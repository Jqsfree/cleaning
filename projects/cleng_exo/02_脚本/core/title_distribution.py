"""Discover title themes in a metadata file without deciding what to drop.

The lexical method uses multilingual character TF-IDF and MiniBatchKMeans.
An optional local sentence encoder can improve semantic grouping. Cluster IDs
are exploratory: neither proximity nor title patterns are topic gold labels.
"""
from __future__ import annotations

import csv
import json
import math
import random
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

from core.data_profile import file_hash, iter_metadata

_LATIN = re.compile(r"[a-z][a-z0-9']{2,}", re.I)
_HAN = re.compile(r"[\u4e00-\u9fff]{2,}")
_SPACE = re.compile(r"\s+")
_STOP = frozenset({
    "the", "and", "for", "with", "from", "this", "that", "your", "you", "our",
    "how", "what", "video", "videos", "shorts", "full", "part", "episode", "new",
    "best", "top", "all", "day", "life", "live", "official", "2026", "2025",
})


def normalize_title(value: str) -> str:
    return _SPACE.sub(" ", unicodedata.normalize("NFKC", value or "").casefold()).strip()


def title_patterns(title: str) -> set[str]:
    """Readable English unigrams/bigrams and Chinese 2–3 character phrases."""
    normalized = normalize_title(title)
    words = [w for w in _LATIN.findall(normalized) if w not in _STOP and not w.isdigit()]
    result = set(words)
    result.update(a + " " + b for a, b in zip(words, words[1:]) if a != b)
    for match in _HAN.finditer(normalized):
        phrase = match.group(0)
        for width in (2, 3):
            result.update(phrase[i:i + width] for i in range(len(phrase) - width + 1))
    return result


def _duration(row: dict) -> float:
    try:
        value = float(row.get("duration_seconds") or "")
        return value if math.isfinite(value) and value > 0 else 0.0
    except (ValueError, TypeError):
        return 0.0


def _write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _features_for_fit(titles: list[str], *, encoder: str | None, max_features: int, seed: int):
    import numpy as np
    if encoder:
        from sentence_transformers import SentenceTransformer
        import torch
        torch.set_num_threads(4)
        source = Path(encoder)
        if not source.is_dir():
            raise ValueError("--encoder must point to an existing local model directory")
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        model = SentenceTransformer(str(source), local_files_only=True, device=device)
        x = model.encode(titles, batch_size=64, normalize_embeddings=True,
                         show_progress_bar=False).astype(np.float64)

        def transform(batch: list[str]):
            values = model.encode(batch, batch_size=64, normalize_embeddings=True,
                                  show_progress_bar=False).astype(np.float64)
            if not np.isfinite(values).all():
                raise ValueError("Non-finite title embeddings")
            return values

        return x, transform, {"method": "local_sentence_embedding", "encoder": str(source.resolve()),
                              "device": device}

    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2,
                                 max_features=max_features, sublinear_tf=True,
                                 dtype=np.float64)
    sparse = vectorizer.fit_transform(titles)
    if sparse.shape[1] < 2:
        raise ValueError("Too few distinct title features for topic discovery")

    def transform(batch: list[str]):
        values = vectorizer.transform(batch)
        if not np.isfinite(values.data).all():
            raise ValueError("Non-finite title TF-IDF features")
        return values

    return sparse, transform, {"method": "char_tfidf_sparse", "features": sparse.shape[1]}


def _fit_sample(input_path: Path, sample_size: int, seed: int) -> tuple[list[str], dict]:
    rng = random.Random(seed)
    reservoir: list[str] = []
    total = empty = 0
    seconds = 0.0
    for row in iter_metadata(input_path):
        total += 1
        seconds += _duration(row)
        title = normalize_title(row["title"])
        if not title:
            empty += 1
            continue
        seen = total - empty
        if len(reservoir) < sample_size:
            reservoir.append(title)
        else:
            slot = rng.randrange(seen)
            if slot < sample_size:
                reservoir[slot] = title
    if total == 0 or total == empty:
        raise ValueError("Input has no nonempty titles")
    # Fitting on unique strings prevents repeated SEO titles dominating centers.
    unique = list(dict.fromkeys(reservoir))
    if len(unique) < 10:
        raise ValueError("At least 10 distinct sampled titles are required")
    return unique, {"input_rows": total, "empty_title_rows": empty,
                    "input_hours": round(seconds / 3600, 4),
                    "sampled_rows": len(reservoir), "unique_fit_titles": len(unique)}


def _candidate_patterns(titles: list[str], labels, topic_count: int) -> set[str]:
    per_topic = [Counter() for _ in range(topic_count)]
    topic_sizes = Counter(int(topic) for topic in labels)
    global_counts = Counter()
    for title, topic in zip(titles, labels):
        patterns = title_patterns(title)
        per_topic[int(topic)].update(patterns)
        global_counts.update(patterns)
    selected: set[str] = set()
    for topic, counts in enumerate(per_topic):
        size = topic_sizes[topic] or 1
        ranked = sorted(counts.items(), key=lambda x: (
            -(x[1] * math.log2(1 + x[1] / max(global_counts[x[0]], 1) * len(titles) / size)),
            -x[1], x[0],
        ))
        selected.update(term for term, _ in ranked[:50])
    return selected


class _CosineTopics:
    """Spherical k-means for unit-length sentence embeddings."""

    def __init__(self, n_clusters: int, seed: int, device: str):
        import torch
        self.n_clusters = n_clusters
        self.seed = seed
        self.device = torch.device(device)
        self.centers = None

    def fit(self, values):
        import numpy as np
        import torch
        points = torch.as_tensor(values, dtype=torch.float32, device=self.device)
        rng = random.Random(self.seed)
        centers = points[rng.sample(range(len(points)), self.n_clusters)].clone()
        previous = None
        for _ in range(50):
            labels = (points @ centers.T).argmax(dim=1)
            if previous is not None and torch.equal(previous, labels):
                break
            previous = labels
            updated = torch.zeros_like(centers)
            updated.index_add_(0, labels, points)
            lengths = torch.linalg.vector_norm(updated, dim=1, keepdim=True)
            nonempty = lengths.squeeze(1) > 0
            updated[nonempty] /= lengths[nonempty]
            updated[~nonempty] = centers[~nonempty]
            centers = updated
        self.centers = centers
        self.cluster_centers_ = centers.cpu().numpy().astype(np.float64)
        return self

    def transform(self, values):
        import numpy as np
        import torch
        points = torch.as_tensor(values, dtype=torch.float32, device=self.device)
        similarities = (points @ self.centers.T).clamp(-1, 1)
        distances = (1 - similarities).cpu().numpy().astype(np.float64)
        if not np.isfinite(distances).all():
            raise ValueError("Non-finite title-to-topic distances")
        return distances

    def predict(self, values):
        return self.transform(values).argmin(axis=1)


def analyze_title_distribution(
    input_path: str | Path, output: str | Path, *, topics: int = 20,
    sample_size: int = 10000, batch_size: int = 2048, seed: int = 42,
    max_features: int = 25000, encoder: str | None = None,
) -> dict:
    """Cluster titles, count the full file, and write human-review evidence."""
    import numpy as np
    from sklearn.cluster import MiniBatchKMeans

    input_path, output = Path(input_path), Path(output)
    if output.exists():
        raise FileExistsError(output)
    if not input_path.is_file() or topics < 2 or sample_size < 10 or batch_size < 1 or max_features < 100:
        raise ValueError("Invalid input or topic analysis parameters")
    before_hash = file_hash(input_path)
    sample, intake = _fit_sample(input_path, sample_size, seed)
    n_topics = min(topics, max(2, len(sample) // 5))
    x_sample, transform, feature_info = _features_for_fit(
        sample, encoder=encoder, max_features=max_features, seed=seed)
    feature_values = x_sample.data if hasattr(x_sample, "data") else x_sample
    if not np.isfinite(feature_values).all():
        raise ValueError("Non-finite fitted title features")
    if encoder:
        model = _CosineTopics(n_topics, seed, feature_info["device"])
    else:
        model = MiniBatchKMeans(n_clusters=n_topics, batch_size=min(1024, len(sample)),
                                n_init=3, max_iter=100, random_state=seed, init="random")
    model.fit(x_sample)
    if not np.isfinite(model.cluster_centers_).all():
        raise ValueError("Non-finite topic centers")
    sample_labels = model.predict(x_sample)
    selected_patterns = _candidate_patterns(sample, sample_labels, n_topics)

    output.mkdir(parents=True)
    topic_counts = Counter()
    topic_seconds = Counter()
    topic_near_ties = Counter()
    topic_channels: dict[int, Counter] = defaultdict(Counter)
    topic_patterns: dict[int, Counter] = defaultdict(Counter)
    global_patterns = Counter()
    nearest: dict[int, list[tuple[float, str, str]]] = defaultdict(list)
    random_examples: dict[int, list[str]] = defaultdict(list)
    seen_by_topic = Counter()
    rng = random.Random(seed + 1)
    assigned = 0

    def process(batch: list[dict], writer) -> None:
        nonlocal assigned
        matrix = transform([normalize_title(row["title"]) for row in batch])
        distances = model.transform(matrix)
        order = np.argsort(distances, axis=1)
        for row, ranking, ds in zip(batch, order, distances):
            topic = int(ranking[0])
            first = float(ds[topic])
            second = float(ds[ranking[1]])
            margin = (second - first) / max(second, 1e-12)
            topic_counts[topic] += 1
            if margin <= .05:
                topic_near_ties[topic] += 1
            topic_seconds[topic] += _duration(row)
            channel = (row.get("channel") or "").strip()
            if channel:
                topic_channels[topic][channel] += 1
            terms = title_patterns(row["title"]) & selected_patterns
            topic_patterns[topic].update(terms)
            global_patterns.update(terms)
            representative = nearest[topic]
            item = (first, row["title"], row["video_id"])
            if all(existing[1] != row["title"] for existing in representative):
                representative.append(item)
                representative.sort()
                del representative[10:]
            seen_by_topic[topic] += 1
            examples = random_examples[topic]
            if len(examples) < 12:
                examples.append(row["title"])
            else:
                slot = rng.randrange(seen_by_topic[topic])
                if slot < 12:
                    examples[slot] = row["title"]
            writer.writerow({"video_id": row["video_id"], "title": row["title"],
                             "topic_id": f"topic_{topic:02d}", "distance": f"{first:.6f}",
                             "assignment_margin": f"{margin:.6f}"})
            assigned += 1

    with (output / "title_topic_assignments.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["video_id", "title", "topic_id", "distance", "assignment_margin"])
        writer.writeheader()
        batch = []
        for row in iter_metadata(input_path):
            if not normalize_title(row["title"]):
                continue
            batch.append(row)
            if len(batch) >= batch_size:
                process(batch, writer)
                batch.clear()
        if batch:
            process(batch, writer)

    if assigned != intake["input_rows"] - intake["empty_title_rows"] or file_hash(input_path) != before_hash:
        raise ValueError("Input changed or row count differed between analysis passes")

    patterns_rows, summary_rows, review_rows, example_rows = [], [], [], []
    for topic in sorted(topic_counts):
        size = topic_counts[topic]
        ranked = []
        for term, count in topic_patterns[topic].items():
            global_count = global_patterns[term]
            lift = (count / size) / (global_count / assigned)
            if count >= max(3, math.ceil(size * .002)):
                ranked.append((count * math.log2(max(lift, 1.0) + 1), count, lift, term, global_count))
        ranked.sort(key=lambda x: (-x[0], -x[1], x[3]))
        top = []
        for _, count, lift, term, global_count in ranked[:20]:
            top.append(term)
            patterns_rows.append({"topic_id": f"topic_{topic:02d}", "pattern": term,
                                  "topic_rows": size, "hit_rows": count,
                                  "global_hit_rows": global_count, "topic_share": round(count / size, 5),
                                  "lift_vs_input": round(lift, 3)})
        central = [title for _, title, _ in nearest[topic]]
        channels = [f"{name} ({count})" for name, count in topic_channels[topic].most_common(5)]
        summary = {"topic_id": f"topic_{topic:02d}", "rows": size,
                   "row_share": round(size / intake["input_rows"], 6),
                   "hours": round(topic_seconds[topic] / 3600, 3),
                   "hour_share": round(topic_seconds[topic] / max(intake["input_hours"] * 3600, 1), 6),
                   "near_tie_rows": topic_near_ties[topic],
                   "near_tie_share": round(topic_near_ties[topic] / size, 5),
                   "top_title_patterns": " | ".join(top[:12]),
                   "top_channels_context_only": " | ".join(channels),
                   "central_titles": " || ".join(central[:5])}
        summary_rows.append(summary)
        review_rows.append({**summary, "human_topic_name": "", "is_desired_category": "",
                            "reviewer": "", "review_notes": ""})
        for kind, titles in (("central", central), ("random", random_examples[topic])):
            for rank, title in enumerate(titles, 1):
                example_rows.append({"topic_id": f"topic_{topic:02d}", "kind": kind,
                                     "rank": rank, "title": title})
    summary_rows.sort(key=lambda x: -x["rows"])
    review_rows.sort(key=lambda x: -x["rows"])
    _write_csv(output / "topic_distribution.csv", summary_rows,
               ["topic_id", "rows", "row_share", "hours", "hour_share", "near_tie_rows", "near_tie_share", "top_title_patterns",
                "top_channels_context_only", "central_titles"])
    _write_csv(output / "title_patterns_by_topic.csv", patterns_rows,
               ["topic_id", "pattern", "topic_rows", "hit_rows", "global_hit_rows", "topic_share", "lift_vs_input"])
    _write_csv(output / "topic_examples.csv", example_rows, ["topic_id", "kind", "rank", "title"])
    _write_csv(output / "topic_review.csv", review_rows,
               ["topic_id", "rows", "row_share", "hours", "hour_share", "near_tie_rows", "near_tie_share", "top_title_patterns",
                "top_channels_context_only", "central_titles", "human_topic_name", "is_desired_category",
                "reviewer", "review_notes"])
    report = {
        "status": "exploratory_title_topics_requires_human_review",
        "input": str(input_path.resolve()), "input_sha256": before_hash,
        "parameters": {"topics_requested": topics, "topics_fitted": n_topics,
                       "sample_size": sample_size, "batch_size": batch_size,
                       "seed": seed, "max_features": max_features, "encoder": encoder},
        "features": feature_info, "intake": intake,
        "assigned_rows": assigned,
        "topic_rows_sum": sum(topic_counts.values()),
        "note": "Clusters are lexical/semantic discovery aids, not validated category labels. near_tie means the two nearest centers differ by <=5% relative distance; it is not a probability. Channels are context only. No video is removed.",
    }
    (output / "manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# Title topic distribution", "",
             f"Input: `{input_path}` · SHA256 `{before_hash}`",
             f"Rows: {intake['input_rows']:,}; titles assigned: {assigned:,}; requested topics: {topics}; fitted: {n_topics}.",
             "", "This is topic discovery based on titles. Review examples and name each topic before any off-topic rule is written.", ""]
    for row in summary_rows:
        lines += [f"## {row['topic_id']} · {row['rows']:,} rows ({row['row_share']:.1%}) · {row['hours']:,.1f} h · near-tie {row['near_tie_share']:.1%}",
                  "", f"Patterns: {row['top_title_patterns'] or '(no stable pattern)' }", "",
                  f"Central titles: {row['central_titles']}", ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return report
