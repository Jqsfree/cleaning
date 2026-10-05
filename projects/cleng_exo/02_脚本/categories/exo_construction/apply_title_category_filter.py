#!/usr/bin/env python3
"""Apply a reviewed construction-title exclusion pass to one exact topic run.

The input must match the title-topic manifest SHA256. Topic IDs are valid only
for that run. The keep file retains the original columns; the drop file adds
topic_id and matched exclusion rules for inspection.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from core.data_profile import file_hash

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


def _sha(path: Path) -> str:
    return file_hash(path)


def apply_filter(input_path: Path, assignments_path: Path, manifest_path: Path,
                 policy_path: Path, output_dir: Path, stem: str) -> dict:
    if not all(p.is_file() for p in (input_path, assignments_path, manifest_path, policy_path)):
        raise FileNotFoundError("Input, assignments, manifest and policy must all exist")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    input_hash = _sha(input_path)
    if manifest.get("input_sha256") != input_hash:
        raise ValueError("Topic run does not belong to this exact input CSV")
    policy = tomllib.loads(policy_path.read_text(encoding="utf-8"))
    topics = set(policy["drop_topics"])
    conditional_topics = set(policy.get("drop_unless_construction_topics", []))
    hint = re.compile(policy["construction_hint_pattern"])
    rules = [(r["id"], re.compile(r["pattern"])) for r in policy["rules"]]
    if len({r[0] for r in rules}) != len(rules):
        raise ValueError("Duplicate rule IDs")

    assignments = {}
    with assignments_path.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f, strict=True):
            vid = row["video_id"]
            if vid in assignments:
                raise ValueError(f"Duplicate assignment video_id: {vid}")
            assignments[vid] = (row["topic_id"], row["title"])
    if len(assignments) != manifest["assigned_rows"]:
        raise ValueError("Assignment count differs from topic manifest")

    output_dir.mkdir(parents=True, exist_ok=False)
    keep_path = output_dir / f"{stem}_title_category_keep.csv"
    drop_path = output_dir / f"{stem}_title_category_drop.csv"
    report_path = output_dir / f"{stem}_title_category_report.json"
    review_path = output_dir / f"{stem}_title_category_review_sample.csv"
    counts = Counter()
    by_rule = Counter()
    by_topic = Counter()
    hours = Counter()
    seen_ids = set()
    assigned_seen = 0
    review_seen = Counter()
    review_samples = {}
    rng = random.Random(42)
    with (input_path.open(encoding="utf-8-sig", newline="")) as source, \
         keep_path.open("w", encoding="utf-8-sig", newline="") as keep_file, \
         drop_path.open("w", encoding="utf-8-sig", newline="") as drop_file:
        reader = csv.DictReader(source, strict=True)
        fields = reader.fieldnames
        if not fields or not {"video_id", "title"} <= set(fields):
            raise ValueError("Input requires video_id,title")
        keep_writer = csv.DictWriter(keep_file, fieldnames=fields)
        drop_writer = csv.DictWriter(drop_file, fieldnames=fields + ["topic_id", "drop_rules"])
        keep_writer.writeheader()
        drop_writer.writeheader()
        for row in reader:
            counts["total"] += 1
            vid, title = row["video_id"], row["title"]
            if vid in seen_ids:
                raise ValueError(f"Duplicate input video_id: {vid}")
            seen_ids.add(vid)
            assignment = assignments.get(vid)
            if assignment:
                topic, fitted_title = assignment
                if title != fitted_title:
                    raise ValueError(f"Title changed since topic run: {vid}")
                assigned_seen += 1
            else:
                if title.strip():
                    raise ValueError(f"Nonempty title has no topic assignment: {vid}")
                topic = ""
            matches = ([f"cluster:{topic}"] if topic in topics else [])
            if topic in conditional_topics and not hint.search(title):
                matches.append(f"cluster_without_construction_hint:{topic}")
            matches.extend(rule_id for rule_id, regex in rules if regex.search(title))
            if not title.strip():
                matches.append("empty_title")
            elif policy.get("require_construction_hint") and not hint.search(title):
                matches.append("no_construction_title_signal")
            try:
                duration = float(row.get("duration_seconds") or "nan")
            except ValueError:
                duration = float("nan")
            duration = duration if math.isfinite(duration) and duration > 0 else 0.0
            if matches:
                drop_writer.writerow({**row, "topic_id": topic, "drop_rules": ",".join(matches)})
                primary = matches[0]
                review_seen[primary] += 1
                sample = review_samples.setdefault(primary, [])
                item = {"primary_rule": primary, "topic_id": topic, "video_id": vid,
                        "title": title, "drop_rules": ",".join(matches)}
                if len(sample) < 15:
                    sample.append(item)
                else:
                    slot = rng.randrange(review_seen[primary])
                    if slot < 15:
                        sample[slot] = item
                counts["drop"] += 1
                hours["drop"] += duration / 3600
                by_topic[topic or "unassigned"] += 1
                by_rule.update(matches)
            else:
                keep_writer.writerow(row)
                counts["keep"] += 1
                hours["keep"] += duration / 3600
                if not topic:
                    counts["empty_title_kept"] += 1
    if assigned_seen != len(assignments) or counts["total"] != manifest["intake"]["input_rows"]:
        raise ValueError("Input and topic assignments are not in one-to-one agreement")
    if _sha(input_path) != input_hash:
        raise ValueError("Input CSV changed during filtering")
    with review_path.open("w", encoding="utf-8-sig", newline="") as sample_file:
        writer = csv.DictWriter(sample_file, fieldnames=["primary_rule", "topic_id", "video_id", "title", "drop_rules"])
        writer.writeheader()
        for primary in sorted(review_samples):
            writer.writerows(review_samples[primary])
    report = {
        "input": str(input_path.resolve()), "input_sha256": input_hash,
        "topic_manifest": str(manifest_path.resolve()),
        "policy": str(policy_path.resolve()), "policy_sha256": _sha(policy_path),
        "policy_version": policy["version"],
        "counts": dict(counts), "by_rule": dict(by_rule.most_common()),
        "hours": {key: round(value, 3) for key, value in hours.items()},
        "dropped_by_topic": dict(by_topic.most_common()),
        "outputs": {"keep": str(keep_path.resolve()), "drop": str(drop_path.resolve()),
                    "review_sample": str(review_path.resolve())},
        "note": "Title and topic based first-pass exclusion. Missing duration and out-of-range duration still require quality filtering. Rules may remove some true construction videos; inspect the drop CSV.",
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--assignments", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--policy", type=Path, default=Path(__file__).parent / "rules/title_category_exclude_v1.toml")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--stem", default="construction_0929_nonliveA")
    args = parser.parse_args()
    result = apply_filter(args.input, args.assignments, args.manifest, args.policy,
                          args.out_dir, args.stem)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
