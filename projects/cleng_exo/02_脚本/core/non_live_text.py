"""Cross-category, title-only signals for non-live content.

Rules stay in review mode until independent human evidence promotes them.
The audit reads annotations; the scan tags source metadata without dropping rows.
"""
from __future__ import annotations

import csv
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib

from core.data_profile import file_hash, iter_metadata

DEFAULT_POLICY = Path(__file__).with_name("non_live_candidates.toml")
_LABELS = {"T", "F", "U"}


def load_candidates(path: str | Path = DEFAULT_POLICY) -> tuple[dict, list[dict]]:
    path = Path(path)
    policy = tomllib.loads(path.read_text(encoding="utf-8"))
    meta = policy.get("meta") or {}
    if not meta.get("version") or meta.get("field") != "title" or meta.get("default_action") != "review":
        raise ValueError("Non-live policy must be title-only and review-only")
    rules, ids = [], set()
    for raw in policy.get("rule") or []:
        name, group, pattern = raw.get("id"), raw.get("group"), raw.get("pattern")
        if (not name or name in ids or group not in {"shared_candidate", "category_conditional"}
                or not pattern or raw.get("action", "review") != "review"):
            raise ValueError(f"Invalid or duplicate non-live rule: {name}")
        ids.add(name)
        rules.append({"id": name, "group": group, "pattern": pattern,
                      "regex": re.compile(pattern, re.I)})
    if not rules:
        raise ValueError("Non-live policy has no rules")
    return meta, rules


def matched_rules(title: str, rules: list[dict]) -> list[dict]:
    normalized = unicodedata.normalize("NFKC", title or "")
    return [rule for rule in rules if rule["regex"].search(normalized)]


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def discover_annotation_files(roots: list[str | Path], files: list[str | Path] | None = None) -> list[Path]:
    """Find likely human annotation files; provenance is retained for review."""
    found = {Path(p).resolve() for p in files or []}
    for root in map(Path, roots):
        if not root.exists():
            raise FileNotFoundError(root)
        if root.is_file():
            found.add(root.resolve())
            continue
        for path in root.rglob("*.csv"):
            # Clean/model outputs may contain "gold" in the run name without
            # containing any independently annotated rows.
            if set(path.parts) & {"01_quality", "02_sample", "05_clean", "06_tools", "07_deliver"}:
                continue
            name = path.name.lower()
            if any(part in name for part in ("qc_result", "gold", "human_label")) or path.parent.name == "gold":
                found.add(path.resolve())
    return sorted(found)


def _label_of(row: dict, path: Path) -> tuple[str, str]:
    source = (row.get("label_source") or "").strip().lower()
    if "topic_label" in row and source == "human":
        return (row.get("topic_label") or "").strip().upper(), "explicit_human"
    if "human_label" in row and (source == "human" or "human" in path.name.lower()):
        return (row.get("human_label") or "").strip().upper(), "explicit_human"
    if "qc_result" in row and any(part in path.name.lower() for part in ("qc_result", "gold")) and source in ("", "human"):
        return (row.get("qc_result") or "").strip().upper(), "qc_result_assumed_human"
    return "", "unsupported_label_source"


def _identity(row: dict) -> tuple[str, str]:
    title = " ".join(unicodedata.normalize("NFKC", (row.get("title") or "")).casefold().split())
    video_id = (row.get("video_id") or "").strip()
    return video_id or (row.get("channel") or "").strip().casefold(), title


def audit_annotations(
    *, roots: list[str | Path], output: str | Path,
    files: list[str | Path] | None = None, policy_path: str | Path = DEFAULT_POLICY,
) -> dict:
    """Retrospective cross-category replay, never automatic rule acceptance."""
    policy_path, output = Path(policy_path), Path(output)
    meta, rules = load_candidates(policy_path)
    candidates = discover_annotation_files(roots, files)
    if not candidates:
        raise ValueError("No annotation files found")
    by_identity: dict[tuple[str, str], dict[str, dict]] = defaultdict(dict)
    file_aliases, inventories = {}, []
    skipped = Counter()
    for path in candidates:
        if "weak" in path.name.casefold():
            skipped["weak_file"] += 1
            continue
        digest = file_hash(path)
        if digest in file_aliases:
            file_aliases[digest].append(str(path))
            skipped["exact_duplicate_file"] += 1
            continue
        file_aliases[digest] = [str(path)]
        file_info = {"path": str(path), "sha256": digest, "rows": 0,
                     "labeled": 0, "label_sources": Counter(), "skipped": Counter()}
        try:
            with path.open(encoding="utf-8-sig", newline="") as f:
                reader = csv.DictReader(f, strict=True)
                if not reader.fieldnames or not {"video_id", "title"} <= set(reader.fieldnames):
                    skipped["missing_identity_columns_file"] += 1
                    continue
                for row in reader:
                    file_info["rows"] += 1
                    label, source = _label_of(row, path)
                    file_info["label_sources"][source] += 1
                    # Playback failure is a technical label, not evidence of
                    # animation, audio-only content, or any other title form.
                    if "无法播放" in (row.get("qc_result") or "") or (row.get("qc_status") or "").lower() in {"error", "unavailable"}:
                        file_info["skipped"]["technical_failure"] += 1
                        continue
                    if label not in _LABELS:
                        file_info["skipped"]["unlabeled_or_nonstandard"] += 1
                        continue
                    key = _identity(row)
                    if not key[1]:
                        file_info["skipped"]["empty_title"] += 1
                        continue
                    file_info["labeled"] += 1
                    if label not in by_identity[key]:
                        by_identity[key][label] = {
                            "video_id": (row.get("video_id") or "").strip(),
                            "title": (row.get("title") or "").strip(),
                            "channel": (row.get("channel") or "").strip(),
                            "label": label, "source": str(path),
                            "label_kind": source,
                            "scope": (row.get("topic_category") or row.get("category") or path.parent.name).strip(),
                        }
        except (OSError, UnicodeError, csv.Error) as exc:
            file_info["error"] = str(exc)
            skipped["unreadable_file"] += 1
        file_info["label_sources"] = dict(file_info["label_sources"])
        file_info["skipped"] = dict(file_info["skipped"])
        inventories.append(file_info)

    records = [row for labels in by_identity.values() for row in labels.values()]
    conflicts = [list(labels.values()) for labels in by_identity.values() if len(labels) > 1]
    hit_counts: dict[str, Counter] = {r["id"]: Counter() for r in rules}
    scope_counts: dict[str, dict[str, Counter]] = {
        r["id"]: defaultdict(Counter) for r in rules
    }
    examples: dict[str, dict[str, list[dict]]] = {
        r["id"]: {lab: [] for lab in _LABELS} for r in rules
    }
    cases = []
    for row in records:
        hits = matched_rules(row["title"], rules)
        for rule in hits:
            hit_counts[rule["id"]][row["label"]] += 1
            scope_counts[rule["id"]][row["scope"]][row["label"]] += 1
            bucket = examples[rule["id"]][row["label"]]
            if len(bucket) < 5:
                bucket.append({k: row[k] for k in ("video_id", "title", "source", "scope")})
            cases.append({**row, "rule_id": rule["id"], "rule_group": rule["group"]})

    result = {
        "status": "retrospective_candidate_audit_not_rule_acceptance",
        "policy": str(policy_path.resolve()), "policy_sha256": file_hash(policy_path),
        "policy_version": meta["version"],
        "input_files_discovered": len(candidates),
        "input_files_read": len(inventories),
        "skipped": dict(skipped),
        "unique_identity_title_keys": len(by_identity),
        "label_records": len(records),
        "label_counts": dict(Counter(row["label"] for row in records)),
        "conflicting_identity_title_keys": len(conflicts),
        "files": inventories,
        "exact_duplicate_file_aliases": [v for v in file_aliases.values() if len(v) > 1],
        "rules": [{
            "id": r["id"], "group": r["group"],
            "counts": dict(hit_counts[r["id"]]),
            "scope_counts": {scope: dict(count) for scope, count in sorted(scope_counts[r["id"]].items())},
            "examples": examples[r["id"]],
            "evidence_status": (
                "blocked_by_T_counterexample" if hit_counts[r["id"]]["T"] else
                "insufficient_F_support" if hit_counts[r["id"]]["F"] < 3 else
                "requires_independent_validation"
            ),
            "decision": "review_only",
        } for r in rules],
        "note": "qc_result provenance is assumed human unless explicitly marked; verify sources and label definitions before promotion. F labels do not encode why a video failed.",
    }
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    _write_json(output / "audit.json", result)
    _write_csv(output / "rule_hits.csv", cases,
               ["video_id", "title", "channel", "label", "source", "label_kind", "scope", "rule_id", "rule_group"])
    _write_csv(output / "label_conflicts.csv",
               [dict(row, conflict_key="|".join(_identity(row))) for group in conflicts for row in group],
               ["conflict_key", "video_id", "title", "channel", "label", "source", "label_kind", "scope"])
    return {k: result[k] for k in ("status", "policy_version", "input_files_discovered", "label_records", "label_counts", "conflicting_identity_title_keys")}


def scan_metadata(*, input_path: str | Path, output: str | Path,
                  policy_path: str | Path = DEFAULT_POLICY) -> dict:
    """Tag title matches in a metadata pool; does not split or drop the input."""
    input_path, output, policy_path = Path(input_path), Path(output), Path(policy_path)
    meta, rules = load_candidates(policy_path)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    counts, total, matched = Counter(), 0, 0
    with (output / "candidates.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["video_id", "title", "matched_rules", "rule_groups", "decision"])
        writer.writeheader()
        for row in iter_metadata(input_path):
            total += 1
            hits = matched_rules(row["title"], rules)
            if not hits:
                continue
            matched += 1
            counts.update(rule["id"] for rule in hits)
            writer.writerow({"video_id": row["video_id"], "title": row["title"],
                             "matched_rules": ",".join(rule["id"] for rule in hits),
                             "rule_groups": ",".join(sorted({rule["group"] for rule in hits})),
                             "decision": "review"})
    result = {"status": "shadow_scan_no_rows_dropped", "input": str(input_path.resolve()),
              "input_sha256": file_hash(input_path), "policy_sha256": file_hash(policy_path),
              "policy_version": meta["version"], "total_rows": total,
              "candidate_rows": matched, "rule_hits": dict(counts),
              "candidate_file": str((output / "candidates.csv").resolve())}
    _write_json(output / "summary.json", result)
    return result
