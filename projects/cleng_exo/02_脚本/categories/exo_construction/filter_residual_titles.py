#!/usr/bin/env python3
"""Filter explicit residual off-category title patterns from a construction CSV."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

DEFAULT_POLICY = Path(__file__).parent / "rules/residual_title_v1.toml"


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(input_path: Path, out_dir: Path, policy_path: Path) -> dict:
    if not input_path.is_file() or not policy_path.is_file() or out_dir.exists():
        raise ValueError("Input/policy must exist and output directory must be new")
    original_hash = file_hash(input_path)
    policy = tomllib.loads(policy_path.read_text(encoding="utf-8"))
    rules = [(item["id"], re.compile(item["pattern"])) for item in policy["rules"]]
    if len(set(rule_id for rule_id, _ in rules)) != len(rules):
        raise ValueError("Duplicate rule ID")
    out_dir.mkdir(parents=True)
    keep_path = out_dir / "construction_0929_residual_keep.csv"
    drop_path = out_dir / "construction_0929_residual_drop.csv"
    counts, hours = Counter(), Counter()
    seen_ids = set()
    with input_path.open(encoding="utf-8-sig", newline="") as source, \
         keep_path.open("w", encoding="utf-8-sig", newline="") as kept, \
         drop_path.open("w", encoding="utf-8-sig", newline="") as dropped:
        reader = csv.DictReader(source, strict=True)
        fields = list(reader.fieldnames or [])
        if not {"video_id", "title", "duration_seconds"} <= set(fields):
            raise ValueError("Input requires video_id,title,duration_seconds")
        keep_writer = csv.DictWriter(kept, fieldnames=fields)
        drop_writer = csv.DictWriter(dropped, fieldnames=fields + ["drop_reasons"])
        keep_writer.writeheader()
        drop_writer.writeheader()
        for row in reader:
            video_id = row["video_id"]
            if video_id in seen_ids:
                raise ValueError(f"Duplicate video_id: {video_id}")
            seen_ids.add(video_id)
            title = row["title"]
            hits = [rule_id for rule_id, regex in rules if regex.search(title)]
            try:
                duration = float(row["duration_seconds"])
            except (ValueError, TypeError):
                duration = 0.0
            valid_hours = duration / 3600 if math.isfinite(duration) and duration > 0 else 0.0
            counts["input"] += 1
            hours["input"] += valid_hours
            if hits:
                drop_writer.writerow({**row, "drop_reasons": ";".join(hits)})
                counts["drop"] += 1
                hours["drop"] += valid_hours
                counts.update(hits)
            else:
                keep_writer.writerow(row)
                counts["keep"] += 1
                hours["keep"] += valid_hours
    if file_hash(input_path) != original_hash or counts["input"] != counts["keep"] + counts["drop"]:
        raise ValueError("Input changed or row counts did not reconcile")
    result = {
        "input": str(input_path.resolve()), "input_sha256": original_hash,
        "policy": str(policy_path.resolve()), "policy_sha256": file_hash(policy_path),
        "policy_version": policy["meta"]["version"],
        "counts": dict(counts), "hours": {key: round(value, 3) for key, value in hours.items()},
        "outputs": {"keep": str(keep_path.resolve()), "drop": str(drop_path.resolve())},
        "note": "Title-only residual exclusion. This does not assess video frames or claim that every retained title is on target.",
    }
    (out_dir / "report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    args = parser.parse_args()
    print(json.dumps(run(args.input, args.out, args.policy), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
