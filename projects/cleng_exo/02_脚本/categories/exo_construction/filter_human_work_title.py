#!/usr/bin/env python3
"""Keep only construction rows whose titles contain hands-on work evidence."""
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

DEFAULT_POLICY = Path(__file__).parent / "rules/human_work_title_v1.toml"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(source: Path, out_dir: Path, policy_path: Path) -> dict:
    if not source.is_file() or not policy_path.is_file() or out_dir.exists():
        raise ValueError("Input/policy must exist and output directory must be new")
    input_hash = sha256(source)
    policy = tomllib.loads(policy_path.read_text(encoding="utf-8"))
    rules = [(rule["id"], re.compile(rule["pattern"])) for rule in policy["rules"]]
    if len({rule_id for rule_id, _ in rules}) != len(rules):
        raise ValueError("Duplicate rule ID")
    out_dir.mkdir(parents=True)
    keep_path = out_dir / "construction_0929_human_work_keep.csv"
    drop_path = out_dir / "construction_0929_human_work_drop.csv"
    counts, hours, ids = Counter(), Counter(), set()
    with source.open(encoding="utf-8-sig", newline="") as src, \
         keep_path.open("w", encoding="utf-8-sig", newline="") as keep, \
         drop_path.open("w", encoding="utf-8-sig", newline="") as drop:
        reader = csv.DictReader(src, strict=True)
        columns = list(reader.fieldnames or [])
        if not {"video_id", "title", "duration_seconds"} <= set(columns):
            raise ValueError("Missing required columns")
        keep_writer = csv.DictWriter(keep, fieldnames=columns)
        drop_writer = csv.DictWriter(drop, fieldnames=columns + ["drop_reason"])
        keep_writer.writeheader()
        drop_writer.writeheader()
        for row in reader:
            video_id = row["video_id"]
            if video_id in ids:
                raise ValueError(f"Duplicate video_id: {video_id}")
            ids.add(video_id)
            title = row["title"] or ""
            hits = [rule_id for rule_id, regex in rules if regex.search(title)]
            try:
                duration = float(row["duration_seconds"])
            except (ValueError, TypeError):
                duration = 0.0
            valid_hours = duration / 3600 if math.isfinite(duration) and duration > 0 else 0.0
            counts["input"] += 1
            hours["input"] += valid_hours
            if hits:
                keep_writer.writerow(row)
                counts["keep"] += 1
                hours["keep"] += valid_hours
                counts.update(hits)
            else:
                drop_writer.writerow({**row, "drop_reason": "no_human_work_title_evidence"})
                counts["drop"] += 1
                hours["drop"] += valid_hours
    if sha256(source) != input_hash or counts["input"] != counts["keep"] + counts["drop"]:
        raise ValueError("Input changed or row counts did not reconcile")
    result = {
        "input": str(source.resolve()), "input_sha256": input_hash,
        "policy": str(policy_path.resolve()), "policy_sha256": sha256(policy_path),
        "policy_version": policy["meta"]["version"],
        "counts": dict(counts), "hours": {k: round(v, 3) for k, v in hours.items()},
        "outputs": {"keep": str(keep_path.resolve()), "drop": str(drop_path.resolve())},
        "note": "Positive title evidence only; a title match does not verify visible humans.",
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
