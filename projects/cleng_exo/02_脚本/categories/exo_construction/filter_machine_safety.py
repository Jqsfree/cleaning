#!/usr/bin/env python3
"""Exclude heavy-machine-operation titles and safety-course titles from construction keep."""
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

DEFAULT_POLICY = Path(__file__).parent / "rules/machine_safety_title_v1.toml"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def filter_file(input_path: Path, output_dir: Path, policy_path: Path) -> dict:
    if not input_path.is_file() or not policy_path.is_file() or output_dir.exists():
        raise ValueError("Input and policy must exist; output directory must be new")
    input_hash = sha256(input_path)
    policy = tomllib.loads(policy_path.read_text(encoding="utf-8"))
    machine = re.compile(policy["machine_operation"]["equipment"])
    brand = re.compile(policy["machine_operation"]["machine_brand"])
    operation = re.compile(policy["machine_operation"]["operation"])
    protection = re.compile(policy["machine_operation"]["manual_work_protection"])
    safety = re.compile(policy["safety_course"]["pattern"])
    output_dir.mkdir(parents=True)
    keep_path = output_dir / "construction_0929_machine_safety_keep.csv"
    drop_path = output_dir / "construction_0929_machine_safety_drop.csv"
    counts, hours = Counter(), Counter()
    seen = set()
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
            if video_id in seen:
                raise ValueError(f"Duplicate video_id: {video_id}")
            seen.add(video_id)
            title = row["title"]
            reasons = []
            if (machine.search(title) or (brand.search(title) and operation.search(title))) and not protection.search(title):
                reasons.append("heavy_machine_operation")
            if safety.search(title):
                reasons.append("safety_course")
            try:
                duration = float(row["duration_seconds"])
            except (ValueError, TypeError):
                duration = 0.0
            valid_hours = duration / 3600 if math.isfinite(duration) and duration > 0 else 0.0
            counts["input"] += 1
            hours["input"] += valid_hours
            if reasons:
                drop_writer.writerow({**row, "drop_reasons": ";".join(reasons)})
                counts["drop"] += 1
                hours["drop"] += valid_hours
                for reason in reasons:
                    counts[reason] += 1
            else:
                keep_writer.writerow(row)
                counts["keep"] += 1
                hours["keep"] += valid_hours
    if sha256(input_path) != input_hash or counts["input"] != counts["keep"] + counts["drop"]:
        raise ValueError("Input changed or output counts did not reconcile")
    report = {
        "input": str(input_path.resolve()), "input_sha256": input_hash,
        "policy": str(policy_path.resolve()), "policy_sha256": sha256(policy_path),
        "policy_version": policy["meta"]["version"],
        "counts": dict(counts), "hours": {key: round(value, 3) for key, value in hours.items()},
        "outputs": {"keep": str(keep_path.resolve()), "drop": str(drop_path.resolve())},
        "note": "Title-only rule. It cannot prove whether a person is visible in the video; review drop and remaining keep samples.",
    }
    (output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    args = parser.parse_args()
    print(json.dumps(filter_file(args.input, args.out, args.policy), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
