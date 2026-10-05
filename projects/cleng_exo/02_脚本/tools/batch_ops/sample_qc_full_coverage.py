#!/usr/bin/env python3
"""Full-coverage keyword-stratified CSV sample for manual QC.

Keywords whose expected proportional quota is below one are combined into one
tail stratum. The original keyword remains in the sample. The reported margin
is a conservative simple-random-sample reference; actual stratified precision
depends on the labels in each stratum.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from statistics import NormalDist

TAIL = "other_low_frequency_keywords"


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def sample(input_path: Path, output_dir: Path, *, confidence: float = .90,
           margin: float = .05, seed: int = 42) -> dict:
    if not input_path.is_file() or output_dir.exists() or not 0 < confidence < 1 or not 0 < margin < 1:
        raise ValueError("Invalid input, output directory, confidence or margin")
    before = _hash(input_path)
    with input_path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, strict=True)
        fields = list(reader.fieldnames or [])
        if not {"video_id", "title", "keyword"} <= set(fields):
            raise ValueError("Input requires video_id,title,keyword")
        rows = list(reader)
    total = len(rows)
    if total < 2:
        raise ValueError("At least two rows are required")
    ids = [row["video_id"] for row in rows]
    if len(set(ids)) != total:
        raise ValueError("video_id must be unique for row sampling")
    z = NormalDist().inv_cdf((1 + confidence) / 2)
    n0 = z * z * .25 / (margin * margin)
    target = min(total, math.ceil(n0 / (1 + (n0 - 1) / total)))
    keyword_counts = Counter((row.get("keyword") or "").strip() for row in rows)
    while True:
        major = {key for key, count in keyword_counts.items() if count * target >= total}
        groups: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            key = (row.get("keyword") or "").strip()
            group = f"keyword:{key}" if key in major else TAIL
            groups[group].append(row)
        if not groups or sum(map(len, groups.values())) != total:
            raise ValueError("Stratum coverage failed")
        quotas = {name: len(items) * target / total for name, items in groups.items()}
        alloc = {name: math.floor(value) for name, value in quotas.items()}
        leftover = target - sum(alloc.values())
        for name in sorted(groups, key=lambda key: (-(quotas[key] - alloc[key]), -len(groups[key]), key))[:leftover]:
            alloc[name] += 1
        if sum(alloc.values()) != target or any(value < 1 or value > len(groups[name]) for name, value in alloc.items()):
            raise ValueError("Invalid stratum allocation")
        worst_variance = sum(
            (len(items) / total) ** 2 * .25 / alloc[name] *
            (len(items) - alloc[name]) / (len(items) - 1)
            for name, items in groups.items() if len(items) > 1
        )
        stratified_margin = z * math.sqrt(worst_variance)
        if stratified_margin <= margin or target == total:
            break
        target += 1

    rng = random.Random(seed)
    selected = []
    strata = []
    for name in sorted(groups):
        count = len(groups[name])
        quota = alloc[name]
        weight = count / quota
        strata.append({"sampling_stratum": name, "pool_rows": count,
                       "sample_rows": quota, "weight": round(weight, 8)})
        for row in rng.sample(groups[name], quota):
            selected.append({**row, "sampling_stratum": name,
                             "sampling_weight": round(weight, 8),
                             "qc_label": "", "qc_notes": ""})
    rng.shuffle(selected)
    for index, row in enumerate(selected, 1):
        row["sample_id"] = f"QC{index:03d}"
    if _hash(input_path) != before:
        raise ValueError("Input changed during sampling")
    effective = z * math.sqrt(.25 / target) * math.sqrt((total - target) / (total - 1))
    output_dir.mkdir(parents=True)
    sample_path = output_dir / "construction_0929_final_keep_stratified_sample.csv"
    strata_path = output_dir / "strata_allocation.csv"
    manifest_path = output_dir / "manifest.json"
    _write(sample_path, ["sample_id"] + fields + ["sampling_stratum", "sampling_weight", "qc_label", "qc_notes"], selected)
    _write(strata_path, ["sampling_stratum", "pool_rows", "sample_rows", "weight"], strata)
    report = {
        "input": str(input_path.resolve()), "input_sha256": before,
        "population_rows": total, "sample_rows": target,
        "confidence_requested": confidence, "margin_requested": margin,
        "srs_reference_margin_effective": round(effective, 6),
        "stratified_worst_case_margin": round(stratified_margin, 6),
        "stratification": "major keywords individually; low-frequency keywords pooled",
        "major_keyword_strata": len(major),
        "tail_keywords": len(keyword_counts) - len(major),
        "tail_rows": len(groups.get(TAIL, [])),
        "covered_rows": sum(map(len, groups.values())), "seed": seed,
        "outputs": {"sample": str(sample_path.resolve()), "strata": str(strata_path.resolve())},
        "note": "Weights are pool_rows/sample_rows. For QC rates, combine labels with these weights. The worst-case margin assumes p=0.5 in every stratum; it is a design bound for the pooled proportion, not an observed error rate or a bound for each stratum.",
    }
    manifest_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--confidence", type=float, default=.90)
    parser.add_argument("--margin", type=float, default=.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(sample(args.input, args.out, confidence=args.confidence,
                            margin=args.margin, seed=args.seed), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
