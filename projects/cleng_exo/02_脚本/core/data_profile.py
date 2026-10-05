"""Strict, streaming metadata intake profiling; no topic decisions or implicit labels."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import random
from collections import Counter
from pathlib import Path

FIELDS = ("title", "channel", "description")


def file_hash(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def iter_metadata(path: str | Path):
    """Read CSV/TSV/Parquet without silent skips or malformed record repair."""
    path = Path(path)
    if path.suffix.lower() in (".parquet", ".pq"):
        import pyarrow.parquet as pq
        source = pq.ParquetFile(path)
        names = source.schema_arrow.names
        _check_fields(names)
        for batch in source.iter_batches(batch_size=4096):
            for row in batch.to_pylist():
                yield {k: "" if v is None else str(v) for k, v in row.items()}
    else:
        with path.open(encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f, delimiter="\t" if path.suffix.lower()==".tsv" else ",", strict=True)
            _check_fields(reader.fieldnames)
            try:
                for row in reader:
                    if None in row or any(v is None or "\x00" in v for v in row.values()):
                        raise ValueError(f"Malformed metadata record near line {reader.line_num}")
                    yield row
            except csv.Error as exc:
                raise ValueError(f"Malformed CSV near line {reader.line_num}: {exc}") from exc


def _check_fields(names):
    if not names or len(names)!=len(set(names)) or any(not x.strip() for x in names):
        raise ValueError("Missing or duplicate column names")
    if not {"video_id", "title"} <= set(names):
        raise ValueError("Metadata requires video_id,title")


def _ratio(n, total):
    return n / total if total else None


def profile_file(path: str | Path, category: str, *, sample_size=5000, seed=42, reference=None) -> dict:
    """Exact counts; bounded reservoir for character-length quantiles. O(unique IDs) identity index."""
    if sample_size < 1:
        raise ValueError("sample_size must be positive")
    path = Path(path)
    before = file_hash(path)
    seen, channels, languages = {}, Counter(), Counter()
    missing, n, duplicates, conflicts, empty_ids = Counter(), 0, 0, 0, 0
    duration_valid, duration_invalid, duration_missing, seconds = 0, 0, 0, 0.0
    sample, rng, columns = [], random.Random(seed), []
    for row in iter_metadata(path):
        n += 1
        columns = list(row)
        vid = row["video_id"].strip()
        identity = tuple(row.get(k, "").strip() for k in FIELDS)
        identity_hash = hashlib.sha256(json.dumps(identity, ensure_ascii=False).encode()).hexdigest()
        if not vid:
            empty_ids += 1
        elif vid in seen:
            duplicates += 1
            conflicts += seen[vid] != identity_hash
        else:
            seen[vid] = identity_hash
        for name in FIELDS:
            missing[name] += not row.get(name, "").strip()
        channels[row.get("channel", "").strip() or "(unknown)"] += 1
        # This is supplied metadata, not automatic language identification.
        languages[row.get("language", "").strip() or "(unknown)"] += 1
        duration = row.get("duration_seconds", "").strip()
        if not duration:
            duration_missing += 1
        else:
            try:
                value = float(duration)
            except ValueError:
                value = float("nan")
            if math.isfinite(value) and value > 0:
                duration_valid += 1
                seconds += value
            else:
                duration_invalid += 1
        lengths = {name: len(row.get(name, "")) for name in FIELDS}
        if len(sample) < sample_size:
            sample.append(lengths)
        else:
            j = rng.randrange(n)
            if j < sample_size:
                sample[j] = lengths
    if file_hash(path) != before:
        raise ValueError("Input changed during profiling")
    quantiles = {}
    for name in FIELDS:
        values = sorted(x[name] for x in sample)
        quantiles[name] = {f"p{p}": values[round((len(values)-1)*p/100)] if values else None for p in (50, 95, 99)}
    metrics = {
        "rows": n, "unique_nonempty_ids": len(seen), "duplicate_rows": duplicates,
        "conflicting_duplicate_rows": conflicts, "empty_id_rows": empty_ids,
        "missing_fields": {k: {"n":missing[k], "rate":_ratio(missing[k],n)} for k in FIELDS},
        "channels": {"distinct_including_unknown":len(channels), "top10":channels.most_common(10),
                     "top10_share":_ratio(sum(c for _,c in channels.most_common(10)),n)},
        "supplied_language":dict(languages),
        "duration": {"valid_rows":duration_valid, "invalid_rows":duration_invalid,
                     "missing_rows":duration_missing, "valid_hours":seconds/3600},
        "character_length_quantiles":quantiles,
    }
    result = {"schema_version":1,"category":category,"stage":"intake", "input":str(path.resolve()),
        "input_sha256":before,"columns":columns,"metrics":metrics,
        "measurement":{"counts":"exact parsed records", "length_quantiles":"reservoir sample",
                       "sample_rows":len(sample),"seed":seed,"token_truncation":"not measured"},
        "status":"execution_checks_passed", "warnings":[], "reference_comparison":None,
        "note":"Structural profile only; no semantic quality or release approval."}
    if not n or empty_ids or conflicts:
        result["status"]="blocked_input_contract"
    for key,count in (("empty_input",int(not n)),("empty_ids",empty_ids),("duplicate_ids",duplicates),
                      ("conflicting_metadata",conflicts),("invalid_duration",duration_invalid)):
        if count: result["warnings"].append({"code":key,"count":count})
    if reference:
        ref = json.loads(Path(reference).read_text())
        if ref.get("category")!=category or ref.get("stage")!="intake" or ref.get("schema_version")!=1:
            raise ValueError("Reference category/stage/schema mismatch")
        changes = {}
        for k in FIELDS:
            a,b = ref["metrics"]["missing_fields"][k]["rate"],metrics["missing_fields"][k]["rate"]
            changes[k+"_missing_rate_delta"] = b-a if a is not None and b is not None else None
        result["reference_comparison"]={"reference_input_sha256":ref["input_sha256"],"deltas":changes,
                                         "interpretation":"descriptive changes, not a quality verdict"}
    return result


def write_profile(report: dict, output: str | Path) -> None:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output/"profile.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n")
    m=report["metrics"]
    (output/"profile.md").write_text(
        f"# Metadata intake: {report['category']}\n\nStatus: {report['status']}\n\n"
        f"Rows: {m['rows']}; unique IDs: {m['unique_nonempty_ids']}; duplicate rows: {m['duplicate_rows']}; "
        f"conflicts: {m['conflicting_duplicate_rows']}; empty IDs: {m['empty_id_rows']}.\n\n"
        "Counts cover parsed records. Length quantiles use the recorded reservoir sample. "
        "Token truncation and semantic purity are not measured. Full metrics: profile.json.\n")
