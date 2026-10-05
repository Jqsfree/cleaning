"""Traceable historical metadata enrichment and rule diagnosis; never release evidence."""
from __future__ import annotations

import json
import re
import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from core.data_profile import file_hash
from core.topic_loop import human_gold, read_rows, title_key, load_policy, classify, write_csv, write_json, digest
from core.stage_eval import decision_metrics
from core.text_benchmark import validate_splits

SPLITS = ("train", "calibration", "test")
ENRICH_FIELDS = ("description", "duration_seconds", "language")


def enrich_gold(gold_dir, category, output):
    """Join ONLY each gold row's recorded human source; preserve labels and split order."""
    paths = {s: Path(gold_dir)/(s+".csv") for s in SPLITS}
    hashes = {s: file_hash(p) for s, p in paths.items()}
    splits = {s: human_gold(p, category) for s, p in paths.items()}
    validate_splits(splits)
    sources = {}
    for rows in splits.values():
        for row in rows:
            name = row.get("label_file", "")
            if not name: raise ValueError("Missing label_file provenance: "+row["video_id"])
            if name not in sources:
                path = Path(name)
                if not path.is_absolute(): raise ValueError("label_file must be an absolute path")
                before = file_hash(path)
                records = defaultdict(list)
                for source in read_rows(path): records[source["video_id"]].append(source)
                if file_hash(path) != before: raise ValueError("Source changed during enrichment")
                sources[name] = (before, records)
    enriched = {}
    stats = {}
    for split, rows in splits.items():
        enriched[split] = []
        for row in rows:
            sha, index = sources[row["label_file"]]
            candidates = index.get(row["video_id"], [])
            if not candidates: raise ValueError("ID missing from recorded source: "+row["video_id"])
            # Duplicate source rows must agree on every identity/label/enriched field.
            signatures = {tuple(r.get(k, "").strip() for k in ("title", "channel", "qc_result")+ENRICH_FIELDS) for r in candidates}
            if len(signatures) != 1: raise ValueError("Conflicting source rows: "+row["video_id"])
            source = candidates[0]
            for field in ("title", "channel"):
                if source.get(field, "").strip() != row.get(field, "").strip():
                    raise ValueError("Source identity mismatch: "+row["video_id"]+" "+field)
            if source.get("qc_result", "").strip().upper() != row["topic_label"]:
                raise ValueError("Source human label mismatch: "+row["video_id"])
            joined = dict(row)
            for field in ENRICH_FIELDS:
                value = source.get(field, "")
                if row.get(field, "").strip() and row[field].strip() != value.strip():
                    raise ValueError("Existing gold field disagrees with source: "+field)
                joined[field] = row.get(field) or value
            joined["metadata_source_sha256"] = sha
            enriched[split].append(joined)
        stats[split] = {"rows": len(rows), "labels": dict(Counter(r["topic_label"] for r in rows)),
                        "nonempty_fields": {k: sum(bool(r[k].strip()) for r in enriched[split]) for k in ENRICH_FIELDS}}
    for split, p in paths.items():
        if file_hash(p) != hashes[split]: raise ValueError("Gold changed during enrichment")
    for name, (sha, _) in sources.items():
        if file_hash(name) != sha: raise ValueError("Source changed during enrichment")
    output = Path(output)
    if output.exists(): raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".enrich-", dir=str(output.parent)))
    try:
        for split, rows in enriched.items():
            fields = list(dict.fromkeys(k for r in rows for k in r))
            write_csv(temporary/(split+".csv"), rows, fields)
        report = {"status": "historical_metadata_enriched_not_new_labels", "category": category,
                  "input_split_sha256": hashes, "output_split_sha256": {s: file_hash(temporary/(s+".csv")) for s in SPLITS},
                  "source_sha256": {name: v[0] for name, v in sources.items()}, "splits": stats,
                  "scope": "Preserves existing human labels and split membership. Does not infer rejection reasons or supply fresh acceptance."}
        write_json(temporary/"enrichment.json", report)
        temporary.rename(output)
    finally:
        if temporary.exists(): shutil.rmtree(temporary)
    return report


def _labels(rows):
    counts = Counter(r["topic_label"] for r in rows)
    return {"n": len(rows), "labels": {k: counts[k] for k in ("T", "F", "U")}}


def diagnose_rules(gold_path, policy_path, output, probes_path=None):
    before = file_hash(gold_path)
    policy = load_policy(policy_path)
    gold = human_gold(gold_path, policy["category"])
    if not gold: raise ValueError("Empty human reference")
    probes = json.loads(Path(probes_path).read_text()) if probes_path else []
    if not isinstance(probes, list): raise ValueError("Probes must be a list")
    used = set()
    for probe in probes:
        if not isinstance(probe.get("id"), str) or not probe["id"] or probe["id"] in used:
            raise ValueError("Probe IDs must be unique and nonempty")
        used.add(probe["id"])
        fields = probe.get("fields", ["title"])
        if not isinstance(fields, list) or not fields or not set(fields) <= {"title", "channel", "description"}:
            raise ValueError("Probe fields must be metadata only")
        re.compile(probe["pattern"], re.I)
    decisions, cases = [], []
    groups = {key: defaultdict(list) for key in ("channel", "label_file")}
    rules = []
    for group in ("related_patterns", "unrelated_patterns"):
        for rule in policy[group]:
            rules.append({"id": group+":"+rule["id"], "pattern": rule["pattern"], "fields": ["title"]})
    probe_hits = defaultdict(list)
    rule_hits = defaultdict(list)
    for row in gold:
        decision, reason = classify(row["title"], policy)
        pred = {"video_id": row["video_id"], "title_hash": title_key(row["title"]), "decision": decision, "reason": reason}
        decisions.append(pred)
        evidence = {}
        for collection, hits in ((rules, rule_hits), (probes, probe_hits)):
            for rule in collection:
                matches = {field: m.group(0) for field in rule.get("fields", ["title"])
                           for m in [re.search(rule["pattern"], row.get(field, ""), re.I)] if m}
                if matches:
                    hits[rule["id"]].append(dict(row, decision=decision))
                    if collection is probes: evidence[rule["id"]] = matches
        error = "false_keep" if decision == "keep" and row["topic_label"] == "F" else (
            "false_drop" if decision == "drop" and row["topic_label"] == "T" else (
            "unknown_auto_decision" if decision != "review" and row["topic_label"] == "U" else ""))
        cases.append(dict(row, **pred, error_kind=error, observed_signals=json.dumps(evidence, ensure_ascii=False),
                          cause_hypothesis="", diagnosis_reviewer=""))
        for key in groups: groups[key][row.get(key, "").strip().casefold() if key == "channel" else row.get(key, "")].append(dict(row, decision=decision))
    def summarize_hits(rule, rows):
        kept = [r for r in rows if r["decision"] == "keep"]
        return {"id": rule["id"], "pattern": rule["pattern"], "fields": rule.get("fields", ["title"]),
                "matched": _labels(rows), "baseline_keep_matched": _labels(kept),
                "if_vetoed_current_keep": {"T_lost": sum(r["topic_label"] == "T" for r in kept),
                                          "F_removed": sum(r["topic_label"] == "F" for r in kept),
                                          "U_dropped": sum(r["topic_label"] == "U" for r in kept)}}
    report = {"status": "retrospective_diagnosis_not_release_evidence", "category": policy["category"],
              "reference_sha256": before, "policy_hash": digest(policy), "probes_hash": digest(probes),
              "baseline": decision_metrics(gold, decisions),
              "rule_hits": [summarize_hits(r, rule_hits[r["id"]]) for r in rules],
              "probes": [summarize_hits(r, probe_hits[r["id"]]) for r in probes],
              "groups": {key: [{"value": name, **_labels(rows), "keep": _labels([r for r in rows if r["decision"] == "keep"])}
                               for name, rows in sorted(index.items())] for key, index in groups.items()},
              "mixed_label_channels": sum(bool(k) and {"T", "F"} <= {r["topic_label"] for r in rows} for k, rows in groups["channel"].items()),
              "error_counts": dict(Counter(r["error_kind"] for r in cases if r["error_kind"])),
              "scope": "Signals overlap and are observations, not human rejection reasons. Veto impact is measured on current keep only. Historical test is now development/regression material."}
    if file_hash(gold_path) != before: raise ValueError("Gold changed during diagnosis")
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    write_json(output/"diagnosis.json", report)
    write_json(output/"probes.json", probes)
    write_json(output/"policy.json", policy)
    fields = list(dict.fromkeys(k for r in cases for k in r))
    write_csv(output/"cases.csv", cases, fields)
    write_csv(output/"errors.csv", [r for r in cases if r["error_kind"]], fields)
    return report
