"""Separate-channel checks for mined rule proposals, not batch acceptance."""
from __future__ import annotations
import re
from core.text_benchmark import validate_splits
from core.topic_loop import classify, title_key
from core.stage_eval import decision_metrics


def validate_proposals(development, validation, rules, policy, min_f=3):
    if min_f < 1: raise ValueError("min_f must be positive")
    validate_splits({"development": development, "validation": validation})
    results = []
    for rule in rules:
        rx = re.compile(rule["pattern"], re.I)
        support = {}
        for name, rows in (("development", development), ("validation", validation)):
            support[name] = {lab: sum(r["topic_label"] == lab and bool(rx.search(r["title"])) for r in rows) for lab in ("T", "F", "U")}
        if any(s["T"] or s["U"] for s in support.values()):
            status = "blocked_human_counterexample"
        elif any(s["F"] < min_f for s in support.values()):
            status = "insufficient_independent_support"
        else:
            status = "candidate_for_review_not_accepted"
        candidate = dict(policy, unrelated_patterns=policy["unrelated_patterns"] + [{"id": rule["name"], "pattern": rule["pattern"]}])
        decisions = []
        transitions = {}
        for row in validation:
            before, _ = classify(row["title"], policy)
            after, reason = classify(row["title"], candidate)
            key = before+"->"+after+":"+row["topic_label"]
            transitions[key] = transitions.get(key, 0) + 1
            decisions.append(dict(video_id=row["video_id"], title_hash=title_key(row["title"]), decision=after, reason=reason))
        results.append({"name": rule["name"], "status": status, "support": support,
                        "validation_transitions": transitions,
                        "validation_candidate_metrics": decision_metrics(validation, decisions)})
    return {"minimum_F_support_per_partition": min_f, "rules": results,
            "scope": "Zero observed T/U is only a regression check. Three F examples are not a precision guarantee. Independent batch acceptance is still required."}
