"""Human-reference replay and stage attribution. Never authorizes batch delivery."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from core.topic_loop import human_gold, title_key, classify, load_policy, digest, write_csv, write_json, wilson
from core.data_profile import file_hash


def ratio(a, b):
    return a/b if b else None


def decision_metrics(gold: list[dict], decisions: list[dict]) -> dict:
    labels = {r["video_id"]:r for r in gold}
    if len(labels)!=len(gold): raise ValueError("Duplicate gold IDs")
    pred={}
    for row in decisions:
        vid=row["video_id"]
        if vid in pred: raise ValueError("Duplicate decision ID")
        if vid not in labels: raise ValueError("Decision outside reference")
        if row.get("title_hash")!=title_key(labels[vid]["title"]): raise ValueError("Decision title fingerprint mismatch")
        if row["decision"] not in ("keep","drop","review"): raise ValueError("Invalid decision")
        pred[vid]=row
    if set(pred)!=set(labels): raise ValueError("Missing reference decisions")
    totals=Counter(r["topic_label"] for r in gold)
    pools={p:Counter(labels[v]["topic_label"] for v,r in pred.items() if r["decision"]==p) for p in ("keep","drop","review")}
    result={"reference_rows":len(gold),"input_labels":dict(totals),"pools":{p:{"n":sum(c.values()),"labels":dict(c)} for p,c in pools.items()}}
    k,d,r=pools["keep"],pools["drop"],pools["review"]
    result.update(keep_precision_known=ratio(k["T"],k["T"]+k["F"]),
                  keep_T_rate_including_U=ratio(k["T"],sum(k.values())),
                  keep_T_ci90=wilson(k["T"],sum(k.values())),
                  T_drop_rate=ratio(d["T"],totals["T"]), F_removed_rate=ratio(d["F"],totals["F"]),
                  drop_T_share=ratio(d["T"],sum(d.values())), T_keep_coverage=ratio(k["T"],totals["T"]),
                  T_review_rate=ratio(r["T"],totals["T"]),review_rate=ratio(sum(r.values()),len(gold)))
    return result


def evaluate_trace(gold: list[dict], stages: list[tuple[str,list[dict]]]) -> dict:
    """keep continues; drop/review stop. Each stage must cover its entire eligible reference set."""
    eligible=gold[:]
    terminal={}
    reports=[]
    names=set()
    for name,decisions in stages:
        if name in names: raise ValueError("Duplicate stage name")
        names.add(name)
        metrics=decision_metrics(eligible,decisions)
        reports.append({"stage":name,**metrics})
        passed=set()
        for r in decisions:
            terminal[r["video_id"]]=r
            if r["decision"]=="keep": passed.add(r["video_id"])
        eligible=[r for r in eligible if r["video_id"] in passed]
    if not stages: raise ValueError("No stages")
    return {"status":"reference_replay_only_not_batch_acceptance", "stages":reports,
            "cumulative":decision_metrics(gold,list(terminal.values()))}


def replay_rules(gold_path: str | Path, policy_path: str | Path, output: str | Path) -> dict:
    policy=load_policy(policy_path)
    gold=human_gold(gold_path,policy["category"])
    if not gold: raise ValueError("Empty human reference")
    decisions=[]
    for r in gold:
        decision,reason=classify(r["title"],policy)
        decisions.append({"video_id":r["video_id"],"title_hash":title_key(r["title"]),
                          "decision":decision,"reason":reason,"stage":"title_rules"})
    report=evaluate_trace(gold,[("title_rules",decisions)])
    report.update(category=policy["category"],policy_hash=digest(policy),reference_sha256=file_hash(gold_path),
                  evidence_scope="retrospective human regression; selection bias and past development use may apply")
    output=Path(output); output.mkdir(parents=True,exist_ok=False)
    write_json(output/"stage_eval.json",report)
    write_csv(output/"decisions.csv",decisions,["video_id","title_hash","stage","decision","reason"])
    lookup={r["video_id"]:r for r in gold}
    errors=[dict(lookup[d["video_id"]],**d) for d in decisions
            if (d["decision"]=="keep" and lookup[d["video_id"]]["topic_label"]!="T")
            or (d["decision"]=="drop" and lookup[d["video_id"]]["topic_label"] in ("T","U"))]
    write_csv(output/"errors.csv",errors,["video_id","title","topic_label","decision","reason"])
    return report
