"""Frozen candidate acceptance and atomic release. Replays never grant publication."""
from __future__ import annotations

import csv
import fcntl
import json
import os
import shutil
import tempfile
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from core.data_profile import file_hash
from core.topic_loop import digest, human_gold, read_rows, validate_human_rows, title_key, wilson, write_csv, write_json

AUDIT_FIELDS=["video_id","title","channel","description","topic_category","topic_title_hash",
              "topic_label","label_source","reviewer","unrelated_topic","review_notes"]


def metadata_hash(row):
    return digest({k:row.get(k,"") for k in ("title","channel","description")})


def candidate_id(manifest):
    return digest({k:manifest[k] for k in ("category","source_hash","policy_hash","model_hash",
        "verifier_hash","pool_hashes","audit_expected","audit_metadata","acceptance_plan","purpose","parent_candidate",
        "counts","unique_rows","sample_size_requested","seed","development_titles","input_sha256")})


@contextmanager
def round_lock(root):
    with (Path(root)/".acceptance.lock").open("a") as f:
        fcntl.flock(f.fileno(),fcntl.LOCK_EX)
        try: yield
        finally: fcntl.flock(f.fileno(),fcntl.LOCK_UN)


def frozen_manifest(root):
    root=Path(root)
    m=json.loads((root/"manifest.json").read_text())
    if m.get("schema_version")!=2: raise ValueError("Legacy round: rerun to freeze candidate and acceptance plan")
    if m["policy_hash"]!=digest(m["policy"]): raise ValueError("Policy fingerprint mismatch")
    if m["candidate_id"]!=candidate_id(m): raise ValueError("Candidate fingerprint mismatch")
    for name,expected in m["pool_hashes"].items():
        if file_hash(root/(name+".csv"))!=expected: raise ValueError("Candidate pool changed: "+name)
    return m


def load_audit_gold(path, manifest):
    """Incomplete worksheets remain pending; nonblank labels must have human provenance."""
    allowed=set().union(*(set(v) for v in manifest["audit_expected"].values()))
    labeled=[]
    for row in read_rows(path):
        if row["video_id"] not in allowed: raise ValueError("Labels outside frozen random audit")
        if metadata_hash(row)!=manifest["audit_metadata"][row["video_id"]]:
            raise ValueError("Audit metadata was changed")
        if row.get("topic_label","").strip(): labeled.append(row)
    return validate_human_rows(labeled,manifest["category"])


def assess(m, gold, target=None):
    plan=m["acceptance_plan"]
    if target is not None and target!=plan["target"]:
        raise ValueError("Acceptance target must be frozen before sampling; create a new candidate")
    target=plan["target"]
    lookup={r["video_id"]:r for r in gold}
    expected=m["audit_expected"]
    allowed=set().union(*(set(v) for v in expected.values()))
    if not set(lookup)<=allowed: raise ValueError("Labels outside frozen random audit")
    development=set(m["development_titles"])
    report={"category":m["category"],"candidate_id":m["candidate_id"],"policy_hash":m["policy_hash"],
            "target":target,"confidence":90,"status":"pending_labels","metrics":{},"errors":[]}
    for pool,members in expected.items():
        rows=[]
        for vid,key in members.items():
            if vid not in lookup: continue
            row=lookup[vid]
            if title_key(row["title"])!=key or metadata_hash(row)!=m["audit_metadata"][vid]:
                raise ValueError("Audit metadata was changed")
            if key in development and m["purpose"]=="candidate":
                raise ValueError("Audit/train overlap: use regression mode or a new independent sample")
            rows.append(row)
        c=Counter(r["topic_label"] for r in rows)
        size=m["counts"].get(pool,0)
        rate=c["T"]/len(rows) if rows else None
        census=size>0 and len(rows)==size
        interval=[rate,rate] if census else wilson(c["T"],len(rows))
        report["metrics"][pool]={"population":size,"requested":len(members),"labeled":len(rows),
            "complete":len(rows)==len(members),"T":c["T"],"F":c["F"],"U":c["U"],
            "topic_related_rate":rate,"related_rate_ci90":interval,"interval_method":"census" if census else "wilson_90"}
        for r in rows:
            if (pool=="keep" and r["topic_label"]!="T") or (pool=="drop" and r["topic_label"] in ("T","U")):
                report["errors"].append(dict(r,predicted_pool=pool))
    loss_limit=plan.get("max_t_loss")
    if loss_limit is not None:
        from core.sampling_risk import risk_bounds
        risk=risk_bounds(report["metrics"])
        report["loss_risk"]=dict(risk,limit=loss_limit)
        # Use the same simultaneous bounds for the keep gate when both apply.
        keep=report["metrics"]["keep"]
        if keep["population"]:
            keep["related_rate_ci90"]=[v/keep["population"] for v in risk["count_bounds"]["keep"]]
            keep["interval_method"]="census" if keep["labeled"]==keep["population"] else "bonferroni_wilson_joint_90"
    k=report["metrics"]["keep"]
    if not k["population"]:
        report["status"]="blocked_no_candidates"
    elif target is None:
        report["status"]="blocked_acceptance_unconfigured"
    elif all(x["complete"] for x in report["metrics"].values()):
        lo,hi=k["related_rate_ci90"]
        if loss_limit is None:
            report["status"]="accepted" if lo>=target else "failed_acceptance" if hi<target else "inconclusive"
        else:
            loss_lo,loss_hi=report["loss_risk"]["interval"]
            if hi<target or loss_lo>loss_limit: report["status"]="failed_acceptance"
            elif lo>=target and loss_hi<=loss_limit: report["status"]="accepted"
            else: report["status"]="inconclusive"
    report["quality_decision"]=report["status"]
    if m["purpose"]=="regression": report["status"]="retrospective_replay_not_acceptance"
    report["next_action"]={"accepted":"release_frozen_candidate", "failed_acceptance":"plan_specialist_for_this_batch",
        "pending_labels":"complete_human_labels", "inconclusive":"review_sampling_plan_no_release",
        "retrospective_replay_not_acceptance":"use_for_regression_only"}.get(report["status"],"resolve_blocker")
    report["note"]="U counts as non-pass. Historical replay is not batch acceptance. Loss bounds apply only when configured before the random audit."
    return report


def verified_evaluation(root, manifest=None):
    """Recompute saved acceptance without trusting the status string or writing files."""
    root=Path(root);m=manifest if manifest is not None else frozen_manifest(root)
    if not (root/"human_topic_evaluation.json").is_file(): raise ValueError("No human evaluation; candidate has not passed acceptance")
    report=json.loads((root/"human_topic_evaluation.json").read_text())
    if report["candidate_id"]!=m["candidate_id"]: raise ValueError("Acceptance belongs to another candidate")
    labels=(root/report["labels_snapshot"]).resolve()
    if root.resolve() not in labels.parents: raise ValueError("Invalid label snapshot path")
    if file_hash(labels)!=report["labels_sha256"]: raise ValueError("Human label snapshot changed")
    checked=assess(m,load_audit_gold(labels,m))
    if report["status"]!=checked["status"]: raise ValueError("Acceptance validation not passed: saved status disagrees with human evidence")
    checked.update(labels_sha256=report["labels_sha256"],labels_snapshot=report["labels_snapshot"])
    return checked


def inspect_round(root, *, verify=True):
    root=Path(root)
    if verify:
        m=frozen_manifest(root)
        report=verified_evaluation(root,m) if (root/"human_topic_evaluation.json").exists() else None
    else:
        m=json.loads((root/"manifest.json").read_text());report=None
    result={"round":str(root.resolve()),"category":m.get("category"),"candidate_id":m.get("candidate_id"),
            "purpose":m.get("purpose"),"counts":m.get("counts",{}),"input_sha256":m.get("input_sha256"),
            "integrity":"verified" if verify else "not_checked","release_eligible":False}
    if not verify: result.update(status="recorded_candidate_unverified",next_action="verify_candidate_and_audit")
    elif report:
        result.update(status=report["status"],next_action=report["next_action"],release_eligible=report["status"]=="accepted")
    else: result.update(status="pending_labels",next_action="complete_human_labels",audit=str(root/"audit.csv"))
    return result


def evaluate_round(root, labels_path, target=None):
    root=Path(root)
    with round_lock(root):
        m=frozen_manifest(root)
        labels_hash=file_hash(labels_path)
        gold=load_audit_gold(labels_path,m)
        report=assess(m,gold,target)
        if file_hash(labels_path)!=labels_hash: raise ValueError("Labels changed during evaluation")
        audit_dir=root/"audits";audit_dir.mkdir(exist_ok=True)
        dest=audit_dir/(labels_hash+".csv")
        if not dest.exists():
            temp=dest.with_suffix(".tmp")
            shutil.copyfile(labels_path,temp)
            if file_hash(temp)!=labels_hash: raise ValueError("Label snapshot changed")
            temp.replace(dest)
        report.update(labels_sha256=labels_hash,labels_snapshot=str(dest.relative_to(root)))
        write_json(audit_dir/(labels_hash+".json"),report)
        # Atomic latest-report pointer; the candidate manifest is immutable.
        temp=root/".human_topic_evaluation.tmp"
        write_json(temp,report);temp.replace(root/"human_topic_evaluation.json")
        write_csv(root/"human_topic_errors.csv",report["errors"],
                  ["video_id","title","topic_label","predicted_pool","unrelated_topic","review_notes"])
        return report


def release_round(root, output):
    root=Path(root);output=Path(output)
    with round_lock(root):
        m=frozen_manifest(root)
        if m["purpose"]!="candidate": raise ValueError("Regression experiments cannot be released")
        report=verified_evaluation(root,m)
        if report["status"]!="accepted": raise ValueError("Candidate has not passed independent human acceptance")
        if output.exists(): raise FileExistsError(output)
        output.parent.mkdir(parents=True,exist_ok=True)
        staging=Path(tempfile.mkdtemp(prefix=".release-",dir=output.parent))
        try:
            shutil.copyfile(root/"keep.csv",staging/"data.csv")
            if file_hash(staging/"data.csv")!=m["pool_hashes"]["keep"]: raise ValueError("Export differs from accepted snapshot")
            frozen_manifest(root)
            result={"status":"released","category":m["category"],"candidate_id":m["candidate_id"],
                    "rows":m["counts"]["keep"],"input_sha256":m["input_sha256"],"data_sha256":m["pool_hashes"]["keep"],
                    "acceptance_plan":m["acceptance_plan"],
                    "labels_sha256":report["labels_sha256"],"round":str(root.resolve())}
            write_json(staging/"release.json",result)
            # Cooperating releases of this candidate share the same round lock.
            if output.exists(): raise FileExistsError(output)
            staging.rename(output)
        except BaseException:
            shutil.rmtree(staging,ignore_errors=True)
            raise
        return result


def verified_release(directory):
    """Validate a release receipt against its frozen round and original human evidence."""
    directory=Path(directory)
    receipt=json.loads((directory/"release.json").read_text())
    m=frozen_manifest(receipt["round"])
    evidence=verified_evaluation(receipt["round"],m)
    expected={"status":"released","category":m["category"],"candidate_id":m["candidate_id"],
              "rows":m["counts"]["keep"],"input_sha256":m["input_sha256"],
              "data_sha256":m["pool_hashes"]["keep"],"labels_sha256":evidence["labels_sha256"]}
    if m["purpose"]!="candidate" or evidence["status"]!="accepted": raise ValueError("Release lacks accepted human evidence")
    if any(receipt.get(k)!=v for k,v in expected.items()): raise ValueError("Release receipt disagrees with frozen evidence")
    if "acceptance_plan" in receipt and receipt["acceptance_plan"]!=m["acceptance_plan"]: raise ValueError("Release acceptance plan changed")
    if file_hash(directory/"data.csv")!=m["pool_hashes"]["keep"]: raise ValueError("Released data changed")
    return receipt
