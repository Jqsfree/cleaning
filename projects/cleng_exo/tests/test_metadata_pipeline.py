"""Cross-stage behavior, human-label isolation, and publication regressions."""
from pathlib import Path
import csv
import json
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"02_脚本"))
from core.data_profile import profile_file, iter_metadata
from core.topic_loop import write_csv, write_json, run_round, evaluate, read_rows, title_key
from core.metadata_acceptance import release_round, frozen_manifest
from core.stage_eval import evaluate_trace, decision_metrics
from core.text_benchmark import validate_splits, choose_thresholds


def make_csv(path, rows):
    write_csv(path,rows,list(rows[0]))
    return path


def make_round(tmp_path, *, n=60, sample=60, target=.95, purpose="candidate", extra=None):
    policy=tmp_path/"policy.json"
    write_json(policy,{"category":"cook","version":"1","definition":"cooking",
        "related_patterns":[{"id":"cook","pattern":"cooking"}],"unrelated_patterns":[]})
    src=make_csv(tmp_path/"input.csv",[dict(video_id=str(i),title=f"cooking {i}",channel="c",**(extra or {})) for i in range(n)])
    root=tmp_path/"round"
    run_round(src,policy,root,sample_n=sample,target=target,purpose=purpose)
    return src,policy,root


def fill(root, path, label="T", limit=None):
    rows=list(read_rows(root/"audit.csv"))
    if limit is not None: rows=rows[:limit]
    for r in rows:r.update(topic_label=label,label_source="human",reviewer="test_reviewer")
    return make_csv(path,rows)


def test_profile_multiline_missing_duration_and_conflicts(tmp_path):
    src=make_csv(tmp_path/"input.csv",[
        dict(video_id="a",title="one\nline",channel="",description="",duration_seconds="120"),
        dict(video_id="a",title="different",channel="",description="",duration_seconds="NaN"),
        dict(video_id=" ",title=" ",channel="x",description="text",duration_seconds="")])
    r=profile_file(src,"cook")
    assert r["metrics"]["rows"]==3
    assert r["metrics"]["conflicting_duplicate_rows"]==1
    assert r["metrics"]["duration"]==dict(valid_rows=1,invalid_rows=1,missing_rows=1,valid_hours=120/3600)
    assert r["metrics"]["missing_fields"]["title"]["n"]==1
    assert r["status"]=="blocked_input_contract"


@pytest.mark.parametrize("text",['video_id,title\na,"broken\n', 'video_id,title,title\na,x,y\n', 'video_id,title\na,x,extra\n'])
def test_intake_rejects_malformed_or_duplicate_columns(tmp_path,text):
    p=tmp_path/"bad.csv";p.write_text(text)
    with pytest.raises(ValueError):profile_file(p,"cook")


def test_profile_empty_and_wrong_reference(tmp_path):
    p=tmp_path/"empty.csv";p.write_text("video_id,title\n")
    r=profile_file(p,"cook")
    assert r["status"]=="blocked_input_contract"
    assert r["metrics"]["missing_fields"]["title"]["rate"] is None
    ref=tmp_path/"ref.json";write_json(ref,r)
    with pytest.raises(ValueError,match="Reference"):profile_file(p,"service",reference=ref)


def test_tsv_and_parquet_same_profile(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    tsv=tmp_path/"in.tsv";tsv.write_text("video_id\ttitle\na\tcooking\n")
    par=tmp_path/"in.parquet";pq.write_table(pa.table({"video_id":["a"],"title":["cooking"]}),par)
    assert profile_file(tsv,"cook")["metrics"]==profile_file(par,"cook")["metrics"]


def test_human_columns_are_blank_even_when_input_is_gold(tmp_path):
    _,_,root=make_round(tmp_path,extra={"topic_label":"T","label_source":"human","reviewer":"previous","review_notes":"old"})
    rows=list(read_rows(root/"audit.csv"))
    assert all(all(r[k]=="" for k in ("topic_label","label_source","reviewer","review_notes")) for r in rows)
    assert all("topic_decision" not in r and "topic_score" not in r for r in rows)
    r=json.loads((root/"stage_eval.json").read_text())
    assert r["unique_input"]==sum(r["pool_counts"].values())


def test_accept_census_then_release_identical_snapshot(tmp_path):
    _,_,root=make_round(tmp_path,n=3,sample=3)
    labels=fill(root,tmp_path/"labels.csv")
    r=evaluate(root,labels)
    assert r["status"]=="accepted"
    assert r["metrics"]["keep"]["interval_method"]=="census"
    dest=tmp_path/"released";released=release_round(root,dest)
    assert released["rows"]==3
    assert (dest/"data.csv").read_bytes()==(root/"keep.csv").read_bytes()
    with pytest.raises(FileExistsError):release_round(root,dest)


def test_unconfigured_or_changed_target_cannot_release(tmp_path):
    _,_,root=make_round(tmp_path,target=None)
    labels=fill(root,tmp_path/"labels.csv")
    assert evaluate(root,labels)["status"]=="blocked_acceptance_unconfigured"
    with pytest.raises(ValueError,match="frozen"):evaluate(root,labels,.90)
    with pytest.raises(ValueError):release_round(root,tmp_path/"released")


def test_partial_labels_wait_and_small_sample_is_inconclusive(tmp_path):
    _,_,root=make_round(tmp_path,sample=2)
    assert evaluate(root,fill(root,tmp_path/"partial.csv",limit=1))["status"]=="pending_labels"
    assert evaluate(root,fill(root,tmp_path/"full.csv"))["status"]=="inconclusive"
    with pytest.raises(ValueError):release_round(root,tmp_path/"released")


def test_failure_routes_specialist_for_same_batch_only(tmp_path):
    src,policy,root=make_round(tmp_path,n=5,sample=5)
    r=evaluate(root,fill(root,tmp_path/"labels.csv",label="F"))
    assert r["status"]=="failed_acceptance"
    assert r["next_action"]=="plan_specialist_for_this_batch"
    with pytest.raises(ValueError):release_round(root,tmp_path/"released")
    child=tmp_path/"child"
    run_round(src,policy,child,target=.95,parent_round=root)
    m=frozen_manifest(child)
    assert m["parent_candidate"]==frozen_manifest(root)["candidate_id"]
    assert len(m["development_titles"])==5
    other=make_csv(tmp_path/"other.csv",[dict(video_id="new",title="cooking")])
    with pytest.raises(ValueError,match="same batch"):run_round(other,policy,tmp_path/"bad",target=.95,parent_round=root)


def test_passed_batch_cannot_trigger_specialist(tmp_path):
    src,policy,root=make_round(tmp_path,n=3)
    evaluate(root,fill(root,tmp_path/"labels.csv"))
    with pytest.raises(ValueError,match="failed human"):run_round(src,policy,tmp_path/"child",target=.95,parent_round=root)


def test_pool_mutation_invalidates_acceptance(tmp_path):
    _,_,root=make_round(tmp_path)
    evaluate(root,fill(root,tmp_path/"labels.csv"))
    with (root/"keep.csv").open("a") as f:f.write("tamper\n")
    with pytest.raises(ValueError,match="pool changed"):release_round(root,tmp_path/"released")
    assert not (tmp_path/"released").exists()


def test_forged_report_cannot_turn_failure_into_release(tmp_path):
    _,_,root=make_round(tmp_path)
    r=evaluate(root,fill(root,tmp_path/"labels.csv",label="F"))
    r["status"]="accepted";write_json(root/"human_topic_evaluation.json",r)
    with pytest.raises(ValueError,match="not passed"):release_round(root,tmp_path/"released")


def test_regression_labels_never_grant_release(tmp_path):
    _,_,root=make_round(tmp_path,purpose="regression")
    r=evaluate(root,fill(root,tmp_path/"labels.csv"))
    assert r["quality_decision"]=="accepted"
    assert r["status"]=="retrospective_replay_not_acceptance"
    with pytest.raises(ValueError,match="Regression"):release_round(root,tmp_path/"released")


def test_channel_edit_invalidates_audit(tmp_path):
    _,_,root=make_round(tmp_path,n=1)
    labels=fill(root,tmp_path/"labels.csv")
    rows=list(read_rows(labels));rows[0]["channel"]="different"
    make_csv(labels,rows)
    with pytest.raises(ValueError,match="metadata was changed"):evaluate(root,labels)


def test_zero_keep_is_blocked(tmp_path):
    src=make_csv(tmp_path/"in.csv",[dict(video_id="a",title="unknown")])
    p=tmp_path/"p.json";write_json(p,dict(category="cook",definition="cook",version="1",related_patterns=[],unrelated_patterns=[]))
    root=tmp_path/"round";run_round(src,p,root,target=.95)
    assert evaluate(root,fill(root,tmp_path/"labels.csv"))["status"]=="blocked_no_candidates"


def test_stage_replay_reports_conditional_and_cumulative_losses():
    gold=[dict(video_id=str(i),title=str(i),topic_label="T" if i<3 else "F") for i in range(5)]
    def d(i,p):return dict(video_id=str(i),title_hash=title_key(str(i)),decision=p)
    first=[d(i,"drop" if i==4 else "keep") for i in range(5)]
    second=[d(i,"drop" if i==0 else "keep") for i in range(4)]
    r=evaluate_trace(gold,[("a",first),("b",second)])
    assert r["stages"][0]["F_removed_rate"]==.5
    assert r["stages"][1]["T_drop_rate"]==pytest.approx(1/3)
    assert r["cumulative"]["T_keep_coverage"]==pytest.approx(2/3)
    assert r["cumulative"]["keep_precision_known"]==pytest.approx(2/3)
    with pytest.raises(ValueError,match="Missing"):evaluate_trace(gold,[("bad",first[:-1])])


def test_unknown_and_zero_denominators_not_silently_positive():
    gold=[dict(video_id="u",title="x",topic_label="U")]
    m=decision_metrics(gold,[dict(video_id="u",title_hash=title_key("x"),decision="keep")])
    assert m["keep_precision_known"] is None
    assert m["keep_T_rate_including_U"]==0
    assert m["T_drop_rate"] is None


def test_channel_leakage_rejected_and_thresholds_fail_closed():
    train=[dict(video_id="1",title="x",channel="same")]
    test=[dict(video_id="2",title="y",channel="same")]
    with pytest.raises(ValueError,match="channel overlap"):validate_splits({"train":train,"test":test})
    t=choose_thresholds(["T","F","U"],[.9,.1,.5],.95,20)
    assert t["keep"] is None and t["drop"] is None


def test_legacy_exo_delivery_cannot_fallback_to_quality(tmp_path):
    import importlib.util
    from types import SimpleNamespace
    path=Path(__file__).resolve().parents[1]/"02_脚本/pipeline/orchestrate.py"
    spec=importlib.util.spec_from_file_location("test_orchestrate_gate",path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    q=tmp_path/"01_quality";q.mkdir()
    make_csv(q/"source_quality_20260921.csv",[dict(video_id="a",title="cooking")])
    with pytest.raises(SystemExit,match="accepted-round"):
        module._run_deliver(tmp_path,"exo_cook","machine",{},{},SimpleNamespace())
    assert not (tmp_path/"07_deliver").exists()


def test_legacy_pending_lot_and_nonempty_folder_do_not_authorize_delivery(tmp_path):
    from core.lot_accept import prepare_deliver
    from core.run_manifest import init_manifest
    from core.batch_sop import stage_done
    init_manifest(tmp_path,category="exo_cook",source="machine",batch="test")
    src=make_csv(tmp_path/"source.csv",[dict(video_id="a",title="cooking")])
    with pytest.raises(ValueError,match="pending lots"):
        prepare_deliver(tmp_path,lot_csv=src,sample_frame="clean_keep",deliver_name="deliver.csv")
    (tmp_path/"07_deliver").mkdir();(tmp_path/"07_deliver/legacy.csv").write_text("legacy")
    assert not stage_done(tmp_path,{"id":"deliver"},{})


@pytest.mark.parametrize("field,value",[("counts",{"keep":1}),("development_titles",["different"]),("input_sha256","other")])
def test_manifest_semantics_are_bound_to_candidate_identity(tmp_path,field,value):
    _,_,root=make_round(tmp_path)
    m=json.loads((root/"manifest.json").read_text());m[field]=value
    write_json(root/"manifest.json",m)
    with pytest.raises(ValueError,match="fingerprint"):frozen_manifest(root)


def test_changed_snapshot_of_human_labels_cannot_release(tmp_path):
    _,_,root=make_round(tmp_path)
    report=evaluate(root,fill(root,tmp_path/"labels.csv"))
    with (root/report["labels_snapshot"]).open("a") as f:f.write("changed\n")
    with pytest.raises(ValueError,match="snapshot changed"):release_round(root,tmp_path/"released")


def test_baseline_can_automatically_replay_human_reference(tmp_path):
    src,policy,_=make_round(tmp_path,n=3)
    gold=make_csv(tmp_path/"reference.csv",[
        dict(video_id="g",title="cooking reference",topic_category="cook",topic_label="F",label_source="human",reviewer="known")])
    out=tmp_path/"with_reference"
    run_round(src,policy,out,target=.95,reference_gold=gold)
    report=json.loads((out/"reference_eval.json").read_text())
    assert report["cumulative"]["keep_T_rate_including_U"]==0
    assert report["status"]=="reference_replay_only_not_batch_acceptance"
    assert not (out/"human_topic_evaluation.json").exists()


def test_manifest_cannot_keep_completed_stages_for_another_input(tmp_path):
    from core.run_manifest import init_manifest,update_stage,load_manifest
    init_manifest(tmp_path,category="exo_cook",source="machine",batch="b",input_path="old.csv")
    update_stage(tmp_path,"quality",stats={"rows":10})
    with pytest.raises(ValueError,match="Input changed"):
        init_manifest(tmp_path,category="exo_cook",source="machine",batch="b",input_path="new.csv")
    assert load_manifest(tmp_path)["input"]=="old.csv"
    with pytest.raises(ValueError,match="identity changed"):
        init_manifest(tmp_path,category="exo_parent",source="machine",batch="b")


def test_partially_filled_worksheet_stays_pending_without_fake_labels(tmp_path):
    _,_,root=make_round(tmp_path,n=3)
    rows=list(read_rows(root/"audit.csv"))
    rows[0].update(topic_label="T",label_source="human",reviewer="r")
    labels=make_csv(tmp_path/"partial.csv",rows)
    report=evaluate(root,labels)
    assert report["status"]=="pending_labels"
    assert report["metrics"]["keep"]["labeled"]==1
    assert report["metrics"]["keep"]["U"]==0
