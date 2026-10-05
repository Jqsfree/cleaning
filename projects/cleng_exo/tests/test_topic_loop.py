from pathlib import Path
import csv,json,sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"02_脚本"))
from core.topic_loop import *

def policy(tmp):
    p=tmp/"policy.json"
    write_json(p,dict(category="test",definition="cooking",version="1",related_patterns=[dict(id="cook",pattern="cooking")],unrelated_patterns=[dict(id="game",pattern="minecraft")]))
    return p

def csvfile(path,rows):
    write_csv(path,rows,list(rows[0]))
    return path

def test_conflict_and_unknown(tmp_path):
    p=load_policy(policy(tmp_path))
    assert classify("cooking minecraft",p)[0]=="review"
    assert classify("unfamiliar language",p)[0]=="review"
    assert classify("",p)[0]=="review"
    assert classify("cooking tutorial",p)[0]=="keep"

def test_round_no_channel_veto_no_label_leak(tmp_path):
    p=policy(tmp_path)
    src=csvfile(tmp_path/"in.csv",[dict(video_id=str(i),title="cooking",channel="minecraft",qc_result="F") for i in range(20)])
    out=tmp_path/"round"
    r=run_round(src,p,out,sample_n=7)
    assert r["counts"]=={"keep":20}
    audit=list(read_rows(out/"audit_keep.csv"))
    assert len(audit)==7
    assert all(x["topic_label"]=="" for x in audit)
    assert "qc_result" not in audit[0]
    with pytest.raises(FileExistsError): run_round(src,p,out)

def test_evaluate_rejects_missing_provenance_and_changed_titles(tmp_path):
    p=policy(tmp_path)
    src=csvfile(tmp_path/"in.csv",[dict(video_id="a",title="cooking")])
    out=tmp_path/"round"; run_round(src,p,out)
    row=list(read_rows(out/"audit_keep.csv"))[0]
    row.update(topic_label="T",label_source="human",reviewer="tester")
    csvfile(tmp_path/"gold.csv",[row])
    result=evaluate(out,tmp_path/"gold.csv")
    assert result["status"]=="blocked_acceptance_unconfigured" # policy must be frozen before sampling
    row["title"]="changed"
    csvfile(tmp_path/"bad.csv",[row])
    with pytest.raises(ValueError,match="changed"): evaluate(out,tmp_path/"bad.csv")

def test_legacy_labels_not_silently_used(tmp_path):
    p=csvfile(tmp_path/"old.csv",[dict(video_id="a",title="cooking",qc_result="T")])
    with pytest.raises(ValueError): human_gold(p,"test")

def test_import_conflicts_playback_and_channel_split(tmp_path):
    source=tmp_path/"source";source.mkdir()
    rows=[dict(video_id=str(i),title="title "+str(i),channel="c"+str(i//2),qc_result="T" if i%2 else "F") for i in range(100)]
    rows.extend([dict(video_id="0",title="title 0",channel="c0",qc_result="T"),
                 dict(video_id="bad",title="x",channel="y",qc_result="F | 无法播放")])
    csvfile(source/"x_qc_result.csv",rows)
    out=tmp_path/"gold"; report=import_metadata_gold(source,"test",out)
    assert report["conflict_rows"]==2
    assert len(report["excluded"])==1
    groups=[{r["channel"] for r in read_rows(out/(s+".csv"))} for s in ("train","calibration","test")]
    assert not groups[0]&groups[1] and not groups[0]&groups[2] and not groups[1]&groups[2]

def test_sample_size_and_wilson():
    assert sample_size()==271
    lo,hi=wilson(271,271)
    assert .98<lo<1 and hi<=1

def test_invalid_csv_fails_closed(tmp_path):
    path=tmp_path/"broken.csv"
    path.write_text("video_id,title\na,title,extra\n")
    with pytest.raises(ValueError): list(read_rows(path))

def test_close_requires_two_independent_passed_rounds(tmp_path):
    with pytest.raises(ValueError): close_loop([],tmp_path/"closed.json")

def test_rule_revision_rejects_human_positive_hits(tmp_path):
    p=policy(tmp_path)
    proposals=tmp_path/"proposals.json"
    write_json(proposals,dict(parent_policy_hash=digest(load_policy(p)),rules=[dict(name="bad",pattern="cooking")]))
    rows=[dict(video_id=str(i),title="cooking "+str(i),topic_category="test",topic_label="T" if i==0 else "F",label_source="human",reviewer="r") for i in range(4)]
    labels=csvfile(tmp_path/"labels.csv",rows)
    with pytest.raises(ValueError,match="regression"):
        revise_policy(p,proposals,labels,["bad"],tmp_path/"new.json")

def test_verified_fingerprint_cannot_be_reused_on_other_input(tmp_path):
    p=policy(tmp_path)
    src=csvfile(tmp_path/"input.csv",[dict(video_id="a",title="cooking")])
    report=tmp_path/"verifier.json"
    write_json(report,dict(verified_sha256="wrong",policy_hash=digest(load_policy(p))))
    with pytest.raises(ValueError,match="fingerprint"):
        run_round(src,p,tmp_path/"round",verifier_report=report)

def test_numerical_prediction_is_finite():
    import numpy as np
    from types import SimpleNamespace
    clf=SimpleNamespace(coef_=np.array([[1.,-1.]]),intercept_=np.array([0.]))
    assert predict_scores(clf,np.array([[1.,1.]]))[0]==.5
    with pytest.raises(ValueError): predict_scores(clf,np.array([[np.nan,1.]]))
