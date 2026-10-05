from pathlib import Path
import csv
import json
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'02_脚本'))
from core.gold_diagnostics import enrich_gold, diagnose_rules
from core.topic_loop import write_csv, write_json, read_rows
from core.text_benchmark import text_field


def csv_file(path, rows):
    write_csv(path, rows, list(dict.fromkeys(k for r in rows for k in r)))


def fixture_gold(tmp_path):
    gold = tmp_path/'gold'; gold.mkdir()
    for i, split in enumerate(('train','calibration','test')):
        raw = tmp_path/(split+'_qc_result.csv')
        source = dict(video_id=str(i), title='recipe '+split, channel=split, qc_result='T', description='make rice', duration_seconds='123')
        csv_file(raw, [source])
        row = dict(video_id=str(i), title=source['title'], channel=split, topic_label='T', topic_category='cook',
                   reviewer='original', label_source='human', label_file=str(raw), qc_result='T')
        csv_file(gold/(split+'.csv'), [row])
    return gold


def test_enrich_preserves_gold_identity_order_and_labels(tmp_path):
    gold = fixture_gold(tmp_path)
    before = {p:p.read_bytes() for p in gold.iterdir()}
    output = tmp_path/'enriched'
    report = enrich_gold(gold,'cook',output)
    assert all(p.read_bytes()==value for p,value in before.items())
    for split in ('train','calibration','test'):
        old = list(read_rows(gold/(split+'.csv')))[0]
        new = list(read_rows(output/(split+'.csv')))[0]
        assert all(new[k]==v for k,v in old.items())
        assert new['description']=='make rice' and new['duration_seconds']=='123'
        assert len(new['metadata_source_sha256'])==64
        assert report['splits'][split]['rows']==1
    with pytest.raises(FileExistsError):enrich_gold(gold,'cook',output)


@pytest.mark.parametrize('field,value', [('title','changed'),('channel','changed'),('qc_result','F')])
def test_enrich_rejects_mismatched_source(tmp_path,field,value):
    gold=fixture_gold(tmp_path);raw=tmp_path/'train_qc_result.csv'
    rows=list(read_rows(raw));rows[0][field]=value;csv_file(raw,rows)
    with pytest.raises(ValueError,match='mismatch'):enrich_gold(gold,'cook',tmp_path/'out')
    assert not (tmp_path/'out').exists()


def test_enrich_rejects_conflicting_duplicates_and_missing_id(tmp_path):
    gold=fixture_gold(tmp_path);raw=tmp_path/'train_qc_result.csv';rows=list(read_rows(raw))
    csv_file(raw,[rows[0],dict(rows[0],description='conflict')])
    with pytest.raises(ValueError,match='Conflicting'):enrich_gold(gold,'cook',tmp_path/'out')
    csv_file(raw,[dict(rows[0],video_id='unknown')])
    with pytest.raises(ValueError,match='missing'):enrich_gold(gold,'cook',tmp_path/'out')


def test_enrich_rejects_existing_description_disagreement(tmp_path):
    gold=fixture_gold(tmp_path);p=gold/'test.csv';rows=list(read_rows(p));rows[0]['description']='different';csv_file(p,rows)
    with pytest.raises(ValueError,match='disagrees'):enrich_gold(gold,'cook',tmp_path/'out')


def test_diagnosis_counts_controls_and_only_vetoes_current_keep(tmp_path):
    gold=tmp_path/'gold.csv'
    rows=[dict(video_id=str(i),title=title,channel='shared',topic_label=label,topic_category='cook',label_source='human',reviewer='test',label_file='source')
          for i,(title,label) in enumerate([('recipe chef','T'),('recipe chef home','F'),('home only','T'),('game chef','T')])]
    csv_file(gold,rows)
    policy=tmp_path/'policy.json';write_json(policy,dict(category='cook',version='1',definition='cook',related_patterns=[dict(id='recipe',pattern='recipe')],unrelated_patterns=[dict(id='game',pattern='game')]))
    probes=tmp_path/'probes.json';write_json(probes,[dict(id='home',pattern='home'),dict(id='chef',pattern='chef')])
    original=policy.read_bytes()
    report=diagnose_rules(gold,policy,tmp_path/'out',probes)
    assert report['error_counts']=={'false_keep':1,'false_drop':1}
    assert report['mixed_label_channels']==1
    assert report['probes'][0]['matched']['labels']['T']==1
    assert report['probes'][0]['if_vetoed_current_keep']==dict(T_lost=0,F_removed=1,U_dropped=0)
    assert report['probes'][1]['if_vetoed_current_keep']==dict(T_lost=1,F_removed=1,U_dropped=0)
    assert policy.read_bytes()==original
    cases=list(read_rows(tmp_path/'out'/'cases.csv'))
    assert len(cases)==4 and all(r['cause_hypothesis']=='' for r in cases)
    assert len(list(read_rows(tmp_path/'out'/'errors.csv')))==2


@pytest.mark.parametrize('probes', [[dict(id='x',pattern='a',fields=['topic_label'])],[dict(id='x',pattern='a'),dict(id='x',pattern='b')]])
def test_diagnosis_rejects_label_features_or_ambiguous_ids(tmp_path,probes):
    gold=fixture_gold(tmp_path)
    policy=tmp_path/'p.json';write_json(policy,dict(category='cook',definition='cook',version='1',related_patterns=[],unrelated_patterns=[]))
    p=tmp_path/'probes.json';write_json(p,probes)
    with pytest.raises(ValueError):diagnose_rules(gold/'test.csv',policy,tmp_path/'out',p)


def test_benchmark_description_features_exclude_provenance():
    row=dict(title='cook',channel='chef',description='make rice https://example.org/link',topic_label='F',label_file='REJECTED',metadata_source_sha256='SECRET')
    assert text_field(row,('title','channel','description'))=='cook chef make rice'


def rule_gold(partition, labels):
    return [dict(video_id=partition+str(i), title='forbidden '+partition+' '+str(i), channel=partition,
                 topic_label=lab, topic_category='cook',label_source='human',reviewer='test') for i,lab in enumerate(labels)]


def test_rule_validation_blocks_counterexamples_and_insufficient_support():
    from core.rule_validation import validate_proposals
    p=dict(category='cook',definition='cook',version='1',related_patterns=[],unrelated_patterns=[])
    rules=[dict(name='forbidden',pattern='forbidden')]
    dev=rule_gold('train',['F']*3)
    for labels,expected in [(['F','F','T'],'blocked_human_counterexample'),(['F','F','U'],'blocked_human_counterexample'),(['F'],'insufficient_independent_support'),(['F']*3,'candidate_for_review_not_accepted')]:
        r=validate_proposals(dev,rule_gold('cal',labels),rules,p)
        assert r['rules'][0]['status']==expected
    with pytest.raises(ValueError,match='overlap'):validate_proposals(dev,dev,rules,p)


def test_revision_requires_independent_evidence_and_retains_all_development_titles(tmp_path):
    from core.topic_loop import revise_policy, digest, title_key
    from core.data_profile import file_hash
    p=dict(category='cook',definition='cook',version='1',related_patterns=[],unrelated_patterns=[])
    policy=tmp_path/'policy.json';write_json(policy,p)
    train=tmp_path/'train.csv';cal=tmp_path/'cal.csv'
    dev=rule_gold('train',['F']*3);validation=rule_gold('cal',['F']*3)
    csv_file(train,dev);csv_file(cal,validation)
    proposal=tmp_path/'proposal.json'
    write_json(proposal,dict(parent_policy_hash=digest(p),rules=[dict(name='forbidden',pattern='forbidden')],
                            development_sha256=file_hash(train),validation_sha256=file_hash(cal),development_titles=['earlier-source-title']))
    with pytest.raises(ValueError,match='Independent'):revise_policy(policy,proposal,train,['forbidden'],tmp_path/'bad.json')
    out=tmp_path/'candidate.json'
    revise_policy(policy,proposal,train,['forbidden'],out,cal)
    new=json.loads(out.read_text())
    assert set(new['development_titles'])=={'earlier-source-title'}|{title_key(r['title']) for r in dev+validation}
    assert new['unrelated_patterns']==[dict(id='forbidden',pattern='forbidden')]
    validation[0]['topic_label']='T';csv_file(cal,validation)
    with pytest.raises(ValueError,match='evidence mismatch'):revise_policy(policy,proposal,train,['forbidden'],tmp_path/'stale.json',cal)


def test_proposals_record_all_label_exposure(tmp_path):
    from core.topic_loop import propose, title_key
    from core.data_profile import file_hash
    p=tmp_path/'p.json';write_json(p,dict(category='cook',definition='cook',version='1',related_patterns=[],unrelated_patterns=[]))
    dev=rule_gold('train',['F']*3);val=rule_gold('cal',['F']*3)
    train=tmp_path/'train.csv';cal=tmp_path/'cal.csv';csv_file(train,dev);csv_file(cal,val)
    report=propose(train,p,tmp_path/'proposal.json',cal)
    assert report['development_sha256']==file_hash(train)
    assert report['validation_sha256']==file_hash(cal)
    assert set(report['development_titles'])=={title_key(r['title']) for r in dev+val}


def test_proposals_without_validation_are_never_addable(tmp_path):
    from core.topic_loop import propose
    p=tmp_path/'p.json';write_json(p,dict(category='cook',definition='cook',version='1',related_patterns=[],unrelated_patterns=[]))
    rows=rule_gold('train',['F']*3)
    for row in rows: row['title']='minecraft '+row['title']
    train=tmp_path/'train.csv';csv_file(train,rows)
    report=propose(train,p,tmp_path/'proposal.json')
    assert report['rules']
    assert all(not r['eligible_for_revision'] and 'addable' not in r for r in report['rules'])
