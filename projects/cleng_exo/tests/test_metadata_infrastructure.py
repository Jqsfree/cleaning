import json
import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'02_脚本'))
from core.topic_loop import write_csv,write_json,read_rows,evaluate,title_key
from core.metadata_workflow import start_batch,batch_status,retry_batch,specialist_batch,publish_batch,catalog
from core.run_manifest import init_manifest,load_manifest,update_stage
from core.data_profile import file_hash
from core.metadata_acceptance import inspect_round


def setup_batch(tmp_path, **options):
    src=tmp_path/'input.csv'
    write_csv(src,[dict(video_id=str(i),title=('cook ' if i<8 else 'game ')+str(i),channel='c') for i in range(10)],['video_id','title','channel'])
    policy=tmp_path/'policy.json';write_json(policy,dict(category='exo_cook',version='1',definition='cook',related_patterns=[dict(id='cook',pattern='cook')],unrelated_patterns=[dict(id='game',pattern='game')]))
    result=start_batch(src,policy,tmp_path/'runs','new',target=.9,sample_size=30,**options)
    return src,policy,result


def labels_for(result,tmp_path,*,fail=False,drop_label='F'):
    rows=list(read_rows(Path(result['round'])/'audit.csv'))
    for r in rows:r.update(topic_label='F' if fail else ('T' if r['title'].startswith('cook') else drop_label),label_source='human',reviewer='test')
    p=tmp_path/'audit.csv';write_csv(p,rows,list(rows[0]));return p


def test_managed_batch_full_lifecycle_and_catalog(tmp_path):
    src,policy,result=setup_batch(tmp_path,max_t_loss=.1)
    root=Path(result['batch_root']);assert batch_status(root)['status']=='pending_labels'
    assert not batch_status(root,verify=False)['release_eligible']
    with pytest.raises(ValueError):publish_batch(root)
    assert evaluate(result['round'],labels_for(result,tmp_path))['status']=='accepted'
    assert batch_status(root)['release_eligible']
    release=publish_batch(root);assert Path(release['deliver_path']).read_bytes()==(Path(result['round'])/'keep.csv').read_bytes()
    assert publish_batch(root)['candidate_id']==release['candidate_id']
    assert batch_status(root)['status']=='released'
    report=catalog(tmp_path/'runs');assert report['batch_count']==1 and not report['batches'][0]['release_eligible']
    with pytest.raises(ValueError):retry_batch(root)
    with pytest.raises(ValueError):specialist_batch(root,policy)
    with pytest.raises(FileExistsError):start_batch(src,policy,tmp_path/'runs','new',target=.9)


def test_same_path_input_change_blocks_publication(tmp_path):
    src,policy,result=setup_batch(tmp_path)
    evaluate(result['round'],labels_for(result,tmp_path))
    src.write_text(src.read_text().replace('cook 0','cook X'))
    assert batch_status(result['batch_root'])['status']=='invalid_batch_input'
    with pytest.raises(ValueError,match='changed'):publish_batch(result['batch_root'])


def test_failed_acceptance_routes_specialist_preserving_targets(tmp_path):
    src,policy,result=setup_batch(tmp_path,max_t_loss=.1)
    assert evaluate(result['round'],labels_for(result,tmp_path,fail=True))['status']=='failed_acceptance'
    assert batch_status(result['batch_root'])['next_action']=='plan_specialist_for_this_batch'
    child=specialist_batch(result['batch_root'],policy)
    m=json.loads((Path(child['round'])/'manifest.json').read_text())
    assert m['acceptance_plan']['target']==.9 and m['acceptance_plan']['max_t_loss']==.1
    assert m['parent_candidate']==result['candidate_id']
    assert len(m['development_titles'])==10


def test_retry_uses_new_attempt_and_same_config(tmp_path,monkeypatch):
    import core.metadata_workflow as workflow
    actual=workflow.run_round
    def fail(*args,**kwargs):raise OSError('simulated interruption')
    monkeypatch.setattr(workflow,'run_round',fail)
    with pytest.raises(OSError):setup_batch(tmp_path)
    root=tmp_path/'runs/exo_cook/machine_new'
    assert batch_status(root)['status']=='execution_failed'
    monkeypatch.setattr(workflow,'run_round',actual)
    result=retry_batch(root)
    assert result['attempt'].endswith('attempt_0002')
    assert batch_status(root)['status']=='pending_labels'
    attempts=list((root/'06_tools/metadata').glob('attempt_*/operation.json'))
    assert len(attempts)==2
    assert json.loads(attempts[0].read_text())['config']==json.loads(attempts[1].read_text())['config']


def test_status_recomputes_forged_accepted_report(tmp_path):
    src,policy,result=setup_batch(tmp_path)
    evaluate(result['round'],labels_for(result,tmp_path,fail=True))
    p=Path(result['round'])/'human_topic_evaluation.json';report=json.loads(p.read_text());report['status']='accepted';write_json(p,report)
    assert batch_status(result['batch_root'])['status']=='invalid_evidence'
    with pytest.raises(ValueError):inspect_round(result['round'])
    with pytest.raises(ValueError):publish_batch(result['batch_root'])


def test_legacy_catalog_does_not_claim_delivery_or_modify_records(tmp_path):
    root=tmp_path/'runs/exo_cook/machine_old';root.mkdir(parents=True)
    p=root/'manifest.json';write_json(p,dict(category='exo_cook',source='machine',batch='old',input='',stages={},deliver_path='old_keep.csv'))
    original=p.read_bytes();result=catalog(tmp_path/'runs')
    assert result['batches'][0]['status']=='legacy_unverified'
    assert not result['batches'][0]['release_eligible'] and p.read_bytes()==original
    broken=tmp_path/'runs/exo_parent/machine_bad';broken.mkdir(parents=True);(broken/'manifest.json').write_text('{')
    assert catalog(tmp_path/'runs')['batch_count']==2


def test_manifest_corruption_and_same_path_input_are_not_reset(tmp_path):
    root=tmp_path/'batch';root.mkdir();p=root/'manifest.json';p.write_text('{broken')
    with pytest.raises(ValueError):init_manifest(root,category='exo_cook',source='machine',batch='x')
    assert p.read_text()=='{broken'
    src=tmp_path/'input.csv';src.write_text('video_id,title\na,cook\n')
    init_manifest(root,category='exo_cook',source='machine',batch='x',input_path=str(src),reinit=True)
    expected=file_hash(src);assert load_manifest(root)['input_sha256']==expected
    update_stage(root,'quality',paths={'data':'out.csv'})
    src.write_text('video_id,title\na,game\n')
    with pytest.raises(ValueError):init_manifest(root,category='exo_cook',source='machine',batch='x',input_path=str(src))
    with pytest.raises(ValueError):update_stage(root,'clean',stats={'n':1})
    assert 'clean' not in load_manifest(root)['stages']


def test_rule_cache_is_outside_source_and_rule_version_specific(tmp_path,monkeypatch):
    from core.rules_loader import save_hit_cache,load_hit_cache,_hit_cache_path
    monkeypatch.setenv('EXO_CACHE_DIR',str(tmp_path/'runtime'))
    rules=tmp_path/'source';rules.mkdir();p=rules/'blacklist.toml';p.write_text('[meta]\nversion="1"\n')
    save_hit_cache(rules,{'pass2':{'a':4}});save_hit_cache(rules,{'pass2':{'a':2,'b':1}})
    assert load_hit_cache(rules)=={'pass2':{'a':4,'b':1}}
    assert _hit_cache_path(rules).is_relative_to(tmp_path/'runtime')
    assert not (rules/'.rule_hits_cache.json').exists()
    p.write_text('[meta]\nversion="2"\n');assert load_hit_cache(rules)=={}


def test_unavailable_cache_does_not_fail_cleaning(tmp_path,monkeypatch):
    from core.rules_loader import save_hit_cache
    blocked=tmp_path/'not_a_dir';blocked.write_text('x');monkeypatch.setenv('EXO_CACHE_DIR',str(blocked))
    save_hit_cache(tmp_path,{'pass2':{'a':1}})


def test_stratified_loss_detects_false_deletion_and_counts_U_conservatively(tmp_path):
    _,_,result=setup_batch(tmp_path,max_t_loss=.1,audit_sizes={'keep':8,'drop':2,'review':1})
    report=evaluate(result['round'],labels_for(result,tmp_path,drop_label='T'))
    assert report['metrics']['keep']['topic_related_rate']==1
    assert report['status']=='failed_acceptance'
    assert report['loss_risk']['interval']==[.2,.2]
    report=evaluate(result['round'],labels_for(result,tmp_path,drop_label='U'))
    assert report['status']=='failed_acceptance' and report['loss_risk']['interval']==[.2,.2]


def test_risk_weights_pool_sizes_and_insufficient_sample_stays_uncertain():
    from core.sampling_risk import risk_bounds
    metrics={'keep':dict(population=100,labeled=100,T=100,F=0,U=0),
             'drop':dict(population=10000,labeled=10,T=1,F=9,U=0),
             'review':dict(population=0,labeled=0,T=0,F=0,U=0)}
    result=risk_bounds(metrics)
    assert result['estimate']==pytest.approx(1000/1100)
    assert result['interval'][1]>.9
    metrics['drop'].update(T=0,F=10)
    result=risk_bounds(metrics);assert result['estimate']==0 and result['interval'][1]>.9


def test_pool_audit_sizes_are_frozen(tmp_path):
    _,_,result=setup_batch(tmp_path,audit_sizes={'keep':2,'drop':1,'review':4})
    m=json.loads((Path(result['round'])/'manifest.json').read_text())
    assert len(m['audit_expected']['keep'])==2 and len(m['audit_expected']['drop'])==1
    m['acceptance_plan']['audit_sizes']['keep']=3;write_json(Path(result['round'])/'manifest.json',m)
    with pytest.raises(ValueError):inspect_round(result['round'])


def test_unified_cli_dispatch_and_compatibility_help(tmp_path,capsys):
    import subprocess
    from core.metadata_cli import main
    _,_,result=setup_batch(tmp_path)
    assert main(['catalog','--runs-root',str(tmp_path/'runs')])==0
    report=json.loads(capsys.readouterr().out);assert report['batch_count']==1
    assert main(['status','--batch-root',result['batch_root']])==0
    assert json.loads(capsys.readouterr().out)['status']=='pending_labels'
    assert main(['next','--round',result['round']])==0
    assert json.loads(capsys.readouterr().out)['status']=='pending_labels'
    root=Path(__file__).resolve().parents[1]
    for path in ('02_脚本/pipeline/metadata.py','02_脚本/tools/batch_ops/topic_loop.py'):
        completed=subprocess.run([sys.executable,str(root/path),'--help'],capture_output=True,text=True)
        assert completed.returncode==0 and 'catalog' in completed.stdout and 'specialist' in completed.stdout


def test_retry_does_not_accept_changed_sampling_config(tmp_path,monkeypatch):
    import core.metadata_workflow as workflow
    def fail(*args,**kwargs):raise OSError('stop')
    monkeypatch.setattr(workflow,'run_round',fail)
    with pytest.raises(OSError):setup_batch(tmp_path)
    root=tmp_path/'runs/exo_cook/machine_new';p=next((root/'06_tools/metadata').glob('attempt_*/operation.json'))
    operation=json.loads(p.read_text());operation['config']['seed']=99;write_json(p,operation)
    with pytest.raises(ValueError,match='configuration changed'):retry_batch(root)


def test_category_configuration_doctor_and_cook_regex_serialization():
    from core.metadata_workflow import doctor
    from core.rules_loader import load_blacklist_individual
    root=Path(__file__).resolve().parents[1]
    report=doctor(root)
    assert report['status']=='configuration_checks_passed'
    assert report['toml_files_parsed']>=50
    rules=load_blacklist_individual(root/'02_脚本/categories/exo_cook/rules')
    import re
    patterns=[re.compile(r['pattern']) for r in rules['channel_pass2']]
    assert any(rx.search("Helen's Recipes (Vietnamese Food)") for rx in patterns)
    assert any(rx.search('Survival Skills Cooking') for rx in patterns)
    assert not any(rx.search('Different channel not in the list') for rx in patterns)


def test_publish_rejects_changed_release_receipt(tmp_path):
    _,_,result=setup_batch(tmp_path)
    evaluate(result['round'],labels_for(result,tmp_path));release=publish_batch(result['batch_root'])
    path=Path(release['deliver_path']).parent/'release.json';receipt=json.loads(path.read_text());receipt['input_sha256']='wrong';write_json(path,receipt)
    with pytest.raises(ValueError):publish_batch(result['batch_root'])
    assert batch_status(result['batch_root'])['status']=='invalid_release'


def test_same_snapshot_cannot_restart_under_a_new_batch_name(tmp_path):
    src,policy,result=setup_batch(tmp_path)
    with pytest.raises(ValueError,match='already belongs'):
        start_batch(src,policy,tmp_path/'runs','another_name',target=.9)
    assert not (tmp_path/'runs/exo_cook/machine_another_name').exists()
