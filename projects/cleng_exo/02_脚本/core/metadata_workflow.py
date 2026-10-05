"""Managed metadata batches and a rebuildable catalog. No implicit delivery or resume."""
from __future__ import annotations
import json
import re
import time
from pathlib import Path
from core.data_profile import file_hash
from core.runtime_files import atomic_json, file_lock
from core.run_manifest import init_manifest, load_manifest, update_stage, verify_input
from core.topic_loop import load_policy, digest, run_round
from core.metadata_acceptance import inspect_round, frozen_manifest, verified_evaluation, release_round, verified_release


def _component(value):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*',value): raise ValueError('Invalid category/batch path component')
    return value


def _attempts(root):
    return sorted((Path(root)/'06_tools/metadata').glob('attempt_*'))


def _managed(root):
    root=Path(root).resolve();m=load_manifest(root)
    if not m.get('input_sha256'): raise ValueError('Batch has no frozen input identity; create a new managed batch')
    if root.name != m['source']+'_'+m['batch'] or root.parent.name!=m['category']:
        raise ValueError('Batch directory and registered identity disagree')
    verify_input(m)
    return root,m


def _execute(root, meta, config):
    attempts=_attempts(root)
    number=max([int(p.name.split('_')[-1]) for p in attempts]+[0])+1
    attempt=root/'06_tools/metadata'/('attempt_%04d'%number)
    attempt.mkdir(parents=True,exist_ok=False)
    operation={'state':'running','started_at':time.time(),'config':config,'config_hash':digest(config)}
    atomic_json(attempt/'operation.json',operation)
    atomic_json(attempt/'policy.json',config['policy'])
    try:
        verify_input(meta)
        if config.get('reference_gold') and file_hash(config['reference_gold'])!=config['reference_sha256']: raise ValueError('Reference gold changed')
        model=config.get('model')
        if model and file_hash(model)!=config['model_sha256']: raise ValueError('Model changed before execution')
        result=run_round(meta['input'],attempt/'policy.json',attempt/'candidate',sample_n=config['sample_size'],
            seed=config['seed'],model_path=model,target=config['target'],purpose=config['purpose'],
            parent_round=config.get('parent_round'),reference_gold=config.get('reference_gold'),
            max_t_loss=config.get('max_t_loss'),audit_sizes=config.get('audit_sizes'))
        verify_input(meta)
        if result['input_sha256']!=meta['input_sha256']: raise ValueError('Candidate input mismatch')
        operation.update(state='complete',finished_at=time.time(),candidate_id=result['candidate_id'])
        atomic_json(attempt/'operation.json',operation)
        update_stage(root,'metadata',paths={'attempt':str(attempt),'round':str(attempt/'candidate'),'audit':str(attempt/'candidate/audit.csv')},stats={'candidate_id':result['candidate_id']})
        return {'batch_root':str(root),'attempt':str(attempt),'round':str(attempt/'candidate'),
                'audit':str(attempt/'candidate/audit.csv'),'status':'pending_labels','candidate_id':result['candidate_id']}
    except BaseException as exc:
        operation.update(state='failed',finished_at=time.time(),error=type(exc).__name__+': '+str(exc))
        atomic_json(attempt/'operation.json',operation)
        raise


def _config(policy, *, model=None, reference_gold=None, target=None, max_t_loss=None,
            sample_size=271, audit_sizes=None, seed=42, purpose='candidate', parent_round=None):
    p=load_policy(policy)
    return {'policy':p,'policy_hash':digest(p),'model':str(Path(model).resolve()) if model else None,
            'model_sha256':file_hash(model) if model else None,
            'reference_gold':str(Path(reference_gold).resolve()) if reference_gold else None,
            'reference_sha256':file_hash(reference_gold) if reference_gold else None,
            'target':target,'max_t_loss':max_t_loss,'sample_size':sample_size,'audit_sizes':audit_sizes,'seed':seed,
            'purpose':purpose,'parent_round':str(Path(parent_round).resolve()) if parent_round else None}


def start_batch(input_path, policy, runs_root, batch, *, source='machine', **options):
    config=_config(policy,**options)
    category=_component(config['policy']['category']);batch=_component(batch)
    if source not in ('human','machine'): raise ValueError('Invalid source')
    path=Path(input_path).resolve()
    if not path.is_file(): raise FileNotFoundError(path)
    category_root=Path(runs_root).resolve()/category
    root=category_root/(source+'_'+batch)
    with file_lock(category_root/'.batches.lock',blocking=False):
        if root.exists(): raise FileExistsError(root)
        if config['purpose']=='candidate':
            snapshot=file_hash(path)
            for previous in category_root.glob('*/manifest.json'):
                old=load_manifest(previous.parent)
                if old.get('input_sha256')==snapshot:
                    raise ValueError('This input snapshot already belongs to a managed batch; use status/retry/specialist: '+str(previous.parent))
        root.mkdir(parents=True,exist_ok=False)
        init_manifest(root,category=category,source=source,batch=batch,input_path=str(path),notes='managed metadata pipeline')
    with file_lock(root/'.workflow.lock',blocking=False):
        return _execute(root,load_manifest(root),config)


def retry_batch(batch_root):
    root=Path(batch_root).resolve()
    with file_lock(root/'.workflow.lock',blocking=False):
        root,meta=_managed(root);attempts=_attempts(root)
        if not attempts: raise ValueError('No prior operation to retry')
        previous=attempts[-1];operation=json.loads((previous/'operation.json').read_text())
        if operation['state']=='complete' or (previous/'candidate/manifest.json').exists():
            raise ValueError('A completed candidate cannot be resampled via retry; evaluate it or resolve its registration')
        config=operation['config']
        if digest(config)!=operation.get('config_hash'): raise ValueError('Retry configuration changed')
        if digest(config['policy'])!=config['policy_hash']: raise ValueError('Retry policy fingerprint mismatch')
        if config.get('reference_gold') and file_hash(config['reference_gold'])!=config['reference_sha256']:
            raise ValueError('Retry reference gold changed')
        # Exact configuration/seed, new attempt folder. Never append to partial pools.
        return _execute(root,meta,config)


def specialist_batch(batch_root, policy, *, model=None, reference_gold=None):
    root=Path(batch_root).resolve()
    with file_lock(root/'.workflow.lock',blocking=False):
        root,meta=_managed(root);attempts=_attempts(root)
        if not attempts: raise ValueError('No prior managed candidate')
        previous=attempts[-1];operation=json.loads((previous/'operation.json').read_text())
        if operation['state']!='complete': raise ValueError('Resolve incomplete execution before specialist cleaning')
        parent=previous/'candidate';m=frozen_manifest(parent);report=verified_evaluation(parent,m)
        if report['status']!='failed_acceptance': raise ValueError('Specialist requires failed human acceptance')
        config=_config(policy,model=model,reference_gold=reference_gold,target=m['acceptance_plan']['target'],
            max_t_loss=m['acceptance_plan'].get('max_t_loss'),sample_size=m['sample_size_requested'],
            audit_sizes=m['acceptance_plan'].get('audit_sizes'),seed=m['seed'],purpose='candidate',parent_round=parent)
        if config['policy']['category']!=meta['category']: raise ValueError('Specialist category mismatch')
        return _execute(root,meta,config)


def publish_batch(batch_root):
    root=Path(batch_root).resolve()
    with file_lock(root/'.workflow.lock',blocking=False):
        root,meta=_managed(root);attempts=_attempts(root)
        if not attempts: raise ValueError('No managed candidate')
        attempt=attempts[-1];operation=json.loads((attempt/'operation.json').read_text())
        if operation['state']!='complete': raise ValueError('Latest attempt is incomplete')
        candidate=attempt/'candidate';m=frozen_manifest(candidate);report=verified_evaluation(candidate,m)
        if report['status']!='accepted' or m['purpose']!='candidate': raise ValueError('Candidate has not passed independent acceptance')
        if m['input_sha256']!=meta['input_sha256'] or m['category']!=meta['category']: raise ValueError('Candidate batch identity mismatch')
        dest=root/'07_deliver'/('metadata_'+m['candidate_id'][:16])
        if dest.exists():
            result=verified_release(dest)
            if result['candidate_id']!=m['candidate_id']: raise ValueError('Release belongs to an earlier candidate')
        else:
            result=release_round(candidate,dest)
        update_stage(root,'deliver',paths={'deliver':str(dest/'data.csv'),'release':str(dest/'release.json')},stats=result,deliver_path=str(dest/'data.csv'))
        return dict(result,deliver_path=str(dest/'data.csv'),batch_root=str(root))


def batch_status(batch_root, *, verify=True):
    root=Path(batch_root).resolve();meta=load_manifest(root)
    if not meta: raise ValueError('Missing batch manifest')
    result={'category':meta.get('category'),'source':meta.get('source'),'batch':meta.get('batch'),
            'batch_root':str(root),'input':meta.get('input'),'input_sha256':meta.get('input_sha256'),
            'manifest':str(root/'manifest.json'),'registered_stages':list(meta.get('stages',{})),
            'deliver_path':meta.get('deliver_path') or None,'rounds':[], 'issues':[],
            'integrity':('verified' if meta.get('input_sha256') else 'legacy_identity_unverified') if verify else 'not_checked','release_eligible':False}
    artifacts=[]
    references=[("input","input",meta.get("input"))]
    for stage,entry in (meta.get("stages") or {}).items():
        for name,path in (entry.get("paths") or {}).items(): references.append((stage,name,path))
    for stage,name,value in references:
        if not value: continue
        path=Path(value)
        resolved=path.is_absolute()
        exists=path.exists() if resolved else None
        artifacts.append({"stage":stage,"name":name,"path":str(path),"exists":exists,
                          "bytes":path.stat().st_size if resolved and path.is_file() else None,
                          "resolution":"absolute" if resolved else "legacy_relative_reference"})
    result["registered_artifacts"]=artifacts
    if verify:
        try: verify_input(meta)
        except (OSError,ValueError) as exc: result['issues'].append(str(exc))
    attempts=_attempts(root)
    for attempt in attempts:
        try:
            op=json.loads((attempt/'operation.json').read_text())
            item={'attempt':str(attempt),'execution':op['state']}
            if op['state']=='complete':
                item.update(inspect_round(attempt/'candidate',verify=verify))
                if item['category']!=meta.get('category') or item['input_sha256']!=meta.get('input_sha256'):
                    raise ValueError('Candidate does not belong to registered batch')
            else: item.update(status='execution_'+op['state'],error=op.get('error'),next_action='inspect_execution_then_retry_same_config')
        except (OSError,ValueError,KeyError,TypeError) as exc:
            item={'attempt':str(attempt),'status':'invalid_evidence','error':str(exc),'next_action':'repair_evidence_no_release'}
        result['rounds'].append(item)
    if not attempts:
        result.update(status='legacy_unverified',next_action='create_managed_batch_from_explicit_input')
        # Discover old topic rounds as references; never infer acceptance from filenames.
        tool_root=root/'06_tools'
        if tool_root.exists():
            for path in sorted(tool_root.rglob('manifest.json')):
                try:
                    old=json.loads(path.read_text())
                    if 'audit_expected' in old:
                        result['rounds'].append({'round':str(path.parent),'status':'legacy_or_unmanaged_reference','schema_version':old.get('schema_version')})
                except (OSError,ValueError): result['issues'].append('Unreadable round manifest: '+str(path))
    else:
        latest=result['rounds'][-1]
        result.update(status=latest['status'],next_action=latest['next_action'],release_eligible=latest.get('release_eligible',False))
    if result['issues']: result.update(status='invalid_batch_input',release_eligible=False,next_action='resolve_batch_identity_no_release')
    if verify and result['release_eligible'] and result['deliver_path']:
        try:
            data=Path(result['deliver_path']);release=verified_release(data.parent)
            latest=result['rounds'][-1];m=frozen_manifest(latest['round'])
            if (release['candidate_id']!=latest['candidate_id'] or release['data_sha256']!=m['pool_hashes']['keep']
                or file_hash(data)!=m['pool_hashes']['keep']): raise ValueError('Registered release fingerprint mismatch')
            result.update(status='released',next_action='none')
        except (OSError,ValueError,KeyError) as exc:
            result['issues'].append(str(exc));result.update(status='invalid_release',release_eligible=False,next_action='repair_release_registration')
    return result


def catalog(runs_root, *, category=None, verify=False):
    root=Path(runs_root).resolve()
    if not root.is_dir(): raise FileNotFoundError(root)
    batches=[]
    for path in sorted(root.glob('*/*/manifest.json')):
        if category and path.parent.parent.name!=category: continue
        try: batches.append(batch_status(path.parent,verify=verify))
        except (OSError,ValueError,KeyError,TypeError) as exc:
            batches.append({'batch_root':str(path.parent),'status':'invalid_manifest','error':str(exc),'release_eligible':False})
    return {'schema_version':1,'generated_at':time.time(),'runs_root':str(root),'integrity':'verified_where_supported' if verify else 'not_checked',
            'batch_count':len(batches),'batches':batches,
            'note':'Rebuildable index of manifests; catalog never authorizes release. Legacy outputs are references, not new acceptance evidence.'}


def doctor(project_root):
    from core.rules_loader import tomllib
    root=Path(project_root).resolve();errors=[];toml_files=0;policies=[]
    rules=root/'02_脚本/categories'
    if not rules.is_dir(): raise FileNotFoundError(rules)
    for path in sorted(rules.rglob('*.toml')):
        try: tomllib.loads(path.read_text());toml_files+=1
        except (ValueError,OSError) as exc: errors.append({'path':str(path),'error':str(exc)})
    for path in sorted(rules.glob('*/rules/title_topic_v1.json')):
        try:
            policy=load_policy(path);policies.append({'category':policy['category'],'policy':str(path),'hash':digest(policy)})
        except (ValueError,OSError,KeyError,re.error) as exc: errors.append({'path':str(path),'error':str(exc)})
    return {'status':'configuration_checks_passed' if not errors else 'blocked_configuration',
            'toml_files_parsed':toml_files,'metadata_policies':policies,'errors':errors,
            'note':'Configuration checks do not certify human quality or model calibration.'}
