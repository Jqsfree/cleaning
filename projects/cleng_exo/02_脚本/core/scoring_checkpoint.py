"""CSV scoring with content-bound checkpoints and atomic regeneration of splits."""
from __future__ import annotations
import csv
import hashlib
import io
import json
import math
import os
import tempfile
from pathlib import Path
from core.data_profile import file_hash
from core.runtime_files import file_lock,atomic_json


def _csv_bytes(frame, header):
    # UTF-8 BOM only at file start, including across resumed chunks.
    clean=frame.copy()
    for col in clean.columns:
        clean[col]=clean[col].map(lambda v:v.replace('\x00','').replace('\r',' ').replace('\n',' ').replace('\u2028',' ').replace('\u2029',' ') if isinstance(v,str) else v)
    return (('\ufeff' if header else '')+clean.to_csv(index=False,header=header,lineterminator='\n')).encode('utf-8')


def _hash_prefix(path, size):
    h=hashlib.sha256();remaining=size
    with path.open('rb') as handle:
        while remaining:
            block=handle.read(min(1024*1024,remaining))
            if not block: raise ValueError('Scored output is shorter than its committed checkpoint')
            h.update(block);remaining-=len(block)
    return h


def score_csv(inp,out,scorer,*,identity,chunksize=30000,sample=0,resume=False,split_paths=None):
    import pandas as pd
    inp=Path(inp).resolve();out=Path(out).resolve()
    if inp==out: raise ValueError('Scoring cannot overwrite its input')
    if chunksize<1 or sample<0: raise ValueError('Invalid chunk/sample size')
    if not identity: raise ValueError('An explicit model/code identity is required')
    split_paths=tuple(Path(p).resolve() for p in split_paths) if split_paths else ()
    if split_paths and (len(split_paths)!=2 or len(set(split_paths+(inp,out)))!=4): raise ValueError('Scored input/output and split paths must be distinct')
    out.parent.mkdir(parents=True,exist_ok=True)
    checkpoint=out.with_suffix(out.suffix+'.checkpoint.json')
    with file_lock(out.with_suffix(out.suffix+'.lock'),blocking=False):
        signature={'input_sha256':file_hash(inp),'engine_sha256':file_hash(__file__),'identity':identity,'sample':sample,'split_paths':[str(p) for p in split_paths]}
        state={'schema_version':1,'signature':signature,'rows':0,'bytes':0,'output_sha256':hashlib.sha256(b'').hexdigest(),'state':'running'}
        if resume:
            if not checkpoint.is_file(): raise ValueError('Legacy output has no trusted checkpoint; start a new output')
            state=json.loads(checkpoint.read_text())
            if state.get('schema_version')!=1 or state['signature']!=signature: raise ValueError('Resume input/model/threshold/config fingerprint mismatch')
            hasher=_hash_prefix(out,state['bytes'])
            if hasher.hexdigest()!=state['output_sha256']: raise ValueError('Committed output changed; refusing resume')
            if out.stat().st_size!=state['bytes']:
                if state['state']=='complete': raise ValueError('Completed output changed; refusing resume')
                # Remove only uncommitted bytes after the verified checkpoint.
                with out.open('r+b') as handle:handle.truncate(state['bytes'])
        else:
            if out.exists() or checkpoint.exists() or any(p.exists() for p in split_paths):
                raise FileExistsError('Output already exists; use a new path or a verified --resume')
            out.touch();hasher=hashlib.sha256();atomic_json(checkpoint,state)
        completed=state['rows'];skipped=0
        for chunk in pd.read_csv(inp,encoding='utf-8-sig',chunksize=chunksize,nrows=sample or None):
            if skipped+len(chunk)<=completed:
                skipped+=len(chunk);continue
            if skipped<completed:
                chunk=chunk.iloc[completed-skipped:];skipped=completed
            scored=scorer(chunk)
            if len(scored)!=len(chunk) or not scored.index.equals(chunk.index): raise ValueError('Scorer changed row count or order')
            if 'ml_score' not in scored or not all(math.isfinite(float(v)) for v in scored['ml_score']): raise ValueError('Nonfinite/missing scores')
            if 'ml_auto_drop' not in scored: raise ValueError('Missing drop decisions')
            payload=_csv_bytes(scored,header=state['rows']==0)
            with out.open('ab') as handle:handle.write(payload);handle.flush();os.fsync(handle.fileno())
            hasher.update(payload);state.update(rows=state['rows']+len(scored),bytes=out.stat().st_size,output_sha256=hasher.hexdigest(),state='running')
            atomic_json(checkpoint,state)
        if file_hash(inp)!=signature['input_sha256']: raise ValueError('Input changed during scoring')
        if state['rows']==0: raise ValueError('Empty scoring input')
        if file_hash(out)!=state['output_sha256']: raise ValueError('Scored output changed during execution')
        # Always rebuild complete split outputs, including after an interrupted split.
        totals={'n_rows':0,'n_drop':0,'n_keep':0,'hours_keep':0.,'hours_drop':0.}
        temp_paths=[];handles=[]
        try:
            for path in split_paths:
                path.parent.mkdir(parents=True,exist_ok=True)
                fd,name=tempfile.mkstemp(prefix='.'+path.name+'-',dir=str(path.parent));temp_paths.append(Path(name));handles.append(os.fdopen(fd,'wb'))
            first=True
            for scored in pd.read_csv(out,encoding='utf-8-sig',chunksize=chunksize):
                raw=scored['ml_auto_drop'].astype(str).str.lower()
                if not raw.isin(['true','false','1','0']).all(): raise ValueError('Invalid stored drop decisions')
                mask=raw.isin(['true','1']);totals['n_rows']+=len(scored);totals['n_drop']+=int(mask.sum());totals['n_keep']+=int((~mask).sum())
                if 'duration_seconds' in scored:
                    dur=pd.to_numeric(scored['duration_seconds'],errors='coerce').fillna(0)
                    dur=dur.where(dur.map(lambda x: math.isfinite(x) and x>0),0)
                    totals['hours_keep']+=float(dur[~mask].sum())/3600;totals['hours_drop']+=float(dur[mask].sum())/3600
                for handle,part in zip(handles,(scored.loc[~mask],scored.loc[mask])):handle.write(_csv_bytes(part,first))
                first=False
            if totals['n_rows']!=state['rows']: raise ValueError('Checkpoint row conservation failed')
            for handle in handles:handle.flush();os.fsync(handle.fileno());handle.close()
            state.update(state='splitting',summary=totals);atomic_json(checkpoint,state)
            for temporary,path in zip(temp_paths,split_paths):temporary.replace(path)
            if file_hash(out)!=state['output_sha256'] or file_hash(inp)!=signature['input_sha256']: raise ValueError('Input/output changed before completion')
            state.update(state='complete',split_sha256={str(p):file_hash(p) for p in split_paths})
            atomic_json(checkpoint,state)
        finally:
            for handle in handles:
                if not handle.closed:handle.close()
            for temporary in temp_paths:temporary.unlink(missing_ok=True)
        return dict(totals,input=str(inp),output=str(out),checkpoint=str(checkpoint),
                    **({'keep':str(split_paths[0]),'drop':str(split_paths[1])} if split_paths else {}))
