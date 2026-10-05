import json
import sys
from pathlib import Path
import pandas as pd
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'02_脚本'))
from core.scoring_checkpoint import score_csv


def scorer(frame):
    out=frame.copy();out['ml_score']=out['value']/10;out['ml_auto_drop']=out['ml_score']<.3
    return out


def source(tmp_path):
    p=tmp_path/'in.csv';pd.DataFrame([dict(video_id=str(i),title='row '+str(i),value=i,duration_seconds=3600) for i in range(5)]).to_csv(p,index=False);return p


def test_resume_after_interrupt_rebuilds_full_splits_and_totals(tmp_path):
    inp=source(tmp_path);out=tmp_path/'score.csv';paths=(tmp_path/'keep.csv',tmp_path/'drop.csv')
    def interrupted(frame):
        if frame.index[0]>=2:raise RuntimeError('interrupt')
        return scorer(frame)
    with pytest.raises(RuntimeError):score_csv(inp,out,interrupted,identity={'model':'a','threshold':.3},chunksize=2,split_paths=paths)
    cp=json.loads(out.with_suffix('.csv.checkpoint.json').read_text());assert cp['rows']==2
    with out.open('ab') as handle:handle.write(b'uncommitted partial')
    result=score_csv(inp,out,scorer,identity={'model':'a','threshold':.3},chunksize=3,resume=True,split_paths=paths)
    assert result['n_rows']==5 and result['n_drop']==3 and result['n_keep']==2
    assert result['hours_keep']==2 and result['hours_drop']==3
    assert len(pd.read_csv(paths[0]))==2 and len(pd.read_csv(paths[1]))==3
    assert out.read_bytes().count(b'\xef\xbb\xbf')==1
    paths[0].unlink()
    result=score_csv(inp,out,scorer,identity={'model':'a','threshold':.3},resume=True,split_paths=paths)
    assert result['n_rows']==5 and len(pd.read_csv(paths[0]))==2


@pytest.mark.parametrize('change',['input','model','threshold','output'])
def test_resume_rejects_changed_identity_or_completed_output(tmp_path,change):
    inp=source(tmp_path);out=tmp_path/'score.csv';identity={'model':'a','threshold':.3}
    score_csv(inp,out,scorer,identity=identity)
    if change=='input':inp.write_text(inp.read_text().replace('row 0','row X'))
    if change=='model':identity['model']='b'
    if change=='threshold':identity['threshold']=.5
    if change=='output':out.write_bytes(out.read_bytes()+b'\n')
    before=out.read_bytes()
    with pytest.raises(ValueError):score_csv(inp,out,scorer,identity=identity,resume=True)
    assert out.read_bytes()==before


def test_legacy_resume_and_implicit_overwrite_are_blocked(tmp_path):
    inp=source(tmp_path);out=tmp_path/'score.csv';out.write_text('old output')
    with pytest.raises(ValueError,match='checkpoint'):score_csv(inp,out,scorer,identity={'model':'a'},resume=True)
    with pytest.raises(FileExistsError):score_csv(inp,out,scorer,identity={'model':'a'})
    with pytest.raises(ValueError):score_csv(inp,inp,scorer,identity={'model':'a'})


def test_nonfinite_scores_not_committed(tmp_path):
    inp=source(tmp_path);out=tmp_path/'score.csv'
    def bad(frame):
        r=scorer(frame);r['ml_score']=float('nan');return r
    with pytest.raises(ValueError,match='Nonfinite'):score_csv(inp,out,bad,identity={'model':'a'})
    assert json.loads(out.with_suffix('.csv.checkpoint.json').read_text())['rows']==0


def test_cook_cli_requires_selected_threshold_and_uses_checkpoint_wrapper(tmp_path):
    import importlib.util
    path=Path(__file__).resolve().parents[1]/'02_脚本/tools/score_exo_cook_text.py'
    spec=importlib.util.spec_from_file_location('cook_scoring_test',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    calibration=tmp_path/'calibration.json'
    with pytest.raises(FileNotFoundError):module.load_threshold(calibration)
    calibration.write_text(json.dumps({'recall':{'drop_threshold':.2}}))
    with pytest.raises(ValueError):module.load_threshold(calibration)
    assert module.load_threshold(calibration,legacy_recall=True)==.2
    calibration.write_text(json.dumps({'t_like':{'drop_threshold':.5}}))
    assert module.load_threshold(calibration)==.5
    class Model:
        def predict_proba(self,texts):
            import numpy as np
            return np.asarray([[.2,.8] for _ in texts])
    inp=source(tmp_path);out=tmp_path/'cook.csv'
    result=module.score_file(inp,out,Model(),threshold=.5,chunksize=2,sample=0,split_keep_drop=True,run_identity={'test_model':'constant'})
    assert result['n_keep']==5 and result['n_drop']==0
