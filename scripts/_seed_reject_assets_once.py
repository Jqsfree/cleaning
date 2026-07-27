#!/usr/bin/env python3
"""一次性种子：用现有 drop/thumb 灌入 _reject_seed_0727 + data/assets/rejects"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "02_脚本"))

import importlib.util

def _load(name: str, rel: str):
    path = ROOT / "02_脚本" / rel
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

propose_mod = _load("propose_reject_tags", "tools/propose_reject_tags.py")
export_mod = _load("export_reject_assets", "tools/export_reject_assets.py")
metrics_mod = _load("reject_source_metrics", "tools/reject_source_metrics.py")
suggest_mod = _load("suggest_reject_opt", "tools/suggest_reject_opt.py")

_read_table = propose_mod._read_table
propose_from_frame = propose_mod.propose_from_frame
propose_from_thumb = propose_mod.propose_from_thumb
write_proposed = propose_mod.write_proposed
export_assets = export_mod.export_assets
compute_metrics_from_assets = metrics_mod.compute_metrics_from_assets
build_suggestions = suggest_mod.build_suggestions
render_md = suggest_mod.render_md


BATCH = ROOT / "data/runs/film_tv/_reject_seed_0727"
ASSETS = ROOT / "data/assets/rejects"
(BATCH / "03_qc").mkdir(parents=True, exist_ok=True)
(BATCH / "04_rules").mkdir(parents=True, exist_ok=True)

texts = [
    ROOT / "data/runs/film_tv/human_0724/02_clean/run01/_merged_all_clean_drop_0724.csv",
    ROOT / "data/runs/film_tv/005_clean/影视剧-播放列表_new_20260723/run01/影视剧-播放列表_771f94e0_records_new_20260723_clean_drop_0723.csv",
    ROOT / "data/runs/film_tv/005_clean/影视剧核心片段批量_4292e310_records/run01/影视剧核心片段批量_4292e310_records_clean_drop_0722.csv",
]
thumbs = [
    ROOT / "data/runs/film_tv/005_clean/影视剧_31768118_records/run01/影视剧_31768118_records_run01_keep_thumb_qc.csv",
    ROOT / "data/runs/film_tv/005_clean/影视剧-播放列表_771f94e0_records/run05/影视剧-播放列表_771f94e0_records_run01_keep_high_clean_0722_thumb_qc.csv",
    ROOT / "data/runs/film_tv/005_clean/影视剧-频道_0316a650_records/run05/影视剧-频道_0316a650_records_clean_0722_thumb_qc.csv",
]

prop = BATCH / "03_qc" / "reject_proposed.csv"
if prop.exists():
    prop.unlink()

t0 = time.time()
n_text = n_thumb = 0

for f in texts:
    print(f"[text] {f.name} ...", flush=True)
    df = _read_table(str(f))
    print(f"  in={len(df):,}", flush=True)
    out = propose_from_frame(df, category="film_tv", modality="text")
    write_proposed(BATCH, out, merge=True)
    n_text += len(out)
    print(f"  +{len(out):,}", flush=True)

for f in thumbs:
    print(f"[thumb] {f.name} ...", flush=True)
    df = _read_table(str(f))
    print(f"  in={len(df):,}", flush=True)
    out = propose_from_thumb(df, category="film_tv")
    write_proposed(BATCH, out, merge=True)
    n_thumb += len(out)
    print(f"  +{len(out):,}", flush=True)

print("[export]", flush=True)
stats = export_assets(batch_roots=[BATCH], assets_root=ASSETS)

print("[metrics]", flush=True)
report = compute_metrics_from_assets(ASSETS)
metrics_dir = ASSETS / "_metrics"
metrics_dir.mkdir(parents=True, exist_ok=True)
(metrics_dir / "by_source.json").write_text(
    json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
suggestions = build_suggestions(report)
(BATCH / "04_rules" / "reject_opt_suggestions.md").write_text(
    render_md(suggestions, report), encoding="utf-8"
)

df = pd.read_csv(prop, dtype=str)
print("=" * 56, flush=True)
print(f"DONE {time.time()-t0:.1f}s", flush=True)
print(f"text_new={n_text:,} thumb_new={n_thumb:,} total={len(df):,}", flush=True)
print("modality:", df["modality"].value_counts().to_dict(), flush=True)
print("top tags:", df["reject_tags"].value_counts().head(12).to_dict(), flush=True)
print(f"asset_tags={len(stats)} alerts={report.get('deadlock_alerts')}", flush=True)
print("=" * 56, flush=True)
