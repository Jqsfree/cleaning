#!/usr/bin/env python3
"""
monitor.py -- Streamlit 管道监控面板

支持多数据集、多轮迭代对比。
每 5 秒自动刷新。

用法:
  conda activate data_cleaning
  streamlit run monitor.py
"""

import streamlit as st
import json, os, time, re
from pathlib import Path

RUNS_DIR = Path(__file__).resolve().parent / "data" / "runs"
REFRESH_SEC = 5


# ═══════════════════ load ═══════════════════

def find_runs(dataset: str) -> dict[str, dict]:
    """返回 {run_name: {phase_id: progress_data}}"""
    base = RUNS_DIR / dataset
    if not base.exists():
        return {}
    runs = {}
    # 递归发现所有 progress.json，按目录结构分组
    for pf_path in sorted(base.rglob("progress.json")):
        try:
            data = json.loads(pf_path.read_text())
        except json.JSONDecodeError:
            continue
        phase = data.get("phase", 0)
        parent = pf_path.parent
        if parent.parent == base:
            # 直接在数据集根下一层的 → shared（如 001_baseline, 002_audit）
            runs.setdefault("shared", {})[phase] = data
        else:
            # 嵌套子目录 → 按 run 名分组（如 003_analysis/run01）
            runs.setdefault(parent.name, {})[phase] = data
    return runs


def load_stats(dataset: str) -> list[dict]:
    sp = RUNS_DIR / dataset / "001_baseline" / "baseline_stats.md"
    if not sp.exists(): return []
    rows = []
    for line in sp.read_text().split("\n"):
        m = re.match(r"\| (.+?) \| ([0-9,]+) \| ([0-9,—]+) \|", line)
        if m and m.group(1).strip() != "Stage":
            rows.append({"阶段": m.group(1).strip(), "保留": m.group(2), "移除": m.group(3)})
    return rows


def list_files(dataset: str, subdir: str) -> list[str]:
    p = RUNS_DIR / dataset / subdir
    if not p.exists(): return []
    # for run-based dirs, dig one level deeper
    files = []
    for f in p.rglob("*"):
        if f.is_file() and f.name != "progress.json":
            files.append(str(f.relative_to(p)))
    return sorted(files)


# ═══════════════════ render ═══════════════════

def render_dataset(ds_name: str, label: str, emoji: str):
    st.subheader(f"{emoji} {label}")

    st.caption("📋 规则: 02_脚本/rules/")

    runs = find_runs(ds_name)
    shared = runs.get("shared", {})
    stats = load_stats(ds_name)

    # Phase 0 — always shared
    p0 = shared.get(0)
    with st.expander(f"Phase 0 — 数据规范化 {'✅' if p0 else '⬜'}", expanded=False):
        if p0 and p0.get("status") == "done":
            st.metric("保留", f"{p0.get('final',0):,} 行", f"{p0.get('retention_pct',0):.1f}%")
            if stats:
                st.dataframe(stats, use_container_width=True, hide_index=True)
            st.caption(" | ".join(list_files(ds_name, "001_baseline")))
        else:
            st.caption("待跑")

    # Phase 2 — shared
    p2 = shared.get(2)
    with st.expander(f"Phase 2 — 抽样 + QC {'✅' if p2 and p2.get('T') is not None else '⬜'}", expanded=False):
        if p2 and p2.get("status") == "done":
            t_val = p2.get("T")
            if t_val is not None:
                c1, c2, c3 = st.columns(3)
                c1.metric("T", t_val)
                c2.metric("F", p2.get("F", 0))
                c3.metric("通过率", f"{t_val/max(p2.get('qc_samples',1),1)*100:.0f}%")
                st.caption(f"{p2.get('model','?')} · {p2.get('qc_samples','?')} 条")
            st.caption(" | ".join(list_files(ds_name, "002_audit")))
        elif p2 and p2.get("status") == "running":
            st.progress(p2.get("pct", 0) / 100)
            c1, c2 = st.columns(2)
            c1.metric("已处理", f"{p2.get('done',0)}/{p2.get('total',1)}")
            if p2.get("rate"): c2.metric("速率", f"{p2['rate']}/s")
        else:
            st.caption("待跑")

    # Phase 3/5 — per-run
    run_names = sorted([k for k in runs if k != "shared"])
    if not run_names:
        run_names = ["run01"]  # show placeholder

    for run_name in run_names:
        run_phases = runs.get(run_name, {})
        p3 = run_phases.get(3)
        p5 = run_phases.get(5)

        with st.expander(f"🔄 {run_name} — Phase 3/5", expanded=run_name == run_names[-1]):
            c3, c5 = st.columns(2)
            with c3:
                st.caption("Phase 3 污染分析")
                if p3 and p3.get("status") == "done":
                    st.metric("T", p3.get("lang", 0))
                    st.metric("F", p3.get("non_lang", 0))
                    st.metric("污染类别", p3.get("pollution_categories", 0))
                else:
                    st.caption("⬜ 待跑")

            with c5:
                st.caption("Phase 5 规则清洗")
                if p5 and p5.get("status") == "done":
                    st.metric("Keep", f"{p5.get('keep',0):,}")
                    st.metric("Drop", f"{p5.get('drop',0):,}")
                    st.metric("通过率", f"{p5.get('retention_pct',0):.1f}%")
                    st.caption(" | ".join(list_files(ds_name, "005_clean")))
                elif p5 and p5.get("status") == "running":
                    st.caption(f"🔄 {p5.get('stage','')}")
                else:
                    st.caption("⬜ 待跑")


# ═══════════════════ page ═══════════════════

st.set_page_config(page_title="teach Monitor", layout="wide", page_icon="📚")
st.markdown(f'<meta http-equiv="refresh" content="{REFRESH_SEC}">', unsafe_allow_html=True)
st.title("📚 teach 管道监控")
st.caption(f"⏱ {time.strftime('%Y-%m-%d %H:%M:%S')} · 每 {REFRESH_SEC}s 刷新")

# 自动发现数据集
datasets = sorted([d.name for d in RUNS_DIR.iterdir() if d.is_dir() and not d.name.startswith(".")])
if not datasets:
    st.warning("暂无数据集，请将数据放入 data/runs/ 目录下")
else:
    # 两列布局渲染所有数据集
    for i in range(0, len(datasets), 2):
        col1, col2 = st.columns(2)
        with col1:
            ds = datasets[i]
            render_dataset(ds, ds, "📚")
        with col2:
            if i + 1 < len(datasets):
                ds = datasets[i + 1]
                render_dataset(ds, ds, "📚")

st.divider()
st.caption("`conda activate data_cleaning && streamlit run monitor.py`")
