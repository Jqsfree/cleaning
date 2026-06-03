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
    for d in sorted(base.iterdir()):
        if not d.is_dir():
            continue
        # Static dirs (001_baseline etc) treated as run "shared"
        if d.name.startswith(("001_", "002_")):
            pf = d / "progress.json"
            if pf.exists():
                try:
                    data = json.loads(pf.read_text())
                    runs.setdefault("shared", {})[data.get("phase", 0)] = data
                except json.JSONDecodeError:
                    pass
        # Run dirs (003_analysis/run01, 005_clean/run01)
        elif d.name.startswith(("003_", "004_", "005_", "006_", "007_")):
            for rd in sorted(d.iterdir()):
                if not rd.is_dir():
                    continue
                pf = rd / "progress.json"
                if pf.exists():
                    try:
                        data = json.loads(pf.read_text())
                        runs.setdefault(rd.name, {})[data.get("phase", 0)] = data
                    except json.JSONDecodeError:
                        pass
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
                    st.metric("T", p3.get("sports", 0))
                    st.metric("F", p3.get("non_sports", 0))
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

st.set_page_config(page_title="sport-live Monitor", layout="wide", page_icon="🏟")
st.markdown(f'<meta http-equiv="refresh" content="{REFRESH_SEC}">', unsafe_allow_html=True)
st.title("🏟 sport-live 管道监控")
st.caption(f"⏱ {time.strftime('%Y-%m-%d %H:%M:%S')} · 每 {REFRESH_SEC}s 刷新")

col1, col2 = st.columns(2)
with col1:
    render_dataset("pingpong", "乒乓球", "🏓")
with col2:
    render_dataset("curling", "冰壶", "🥌")

st.divider()
st.caption("`conda activate data_cleaning && streamlit run monitor.py`")
