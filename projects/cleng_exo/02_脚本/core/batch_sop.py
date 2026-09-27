#!/usr/bin/env python3
"""
core/batch_sop.py — 按 recipe 计算 checklist / next_action
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from core.recipe import Stage, flow_for, load_recipe


def _dir_nonempty(path: Path) -> bool:
    return path.is_dir() and any(path.iterdir())


def _file_exists(path: Path) -> bool:
    return path.is_file()


def stage_done(batch_root: Path, stage: Stage, paths: dict[str, str]) -> bool:
    """根据 stage id 与批次目录判断是否已完成。"""
    sid = stage.get("id") or ""
    root = Path(batch_root)

    mapping = {
        "quality": paths.get("quality", "01_quality"),
        "sample": paths.get("sample", "02_sample"),
        "human_qc": paths.get("qc", "03_qc"),
        "text_qc": paths.get("qc", "03_qc"),
        "thumb_qc": paths.get("qc", "03_qc"),
        "storyboard_qc": paths.get("qc", "03_qc"),
        "rules": paths.get("rules", "04_rules"),
        "clean": paths.get("clean", "05_clean"),
        "deliver": paths.get("deliver", "07_deliver"),
    }
    rel = mapping.get(sid)
    if rel is None:
        # 未知 id：看 manifest stages
        return False

    target = root / rel
    if sid in ("human_qc", "text_qc", "thumb_qc", "storyboard_qc"):
        # QC：目录非空即可；human 另认 pass.csv
        if sid == "human_qc":
            return (target / "pass.csv").is_file() or (target / "labeled.csv").is_file() or _dir_nonempty(target)
        return _dir_nonempty(target)
    if sid == "rules":
        return _dir_nonempty(target) or (target / "NOTES.md").is_file()
    if sid == "clean":
        if not target.is_dir():
            return False
        return any(target.rglob("*.csv")) or any(target.rglob("*.parquet"))
    if sid == "deliver":
        return _dir_nonempty(target)
    return _dir_nonempty(target)


def has_fail(batch_root: Path, paths: dict[str, str]) -> bool:
    qc = Path(batch_root) / paths.get("qc", "03_qc")
    fail = qc / "fail.csv"
    if fail.is_file() and fail.stat().st_size > 10:
        return True
    return False


def deps_satisfied(
    batch_root: Path,
    stage: Stage,
    done: dict[str, bool],
) -> bool:
    for d in stage.get("depends") or []:
        if not done.get(d, False):
            # optional 依赖未完成时：若依赖本身 optional 且缺失，仍阻塞
            return False
    return True


def evaluate_recipe_checklist(
    batch_root: str | Path,
    *,
    category: str,
    source: str,
) -> list[dict[str, str]]:
    root = Path(batch_root)
    recipe = load_recipe(category)
    paths = dict(recipe.get("paths") or {})
    rows: list[dict[str, str]] = []
    flow = flow_for(recipe, source)
    done: dict[str, bool] = {}

    for st in flow:
        sid = str(st["id"])
        ok = stage_done(root, st, paths)
        # when=has_fail：无 fail 则 skip
        when = st.get("when")
        if when == "has_fail" and not has_fail(root, paths):
            status = "skip"
            ok = True  # 不算 missing
        elif ok:
            status = "ok"
        elif st.get("optional") or st.get("kind") in ("optional_auto",):
            status = "optional_missing"
        elif st.get("kind") == "external":
            status = "missing"
        else:
            status = "missing"
        done[sid] = ok or status == "skip"
        rows.append({
            "id": sid,
            "kind": str(st.get("kind") or ""),
            "status": status,
            "hint": str(st.get("hint") or ""),
            "tool": str(st.get("tool") or ""),
            "optional": "yes" if st.get("optional") else "no",
        })
    # manifest
    man = root / "manifest.json"
    rows.append({
        "id": "manifest",
        "kind": "file",
        "status": "ok" if man.is_file() else "missing",
        "hint": "run_manifest.py init / run.py",
        "tool": "",
        "optional": "no",
    })
    return rows


def next_action(
    batch_root: str | Path,
    *,
    category: str,
    source: str,
) -> dict[str, Any] | None:
    """返回下一建议动作；全部完成则 None。"""
    root = Path(batch_root)
    recipe = load_recipe(category)
    paths = dict(recipe.get("paths") or {})
    flow = flow_for(recipe, source)
    done: dict[str, bool] = {}

    for st in flow:
        sid = str(st["id"])
        when = st.get("when")
        if when == "has_fail" and not has_fail(root, paths):
            done[sid] = True
            continue
        ok = stage_done(root, st, paths)
        done[sid] = ok

    for st in flow:
        sid = str(st["id"])
        when = st.get("when")
        if when == "has_fail" and not has_fail(root, paths):
            continue
        if done.get(sid):
            continue
        if not deps_satisfied(root, st, done):
            continue
        kind = st.get("kind") or "auto"
        hint = st.get("hint") or ""
        argv_hint = _argv_hint(sid, kind, st, category=category, source=source)
        return {
            "id": sid,
            "kind": kind,
            "reason": hint or f"下一阶段: {sid}",
            "argv_hint": argv_hint,
            "optional": bool(st.get("optional")),
            "tool": st.get("tool"),
        }
    return None


def _argv_hint(
    sid: str,
    kind: str,
    st: Stage,
    *,
    category: str,
    source: str,
) -> str:
    if sid == "quality":
        return "pipeline/run.py <raw.csv> --category {cat} --source {src} -o $BATCH/"
    if sid == "sample":
        return "pipeline/03_sample.py $BATCH/01_quality/*quality*.csv -o $BATCH/02_sample/ -n 385"
    if sid == "human_qc":
        return "tools/ingest_human_qc.py labels.csv -o $BATCH/ --category {cat} --source {src} --batch <id>"
    if sid == "text_qc":
        return "qc/text.py $BATCH/02_sample/*sample*.csv --category {cat} -o $BATCH/03_qc/ -w 20"
    if sid == "rules":
        return "mkdir -p $BATCH/04_rules && echo '规则依据' > $BATCH/04_rules/NOTES.md"
    if sid == "clean":
        if source == "human":
            return (
                "pipeline/02_clean.py $BATCH/03_qc/fail.csv --category {cat} "
                "--source human --allow-clean -o $BATCH/05_clean/run01/"
            )
        return (
            "pipeline/02_clean.py $Q --category {cat} --source machine "
            "--rules-ready -o $BATCH/05_clean/run01/"
        )
    if sid == "deliver":
        tool = st.get("tool") or "copy_keep"
        if tool == "ge720":
            return "tools/batch_deliver_ge720.py <keep.csv> --batch-root $BATCH/ --batch-id <id>"
        if tool == "copy_pass":
            return "cp $BATCH/03_qc/pass.csv $BATCH/07_deliver/ && run_manifest update …"
        if tool == "copy_remain":
            return (
                "tools/lot_accept.py prepare -o $BATCH/ --lot-csv <remain.csv> "
                "--frame remain --deliver-name <id>_deliver_remain.csv "
                "&& lot_accept verify -o $BATCH/"
            )
        return "将 keep 拷入 $BATCH/07_deliver/ 并更新 manifest.deliver_path"
    return hint if (hint := st.get("hint")) else f"完成阶段 {sid}"


def upto_order(recipe: dict, source: str) -> list[str]:
    return [str(s["id"]) for s in flow_for(recipe, source)]
