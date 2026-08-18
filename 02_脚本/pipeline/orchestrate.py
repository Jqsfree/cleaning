#!/usr/bin/env python3
"""
pipeline/orchestrate.py — 按品类 recipe 解释流程（非统一 Phase0–7）

用法:
  02_脚本/pipeline/orchestrate.py status -o $BATCH/
  02_脚本/pipeline/orchestrate.py next -o $BATCH/
  02_脚本/pipeline/orchestrate.py run -o $BATCH/ --upto sample --input raw.csv

默认 status/next 只读。run 必须 --upto；禁止无 upto 全自动串完。
gated 阶段仍走 clean_gates；external 阶段只提示不执行。
"""

from __future__ import annotations

import argparse
import importlib.util
import shutil
import sys
from pathlib import Path
from types import ModuleType

_PIPE = Path(__file__).resolve().parent
_ROOT = _PIPE.parent
sys.path.insert(0, str(_ROOT))

from core.batch_layout import looks_like_batch_root  # noqa: E402
from core.batch_sop import evaluate_recipe_checklist, next_action, upto_order  # noqa: E402
from core.clean_gates import assert_clean_gates  # noqa: E402
from core.contracts import assert_contracts  # noqa: E402
from core.log import log  # noqa: E402
from core.recipe import flow_for, load_recipe, stage_by_id  # noqa: E402
from core.run_manifest import init_manifest, load_manifest, update_stage  # noqa: E402
from core.provenance import build_provenance  # noqa: E402

QUALITY_PY = _PIPE / "01_quality.py"
SAMPLE_PY = _PIPE / "03_sample.py"
CLEAN_PY = _PIPE / "02_clean.py"
TEXT_QC_PY = _PIPE.parent / "qc" / "text.py"


def _load_mod(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _batch_meta(batch_root: Path, args) -> tuple[str, str, str]:
    data = load_manifest(batch_root)
    category = args.category or data.get("category")
    source = args.source or data.get("source")
    batch = args.batch or data.get("batch")
    if not category or not source:
        # 目录名兜底：…/film_tv/human_0804
        if looks_like_batch_root(batch_root):
            source = source or batch_root.name.split("_", 1)[0].lower()
            batch = batch or batch_root.name.split("_", 1)[1]
            category = category or batch_root.parent.name
    if not category or not source:
        raise SystemExit(
            "[ERROR] 需要 --category/--source，或先 init manifest"
        )
    source = str(source).lower()
    batch = str(batch or batch_root.name.split("_", 1)[-1])
    return str(category), source, batch


def cmd_status(args) -> None:
    root = Path(args.batch_root)
    category, source, _ = _batch_meta(root, args)
    recipe = load_recipe(category)
    meta = recipe.get("meta") or {}
    print(f"category={category} source={source} deliver_format="
          f"{meta.get('deliver_format')}")
    if meta.get("layer1_stop"):
        print(f"layer1_stop={meta.get('layer1_stop')}  （第一层停点；其后阶段勿默认跑）")
    print(f"layers: {recipe.get('layers')}")
    rows = evaluate_recipe_checklist(root, category=category, source=source)
    print(f"{'id':<16} {'status':<18} {'kind':<14} hint")
    print("-" * 72)
    missing = 0
    for r in rows:
        print(f"{r['id']:<16} {r['status']:<18} {r['kind']:<14} {r.get('hint','')[:40]}")
        if r["status"] == "missing" and r.get("optional") != "yes":
            missing += 1
    if args.strict and missing:
        sys.exit(2)


def cmd_next(args) -> None:
    root = Path(args.batch_root)
    category, source, _ = _batch_meta(root, args)
    act = next_action(root, category=category, source=source)
    if act is None:
        print("NEXT: (done) 配方流程无待办阶段")
        return
    print(f"NEXT: {act['id']}  kind={act['kind']}  optional={act['optional']}")
    print(f"  reason: {act['reason']}")
    hint = str(act["argv_hint"]).format(cat=category, src=source)
    print(f"  hint:   {hint}")
    if act["kind"] == "external":
        print("  （external：编排不自动执行，请人工完成后重跑 next）")


def _latest_quality_keep(q_dir: Path) -> Path | None:
    cands = sorted(
        q_dir.glob("*_quality_*.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    cands = [p for p in cands if "_quality_drop_" not in p.name]
    return cands[0] if cands else None


def _latest_sample(s_dir: Path) -> Path | None:
    cands = sorted(
        list(s_dir.glob("*sample*.csv")) + list(s_dir.glob("*sample*.parquet")),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return cands[0] if cands else None


def _run_text_qc(root: Path, category: str, paths: dict, args) -> None:
    """调用 qc/text.py；须 --run-text-qc。"""
    import subprocess

    s_dir = root / paths.get("sample", "02_sample")
    sample = _latest_sample(s_dir)
    if sample is None:
        raise SystemExit(f"[ERROR] 无 sample 文件: {s_dir}")
    qc_dir = root / paths.get("qc", "03_qc")
    qc_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(TEXT_QC_PY), str(sample),
        "--category", category,
        "-o", str(qc_dir),
        "-w", str(args.text_workers),
    ]
    if getattr(args, "dry_run_qc", False):
        cmd.append("--dry-run")
    log(f"text_qc: {' '.join(cmd)}")
    proc = subprocess.run(cmd, check=False)
    if proc.returncode != 0:
        sys.exit(proc.returncode)
    update_stage(
        root,
        "text_qc",
        paths={"qc_dir": str(qc_dir), "sample": str(sample)},
        provenance=build_provenance(input_path=sample, extra={"category": category}),
    )


def cmd_run(args) -> None:
    if not args.upto:
        raise SystemExit("[ERROR] run 必须指定 --upto（禁止无边界全自动串完）")
    root = Path(args.batch_root)
    root.mkdir(parents=True, exist_ok=True)
    category, source, batch = _batch_meta(root, args)
    recipe = load_recipe(category)
    order = upto_order(recipe, source)
    if args.upto not in order:
        raise SystemExit(
            f"[ERROR] --upto {args.upto!r} 不在 recipe flow 中: {order}"
        )
    stop_at = order.index(args.upto)

    try:
        init_manifest(
            root, category=category, source=source, batch=batch,
            input_path=args.input or "",
        )
    except Exception as e:
        log(f"manifest init: {e}", level="WARN")

    for i, sid in enumerate(order):
        if i > stop_at:
            break
        st = stage_by_id(recipe, source, sid) or {}
        kind = st.get("kind") or "auto"
        # 已完成则跳过
        from core.batch_sop import stage_done, has_fail
        paths = dict(recipe.get("paths") or {})
        if st.get("when") == "has_fail" and not has_fail(root, paths):
            log(f"skip {sid} (when=has_fail 无 fail)")
            continue
        if stage_done(root, st, paths):
            log(f"skip {sid} (already done)")
            continue

        if kind == "external":
            log(f"STOP at external stage={sid}；请人工处理后重跑", level="WARN")
            print(f"hint: {st.get('hint') or sid}")
            sys.exit(0)

        if kind == "optional_auto" and not args.run_text_qc and sid in (
            "text_qc", "thumb_qc", "storyboard_qc",
        ):
            if args.upto == sid:
                raise SystemExit(
                    f"[ERROR] --upto {sid} 需要同时 --run-text-qc（避免静默跳过）"
                )
            log(f"skip optional {sid}（需要 --run-text-qc 才自动跑 QC）")
            continue

        if kind == "gated" and sid == "clean":
            _run_clean(root, category, source, args, st)
            continue

        if sid == "quality":
            if not args.input:
                raise SystemExit("[ERROR] --upto quality 需要 --input raw.csv")
            q_dir = root / paths.get("quality", "01_quality")
            q_dir.mkdir(parents=True, exist_ok=True)
            mod = _load_mod("orch_quality", QUALITY_PY)
            mod.quality_check(args.input, str(q_dir))
            keep = _latest_quality_keep(q_dir)
            if keep:
                assert_contracts(keep, layer="quality", category=category)
            continue

        if sid == "sample":
            q_dir = root / paths.get("quality", "01_quality")
            keep = _latest_quality_keep(q_dir)
            if keep is None:
                raise SystemExit(f"[ERROR] 无 quality keep: {q_dir}")
            s_dir = root / paths.get("sample", "02_sample")
            s_dir.mkdir(parents=True, exist_ok=True)
            import subprocess
            cmd = [
                sys.executable, str(SAMPLE_PY), str(keep),
                "-o", str(s_dir),
                "--confidence", str(args.sample_confidence),
                "--margin", str(args.sample_margin),
            ]
            if args.sample_n is not None:
                cmd.extend(["-n", str(args.sample_n)])
            proc = subprocess.run(cmd, check=False)
            if proc.returncode != 0:
                sys.exit(proc.returncode)
            continue

        if sid == "text_qc":
            _run_text_qc(root, category, paths, args)
            continue

        if sid == "deliver":
            _run_deliver(root, category, source, recipe, st, args)
            continue

        log(f"未实现自动执行的 stage={sid} kind={kind}，请按 next 提示手工跑", level="WARN")
        sys.exit(0)

    log(f"run --upto {args.upto} 完成 → {root}")


def _run_clean(root: Path, category: str, source: str, args, st: dict) -> None:
    if source == "machine" and not args.rules_ready:
        raise SystemExit("[ERROR] 机采 clean 须 --rules-ready")
    if source == "human" and not args.allow_clean:
        raise SystemExit("[ERROR] 人工 clean 须 --allow-clean")

    if source == "human":
        inp = root / "03_qc" / "fail.csv"
        if not inp.is_file():
            raise SystemExit(f"[ERROR] 无 fail 集: {inp}")
    else:
        if args.input and Path(args.input).is_file():
            inp = Path(args.input)
        else:
            keep = _latest_quality_keep(root / "01_quality")
            if keep is None:
                raise SystemExit("[ERROR] 机采 clean 需要 quality keep 或 --input")
            inp = keep

    c_dir = root / "05_clean" / (args.run or "run01")
    c_dir.mkdir(parents=True, exist_ok=True)
    assert_clean_gates(
        source=source,
        input_path=str(inp),
        output_dir=str(c_dir),
        rules_ready=bool(args.rules_ready),
        allow_clean=bool(args.allow_clean),
        skip_evidence=bool(args.skip_evidence),
    )
    mod = _load_mod("orch_clean", CLEAN_PY)
    mod.run_clean(
        str(inp),
        category,
        str(c_dir),
        source=source,
        rules_ready=bool(args.rules_ready),
        allow_clean=bool(args.allow_clean),
        skip_evidence=bool(args.skip_evidence),
        run=args.run or "run01",
        enforce_gates=True,
    )


def _run_deliver(root: Path, category: str, source: str, recipe: dict, st: dict, args) -> None:
    tool = st.get("tool") or "copy_keep"
    ddir = root / "07_deliver"
    ddir.mkdir(parents=True, exist_ok=True)
    if tool == "ge720":
        raise SystemExit(
            "[ERROR] deliver.tool=ge720 请显式调用 "
            "tools/batch_deliver_ge720.py（需 YOUTUBE_API_KEY）"
        )
    src: Path | None = None
    if tool == "copy_pass":
        src = root / "03_qc" / "pass.csv"
    elif tool == "copy_keep":
        # prefer clean keep
        cleans = sorted((root / "05_clean").rglob("*keep*.parquet")) + sorted(
            (root / "05_clean").rglob("*keep*.csv")
        )
        if cleans:
            src = cleans[-1]
        else:
            src = _latest_quality_keep(root / "01_quality")
    elif tool == "copy_qc":
        qc = root / "03_qc"
        cands = sorted(qc.glob("*.csv")) if qc.is_dir() else []
        src = cands[0] if cands else None

    if src is None or not src.is_file():
        raise SystemExit(f"[ERROR] deliver 找不到源文件 tool={tool}")

    dest = ddir / src.name
    shutil.copy2(src, dest)
    assert_contracts(dest, layer="deliver", category=category, soft=False)
    update_stage(
        root,
        "deliver",
        paths={"deliver": str(dest)},
        deliver_path=str(dest),
        provenance=build_provenance(input_path=src, extra={"tool": tool}),
    )
    log(f"deliver → {dest}")


def main() -> None:
    p = argparse.ArgumentParser(
        description="配方编排：status|next|run（解释 categories/*/recipe.toml）",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_batch(sp):
        sp.add_argument("-o", "--batch-root", required=True)
        sp.add_argument("--category", default=None)
        sp.add_argument("--source", default=None, choices=("human", "machine"))
        sp.add_argument("--batch", default=None)

    p_st = sub.add_parser("status", help="按 recipe 打印 checklist")
    add_batch(p_st)
    p_st.add_argument("--strict", action="store_true")

    p_nx = sub.add_parser("next", help="下一建议动作（不执行）")
    add_batch(p_nx)

    p_run = sub.add_parser("run", help="执行至 --upto（须显式）")
    add_batch(p_run)
    p_run.add_argument("--upto", required=True, help="quality|sample|clean|deliver|…")
    p_run.add_argument("--input", default="", help="raw / clean 输入")
    p_run.add_argument(
        "--sample-n", type=int, default=None,
        help="固定样本量（指定则不用公式）；默认按 --sample-confidence 计算",
    )
    p_run.add_argument(
        "--sample-confidence", type=int, default=90, choices=(90, 95, 99),
        help="抽样置信度（默认 90）",
    )
    p_run.add_argument(
        "--sample-margin", type=float, default=0.05,
        help="抽样误差（默认 0.05）",
    )
    p_run.add_argument("--run", default="run01", help="clean run 名")
    p_run.add_argument("--rules-ready", action="store_true")
    p_run.add_argument("--allow-clean", action="store_true")
    p_run.add_argument("--skip-evidence", action="store_true")
    p_run.add_argument(
        "--run-text-qc", action="store_true",
        help="允许自动跑 optional_auto 的 text/thumb QC（默认跳过；--upto text_qc 必填）",
    )
    p_run.add_argument(
        "--text-workers", type=int, default=32,
        help="text_qc 并发（传给 qc/text.py -w；默认 32，模型见 text.py DEFAULT_MODEL=qwen-plus）",
    )
    p_run.add_argument(
        "--dry-run-qc", action="store_true",
        help="text_qc 传 --dry-run（不调 API）",
    )

    args = p.parse_args()
    if args.cmd == "status":
        cmd_status(args)
    elif args.cmd == "next":
        cmd_next(args)
    elif args.cmd == "run":
        cmd_run(args)


if __name__ == "__main__":
    main()
