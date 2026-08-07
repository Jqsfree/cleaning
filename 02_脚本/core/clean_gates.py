#!/usr/bin/env python3
"""
core/clean_gates.py — clean 证据门禁（相对 honor-system boolean）

机采 --rules-ready：须能在批次内看到 sample/QC/规则依据。
人工 --allow-clean：输入须像不合格集（fail / 03_qc）。
"""

from __future__ import annotations

from pathlib import Path

from core.batch_layout import infer_batch_root
from core.run_manifest import load_manifest


def _dir_has_files(path: Path) -> bool:
    if not path.is_dir():
        return False
    return any(p.is_file() for p in path.rglob("*") if not p.name.startswith("."))


def machine_rules_evidence(batch_root: Path) -> tuple[bool, str]:
    """
    机采规则就绪证据：04_rules 有文件，或 manifest 含 sample+(qc|text_qc|human_qc)，
    或 03_qc 下有文件。
    """
    root = Path(batch_root)
    rules = root / "04_rules"
    if _dir_has_files(rules):
        return True, f"04_rules 有依据: {rules}"

    qc = root / "03_qc"
    sample = root / "02_sample"
    if _dir_has_files(sample) and _dir_has_files(qc):
        return True, f"02_sample + 03_qc 均有产物"

    data = load_manifest(root)
    stages = set((data.get("stages") or {}).keys())
    has_sample = "sample" in stages or _dir_has_files(sample)
    has_qc = bool(stages & {"qc", "text_qc", "human_qc", "qc_text"}) or _dir_has_files(qc)
    if has_sample and has_qc:
        return True, f"manifest/目录证据 sample+qc: {sorted(stages)}"

    if _dir_has_files(qc):
        # 仅有 QC 也接受（可能 sample 在别处），但提示偏弱
        return True, f"03_qc 有产物（建议补 04_rules/NOTES.md）"

    return False, (
        f"机采 --rules-ready 缺证据：请在 {root} 下准备 "
        f"02_sample + 03_qc，或 04_rules/NOTES.md；"
        f"临时跳过加 --skip-evidence"
    )


def human_fail_input_ok(input_path: str | Path) -> tuple[bool, str]:
    """人工 clean 输入应像不合格集。"""
    p = Path(input_path)
    name = p.name.lower()
    parts = {x.lower() for x in p.parts}
    if "fail" in name or name.startswith("fail") or "_fail" in name:
        return True, f"输入名含 fail: {p.name}"
    if "03_qc" in parts and ("fail" in name or "unqual" in name or "reject" in name):
        return True, f"位于 03_qc 且像不合格集: {p}"
    if "03_qc" in parts and "pass" not in name and "keep" not in name:
        # 03_qc 下非 pass 文件允许，但提示
        return True, f"位于 03_qc/: {p}"
    return False, (
        f"人工 --allow-clean 要求输入为不合格集（路径/文件名含 fail，"
        f"或位于 03_qc/）；当前: {p}；临时跳过加 --skip-evidence"
    )


def assert_clean_gates(
    *,
    source: str | None,
    input_path: str,
    output_dir: str,
    rules_ready: bool,
    allow_clean: bool,
    skip_evidence: bool = False,
    legacy: bool = False,
) -> None:
    """
    在 boolean 门禁通过后做证据检查。失败 → SystemExit(2)。
    legacy / skip_evidence 跳过证据（仍建议尽快去掉）。
    """
    if legacy or skip_evidence or source is None:
        return

    batch_root = infer_batch_root(output_dir) or infer_batch_root(input_path)

    if source == "machine" and rules_ready:
        if batch_root is None:
            raise SystemExit(
                "[ERROR] 机采 --rules-ready：无法从 -o/输入推断批次根 "
                f"(data/runs/{{cat}}/{{source}}_{{batch}}/)；"
                f"output={output_dir}；加 --skip-evidence 可临时跳过"
            )
        ok, msg = machine_rules_evidence(batch_root)
        if not ok:
            raise SystemExit(f"[ERROR] {msg}")
        print(f"[GATE] machine rules evidence: {msg}", flush=True)

    if source == "human" and allow_clean:
        ok, msg = human_fail_input_ok(input_path)
        if not ok:
            raise SystemExit(f"[ERROR] {msg}")
        print(f"[GATE] human fail input: {msg}", flush=True)


def assert_clean_not_raw(
    *,
    keep_path: Path,
    ran_quality: bool,
    source: str,
    skip_evidence: bool = False,
) -> None:
    """禁止 clean-only 直接吃 raw（除非 skip）。"""
    if skip_evidence or ran_quality:
        return
    name = keep_path.name.lower()
    parts = {x.lower() for x in keep_path.parts}
    if "01_quality" in parts or "quality" in name:
        return
    if "03_qc" in parts or "fail" in name:
        return
    if "05_clean" in parts:
        return
    raise SystemExit(
        f"[ERROR] clean 未跑 quality 且输入不像 quality/fail 产物: {keep_path}；"
        f"请先 quality，或对不合格集 clean，或加 --skip-evidence"
    )
