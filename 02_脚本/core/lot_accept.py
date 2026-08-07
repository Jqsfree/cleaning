"""
core/lot_accept.py — 交批总体 (=抽批总体) 的 lot 验收元数据

约定：
- screening（规则/ML 高置信 drop）≠ lot 验收
- sample_frame 必须等于 deliver_frame（交谁验谁）
- 默认 method=ci_estimate（比例估计 + 客户最低合格线）；AQL 需合同约定
- KPI 只认人工 pass_rate，禁止 ml keep%/uncertain% 冒充 lot 质量
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Any

import pandas as pd

from core.human_qc import normalize_frame, pass_rate
from core.run_manifest import load_manifest, save_manifest, update_stage

ALLOWED_METHODS = frozenset({
    "ci_estimate",
    "aql",
    "prescreen_plus_screening",  # 默认：复用已有人工抽检 + overturn，不加第二轮人工
})
ALLOWED_FRAMES = frozenset({"quality", "remain", "clean_keep", "human_pass"})
STAGE_NAME = "lot_accept"


def _repo_rel(path: Path) -> str:
    path = path.resolve()
    cur = path.parent
    while cur != cur.parent:
        if (cur / "AGENTS.md").exists():
            try:
                return str(path.relative_to(cur))
            except ValueError:
                break
        cur = cur.parent
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


def count_rows(csv_path: str | Path) -> int:
    """按首列非空计数（避免 CSV 坏行导致 wc -l 与加载行数不一致）。"""
    path = Path(csv_path)
    col = pd.read_csv(path, usecols=[0], dtype=str)
    return int(col.iloc[:, 0].dropna().astype(str).str.strip().ne("").sum())


def assert_ids_subset(
    sample_ids: set[str],
    lot_ids: set[str],
    *,
    label: str = "sample",
) -> None:
    missing = sample_ids - lot_ids
    if missing:
        preview = ", ".join(sorted(missing)[:8])
        more = f" …(+{len(missing) - 8})" if len(missing) > 8 else ""
        raise ValueError(
            f"{label} 中有 {len(missing)} 条不在 lot（sample_frame≠deliver_frame）: "
            f"{preview}{more}"
        )


def load_id_set(csv_path: str | Path, id_col: str = "video_id") -> set[str]:
    df = pd.read_csv(csv_path, usecols=[id_col], dtype=str)
    return set(df[id_col].dropna().astype(str).str.strip()) - {""}


def prepare_deliver(
    batch_root: str | Path,
    *,
    lot_csv: str | Path,
    sample_frame: str,
    deliver_name: str,
    method: str = "ci_estimate",
    notes: str = "",
) -> dict[str, Any]:
    """将 lot 拷入 07_deliver/，登记 deliver_path 与 lot 元数据（decision=pending）。"""
    if sample_frame not in ALLOWED_FRAMES:
        raise ValueError(f"sample_frame 须为 {sorted(ALLOWED_FRAMES)}")
    if method not in ALLOWED_METHODS:
        raise ValueError(f"method 须为 {sorted(ALLOWED_METHODS)}")

    root = Path(batch_root)
    lot_path = Path(lot_csv)
    if not lot_path.is_file():
        raise FileNotFoundError(f"lot 不存在: {lot_path}")

    deliver_dir = root / "07_deliver"
    deliver_dir.mkdir(parents=True, exist_ok=True)
    dest = deliver_dir / deliver_name
    if lot_path.resolve() != dest.resolve():
        shutil.copy2(lot_path, dest)

    lot_size = count_rows(dest)
    lot_rel = _repo_rel(dest)

    lot_meta: dict[str, Any] = {
        "lot_id": f"{root.name}",
        "lot_size": lot_size,
        "sample_frame": sample_frame,
        "deliver_frame": sample_frame,
        "method": method,
        "n": None,
        "pass_rate": None,
        "decision": "pending",
        "min_pass_rate": None,
        "notes": notes
        or (
            "默认交 remain；lot 判决用 verify（已有 human_qc + overturn），"
            "不加第二轮人工。禁止用 ml% 冒充 lot 质量。"
        ),
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    data = load_manifest(root)
    if not data:
        raise FileNotFoundError(f"无 manifest: {root}/manifest.json")
    data["lot"] = lot_meta
    data["deliver_path"] = lot_rel
    save_manifest(root, data)

    update_stage(
        root,
        STAGE_NAME,
        paths={"lot": lot_rel, "deliver": lot_rel},
        stats={
            "lot_size": lot_size,
            "sample_frame": sample_frame,
            "method": method,
            "decision": "pending",
        },
        deliver_path=lot_rel,
    )
    return lot_meta


def record_sample(
    batch_root: str | Path,
    *,
    sample_csv: str | Path,
    lot_csv: str | Path | None = None,
    id_col: str = "video_id",
) -> dict[str, Any]:
    """登记验收样本路径；校验 sample ⊆ lot。"""
    root = Path(batch_root)
    data = load_manifest(root)
    if not data:
        raise FileNotFoundError(f"无 manifest: {root}")
    lot = dict(data.get("lot") or {})
    if not lot:
        raise ValueError("请先 prepare（manifest.lot 为空）")

    sample_path = Path(sample_csv)
    n = count_rows(sample_path)
    sample_ids = load_id_set(sample_path, id_col=id_col)

    lot_path = Path(lot_csv) if lot_csv else Path(
        (data.get("stages") or {}).get(STAGE_NAME, {}).get("paths", {}).get("lot")
        or data.get("deliver_path")
        or ""
    )
    if not lot_path.is_file():
        raise FileNotFoundError(f"找不到 lot 文件以校验 frame: {lot_path}")
    lot_ids = load_id_set(lot_path, id_col=id_col)
    assert_ids_subset(sample_ids, lot_ids, label="验收样本")

    lot["n"] = n
    lot["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    data["lot"] = lot
    save_manifest(root, data)

    sample_rel = str(sample_path)
    update_stage(
        root,
        STAGE_NAME,
        paths={"sample": sample_rel},
        stats={"n": n, "decision": lot.get("decision") or "pending"},
    )
    return lot


def decide(
    batch_root: str | Path,
    *,
    labeled_csv: str | Path,
    method: str = "ci_estimate",
    min_pass_rate: float = 0.85,
    aql: float | None = None,
    ac: int | None = None,
    re: int | None = None,
    category: str = "live_sell",
    source: str = "human",
    batch: str = "",
    id_col: str | None = None,
    label_col: str | None = None,
) -> dict[str, Any]:
    """根据人工标注写 lot decision（accept/reject）。强制 sample ⊆ lot。"""
    if method not in ALLOWED_METHODS:
        raise ValueError(f"method 须为 {sorted(ALLOWED_METHODS)}")

    root = Path(batch_root)
    data = load_manifest(root)
    if not data:
        raise FileNotFoundError(f"无 manifest: {root}")
    lot = dict(data.get("lot") or {})
    if not lot:
        raise ValueError("请先 prepare（manifest.lot 为空）")

    lot_path = Path(
        (data.get("stages") or {}).get(STAGE_NAME, {}).get("paths", {}).get("lot")
        or data.get("deliver_path")
        or ""
    )
    if not lot_path.is_file():
        raise FileNotFoundError(f"找不到 lot: {lot_path}")

    raw = pd.read_csv(labeled_csv, dtype=str, low_memory=False)
    labeled = normalize_frame(
        raw,
        category=category,
        source=source,
        batch=batch or str(data.get("batch") or ""),
        dimension="lot_accept",
        label_col=label_col,
        id_col=id_col,
        labeled_at=time.strftime("%Y-%m-%d %H:%M:%S"),
    )
    if labeled.empty:
        raise ValueError("标注表无有效标签")

    sample_ids = set(labeled["video_id"].astype(str))
    lot_ids = load_id_set(lot_path)
    assert_ids_subset(sample_ids, lot_ids, label="验收标注")

    rate = pass_rate(labeled)
    n = len(labeled)
    n_fail = int((labeled["human_label"] == "fail").sum())

    decision = "reject"
    detail: dict[str, Any] = {
        "n": n,
        "n_pass": int((labeled["human_label"] == "pass").sum()),
        "n_fail": n_fail,
        "pass_rate": round(rate, 4),
    }

    if method == "ci_estimate":
        decision = "accept" if rate >= min_pass_rate else "reject"
        detail["min_pass_rate"] = min_pass_rate
    else:
        # AQL：须显式 Ac/Re；未提供则拒写
        if ac is None or re is None:
            raise ValueError("method=aql 时须提供 --ac 与 --re（查 ISO 2859 表）")
        decision = "accept" if n_fail <= ac else "reject"
        detail["aql"] = aql
        detail["ac"] = ac
        detail["re"] = re

    qc_dir = root / "03_qc"
    qc_dir.mkdir(parents=True, exist_ok=True)
    labeled_out = qc_dir / "lot_accept_labeled.csv"
    labeled.to_csv(labeled_out, index=False)

    lot.update({
        "method": method,
        "n": n,
        "pass_rate": detail["pass_rate"],
        "decision": decision,
        "min_pass_rate": min_pass_rate if method == "ci_estimate" else None,
        "aql": detail.get("aql"),
        "ac": detail.get("ac"),
        "re": detail.get("re"),
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    })
    data["lot"] = lot
    save_manifest(root, data)

    update_stage(
        root,
        STAGE_NAME,
        paths={"labeled": str(labeled_out)},
        stats={**detail, "decision": decision, "method": method},
    )
    return lot


def verify_existing(
    batch_root: str | Path,
    *,
    min_pass_rate: float = 0.85,
    max_overturn: float = 0.08,
) -> dict[str, Any]:
    """用已有 human_qc + screening overturn 写 lot 判决——不加第二轮人工。

    诚实口径：pass_rate 来自筛前抽检（evidence_frame=quality）；
    deliver_frame 可为 remain。不宣称「remain 独立 CI」。
    """
    root = Path(batch_root)
    data = load_manifest(root)
    if not data:
        raise FileNotFoundError(f"无 manifest: {root}")
    lot = dict(data.get("lot") or {})
    if not lot:
        raise ValueError("请先 prepare（manifest.lot 为空）")

    stages = data.get("stages") or {}
    hq = (stages.get("human_qc") or {}).get("stats") or {}
    if "pass_rate" not in hq or "n_labeled" not in hq:
        raise ValueError("缺少 stages.human_qc.stats（n_labeled/pass_rate）；请先 ingest_human_qc")

    n = int(hq["n_labeled"])
    rate = float(hq["pass_rate"])
    overturn = None
    mh = (stages.get("ml_highconf_drop") or {}).get("stats") or {}
    if "overturn_rate" in mh:
        overturn = float(mh["overturn_rate"])
    else:
        summary = (stages.get("ml_highconf_drop") or {}).get("paths", {}).get("summary")
        if summary and Path(summary).is_file():
            import json

            blob = json.loads(Path(summary).read_text(encoding="utf-8"))
            overturn = float((blob.get("overturn") or {}).get("overturn_rate") or 0)

    ok_rate = rate >= min_pass_rate
    ok_overturn = True if overturn is None else overturn <= max_overturn
    decision = "accept" if (ok_rate and ok_overturn) else "reject"

    notes = (
        f"verify_existing：复用筛前 human_qc pass_rate={rate:.4f} (n={n})；"
        f"deliver_frame={lot.get('deliver_frame')}；"
        f"screening overturn={overturn} (max={max_overturn})。"
        "不加第二轮人工；非 remain 独立 CI。"
    )
    lot.update({
        "method": "prescreen_plus_screening",
        "evidence_frame": "quality",
        "n": n,
        "pass_rate": round(rate, 4),
        "min_pass_rate": min_pass_rate,
        "overturn_rate": overturn,
        "max_overturn": max_overturn,
        "decision": decision,
        "notes": notes,
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    })
    # sample_frame 记录证据来源，避免与 deliver 混淆
    lot["sample_frame"] = "quality"
    data["lot"] = lot
    save_manifest(root, data)

    update_stage(
        root,
        STAGE_NAME,
        stats={
            "n": n,
            "pass_rate": round(rate, 4),
            "overturn_rate": overturn,
            "decision": decision,
            "method": "prescreen_plus_screening",
            "evidence_frame": "quality",
            "deliver_frame": lot.get("deliver_frame"),
        },
    )
    return lot
