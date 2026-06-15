#!/usr/bin/env python3
"""
core/rules_manager.py -- 规则加载、备份、保护

规则优先级：数据集专属 rules.toml > 主规则 (02_脚本/rules/)

备份：修改前自动创建 .bak_YYYYMMDD_HHmmss 副本
保护：主规则修改需要用户确认
"""

import os, re, shutil, tomllib, json
from datetime import datetime
from pathlib import Path

MAIN_RULES_DIR = Path(__file__).resolve().parent.parent / "rules"


def sync_current_to_main():
    """将 current/ (真理源) 同步到 rules/ (管道实际读取)。

    SOP 要求总是在 current/ 中修改规则，然后调用此函数同步。
    """
    src_dir = MAIN_RULES_DIR / "current"
    if not src_dir.exists():
        print("[SYNC] current/ 不存在，跳过")
        return
    for name in ["blacklist.toml", "whitelist.toml", "entities.toml"]:
        src = src_dir / name
        dst = MAIN_RULES_DIR / name
        if src.exists():
            shutil.copy2(src, dst)
    print("[SYNC] current/ → rules/")


# ══════════════════════════════════
# 加载
# ══════════════════════════════════

def load_rules(dataset_rules_path: str | None = None) -> dict:
    """
    加载完整规则集。
    优先 dataset_rules_path（数据集专属），回退主规则目录。

    返回 {blacklist, whitelist, entities, _source}
    """
    if dataset_rules_path and os.path.exists(dataset_rules_path):
        return _load_from_dir(os.path.dirname(dataset_rules_path), source="dataset")
    return _load_from_dir(str(MAIN_RULES_DIR), source="main")


def _load_from_dir(dir_path: str, source: str) -> dict:
    """从指定目录加载三个 TOML 文件。"""
    d = Path(dir_path)
    result = {"_source": source, "_dir": str(d)}

    for name in ["blacklist", "whitelist", "entities"]:
        p = d / f"{name}.toml"
        if p.exists():
            with open(p, "rb") as f:
                result[name] = tomllib.load(f)

    return result


def get_rules_summary(rules: dict) -> str:
    """规则集摘要：来源 + 规则数量。"""
    src = rules.get("_source", "?")
    d = rules.get("_dir", "?")
    bl_pass2 = len(rules.get("blacklist", {}).get("pass2", []))
    bl_r2 = len(rules.get("blacklist", {}).get("r2", []))
    pos = len(rules.get("whitelist", {}).get("positive", []))
    neg = len(rules.get("whitelist", {}).get("negative", []))
    lex = len(rules.get("entities", {}).get("sports", []))
    return f"[{src}] {d} — blacklist:{bl_pass2}+{bl_r2} pos:{pos} neg:{neg} lexicon:{lex}"


# ══════════════════════════════════
# 备份
# ══════════════════════════════════

def backup_rules(rules_dir: str) -> str:
    """备份整个规则目录到 .bak_时间戳。返回备份路径。"""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    bak = f"{rules_dir}.bak_{ts}"
    if os.path.exists(rules_dir):
        shutil.copytree(rules_dir, bak)
        print(f"[BACKUP] {rules_dir} → {bak}")
    return bak


def backup_single(file_path: str) -> str:
    """备份单个规则文件。"""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    bak = f"{file_path}.bak_{ts}"
    if os.path.exists(file_path):
        shutil.copy2(file_path, bak)
        print(f"[BACKUP] {file_path} → {bak}")
    return bak


# ══════════════════════════════════
# 保护
# ══════════════════════════════════

def is_main_rules(rules_dir: str) -> bool:
    """判断是否是受保护的主规则目录。"""
    return str(Path(rules_dir).resolve()) == str(MAIN_RULES_DIR.resolve())


def check_modification_allowed(rules_dir: str) -> tuple[bool, str]:
    """
    检查是否允许修改。
    主规则 — 返回 (False, 原因)
    数据集规则 — 返回 (True, "")
    """
    if is_main_rules(rules_dir):
        return (False,
                "主规则受保护。请先修改数据集专属 rules.toml，验证通过后再推广。\n"
                f"推广命令: python3 promote_rules.py {rules_dir}")
    return (True, "")


# ══════════════════════════════════
# 推广
# ══════════════════════════════════

def promote_rules(dataset_rules_dir: str) -> tuple[bool, str]:
    """
    将数据集规则推广到主规则。
    1. 备份主规则
    2. 复制数据集规则覆盖主规则
    3. 记录推广日志

    返回 (成功, 消息)
    """
    dataset_dir = Path(dataset_rules_dir)
    if not dataset_dir.exists():
        return (False, f"数据集规则目录不存在: {dataset_dir}")

    if not (dataset_dir / "blacklist.toml").exists():
        return (False, f"数据集规则不完整，缺少 blacklist.toml")

    # 备份主规则
    bak = backup_rules(str(MAIN_RULES_DIR))

    # 复制
    for name in ["blacklist.toml", "whitelist.toml", "entities.toml"]:
        src = dataset_dir / name
        dst = MAIN_RULES_DIR / name
        if src.exists():
            shutil.copy2(src, dst)

    # 记录推广日志
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "source": str(dataset_dir),
        "backup": bak,
    }
    log_path = MAIN_RULES_DIR / "promote_log.json"
    history = []
    if log_path.exists():
        history = json.loads(log_path.read_text())
    history.append(log_entry)
    log_path.write_text(json.dumps(history, indent=2, ensure_ascii=False))

    return (True, f"推广成功。备份: {bak}")


# ══════════════════════════════════
# 初始化数据集规则
# ══════════════════════════════════

def init_dataset_rules(dataset_rules_dir: str, force: bool = True) -> str:
    """
    初始化数据集规则目录。
    每次 Phase 5 启动时同步主规则到数据集目录。
    force=True 可强制覆盖。
    """
    dst = Path(dataset_rules_dir)
    if (dst / "blacklist.toml").exists() and not force:
        return str(dst)
    os.makedirs(dst, exist_ok=True)
    for name in ["blacklist.toml", "whitelist.toml", "entities.toml"]:
        src = MAIN_RULES_DIR / name
        if src.exists():
            shutil.copy2(src, dst / name)
    action = "强制同步" if force else "初始化"
    print(f"[INIT] 数据集规则 {action}: {dst}")
    return str(dst)
