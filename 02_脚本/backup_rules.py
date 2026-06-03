#!/usr/bin/env python3
"""
backup_rules.py -- 主规则备份 & 恢复

每次修改规则前自动备份，支持回滚。

用法:
  python3 backup_rules.py              # 备份当前主规则
  python3 backup_rules.py --list       # 列出所有备份
  python3 backup_rules.py --restore N  # 恢复到第 N 个备份
"""

import sys, os, shutil
from datetime import datetime
from pathlib import Path

MAIN_RULES = Path(__file__).resolve().parent / "rules"
BACKUP_DIR = MAIN_RULES.parent / "rules_backups"


def backup():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    dst = BACKUP_DIR / ts
    dst.mkdir(parents=True, exist_ok=True)
    for f in MAIN_RULES.glob("*.toml"):
        shutil.copy2(f, dst / f.name)
    print(f"✅ 备份: {dst}")
    return dst


def list_backups():
    if not BACKUP_DIR.exists():
        print("无备份。")
        return []
    backups = sorted(BACKUP_DIR.iterdir(), reverse=True)
    for i, b in enumerate(backups):
        print(f"  [{i}] {b.name}")
    return backups


def restore(index: int):
    backups = sorted(BACKUP_DIR.iterdir())
    if index < 0 or index >= len(backups):
        print(f"无效索引: {index}")
        sys.exit(1)
    src = backups[index]
    # 先备份当前
    backup()
    for f in src.glob("*.toml"):
        shutil.copy2(f, MAIN_RULES / f.name)
    print(f"✅ 恢复: {src.name} → {MAIN_RULES}")


def main():
    if "--list" in sys.argv:
        list_backups()
    elif "--restore" in sys.argv:
        idx = int(sys.argv[sys.argv.index("--restore") + 1])
        restore(idx)
    else:
        backup()


if __name__ == "__main__":
    main()
