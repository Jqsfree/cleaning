#!/usr/bin/env python3
"""
promote_rules.py -- 规则推广：数据集规则 → 主规则

前提: 数据集规则已在 Phase 5/6 验证通过。

流程:
  1. 调用 backup_rules.backup() 备份当前主规则
  2. 复制数据集规则覆盖主规则
  3. 要求用户确认

用法:
  python3 promote_rules.py data/runs/pingpong/rules/
"""

import sys, os, shutil, tomllib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from backup_rules import backup, MAIN_RULES


def promote(dataset_dir: str) -> tuple[bool, str]:
    src = Path(dataset_dir)
    if not src.exists():
        return False, f"目录不存在: {dataset_dir}"
    src_bl = src / "blacklist.toml"
    if not src_bl.exists():
        return False, f"缺少 blacklist.toml: {dataset_dir}"

    try:
        r = tomllib.loads(src_bl.read_text())
        p2 = len(r.get("pass2", []))
        r2 = len(r.get("r2", []))
    except Exception as e:
        return False, f"TOML 解析失败: {e}"

    bak = backup()
    for name in ["blacklist.toml", "whitelist.toml", "entities.toml"]:
        s = src / name
        if s.exists():
            shutil.copy2(s, MAIN_RULES / name)

    return True, f"已推广 {p2+r2} 条 blacklist 规则 | 备份: {bak.name}"


def main():
    if len(sys.argv) < 2:
        print("用法: python3 promote_rules.py <数据集规则目录>")
        sys.exit(1)

    dataset_dir = sys.argv[1]
    src = Path(dataset_dir)

    src_bl = src / "blacklist.toml"
    if src_bl.exists():
        r = tomllib.loads(src_bl.read_text())
        p2 = len(r.get("pass2", []))
        r2 = len(r.get("r2", []))
        print(f"数据集规则: {p2} pass2 + {r2} r2 = {p2+r2} total")
    else:
        print(f"[ERROR] 缺少 blacklist.toml"); sys.exit(1)

    main_bl = MAIN_RULES / "blacklist.toml"
    mr = tomllib.loads(main_bl.read_text())
    mp2 = len(mr.get("pass2", []))
    mr2 = len(mr.get("r2", []))
    print(f"主规则:     {mp2} pass2 + {mr2} r2 = {mp2+mr2} total\n")

    answer = input("确认推广? [y/N]: ").strip().lower()
    if answer not in ("y", "yes"):
        print("已取消。"); sys.exit(0)

    ok, msg = promote(dataset_dir)
    print(f"\n{'✅' if ok else '❌'} {msg}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
