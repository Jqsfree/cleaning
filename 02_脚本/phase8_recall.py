#!/usr/bin/env python3
"""
phase8_recall.py — DROP 强信号召回

用强信号正则从 DROP 中筛选候选体育视频。

模式:
  tight   — vs/final/championship + 运动专属词 (默认)
  medium  — 加 olympic/world cup/grand prix 等
  wide    — 加 channels 白名单

用法:
  python3 phase8_recall.py drop.parquet -s curling -o recovered.parquet
  python3 phase8_recall.py drop.parquet -s equestrian --mode medium --channels "Olympics|FEI" -o recovered.parquet
"""

import sys, os, argparse
import duckdb

SPORT_TERMS = {
    "curling": r"curl|curling|broom|skip|stone|sheet",
    "equestrian": r"horse|equestrian|dressage|show.?jumping|eventing|steeplechase|polo|rodeo|derby|FEI|Longines",
    "climbing": r"climb|boulder|lead.?climbing|speed.?climbing|IFSC|rock.?master",
    "ice_hockey": r"hockey|NHL|ice.?hockey|puck|slapshot",
    "wrestling": r"wrestl|UFC|mma|grappl|jiu.?jitsu",
}


def build_signals(sport: str, mode: str, channels: str) -> list:
    signals = []
    signals.append(r"\b(vs\.?\b|versus)\b")
    signals.append(r"\b(championship|championships|final\b|finals\b|semi.?final|quarter.?final|playoff|round.?robin)\b")
    st = SPORT_TERMS.get(sport, "")
    if st:
        signals.append(rf"\b({st})\b")
    if mode in ("medium", "wide"):
        signals.append(r"\b(olympic|world cup|grand prix|grand slam|nationals?)\b")
        signals.append(r"\b(highlights|full match|full replay|commentary)\b")
    if mode == "wide":
        signals.append(r"\b(NBA|NFL|MLB|NHL|FIFA|ATP|WTA|UFC|NCAA|WNBA|MLS|PGA|LPGA)\b")
    if channels:
        signals.append(rf"\b({channels})\b")
    return signals


def main():
    parser = argparse.ArgumentParser(description="DROP 强信号召回")
    parser.add_argument("input", help="DROP parquet")
    parser.add_argument("-s", "--sport", default="", help="运动名")
    parser.add_argument("-o", "--output", required=True, help="输出 parquet")
    parser.add_argument("--mode", default="tight", choices=["tight", "medium", "wide"])
    parser.add_argument("--channels", default="", help="频道白名单正则")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"[ERROR] 文件不存在: {args.input}")
        sys.exit(1)

    signals = build_signals(args.sport, args.mode, args.channels)
    con = duckdb.connect()
    n_total = con.execute(f"SELECT COUNT(*) FROM read_parquet('{args.input}')").fetchone()[0]

    conditions = []
    for s in signals:
        conditions.append(f"regexp_matches(lower(title), '{s}', 'i')")
    if args.channels:
        conditions.append(f"regexp_matches(lower(channel), '{args.channels}', 'i')")

    where_clause = " OR\n           ".join(conditions)
    con.execute(f"CREATE TEMP TABLE recovered AS SELECT * FROM read_parquet('{args.input}') WHERE {where_clause}")
    n = con.execute("SELECT COUNT(*) FROM recovered").fetchone()[0]

    print(f"模式: {args.mode} | 运动: {args.sport or '(none)'}")
    print(f"DROP: {n_total:,} → 召回: {n:,} ({n/max(n_total,1)*100:.1f}%)")

    if not args.dry_run:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        con.execute(f"COPY (SELECT * FROM recovered) TO '{args.output}' (FORMAT PARQUET)")
        print(f"已写出: {args.output}")
    else:
        print("[dry-run]")

    con.close()


if __name__ == "__main__":
    main()

