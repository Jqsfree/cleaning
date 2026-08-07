#!/usr/bin/env python3
"""导出 vision_pass_19 在 sb0/sb1/sb2 下的 storyboard 抽帧图片。"""

import sys
from pathlib import Path

import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))

from qc_vision_welding import (  # noqa: E402
    DEFAULT_FRAMES,
    YtDlpAuth,
    detect_js_runtimes,
    fetch_storyboard_frames,
    get_storyboard_info,
    log,
    prefetch_browser_cookies,
)

PASS_CSV = _SCRIPT_DIR.parent / "data/runs/welding/vision_qc_pass/vision_pass_19.csv"
OUT_DIR = _SCRIPT_DIR.parent / "data/runs/welding/vision_qc_pass/pass19_sb_frames"
TIERS = ("sb0", "sb1", "sb2")


def main():
    df = pd.read_csv(PASS_CSV)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cookie_path = OUT_DIR / ".cookies_chrome.txt"
    auth = prefetch_browser_cookies(YtDlpAuth(cookies_from_browser=("chrome",)), str(cookie_path))
    js = detect_js_runtimes()

    manifest = []
    ok = fail = 0

    for _, row in df.iterrows():
        vid = row["video_id"]
        title = str(row.get("title", ""))[:60]
        for tier in TIERS:
            sb_info, err = get_storyboard_info(
                vid, auth=auth, sb_prefer_order=[tier], js_runtimes=js,
            )
            dest = OUT_DIR / tier / vid
            dest.mkdir(parents=True, exist_ok=True)

            if sb_info is None:
                (dest / "FAILED.txt").write_text(err or "unknown", encoding="utf-8")
                fail += 1
                log(f"[FAIL] {tier} {vid} {err}")
                continue

            fmt_id = tier  # requested tier; actual may differ if fallback — read from log
            w, h = sb_info["frame_w"], sb_info["frame_h"]
            frames = fetch_storyboard_frames(sb_info, n_frames=DEFAULT_FRAMES)
            if not frames:
                (dest / "FAILED.txt").write_text("no_frames", encoding="utf-8")
                fail += 1
                continue

            for i, img in enumerate(frames):
                img.save(dest / f"frame_{i:02d}.jpg", format="JPEG", quality=90)

            (dest / "meta.txt").write_text(
                f"video_id={vid}\n"
                f"title={title}\n"
                f"tier={tier}\n"
                f"frame={w}x{h}\n"
                f"frames_saved={len(frames)}\n",
                encoding="utf-8",
            )
            manifest.append({"video_id": vid, "tier": tier, "w": w, "h": h, "n": len(frames)})
            ok += 1
            log(f"[OK] {tier} {vid} {w}x{h} x{len(frames)}")

    pd.DataFrame(manifest).to_csv(OUT_DIR / "manifest.csv", index=False, encoding="utf-8-sig")
    log(f"完成: {ok} 成功, {fail} 失败 → {OUT_DIR}")

    from view_sb_compare import write_compare_html  # noqa: E402
    html_path = write_compare_html(OUT_DIR)
    log(f"对比页: {html_path}  （打开: python3 {_SCRIPT_DIR}/view_sb_compare.py {OUT_DIR}）")


if __name__ == "__main__":
    main()
