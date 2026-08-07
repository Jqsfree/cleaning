"""
core/exemplar_sim.py — 样例视频抽帧原型 + 候选缩略图相似度

直播 API 不是硬门禁；以本地样例视频视觉原型为主滤。
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
import threading
import torch
from PIL import Image

VIDEO_EXTS = {".mp4", ".mkv", ".webm", ".avi", ".mov"}
DEFAULT_MODEL = "ViT-B-32"
DEFAULT_PRETRAINED = "openai"
DEFAULT_N_FRAMES = 12
THUMB_URL = "https://i.ytimg.com/vi/{vid}/hqdefault.jpg"


def list_videos(video_dir: str | Path) -> list[Path]:
    root = Path(video_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"样例目录不存在: {root}")
    files = sorted(
        p for p in root.iterdir()
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS
    )
    if not files:
        raise FileNotFoundError(f"目录无视频: {root}")
    return files


def extract_frames(
    video_path: Path,
    out_dir: Path,
    *,
    n_frames: int = DEFAULT_N_FRAMES,
    trim_pct: float = 0.05,
) -> list[Path]:
    """均匀抽帧（跳过首尾 trim_pct）；依赖 ffmpeg。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    # 时长
    probe = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(video_path),
        ],
        capture_output=True, text=True, check=False,
    )
    try:
        duration = float((probe.stdout or "").strip() or "0")
    except ValueError:
        duration = 0.0
    if duration <= 0:
        duration = 60.0

    start = duration * trim_pct
    end = duration * (1.0 - trim_pct)
    if end <= start:
        start, end = 0.0, duration
    span = max(end - start, 0.1)
    paths: list[Path] = []
    for i in range(n_frames):
        t = start + (i + 0.5) / n_frames * span
        out = out_dir / f"f{i:02d}.jpg"
        r = subprocess.run(
            [
                "ffmpeg", "-y", "-ss", f"{t:.3f}", "-i", str(video_path),
                "-frames:v", "1", "-q:v", "2", str(out),
            ],
            capture_output=True, text=True, check=False,
        )
        if r.returncode == 0 and out.is_file() and out.stat().st_size > 0:
            paths.append(out)
    if not paths:
        raise RuntimeError(f"抽帧失败: {video_path}\n{r.stderr[-400:] if r else ''}")
    return paths


class ClipEncoder:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        pretrained: str = DEFAULT_PRETRAINED,
        device: str | None = None,
    ):
        import open_clip

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained,
        )
        self.model = self.model.to(self.device).eval()
        self.model_name = model_name
        self.pretrained = pretrained

    @torch.no_grad()
    def encode_images(self, images: list[Image.Image]) -> np.ndarray:
        if not images:
            return np.zeros((0, 512), dtype=np.float32)
        tensors = torch.stack([self.preprocess(im.convert("RGB")) for im in images])
        tensors = tensors.to(self.device)
        feats = self.model.encode_image(tensors)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.detach().float().cpu().numpy()

    def encode_paths(self, paths: list[Path], *, batch_size: int = 32) -> np.ndarray:
        feats: list[np.ndarray] = []
        batch: list[Image.Image] = []
        for p in paths:
            batch.append(Image.open(p).convert("RGB"))
            if len(batch) >= batch_size:
                feats.append(self.encode_images(batch))
                batch = []
        if batch:
            feats.append(self.encode_images(batch))
        return np.concatenate(feats, axis=0) if feats else np.zeros((0, 512), dtype=np.float32)


def build_bank(
    video_dir: str | Path,
    out_dir: str | Path,
    *,
    n_frames: int = DEFAULT_N_FRAMES,
    model_name: str = DEFAULT_MODEL,
    pretrained: str = DEFAULT_PRETRAINED,
    symlink: bool = True,
) -> dict[str, Any]:
    """样例视频 → frames + prototypes.npy + manifest.csv。"""
    out = Path(out_dir)
    videos_dir = out / "videos"
    frames_root = out / "frames"
    videos_dir.mkdir(parents=True, exist_ok=True)
    frames_root.mkdir(parents=True, exist_ok=True)

    src_files = list_videos(video_dir)
    encoder = ClipEncoder(model_name, pretrained)

    rows: list[dict[str, Any]] = []
    prototypes: list[np.ndarray] = []

    for src in src_files:
        eid = src.stem
        dest = videos_dir / src.name
        if not dest.exists():
            if symlink:
                dest.symlink_to(src.resolve())
            else:
                import shutil
                shutil.copy2(src, dest)

        frame_dir = frames_root / eid
        frame_paths = extract_frames(src, frame_dir, n_frames=n_frames)
        feats = encoder.encode_paths(frame_paths)
        proto = feats.mean(axis=0)
        proto = proto / (np.linalg.norm(proto) + 1e-8)
        prototypes.append(proto.astype(np.float32))
        rows.append({
            "exemplar_id": eid,
            "path": str(dest),
            "source_path": str(src.resolve()),
            "n_frames": len(frame_paths),
            "frame_dir": str(frame_dir),
        })
        print(f"  [{len(rows)}/{len(src_files)}] {eid}  frames={len(frame_paths)}")

    proto_arr = np.stack(prototypes, axis=0)
    np.save(out / "prototypes.npy", proto_arr)
    manifest = pd.DataFrame(rows)
    manifest.to_csv(out / "manifest.csv", index=False)
    meta = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "video_dir": str(Path(video_dir).resolve()),
        "n_exemplars": len(rows),
        "n_frames": n_frames,
        "model": model_name,
        "pretrained": pretrained,
        "note": "直播场景视觉原型；liveBroadcastContent 非硬门禁",
    }
    (out / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    return meta


def load_bank(bank_dir: str | Path) -> tuple[np.ndarray, pd.DataFrame, dict[str, Any]]:
    root = Path(bank_dir)
    proto = np.load(root / "prototypes.npy")
    manifest = pd.read_csv(root / "manifest.csv")
    meta = json.loads((root / "meta.json").read_text(encoding="utf-8"))
    return proto, manifest, meta


def build_bank_from_video_ids(
    video_ids: list[str],
    out_dir: str | Path,
    *,
    cache_dir: str | Path,
    model_name: str = DEFAULT_MODEL,
    pretrained: str = DEFAULT_PRETRAINED,
    thumb_workers: int = 16,
    batch_size: int = 64,
    note: str = "负例缩略图原型（人工 F）",
    labels: dict[str, str] | None = None,
) -> dict[str, Any]:
    """video_id 列表 → 下载缩略图编码为 prototypes（每 id 一原型）。"""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cache = Path(cache_dir)
    ids = [str(v).strip() for v in video_ids if str(v).strip()]
    if not ids:
        raise ValueError("video_ids 为空")

    encoder = ClipEncoder(model_name, pretrained)
    paths = fetch_thumbnails_batch(ids, cache, workers=thumb_workers)
    rows: list[dict[str, Any]] = []
    prototypes: list[np.ndarray] = []
    ok_ids: list[str] = []
    ok_paths: list[Path] = []
    for vid, path in zip(ids, paths):
        if path is None:
            print(f"  [skip] thumb fail {vid}")
            continue
        ok_ids.append(vid)
        ok_paths.append(path)

    for start in range(0, len(ok_paths), batch_size):
        chunk_paths = ok_paths[start : start + batch_size]
        chunk_ids = ok_ids[start : start + batch_size]
        imgs = [Image.open(p).convert("RGB") for p in chunk_paths]
        feats = encoder.encode_images(imgs)
        for j, vid in enumerate(chunk_ids):
            proto = feats[j]
            proto = proto / (np.linalg.norm(proto) + 1e-8)
            prototypes.append(proto.astype(np.float32))
            rows.append({
                "exemplar_id": vid,
                "path": str(chunk_paths[j]),
                "source_path": thumb_url(vid),
                "n_frames": 1,
                "frame_dir": "",
                "label": (labels or {}).get(vid, ""),
            })
        print(f"  encode neg {min(start + len(chunk_ids), len(ok_ids))}/{len(ok_ids)}", flush=True)

    if not prototypes:
        raise RuntimeError("无可用负例缩略图")

    proto_arr = np.stack(prototypes, axis=0)
    np.save(out / "prototypes.npy", proto_arr)
    pd.DataFrame(rows).to_csv(out / "manifest.csv", index=False)
    meta = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_exemplars": len(rows),
        "n_requested": len(ids),
        "n_frames": 1,
        "model": model_name,
        "pretrained": pretrained,
        "source": "video_id_thumbs",
        "note": note,
    }
    (out / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    return meta


def score_max_sim_to_bank(
    video_ids: list[str],
    prototypes: np.ndarray,
    exemplar_ids: list[str],
    encoder: ClipEncoder,
    *,
    cache_dir: str | Path,
    batch_size: int = 64,
    thumb_workers: int = 16,
    exclude_self: bool = True,
) -> pd.DataFrame:
    """候选缩略图 vs bank：max cosine；exclude_self 时跳过同 video_id 原型（LOO）。"""
    cache = Path(cache_dir)
    paths = fetch_thumbnails_batch(video_ids, cache, workers=thumb_workers)
    scores = np.full(len(video_ids), np.nan, dtype=np.float32)
    nearest = np.full(len(video_ids), "", dtype=object)
    eid_list = [str(e) for e in exemplar_ids]
    eid_index = {e: i for i, e in enumerate(eid_list)}

    ok_idx = [i for i, p in enumerate(paths) if p is not None]
    for start in range(0, len(ok_idx), batch_size):
        chunk_i = ok_idx[start : start + batch_size]
        imgs = [Image.open(paths[i]).convert("RGB") for i in chunk_i]
        feats = encoder.encode_images(imgs)
        sims = feats @ prototypes.T  # (B, E)
        for j, ii in enumerate(chunk_i):
            row = sims[j].copy()
            if exclude_self:
                self_i = eid_index.get(str(video_ids[ii]))
                if self_i is not None:
                    row[self_i] = -np.inf
            if not np.isfinite(row).any():
                continue
            best = int(np.argmax(row))
            scores[ii] = float(row[best])
            nearest[ii] = eid_list[best]
        done = min(start + len(chunk_i), len(ok_idx))
        if start % (batch_size * 5) == 0 or done >= len(ok_idx):
            print(f"  encode {done}/{len(ok_idx)}", flush=True)

    return pd.DataFrame({
        "video_id": video_ids,
        "neg_sim": scores,
        "nearest_neg_id": nearest,
        "thumb_ok": [p is not None for p in paths],
    })


def thumb_url(video_id: str) -> str:
    return THUMB_URL.format(vid=str(video_id).strip())


def fetch_thumbnail(
    video_id: str,
    cache_dir: Path,
    *,
    timeout: float = 15.0,
    session: requests.Session | None = None,
) -> Path | None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{video_id}.jpg"
    if path.is_file() and path.stat().st_size > 0:
        return path
    sess = session or requests.Session()
    url = thumb_url(video_id)
    try:
        r = sess.get(url, timeout=timeout)
        if r.status_code != 200 or not r.content:
            return None
        path.write_bytes(r.content)
        return path
    except requests.RequestException:
        return None


def fetch_thumbnails_batch(
    video_ids: list[str],
    cache_dir: Path,
    *,
    workers: int = 16,
    timeout: float = 15.0,
) -> list[Path | None]:
    """并发下载缩略图；已缓存则跳过。"""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    cache_dir.mkdir(parents=True, exist_ok=True)
    out: list[Path | None] = [None] * len(video_ids)
    sess_local = threading.local()

    def _session() -> requests.Session:
        s = getattr(sess_local, "s", None)
        if s is None:
            s = requests.Session()
            sess_local.s = s
        return s

    def _one(i: int, vid: str) -> tuple[int, Path | None]:
        return i, fetch_thumbnail(vid, cache_dir, timeout=timeout, session=_session())

    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futs = [ex.submit(_one, i, vid) for i, vid in enumerate(video_ids)]
        done = 0
        for fut in as_completed(futs):
            i, path = fut.result()
            out[i] = path
            done += 1
            if done % 500 == 0 or done == len(video_ids):
                print(f"  thumbs {done}/{len(video_ids)}", flush=True)
    return out


def score_candidates(
    video_ids: list[str],
    prototypes: np.ndarray,
    exemplar_ids: list[str],
    encoder: ClipEncoder,
    *,
    cache_dir: str | Path,
    batch_size: int = 64,
    high: float = 0.28,
    mid: float = 0.22,
    thumb_workers: int = 16,
) -> pd.DataFrame:
    """对候选 video_id 缩略图打分；band=high|mid|low。"""
    cache = Path(cache_dir)
    paths = fetch_thumbnails_batch(
        video_ids, cache, workers=thumb_workers,
    )

    # encode in batches of existing thumbs
    scores = np.full(len(video_ids), np.nan, dtype=np.float32)
    nearest = np.full(len(video_ids), "", dtype=object)
    ok_idx = [i for i, p in enumerate(paths) if p is not None]
    for start in range(0, len(ok_idx), batch_size):
        chunk_i = ok_idx[start : start + batch_size]
        imgs = [Image.open(paths[i]).convert("RGB") for i in chunk_i]
        feats = encoder.encode_images(imgs)  # (B, D)
        # cosine: feats @ prototypes.T
        sims = feats @ prototypes.T  # (B, E)
        best = sims.argmax(axis=1)
        best_s = sims.max(axis=1)
        for j, ii in enumerate(chunk_i):
            scores[ii] = float(best_s[j])
            nearest[ii] = exemplar_ids[int(best[j])]
        if start % (batch_size * 5) == 0 or start + len(chunk_i) >= len(ok_idx):
            print(f"  encode {min(start + len(chunk_i), len(ok_idx))}/{len(ok_idx)}", flush=True)

    bands = []
    for s in scores:
        if np.isnan(s):
            bands.append("error")
        elif s >= high:
            bands.append("high")
        elif s >= mid:
            bands.append("mid")
        else:
            bands.append("low")

    return pd.DataFrame({
        "video_id": video_ids,
        "sim_score": scores,
        "nearest_exemplar_id": nearest,
        "band": bands,
        "thumb_ok": [p is not None for p in paths],
    })


def assign_bands_by_quantile(
    scores: pd.Series,
    *,
    high_q: float = 0.70,
    mid_q: float = 0.40,
) -> tuple[float, float]:
    """用有效分数分位数建议阈值（宁留：mid 门槛偏低）。"""
    s = scores.dropna()
    if s.empty:
        return 0.28, 0.22
    high = float(s.quantile(high_q))
    mid = float(s.quantile(mid_q))
    return high, mid
