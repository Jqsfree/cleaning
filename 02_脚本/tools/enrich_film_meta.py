#!/home/jqs/miniconda3/envs/data_cleaning/bin/python
"""
enrich_film_meta.py — 影视剧补「频道国家」+ 类型（粗 category / 细 地区剧种·剧种）

口径（见计划）:
  - 频道国家 = channels.snippet.country（ISO；常空）
  - YouTube 粗类 = videos.snippet.categoryId
  - 细类 = 参考标注 join；未命中用标题/频道规则（可选 LLM）

用法:
  export YOUTUBE_API_KEY='...'
  02_脚本/tools/enrich_film_meta.py TARGET.csv --ref REF.csv -o OUT.csv
  # 无 YouTube API：跳过接口，用「地区剧种→ISO」推断频道国家
  02_脚本/tools/enrich_film_meta.py TARGET.csv --ref REF.csv -o OUT.csv --skip-yt --infer-country
  02_脚本/tools/enrich_film_meta.py TARGET.csv --ref REF.csv -o OUT.csv --resume
  02_脚本/tools/enrich_film_meta.py TARGET.csv --ref REF.csv -o OUT.csv --llm -w 8

无 API 时「频道国家」来自内容口径（地区剧种），不是频道登记国；见 country_source=infer_region。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from core.progress import ThrottledProgress, mark_done  # noqa: E402
from core.sop import write_run_log  # noqa: E402

VIDEOS_URL = "https://www.googleapis.com/youtube/v3/videos"
CHANNELS_URL = "https://www.googleapis.com/youtube/v3/channels"
BATCH_SIZE = 50
DEFAULT_SLEEP = 0.05

# YouTube 固定类目（中文名便于交付）
YT_CATEGORY_ZH: dict[str, str] = {
    "1": "电影与动画",
    "2": "汽车",
    "10": "音乐",
    "15": "宠物与动物",
    "17": "体育",
    "19": "旅游与活动",
    "20": "游戏",
    "22": "人物与博客",
    "23": "喜剧",
    "24": "娱乐",
    "25": "新闻与政治",
    "26": "如何创作与风格",
    "27": "教育",
    "28": "科学与技术",
    "29": "非营利与社会活动",
}

REGION_LABELS = ("国产剧", "港剧", "台剧", "韩剧", "日剧", "美剧", "英剧", "泰剧")

# 无 YT API 时：由地区剧种推断 ISO「频道国家」（内容产地代理，非频道登记国）
REGION_TO_COUNTRY: dict[str, str] = {
    "国产剧": "CN",
    "港剧": "HK",
    "台剧": "TW",
    "韩剧": "KR",
    "日剧": "JP",
    "美剧": "US",
    "英剧": "GB",
    "泰剧": "TH",
}

# 标题/频道 → 地区剧种（先匹配先生效）
_REGION_RULES: list[tuple[str, re.Pattern[str]]] = [
    ("港剧", re.compile(r"(港剧|TVB|邵氏|无线电视|香港粤语|香港剧)", re.I)),
    ("台剧", re.compile(r"(台剧|三立|民视|台視|台视|公视|SET\s*Drama|台灣|台湾偶像剧)", re.I)),
    ("韩剧", re.compile(r"(韩剧|韓劇|韩国|韓國|K-?Drama|Netflix.*Korea|/kr\b)", re.I)),
    ("日剧", re.compile(r"(日剧|日劇|日本电视剧|日劇|J-?Drama)", re.I)),
    ("泰剧", re.compile(r"(泰剧|泰劇|泰国|泰國|Lakorn)", re.I)),
    ("英剧", re.compile(r"(英剧|英劇|BBC\b|ITV\b|英国电视剧|英國)", re.I)),
    ("美剧", re.compile(r"(美剧|美劇|Hollywood|美国电影|美國|Netflix\s*US|HBO\b|Disney\+)", re.I)),
    ("国产剧", re.compile(r"(国产|大陆剧|内地剧|央视|芒果|优酷|爱奇艺|腾讯视频|华语|年代剧|家庭伦理剧)", re.I)),
]

_GENRE_RULES: list[tuple[str, re.Pattern[str]]] = [
    ("短剧", re.compile(r"(短剧|短劇|#短剧|竖屏短剧|reel.?drama)", re.I)),
    ("刑侦/犯罪", re.compile(r"(刑侦|犯罪|破案|警察|刑警|悬疑推理|detective|crime)", re.I)),
    ("缉毒/禁毒", re.compile(r"(缉毒|禁毒|毒贩)", re.I)),
    ("反腐/廉政", re.compile(r"(反腐|廉政|纪委)", re.I)),
    ("职场/商战", re.compile(r"(职场|商战|律师|律政|总裁|公司|workplace)", re.I)),
    ("医疗", re.compile(r"(医疗|医院|医生|护士|外科)", re.I)),
    ("校园/青春", re.compile(r"(校园|青春|大学|高中|学院)", re.I)),
    ("婚姻家庭", re.compile(r"(婚姻|离婚|出轨|夫妻)", re.I)),
    ("家庭伦理", re.compile(r"(家庭伦理|婆媳|家族)", re.I)),
    ("都市情感", re.compile(r"(都市情感|都市爱情|甜宠|虐恋)", re.I)),
    ("喜剧/家庭喜剧", re.compile(r"(喜剧|搞笑|欢喜)", re.I)),
    ("年代/改革", re.compile(r"(年代|改革开放|知青)", re.I)),
    ("农村/乡土", re.compile(r"(农村|乡土|乡村)", re.I)),
    ("军旅/现代军事", re.compile(r"(军旅|特种兵|部队)", re.I)),
    ("消防/救援", re.compile(r"(消防|救援|应急)", re.I)),
    ("女性成长", re.compile(r"(女性成长|大女主)", re.I)),
    ("爱情/电影", re.compile(r"(爱情电影|浪漫电影|爱情片|rom-?com|romance)", re.I)),
    ("现实主义/社会", re.compile(r"(现实题材|社会派|现实主义)", re.I)),
]


def resolve_api_key(cli_key: str | None) -> str:
    key = (cli_key or os.getenv("YOUTUBE_API_KEY", "") or "").strip()
    if not key:
        print("[ERROR] 未设置 YOUTUBE_API_KEY")
        sys.exit(1)
    return key


def _read_table(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in (".parquet", ".pq"):
        return pd.read_parquet(path).astype(str).fillna("")
    return pd.read_csv(path, dtype=str).fillna("")


def _http_get_json(url: str, timeout: float = 60.0) -> dict:
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")[:500]
        raise RuntimeError(f"HTTP {e.code}: {body}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"网络错误: {e}") from e


def fetch_videos_meta(api_key: str, video_ids: list[str]) -> dict[str, dict[str, str]]:
    """videos.list → categoryId, channelId, defaultAudioLanguage."""
    params = urllib.parse.urlencode(
        {
            "part": "snippet",
            "id": ",".join(video_ids),
            "key": api_key,
            "maxResults": BATCH_SIZE,
        }
    )
    payload = _http_get_json(f"{VIDEOS_URL}?{params}")
    found: dict[str, dict[str, str]] = {}
    for item in payload.get("items") or []:
        vid = str(item.get("id") or "").strip()
        sn = item.get("snippet") or {}
        if not vid:
            continue
        cid = str(sn.get("categoryId") or "").strip()
        found[vid] = {
            "yt_category_id": cid,
            "yt_category_name": YT_CATEGORY_ZH.get(cid, ""),
            "yt_channel_id": str(sn.get("channelId") or "").strip(),
            "yt_default_audio_language": str(sn.get("defaultAudioLanguage") or "").strip(),
            "yt_meta_status": "ok",
        }
    out: dict[str, dict[str, str]] = {}
    for vid in video_ids:
        out[vid] = found.get(
            vid,
            {
                "yt_category_id": "",
                "yt_category_name": "",
                "yt_channel_id": "",
                "yt_default_audio_language": "",
                "yt_meta_status": "not_found",
            },
        )
    return out


def fetch_channels_country(api_key: str, channel_ids: list[str]) -> dict[str, str]:
    """channels.list → snippet.country."""
    params = urllib.parse.urlencode(
        {
            "part": "snippet",
            "id": ",".join(channel_ids),
            "key": api_key,
            "maxResults": BATCH_SIZE,
        }
    )
    payload = _http_get_json(f"{CHANNELS_URL}?{params}")
    out: dict[str, str] = {}
    for item in payload.get("items") or []:
        cid = str(item.get("id") or "").strip()
        country = str((item.get("snippet") or {}).get("country") or "").strip()
        if cid:
            out[cid] = country
    for cid in channel_ids:
        out.setdefault(cid, "")
    return out


def load_ref_labels(ref_path: str) -> pd.DataFrame:
    ref = _read_table(ref_path)
    need = {"video_id", "地区剧种", "剧种"}
    missing = need - set(ref.columns)
    if missing:
        print(f"[ERROR] 参考表缺列: {sorted(missing)}")
        sys.exit(1)
    slim = (
        ref[["video_id", "地区剧种", "剧种"]]
        .astype(str)
        .fillna("")
        .assign(video_id=lambda d: d["video_id"].str.strip())
    )
    slim = slim[slim["video_id"] != ""].drop_duplicates(subset=["video_id"], keep="last")
    return slim


def classify_region_genre(title: str, channel: str) -> tuple[str, str, str]:
    """返回 (地区剧种, 剧种, source)。"""
    text = f"{title} {channel}"
    region = ""
    for label, pat in _REGION_RULES:
        if pat.search(text):
            region = label
            break
    genre = ""
    for label, pat in _GENRE_RULES:
        if pat.search(text):
            genre = label
            break
    if region or genre:
        return region, genre or "未分类", "title_rule"
    return "", "", ""


def llm_classify_batch(
    rows: list[dict[str, str]],
    *,
    workers: int = 8,
) -> dict[str, tuple[str, str]]:
    """可选：用 DashScope 文本模型补 地区剧种/剧种。返回 video_id → (地区剧种, 剧种)。"""
    api_key = (os.getenv("DASHSCOPE_API_KEY") or "").strip()
    if not api_key:
        print("[WARN] --llm 但未设置 DASHSCOPE_API_KEY，跳过")
        return {}

    import requests

    url = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
    region_opts = "、".join(REGION_LABELS)
    system = (
        "你是影视内容标注员。根据标题和频道，判断地区剧种与剧种。"
        f"地区剧种只能是：{region_opts}。"
        "剧种只能是参考集常见类：刑侦/犯罪、职场/商战、都市情感、短剧、现代/通用、爱情/电影、"
        "家庭伦理、婚姻家庭、校园/青春、喜剧/家庭喜剧、缉毒/禁毒、医疗、年代/改革、未分类、"
        "农村/乡土、现实主义/社会、女性成长、反腐/廉政、消防/救援、军旅/现代军事。"
        '只输出 JSON：{"地区剧种":"...","剧种":"..."}，不要解释。'
    )

    def one(row: dict[str, str]) -> tuple[str, str, str]:
        vid = row["video_id"]
        user = f"标题: {row.get('title','')[:120]}\n频道: {row.get('channel','')[:80]}"
        try:
            r = requests.post(
                url,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={
                    "model": "qwen-flash",
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    "temperature": 0,
                    "max_tokens": 80,
                },
                timeout=30,
            )
            if r.status_code != 200:
                return vid, "", ""
            text = r.json()["choices"][0]["message"]["content"].strip()
            m = re.search(r"\{.*\}", text, re.S)
            if not m:
                return vid, "", ""
            obj = json.loads(m.group(0))
            region = str(obj.get("地区剧种") or "").strip()
            genre = str(obj.get("剧种") or "").strip()
            if region not in REGION_LABELS:
                region = ""
            return vid, region, genre
        except Exception:
            return vid, "", ""

    out: dict[str, tuple[str, str]] = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(one, r) for r in rows]
        for fut in as_completed(futs):
            vid, region, genre = fut.result()
            if region or genre:
                out[vid] = (region, genre)
    return out


def infer_country_from_region(merged: pd.DataFrame) -> int:
    """空「频道国家」时，用地区剧种映射 ISO；写入 country_source=infer_region。返回新填行数。"""
    if "country_source" not in merged.columns:
        merged["country_source"] = ""
    empty = merged["频道国家"].fillna("").astype(str).str.strip() == ""
    region = merged["地区剧种"].fillna("").astype(str).str.strip()
    mapped = region.map(REGION_TO_COUNTRY).fillna("")
    fill = empty & (mapped != "")
    n = int(fill.sum())
    if n:
        merged.loc[fill, "频道国家"] = mapped[fill]
        merged.loc[fill, "country_source"] = "infer_region"
    # API 已写入且尚未标注来源
    if "yt_meta_status" in merged.columns:
        from_api = (
            (~empty)
            & (merged["yt_meta_status"].fillna("").astype(str).str.strip() == "ok")
            & (merged["country_source"].fillna("").astype(str).str.strip() == "")
        )
        merged.loc[from_api, "country_source"] = "yt_api"
    return n


def enrich(
    target_path: str,
    ref_path: str,
    output_path: str,
    *,
    skip_yt: bool = False,
    infer_country: bool = False,
    api_key: str | None = None,
    resume: bool = False,
    use_llm: bool = False,
    llm_workers: int = 8,
    sleep: float = DEFAULT_SLEEP,
    limit: int = 0,
) -> dict:
    df = _read_table(target_path)
    if "video_id" not in df.columns:
        print("[ERROR] 目标表需要 video_id")
        sys.exit(1)
    if limit and limit > 0:
        df = df.head(limit).copy()
        print(f"[抽样] limit={limit}")

    for col in (
        "频道国家",
        "country_source",
        "yt_category_id",
        "yt_category_name",
        "yt_channel_id",
        "yt_default_audio_language",
        "yt_meta_status",
        "地区剧种",
        "剧种",
        "label_source",
    ):
        if col not in df.columns:
            df[col] = ""

    # 无 API key 且未显式要求拉 YT → 自动 skip
    key_present = bool((api_key or os.getenv("YOUTUBE_API_KEY", "") or "").strip())
    if not skip_yt and not key_present:
        print("[WARN] 未设置 YOUTUBE_API_KEY，自动 --skip-yt；可用 --infer-country 由地区剧种推断国家")
        skip_yt = True
        if not infer_country:
            infer_country = True
            print("[WARN] 已自动开启 --infer-country")

    # --- join 参考 ---
    ref = load_ref_labels(ref_path)
    print(f"[参考] unique video_id={len(ref):,}")
    base_cols = [c for c in df.columns if c not in ("地区剧种", "剧种", "label_source")]
    # 保留已有 label_source 以便 resume
    merged = df[base_cols].merge(ref, on="video_id", how="left", suffixes=("", "_ref"))
    # merge 后 地区剧种/剧种来自 ref
    for c in ("地区剧种", "剧种"):
        if c not in merged.columns:
            merged[c] = ""
        merged[c] = merged[c].fillna("").astype(str)
    if "label_source" not in merged.columns:
        merged["label_source"] = ""
    # 若 resume 且输出已存在，合并 YT 进度列
    if resume and os.path.exists(output_path):
        prev = _read_table(output_path)
        yt_cols = [
            "频道国家",
            "country_source",
            "yt_category_id",
            "yt_category_name",
            "yt_channel_id",
            "yt_default_audio_language",
            "yt_meta_status",
            "label_source",
            "地区剧种",
            "剧种",
        ]
        keep = ["video_id"] + [c for c in yt_cols if c in prev.columns]
        prev = prev[keep].drop_duplicates("video_id", keep="last")
        merged = merged.drop(columns=[c for c in yt_cols if c in merged.columns], errors="ignore")
        merged = merged.merge(prev, on="video_id", how="left")
        for c in yt_cols:
            if c in merged.columns:
                merged[c] = merged[c].fillna("")
        print(f"[续跑] 合并已有输出 {output_path}")

    hit = (merged["地区剧种"].astype(str).str.strip() != "") | (
        merged["剧种"].astype(str).str.strip() != ""
    )
    # 仅当来自本次 join（label_source 空且有标签）标 ref_join
    newly = hit & (merged["label_source"].astype(str).str.strip() == "")
    merged.loc[newly, "label_source"] = "ref_join"
    n_ref = int((merged["label_source"] == "ref_join").sum())
    print(f"[join] label_source=ref_join → {n_ref:,} / {len(merged):,} ({n_ref / max(len(merged), 1):.1%})")

    # --- YouTube ---
    if not skip_yt:
        key = resolve_api_key(api_key)
        # pending: 无终态 yt_meta_status
        status = merged["yt_meta_status"].fillna("").astype(str).str.strip()
        pending_idx = merged.index[~status.isin(("ok", "not_found"))]
        vids = (
            merged.loc[pending_idx, "video_id"]
            .astype(str)
            .str.strip()
            .replace("", pd.NA)
            .dropna()
            .unique()
            .tolist()
        )
        print(f"[YT] 待拉 videos={len(vids):,}")
        out_dir = os.path.dirname(os.path.abspath(output_path)) or "."
        os.makedirs(out_dir, exist_ok=True)
        prog = ThrottledProgress(
            out_dir,
            "enrich_film_meta",
            interval_sec=5.0,
            every_n=50,
            input=target_path,
            output=output_path,
            total=len(vids),
        )
        done = 0
        channel_cache: dict[str, str] = {}
        # preload countries already known
        for _, row in merged.iterrows():
            cid = str(row.get("yt_channel_id") or "").strip()
            ctry = str(row.get("频道国家") or "").strip()
            if cid and ctry:
                channel_cache[cid] = ctry

        for i in range(0, len(vids), BATCH_SIZE):
            batch = vids[i : i + BATCH_SIZE]
            try:
                meta = fetch_videos_meta(key, batch)
            except Exception as e:
                print(f"[WARN] videos.list 失败: {e}")
                meta = {
                    v: {
                        "yt_category_id": "",
                        "yt_category_name": "",
                        "yt_channel_id": "",
                        "yt_default_audio_language": "",
                        "yt_meta_status": "error",
                    }
                    for v in batch
                }
            # collect new channel ids
            need_ch = []
            for v in batch:
                cid = meta[v].get("yt_channel_id") or ""
                if cid and cid not in channel_cache:
                    need_ch.append(cid)
            for j in range(0, len(need_ch), BATCH_SIZE):
                ch_batch = need_ch[j : j + BATCH_SIZE]
                try:
                    channel_cache.update(fetch_channels_country(key, ch_batch))
                except Exception as e:
                    print(f"[WARN] channels.list 失败: {e}")
                    for cid in ch_batch:
                        channel_cache.setdefault(cid, "")

            for v in batch:
                m = meta[v]
                mask = merged["video_id"].astype(str) == v
                for col in (
                    "yt_category_id",
                    "yt_category_name",
                    "yt_channel_id",
                    "yt_default_audio_language",
                    "yt_meta_status",
                ):
                    merged.loc[mask, col] = m.get(col, "")
                cid = m.get("yt_channel_id") or ""
                merged.loc[mask, "频道国家"] = channel_cache.get(cid, "") if cid else ""

            done += len(batch)
            prog.tick(done=done)
            if sleep:
                time.sleep(sleep)
            # 周期性落盘断点
            if done % 2000 < BATCH_SIZE:
                tmp = output_path + ".ckpt.csv"
                merged.to_csv(tmp, index=False)

        prog.tick(force=True, done=done)
    else:
        print("[YT] --skip-yt，跳过 API")

    # --- 规则补全未命中细类（不覆盖 ref_join / llm）---
    need = merged["label_source"].fillna("").astype(str).str.strip().isin(("", "yt_only"))
    n_need = int(need.sum())
    print(f"[规则] 待补细类行≈{n_need:,}")
    rule_hit = 0
    titles = merged["title"] if "title" in merged.columns else pd.Series("", index=merged.index)
    channels = merged["channel"] if "channel" in merged.columns else pd.Series("", index=merged.index)
    for idx in merged.index[need]:
        region, genre, src = classify_region_genre(str(titles.at[idx]), str(channels.at[idx]))
        if not src:
            continue
        if not str(merged.at[idx, "地区剧种"] or "").strip() and region:
            merged.at[idx, "地区剧种"] = region
        if not str(merged.at[idx, "剧种"] or "").strip() and genre:
            merged.at[idx, "剧种"] = genre
        merged.at[idx, "label_source"] = src
        rule_hit += 1
    print(f"[规则] 新写入 title_rule ≈{rule_hit:,}")

    # --- 可选 LLM ---
    if use_llm:
        still = merged["label_source"].fillna("").astype(str).str.strip().isin(("", "yt_only"))
        still &= merged["地区剧种"].fillna("").astype(str).str.strip() == ""
        rows = []
        for idx in merged.index[still]:
            rows.append(
                {
                    "video_id": str(merged.at[idx, "video_id"]),
                    "title": str(merged.at[idx, "title"] if "title" in merged.columns else ""),
                    "channel": str(merged.at[idx, "channel"] if "channel" in merged.columns else ""),
                    "_idx": idx,
                }
            )
        print(f"[LLM] 待补 {len(rows):,}")
        # map video_id → idx（可能重复，取首次）
        id2idx: dict[str, object] = {}
        for r in rows:
            id2idx.setdefault(r["video_id"], r["_idx"])
        llm_out = llm_classify_batch(
            [{"video_id": r["video_id"], "title": r["title"], "channel": r["channel"]} for r in rows],
            workers=llm_workers,
        )
        n_llm = 0
        for vid, (region, genre) in llm_out.items():
            idx = id2idx.get(vid)
            if idx is None:
                continue
            if region:
                merged.at[idx, "地区剧种"] = region
            if genre:
                merged.at[idx, "剧种"] = genre
            merged.at[idx, "label_source"] = "llm"
            n_llm += 1
        print(f"[LLM] 写入 {n_llm:,}")

    # 仅有 YT、无细类
    empty_label = merged["label_source"].fillna("").astype(str).str.strip() == ""
    merged.loc[empty_label, "label_source"] = "yt_only"

    # --- 无 API：由地区剧种推断频道国家 ---
    if infer_country:
        n_inf = infer_country_from_region(merged)
        print(f"[推断] 地区剧种→频道国家 新填 {n_inf:,}（country_source=infer_region）")

    # 列顺序：原列 + 新列靠后
    new_cols = [
        "频道国家",
        "country_source",
        "yt_category_id",
        "yt_category_name",
        "yt_channel_id",
        "yt_default_audio_language",
        "yt_meta_status",
        "地区剧种",
        "剧种",
        "label_source",
    ]
    orig = [c for c in _read_table(target_path).columns if c in merged.columns]
    if limit:
        orig = [c for c in df.columns if c in merged.columns and c not in new_cols]
    ordered = orig + [c for c in new_cols if c not in orig]
    # 去重保序
    seen = set()
    final_cols = []
    for c in ordered + list(merged.columns):
        if c not in seen and c in merged.columns:
            seen.add(c)
            final_cols.append(c)
    out_df = merged[final_cols]

    parent = os.path.dirname(os.path.abspath(output_path)) or "."
    os.makedirs(parent, exist_ok=True)
    tmp = output_path + ".tmp"
    out_df.to_csv(tmp, index=False)
    os.replace(tmp, output_path)
    ckpt = output_path + ".ckpt.csv"
    if os.path.exists(ckpt):
        try:
            os.unlink(ckpt)
        except OSError:
            pass

    # 报告
    def nonempty(col: str) -> int:
        return int((out_df[col].fillna("").astype(str).str.strip() != "").sum()) if col in out_df.columns else 0

    n = len(out_df)
    stats = {
        "rows": n,
        "频道国家_nonempty": nonempty("频道国家"),
        "频道国家_rate": round(nonempty("频道国家") / max(n, 1), 4),
        "country_source": out_df["country_source"].value_counts().to_dict()
        if "country_source" in out_df.columns
        else {},
        "yt_category_nonempty": nonempty("yt_category_id"),
        "地区剧种_nonempty": nonempty("地区剧种"),
        "剧种_nonempty": nonempty("剧种"),
        "label_source": out_df["label_source"].value_counts().to_dict() if "label_source" in out_df.columns else {},
        "地区剧种_dist": out_df["地区剧种"].value_counts().head(20).to_dict() if "地区剧种" in out_df.columns else {},
        "yt_meta_status": out_df["yt_meta_status"].value_counts().to_dict()
        if "yt_meta_status" in out_df.columns
        else {},
        "note": "无 YT API 时频道国家可为 infer_region（内容产地代理，非频道登记国）",
    }
    report_path = os.path.splitext(output_path)[0] + "_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"[完成] → {output_path}")
    print(f"[报告] → {report_path}")
    print(
        f"  频道国家非空 {stats['频道国家_nonempty']:,} ({stats['频道国家_rate']:.1%})  "
        f"category {stats['yt_category_nonempty']:,}  "
        f"地区剧种 {stats['地区剧种_nonempty']:,}  剧种 {stats['剧种_nonempty']:,}"
    )
    print(f"  label_source: {stats['label_source']}")
    if stats.get("country_source"):
        print(f"  country_source: {stats['country_source']}")

    mark_done(
        parent,
        "enrich_film_meta",
        input=target_path,
        output=output_path,
        **{k: v for k, v in stats.items() if k != "地区剧种_dist"},
    )
    write_run_log(
        "enrich_film_meta",
        target_path,
        parent,
        stats={k: v for k, v in stats.items() if not isinstance(v, dict)},
        command=f"enrich_film_meta.py {target_path} --ref {ref_path} -o {output_path}",
        category="film_tv",
    )
    return stats


def main() -> None:
    p = argparse.ArgumentParser(description="补频道国家 + 影视类型（粗/细）")
    p.add_argument("input", help="目标 CSV/parquet（如 ge720 表）")
    p.add_argument("--ref", required=True, help="参考标注 CSV（含 地区剧种/剧种）")
    p.add_argument("-o", "--output", required=True, help="输出旁路 CSV")
    p.add_argument("--skip-yt", action="store_true", help="不调 YouTube API，只 join+规则")
    p.add_argument(
        "--infer-country",
        action="store_true",
        help="无 API 时用地区剧种推断频道国家（ISO）；无 key 时默认自动开启",
    )
    p.add_argument("--api-key", default=None, help="YOUTUBE_API_KEY（默认读环境变量）")
    p.add_argument("--resume", action="store_true", help="合并已有输出续跑 YT")
    p.add_argument("--llm", action="store_true", help="规则仍空时用 DashScope 补细类")
    p.add_argument("-w", "--llm-workers", type=int, default=8)
    p.add_argument("--sleep", type=float, default=DEFAULT_SLEEP)
    p.add_argument("-n", "--limit", type=int, default=0, help="只处理前 N 行（调试）")
    args = p.parse_args()

    enrich(
        args.input,
        args.ref,
        args.output,
        skip_yt=args.skip_yt,
        infer_country=args.infer_country,
        api_key=args.api_key,
        resume=args.resume,
        use_llm=args.llm,
        llm_workers=args.llm_workers,
        sleep=args.sleep,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
