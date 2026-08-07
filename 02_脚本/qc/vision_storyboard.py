#!/home/jqs/miniconda3/envs/data_cleaning/bin/python
from __future__ import annotations
"""
qc_vision_welding.py — 电焊视频 Storyboard 视觉质检

流程：
  1. yt-dlp 提取视频 info（不下载视频），拿到 storyboard URL（含签名）
  2. 下载最高分辨率 storyboard 拼图（一张大图，含若干小帧拼在一起）
  3. 切割出均匀分布的帧（默认取 4 帧）
  4. 调通义千问视觉模型，判定画面是否含真实焊接实操

通过标准（窄）：
  画面中有真实焊接操作/焊工实操演示（弧焊、MIG、TIG、气焊等）

依赖：
  pip install yt-dlp pandas Pillow tqdm openai
  export DASHSCOPE_API_KEY=sk-...
  # 默认从 Chrome 读取 YouTube 登录 cookies（Chrome 需已登录 YouTube）
  # 可选：export YT_DLP_COOKIES_FROM_BROWSER=firefox
  # 可选：export YT_DLP_COOKIES_FILE=/path/to/cookies.txt  或 --cookies

用法:
  # 试跑 5 条
  python3 qc_vision_welding.py input.parquet -o run02/ --max-rows 5

  # 全量续跑（默认 sb2、流水线、sidecar checkpoint；sb2 召回低于 sb0）
  python3 qc_vision_welding.py input.parquet -o run02/ -w 2

  # 随机抽样 / 干跑 / benchmark
  python3 qc_vision_welding.py input.parquet --sample 200 -w 2
  python3 qc_vision_welding.py input.parquet --dry-run
  python3 qc_vision_welding.py input.parquet --benchmark 20

  # 高级项用环境变量（见脚本顶部配置）：
  #   QC_VISION_META_SLEEP  QC_VISION_FRAMES  QC_VISION_VERBOSE
  #   YT_DLP_COOKIES_FILE   QC_VISION_SIDECAR=0

  # 监控面板（另开终端）
  """

import sys, os, time, argparse, shutil, random, queue, threading, tempfile, signal, json, re, tomllib
import urllib.request
from contextlib import nullcontext
from dataclasses import dataclass, field
from io import BytesIO
from datetime import datetime, timezone
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.request import Request

_SCRIPT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_SCRIPT_DIR))

import pandas as pd
from PIL import Image

from core.progress import update as progress_update, mark_done as progress_mark_done
from core.qc_stats import QcStatsBoard, QcTfErrCounts
from core.welding_l0 import welding_l0_prefilter
from core.adaptive_api import (
    AdaptiveConcurrencyGate,
    DualResourceScheduler,
    is_transient_error,
)
from core.yt_dlp_auth import (
    YtDlpAuth,
    apply_yt_dlp_auth,
    prefetch_browser_cookies,
    resolve_yt_dlp_auth,
)
from core.sop import write_run_log

# ─────────────────────────────────────────────
# 配置
# ─────────────────────────────────────────────

DEFAULT_MODEL    = "qwen3-vl-flash"
DEFAULT_WORKERS  = int(os.getenv("QC_VISION_WORKERS", "2"))
DEFAULT_META_WORKERS = int(os.getenv("QC_VISION_META_WORKERS", "3"))
DEFAULT_YT_CONCURRENCY = int(os.getenv("QC_VISION_YT_CONCURRENCY", "2"))
DEFAULT_FRAMES   = int(os.getenv("QC_VISION_FRAMES", "4"))
MAX_RETRIES      = 3       # 视觉 API 单条重试
CHECKPOINT_EVERY = int(os.getenv("QC_VISION_CHECKPOINT_EVERY", "200"))
CHECKPOINT_MIN_SEC = float(os.getenv("QC_VISION_CHECKPOINT_MIN_SEC", "600"))  # 最久间隔落盘（秒）
PROGRESS_JSON_SEC = 5        # progress.json 最短写入间隔（秒）
PROGRESS_JSON_EVERY = 10       # 每 N 条强制刷新监控文件
VISION_PHASE = "qc_vision_sb"  # 写入 progress.json 的 stage 名
ERROR_RETRY_ROUNDS = int(os.getenv("QC_VISION_ERROR_RETRY_ROUNDS", "3"))
ERROR_RETRY_PAUSE_SEC = 15 # 重试轮次间暂停并刷新 cookies（秒）
ERROR_RETRY_ENABLED = os.getenv("QC_VISION_NO_ERROR_RETRY", "").lower() not in ("1", "true", "yes")
# 内置 yt-dlp 节流 / cookies 策略（参考 yt-dlp troubleshooting、#15911）
META_SLEEP_SEC = float(os.getenv("QC_VISION_META_SLEEP", "0.5"))
COOKIE_REFRESH_EVERY = int(os.getenv("QC_VISION_COOKIE_REFRESH_EVERY", "40"))
COOKIE_REFRESH_SEC = float(os.getenv("QC_VISION_COOKIE_REFRESH_SEC", str(10 * 60)))
STORYBOARD_ATTEMPTS = 3    # 单条 storyboard 失败后的即时重试（含刷新 cookies）
MAX_API_TOKENS = 8         # T/F 仅需极少 token
SHEET_DL_WORKERS = 3       # storyboard sheet 并行下载
USE_SIDECAR_CHECKPOINT = os.getenv("QC_VISION_SIDECAR", "1").lower() not in ("0", "false", "no")
PREFETCH_COOKIES = os.getenv("QC_VISION_NO_PREFETCH_COOKIES", "").lower() not in ("1", "true", "yes")
USE_PIPELINE = os.getenv("QC_VISION_NO_PIPELINE", "").lower() not in ("1", "true", "yes")
TRANSIENT_SB_ERRORS = frozenset({
    "empty_formats", "bot_challenge", "rate_limited",
    "yt_dlp_error", "storyboard_not_found",
})
QC_VISION_COLS = (
    "qc_vision_result", "qc_vision_model", "qc_vision_run_id", "qc_vision_error_reason",
)
# sb2 默认：下载更小更快；电焊召回低于 sb0（见 qc_smoke/sb_ab_report.csv）
SB_ALL = ("sb0", "sb1", "sb2", "sb3")
DEFAULT_SB_PREFER = "sb2"
SB_ONLY_DEFAULT = True     # 生产不回退其他档

# 默认安静模式；QC_VISION_VERBOSE=1 或 --verbose 打开逐条日志
_VERBOSE = os.getenv("QC_VISION_VERBOSE", "").lower() in ("1", "true", "yes")


SEC_PER_VIDEO_ESTIMATE = float(os.getenv("QC_VISION_SEC_PER_VIDEO", "5.0"))
FRAME_MAX_SIZE   = (320, 180)
IMAGE_QUALITY    = 82


def resolve_sb_prefer_order(prefer: str | None = None, sb_only: bool | None = None) -> list[str]:
    """将首选档位置于首位；sb_only 默认 True（生产仅 sb2）。"""
    prefer = (prefer or DEFAULT_SB_PREFER).lower()
    if prefer not in SB_ALL:
        raise ValueError(f"无效 storyboard 档位: {prefer!r}，可选 {SB_ALL}")
    only = SB_ONLY_DEFAULT if sb_only is None else sb_only
    if only:
        return [prefer]
    return [prefer] + [x for x in SB_ALL if x != prefer]


SB_PREFER_ORDER = resolve_sb_prefer_order(DEFAULT_SB_PREFER)

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124.0 Safari/537.36"

# ─────────────────────────────────────────────
# 提示词（可被 --category 覆盖）
# ─────────────────────────────────────────────

_CATEGORIES_DIR = Path(__file__).resolve().parent.parent / "categories"

SYSTEM_PROMPT = """\
你是严谨的焊接工艺视频视觉审核员。根据视频抽帧画面，判断该视频是否展示真实焊接操作或焊工实操演示。仅输出 T 或 F，禁止任何解释。"""

USER_PROMPT_TMPL = """\
以下是从视频中均匀抽取的 {n} 帧画面，请综合判断：

符合通过（T）:
- 画面中有真实焊接操作：电弧焊、MIG/MAG、TIG、气焊/氧乙炔焊、点焊、埋弧焊等
- 可见焊枪/焊条/焊机在工作，或有焊工穿戴 PPE 进行实操
- 教学/演示类焊接工艺视频（讲解焊姿、焊缝、参数均可）

不符合（F）:
- 无焊接场景：纯讲解无实操、PPT/动画、仅有设备外观无作业
- 非焊接内容：游戏、美妆、语言教学、新闻、音乐、综艺
- 仅有切割/打磨/抛光而无焊接；纯机械加工、管道安装无焊点作业
- 仅有文字标题卡、缩略图拼图、广告片无实操画面

严格按照要求输出，仅输出 T 或 F，禁止任何解释。"""

_VISION_CATEGORY_LABEL = "焊接"


def load_vision_qc_config(category: str) -> dict:
    """从 categories/<category>/rules/vision_sb.toml 加载视觉 QC 提示词。"""
    config_path = _CATEGORIES_DIR / category / "rules" / "vision_sb.toml"
    if not config_path.exists():
        print(f"[ERROR] 类别 '{category}' 的 vision_sb.toml 不存在: {config_path}")
        sys.exit(1)
    try:
        cfg = tomllib.loads(config_path.read_text("utf-8"))
        return {
            "category": cfg["meta"]["category"],
            "category_label": cfg["meta"]["category_label"],
            "system_prompt": cfg["prompts"]["system_prompt"],
            "user_prompt_tmpl": cfg["prompts"]["user_prompt_tmpl"],
            "pass_label": cfg["labels"]["pass_label"],
            "fail_label": cfg["labels"]["fail_label"],
        }
    except (KeyError, tomllib.TOMLDecodeError) as e:
        print(f"[ERROR] vision_sb.toml 解析失败: {config_path}  {e}")
        sys.exit(1)


def apply_vision_qc_config(cfg: dict):
    """将 TOML 配置加载到模块级提示词变量。"""
    global SYSTEM_PROMPT, USER_PROMPT_TMPL, _VISION_CATEGORY_LABEL
    SYSTEM_PROMPT = cfg["system_prompt"]
    USER_PROMPT_TMPL = cfg["user_prompt_tmpl"]
    _VISION_CATEGORY_LABEL = cfg["category_label"]


# ─────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────

def log(msg: str):
    if _VERBOSE:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def log_always(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def make_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def fmt_duration(sec: float) -> str:
    """将秒数格式化为可读时长。"""
    sec = max(0, int(sec + 0.5))
    if sec < 60:
        return f"{sec}s"
    m, s = divmod(sec, 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m"


@dataclass
class RunConfig:
    """运行时配置（贯穿 stage1/2）。"""
    meta_sleep_sec: float = META_SLEEP_SEC
    use_sidecar: bool = False
    quiet: bool = False
    sb_cache_dir: str | None = None


@dataclass
class RunContext:
    """供 SIGINT 落盘与续跑恢复。"""
    df: pd.DataFrame | None = None
    input_path: str = ""
    run_cfg: RunConfig | None = None


_RUN_CTX = RunContext()
_shutdown_requested = False
_API_GATE: AdaptiveConcurrencyGate | None = None
_SCHEDULER: DualResourceScheduler | None = None


@dataclass
class PreparedItem:
    """流水线 stage1 输出：预编码 JPEG base64 或错误。"""
    frames_b64: list[str] | None = None
    error: str = ""
    t0: float = 0.0
    l0_reason: str = ""


@dataclass
class BenchmarkStats:
    """--benchmark 模式计时。"""
    meta_secs: list[float] = field(default_factory=list)
    api_secs: list[float] = field(default_factory=list)
    total_secs: list[float] = field(default_factory=list)


_BENCH = BenchmarkStats()
_ydl_local = threading.local()
_http_local = threading.local()


class ProgressTracker:
    """单轮进度与 ETA（tqdm 建议：miniters=1 应对网络抖动）。"""

    def __init__(
        self,
        total: int,
        estimate_sec_per_item: float,
        pass_label: str = "",
        stats: QcStatsBoard | None = None,
    ):
        self.total = total
        self.completed = 0
        self.stats = stats
        self.estimate_sec = estimate_sec_per_item
        self.pass_label = pass_label
        self.t0 = time.perf_counter()
        self._last_log = 0.0

    @property
    def n_t(self) -> int:
        return self.stats.pass_counts.n_t if self.stats else 0

    @property
    def n_f(self) -> int:
        return self.stats.pass_counts.n_f if self.stats else 0

    @property
    def n_err(self) -> int:
        return self.stats.pass_counts.n_err if self.stats else 0

    def mark_done(self, label: str = "", df: pd.DataFrame | None = None) -> None:
        self.completed += 1
        if self.stats is not None:
            if df is not None:
                self.stats.record_and_sync(label, df)
            else:
                self.stats.record_pass(label)
        elif label == "T":
            self._n_t = getattr(self, "_n_t", 0) + 1
        elif label == "F":
            self._n_f = getattr(self, "_n_f", 0) + 1
        elif label == "ERROR":
            self._n_err = getattr(self, "_n_err", 0) + 1

    def elapsed(self) -> float:
        return time.perf_counter() - self.t0

    def avg_sec(self) -> float:
        if self.completed == 0:
            return self.estimate_sec
        return self.elapsed() / self.completed

    def eta_sec(self) -> float:
        remaining = max(0, self.total - self.completed)
        return remaining * self.avg_sec()

    def pct(self) -> float:
        return self.completed / max(self.total, 1) * 100

    def pass_rate_pct(self) -> float:
        ok = self.n_t + self.n_f
        return ok / max(self.completed, 1) * 100

    def postfix(self) -> dict:
        if self.stats is not None:
            return self.stats.tqdm_postfix(self.avg_sec())
        return {
            "T": getattr(self, "_n_t", 0),
            "F": getattr(self, "_n_f", 0),
            "E": getattr(self, "_n_err", 0),
            "s": f"{self.avg_sec():.1f}",
        }

    def status_line(self) -> str:
        phase = f"[{self.pass_label}] " if self.pass_label else ""
        stats_part = (
            self.stats.status_suffix()
            if self.stats is not None
            else f"T={self.n_t} F={self.n_f} err={self.n_err}"
        )
        return (
            f"{phase}{self.completed:,}/{self.total:,} ({self.pct():.1f}%) | "
            f"{stats_part} | "
            f"已用 {fmt_duration(self.elapsed())} | "
            f"剩余 {fmt_duration(self.eta_sec())} | "
            f"{self.avg_sec():.1f}s/条"
        )

    def maybe_log(self, interval_sec: float = 60.0) -> None:
        """无 tqdm 时定期文本进度。"""
        now = time.perf_counter()
        if self.completed == 1 or now - self._last_log >= interval_sec:
            log_progress(self)
            self._last_log = now


class VisionRunMonitor:
    """写 progress.json 供查看进度。"""

    def __init__(
        self,
        output_dir: str,
        run_id: str,
        model: str,
        sb_prefer: str,
        total_rows: int,
    ):
        self.output_dir = output_dir
        self.run_id = run_id
        self.model = model
        self.sb_prefer = sb_prefer
        self.total_rows = total_rows
        self.pass_label = ""
        self.pass_done = 0
        self.pass_total = 0
        self.t0 = time.perf_counter()
        self._lock = threading.Lock()
        self._last_flush = 0.0
        self._since_flush = 0

    def set_pass(self, label: str, total: int) -> None:
        with self._lock:
            self.pass_label = label
            self.pass_done = 0
            self.pass_total = total
            self._flush(force=True)

    def tick(self, tracker: ProgressTracker, df: pd.DataFrame) -> None:
        with self._lock:
            self.pass_done = tracker.completed
            self._since_flush += 1
            self._flush(df=df)

    def finish_pass(self, df: pd.DataFrame) -> None:
        with self._lock:
            self._flush(df=df, force=True)

    def mark_complete(self, df: pd.DataFrame) -> None:
        snap = self._snapshot(df)
        progress_mark_done(
            self.output_dir,
            VISION_PHASE,
            run_id=self.run_id,
            model=self.model,
            sb_prefer=self.sb_prefer,
            elapsed_sec=round(time.perf_counter() - self.t0, 1),
            **snap,
        )

    def _snapshot(self, df: pd.DataFrame) -> dict:
        col = df["qc_vision_result"] if "qc_vision_result" in df.columns else pd.Series(dtype=str)
        n_t = int((col == "T").sum())
        n_f = int((col == "F").sum())
        n_err = int((col == "ERROR").sum())
        done = n_t + n_f + n_err
        pending = max(0, self.total_rows - done)
        return {
            "total": self.total_rows,
            "done": done,
            "pending": pending,
            "n_t": n_t,
            "n_f": n_f,
            "n_err": n_err,
            "overall_pct": round(done / max(self.total_rows, 1) * 100, 2),
            "t_pct": round(n_t / max(self.total_rows, 1) * 100, 2),
        }

    def _flush(self, df: pd.DataFrame | None = None, force: bool = False) -> None:
        now = time.time()
        if not force:
            if self._since_flush < PROGRESS_JSON_EVERY and now - self._last_flush < PROGRESS_JSON_SEC:
                return
        self._last_flush = now
        self._since_flush = 0
        extra = self._snapshot(df) if df is not None else {}
        pass_pct = round(self.pass_done / max(self.pass_total, 1) * 100, 1)
        progress_update(
            self.output_dir,
            VISION_PHASE,
            status="running",
            run_id=self.run_id,
            model=self.model,
            sb_prefer=self.sb_prefer,
            pass_label=self.pass_label,
            pass_done=self.pass_done,
            pass_total=self.pass_total,
            pass_pct=pass_pct,
            elapsed_sec=round(time.perf_counter() - self.t0, 1),
            **extra,
        )


def create_progress_bar(total: int, desc: str):
    """tqdm 配置：miniters=1 + maxinterval 保证慢请求时仍刷新（见 tqdm 文档）。"""
    from tqdm import tqdm
    return tqdm(
        total=total,
        desc=desc[:20],
        unit="条",
        miniters=1,
        mininterval=0.3,
        maxinterval=3.0,
        smoothing=0.05,
        dynamic_ncols=True,
        bar_format=(
            "{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} "
            "[{elapsed}<{remaining}] {postfix}"
        ),
    )


def _after_item(
    tracker: ProgressTracker,
    pbar,
    monitor: VisionRunMonitor | None,
    df: pd.DataFrame,
    label: str,
    completed: int,
    pending_total: int,
    input_path: str,
    last_ckpt: float,
    *,
    run_cfg: RunConfig | None = None,
    checkpoint_lock: threading.Lock | None = None,
) -> float:
    """统一更新 tqdm / checkpoint / 监控（checkpoint 优先，监控失败不丢数据）。"""
    tracker.mark_done(label, df=df)
    if pbar:
        pbar.update(1)
        pbar.set_postfix(tracker.postfix(), refresh=True)
    else:
        tracker.maybe_log()

    now = time.time()
    due_count = completed % CHECKPOINT_EVERY == 0
    due_time = (now - last_ckpt) >= CHECKPOINT_MIN_SEC
    due_final = completed == pending_total
    if due_count or due_final or due_time:
        def _do_ckpt():
            nonlocal last_ckpt
            flush_checkpoint_state(
                sync_main=False, df=df, input_path=input_path, run_cfg=run_cfg,
            )
            log(f"  checkpoint ✓ {completed:,}/{pending_total:,} → {input_path}")
            log(f"  {tracker.status_line()}")
            last_ckpt = now

        if checkpoint_lock:
            with checkpoint_lock:
                _do_ckpt()
        else:
            _do_ckpt()

    if monitor:
        try:
            monitor.tick(tracker, df)
        except Exception as e:
            log(f"  [WARN] progress.json 更新失败（不影响 checkpoint）: {e}")

    return last_ckpt


def log_progress(tracker: ProgressTracker, prefix: str = "进度") -> None:
    log(f"{prefix}: {tracker.status_line()}")


def prepare_auth_for_run(
    auth: YtDlpAuth,
    run_id: str,
    output_dir: str,
    prefetch: bool,
) -> tuple[YtDlpAuth, str | None, tuple[str, ...] | None]:
    """解析认证；浏览器模式且 prefetch 时导出 cookie 文件。返回 (auth, cache_path, browser_spec)。"""
    browser_spec = auth.cookies_from_browser
    if auth.cookies_file or not prefetch or not auth.cookies_from_browser:
        return auth, auth.cookies_file, browser_spec
    browser_tag = "_".join(auth.cookies_from_browser).replace(":", "_")
    cache_path = os.path.join(output_dir, f".qc_vision_cookies_{browser_tag}_{run_id}.txt")
    try:
        log(f"预取浏览器 cookies: {':'.join(auth.cookies_from_browser)} → {cache_path}")
        t0 = time.perf_counter()
        auth2 = prefetch_browser_cookies(auth, cache_path)
        log(f"  cookies 已缓存 ({time.perf_counter() - t0:.1f}s)，workers 可 >1")
        return auth2, cache_path, browser_spec
    except Exception as e:
        log(f"  cookies 预取失败: {e}，回退单线程浏览器模式")
        return auth, None, browser_spec


def detect_js_runtimes() -> dict | None:
    """自动检测 node/deno，应对 YouTube SABR（yt-dlp#15911）。"""
    if shutil.which("deno"):
        return {"deno": {}}
    if shutil.which("node"):
        return {"node": {}}
    return None


class AuthManager:
    """线程安全的 cookies 管理：定期从浏览器刷新缓存文件。"""

    def __init__(
        self,
        auth: YtDlpAuth,
        cache_path: str | None,
        browser_spec: tuple[str, ...] | None,
        refresh_every_n: int = COOKIE_REFRESH_EVERY,
        refresh_interval_sec: float = COOKIE_REFRESH_SEC,
        yt_gate: AdaptiveConcurrencyGate | None = None,
        yt_concurrency: int = DEFAULT_YT_CONCURRENCY,
    ):
        self._lock = threading.Lock()
        self._auth = auth
        self._cache_path = cache_path
        self._browser_spec = browser_spec
        self._refresh_every_n = refresh_every_n
        self._refresh_interval_sec = refresh_interval_sec
        self._request_count = 0
        self._last_refresh = time.perf_counter()
        self._yt_gate = yt_gate or AdaptiveConcurrencyGate(
            yt_concurrency, label="yt_meta",
        )

    def get_auth(self) -> YtDlpAuth:
        with self._lock:
            self._request_count += 1
            if self._maybe_refresh_locked(trigger="periodic"):
                pass
            return self._auth

    def yt_dlp_section(self):
        """控制 yt-dlp 并发（自适应 gate），避免过载 + 刷新 cookies 竞态。"""
        return self._yt_gate.slot()

    @property
    def yt_gate(self) -> AdaptiveConcurrencyGate:
        return self._yt_gate

    def refresh_now(self, reason: str = "") -> bool:
        with self._lock:
            return self._maybe_refresh_locked(trigger=reason or "manual", force=True)

    def _maybe_refresh_locked(self, trigger: str = "", force: bool = False) -> bool:
        if not self._browser_spec or not self._cache_path:
            return False
        due_time = (
            self._refresh_interval_sec > 0
            and time.perf_counter() - self._last_refresh >= self._refresh_interval_sec
        )
        due_count = (
            self._refresh_every_n > 0
            and self._request_count > 0
            and self._request_count % self._refresh_every_n == 0
        )
        if not force and not due_time and not due_count:
            return False
        try:
            log(f"刷新 cookies ({trigger}) → {self._cache_path}")
            self._auth = prefetch_browser_cookies(
                YtDlpAuth(cookies_from_browser=self._browser_spec),
                self._cache_path,
            )
            self._last_refresh = time.perf_counter()
            return True
        except Exception as e:
            log(f"  cookies 刷新失败: {e}")
            return False


def build_ydl_opts(
    auth: YtDlpAuth,
    js_runtimes: dict | None = None,
    meta_sleep_sec: float = META_SLEEP_SEC,
    player_clients: list[str] | None = None,
) -> dict:
    yt_args: dict = {"skip": ["hls", "dash"]}
    if player_clients:
        yt_args["player_client"] = list(player_clients)
    opts = {
        "quiet": True,
        "skip_download": True,
        "no_warnings": True,
        "ignore_no_formats_error": True,
        "extractor_args": {"youtube": yt_args},
    }
    if meta_sleep_sec > 0:
        opts["sleep_interval_requests"] = meta_sleep_sec
    if js_runtimes:
        opts["js_runtimes"] = js_runtimes
    apply_yt_dlp_auth(opts, auth)
    return opts


def storyboard_fragment_urls(fragments: list) -> list[str]:
    """从 yt-dlp storyboard fragments 提取 URL；缺字段时跳过，避免 KeyError。"""
    urls: list[str] = []
    for frag in fragments or []:
        if isinstance(frag, str) and frag.startswith("http"):
            urls.append(frag)
            continue
        if not isinstance(frag, dict):
            continue
        url = frag.get("url") or frag.get("URL") or ""
        if url:
            urls.append(url)
    return urls


def classify_storyboard_failure(err_msg: str, n_formats: int, n_storyboard: int) -> str:
    """将 yt-dlp / formats 状态映射为可区分的错误码。"""
    el = (err_msg or "").lower()
    if "sign in" in el or "not a bot" in el or "confirm you're" in el:
        return "bot_challenge"
    if "429" in el or "too many requests" in el or "rate limit" in el:
        return "rate_limited"
    if "403" in el or "forbidden" in el:
        return "bot_challenge"
    if n_formats == 0:
        return "empty_formats"
    if n_storyboard == 0:
        return "storyboard_not_found"
    return "yt_dlp_error"


def fetch_url(url: str, timeout: int = 15) -> bytes:
    """HTTP GET；thread-local opener 复用连接。"""
    if not getattr(_http_local, "opener", None):
        _http_local.opener = urllib.request.build_opener()
    req = Request(url, headers={"User-Agent": UA, "Referer": "https://www.youtube.com/"})
    with _http_local.opener.open(req, timeout=timeout) as resp:
        return resp.read()


def compute_sample_indices(total_frames: int, n_frames: int) -> list[int]:
    """均匀采样 storyboard 帧索引（可单测）。"""
    if total_frames <= n_frames:
        return list(range(total_frames))
    step = total_frames / n_frames
    return [int(i * step + step / 2) for i in range(n_frames)]


def _ydl_extract_info(url: str, ydl_opts: dict):
    """Thread-local 复用 YoutubeDL 实例。"""
    import yt_dlp
    cookiefile = ydl_opts.get("cookiefile")
    sleep_iv = ydl_opts.get("sleep_interval_requests")
    opts_key = (cookiefile, sleep_iv, tuple(sorted((ydl_opts.get("js_runtimes") or {}).keys())))
    if getattr(_ydl_local, "opts_key", None) != opts_key:
        _ydl_local.ydl = yt_dlp.YoutubeDL(ydl_opts)
        _ydl_local.opts_key = opts_key
    return _ydl_local.ydl.extract_info(url, download=False)


# ─────────────────────────────────────────────
# Storyboard 核心逻辑
# ─────────────────────────────────────────────

def get_storyboard_info(
    video_id: str,
    auth: YtDlpAuth | None = None,
    sb_prefer_order: list[str] | None = None,
    js_runtimes: dict | None = None,
    meta_sleep_sec: float = META_SLEEP_SEC,
    player_clients: list[str] | None = None,
) -> tuple[dict | None, str]:
    try:
        import yt_dlp
    except ImportError:
        raise RuntimeError("请先安装 yt-dlp：pip install yt-dlp")

    auth = auth or YtDlpAuth(cookies_from_browser=("chrome",))
    ydl_opts = build_ydl_opts(
        auth,
        js_runtimes=js_runtimes,
        meta_sleep_sec=meta_sleep_sec,
        player_clients=player_clients,
    )

    url = f"https://www.youtube.com/watch?v={video_id}"
    err_msg = ""
    try:
        info = _ydl_extract_info(url, ydl_opts)
    except Exception as e:
        err_msg = str(e)
        log(f"  yt-dlp 失败 [{video_id}]: {err_msg[:120]}")
        code = classify_storyboard_failure(err_msg, 0, 0)
        if "nsig" in err_msg.lower() or "challenge" in err_msg.lower():
            log("  提示: 可安装 deno/node 并配置 --js-runtime auto（见 yt-dlp#15911）")
        return None, code

    try:
        formats = info.get("formats") or []
        sb_formats = {}
        for f in formats:
            if not isinstance(f, dict) or f.get("format_note") != "storyboard":
                continue
            fid = f.get("format_id")
            if fid:
                sb_formats[fid] = f

        if not sb_formats:
            code = classify_storyboard_failure(err_msg, len(formats), 0)
            log(f"  [{video_id}] 无 storyboard ({code}, formats={len(formats)})")
            return None, code

        chosen = None
        prefer_order = sb_prefer_order or SB_PREFER_ORDER
        for pref in prefer_order:
            if pref in sb_formats:
                chosen = sb_formats[pref]
                break
        if chosen is None:
            chosen = list(sb_formats.values())[0]

        rows = int(chosen.get("rows") or 10)
        cols = int(chosen.get("columns") or 10)
        frame_w = int(chosen.get("width") or 160)
        frame_h = int(chosen.get("height") or 90)
        fragments = chosen.get("fragments") or []

        if not fragments:
            top_url = chosen.get("url") or ""
            if top_url:
                fragments = [{"url": top_url}]
            else:
                log(f"  [{video_id}] storyboard fragments 为空")
                return None, "storyboard_fragments_empty"

        sheet_urls = storyboard_fragment_urls(fragments)
        if not sheet_urls:
            log(f"  [{video_id}] storyboard fragments 无有效 url")
            return None, "storyboard_fragments_empty"

        total_frames = rows * cols * len(sheet_urls)
        fmt_id = chosen.get("format_id") or "sb?"

        log(
            f"  [{video_id}] storyboard={fmt_id} "
            f"{frame_w}x{frame_h}/帧 {cols}列x{rows}行 x{len(sheet_urls)}张 = {total_frames}帧"
        )

        return {
            "sheet_urls": sheet_urls,
            "cols": cols,
            "rows": rows,
            "frame_w": frame_w,
            "frame_h": frame_h,
            "total_frames": total_frames,
        }, ""
    except Exception as e:
        log(f"  [{video_id}] storyboard 解析失败: {type(e).__name__}: {e}")
        return None, f"storyboard_parse_error:{type(e).__name__}"



def crop_selected_frames_from_sheet(
    sheet_bytes: bytes,
    cols: int,
    rows: int,
    frame_w: int,
    frame_h: int,
    local_indices: list[int],
) -> dict[int, Image.Image]:
    """只解码并裁切需要的格，避免整张 sheet 全量 crop。"""
    if not local_indices:
        return {}
    sheet = Image.open(BytesIO(sheet_bytes)).convert("RGB")
    out: dict[int, Image.Image] = {}
    for local_idx in local_indices:
        if local_idx < 0:
            continue
        r, c = divmod(local_idx, cols)
        if r >= rows:
            continue
        x, y = c * frame_w, r * frame_h
        if x + frame_w > sheet.width or y + frame_h > sheet.height:
            continue
        out[local_idx] = sheet.crop((x, y, x + frame_w, y + frame_h))
    return out


def fetch_storyboard_frames(sb_info: dict, n_frames: int = DEFAULT_FRAMES) -> list[Image.Image]:
    sheet_urls   = sb_info["sheet_urls"]
    cols         = sb_info["cols"]
    rows         = sb_info["rows"]
    frame_w      = sb_info["frame_w"]
    frame_h      = sb_info["frame_h"]
    total_frames = sb_info["total_frames"]
    frames_per_sheet = cols * rows

    sample_indices = compute_sample_indices(total_frames, n_frames)
    needed_sheets = sorted(set(idx // frames_per_sheet for idx in sample_indices))

    all_frames: dict[int, Image.Image] = {}

    def _load_sheet(sheet_idx: int) -> tuple[int, dict[int, Image.Image]]:
        if sheet_idx >= len(sheet_urls):
            return sheet_idx, {}
        try:
            local_needed = sorted({
                idx - sheet_idx * frames_per_sheet
                for idx in sample_indices
                if idx // frames_per_sheet == sheet_idx
            })
            data = fetch_url(sheet_urls[sheet_idx])
            base = sheet_idx * frames_per_sheet
            picked = crop_selected_frames_from_sheet(
                data, cols, rows, frame_w, frame_h, local_needed,
            )
            return sheet_idx, {base + i: img for i, img in picked.items()}
        except Exception as e:
            log(f"    下载 sheet[{sheet_idx}] 失败: {e}")
            return sheet_idx, {}

    if len(needed_sheets) <= 1:
        for si in needed_sheets:
            _, chunk = _load_sheet(si)
            all_frames.update(chunk)
    else:
        with ThreadPoolExecutor(max_workers=min(SHEET_DL_WORKERS, len(needed_sheets))) as ex:
            for _, chunk in ex.map(_load_sheet, needed_sheets):
                all_frames.update(chunk)

    result = []
    for idx in sample_indices:
        if idx in all_frames:
            img = all_frames[idx].copy()
            img.thumbnail(FRAME_MAX_SIZE, Image.LANCZOS)
            result.append(img)

    return result


def frames_to_b64_list(frames: list[Image.Image]) -> list[str]:
    return [image_to_b64(img) for img in frames]


# ─────────────────────────────────────────────
# 模型调用
# ─────────────────────────────────────────────

def image_to_b64(img: Image.Image) -> str:
    import base64
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=IMAGE_QUALITY)
    return base64.b64encode(buf.getvalue()).decode()


def _yt_gate_for(auth_mgr: AuthManager | YtDlpAuth | None) -> AdaptiveConcurrencyGate | None:
    if isinstance(auth_mgr, AuthManager):
        return auth_mgr.yt_gate
    if _SCHEDULER is not None:
        return _SCHEDULER.yt_meta
    return None


def prepare_storyboard(
    video_id: str,
    auth_mgr: AuthManager | YtDlpAuth | None,
    n_frames: int = DEFAULT_FRAMES,
    sb_prefer_order: list[str] | None = None,
    js_runtimes: dict | None = None,
    run_cfg: RunConfig | None = None,
) -> PreparedItem:
    """Stage1：yt-dlp 元数据 + storyboard 切帧 + JPEG base64 预编码。"""
    run_cfg = run_cfg or RunConfig()
    meta_sleep = run_cfg.meta_sleep_sec
    cache_dir = run_cfg.sb_cache_dir
    last_err = "storyboard_not_found"
    t0 = time.perf_counter()
    yt_gate = _yt_gate_for(auth_mgr)

    def _note_yt_ok():
        if yt_gate:
            yt_gate.record_outcome(ok=True)

    def _note_yt_fail(err: str):
        if not yt_gate:
            return
        if is_transient_error(err):
            if err in ("rate_limited", "bot_challenge"):
                yt_gate.on_rate_limit(log)
            yt_gate.record_outcome(transient_error=True, log_fn=log)

    if cache_dir:
        cached = load_cached_frames(cache_dir, video_id, n_frames)
        if cached:
            meta_sec = time.perf_counter() - t0
            _BENCH.meta_secs.append(meta_sec)
            log(f"  [{video_id}] 磁盘缓存 {len(cached)} 帧，调模型...")
            _note_yt_ok()
            return PreparedItem(frames_b64=frames_to_b64_list(cached), t0=t0)

        # storyboard URL 缓存：命中时跳过 Player API，直接下载切帧
        sb_cached = load_cached_sb_info(cache_dir, video_id)
        if sb_cached:
            frames = fetch_storyboard_frames(sb_cached, n_frames=n_frames)
            if frames:
                save_cached_frames(cache_dir, video_id, n_frames, frames)
                meta_sec = time.perf_counter() - t0
                _BENCH.meta_secs.append(meta_sec)
                log(f"  [{video_id}] sb缓存 {(time.perf_counter()-t0)*1000:.0f}ms（跳过Player API）")
                _note_yt_ok()
                return PreparedItem(frames_b64=frames_to_b64_list(frames), t0=t0)
            # URL 过期导致下载失败 → 回退到 Player API
            log(f"  [{video_id}] sb缓存过期，回退 Player API...")

    for attempt in range(STORYBOARD_ATTEMPTS):
        if _shutdown_requested:
            return PreparedItem(error="shutdown_requested", t0=t0)

        if isinstance(auth_mgr, AuthManager):
            auth = auth_mgr.get_auth()
            lock_ctx = auth_mgr.yt_dlp_section()
        else:
            auth = auth_mgr
            lock_ctx = yt_gate.slot() if yt_gate else nullcontext()

        # empty_formats / bot 时换 android+web client 再试，减轻默认 client 空 formats
        alt_clients = None
        if attempt > 0 and last_err in ("empty_formats", "bot_challenge", "storyboard_not_found"):
            alt_clients = ["android", "web"]

        with lock_ctx:
            sb_info, err = get_storyboard_info(
                video_id, auth=auth, sb_prefer_order=sb_prefer_order,
                js_runtimes=js_runtimes, meta_sleep_sec=meta_sleep,
                player_clients=alt_clients,
            )

        if sb_info is None:
            last_err = err or "storyboard_not_found"
        else:
            frames = fetch_storyboard_frames(sb_info, n_frames=n_frames)
            if frames:
                if cache_dir:
                    try:
                        save_cached_sb_info(cache_dir, video_id, sb_info)
                        save_cached_frames(cache_dir, video_id, n_frames, frames)
                    except Exception as ex:
                        log(f"  [{video_id}] storyboard 缓存写入失败: {ex}")
                meta_sec = time.perf_counter() - t0
                _BENCH.meta_secs.append(meta_sec)
                log(f"  [{video_id}] 实际取得 {len(frames)} 帧，调模型...")
                _note_yt_ok()
                return PreparedItem(frames_b64=frames_to_b64_list(frames), t0=t0)
            last_err = "storyboard_download_failed"

        if last_err not in TRANSIENT_SB_ERRORS or attempt >= STORYBOARD_ATTEMPTS - 1:
            break

        if isinstance(auth_mgr, AuthManager):
            auth_mgr.refresh_now(f"{last_err}#{attempt + 1}")
        time.sleep(1.0 + attempt * 0.5)

    _note_yt_fail(last_err)
    return PreparedItem(error=last_err, t0=t0)


def call_vision_api(
    client,
    video_id: str,
    frames: list[Image.Image] | None = None,
    model: str = DEFAULT_MODEL,
    frames_b64: list[str] | None = None,
) -> tuple[dict | None, str]:
    """Stage2：多图送视觉模型，返回 T/F。"""
    t0 = time.perf_counter()
    if frames_b64 is None:
        if not frames:
            return None, "no_frames"
        frames_b64 = frames_to_b64_list(frames)

    content = []
    for b64 in frames_b64:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
        })
    content.append({"type": "text", "text": USER_PROMPT_TMPL.format(n=len(frames_b64))})

    messages = []
    if SYSTEM_PROMPT and SYSTEM_PROMPT.strip():
        messages.append({"role": "system", "content": SYSTEM_PROMPT.strip()})
    messages.append({"role": "user", "content": content})

    for attempt in range(MAX_RETRIES):
        gate = _API_GATE
        if gate:
            gate.acquire()
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.0,
                max_tokens=MAX_API_TOKENS,
            )
            raw = resp.choices[0].message.content.strip()
            api_sec = time.perf_counter() - t0
            _BENCH.api_secs.append(api_sec)
            label = parse_vision_label(raw)
            if label in ("T", "F", "U"):
                if gate:
                    gate.record_outcome(ok=True)
                return {"label": label, "overall": label == "T", "reason": raw[:20]}, ""
            if attempt < MAX_RETRIES - 1:
                time.sleep(random.uniform(0.05, 0.25))
                continue
            return None, f"invalid_response:{raw[:50]}"

        except Exception as e:
            err_str = str(e).lower()
            if "rate_limit" in err_str or "429" in err_str:
                if gate:
                    gate.on_rate_limit(log)
                    gate.record_outcome(transient_error=True, log_fn=log)
                # 靠 gate 降并发；仅极短 jitter，避免与 gate 双限流
                time.sleep(random.uniform(0.05, 0.25))
                if attempt < MAX_RETRIES - 1:
                    continue
                return None, f"api_error:{type(e).__name__}:{str(e)[:80]}"
            if gate and is_transient_error(str(e)):
                gate.record_outcome(transient_error=True, log_fn=log)
            return None, f"api_error:{type(e).__name__}:{str(e)[:80]}"
        finally:
            if gate:
                gate.release()

    return None, "max_retries_exceeded"


def call_vision(client, video_id: str, model: str,
                n_frames: int = DEFAULT_FRAMES,
                auth_mgr: AuthManager | YtDlpAuth | None = None,
                sb_prefer_order: list[str] | None = None,
                js_runtimes: dict | None = None,
                run_cfg: RunConfig | None = None) -> tuple[dict | None, str]:
    """串行单条：storyboard 准备 + API。"""
    t0 = time.perf_counter()
    prep = prepare_storyboard(
        video_id, auth_mgr, n_frames, sb_prefer_order, js_runtimes, run_cfg,
    )
    if prep.error:
        return None, prep.error
    result, err = call_vision_api(
        client, video_id, model=model, frames_b64=prep.frames_b64,
    )
    _BENCH.total_secs.append(time.perf_counter() - t0)
    return result, err


def _apply_qc_result(
    df, idx, result, error, model, run_id, l0_reason: str = "",
) -> tuple[str, bool]:
    """写入一行 QC 结果，返回 (overall_label, is_ok)。"""
    if l0_reason:
        df.at[idx, "qc_vision_result"] = "F"
        df.at[idx, "qc_vision_error_reason"] = l0_reason
        df.at[idx, "qc_vision_model"] = model
        df.at[idx, "qc_vision_run_id"] = run_id
        return "F", True
    if error:
        df.at[idx, "qc_vision_result"] = "ERROR"
        df.at[idx, "qc_vision_error_reason"] = error
        df.at[idx, "qc_vision_model"] = model
        df.at[idx, "qc_vision_run_id"] = run_id
        return "ERROR", False
    if result:
        overall = result.get("label") or ("T" if result.get("overall") else "F")
        df.at[idx, "qc_vision_result"] = overall
        df.at[idx, "qc_vision_error_reason"] = ""
        df.at[idx, "qc_vision_model"] = model
        df.at[idx, "qc_vision_run_id"] = run_id
        return overall, True
    df.at[idx, "qc_vision_result"] = "ERROR"
    df.at[idx, "qc_vision_error_reason"] = "unknown_error"
    df.at[idx, "qc_vision_model"] = model
    df.at[idx, "qc_vision_run_id"] = run_id
    return "ERROR", False


def run_qc_sequential(
    client, df, pending_idx, auth_mgr, model, n_frames, workers,
    input_path, run_id, tracker, pbar, sb_prefer_order=None,
    js_runtimes=None, monitor=None, run_cfg: RunConfig | None = None,
) -> tuple[int, int, int, int, int]:
    """原模式：每条独立完成 storyboard + API。"""
    run_cfg = run_cfg or RunConfig()
    completed = 0
    n_ok = n_err = n_t = n_f = 0
    last_ckpt = time.time()
    ckpt_lock = threading.Lock()
    n_total = len(pending_idx)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_map = {}
        for idx in pending_idx:
            if _shutdown_requested:
                break
            l0 = row_l0_reason(df, idx)
            if l0:
                label, ok = _apply_qc_result(
                    df, idx, None, "", model, run_id, l0_reason=l0,
                )
                if label == "T":
                    n_t += 1
                    n_ok += 1
                elif label == "F":
                    n_f += 1
                    n_ok += 1
                elif label == "U":
                    n_ok += 1
                else:
                    n_err += 1
                completed += 1
                try:
                    last_ckpt = _after_item(
                        tracker, pbar, monitor, df, label,
                        completed, n_total, input_path, last_ckpt,
                        run_cfg=run_cfg, checkpoint_lock=ckpt_lock,
                    )
                except Exception as ex:
                    log(f"  [ERROR] 条目处理后续失败: {ex}")
                continue
            future_map[
                executor.submit(
                    call_vision, client, str(df.at[idx, "video_id"]),
                    model, n_frames, auth_mgr, sb_prefer_order, js_runtimes, run_cfg,
                )
            ] = idx
        for future in as_completed(future_map):
            if _shutdown_requested:
                break
            idx = future_map[future]
            try:
                result, error = future.result()
            except Exception as ex:
                result, error = None, (
                    f"future_exception:{type(ex).__name__}:{str(ex)[:160]}"
                )

            label, ok = _apply_qc_result(df, idx, result, error, model, run_id)
            if label == "T":
                n_t += 1
                n_ok += 1
            elif label == "F":
                n_f += 1
                n_ok += 1
            elif label == "U":
                n_ok += 1
            else:
                n_err += 1

            completed += 1
            try:
                last_ckpt = _after_item(
                    tracker, pbar, monitor, df, label,
                    completed, n_total, input_path, last_ckpt,
                    run_cfg=run_cfg, checkpoint_lock=ckpt_lock,
                )
            except Exception as ex:
                log(f"  [ERROR] 条目处理后续失败: {ex}")

    return completed, n_ok, n_t, n_f, n_err


def run_qc_pipeline(
    client, df, pending_idx, auth_mgr, model, n_frames,
    meta_workers, api_workers, input_path, run_id, tracker, pbar,
    sb_prefer_order=None,
    js_runtimes=None, monitor=None, run_cfg: RunConfig | None = None,
) -> tuple[int, int, int, int, int]:
    """
    方案 B：流水线 — stage1(meta) 与 stage2(API) 重叠执行；有界 meta 队列。
    """
    run_cfg = run_cfg or RunConfig()
    n_total = len(pending_idx)
    frame_q: queue.Queue = queue.Queue(maxsize=max(meta_workers, api_workers) * 4)
    pending_q: queue.Queue = queue.Queue()
    for idx in pending_idx:
        pending_q.put(idx)

    counters = {"completed": 0, "n_t": 0, "n_f": 0, "n_err": 0, "n_ok": 0}
    lock = threading.Lock()
    ckpt_lock = threading.Lock()
    last_ckpt = time.time()

    def meta_worker():
        while not _shutdown_requested:
            try:
                idx = pending_q.get_nowait()
            except queue.Empty:
                break
            vid = str(df.at[idx, "video_id"])
            l0 = row_l0_reason(df, idx)
            if l0:
                frame_q.put((idx, vid, PreparedItem(l0_reason=l0, t0=time.perf_counter())))
                continue
            prep = prepare_storyboard(
                vid, auth_mgr, n_frames, sb_prefer_order, js_runtimes, run_cfg,
            )
            frame_q.put((idx, vid, prep))

    def meta_producer():
        with ThreadPoolExecutor(max_workers=meta_workers) as meta_ex:
            futs = [meta_ex.submit(meta_worker) for _ in range(meta_workers)]
            for fut in as_completed(futs):
                fut.result()
        for _ in range(api_workers):
            frame_q.put(None)

    def api_worker():
        nonlocal last_ckpt
        while True:
            if _shutdown_requested:
                break
            item = frame_q.get()
            if item is None:
                break
            idx, vid, prep = item
            if prep.l0_reason:
                result, error = None, ""
                l0_reason = prep.l0_reason
            elif prep.error:
                result, error = None, prep.error
                l0_reason = ""
            else:
                try:
                    result, error = call_vision_api(
                        client, vid, model=model, frames_b64=prep.frames_b64,
                    )
                except Exception as ex:
                    result, error = None, (
                        f"future_exception:{type(ex).__name__}:{str(ex)[:160]}"
                    )
                l0_reason = ""

            with lock:
                label, ok = _apply_qc_result(
                    df, idx, result, error, model, run_id, l0_reason=l0_reason,
                )
                if label == "T":
                    counters["n_t"] += 1
                    counters["n_ok"] += 1
                elif label == "F":
                    counters["n_f"] += 1
                    counters["n_ok"] += 1
                elif label == "U":
                    counters["n_ok"] += 1
                else:
                    counters["n_err"] += 1
                counters["completed"] += 1
                completed = counters["completed"]
                if prep.t0:
                    _BENCH.total_secs.append(time.perf_counter() - prep.t0)

            try:
                last_ckpt = _after_item(
                    tracker, pbar, monitor, df, label,
                    completed, n_total, input_path, last_ckpt,
                    run_cfg=run_cfg, checkpoint_lock=ckpt_lock,
                )
            except Exception as ex:
                log(f"  [ERROR] 条目处理后续失败 [{vid}]: {ex}")

    producer = threading.Thread(target=meta_producer, daemon=True)
    producer.start()

    with ThreadPoolExecutor(max_workers=api_workers) as api_ex:
        api_futs = [api_ex.submit(api_worker) for _ in range(api_workers)]
        for f in api_futs:
            f.result()

    producer.join()

    return (
        counters["completed"], counters["n_ok"],
        counters["n_t"], counters["n_f"], counters["n_err"],
    )


# ─────────────────────────────────────────────
# IO 工具
# ─────────────────────────────────────────────

def sidecar_path_for(input_path: str) -> str:
    base, _ext = os.path.splitext(input_path)
    return f"{base}.qc_vision.parquet"


def sb_cache_root(output_dir: str) -> str:
    return os.path.join(output_dir, ".sb_cache")


def _sb_cache_meta_path(cache_dir: str, video_id: str) -> str:
    return os.path.join(cache_dir, video_id, "meta.json")


def _sb_cache_sbinfo_path(cache_dir: str, video_id: str) -> str:
    return os.path.join(cache_dir, video_id, "storyboard.json")


def _sb_cache_frame_path(cache_dir: str, video_id: str, i: int) -> str:
    return os.path.join(cache_dir, video_id, f"frame_{i:02d}.jpg")


def load_cached_frames(cache_dir: str, video_id: str, n_frames: int) -> list[Image.Image] | None:
    """从磁盘缓存加载已裁切的 storyboard 帧。"""
    meta_path = _sb_cache_meta_path(cache_dir, video_id)
    if not os.path.isfile(meta_path):
        return None
    try:
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        if int(meta.get("n_frames", 0)) != n_frames:
            return None
        frames = []
        for i in range(n_frames):
            fp = _sb_cache_frame_path(cache_dir, video_id, i)
            if not os.path.isfile(fp):
                return None
            frames.append(Image.open(fp).convert("RGB"))
        return frames
    except Exception:
        return None


def load_cached_sb_info(cache_dir: str, video_id: str) -> dict | None:
    """从磁盘缓存加载 storyboard 元数据（URL + 网格参数），跳过 Player API。"""
    path = _sb_cache_sbinfo_path(cache_dir, video_id)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def save_cached_sb_info(cache_dir: str, video_id: str, sb_info: dict) -> None:
    """保存 storyboard 元数据到磁盘缓存。"""
    d = os.path.join(cache_dir, video_id)
    os.makedirs(d, exist_ok=True)
    with open(_sb_cache_sbinfo_path(cache_dir, video_id), "w", encoding="utf-8") as f:
        json.dump({
            "video_id": video_id,
            "sheet_urls": sb_info["sheet_urls"],
            "cols": sb_info["cols"],
            "rows": sb_info["rows"],
            "frame_w": sb_info["frame_w"],
            "frame_h": sb_info["frame_h"],
            "total_frames": sb_info["total_frames"],
        }, f)


def save_cached_frames(cache_dir: str, video_id: str, n_frames: int, frames: list[Image.Image]) -> None:
    """保存裁切后的帧到磁盘缓存。"""
    d = os.path.join(cache_dir, video_id)
    os.makedirs(d, exist_ok=True)
    for i, img in enumerate(frames):
        img.save(_sb_cache_frame_path(cache_dir, video_id, i), format="JPEG", quality=IMAGE_QUALITY)
    with open(_sb_cache_meta_path(cache_dir, video_id), "w", encoding="utf-8") as f:
        json.dump({"n_frames": n_frames, "video_id": video_id}, f)


def row_l0_reason(df: pd.DataFrame, idx) -> str | None:
    """从 DataFrame 行提取 L0 预判。

    仅在没有指定 --category 时生效（即默认焊接模式）。
    指定 --category 后跳过 L0，由视觉模型自行判断。
    """
    if _VISION_CATEGORY_LABEL != "焊接":
        return None
    row = df.loc[idx]
    return welding_l0_prefilter(
        title=str(row.get("title", "")),
        channel=str(row.get("channel", "")),
        duration_str=str(row.get("duration_seconds", row.get("duration", ""))),
    )


def parse_vision_label(raw: str) -> str | None:
    """解析模型输出为 T/F/U；无法解析返回 None。"""
    text = (raw or "").strip().upper()
    if re.fullmatch(r"[TFU]", text):
        return text
    found = set(re.findall(r"[TFU]", text))
    if len(found) == 1:
        return found.pop()
    return None


def flush_checkpoint_state(
    sync_main: bool = False,
    df: pd.DataFrame | None = None,
    input_path: str | None = None,
    run_cfg: RunConfig | None = None,
) -> None:
    """sidecar + 可选 merge 写回主 parquet（SIGINT / 统一 checkpoint）。"""
    work_df = df if df is not None else _RUN_CTX.df
    path = input_path or _RUN_CTX.input_path
    cfg = run_cfg if run_cfg is not None else (_RUN_CTX.run_cfg or RunConfig())
    if work_df is None or not path:
        return
    if cfg.use_sidecar:
        write_qc_sidecar(work_df, path)
        if sync_main:
            merge_qc_sidecar(work_df, path)
            atomic_write(work_df, path)
    else:
        atomic_write(work_df, path)


def restore_from_sidecar_on_startup(df: pd.DataFrame, input_path: str) -> pd.DataFrame:
    """启动时合并 sidecar，避免崩溃后续跑丢进度。"""
    path = sidecar_path_for(input_path)
    if not os.path.isfile(path):
        return df
    before_done = int((df["qc_vision_result"].isin(["T", "F", "ERROR"])).sum())
    df = merge_qc_sidecar(df, input_path)
    after_done = int((df["qc_vision_result"].isin(["T", "F", "ERROR"])).sum())
    if after_done > before_done:
        log_always(f"已从 sidecar 恢复 QC 列（+{after_done - before_done:,} 条已QC）")
        atomic_write(df, input_path)
    return df


def _signal_handler(signum, _frame) -> None:
    global _shutdown_requested
    _shutdown_requested = True
    log_always(f"收到信号 {signum}，正在 checkpoint 落盘...")
    try:
        flush_checkpoint_state(sync_main=True)
        log_always("checkpoint 落盘完成，退出")
    except Exception as e:
        log_always(f"checkpoint 落盘失败: {e}")
    raise SystemExit(128 + signum if signum < 128 else 1)


def install_signal_handlers() -> None:
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)


def write_qc_sidecar(df: pd.DataFrame, input_path: str) -> None:
    """仅写 QC 列到 sidecar，加速 checkpoint。"""
    path = sidecar_path_for(input_path)
    cols = ["video_id", *QC_VISION_COLS]
    out = df[cols].copy()
    tmp = path + ".tmp"
    out.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def merge_qc_sidecar(df: pd.DataFrame, input_path: str) -> pd.DataFrame:
    """将 sidecar QC 列合并回主表（按 video_id）。"""
    path = sidecar_path_for(input_path)
    if not os.path.exists(path):
        return df
    side = pd.read_parquet(path)
    if "video_id" not in side.columns:
        return df
    side = side.drop_duplicates("video_id", keep="last").set_index("video_id")
    for col in QC_VISION_COLS:
        if col in side.columns:
            mapped = df["video_id"].map(side[col])
            df[col] = mapped.where(mapped.notna(), df[col])
    return df


def atomic_write(df: pd.DataFrame, target_path: str):
    """原子写入；失败时抛出异常并保留 .tmp 便于排查。"""
    tmp = target_path + ".tmp"
    ext = os.path.splitext(target_path)[1].lower()
    try:
        if ext == ".parquet":
            df.to_parquet(tmp, index=False)
        else:
            df.to_csv(tmp, index=False, encoding="utf-8-sig")
        os.replace(tmp, target_path)
    except Exception as e:
        log(f"  [ERROR] 写入失败 {target_path}: {e}")
        if os.path.exists(tmp):
            log(f"  [ERROR] 临时文件保留: {tmp}")
        raise


def assert_writable(path: str) -> None:
    """启动时检查输入文件可写，避免跑几小时无法 checkpoint。"""
    abspath = os.path.abspath(path)
    if not os.path.exists(abspath):
        raise FileNotFoundError(abspath)
    if not os.access(abspath, os.W_OK):
        raise PermissionError(
            f"输入文件不可写: {abspath}\n"
            f"  请 chown/chmod 后重跑，例如: sudo chown -R $USER:$USER {os.path.dirname(abspath)}"
        )
    d = os.path.dirname(abspath) or "."
    probe = os.path.join(d, f".write_probe_{os.getpid()}")
    try:
        with open(probe, "w") as f:
            f.write("ok")
        os.remove(probe)
    except OSError as e:
        raise PermissionError(f"输出目录不可写: {d} ({e})") from e


def collect_error_indices(df: pd.DataFrame) -> list:
    """收集仍为 ERROR 的行索引。"""
    mask = df["qc_vision_result"] == "ERROR"
    return df[mask].index.tolist()


def collect_retryable_error_indices(df: pd.DataFrame) -> list:
    """仅收集瞬时 ERROR（永久错误跳过重试轮）。"""
    mask = df["qc_vision_result"] == "ERROR"
    out = []
    for idx in df[mask].index:
        reason = df.at[idx, "qc_vision_error_reason"] if "qc_vision_error_reason" in df.columns else ""
        if is_transient_error(reason):
            out.append(idx)
    return out


def log_error_summary(df: pd.DataFrame, output_dir: str | None = None, run_id: str = "") -> None:
    err_df = df.loc[df["qc_vision_result"] == "ERROR", "qc_vision_error_reason"]
    if len(err_df) == 0:
        return
    log("ERROR 原因 Top:")
    counts = err_df.value_counts().head(12)
    for reason, cnt in counts.items():
        log(f"  {reason}: {cnt:,}")

    if not output_dir:
        return
    try:
        os.makedirs(output_dir, exist_ok=True)
        stem = f"qc_vision_errors_{run_id}" if run_id else "qc_vision_errors"
        summary_path = os.path.join(output_dir, f"{stem}_summary.json")
        list_path = os.path.join(output_dir, f"{stem}.txt")
        payload = {
            "run_id": run_id,
            "n_error": int(len(err_df)),
            "reasons": {str(k): int(v) for k, v in counts.items()},
        }
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        err_rows = df.loc[df["qc_vision_result"] == "ERROR", ["video_id", "qc_vision_error_reason"]]
        err_rows.to_csv(list_path, index=False, encoding="utf-8-sig")
        log(f"ERROR 明细已写入: {list_path}")
        log(f"ERROR 汇总已写入: {summary_path}")
    except Exception as e:
        log(f"  [WARN] 写入 ERROR 汇总失败: {e}")


def execute_qc_pass(
    client,
    df: pd.DataFrame,
    pending_idx: list,
    *,
    pass_label: str,
    auth_mgr,
    model: str,
    n_frames: int,
    workers: int,
    meta_workers: int,
    use_pipeline: bool,
    input_path: str,
    run_id: str,
    sb_prefer_order: list[str],
    js_runtimes: dict | None,
    monitor: VisionRunMonitor | None = None,
    run_cfg: RunConfig | None = None,
    stats_board: QcStatsBoard | None = None,
) -> tuple[int, int, int, int, int]:
    """执行一轮 QC（主流程或失败重试）。"""
    n = len(pending_idx)
    if n == 0:
        return 0, 0, 0, 0, 0

    log(f"── {pass_label}: {n:,} 条 ──")
    if monitor:
        monitor.set_pass(pass_label, n)

    tracker = ProgressTracker(
        n, SEC_PER_VIDEO_ESTIMATE, pass_label=pass_label, stats=stats_board,
    )
    if stats_board is not None:
        stats_board.reset_pass(pass_total=n)
    pbar = None
    try:
        pbar = create_progress_bar(n, pass_label)
    except ImportError:
        log("未安装 tqdm，将使用文本进度（pip install tqdm）")

    if use_pipeline:
        completed, n_ok, n_t, n_f, n_err = run_qc_pipeline(
            client, df, pending_idx, auth_mgr, model, n_frames,
            meta_workers, workers, input_path, run_id, tracker, pbar,
            sb_prefer_order, js_runtimes, monitor, run_cfg,
        )
    else:
        completed, n_ok, n_t, n_f, n_err = run_qc_sequential(
            client, df, pending_idx, auth_mgr, model, n_frames, workers,
            input_path, run_id, tracker, pbar, sb_prefer_order, js_runtimes,
            monitor, run_cfg,
        )

    if pbar:
        pbar.close()

    if monitor:
        monitor.finish_pass(df)

    if run_cfg and run_cfg.use_sidecar:
        df = merge_qc_sidecar(df, input_path)
    # 最终强制 sync 主文件，避免只写 sidecar 导致 progress 有结果、输入表无列
    flush_checkpoint_state(sync_main=True, df=df, input_path=input_path, run_cfg=run_cfg)
    log(f"  {pass_label} 完成: T={n_t:,} F={n_f:,} ERR={n_err:,} | {tracker.status_line()}")
    return completed, n_ok, n_t, n_f, n_err


def _percentile(vals: list[float], p: float) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    k = (len(s) - 1) * p / 100.0
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


def _print_benchmark_report() -> None:
    def row(name: str, vals: list[float]):
        if not vals:
            return
        log_always(
            f"  {name}: n={len(vals)} "
            f"P50={_percentile(vals, 50):.2f}s "
            f"P95={_percentile(vals, 95):.2f}s "
            f"avg={sum(vals)/len(vals):.2f}s"
        )

    log_always("── benchmark 耗时 ──")
    row("meta", _BENCH.meta_secs)
    row("api", _BENCH.api_secs)
    row("total", _BENCH.total_secs)


# ─────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="电焊视频 Storyboard 视觉质检",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "内置默认: sb2 单档 | meta_sleep=%ss | sidecar_checkpoint=%s | "
            "流水线=workers>1 | cookies 预取+刷新\n"
            "环境变量: QC_VISION_META_SLEEP QC_VISION_FRAMES QC_VISION_WORKERS "
            "QC_VISION_META_WORKERS QC_VISION_YT_CONCURRENCY "
            "QC_VISION_VERBOSE QC_VISION_SIDECAR=0 YT_DLP_COOKIES_FILE\n"
            "调参: 先 --benchmark 30，再调 --yt-concurrency / -w；看 ERROR%% 勿盲加并发"
            % (META_SLEEP_SEC, USE_SIDECAR_CHECKPOINT)
        ),
    )
    parser.add_argument("input", help="输入 parquet/csv（需含 video_id 列）")
    parser.add_argument("-o", "--output-dir", default=None,
                        help="progress.json / cookies 缓存目录（默认与输入同目录）")
    parser.add_argument("-m", "--model", default=DEFAULT_MODEL,
                        help=f"模型名（默认 {DEFAULT_MODEL}）")
    parser.add_argument("-w", "--workers", type=int, default=DEFAULT_WORKERS,
                        help=f"API 并发（默认 {DEFAULT_WORKERS}）")
    parser.add_argument("--meta-workers", type=int, default=DEFAULT_META_WORKERS,
                        help=f"storyboard/meta 流水线并发（默认 {DEFAULT_META_WORKERS}）")
    parser.add_argument("--yt-concurrency", type=int, default=DEFAULT_YT_CONCURRENCY,
                        help=f"yt-dlp Player API 并发上限（默认 {DEFAULT_YT_CONCURRENCY}）")
    parser.add_argument("--cookies", default=None,
                        help="cookies 文件（Netscape .txt）；否则读浏览器/环境变量")
    parser.add_argument("--max-rows", type=int, default=0,
                        help="最多处理 N 行（0=全部，建议先试 5）")
    parser.add_argument("--sample", type=int, default=0,
                        help="从待处理行中随机抽样 N 条（0=不抽样）")
    parser.add_argument("--dry-run", action="store_true", help="只统计待处理量，不调 API")
    parser.add_argument("--force", action="store_true", help="清除已有 QC 结果重新跑")
    parser.add_argument("--verbose", action="store_true", help="打印逐条 storyboard/API 日志")
    parser.add_argument("--benchmark", type=int, default=0, metavar="N",
                        help="跑 N 条并输出 meta/api/total 耗时 P50/P95")
    parser.add_argument("-c", "--category", default=None,
                        help="类别名（加载 categories/<name>/rules/vision_sb.toml 替代内置提示词）")
    args = parser.parse_args()

    global _VERBOSE, _API_GATE, _SCHEDULER
    if args.verbose:
        _VERBOSE = True

    if args.category:
        cfg = load_vision_qc_config(args.category)
        apply_vision_qc_config(cfg)
        log_always(f"视觉 QC 类别: {cfg['category_label']}")

    n_frames = DEFAULT_FRAMES
    meta_sleep = META_SLEEP_SEC

    if not os.path.exists(args.input):
        print(f"[ERROR] 文件不存在: {args.input}")
        sys.exit(1)

    output_dir = args.output_dir or os.path.dirname(os.path.abspath(args.input))
    os.makedirs(output_dir, exist_ok=True)

    run_cfg = RunConfig(
        meta_sleep_sec=meta_sleep,
        use_sidecar=USE_SIDECAR_CHECKPOINT,
        quiet=not _VERBOSE,
        sb_cache_dir=sb_cache_root(output_dir),
    )
    install_signal_handlers()

    run_id = make_run_id()
    sb_prefer_order = resolve_sb_prefer_order()
    auth = resolve_yt_dlp_auth(args.cookies, os.getenv("YT_DLP_COOKIES_FROM_BROWSER", "chrome"))
    auth, cookie_cache, browser_spec = prepare_auth_for_run(
        auth, run_id, output_dir, prefetch=PREFETCH_COOKIES,
    )

    js_runtimes = detect_js_runtimes()

    if auth.cookies_file:
        log(f"使用 cookies 文件: {auth.cookies_file}")
    elif auth.cookies_from_browser:
        log(f"使用浏览器 cookies: {':'.join(auth.cookies_from_browser)}（未预取，workers 强制为 1）")
        if args.workers > 1:
            args.workers = 1

    use_pipeline = USE_PIPELINE and args.workers > 1
    meta_workers = max(1, args.meta_workers)
    api_workers = args.workers

    _SCHEDULER = DualResourceScheduler(
        yt_initial=max(1, args.yt_concurrency),
        api_initial=max(1, api_workers),
    )
    _API_GATE = _SCHEDULER.api

    auth_mgr: AuthManager | YtDlpAuth = auth
    if cookie_cache and browser_spec:
        auth_mgr = AuthManager(
            auth, cookie_cache, browser_spec,
            refresh_every_n=COOKIE_REFRESH_EVERY,
            refresh_interval_sec=float(COOKIE_REFRESH_SEC),
            yt_gate=_SCHEDULER.yt_meta,
        )

    log_always(
        f"storyboard: {DEFAULT_SB_PREFER}（仅本档）| "
        f"每视频 {n_frames} 帧 | api_workers={api_workers} | "
        f"meta_workers={meta_workers} | yt_concurrency={args.yt_concurrency}"
    )
    if js_runtimes:
        log_always(f"yt-dlp JS runtime: {list(js_runtimes.keys())}")
    log_always(
        f"yt-dlp 请求间隔 {meta_sleep}s | "
        f"cookies 每 {COOKIE_REFRESH_EVERY} 次或 {int(COOKIE_REFRESH_SEC // 60)}min 刷新 | "
        f"checkpoint 每 {CHECKPOINT_EVERY} 条（最久 {int(CHECKPOINT_MIN_SEC)}s）"
        + (" | sidecar checkpoint" if run_cfg.use_sidecar else "")
    )
    if ERROR_RETRY_ENABLED:
        log(f"失败重试: 主流程结束后最多 {ERROR_RETRY_ROUNDS} 轮（间隔 {ERROR_RETRY_PAUSE_SEC}s）")

    if use_pipeline:
        log(f"流水线模式: meta_workers={meta_workers}, api_workers={api_workers}")
    else:
        log(f"串行池模式: workers={api_workers}")

    try:
        import yt_dlp
        log(f"yt-dlp 版本: {yt_dlp.version.__version__}")
    except Exception:
        pass

    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key and not args.dry_run:
        print("[ERROR] 未设置 DASHSCOPE_API_KEY")
        sys.exit(1)

    from openai import OpenAI
    client = None if args.dry_run else OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    t0 = time.perf_counter()

    log(f"读取: {args.input}")
    assert_writable(args.input)
    ext = os.path.splitext(args.input)[1].lower()
    df  = (pd.read_parquet(args.input) if ext == ".parquet"
           else pd.read_csv(args.input, dtype=str, low_memory=False)).fillna("").astype(str)
    log(f"  总行数: {len(df):,}")

    bak = args.input + f".bak_{run_id}"
    if not os.path.exists(bak):
        shutil.copy2(args.input, bak)
        log(f"已备份: {bak}")

    qc_cols = ["qc_vision_result", "qc_vision_model", "qc_vision_run_id",
               "qc_vision_error_reason"]
    for col in qc_cols:
        if col not in df.columns:
            df[col] = ""
    if args.force:
        for col in qc_cols:
            df[col] = ""

    if run_cfg.use_sidecar:
        df = restore_from_sidecar_on_startup(df, args.input)

    _RUN_CTX.df = df
    _RUN_CTX.input_path = args.input
    _RUN_CTX.run_cfg = run_cfg

    monitor = VisionRunMonitor(
        output_dir, run_id, args.model, DEFAULT_SB_PREFER, len(df),
    )
    stats_board = QcStatsBoard.from_dataframe(df, total_rows=len(df))
    snap = monitor._snapshot(df)
    g = stats_board.global_counts
    log_always(
        f"全表 QC: T={g.n_t:,} F={g.n_f:,} E={g.n_err:,} | "
        f"待处理 {g.pending:,} / {g.total:,}"
    )
    progress_update(
        output_dir, VISION_PHASE, status="starting",
        run_id=run_id, model=args.model, sb_prefer=DEFAULT_SB_PREFER, **snap,
    )
    log(f"进度: {output_dir}/progress.json（stage={VISION_PHASE}）")

    pending_mask = df["qc_vision_result"].isin(["", "ERROR"]) | df["qc_vision_result"].isna()
    pending_idx  = df[pending_mask].index.tolist()

    if args.sample > 0 and len(pending_idx) > args.sample:
        pending_idx = random.sample(pending_idx, args.sample)
        log_always(f"随机抽样: {len(pending_idx):,} 条")


    if args.benchmark > 0 and len(pending_idx) > args.benchmark:
        pending_idx = pending_idx[: args.benchmark]
        log_always(f"benchmark 模式: {len(pending_idx):,} 条")

    if args.max_rows > 0 and len(pending_idx) > args.max_rows:
        pending_idx = pending_idx[:args.max_rows]

    log(f"待处理: {len(pending_idx):,} / {len(df):,}")

    n_pending = len(pending_idx)
    if n_pending > 0:
        log(
            f"预估主流程耗时: ~{fmt_duration(n_pending * SEC_PER_VIDEO_ESTIMATE)} "
            f"({SEC_PER_VIDEO_ESTIMATE:.0f}s/条 × {n_pending:,} 条)"
        )

    if len(pending_idx) == 0:
        log("全部已质检，无需重跑。")
        return

    if args.dry_run:
        log(f"[dry-run] 将处理 {len(pending_idx)} 条，模型={args.model}，帧数={n_frames}")
        log(f"[dry-run] 预估耗时: ~{fmt_duration(n_pending * SEC_PER_VIDEO_ESTIMATE)}")
        if ERROR_RETRY_ENABLED:
            log(f"[dry-run] 主流程后将自动重试 ERROR（最多 {ERROR_RETRY_ROUNDS} 轮）")
        return

    # scheduler 已在读表前初始化；此处仅确认 API gate 指向同一对象
    if _SCHEDULER is None:
        _SCHEDULER = DualResourceScheduler(
            yt_initial=max(1, args.yt_concurrency),
            api_initial=max(1, api_workers),
        )
    _API_GATE = _SCHEDULER.api

    pass_kw = dict(
        auth_mgr=auth_mgr,
        model=args.model,
        n_frames=n_frames,
        workers=api_workers,
        meta_workers=meta_workers,
        use_pipeline=use_pipeline,
        input_path=args.input,
        run_id=run_id,
        sb_prefer_order=sb_prefer_order,
        js_runtimes=js_runtimes,
        monitor=monitor,
        run_cfg=run_cfg,
        stats_board=stats_board,
    )

    _BENCH.meta_secs.clear()
    _BENCH.api_secs.clear()
    _BENCH.total_secs.clear()

    total_completed = 0

    c, _, t, f, e = execute_qc_pass(
        client, df, pending_idx, pass_label="主流程", **pass_kw,
    )
    total_completed += c

    if ERROR_RETRY_ENABLED:
        for round_i in range(ERROR_RETRY_ROUNDS):
            if _SCHEDULER and not (
                _SCHEDULER.yt_ready_for_retry() or _SCHEDULER.api_ready_for_retry()
            ):
                log("yt/API 仍处于熔断态，跳过本轮及后续 ERROR 重试")
                break

            err_idx = collect_retryable_error_indices(df)
            permanent_left = len(collect_error_indices(df)) - len(err_idx)
            if not err_idx:
                if permanent_left:
                    log(f"无可重试瞬时 ERROR（剩余永久 ERROR {permanent_left:,}），跳过重试")
                else:
                    log("无 ERROR 行，跳过重试")
                break

            before_err = len(err_idx)
            if isinstance(auth_mgr, AuthManager):
                auth_mgr.refresh_now(f"retry_round_{round_i + 1}")
            log(
                f"准备失败重试 第 {round_i + 1}/{ERROR_RETRY_ROUNDS} 轮: "
                f"瞬时 ERROR {before_err:,} 条"
                + (f"（跳过永久 {permanent_left:,}）" if permanent_left else "")
                + f"，等待 {ERROR_RETRY_PAUSE_SEC}s..."
            )
            time.sleep(ERROR_RETRY_PAUSE_SEC)

            c, _, t, f, e = execute_qc_pass(
                client, df, err_idx,
                pass_label=f"失败重试 {round_i + 1}/{ERROR_RETRY_ROUNDS}",
                **pass_kw,
            )
            total_completed += c

            after_retryable = len(collect_retryable_error_indices(df))
            recovered = before_err - after_retryable
            log(f"  本轮恢复 {recovered:,} 条，剩余瞬时 ERROR {after_retryable:,}")
            if recovered == 0:
                log("  本轮无进展，停止后续重试")
                break

    log_error_summary(df, output_dir=output_dir, run_id=run_id)
    monitor.mark_complete(df)

    elapsed = time.perf_counter() - t0
    final_err = len(collect_error_indices(df))
    final_t = int((df["qc_vision_result"] == "T").sum())
    final_f = int((df["qc_vision_result"] == "F").sum())

    sidecar = sidecar_path_for(args.input) if run_cfg.use_sidecar else ""
    write_run_log(
        "qc_vision_sb", args.input, output_dir,
        stats={
            "run_id": run_id,
            "model": args.model,
            "T": final_t,
            "F": final_f,
            "ERROR": final_err,
            "completed": total_completed,
            "elapsed_sec": round(elapsed, 1),
            "sidecar": sidecar,
            "input": args.input,
        },
        command=f"vision_storyboard.py {args.input} -o {output_dir}",
        category=args.category,
    )

    if args.benchmark > 0 and _BENCH.total_secs:
        _print_benchmark_report()

    print()
    print("=" * 60)
    print("  电焊 Storyboard 视觉 QC 完成")
    print("=" * 60)
    print(f"  run_id:    {run_id}")
    print(f"  模型:      {args.model}")
    print(f"  每视频帧数: {n_frames}")
    print(f"  本轮处理:  {total_completed:,} 次任务")
    print(f"  最终:      T={final_t:,}  F={final_f:,}  ERR={final_err:,}")
    print(f"  耗时:      {fmt_duration(elapsed)}  ({elapsed/max(total_completed,1):.1f}s/次)")
    print(f"  回写至:    {args.input}")
    print("=" * 60)


if __name__ == "__main__":
    main()
