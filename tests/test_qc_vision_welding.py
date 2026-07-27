"""qc_vision_welding 单元测试（mock 外部依赖）。"""

import os
import sys
import tempfile
from pathlib import Path
from unittest import mock

import pandas as pd
import pytest

_SCRIPT_DIR = Path(__file__).resolve().parents[1] / "02_脚本"
sys.path.insert(0, str(_SCRIPT_DIR))

import qc.vision_storyboard as qw  # noqa: E402


class TestStoryboardFragmentUrls:
    def test_skips_missing_url(self):
        frags = [
            {"url": "https://a/1.jpg"},
            {"path": "nope"},
            {"URL": "https://a/2.jpg"},
            "https://a/3.jpg",
            None,
        ]
        assert qw.storyboard_fragment_urls(frags) == [
            "https://a/1.jpg",
            "https://a/2.jpg",
            "https://a/3.jpg",
        ]

    def test_empty(self):
        assert qw.storyboard_fragment_urls([]) == []
        assert qw.storyboard_fragment_urls(None) == []

    def test_storyboard_not_found(self):
        assert qw.classify_storyboard_failure("", 10, 0) == "storyboard_not_found"

    def test_bot_challenge(self):
        assert qw.classify_storyboard_failure("Please sign in to confirm you're not a bot", 0, 0) == "bot_challenge"

    def test_rate_limited(self):
        assert qw.classify_storyboard_failure("HTTP Error 429: Too Many Requests", 0, 0) == "rate_limited"


class TestComputeSampleIndices:
    def test_fewer_than_total(self):
        assert qw.compute_sample_indices(3, 6) == [0, 1, 2]

    def test_evenly_spaced(self):
        idx = qw.compute_sample_indices(100, 6)
        assert len(idx) == 6
        assert idx[0] == 8
        assert idx[-1] == 91

    def test_single_frame(self):
        assert qw.compute_sample_indices(50, 1) == [25]


class TestCropSelectedFrames:
    def test_only_requested_tiles(self):
        from PIL import Image
        cols, rows, fw, fh = 5, 2, 10, 10
        sheet = Image.new("RGB", (cols * fw, rows * fh), color=(0, 0, 0))
        buf = __import__("io").BytesIO()
        sheet.save(buf, format="JPEG")
        picked = qw.crop_selected_frames_from_sheet(
            buf.getvalue(), cols, rows, fw, fh, [0, 7],
        )
        assert set(picked) == {0, 7}
        assert all(img.size == (fw, fh) for img in picked.values())


class TestResolveSbPreferOrder:
    def test_default_sb2_only(self):
        assert qw.resolve_sb_prefer_order() == ["sb2"]

    def test_fallback_when_disabled(self):
        order = qw.resolve_sb_prefer_order("sb2", sb_only=False)
        assert order[0] == "sb2"
        assert len(order) == 4

    def test_sb_only_explicit(self):
        assert qw.resolve_sb_prefer_order("sb2", sb_only=True) == ["sb2"]

    def test_invalid_tier(self):
        with pytest.raises(ValueError, match="无效 storyboard"):
            qw.resolve_sb_prefer_order("sb9")


class TestAfterItem:
    def test_checkpoint_before_monitor(self):
        df = pd.DataFrame({
            "video_id": ["a"],
            "qc_vision_result": [""],
            "qc_vision_model": [""],
            "qc_vision_run_id": [""],
            "qc_vision_error_reason": [""],
        })
        tracker = qw.ProgressTracker(1, 5.0)
        calls: list[str] = []

        def fake_atomic_write(*_a, **_k):
            calls.append("checkpoint")

        class BoomMonitor:
            def tick(self, *_a, **_k):
                calls.append("monitor")
                raise RuntimeError("monitor down")

        with mock.patch.object(qw, "atomic_write", side_effect=fake_atomic_write), \
             mock.patch.object(qw, "CHECKPOINT_EVERY", 1):
            qw._after_item(
                tracker, None, BoomMonitor(), df, "T",
                1, 1, "/tmp/x.parquet", 0.0,
            )

        assert calls == ["checkpoint", "monitor"]

    def test_monitor_failure_does_not_block_checkpoint(self, tmp_path):
        df = pd.DataFrame({
            "video_id": ["a"],
            "qc_vision_result": [""],
            "qc_vision_model": [""],
            "qc_vision_run_id": [""],
            "qc_vision_error_reason": [""],
        })
        target = str(tmp_path / "out.parquet")
        df.to_parquet(target, index=False)
        tracker = qw.ProgressTracker(1, 5.0)

        class BoomMonitor:
            def tick(self, *_a, **_k):
                raise RuntimeError("boom")

        with mock.patch.object(qw, "CHECKPOINT_EVERY", 1):
            qw._after_item(
                tracker, None, BoomMonitor(), df, "T",
                1, 1, target, 0.0,
            )
        reread = pd.read_parquet(target)
        assert reread.at[0, "qc_vision_result"] == ""  # df mutated in memory only
        assert os.path.exists(target)

    def test_sidecar_checkpoint(self, tmp_path):
        df = pd.DataFrame({
            "video_id": ["v1"],
            "qc_vision_result": ["T"],
            "qc_vision_model": ["m"],
            "qc_vision_run_id": ["r"],
            "qc_vision_error_reason": [""],
        })
        target = str(tmp_path / "out.parquet")
        df.to_parquet(target, index=False)
        tracker = qw.ProgressTracker(1, 5.0)
        run_cfg = qw.RunConfig(use_sidecar=True)

        with mock.patch.object(qw, "CHECKPOINT_EVERY", 1):
            qw._after_item(
                tracker, None, None, df, "T",
                1, 1, target, 0.0, run_cfg=run_cfg,
            )
        side = tmp_path / "out.qc_vision.parquet"
        assert side.exists()
        side_df = pd.read_parquet(side)
        assert side_df.at[0, "qc_vision_result"] == "T"


class TestVisionRunMonitorSnapshot:
    def test_snapshot_uses_overall_pct_not_duplicate_pass_pct(self, tmp_path):
        with mock.patch.object(qw, "progress_update") as mock_update:
            mon = qw.VisionRunMonitor(str(tmp_path), "run1", "model", "sb2", 100)
            df = pd.DataFrame({"qc_vision_result": ["T", "F", "ERROR"] + [""] * 97})
            mon._flush(df=df, force=True)

        kwargs = mock_update.call_args.kwargs
        assert "overall_pct" in kwargs
        assert "pass_pct" in kwargs
        assert kwargs["overall_pct"] != kwargs.get("pass_pct") or kwargs["pass_done"] == 0


class TestAssertWritable:
    def test_missing_file(self):
        with pytest.raises(FileNotFoundError):
            qw.assert_writable("/nonexistent/path/file.parquet")

    def test_readonly_file(self, tmp_path):
        p = tmp_path / "ro.parquet"
        pd.DataFrame({"a": [1]}).to_parquet(p, index=False)
        os.chmod(p, 0o444)
        try:
            with pytest.raises(PermissionError, match="不可写"):
                qw.assert_writable(str(p))
        finally:
            os.chmod(p, 0o644)

    def test_writable_ok(self, tmp_path):
        p = tmp_path / "ok.parquet"
        pd.DataFrame({"a": [1]}).to_parquet(p, index=False)
        qw.assert_writable(str(p))


class TestAtomicWrite:
    def test_success_replaces_target(self, tmp_path):
        target = tmp_path / "out.parquet"
        df = pd.DataFrame({"x": [1]})
        qw.atomic_write(df, str(target))
        assert target.exists()
        assert not (tmp_path / "out.parquet.tmp").exists()

    def test_failure_keeps_tmp(self, tmp_path):
        target = str(tmp_path / "out.parquet")
        df = pd.DataFrame({"x": [1]})
        with mock.patch("os.replace", side_effect=OSError("replace failed")):
            with pytest.raises(OSError):
                qw.atomic_write(df, target)
        assert os.path.exists(target + ".tmp")


class TestMergeQcSidecar:
    def test_merge_updates_qc_columns(self, tmp_path):
        main = pd.DataFrame({
            "video_id": ["a", "b"],
            "qc_vision_result": ["", ""],
            "qc_vision_model": ["", ""],
            "qc_vision_run_id": ["", ""],
            "qc_vision_error_reason": ["", ""],
        })
        path = str(tmp_path / "data.parquet")
        main.to_parquet(path, index=False)
        side = pd.DataFrame({
            "video_id": ["a"],
            "qc_vision_result": ["T"],
            "qc_vision_model": ["m"],
            "qc_vision_run_id": ["r1"],
            "qc_vision_error_reason": [""],
        })
        side.to_parquet(qw.sidecar_path_for(path), index=False)
        merged = qw.merge_qc_sidecar(main, path)
        assert merged.loc[merged["video_id"] == "a", "qc_vision_result"].iloc[0] == "T"
        assert merged.loc[merged["video_id"] == "b", "qc_vision_result"].iloc[0] == ""


class TestRestoreFromSidecar:
    def test_startup_merge_skips_completed_on_resume(self, tmp_path):
        main = pd.DataFrame({
            "video_id": ["a", "b", "c"],
            "qc_vision_result": ["T", "", ""],
            "qc_vision_model": ["m", "", ""],
            "qc_vision_run_id": ["r0", "", ""],
            "qc_vision_error_reason": ["", "", ""],
        })
        path = str(tmp_path / "data.parquet")
        main.to_parquet(path, index=False)
        side = pd.DataFrame({
            "video_id": ["b", "c"],
            "qc_vision_result": ["F", "T"],
            "qc_vision_model": ["m", "m"],
            "qc_vision_run_id": ["r1", "r1"],
            "qc_vision_error_reason": ["l0:title:music", ""],
        })
        side.to_parquet(qw.sidecar_path_for(path), index=False)

        restored = qw.restore_from_sidecar_on_startup(main, path)
        assert restored.loc[restored["video_id"] == "b", "qc_vision_result"].iloc[0] == "F"
        assert restored.loc[restored["video_id"] == "c", "qc_vision_result"].iloc[0] == "T"

        pending = restored["qc_vision_result"].isin(["", "ERROR"]) | restored["qc_vision_result"].isna()
        assert pending.sum() == 0

        reread = pd.read_parquet(path)
        assert reread.loc[reread["video_id"] == "b", "qc_vision_result"].iloc[0] == "F"


class TestParseVisionLabel:
    def test_strict_t(self):
        assert qw.parse_vision_label("T") == "T"

    def test_strict_f(self):
        assert qw.parse_vision_label("f") == "F"

    def test_ambiguous(self):
        assert qw.parse_vision_label("TF") is None


class TestStoryboardDiskCache:
    def test_save_and_load_roundtrip(self, tmp_path):
        from PIL import Image

        cache_dir = str(tmp_path / ".sb_cache")
        vid = "abc123"
        frames = [Image.new("RGB", (10, 10), color=(i, 0, 0)) for i in range(4)]
        qw.save_cached_frames(cache_dir, vid, 4, frames)
        loaded = qw.load_cached_frames(cache_dir, vid, 4)
        assert loaded is not None
        assert len(loaded) == 4

    def test_wrong_frame_count_misses(self, tmp_path):
        from PIL import Image

        cache_dir = str(tmp_path / ".sb_cache")
        vid = "x"
        frames = [Image.new("RGB", (5, 5))]
        qw.save_cached_frames(cache_dir, vid, 1, frames)
        assert qw.load_cached_frames(cache_dir, vid, 4) is None


class TestSignalCheckpoint:
    def test_sigint_flushes_sidecar_and_main(self, tmp_path):
        df = pd.DataFrame({
            "video_id": ["v1"],
            "qc_vision_result": ["T"],
            "qc_vision_model": ["m"],
            "qc_vision_run_id": ["r"],
            "qc_vision_error_reason": [""],
        })
        path = str(tmp_path / "out.parquet")
        df.to_parquet(path, index=False)
        qw._RUN_CTX.df = df
        qw._RUN_CTX.input_path = path
        qw._RUN_CTX.run_cfg = qw.RunConfig(use_sidecar=True)

        with mock.patch.object(qw, "log_always"):
            qw.flush_checkpoint_state(sync_main=True)

        assert os.path.exists(qw.sidecar_path_for(path))
        reread = pd.read_parquet(path)
        assert reread.at[0, "qc_vision_result"] == "T"


class TestL0InPipeline:
    def test_l0_skips_api(self, tmp_path):
        target = tmp_path / "keep.parquet"
        df = pd.DataFrame({
            "video_id": ["v1"],
            "title": ["Best Pop Music Mix"],
            "channel": ["Music Channel"],
            "duration_seconds": ["600"],
            "qc_vision_result": [""],
            "qc_vision_model": [""],
            "qc_vision_run_id": [""],
            "qc_vision_error_reason": [""],
        })
        df.to_parquet(target, index=False)

        api_called = []

        def fake_api(*_a, **_k):
            api_called.append(1)
            return {"overall": True}, ""

        tracker = qw.ProgressTracker(1, 1.0)
        with mock.patch.object(qw, "prepare_storyboard") as mock_prep, \
             mock.patch.object(qw, "call_vision_api", side_effect=fake_api):
            qw.run_qc_pipeline(
                client=object(),
                df=df,
                pending_idx=[0],
                auth_mgr=None,
                model="test-model",
                n_frames=4,
                meta_workers=1,
                api_workers=1,
                input_path=str(target),
                run_id="test_run",
                tracker=tracker,
                pbar=None,
                sb_prefer_order=["sb2"],
            )
            mock_prep.assert_not_called()

        assert len(api_called) == 0
        assert df.at[0, "qc_vision_result"] == "F"
        assert df.at[0, "qc_vision_error_reason"].startswith("l0:")


class TestBuildYdlOpts:
    def test_meta_sleep_in_opts(self):
        auth = qw.YtDlpAuth(cookies_file="/tmp/c.txt")
        opts = qw.build_ydl_opts(auth, meta_sleep_sec=0.5)
        assert opts["sleep_interval_requests"] == 0.5

    def test_zero_meta_sleep_omits_key(self):
        auth = qw.YtDlpAuth(cookies_file="/tmp/c.txt")
        opts = qw.build_ydl_opts(auth, meta_sleep_sec=0)
        assert "sleep_interval_requests" not in opts


class TestPercentile:
    def test_p50(self):
        assert qw._percentile([1.0, 2.0, 3.0, 4.0], 50) == pytest.approx(2.5)


class TestPipelineCheckpointSmoke:
    """模拟 stage1/2，验证 checkpoint 落盘（无需 DASHSCOPE / yt-dlp）。"""

    def test_pipeline_writes_checkpoint(self, tmp_path):
        target = tmp_path / "keep.parquet"
        df = pd.DataFrame({
            "video_id": [f"v{i}" for i in range(5)],
            "qc_vision_result": [""] * 5,
            "qc_vision_model": [""] * 5,
            "qc_vision_run_id": [""] * 5,
            "qc_vision_error_reason": [""] * 5,
        })
        df.to_parquet(target, index=False)
        mtime_before = target.stat().st_mtime

        def fake_prepare(_vid, *_a, **_k):
            return qw.PreparedItem(
                frames_b64=["ZmFrZQ=="],
                t0=qw.time.perf_counter(),
            )

        def fake_api(_client, _vid, **_k):
            return {"overall": True, "reason": "T"}, ""

        tracker = qw.ProgressTracker(5, 1.0)
        with mock.patch.object(qw, "prepare_storyboard", side_effect=fake_prepare), \
             mock.patch.object(qw, "call_vision_api", side_effect=fake_api), \
             mock.patch.object(qw, "CHECKPOINT_EVERY", 2):
            qw.run_qc_pipeline(
                client=object(),
                df=df,
                pending_idx=list(range(5)),
                auth_mgr=None,
                model="test-model",
                n_frames=2,
                meta_workers=1,
                api_workers=2,
                input_path=str(target),
                run_id="test_run",
                tracker=tracker,
                pbar=None,
                sb_prefer_order=["sb2"],
            )

        assert target.stat().st_mtime >= mtime_before
        assert (df["qc_vision_result"] == "T").sum() == 5
        reread = pd.read_parquet(target)
        assert (reread["qc_vision_result"] == "T").sum() >= 4

