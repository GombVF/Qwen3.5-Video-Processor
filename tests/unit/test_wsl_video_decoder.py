from __future__ import annotations

import sys
import types

import pytest
from PIL import Image

from visual_service.settings import Settings
from visual_service.video import decoder
from visual_service.video.decoder import (
    SampledVideoFrames,
    VideoDecodeError,
    _is_transient_decode_error,
    decode_video_window,
    plan_video_timestamps,
)
from visual_service.video.profiles import overview_profile, semantic_profile


class _FakeTensor:
    shape = (4, 3, 64, 32)


def _install_fake_qwen_utils(monkeypatch, process_vision_info):
    package = types.ModuleType("qwen_vl_utils")
    package.process_vision_info = process_vision_info
    monkeypatch.setitem(sys.modules, "qwen_vl_utils", package)


def _decoded_payload():
    metadata = {
        "fps": 30.0,
        "total_num_frames": 60,
        "frames_indices": [0, 15, 30, 45],
        "video_backend": "torchvision",
    }
    return None, [(_FakeTensor(), metadata)], {"fps": [2.0]}


def _sampled_payload() -> SampledVideoFrames:
    return SampledVideoFrames(
        images=[Image.new("RGB", (32, 64)) for _ in range(4)],
        timestamps_ms=[250, 750, 1_250, 1_750],
        frame_indices=[8, 23, 38, 53],
        source_fps=30.0,
        effective_fps=2.0,
        total_frames=60,
        decode_ms=1,
    )


def test_video_metadata_is_forwarded_and_second_resize_is_disabled(monkeypatch, tmp_path):
    captured_messages = None

    def process_vision_info(*args, **kwargs):
        nonlocal captured_messages
        captured_messages = args[0]
        return _decoded_payload()

    _install_fake_qwen_utils(monkeypatch, process_vision_info)
    monkeypatch.setattr(decoder, "_sample_window_frames", lambda *args: _sampled_payload())
    decoded = decode_video_window(
        tmp_path / "video.mp4",
        0,
        2_000,
        semantic_profile(Settings()),
        retry_delay_ms=0,
    )

    assert decoded.video_kwargs["video_metadata"][0]["fps"] == 30.0
    assert decoded.video_kwargs["video_metadata"][0]["frames_indices"] == [8, 23, 38, 53]
    assert decoded.video_kwargs["video_metadata"][0]["video_backend"] == "pyav-uniform"
    assert decoded.video_kwargs["do_resize"] is False
    assert decoded.video_kwargs["cap_pixels_per_frame"] is False
    assert captured_messages[0]["content"][0]["video"][0].size == (32, 64)
    assert decoded.backend == "pyav-uniform"
    assert decoded.decode_strategy == "pyav-uniform"
    assert decoded.decode_attempts == 1


def test_transient_eagain_is_retried_once(monkeypatch, tmp_path):
    calls = 0

    def process_vision_info(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise BlockingIOError(11, "Resource temporarily unavailable")
        return _decoded_payload()

    _install_fake_qwen_utils(monkeypatch, process_vision_info)
    monkeypatch.setattr(decoder, "_sample_window_frames", lambda *args: _sampled_payload())
    decoded = decode_video_window(
        tmp_path / "video.mp4",
        0,
        2_000,
        semantic_profile(Settings()),
        max_attempts=2,
        retry_delay_ms=0,
    )

    assert calls == 2
    assert decoded.decode_attempts == 2


def test_non_transient_decode_failure_is_not_retried(monkeypatch, tmp_path):
    calls = 0

    def process_vision_info(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise ValueError("unsupported codec")

    _install_fake_qwen_utils(monkeypatch, process_vision_info)
    monkeypatch.setattr(decoder, "_sample_window_frames", lambda *args: _sampled_payload())
    with pytest.raises(VideoDecodeError, match="after 1 attempt"):
        decode_video_window(
            tmp_path / "video.mp4",
            0,
            2_000,
            semantic_profile(Settings()),
            max_attempts=2,
            retry_delay_ms=0,
        )
    assert calls == 1


@pytest.mark.parametrize(
    "error",
    [
        BlockingIOError(11, "Resource temporarily unavailable"),
        RuntimeError("[Errno 11] Resource temporarily unavailable"),
    ],
)
def test_only_eagain_decoder_errors_are_classified_as_transient(error):
    assert _is_transient_decode_error(error) is True
    assert _is_transient_decode_error(ValueError("invalid media")) is False


def test_decode_retry_settings_are_validated():
    with pytest.raises(ValueError, match="attempts"):
        Settings(video_decode_max_attempts=0)
    with pytest.raises(ValueError, match="delay"):
        Settings(video_decode_retry_delay_ms=-1)


def test_semantic_sampling_uses_24_uniform_frames_for_full_window():
    profile = semantic_profile(Settings())
    timestamps = plan_video_timestamps(10_000, 22_000, profile)

    assert len(timestamps) == 24
    assert timestamps[0] == 10_250
    assert timestamps[-1] == 21_750


def test_no_physical_subvideo_files_written(tmp_path):
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    ("duration_ms", "expected_frames"),
    [(12_000, 24), (27_000, 24), (65_000, 24)],
)
def test_overview_sampling_fixed_max_always_uses_max_frames(duration_ms, expected_frames):
    timestamps = plan_video_timestamps(0, duration_ms, overview_profile(Settings()))
    assert len(timestamps) == expected_frames
    assert timestamps == sorted(timestamps)
    assert timestamps[-1] < duration_ms


@pytest.mark.parametrize(
    ("duration_ms", "expected_frames"),
    [(12_000, 12), (27_000, 24), (65_000, 24)],
)
def test_overview_adaptive_scales_then_caps_frame_budget(duration_ms, expected_frames):
    config = Settings(global_overview_frame_policy="ADAPTIVE")
    timestamps = plan_video_timestamps(0, duration_ms, overview_profile(config))
    assert len(timestamps) == expected_frames
    assert timestamps == sorted(timestamps)
    assert timestamps[-1] < duration_ms
