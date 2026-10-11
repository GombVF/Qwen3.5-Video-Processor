import asyncio
import threading

import pytest

from visual_service.model import qwen_analyzer
from visual_service.model.qwen_analyzer import GenerationAttempt
from visual_service.settings import Settings
from visual_service.video.profiles import semantic_profile


def _attempt() -> GenerationAttempt:
    return GenerationAttempt(
        pass_name="observations",
        raw_output='{"observations":[],"objects":[]}',
        generated_tokens=8,
        visual_tokens=10,
        finish_reason="stop",
        decode_ms=1,
        preprocess_ms=1,
        inference_ms=1,
        peak_vram_bytes=1,
        selected_frames=4,
        width=100,
        height=100,
    )


@pytest.mark.asyncio
async def test_cancellation_does_not_release_gpu_lock(monkeypatch, tmp_path):
    first_started = threading.Event()
    first_release = threading.Event()
    second_started = threading.Event()
    counter_lock = threading.Lock()
    call_count = 0

    def fake_generate(*args, **kwargs):
        nonlocal call_count
        with counter_lock:
            call_count += 1
            current = call_count
        if current == 1:
            first_started.set()
            first_release.wait(timeout=5)
        else:
            second_started.set()
        return _attempt()

    monkeypatch.setattr(qwen_analyzer, "_generate_video", fake_generate)
    config = Settings()
    profile = semantic_profile(config)
    media = tmp_path / "video.mp4"

    first = asyncio.create_task(
        qwen_analyzer.analyze_window(object(), media, 0, 1_000, profile, config)
    )
    assert await asyncio.to_thread(first_started.wait, 2)
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first

    second = asyncio.create_task(
        qwen_analyzer.analyze_window(object(), media, 0, 1_000, profile, config)
    )
    await asyncio.sleep(0.05)
    assert not second_started.is_set()

    first_release.set()
    await second
    assert second_started.is_set()


def test_no_generative_repair_path_exists():
    source = __import__("inspect").getsource(qwen_analyzer)
    assert "build_repair_prompt" not in source
    assert "repair_inference" not in source


def test_inference_lock_is_threading_lock():
    assert isinstance(qwen_analyzer._inference_lock, type(threading.Lock()))
