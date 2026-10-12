"""Tests for Gap 5 (independent degradation) and Gap 6 (analysis modes)."""

import asyncio
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from visual_service.contracts.responses import PassStatus
from visual_service.model.pipeline import run_visual_pipeline
from visual_service.model.qwen_analyzer import (
    GenerationAttempt,
    GlobalOverviewResult,
    OcrAnalysisResult,
    SynthesisResult,
    WindowAnalysisResult,
    WindowValidationFailed,
)
from visual_service.settings import AnalysisMode, Settings
from visual_service.validation.output_validator import ObservationItem


def _mock_runtime():
    runtime = MagicMock()
    runtime.processor.tokenizer.encode.return_value = [1, 2, 3]
    return runtime


def _attempt(name="test", partial=False) -> GenerationAttempt:
    return GenerationAttempt(
        pass_name=name,
        raw_output="{}",
        generated_tokens=8,
        visual_tokens=10,
        finish_reason="stop",
        decode_ms=1,
        preprocess_ms=1,
        inference_ms=1,
        peak_vram_bytes=1000,
        selected_frames=4,
        width=100,
        height=100,
        is_partial=partial,
    )


def _global_result(summary="Video summary.", topics=None):
    return GlobalOverviewResult(
        summary=summary,
        topics=topics or ["topic1"],
        attempt=_attempt("global-overview"),
    )


def _window_result(observations=None, objects=None):
    return WindowAnalysisResult(
        observations=observations or [
            ObservationItem(order=1, type="action", description="A person walks."),
        ],
        objects=objects or ["person"],
        attempt=_attempt("observations"),
    )


def _ocr_result(texts=None, frame_seqs=None):
    t = texts if texts is not None else ["HELLO"]
    f = frame_seqs if frame_seqs is not None else [0]
    return OcrAnalysisResult(
        visible_texts=t,
        text_frame_sequences=f,
        attempts=[_attempt("ocr-0")],
        decode_ms=10,
        selected_frames=4,
    )


# --- Gap 5: Independent degradation ---

@pytest.mark.asyncio
async def test_global_overview_failure_does_not_block_semantic(tmp_path):
    media = tmp_path / "video.mp4"

    async def failing_overview(*args, **kwargs):
        raise RuntimeError("Global overview failed")

    with patch("visual_service.model.pipeline.analyze_global_overview", failing_overview), \
         patch("visual_service.model.pipeline.analyze_window", AsyncMock(return_value=_window_result())), \
         patch("visual_service.model.pipeline.analyze_ocr_video", AsyncMock(return_value=_ocr_result())):
        result = await run_visual_pipeline(_mock_runtime(), media, 12_000, Settings())

    assert result.consolidated.summary == ""
    assert len(result.consolidated.observations) > 0
    assert result.metrics.globalOverviewPass.status == PassStatus.FAILED
    assert result.metrics.semanticPass.status == PassStatus.COMPLETE
    assert any("Global overview failed" in w for w in result.warnings)


@pytest.mark.asyncio
async def test_window_failure_does_not_block_later_windows(tmp_path):
    media = tmp_path / "video.mp4"
    call_count = 0

    async def selective_window(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("Window 0 failed")
        return _window_result()

    with patch("visual_service.model.pipeline.analyze_global_overview", AsyncMock(return_value=_global_result())), \
         patch("visual_service.model.pipeline.analyze_window", selective_window), \
         patch("visual_service.model.pipeline.analyze_ocr_video", AsyncMock(return_value=_ocr_result())):
        config = Settings(semantic_window_duration_ms=12_000)
        result = await run_visual_pipeline(_mock_runtime(), media, 30_000, config)

    assert len(result.consolidated.observations) > 0
    assert any("Window 0 failed" in w for w in result.warnings)


@pytest.mark.asyncio
async def test_all_windows_fail_raises(tmp_path):
    media = tmp_path / "video.mp4"

    async def all_fail(*args, **kwargs):
        raise RuntimeError("Failed")

    with patch("visual_service.model.pipeline.analyze_global_overview", AsyncMock(return_value=_global_result())), \
         patch("visual_service.model.pipeline.analyze_window", all_fail), \
         patch("visual_service.model.pipeline.analyze_ocr_video", AsyncMock(return_value=_ocr_result())):
        with pytest.raises(RuntimeError, match="No semantic window"):
            await run_visual_pipeline(_mock_runtime(), media, 12_000, Settings())


@pytest.mark.asyncio
async def test_ocr_failure_does_not_block_semantic(tmp_path):
    media = tmp_path / "video.mp4"

    async def failing_ocr(*args, **kwargs):
        raise RuntimeError("OCR failed")

    with patch("visual_service.model.pipeline.analyze_global_overview", AsyncMock(return_value=_global_result())), \
         patch("visual_service.model.pipeline.analyze_window", AsyncMock(return_value=_window_result())), \
         patch("visual_service.model.pipeline.analyze_ocr_video", failing_ocr):
        result = await run_visual_pipeline(_mock_runtime(), media, 12_000, Settings())

    assert len(result.consolidated.observations) > 0
    assert result.consolidated.visible_texts == []
    assert result.metrics.ocrPass.status == PassStatus.FAILED
    assert any("OCR pass failed" in w for w in result.warnings)


# --- Gap 6: Analysis modes ---

@pytest.mark.asyncio
async def test_semantic_only_skips_ocr_completely(tmp_path):
    media = tmp_path / "video.mp4"
    ocr_called = False

    async def track_ocr(*args, **kwargs):
        nonlocal ocr_called
        ocr_called = True
        return _ocr_result()

    with patch("visual_service.model.pipeline.analyze_global_overview", AsyncMock(return_value=_global_result())), \
         patch("visual_service.model.pipeline.analyze_window", AsyncMock(return_value=_window_result())), \
         patch("visual_service.model.pipeline.analyze_ocr_video", track_ocr):
        result = await run_visual_pipeline(
            _mock_runtime(), media, 12_000, Settings(),
            analysis_mode=AnalysisMode.SEMANTIC_ONLY,
        )

    assert not ocr_called
    assert result.consolidated.visible_texts == []
    assert result.metrics.ocrPass.status == PassStatus.SKIPPED
    assert result.metrics.ocrPass.inferenceCount == 0
    assert result.metrics.analysisMode == "SEMANTIC_ONLY"


@pytest.mark.asyncio
async def test_semantic_and_ocr_runs_ocr(tmp_path):
    media = tmp_path / "video.mp4"

    with patch("visual_service.model.pipeline.analyze_global_overview", AsyncMock(return_value=_global_result())), \
         patch("visual_service.model.pipeline.analyze_window", AsyncMock(return_value=_window_result())), \
         patch("visual_service.model.pipeline.analyze_ocr_video", AsyncMock(return_value=_ocr_result())):
        result = await run_visual_pipeline(
            _mock_runtime(), media, 12_000, Settings(),
            analysis_mode=AnalysisMode.SEMANTIC_AND_OCR,
        )

    assert result.consolidated.visible_texts == ["HELLO"]
    assert result.metrics.ocrPass.status != PassStatus.SKIPPED
    assert result.metrics.analysisMode == "SEMANTIC_AND_OCR"


@pytest.mark.asyncio
async def test_semantic_only_reports_ocr_skipped_not_failed(tmp_path):
    media = tmp_path / "video.mp4"

    with patch("visual_service.model.pipeline.analyze_global_overview", AsyncMock(return_value=_global_result())), \
         patch("visual_service.model.pipeline.analyze_window", AsyncMock(return_value=_window_result())), \
         patch("visual_service.model.pipeline.analyze_ocr_video", AsyncMock(return_value=_ocr_result())):
        result = await run_visual_pipeline(
            _mock_runtime(), media, 12_000, Settings(),
            analysis_mode=AnalysisMode.SEMANTIC_ONLY,
        )

    assert result.metrics.ocrPass.status == PassStatus.SKIPPED
    assert result.metrics.ocrPass.isPartial is False
    assert not any("OCR" in w and "fail" in w.lower() for w in result.warnings)


@pytest.mark.asyncio
async def test_analysis_mode_defaults_to_config(tmp_path):
    media = tmp_path / "video.mp4"
    config = Settings(analysis_mode="SEMANTIC_ONLY")

    with patch("visual_service.model.pipeline.analyze_global_overview", AsyncMock(return_value=_global_result())), \
         patch("visual_service.model.pipeline.analyze_window", AsyncMock(return_value=_window_result())), \
         patch("visual_service.model.pipeline.analyze_ocr_video", AsyncMock(return_value=_ocr_result())):
        result = await run_visual_pipeline(_mock_runtime(), media, 12_000, config)

    assert result.metrics.ocrPass.status == PassStatus.SKIPPED
    assert result.metrics.analysisMode == "SEMANTIC_ONLY"


@pytest.mark.asyncio
async def test_request_mode_overrides_config(tmp_path):
    media = tmp_path / "video.mp4"
    config = Settings(analysis_mode="SEMANTIC_AND_OCR")

    with patch("visual_service.model.pipeline.analyze_global_overview", AsyncMock(return_value=_global_result())), \
         patch("visual_service.model.pipeline.analyze_window", AsyncMock(return_value=_window_result())), \
         patch("visual_service.model.pipeline.analyze_ocr_video", AsyncMock(return_value=_ocr_result())):
        result = await run_visual_pipeline(
            _mock_runtime(), media, 12_000, config,
            analysis_mode=AnalysisMode.SEMANTIC_ONLY,
        )

    assert result.metrics.ocrPass.status == PassStatus.SKIPPED


@pytest.mark.asyncio
async def test_both_modes_share_semantic_processing(tmp_path):
    media = tmp_path / "video.mp4"

    for mode in (AnalysisMode.SEMANTIC_ONLY, AnalysisMode.SEMANTIC_AND_OCR):
        with patch("visual_service.model.pipeline.analyze_global_overview", AsyncMock(return_value=_global_result())), \
             patch("visual_service.model.pipeline.analyze_window", AsyncMock(return_value=_window_result())), \
             patch("visual_service.model.pipeline.analyze_ocr_video", AsyncMock(return_value=_ocr_result())):
            result = await run_visual_pipeline(
                _mock_runtime(), media, 12_000, Settings(), analysis_mode=mode,
            )
        assert len(result.consolidated.observations) > 0
        assert result.consolidated.summary != ""


# --- Window accounting ---

@pytest.mark.asyncio
async def test_window_accounting_all_successful(tmp_path):
    media = tmp_path / "video.mp4"

    with patch("visual_service.model.pipeline.analyze_global_overview", AsyncMock(return_value=_global_result())), \
         patch("visual_service.model.pipeline.analyze_window", AsyncMock(return_value=_window_result())), \
         patch("visual_service.model.pipeline.analyze_ocr_video", AsyncMock(return_value=_ocr_result())):
        result = await run_visual_pipeline(_mock_runtime(), media, 12_000, Settings())

    acct = result.metrics.windowAccounting
    assert acct.plannedWindows == 1
    assert acct.attemptedWindows == 1
    assert acct.successfulWindows == 1
    assert acct.partialWindows == 0
    assert acct.failedWindows == 0
    assert result.metrics.semanticPass.status == PassStatus.COMPLETE


@pytest.mark.asyncio
async def test_window_accounting_one_partial(tmp_path):
    media = tmp_path / "video.mp4"
    partial_attempt = _attempt("observations", partial=True)
    partial_result = WindowAnalysisResult(
        observations=[ObservationItem(order=1, type="action", description="Walks.")],
        objects=[],
        attempt=partial_attempt,
    )

    with patch("visual_service.model.pipeline.analyze_global_overview", AsyncMock(return_value=_global_result())), \
         patch("visual_service.model.pipeline.analyze_window", AsyncMock(return_value=partial_result)), \
         patch("visual_service.model.pipeline.analyze_ocr_video", AsyncMock(return_value=_ocr_result())):
        result = await run_visual_pipeline(_mock_runtime(), media, 12_000, Settings())

    acct = result.metrics.windowAccounting
    assert acct.partialWindows == 1
    assert acct.successfulWindows == 1
    assert acct.failedWindows == 0
    assert result.metrics.semanticPass.status == PassStatus.PARTIAL


@pytest.mark.asyncio
async def test_window_accounting_one_failed_one_succeeded(tmp_path):
    media = tmp_path / "video.mp4"
    call_count = 0

    async def selective_window(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise WindowValidationFailed("bad", _attempt("observations"))
        return _window_result()

    with patch("visual_service.model.pipeline.analyze_global_overview", AsyncMock(return_value=_global_result())), \
         patch("visual_service.model.pipeline.analyze_window", selective_window), \
         patch("visual_service.model.pipeline.analyze_ocr_video", AsyncMock(return_value=_ocr_result())):
        config = Settings(semantic_window_duration_ms=12_000)
        result = await run_visual_pipeline(_mock_runtime(), media, 30_000, config)

    acct = result.metrics.windowAccounting
    assert acct.plannedWindows == 3
    assert acct.failedWindows == 1
    assert acct.successfulWindows >= 1
    assert result.metrics.semanticPass.status == PassStatus.PARTIAL


@pytest.mark.asyncio
async def test_window_accounting_no_successful_windows_fails(tmp_path):
    media = tmp_path / "video.mp4"

    async def all_fail(*args, **kwargs):
        raise WindowValidationFailed("bad", _attempt("observations"))

    with patch("visual_service.model.pipeline.analyze_global_overview", AsyncMock(return_value=_global_result())), \
         patch("visual_service.model.pipeline.analyze_window", all_fail), \
         patch("visual_service.model.pipeline.analyze_ocr_video", AsyncMock(return_value=_ocr_result())):
        with pytest.raises(RuntimeError, match="No semantic window"):
            await run_visual_pipeline(_mock_runtime(), media, 12_000, Settings())


# --- Attempt preservation ---

@pytest.mark.asyncio
async def test_failed_validation_attempt_preserved_in_trace(tmp_path):
    media = tmp_path / "video.mp4"
    failed_attempt = _attempt("observations")
    call_count = 0

    async def selective_window(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise WindowValidationFailed("invalid output", failed_attempt)
        return _window_result()

    with patch("visual_service.model.pipeline.analyze_global_overview", AsyncMock(return_value=_global_result())), \
         patch("visual_service.model.pipeline.analyze_window", selective_window), \
         patch("visual_service.model.pipeline.analyze_ocr_video", AsyncMock(return_value=_ocr_result())):
        config = Settings(semantic_window_duration_ms=12_000)
        result = await run_visual_pipeline(_mock_runtime(), media, 30_000, config)

    assert failed_attempt in result.trace.attempts
    assert any("invalid output" in w for w in result.warnings)


# --- Grounded synthesis pipeline tests ---

def _synthesis_result(ideas=None, themes=None, partial=False):
    return SynthesisResult(
        communicated_ideas=ideas or ["On-screen text presents motivation."],
        themes=themes or ["motivation"],
        evidence_given=[],
        raw_output="{}",
        validated_ideas=[],
        validated_themes=[],
        attempt=_attempt("grounded-synthesis"),
        is_partial=partial,
    )


@pytest.mark.asyncio
async def test_synthesis_runs_after_ocr(tmp_path):
    media = tmp_path / "video.mp4"
    order: list[str] = []

    async def track_overview(*args, **kwargs):
        order.append("overview")
        return _global_result()

    async def track_window(*args, **kwargs):
        order.append("semantic")
        return _window_result()

    async def track_ocr(*args, **kwargs):
        order.append("ocr")
        return _ocr_result()

    async def track_synthesis(*args, **kwargs):
        order.append("synthesis")
        return _synthesis_result()

    with patch("visual_service.model.pipeline.analyze_global_overview", track_overview), \
         patch("visual_service.model.pipeline.analyze_window", track_window), \
         patch("visual_service.model.pipeline.analyze_ocr_video", track_ocr), \
         patch("visual_service.model.pipeline.analyze_grounded_synthesis", track_synthesis):
        result = await run_visual_pipeline(
            _mock_runtime(), media, 12_000,
            Settings(grounded_text_understanding_enabled=True),
        )

    assert order.index("overview") < order.index("semantic")
    assert order.index("semantic") < order.index("ocr")
    assert order.index("ocr") < order.index("synthesis")
    assert result.consolidated.communicated_ideas == ["On-screen text presents motivation."]
    assert result.consolidated.themes == ["motivation"]


@pytest.mark.asyncio
async def test_feature_disabled_skips_synthesis(tmp_path):
    media = tmp_path / "video.mp4"
    synthesis_called = False

    async def track_synthesis(*args, **kwargs):
        nonlocal synthesis_called
        synthesis_called = True
        return _synthesis_result()

    with patch("visual_service.model.pipeline.analyze_global_overview", AsyncMock(return_value=_global_result())), \
         patch("visual_service.model.pipeline.analyze_window", AsyncMock(return_value=_window_result())), \
         patch("visual_service.model.pipeline.analyze_ocr_video", AsyncMock(return_value=_ocr_result())), \
         patch("visual_service.model.pipeline.analyze_grounded_synthesis", track_synthesis):
        result = await run_visual_pipeline(
            _mock_runtime(), media, 12_000,
            Settings(grounded_text_understanding_enabled=False),
        )

    assert not synthesis_called
    assert result.consolidated.communicated_ideas == []
    assert result.consolidated.themes == []
    assert result.metrics.groundedTextPass.status == PassStatus.SKIPPED


@pytest.mark.asyncio
async def test_empty_ocr_skips_synthesis(tmp_path):
    media = tmp_path / "video.mp4"
    synthesis_called = False

    async def track_synthesis(*args, **kwargs):
        nonlocal synthesis_called
        synthesis_called = True
        return _synthesis_result()

    with patch("visual_service.model.pipeline.analyze_global_overview", AsyncMock(return_value=_global_result())), \
         patch("visual_service.model.pipeline.analyze_window", AsyncMock(return_value=_window_result())), \
         patch("visual_service.model.pipeline.analyze_ocr_video", AsyncMock(return_value=_ocr_result(texts=[]))), \
         patch("visual_service.model.pipeline.analyze_grounded_synthesis", track_synthesis):
        result = await run_visual_pipeline(
            _mock_runtime(), media, 12_000,
            Settings(grounded_text_understanding_enabled=True),
        )

    assert not synthesis_called
    assert result.metrics.groundedTextPass.status == PassStatus.SKIPPED


@pytest.mark.asyncio
async def test_semantic_only_skips_synthesis(tmp_path):
    media = tmp_path / "video.mp4"
    synthesis_called = False

    async def track_synthesis(*args, **kwargs):
        nonlocal synthesis_called
        synthesis_called = True
        return _synthesis_result()

    with patch("visual_service.model.pipeline.analyze_global_overview", AsyncMock(return_value=_global_result())), \
         patch("visual_service.model.pipeline.analyze_window", AsyncMock(return_value=_window_result())), \
         patch("visual_service.model.pipeline.analyze_ocr_video", AsyncMock(return_value=_ocr_result())), \
         patch("visual_service.model.pipeline.analyze_grounded_synthesis", track_synthesis):
        result = await run_visual_pipeline(
            _mock_runtime(), media, 12_000, Settings(),
            analysis_mode=AnalysisMode.SEMANTIC_ONLY,
        )

    assert not synthesis_called
    assert result.metrics.groundedTextPass.status == PassStatus.SKIPPED


@pytest.mark.asyncio
async def test_synthesis_failure_does_not_block_pipeline(tmp_path):
    media = tmp_path / "video.mp4"

    async def failing_synthesis(*args, **kwargs):
        raise RuntimeError("Synthesis failed")

    with patch("visual_service.model.pipeline.analyze_global_overview", AsyncMock(return_value=_global_result())), \
         patch("visual_service.model.pipeline.analyze_window", AsyncMock(return_value=_window_result())), \
         patch("visual_service.model.pipeline.analyze_ocr_video", AsyncMock(return_value=_ocr_result())), \
         patch("visual_service.model.pipeline.analyze_grounded_synthesis", failing_synthesis):
        result = await run_visual_pipeline(
            _mock_runtime(), media, 12_000,
            Settings(grounded_text_understanding_enabled=True),
        )

    assert len(result.consolidated.observations) > 0
    assert result.consolidated.visible_texts == ["HELLO"]
    assert result.consolidated.communicated_ideas == []
    assert result.metrics.groundedTextPass.status == PassStatus.FAILED
    assert any("Grounded synthesis failed" in w for w in result.warnings)


@pytest.mark.asyncio
async def test_synthesis_adds_exactly_one_inference(tmp_path):
    media = tmp_path / "video.mp4"

    with patch("visual_service.model.pipeline.analyze_global_overview", AsyncMock(return_value=_global_result())), \
         patch("visual_service.model.pipeline.analyze_window", AsyncMock(return_value=_window_result())), \
         patch("visual_service.model.pipeline.analyze_ocr_video", AsyncMock(return_value=_ocr_result())), \
         patch("visual_service.model.pipeline.analyze_grounded_synthesis", AsyncMock(return_value=_synthesis_result())):
        result = await run_visual_pipeline(
            _mock_runtime(), media, 12_000,
            Settings(grounded_text_understanding_enabled=True),
        )

    total_inferences = (
        result.metrics.globalOverviewPass.inferenceCount
        + result.metrics.semanticPass.inferenceCount
        + result.metrics.ocrPass.inferenceCount
        + result.metrics.groundedTextPass.inferenceCount
    )
    assert total_inferences == 4
    assert result.metrics.groundedTextPass.inferenceCount == 1


@pytest.mark.asyncio
async def test_synthesis_metrics_in_response(tmp_path):
    media = tmp_path / "video.mp4"

    with patch("visual_service.model.pipeline.analyze_global_overview", AsyncMock(return_value=_global_result())), \
         patch("visual_service.model.pipeline.analyze_window", AsyncMock(return_value=_window_result())), \
         patch("visual_service.model.pipeline.analyze_ocr_video", AsyncMock(return_value=_ocr_result())), \
         patch("visual_service.model.pipeline.analyze_grounded_synthesis", AsyncMock(return_value=_synthesis_result())):
        result = await run_visual_pipeline(
            _mock_runtime(), media, 12_000,
            Settings(grounded_text_understanding_enabled=True),
        )

    assert result.metrics.groundedTextPass.status == PassStatus.COMPLETE
    assert result.metrics.groundedTextPass.inferenceCount == 1
    assert result.metrics.groundedTextPass.inferenceMs >= 0


@pytest.mark.asyncio
async def test_long_video_segmentation_still_works(tmp_path):
    media = tmp_path / "video.mp4"

    with patch("visual_service.model.pipeline.analyze_global_overview", AsyncMock(return_value=_global_result())), \
         patch("visual_service.model.pipeline.analyze_window", AsyncMock(return_value=_window_result())), \
         patch("visual_service.model.pipeline.analyze_ocr_video", AsyncMock(return_value=_ocr_result())):
        config = Settings(semantic_window_duration_ms=12_000)
        result = await run_visual_pipeline(_mock_runtime(), media, 65_000, config)

    assert result.metrics.windowAccounting.plannedWindows >= 6


# --- Budget exhaustion / metrics preservation ---

def _exhausted_synthesis_result():
    attempt = _attempt("grounded-synthesis", partial=True)
    attempt.finish_reason = "length"
    attempt.inference_ms = 500
    attempt.input_tokens = 2048
    return SynthesisResult(
        communicated_ideas=[],
        themes=[],
        evidence_given=[],
        raw_output='{"communicatedIdeas": [{"idea": "trun',
        validated_ideas=[],
        validated_themes=[],
        attempt=attempt,
        is_partial=True,
    )


@pytest.mark.asyncio
async def test_synthesis_budget_exhaustion_preserves_attempt(tmp_path):
    media = tmp_path / "video.mp4"

    with patch("visual_service.model.pipeline.analyze_global_overview", AsyncMock(return_value=_global_result())), \
         patch("visual_service.model.pipeline.analyze_window", AsyncMock(return_value=_window_result())), \
         patch("visual_service.model.pipeline.analyze_ocr_video", AsyncMock(return_value=_ocr_result())), \
         patch("visual_service.model.pipeline.analyze_grounded_synthesis", AsyncMock(return_value=_exhausted_synthesis_result())):
        result = await run_visual_pipeline(
            _mock_runtime(), media, 12_000,
            Settings(grounded_text_understanding_enabled=True),
        )

    assert result.metrics.groundedTextPass.inferenceCount == 1
    assert result.metrics.groundedTextPass.inferenceMs > 0
    assert result.metrics.groundedTextPass.status == PassStatus.FAILED
    assert any(a.pass_name == "grounded-synthesis" for a in result.trace.attempts)


@pytest.mark.asyncio
async def test_synthesis_partial_content_is_partial_status(tmp_path):
    media = tmp_path / "video.mp4"
    partial = _synthesis_result(ideas=["Partial idea."], partial=True)

    with patch("visual_service.model.pipeline.analyze_global_overview", AsyncMock(return_value=_global_result())), \
         patch("visual_service.model.pipeline.analyze_window", AsyncMock(return_value=_window_result())), \
         patch("visual_service.model.pipeline.analyze_ocr_video", AsyncMock(return_value=_ocr_result())), \
         patch("visual_service.model.pipeline.analyze_grounded_synthesis", AsyncMock(return_value=partial)):
        result = await run_visual_pipeline(
            _mock_runtime(), media, 12_000,
            Settings(grounded_text_understanding_enabled=True),
        )

    assert result.metrics.groundedTextPass.status == PassStatus.PARTIAL
    assert result.consolidated.communicated_ideas == ["Partial idea."]


@pytest.mark.asyncio
async def test_synthesis_budget_preserves_raw_output(tmp_path):
    media = tmp_path / "video.mp4"

    with patch("visual_service.model.pipeline.analyze_global_overview", AsyncMock(return_value=_global_result())), \
         patch("visual_service.model.pipeline.analyze_window", AsyncMock(return_value=_window_result())), \
         patch("visual_service.model.pipeline.analyze_ocr_video", AsyncMock(return_value=_ocr_result())), \
         patch("visual_service.model.pipeline.analyze_grounded_synthesis", AsyncMock(return_value=_exhausted_synthesis_result())):
        result = await run_visual_pipeline(
            _mock_runtime(), media, 12_000,
            Settings(grounded_text_understanding_enabled=True),
        )

    synthesis_attempts = [a for a in result.trace.attempts if a.pass_name == "grounded-synthesis"]
    assert len(synthesis_attempts) == 1
    assert synthesis_attempts[0].raw_output != ""


@pytest.mark.asyncio
async def test_synthesis_budget_preserves_input_tokens(tmp_path):
    media = tmp_path / "video.mp4"

    with patch("visual_service.model.pipeline.analyze_global_overview", AsyncMock(return_value=_global_result())), \
         patch("visual_service.model.pipeline.analyze_window", AsyncMock(return_value=_window_result())), \
         patch("visual_service.model.pipeline.analyze_ocr_video", AsyncMock(return_value=_ocr_result())), \
         patch("visual_service.model.pipeline.analyze_grounded_synthesis", AsyncMock(return_value=_exhausted_synthesis_result())):
        result = await run_visual_pipeline(
            _mock_runtime(), media, 12_000,
            Settings(grounded_text_understanding_enabled=True),
        )

    assert result.metrics.groundedTextPass.inputTokens is not None


@pytest.mark.asyncio
async def test_visible_texts_preserved_in_consolidation(tmp_path):
    media = tmp_path / "video.mp4"
    ocr = _ocr_result(texts=["CALIFORNIA", "123 Main St", "philosophical caption"], frame_seqs=[0, 5, 10])

    with patch("visual_service.model.pipeline.analyze_global_overview", AsyncMock(return_value=_global_result())), \
         patch("visual_service.model.pipeline.analyze_window", AsyncMock(return_value=_window_result())), \
         patch("visual_service.model.pipeline.analyze_ocr_video", AsyncMock(return_value=ocr)), \
         patch("visual_service.model.pipeline.analyze_grounded_synthesis", AsyncMock(return_value=_synthesis_result(ideas=["Philosophical idea."]))):
        result = await run_visual_pipeline(
            _mock_runtime(), media, 12_000,
            Settings(grounded_text_understanding_enabled=True),
        )

    assert "CALIFORNIA" in result.consolidated.visible_texts
    assert "123 Main St" in result.consolidated.visible_texts
    assert "philosophical caption" in result.consolidated.visible_texts
