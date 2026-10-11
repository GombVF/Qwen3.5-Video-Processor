from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from visual_service.contracts.responses import (
    PassMetrics,
    PassStatus,
    VisualMetricsResponse,
    WindowAccounting,
)
from visual_service.model.consolidator import (
    ConsolidatedResult,
    ObservationEntry,
    WindowObservationResult,
    consolidate_results,
)
from visual_service.model.qwen_analyzer import (
    GenerationAttempt,
    WindowValidationFailed,
    analyze_global_overview,
    analyze_ocr_video,
    analyze_window,
)
from visual_service.settings import AnalysisMode, Settings
from visual_service.video.planner import create_windows
from visual_service.video.profiles import overview_profile, semantic_profile

logger = logging.getLogger(__name__)


@dataclass
class PipelineTrace:
    attempts: list[GenerationAttempt] = field(default_factory=list)


@dataclass
class PipelineResult:
    consolidated: ConsolidatedResult
    metrics: VisualMetricsResponse
    warnings: list[str]
    trace: PipelineTrace


async def run_visual_pipeline(
    runtime: Any,
    media_path: Path,
    duration_ms: int,
    config: Settings,
    *,
    analysis_mode: AnalysisMode | None = None,
) -> PipelineResult:
    mode = analysis_mode or config.analysis_mode
    started = time.perf_counter()
    semantic = semantic_profile(config)
    overview = overview_profile(config)
    windows, strategy = create_windows(duration_ms, semantic)
    warnings: list[str] = []
    trace = PipelineTrace()

    logger.info("qwen35_pipeline_start", extra={"data": {
        "durationMs": duration_ms,
        "strategy": strategy,
        "windowCount": len(windows),
        "analysisMode": mode.value,
    }})

    global_summary = ""
    global_topics: list[str] = []
    overview_attempts: list[GenerationAttempt] = []
    overview_status = PassStatus.COMPLETE
    try:
        global_result = await analyze_global_overview(
            runtime, media_path, duration_ms, overview, config,
        )
        global_summary = global_result.summary
        global_topics = global_result.topics
        overview_attempts.append(global_result.attempt)
        trace.attempts.append(global_result.attempt)
        warnings.extend(global_result.attempt.warnings)
        if global_result.attempt.is_partial:
            overview_status = PassStatus.PARTIAL
    except Exception as exc:
        logger.warning("global_overview_failed", extra={"data": {
            "error": str(exc),
        }})
        warnings.append(f"Global overview failed: {exc}")
        overview_status = PassStatus.FAILED

    window_results: list[WindowObservationResult] = []
    semantic_attempts: list[GenerationAttempt] = []
    failed_window_count = 0
    partial_window_count = 0
    for window in windows:
        try:
            result = await analyze_window(
                runtime,
                media_path,
                window.start_ms,
                window.end_ms,
                semantic,
                config,
            )
        except WindowValidationFailed as exc:
            semantic_attempts.append(exc.attempt)
            trace.attempts.append(exc.attempt)
            warnings.extend(exc.attempt.warnings)
            warnings.append(f"Window {window.sequence} validation failed: {exc}")
            failed_window_count += 1
            logger.warning("semantic_window_validation_failed", extra={"data": {
                "window": window.sequence,
                "error": str(exc),
            }})
            continue
        except Exception as exc:
            logger.warning("semantic_window_failed", extra={"data": {
                "window": window.sequence,
                "error": str(exc),
            }})
            warnings.append(f"Window {window.sequence} failed: {exc}")
            failed_window_count += 1
            continue
        semantic_attempts.append(result.attempt)
        trace.attempts.append(result.attempt)
        warnings.extend(result.attempt.warnings)
        if result.attempt.is_partial:
            partial_window_count += 1
        window_results.append(WindowObservationResult(
            window_index=window.sequence,
            observations=[
                ObservationEntry(item.order, item.type, item.description)
                for item in result.observations
            ],
            objects=result.objects,
            is_partial=result.attempt.is_partial,
            warnings=list(result.attempt.warnings),
        ))

    if not window_results:
        raise RuntimeError("No semantic window produced a valid result")

    ocr_visible_texts: list[str] = []
    ocr_frame_sequences: list[int] | None = None
    ocr_attempts: list[GenerationAttempt] = []
    ocr_decode_ms = 0
    ocr_selected_frames = 0
    ocr_is_partial = False
    ocr_status = PassStatus.COMPLETE

    if mode == AnalysisMode.SEMANTIC_AND_OCR:
        try:
            ocr_result = await analyze_ocr_video(runtime, media_path, duration_ms, config)
            ocr_visible_texts = ocr_result.visible_texts
            ocr_frame_sequences = ocr_result.text_frame_sequences
            ocr_attempts = ocr_result.attempts
            ocr_decode_ms = ocr_result.decode_ms
            ocr_selected_frames = ocr_result.selected_frames
            ocr_is_partial = ocr_result.is_partial
            trace.attempts.extend(ocr_result.attempts)
            warnings.extend(ocr_result.warnings)
            if ocr_is_partial:
                ocr_status = PassStatus.PARTIAL
        except Exception as exc:
            logger.warning("ocr_pass_failed", extra={"data": {"error": str(exc)}})
            warnings.append(f"OCR pass failed: {exc}")
            ocr_status = PassStatus.FAILED
    else:
        ocr_status = PassStatus.SKIPPED

    consolidated = consolidate_results(
        window_results,
        global_summary=global_summary,
        global_topics=global_topics,
        visible_texts=ocr_visible_texts,
        adjacent_threshold=config.consolidation_adjacent_threshold,
        nonadjacent_threshold=config.consolidation_nonadjacent_threshold,
        visible_text_threshold=config.consolidation_visible_text_threshold,
        text_frame_sequences=ocr_frame_sequences,
        fragment_neighborhood=config.consolidation_fragment_neighborhood,
    )
    total_ms = int((time.perf_counter() - started) * 1000)
    overview_metrics = _aggregate_attempts(overview_attempts)
    overview_metrics.status = overview_status
    semantic_metrics = _aggregate_attempts(semantic_attempts)
    successful_windows = len(window_results)
    accounting = WindowAccounting(
        plannedWindows=len(windows),
        attemptedWindows=successful_windows + failed_window_count,
        successfulWindows=successful_windows,
        partialWindows=partial_window_count,
        failedWindows=failed_window_count,
    )
    if successful_windows == len(windows) and partial_window_count == 0:
        semantic_metrics.status = PassStatus.COMPLETE
    elif successful_windows > 0:
        semantic_metrics.status = PassStatus.PARTIAL
    else:
        semantic_metrics.status = PassStatus.FAILED
    ocr_metrics = _aggregate_attempts(ocr_attempts)
    ocr_metrics.decodeMs += ocr_decode_ms
    ocr_metrics.selectedFrames = ocr_selected_frames
    ocr_metrics.isPartial = ocr_metrics.isPartial or ocr_is_partial
    ocr_metrics.status = ocr_status
    peak_vram = max(
        overview_metrics.peakVramBytes,
        semantic_metrics.peakVramBytes,
        ocr_metrics.peakVramBytes,
        0,
    )

    logger.info("qwen35_pipeline_complete", extra={"data": {
        "totalMs": total_ms,
        "windowCount": len(windows),
        "observationCount": len(consolidated.observations),
        "visibleTextCount": len(consolidated.visible_texts),
        "peakVramBytes": peak_vram,
        "analysisMode": mode.value,
    }})
    return PipelineResult(
        consolidated=consolidated,
        metrics=VisualMetricsResponse(
            totalMs=total_ms,
            peakVramBytes=peak_vram,
            windowCount=len(windows),
            analysisMode=mode.value,
            globalOverviewPass=overview_metrics,
            semanticPass=semantic_metrics,
            ocrPass=ocr_metrics,
            windowAccounting=accounting,
        ),
        warnings=warnings,
        trace=trace,
    )


def _aggregate_attempts(attempts: list[GenerationAttempt]) -> PassMetrics:
    if not attempts:
        return PassMetrics()
    visual_counts = [item.visual_tokens for item in attempts if item.visual_tokens is not None]
    return PassMetrics(
        decodeMs=sum(item.decode_ms for item in attempts),
        preprocessMs=sum(item.preprocess_ms for item in attempts),
        inferenceMs=sum(item.inference_ms for item in attempts),
        generatedTokens=sum(item.generated_tokens for item in attempts),
        visualTokens=sum(visual_counts) if visual_counts else None,
        peakVramBytes=max(item.peak_vram_bytes for item in attempts),
        inferenceCount=len(attempts),
        selectedFrames=sum(item.selected_frames for item in attempts),
        finishReason="length" if any(item.finish_reason == "length" for item in attempts) else "stop",
        isPartial=any(item.is_partial for item in attempts),
    )
