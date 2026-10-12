"""End-to-end benchmark for the single Qwen3.5-4B visual pipeline."""

from __future__ import annotations

import argparse
import asyncio
import importlib.metadata
import json
import os
import platform
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from visual_service.model.lifecycle import load_qwen_runtime
from visual_service.model.pipeline import run_visual_pipeline
from visual_service.model.qwen_analyzer import (
    analyze_global_overview,
    analyze_ocr_video,
    analyze_window,
)
from visual_service.settings import AnalysisMode, FramePolicy, settings
from visual_service.video.profiles import overview_profile, semantic_profile


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--media",
        action="append",
        type=Path,
        help="Video path. Repeat for multiple videos. Defaults to data/test_videos/*.mp4.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "benchmark-qwen35-4b.json",
    )
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="Do not warm all three inference shapes before the benchmark.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=[m.value for m in AnalysisMode],
        default=None,
        help="Analysis mode. Defaults to the configured setting.",
    )
    return parser.parse_args()


def video_duration_ms(path: Path) -> int:
    import av

    container = av.open(str(path))
    try:
        stream = container.streams.video[0]
        if stream.duration is not None:
            return int(float(stream.duration * stream.time_base) * 1000)
        if container.duration is not None:
            return int(container.duration / 1000)
    finally:
        container.close()
    raise RuntimeError(f"Could not determine duration for {path}")


async def run() -> None:
    args = parse_args()
    media_paths = args.media or sorted((PROJECT_ROOT / "data" / "test_videos").glob("*.mp4"))
    if not media_paths:
        raise SystemExit("No media files provided or found")
    for path in media_paths:
        if not path.is_file():
            raise SystemExit(f"Video does not exist: {path}")

    mode = AnalysisMode(args.mode) if args.mode else None

    model_load_started = time.perf_counter()
    runtime = await load_qwen_runtime(settings)
    model_load_ms = int((time.perf_counter() - model_load_started) * 1000)
    try:
        warmup_record = None
        if not args.no_warmup:
            # Triton compiles separate kernels for materially different visual
            # shapes. Warm the maximum overview shape, a full semantic window,
            # and one full OCR batch so measured timings represent the resident
            # long-running service instead of one-time compilation latency.
            warmup_path = max(media_paths, key=video_duration_ms)
            available_ms = video_duration_ms(warmup_path)
            overview_duration = min(
                available_ms,
                int(settings.global_overview_max_frames / settings.global_overview_fps * 1000),
            )
            semantic_duration = min(available_ms, settings.semantic_window_duration_ms)
            ocr_duration = min(
                available_ms,
                max(
                    1_000,
                    int(settings.ocr_batch_size / settings.ocr_fps * 1000),
                ),
            )
            warmup_started = time.perf_counter()
            overview_warmup = await analyze_global_overview(
                runtime,
                warmup_path,
                overview_duration,
                overview_profile(settings),
                settings,
            )
            semantic_warmup = await analyze_window(
                runtime,
                warmup_path,
                0,
                semantic_duration,
                semantic_profile(settings),
                settings,
            )
            ocr_warmup = await analyze_ocr_video(
                runtime, warmup_path, ocr_duration, settings,
            )
            sem_prof = semantic_profile(settings)
            ovr_prof = overview_profile(settings)
            warmup_record = {
                "media": str(warmup_path),
                "totalMs": int((time.perf_counter() - warmup_started) * 1000),
                "warmedShapes": [
                    {
                        "pass": "global-overview",
                        "frames": overview_warmup.attempt.selected_frames,
                        "framePolicy": ovr_prof.frame_policy.value,
                    },
                    {
                        "pass": "semantic",
                        "frames": semantic_warmup.attempt.selected_frames,
                        "framePolicy": sem_prof.frame_policy.value,
                    },
                    {
                        "pass": "ocr",
                        "frames": ocr_warmup.selected_frames,
                        "framePolicy": "ADAPTIVE",
                    },
                ],
                "globalOverview": {
                    "durationMs": overview_duration,
                    "inferenceMs": overview_warmup.attempt.inference_ms,
                    "selectedFrames": overview_warmup.attempt.selected_frames,
                    "generatedTokens": overview_warmup.attempt.generated_tokens,
                },
                "semantic": {
                    "durationMs": semantic_duration,
                    "inferenceMs": semantic_warmup.attempt.inference_ms,
                    "selectedFrames": semantic_warmup.attempt.selected_frames,
                    "generatedTokens": semantic_warmup.attempt.generated_tokens,
                },
                "ocr": {
                    "durationMs": ocr_duration,
                    "inferenceMs": sum(a.inference_ms for a in ocr_warmup.attempts),
                    "selectedFrames": ocr_warmup.selected_frames,
                    "generatedTokens": sum(a.generated_tokens for a in ocr_warmup.attempts),
                },
            }

        results = []
        for media_path in media_paths:
            duration_ms = video_duration_ms(media_path)
            result = await run_visual_pipeline(
                runtime, media_path, duration_ms, settings, analysis_mode=mode,
            )
            video_entry = {
                "media": str(media_path.resolve()),
                "durationMs": duration_ms,
                "metrics": result.metrics.model_dump(),
                "validationWarnings": result.warnings,
                "consolidationDiagnostics": result.consolidated.diagnostics,
                "finalResult": {
                    "schemaVersion": settings.qwen_schema_version,
                    "summary": result.consolidated.summary,
                    "observations": result.consolidated.observations,
                    "visibleTexts": result.consolidated.visible_texts,
                    "communicatedIdeas": result.consolidated.communicated_ideas,
                    "themes": result.consolidated.themes,
                    "objects": result.consolidated.objects,
                    "topics": result.consolidated.topics,
                },
                "attempts": [asdict(attempt) for attempt in result.trace.attempts],
            }
            synthesis_attempts = [
                a for a in result.trace.attempts if a.pass_name == "grounded-synthesis"
            ]
            if synthesis_attempts:
                sa = synthesis_attempts[0]
                video_entry["groundedSynthesis"] = {
                    "rawOutput": sa.raw_output,
                    "inputTokens": sa.input_tokens,
                    "generatedTokens": sa.generated_tokens,
                    "inferenceMs": sa.inference_ms,
                    "finishReason": sa.finish_reason,
                }
            results.append(video_entry)

        kd = runtime.kernel_diagnostics
        fla_calls = (
            kd.flash_linear_attention_chunk_calls
            + kd.flash_linear_attention_recurrent_calls
        )
        if settings.qwen_require_fla and fla_calls == 0:
            raise RuntimeError(
                "FLA was required and selected, but the benchmark observed zero gated-delta calls"
            )
        report = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "system": {
                "platform": platform.platform(),
                "python": platform.python_version(),
                "device": runtime.device_name,
                "torch": runtime.torch_version,
                "hip": runtime.hip_version,
                "transformers": runtime.transformers_version,
                "qwenVlUtils": runtime.toolkit_version,
                "pyav": package_version("av"),
                "triton": package_version("triton"),
                "flashLinearAttention": package_version("flash-linear-attention"),
                "videoBackend": runtime.video_backend,
                "videoDecodeStrategy": "pyav-uniform",
                "wsl": bool(
                    os.getenv("WSL_INTEROP")
                    or "microsoft" in platform.release().lower()
                ),
            },
            "model": {
                "id": runtime.model_id,
                "revision": runtime.model_revision,
                "configuredDtype": runtime.configured_dtype,
                "actualDtypeDistribution": runtime.actual_dtype_distribution,
                "attentionImplementation": runtime.attention_implementation,
                "thinkingEnabled": runtime.thinking_enabled,
                "forceFlaBinding": settings.qwen_force_fla_binding,
                "requireFla": settings.qwen_require_fla,
                "loadMs": model_load_ms,
            },
            "contract": {
                "promptVersion": settings.qwen_prompt_version,
                "schemaVersion": settings.qwen_schema_version,
                "analysisMode": (mode or settings.analysis_mode).value,
                "groundedTextUnderstandingEnabled": settings.grounded_text_understanding_enabled,
            },
            "kernelDiagnostics": kd.to_dict(),
            "configuration": {
                "semanticWindowDurationMs": settings.semantic_window_duration_ms,
                "semanticWindowOverlapMs": settings.semantic_window_overlap_ms,
                "semanticFps": settings.semantic_fps,
                "semanticMaxFrames": settings.semantic_max_frames,
                "semanticMaxPixels": settings.semantic_max_pixels,
                "semanticFramePolicy": settings.semantic_frame_policy.value,
                "ocrFps": settings.ocr_fps,
                "ocrMaxFramesTotal": settings.ocr_max_frames_total,
                "ocrBatchSize": settings.ocr_batch_size,
                "ocrMaxPixels": settings.ocr_max_pixels,
                "globalOverviewMaxFrames": settings.global_overview_max_frames,
                "globalOverviewMaxPixels": settings.global_overview_max_pixels,
                "globalOverviewFramePolicy": settings.global_overview_frame_policy.value,
                "videoDecodeMaxAttempts": settings.video_decode_max_attempts,
                "videoDecodeRetryDelayMs": settings.video_decode_retry_delay_ms,
                "videoDecodeStrategy": "pyav-uniform",
                "videoFramePolicy": settings.semantic_frame_policy.value,
                "groundedSynthesisMaxNewTokens": settings.grounded_synthesis_max_new_tokens,
                "maxEvidenceReferencesPerSynthesisItem": settings.max_evidence_references_per_synthesis_item,
                "maxFinalCommunicatedIdeas": settings.max_final_communicated_ideas,
                "maxFinalThemes": settings.max_final_themes,
            },
            "warmup": warmup_record,
            "results": results,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Benchmark written to {args.output}")
    finally:
        await runtime.close()


if __name__ == "__main__":
    asyncio.run(run())
