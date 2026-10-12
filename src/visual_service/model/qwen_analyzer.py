from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from visual_service.model.evidence import EvidenceSet
from visual_service.model.prompt import (
    SYNTHESIS_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    build_global_overview_prompt,
    build_grounded_synthesis_prompt,
    build_observation_prompt,
    build_ocr_prompt,
)
from visual_service.settings import Settings
from visual_service.validation.output_validator import (
    ModelOutputInvalid,
    ModelOutputTruncated,
    salvage_truncated_output,
    validate_global_overview,
    validate_observations,
    validate_ocr_batch,
    validate_synthesis_output,
)
from visual_service.video.decoder import (
    build_video_content,
    decode_video_window,
    extract_ocr_frames,
)
from visual_service.video.profiles import VideoAnalysisProfile

logger = logging.getLogger(__name__)
_inference_lock = threading.Lock()


class InferenceError(Exception):
    pass


class WindowValidationFailed(Exception):
    """Raised when a window's validation fails but the inference attempt completed."""

    def __init__(self, message: str, attempt: "GenerationAttempt"):
        super().__init__(message)
        self.attempt = attempt


@dataclass
class GenerationAttempt:
    pass_name: str
    raw_output: str
    generated_tokens: int
    visual_tokens: int | None
    finish_reason: str
    decode_ms: int
    preprocess_ms: int
    inference_ms: int
    peak_vram_bytes: int
    selected_frames: int
    width: int
    height: int
    warnings: list[str] = field(default_factory=list)
    is_partial: bool = False
    decode_attempts: int = 1
    decode_strategy: str = "not-applicable"
    input_tokens: int | None = None
    evidence_item_count: int | None = None
    evidence_group_count: int | None = None
    evidence_group_sizes: list[int] | None = None
    evidence_largest_group_span: int | None = None


@dataclass
class GlobalOverviewResult:
    summary: str
    topics: list[str]
    attempt: GenerationAttempt


@dataclass
class WindowAnalysisResult:
    observations: list[Any]
    objects: list[str]
    attempt: GenerationAttempt


@dataclass
class OcrAnalysisResult:
    visible_texts: list[str]
    text_frame_sequences: list[int]
    attempts: list[GenerationAttempt]
    decode_ms: int
    selected_frames: int
    warnings: list[str] = field(default_factory=list)
    is_partial: bool = False


@dataclass
class SynthesisResult:
    communicated_ideas: list[str]
    themes: list[str]
    evidence_given: list[dict]
    raw_output: str
    validated_ideas: list[dict]
    validated_themes: list[dict]
    attempt: GenerationAttempt
    is_partial: bool = False


async def wait_for_inference_complete() -> None:
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, _inference_lock.acquire)
    _inference_lock.release()


async def analyze_global_overview(
    runtime: Any,
    media_path: Path,
    duration_ms: int,
    profile: VideoAnalysisProfile,
    config: Settings,
) -> GlobalOverviewResult:
    return await asyncio.to_thread(
        _locked_global_overview, runtime, media_path, duration_ms, profile, config,
    )


async def analyze_window(
    runtime: Any,
    media_path: Path,
    start_ms: int,
    end_ms: int,
    profile: VideoAnalysisProfile,
    config: Settings,
) -> WindowAnalysisResult:
    return await asyncio.to_thread(
        _locked_window, runtime, media_path, start_ms, end_ms, profile, config,
    )


async def analyze_ocr_video(
    runtime: Any,
    media_path: Path,
    duration_ms: int,
    config: Settings,
) -> OcrAnalysisResult:
    return await asyncio.to_thread(
        _locked_ocr_video, runtime, media_path, duration_ms, config,
    )


async def analyze_grounded_synthesis(
    runtime: Any,
    evidence_set: EvidenceSet,
    config: Settings,
) -> SynthesisResult:
    return await asyncio.to_thread(
        _locked_grounded_synthesis, runtime, evidence_set, config,
    )


def _locked_global_overview(
    runtime: Any,
    media_path: Path,
    duration_ms: int,
    profile: VideoAnalysisProfile,
    config: Settings,
) -> GlobalOverviewResult:
    with _inference_lock:
        prompt = build_global_overview_prompt(
            language=config.analysis_language,
            max_summary_sentences=config.max_summary_sentences,
            max_topics=config.max_global_topics,
        )
        attempt = _generate_video(
            runtime,
            media_path,
            0,
            duration_ms,
            profile,
            prompt,
            "global-overview",
            config,
        )
        validation, partial, warnings = _validate_with_salvage(
            attempt,
            lambda raw: validate_global_overview(raw, max_topics=config.max_global_topics),
        )
        attempt.is_partial = partial
        attempt.warnings.extend(warnings)
        return GlobalOverviewResult(
            summary=validation.output.summary,
            topics=validation.output.topics,
            attempt=attempt,
        )


def _locked_window(
    runtime: Any,
    media_path: Path,
    start_ms: int,
    end_ms: int,
    profile: VideoAnalysisProfile,
    config: Settings,
) -> WindowAnalysisResult:
    with _inference_lock:
        prompt = build_observation_prompt(
            language=config.analysis_language,
            max_observations=config.max_observations_per_window,
            max_objects=config.max_objects_per_window,
        )
        attempt = _generate_video(
            runtime,
            media_path,
            start_ms,
            end_ms,
            profile,
            prompt,
            "observations",
            config,
        )
        try:
            validation, partial, warnings = _validate_with_salvage(
                attempt,
                lambda raw: validate_observations(
                    raw,
                    max_observations=config.max_observations_per_window,
                    max_objects=config.max_objects_per_window,
                ),
            )
        except ModelOutputInvalid as exc:
            attempt.is_partial = True
            attempt.warnings.append(f"Validation failed: {exc}")
            raise WindowValidationFailed(str(exc), attempt) from exc
        attempt.is_partial = partial
        attempt.warnings.extend(warnings)
        if not validation.output.observations and validation.is_partial:
            raise WindowValidationFailed(
                "All observations discarded during item-level validation",
                attempt,
            )
        return WindowAnalysisResult(
            observations=validation.output.observations,
            objects=validation.output.objects,
            attempt=attempt,
        )


def _locked_ocr_video(
    runtime: Any,
    media_path: Path,
    duration_ms: int,
    config: Settings,
) -> OcrAnalysisResult:
    with _inference_lock:
        frames, decode_ms = extract_ocr_frames(
            media_path,
            duration_ms=duration_ms,
            fps=config.ocr_fps,
            max_frames=config.ocr_max_frames_total,
            max_pixels=config.ocr_max_pixels,
        )
        attempts: list[GenerationAttempt] = []
        visible_texts: list[str] = []
        text_frame_sequences: list[int] = []
        warnings: list[str] = []
        is_partial = False

        for batch_start in range(0, len(frames), config.ocr_batch_size):
            batch = frames[batch_start:batch_start + config.ocr_batch_size]
            prompt = build_ocr_prompt(
                image_count=len(batch),
                max_texts_per_image=config.max_visible_texts_per_image,
            )
            attempt = _generate_images(
                runtime,
                [item.image for item in batch],
                prompt,
                config.ocr_max_new_tokens,
                pass_name=f"ocr-{batch_start // config.ocr_batch_size}",
            )
            attempts.append(attempt)
            try:
                validation, partial, attempt_warnings = _validate_ocr_with_salvage(
                    attempt,
                    expected_image_count=len(batch),
                    max_texts_per_image=config.max_visible_texts_per_image,
                )
            except ModelOutputInvalid as exc:
                attempt.is_partial = True
                attempt.warnings.append(f"OCR batch rejected: {exc}")
                warnings.extend(attempt.warnings)
                is_partial = True
                continue
            attempt.is_partial = partial
            attempt.warnings.extend(attempt_warnings)
            warnings.extend(attempt.warnings)
            is_partial = is_partial or partial
            for image in validation.output.images:
                global_frame_seq = batch[image.imageIndex].sequence
                for text in image.visibleTexts:
                    visible_texts.append(text)
                    text_frame_sequences.append(global_frame_seq)

        return OcrAnalysisResult(
            visible_texts=visible_texts,
            text_frame_sequences=text_frame_sequences,
            attempts=attempts,
            decode_ms=decode_ms,
            selected_frames=len(frames),
            warnings=warnings,
            is_partial=is_partial,
        )


def _locked_grounded_synthesis(
    runtime: Any,
    evidence_set: EvidenceSet,
    config: Settings,
) -> SynthesisResult:
    with _inference_lock:
        prompt = build_grounded_synthesis_prompt(
            evidence_block=evidence_set.evidence_block,
            evidence_count=len(evidence_set.items),
            max_communicated_ideas=config.max_final_communicated_ideas,
            max_themes=config.max_final_themes,
            max_evidence_references=config.max_evidence_references_per_synthesis_item,
        )
        messages = [
            {"role": "system", "content": SYNTHESIS_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        attempt = _generate_text_only(
            runtime,
            messages,
            max_new_tokens=config.grounded_synthesis_max_new_tokens,
            pass_name="grounded-synthesis",
        )
        attempt.evidence_item_count = len(evidence_set.items)
        attempt.evidence_group_count = evidence_set.group_count
        if evidence_set.groups:
            attempt.evidence_group_sizes = [g.item_count for g in evidence_set.groups]
            attempt.evidence_largest_group_span = max(g.span for g in evidence_set.groups)

        evidence_given = [
            {"id": item.id, "text": item.text}
            for item in evidence_set.items
        ]

        try:
            validation, partial, warnings = _validate_synthesis_with_salvage(
                attempt,
                valid_evidence_ids=evidence_set.id_set,
                max_communicated_ideas=config.max_final_communicated_ideas,
                max_themes=config.max_final_themes,
                max_evidence_references_per_item=config.max_evidence_references_per_synthesis_item,
            )
            attempt.is_partial = partial
            attempt.warnings.extend(warnings)

            public_ideas = [item.idea for item in validation.output.communicatedIdeas]
            public_themes = [item.theme for item in validation.output.themes]
            validated_ideas = [
                {"idea": item.idea, "evidence": item.evidence}
                for item in validation.output.communicatedIdeas
            ]
            validated_themes = [
                {"theme": item.theme, "evidence": item.evidence}
                for item in validation.output.themes
            ]
            return SynthesisResult(
                communicated_ideas=public_ideas,
                themes=public_themes,
                evidence_given=evidence_given,
                raw_output=attempt.raw_output,
                validated_ideas=validated_ideas,
                validated_themes=validated_themes,
                attempt=attempt,
                is_partial=partial,
            )
        except ModelOutputInvalid as exc:
            attempt.is_partial = attempt.finish_reason == "length"
            attempt.warnings.append(str(exc))
            return SynthesisResult(
                communicated_ideas=[],
                themes=[],
                evidence_given=evidence_given,
                raw_output=attempt.raw_output,
                validated_ideas=[],
                validated_themes=[],
                attempt=attempt,
                is_partial=attempt.is_partial,
            )


def _generate_video(
    runtime: Any,
    media_path: Path,
    start_ms: int,
    end_ms: int,
    profile: VideoAnalysisProfile,
    prompt: str,
    pass_name: str,
    config: Settings,
) -> GenerationAttempt:
    patch_size = int(getattr(runtime.processor.image_processor, "patch_size", 16))
    decoded = decode_video_window(
        media_path,
        start_ms,
        end_ms,
        profile,
        patch_size=patch_size,
        max_attempts=config.video_decode_max_attempts,
        retry_delay_ms=config.video_decode_retry_delay_ms,
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                build_video_content(media_path, start_ms, end_ms, profile),
                {"type": "text", "text": prompt},
            ],
        },
    ]
    attempt = _generate(
        runtime,
        messages,
        max_new_tokens=profile.max_new_tokens,
        pass_name=pass_name,
        image_inputs=decoded.image_inputs,
        video_inputs=decoded.video_inputs,
        media_kwargs=decoded.video_kwargs,
    )
    attempt.decode_ms = decoded.decode_ms
    attempt.selected_frames = decoded.selected_frames
    attempt.width = decoded.width
    attempt.height = decoded.height
    attempt.decode_attempts = decoded.decode_attempts
    attempt.decode_strategy = decoded.decode_strategy
    return attempt


def _generate_images(
    runtime: Any,
    images: list[Any],
    prompt: str,
    max_new_tokens: int,
    pass_name: str,
) -> GenerationAttempt:
    from qwen_vl_utils import process_vision_info

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                *[{"type": "image", "image": image} for image in images],
                {"type": "text", "text": prompt},
            ],
        },
    ]
    started = time.perf_counter()
    image_inputs, video_inputs = process_vision_info(messages)
    decode_ms = int((time.perf_counter() - started) * 1000)
    attempt = _generate(
        runtime,
        messages,
        max_new_tokens=max_new_tokens,
        pass_name=pass_name,
        image_inputs=image_inputs,
        video_inputs=video_inputs,
        media_kwargs={},
    )
    attempt.decode_ms = decode_ms
    attempt.selected_frames = len(images)
    if images:
        attempt.width, attempt.height = images[0].size
    return attempt


def _generate_text_only(
    runtime: Any,
    messages: list[dict[str, Any]],
    *,
    max_new_tokens: int,
    pass_name: str,
) -> GenerationAttempt:
    import torch

    if runtime.thinking_enabled:
        raise InferenceError("Thinking mode must be disabled for structured extraction")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(0)

    preprocess_started = time.perf_counter()
    prompt_text = runtime.processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = runtime.processor(
        text=[prompt_text],
        images=None,
        videos=None,
        padding=True,
        return_tensors="pt",
    ).to(runtime.model.device)
    preprocess_ms = int((time.perf_counter() - preprocess_started) * 1000)
    input_tokens = int(inputs.input_ids.shape[1])

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    inference_started = time.perf_counter()
    try:
        with torch.inference_mode():
            output_ids = runtime.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=runtime.processor.tokenizer.eos_token_id,
            )
    except Exception as exc:
        raise InferenceError(f"{pass_name} inference failed: {exc}") from exc
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    inference_ms = int((time.perf_counter() - inference_started) * 1000)

    generated_ids = output_ids[:, inputs.input_ids.shape[1]:]
    generated_tokens = int(generated_ids.shape[1])
    raw_output = runtime.processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    finish_reason = "length" if generated_tokens >= max_new_tokens else "stop"
    peak_vram = int(torch.cuda.max_memory_allocated(0)) if torch.cuda.is_available() else 0

    del inputs, output_ids, generated_ids
    attempt = GenerationAttempt(
        pass_name=pass_name,
        raw_output=raw_output,
        generated_tokens=generated_tokens,
        visual_tokens=None,
        finish_reason=finish_reason,
        decode_ms=0,
        preprocess_ms=preprocess_ms,
        inference_ms=inference_ms,
        peak_vram_bytes=peak_vram,
        selected_frames=0,
        width=0,
        height=0,
    )
    attempt.input_tokens = input_tokens
    return attempt


def _generate(
    runtime: Any,
    messages: list[dict[str, Any]],
    *,
    max_new_tokens: int,
    pass_name: str,
    image_inputs: Any,
    video_inputs: Any,
    media_kwargs: dict[str, Any],
) -> GenerationAttempt:
    import torch

    if runtime.thinking_enabled:
        raise InferenceError("Thinking mode must be disabled for structured extraction")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(0)

    preprocess_started = time.perf_counter()
    prompt_text = runtime.processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = runtime.processor(
        text=[prompt_text],
        images=image_inputs,
        videos=video_inputs,
        **media_kwargs,
        padding=True,
        return_tensors="pt",
    ).to(runtime.model.device)
    preprocess_ms = int((time.perf_counter() - preprocess_started) * 1000)
    visual_tokens = _count_visual_tokens(inputs, runtime.processor)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    inference_started = time.perf_counter()
    try:
        with torch.inference_mode():
            output_ids = runtime.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=runtime.processor.tokenizer.eos_token_id,
            )
    except Exception as exc:
        raise InferenceError(f"{pass_name} inference failed: {exc}") from exc
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    inference_ms = int((time.perf_counter() - inference_started) * 1000)

    generated_ids = output_ids[:, inputs.input_ids.shape[1]:]
    generated_tokens = int(generated_ids.shape[1])
    raw_output = runtime.processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    finish_reason = "length" if generated_tokens >= max_new_tokens else "stop"
    peak_vram = int(torch.cuda.max_memory_allocated(0)) if torch.cuda.is_available() else 0

    del inputs, output_ids, generated_ids
    return GenerationAttempt(
        pass_name=pass_name,
        raw_output=raw_output,
        generated_tokens=generated_tokens,
        visual_tokens=visual_tokens,
        finish_reason=finish_reason,
        decode_ms=0,
        preprocess_ms=preprocess_ms,
        inference_ms=inference_ms,
        peak_vram_bytes=peak_vram,
        selected_frames=0,
        width=0,
        height=0,
    )


def _count_visual_tokens(inputs: Any, processor: Any) -> int | None:
    merge_size = int(getattr(processor.image_processor, "merge_size", 2))
    total = 0
    found = False
    for name in ("image_grid_thw", "video_grid_thw"):
        grid = inputs.get(name) if hasattr(inputs, "get") else getattr(inputs, name, None)
        if grid is None:
            continue
        found = True
        for row in grid:
            total += int(row.prod().item()) // (merge_size * merge_size)
    return total if found else None


def _validate_with_salvage(
    attempt: GenerationAttempt,
    validator: Callable[[str], Any],
) -> tuple[Any, bool, list[str]]:
    warnings: list[str] = []
    try:
        if attempt.finish_reason == "length":
            raise ModelOutputTruncated(
                f"{attempt.pass_name} reached its token budget",
                attempt.raw_output,
                attempt.generated_tokens,
            )
        result = validator(attempt.raw_output)
        warnings.extend(result.warnings)
        return result, result.is_partial, warnings
    except ModelOutputInvalid as original_error:
        salvage = salvage_truncated_output(attempt.raw_output)
        if salvage is None:
            raise
        try:
            result = validator(json.dumps(salvage.data, ensure_ascii=False))
        except ModelOutputInvalid as salvage_error:
            raise ModelOutputInvalid(
                f"{attempt.pass_name} invalid and deterministic salvage failed. "
                f"Original: {original_error}; salvage: {salvage_error}"
            ) from original_error
        warnings.append(
            f"{attempt.pass_name}: deterministic structural salvage used after: "
            f"{original_error}"
        )
        warnings.extend(salvage.diagnostics)
        warnings.extend(result.warnings)
        return result, True, warnings


def _validate_ocr_with_salvage(
    attempt: GenerationAttempt,
    *,
    expected_image_count: int,
    max_texts_per_image: int,
) -> tuple[Any, bool, list[str]]:
    try:
        return _validate_with_salvage(
            attempt,
            lambda raw: validate_ocr_batch(
                raw,
                image_count=expected_image_count,
                max_texts_per_image=max_texts_per_image,
            ),
        )
    except ModelOutputInvalid as original_error:
        salvage = salvage_truncated_output(attempt.raw_output)
        images = salvage.data.get("images") if salvage is not None else None
        if not isinstance(images, list) or not images:
            raise
        recovered_count = len(images)
        try:
            result = validate_ocr_batch(
                json.dumps(salvage.data, ensure_ascii=False),
                image_count=recovered_count,
                max_texts_per_image=max_texts_per_image,
            )
        except ModelOutputInvalid:
            raise original_error
        warnings = [
            f"{attempt.pass_name}: recovered {recovered_count} of "
            f"{expected_image_count} OCR image results after structural truncation"
        ]
        warnings.extend(salvage.diagnostics)
        warnings.extend(result.warnings)
        return result, True, warnings


def _validate_synthesis_with_salvage(
    attempt: GenerationAttempt,
    *,
    valid_evidence_ids: frozenset[str],
    max_communicated_ideas: int,
    max_themes: int,
    max_evidence_references_per_item: int | None = None,
) -> tuple[Any, bool, list[str]]:
    return _validate_with_salvage(
        attempt,
        lambda raw: validate_synthesis_output(
            raw,
            valid_evidence_ids=valid_evidence_ids,
            max_communicated_ideas=max_communicated_ideas,
            max_themes=max_themes,
            max_evidence_references_per_item=max_evidence_references_per_item,
        ),
    )
