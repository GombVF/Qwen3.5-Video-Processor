from __future__ import annotations

import errno
import gc
import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from visual_service.settings import FramePolicy
from visual_service.video.profiles import VideoAnalysisProfile

logger = logging.getLogger(__name__)


class VideoDecodeError(Exception):
    pass


def validate_video_backend_compatibility() -> str:
    """Validate the explicit decoder used by every semantic video pass.

    qwen-vl-utils remains responsible for converting an in-memory frame list
    into the model tensor, but it no longer chooses timestamps or opens the
    media file. This avoids backend-dependent sampling and deprecated
    torchvision video I/O.
    """

    try:
        import av
    except ImportError as exc:
        raise VideoDecodeError("PyAV is required for uniform video sampling") from exc
    if not getattr(av, "__version__", None):
        raise VideoDecodeError("PyAV did not expose a usable version")
    return "pyav-uniform"


@dataclass
class DecodedVideo:
    image_inputs: list[Any] | None
    video_inputs: list[Any]
    video_kwargs: dict[str, Any]
    selected_frames: int
    effective_fps: float
    width: int
    height: int
    backend: str
    decode_ms: int
    decode_attempts: int = 1
    decode_strategy: str = "pyav-uniform"


@dataclass(frozen=True)
class SampledVideoFrames:
    images: list[Image.Image]
    timestamps_ms: list[int]
    frame_indices: list[int]
    source_fps: float
    effective_fps: float
    total_frames: int
    decode_ms: int


@dataclass(frozen=True)
class OcrFrame:
    sequence: int
    timestamp_ms: int
    image: Image.Image


def build_video_content(
    media_path: Path,
    start_ms: int,
    end_ms: int,
    profile: VideoAnalysisProfile,
) -> dict[str, Any]:
    return {
        "type": "video",
        "video": str(media_path),
        "video_start": start_ms / 1000.0,
        "video_end": end_ms / 1000.0,
        "fps": profile.fps,
        "min_frames": profile.min_frames,
        "max_frames": profile.max_frames,
        "max_pixels": profile.max_pixels,
    }


def build_sampled_video_content(
    sample: SampledVideoFrames,
    profile: VideoAnalysisProfile,
) -> dict[str, Any]:
    """Build the qwen-vl-utils input without asking it to decode the file."""

    return {
        "type": "video",
        "video": sample.images,
        "sample_fps": sample.effective_fps,
        "fps": profile.fps,
        "min_frames": len(sample.images),
        "max_frames": len(sample.images),
        "max_pixels": profile.max_pixels,
    }


def plan_video_timestamps(
    start_ms: int,
    end_ms: int,
    profile: VideoAnalysisProfile,
) -> list[int]:
    """Plan uniformly spaced frame timestamps for one logical video window.

    Qwen3.5 groups video frames in temporal pairs. The frame count is therefore
    rounded to a multiple of two before applying the configured bounds. The
    midpoint of each bin is used so the first and final frames are not biased
    toward container boundaries.
    """

    duration_ms = end_ms - start_ms
    if duration_ms <= 0:
        return []
    maximum = max(2, _round_down_even(profile.max_frames))
    minimum = min(maximum, max(2, _round_up_even(profile.min_frames)))
    if profile.frame_policy == FramePolicy.FIXED_MAX:
        count = maximum
    else:
        desired = _round_nearest_even(duration_ms / 1000.0 * profile.fps)
        count = min(max(desired, minimum), maximum)
    step = duration_ms / count
    return [
        min(end_ms - 1, max(start_ms, int(start_ms + (index + 0.5) * step)))
        for index in range(count)
    ]


def decode_video_window(
    media_path: Path,
    start_ms: int,
    end_ms: int,
    profile: VideoAnalysisProfile,
    patch_size: int = 16,
    max_attempts: int = 2,
    retry_delay_ms: int = 250,
) -> DecodedVideo:
    from qwen_vl_utils import process_vision_info

    if max_attempts < 1:
        raise ValueError("max_attempts must be at least one")
    if retry_delay_ms < 0:
        raise ValueError("retry_delay_ms cannot be negative")

    started = time.perf_counter()
    raw_video_inputs: list[Any] | None = None
    image_inputs: list[Any] | None = None
    video_kwargs: dict[str, Any] | None = None
    sample: SampledVideoFrames | None = None
    decode_attempts = 0
    for decode_attempts in range(1, max_attempts + 1):
        try:
            sample = _sample_window_frames(media_path, start_ms, end_ms, profile)
            content = build_sampled_video_content(sample, profile)
            messages = [{"role": "user", "content": [content]}]
            try:
                image_inputs, raw_video_inputs, video_kwargs = process_vision_info(
                    messages,
                    image_patch_size=patch_size,
                    return_video_kwargs=True,
                    return_video_metadata=True,
                )
            except TypeError as exc:
                # Older compatible qwen-vl-utils builds may not expose
                # image_patch_size. Do not mask unrelated decoder TypeErrors.
                if "image_patch_size" not in str(exc):
                    raise
                image_inputs, raw_video_inputs, video_kwargs = process_vision_info(
                    messages, return_video_kwargs=True, return_video_metadata=True,
                )
            break
        except Exception as exc:
            if decode_attempts >= max_attempts or not _is_transient_decode_error(exc):
                raise VideoDecodeError(
                    f"Failed to decode {media_path} [{start_ms}ms-{end_ms}ms] "
                    f"after {decode_attempts} attempt(s): {exc}"
                ) from exc
            logger.warning("video_decode_retry", extra={"data": {
                "path": str(media_path),
                "startMs": start_ms,
                "endMs": end_ms,
                "attempt": decode_attempts,
                "maxAttempts": max_attempts,
                "error": str(exc),
            }})
            # Release decoder-side Python objects before retrying a transient
            # FFmpeg/swscale EAGAIN failure under WSL.
            image_inputs = None
            raw_video_inputs = None
            video_kwargs = None
            sample = None
            gc.collect()
            if retry_delay_ms:
                time.sleep(retry_delay_ms / 1000.0)
    decode_ms = int((time.perf_counter() - started) * 1000)

    if not raw_video_inputs or sample is None:
        raise VideoDecodeError(
            f"No video frames decoded from {media_path} [{start_ms}ms-{end_ms}ms]"
        )

    first = raw_video_inputs[0]
    if isinstance(first, tuple):
        tensor, _toolkit_metadata = first
    else:
        tensor = first

    selected_frames = int(tensor.shape[0])
    if selected_frames != len(sample.frame_indices):
        raise VideoDecodeError(
            "qwen-vl-utils changed the preselected frame count: "
            f"planned={len(sample.frame_indices)}, tensor={selected_frames}. "
            "The temporal metadata would be invalid."
        )
    height = int(tensor.shape[-2])
    width = int(tensor.shape[-1])
    effective_fps = sample.effective_fps
    metadata = {
        "fps": sample.source_fps,
        "frames_indices": sample.frame_indices,
        "total_num_frames": sample.total_frames,
        "video_backend": "pyav-uniform",
    }
    video_kwargs = dict(video_kwargs or {})
    # Preserve source timeline metadata for Qwen3.5 3D position construction.
    # qwen-vl-utils already resized the in-memory frames, so prevent a second
    # resize in AutoProcessor.
    video_kwargs["video_metadata"] = [metadata]
    video_kwargs["do_resize"] = False
    video_kwargs["cap_pixels_per_frame"] = False
    backend = "pyav-uniform"

    logger.info("video_window_decoded", extra={"data": {
        "path": str(media_path),
        "startMs": start_ms,
        "endMs": end_ms,
        "selectedFrames": selected_frames,
        "effectiveFps": effective_fps,
        "width": width,
        "height": height,
        "backend": backend,
        "decodeMs": decode_ms,
        "decodeAttempts": decode_attempts,
    }})
    return DecodedVideo(
        image_inputs=image_inputs,
        video_inputs=[tensor],
        video_kwargs=video_kwargs,
        selected_frames=selected_frames,
        effective_fps=effective_fps,
        width=width,
        height=height,
        backend=backend,
        decode_ms=decode_ms,
        decode_attempts=decode_attempts,
        decode_strategy="pyav-uniform",
    )


def _sample_window_frames(
    media_path: Path,
    start_ms: int,
    end_ms: int,
    profile: VideoAnalysisProfile,
) -> SampledVideoFrames:
    """Decode only the planned frames for a logical window and keep them in RAM."""

    import av

    targets = plan_video_timestamps(start_ms, end_ms, profile)
    if not targets:
        raise VideoDecodeError(
            f"Invalid video window for {media_path}: [{start_ms}ms-{end_ms}ms]"
        )
    started = time.perf_counter()
    container = av.open(str(media_path))
    images: list[Image.Image] = []
    selected_timestamps: list[int] = []
    selected_indices: list[int] = []
    last_image: Image.Image | None = None
    last_frame: Any = None
    last_timestamp_ms: int | None = None
    target_index = 0
    try:
        if not container.streams.video:
            raise VideoDecodeError(f"No video stream found in {media_path}")
        stream = container.streams.video[0]
        source_fps = _source_fps(stream)
        total_frames = _source_total_frames(container, stream, source_fps)
        origin_pts = int(getattr(stream, "start_time", 0) or 0)

        if start_ms > 0 and stream.time_base is not None:
            seek_offset = origin_pts + int((start_ms / 1000.0) / float(stream.time_base))
            container.seek(seek_offset, stream=stream, backward=True, any_frame=False)

        for frame in container.decode(stream):
            if frame.pts is None or stream.time_base is None:
                continue
            timestamp_ms = max(
                0,
                int(float((frame.pts - origin_pts) * stream.time_base) * 1000),
            )
            if timestamp_ms < start_ms:
                continue
            last_frame = frame
            if timestamp_ms >= end_ms:
                break
            last_image = None
            last_timestamp_ms = timestamp_ms
            while target_index < len(targets) and timestamp_ms >= targets[target_index]:
                if last_image is None:
                    last_image = frame.to_image()
                images.append(last_image)
                selected_timestamps.append(timestamp_ms)
                selected_indices.append(
                    min(
                        total_frames - 1,
                        max(0, round(timestamp_ms / 1000.0 * source_fps)),
                    )
                )
                target_index += 1
            if target_index >= len(targets):
                break

        # Sparse/corrupt tails can end before the last planned timestamps. A
        # repeated final observation is safer than changing the temporal shape
        # or manufacturing an intermediate frame.
        if target_index < len(targets):
            if last_image is None and images:
                last_image = images[-1]
                last_timestamp_ms = selected_timestamps[-1]
            if last_image is None and last_frame is not None:
                last_image = last_frame.to_image()
            if last_image is None or last_timestamp_ms is None:
                raise VideoDecodeError(
                    f"No frames decoded from {media_path} [{start_ms}ms-{end_ms}ms]"
                )
            while target_index < len(targets):
                images.append(last_image)
                selected_timestamps.append(last_timestamp_ms)
                selected_indices.append(
                    min(
                        total_frames - 1,
                        max(0, round(last_timestamp_ms / 1000.0 * source_fps)),
                    )
                )
                target_index += 1
    finally:
        container.close()

    return SampledVideoFrames(
        images=images,
        timestamps_ms=selected_timestamps,
        frame_indices=selected_indices,
        source_fps=source_fps,
        effective_fps=len(images) / max((end_ms - start_ms) / 1000.0, 0.001),
        total_frames=total_frames,
        decode_ms=int((time.perf_counter() - started) * 1000),
    )


def _source_fps(stream: Any) -> float:
    for value in (getattr(stream, "average_rate", None), getattr(stream, "base_rate", None)):
        if value is not None:
            fps = float(value)
            if math.isfinite(fps) and fps > 0:
                return fps
    return 30.0


def _source_total_frames(container: Any, stream: Any, source_fps: float) -> int:
    stream_frames = int(getattr(stream, "frames", 0) or 0)
    if stream_frames > 0:
        return stream_frames
    if getattr(stream, "duration", None) is not None and stream.time_base is not None:
        return max(1, math.ceil(float(stream.duration * stream.time_base) * source_fps))
    if getattr(container, "duration", None) is not None:
        return max(1, math.ceil(float(container.duration) / 1_000_000.0 * source_fps))
    return 1


def _round_up_even(value: int) -> int:
    return value if value % 2 == 0 else value + 1


def _round_down_even(value: int) -> int:
    return value if value % 2 == 0 else value - 1


def _round_nearest_even(value: float) -> int:
    return max(2, int(math.floor(value / 2.0 + 0.5)) * 2)


def _metadata_value(metadata: Any, name: str, default: Any) -> Any:
    if isinstance(metadata, dict):
        value = metadata.get(name, default)
    else:
        value = getattr(metadata, name, default)
    return default if value is None else value


def _is_transient_decode_error(exc: BaseException) -> bool:
    """Return True only for the retryable FFmpeg/swscale EAGAIN condition."""

    current: BaseException | None = exc
    while current is not None:
        if isinstance(current, BlockingIOError) or getattr(current, "errno", None) == errno.EAGAIN:
            return True
        message = str(current).lower()
        if "resource temporarily unavailable" in message or "errno 11" in message:
            return True
        current = current.__cause__ or current.__context__
    return False


def plan_ocr_timestamps(duration_ms: int, fps: float, max_frames: int) -> list[int]:
    if duration_ms <= 0 or fps <= 0 or max_frames <= 0:
        return []
    desired = max(1, math.ceil(duration_ms / 1000.0 * fps))
    count = min(desired, max_frames)
    step = duration_ms / count
    return [min(duration_ms - 1, max(0, int((index + 0.5) * step))) for index in range(count)]


def extract_ocr_frames(
    media_path: Path,
    *,
    duration_ms: int,
    fps: float,
    max_frames: int,
    max_pixels: int,
) -> tuple[list[OcrFrame], int]:
    import av

    targets = plan_ocr_timestamps(duration_ms, fps, max_frames)
    if not targets:
        return [], 0
    started = time.perf_counter()
    frames: list[OcrFrame] = []
    target_index = 0
    container = av.open(str(media_path))
    try:
        stream = container.streams.video[0]
        for frame in container.decode(video=0):
            if frame.pts is None:
                continue
            timestamp_ms = int(float(frame.pts * stream.time_base) * 1000)
            while target_index < len(targets) and timestamp_ms >= targets[target_index]:
                image = _resize_to_pixel_budget(frame.to_image(), max_pixels)
                frames.append(OcrFrame(target_index, targets[target_index], image))
                target_index += 1
            if target_index >= len(targets):
                break
    except Exception as exc:
        raise VideoDecodeError(f"Failed to extract OCR frames from {media_path}: {exc}") from exc
    finally:
        container.close()
    decode_ms = int((time.perf_counter() - started) * 1000)
    return frames, decode_ms


def _resize_to_pixel_budget(image: Image.Image, max_pixels: int) -> Image.Image:
    width, height = image.size
    if width * height <= max_pixels:
        return image
    scale = math.sqrt(max_pixels / (width * height))
    resized = (max(1, int(width * scale)), max(1, int(height * scale)))
    return image.resize(resized, Image.Resampling.LANCZOS)
