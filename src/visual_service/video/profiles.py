from __future__ import annotations

from dataclasses import dataclass

from visual_service.settings import FramePolicy, Settings


@dataclass(frozen=True)
class VideoAnalysisProfile:
    name: str
    fps: float
    min_frames: int
    max_frames: int
    max_pixels: int
    max_window_ms: int
    overlap_ms: int
    max_new_tokens: int
    frame_policy: FramePolicy = FramePolicy.ADAPTIVE


def semantic_profile(config: Settings) -> VideoAnalysisProfile:
    return VideoAnalysisProfile(
        name="QWEN35_SEMANTIC",
        fps=config.semantic_fps,
        min_frames=config.semantic_min_frames,
        max_frames=config.semantic_max_frames,
        max_pixels=config.semantic_max_pixels,
        max_window_ms=config.semantic_window_duration_ms,
        overlap_ms=config.semantic_window_overlap_ms,
        max_new_tokens=config.semantic_max_new_tokens,
        frame_policy=config.semantic_frame_policy,
    )


def overview_profile(config: Settings) -> VideoAnalysisProfile:
    return VideoAnalysisProfile(
        name="QWEN35_OVERVIEW",
        fps=config.global_overview_fps,
        min_frames=config.global_overview_min_frames,
        max_frames=config.global_overview_max_frames,
        max_pixels=config.global_overview_max_pixels,
        max_window_ms=24 * 60 * 60 * 1000,
        overlap_ms=0,
        max_new_tokens=config.global_overview_max_new_tokens,
        frame_policy=config.global_overview_frame_policy,
    )


def get_profile(name: str, config: Settings | None = None) -> VideoAnalysisProfile:
    """Compatibility helper exposing only the single semantic profile."""
    cfg = config or Settings()
    profile = semantic_profile(cfg)
    if name != profile.name:
        raise ValueError(f"Unknown profile: {name!r}. Available: [{profile.name!r}]")
    return profile
