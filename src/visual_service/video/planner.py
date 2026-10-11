from __future__ import annotations

from dataclasses import dataclass

from visual_service.video.profiles import VideoAnalysisProfile


@dataclass(frozen=True)
class Window:
    sequence: int
    start_ms: int
    end_ms: int


def create_windows(
    duration_ms: int,
    profile: VideoAnalysisProfile,
) -> tuple[list[Window], str]:
    if duration_ms <= 0:
        raise ValueError(f"duration_ms must be positive, got {duration_ms}")
    if profile.overlap_ms >= profile.max_window_ms:
        raise ValueError(
            f"overlap_ms ({profile.overlap_ms}) must be less than "
            f"max_window_ms ({profile.max_window_ms})"
        )

    if duration_ms <= profile.max_window_ms:
        return [Window(sequence=0, start_ms=0, end_ms=duration_ms)], "FULL_NATIVE"

    windows: list[Window] = []
    start = 0
    sequence = 0
    chunk_ms = profile.max_window_ms

    while start < duration_ms:
        end = min(start + chunk_ms, duration_ms)
        windows.append(Window(sequence=sequence, start_ms=start, end_ms=end))

        if end == duration_ms:
            break

        start = end - profile.overlap_ms
        sequence += 1

    return windows, "CHUNKED_NATIVE"
