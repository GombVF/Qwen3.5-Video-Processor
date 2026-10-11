from pathlib import Path

import pytest

from visual_service.settings import Settings
from visual_service.video.decoder import _sample_window_frames
from visual_service.video.profiles import overview_profile, semantic_profile


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LONG_VIDEO = PROJECT_ROOT / "data" / "test_videos" / "video_65_sec.mp4"


@pytest.mark.skipif(not LONG_VIDEO.is_file(), reason="benchmark media is not present")
def test_global_overview_samples_the_entire_long_video_in_memory():
    sample = _sample_window_frames(
        LONG_VIDEO,
        0,
        65_300,
        overview_profile(Settings()),
    )

    assert len(sample.images) == 24
    assert len(sample.timestamps_ms) == len(sample.frame_indices) == 24
    assert sample.timestamps_ms[0] < 3_000
    assert sample.timestamps_ms[-1] > 62_000
    assert all(image.width > 0 and image.height > 0 for image in sample.images)


@pytest.mark.skipif(not LONG_VIDEO.is_file(), reason="benchmark media is not present")
def test_semantic_sampler_seeks_to_a_logical_window_without_writing_a_clip(tmp_path):
    sample = _sample_window_frames(
        LONG_VIDEO,
        40_000,
        52_000,
        semantic_profile(Settings()),
    )

    assert len(sample.images) == 24
    assert sample.timestamps_ms[0] >= 40_000
    assert sample.timestamps_ms[-1] < 52_000
    assert list(tmp_path.iterdir()) == []
