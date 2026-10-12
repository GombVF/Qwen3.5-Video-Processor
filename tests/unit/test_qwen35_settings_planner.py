import pytest

from visual_service.settings import (
    AnalysisMode,
    FramePolicy,
    PROMPT_VERSION,
    SCHEMA_VERSION,
    Settings,
    is_wsl,
    windows_path_to_wsl,
)
from visual_service.video.decoder import plan_ocr_timestamps, plan_video_timestamps
from visual_service.video.planner import create_windows
from visual_service.video.profiles import overview_profile, semantic_profile


def test_single_version_defaults():
    config = Settings()
    assert config.qwen_prompt_version == PROMPT_VERSION
    assert config.qwen_schema_version == SCHEMA_VERSION
    assert config.qwen_model_id.endswith("Qwen3.5-4B")
    assert config.qwen_dtype == "bfloat16"
    assert config.qwen_enable_thinking is False
    assert config.qwen_max_concurrency == 1
    assert config.qwen_force_fla_binding is is_wsl()
    assert config.qwen_require_fla is is_wsl()
    assert config.semantic_frame_policy == FramePolicy.FIXED_MAX
    assert config.global_overview_frame_policy == FramePolicy.FIXED_MAX


def test_windows_paths_can_be_translated_for_wsl():
    assert windows_path_to_wsl("E:/Trabajo/Qwen/model") == "/mnt/e/Trabajo/Qwen/model"
    assert windows_path_to_wsl(r"C:\models\qwen") == "/mnt/c/models/qwen"
    assert windows_path_to_wsl("Qwen/Qwen3.5-4B") == "Qwen/Qwen3.5-4B"


def test_default_analysis_mode():
    config = Settings()
    assert config.analysis_mode == AnalysisMode.SEMANTIC_AND_OCR


def test_analysis_mode_semantic_only():
    config = Settings(analysis_mode="SEMANTIC_ONLY")
    assert config.analysis_mode == AnalysisMode.SEMANTIC_ONLY


def test_analysis_mode_semantic_and_ocr():
    config = Settings(analysis_mode="SEMANTIC_AND_OCR")
    assert config.analysis_mode == AnalysisMode.SEMANTIC_AND_OCR


def test_unsupported_analysis_mode_rejected():
    with pytest.raises(ValueError, match="Unsupported analysis mode"):
        Settings(analysis_mode="INVALID_MODE")


def test_unsupported_version_is_rejected():
    with pytest.raises(ValueError):
        Settings(qwen_prompt_version="visual-analysis-v5")


def test_v2_prompt_version_still_accepted():
    config = Settings(qwen_prompt_version="visual-analysis-qwen35-v2")
    assert config.qwen_prompt_version == "visual-analysis-qwen35-v2"


def test_v3_prompt_version_still_accepted():
    config = Settings(qwen_prompt_version="visual-analysis-qwen35-v3")
    assert config.qwen_prompt_version == "visual-analysis-qwen35-v3"


def test_v4_prompt_version_still_accepted():
    config = Settings(qwen_prompt_version="visual-analysis-qwen35-v4")
    assert config.qwen_prompt_version == "visual-analysis-qwen35-v4"


def test_v5_prompt_version_still_accepted():
    config = Settings(qwen_prompt_version="visual-analysis-qwen35-v5")
    assert config.qwen_prompt_version == "visual-analysis-qwen35-v5"


def test_v6_prompt_version_still_accepted():
    config = Settings(qwen_prompt_version="visual-analysis-qwen35-v6")
    assert config.qwen_prompt_version == "visual-analysis-qwen35-v6"


def test_v7_prompt_version_still_accepted():
    config = Settings(qwen_prompt_version="visual-analysis-qwen35-v7")
    assert config.qwen_prompt_version == "visual-analysis-qwen35-v7"


def test_grounded_text_understanding_default():
    config = Settings()
    assert config.grounded_text_understanding_enabled is True


def test_grounded_text_understanding_disabled():
    config = Settings(grounded_text_understanding_enabled=False)
    assert config.grounded_text_understanding_enabled is False


def test_grounded_text_limits_configurable():
    config = Settings(
        max_final_communicated_ideas=5,
        max_final_themes=3,
    )
    assert config.max_final_communicated_ideas == 5
    assert config.max_final_themes == 3


def test_grounded_text_limits_defaults():
    config = Settings()
    assert config.max_final_communicated_ideas == 6
    assert config.max_final_themes == 6


def test_grounded_synthesis_max_new_tokens_configurable():
    config = Settings()
    assert config.grounded_synthesis_max_new_tokens == 1024
    config2 = Settings(grounded_synthesis_max_new_tokens=256)
    assert config2.grounded_synthesis_max_new_tokens == 256


def test_evidence_references_per_item_configurable():
    config = Settings()
    assert config.max_evidence_references_per_synthesis_item == 6
    config2 = Settings(max_evidence_references_per_synthesis_item=3)
    assert config2.max_evidence_references_per_synthesis_item == 3


def test_invalid_synthesis_budget_rejected():
    with pytest.raises(ValueError, match="Grounded synthesis max_new_tokens must be positive"):
        config = Settings(grounded_synthesis_max_new_tokens=0)
        config.validate_runtime_budget()


def test_invalid_evidence_references_rejected():
    with pytest.raises(ValueError, match="Max evidence references per synthesis item must be positive"):
        config = Settings(max_evidence_references_per_synthesis_item=0)
        config.validate_runtime_budget()


def test_thinking_cannot_be_enabled():
    with pytest.raises(ValueError):
        Settings(qwen_enable_thinking=True)


def test_windows_use_full_12_second_budget_with_overlap():
    windows, strategy = create_windows(27_000, semantic_profile(Settings()))
    assert strategy == "CHUNKED_NATIVE"
    assert [(w.start_ms, w.end_ms) for w in windows] == [
        (0, 12_000), (10_000, 22_000), (20_000, 27_000),
    ]


def test_ocr_plan_has_no_overlap_duplicates_and_respects_cap():
    timestamps = plan_ocr_timestamps(65_000, 1.0, 120)
    assert len(timestamps) == 65
    assert len(set(timestamps)) == 65
    capped = plan_ocr_timestamps(300_000, 1.0, 120)
    assert len(capped) == 120


def test_fragment_neighborhood_configurable():
    config = Settings()
    assert config.consolidation_fragment_neighborhood == 3
    config2 = Settings(consolidation_fragment_neighborhood=5)
    assert config2.consolidation_fragment_neighborhood == 5


# --- Frame policy ---

def test_adaptive_preserves_current_frame_count_behavior():
    profile = semantic_profile(Settings(semantic_frame_policy="ADAPTIVE"))
    assert profile.frame_policy == FramePolicy.ADAPTIVE
    timestamps = plan_video_timestamps(0, 12_000, profile)
    assert len(timestamps) == 24

    timestamps_short = plan_video_timestamps(0, 3_000, profile)
    assert len(timestamps_short) < 24


def test_fixed_max_selects_exactly_max_frames_for_12_second_video():
    profile = semantic_profile(Settings(semantic_frame_policy="FIXED_MAX"))
    timestamps = plan_video_timestamps(0, 12_000, profile)
    assert len(timestamps) == 24


def test_fixed_max_selects_exactly_max_frames_for_27_second_overview():
    profile = overview_profile(Settings(global_overview_frame_policy="FIXED_MAX"))
    timestamps = plan_video_timestamps(0, 27_000, profile)
    assert len(timestamps) == 24


def test_fixed_max_selects_exactly_max_frames_for_5_3_second_window():
    profile = semantic_profile(Settings(semantic_frame_policy="FIXED_MAX"))
    timestamps = plan_video_timestamps(0, 5_300, profile)
    assert len(timestamps) == 24


def test_fixed_max_selects_exactly_max_frames_for_65_second_overview():
    profile = overview_profile(Settings(global_overview_frame_policy="FIXED_MAX"))
    timestamps = plan_video_timestamps(0, 65_000, profile)
    assert len(timestamps) == 24


def test_fixed_max_distributes_timestamps_throughout_interval():
    profile = semantic_profile(Settings(semantic_frame_policy="FIXED_MAX"))
    timestamps = plan_video_timestamps(0, 5_300, profile)
    assert len(timestamps) == 24
    assert timestamps == sorted(timestamps)
    assert timestamps[0] < 300
    assert timestamps[-1] > 5_000
    assert timestamps[-1] < 5_300


def test_fixed_max_on_semantic_window_within_longer_video():
    profile = semantic_profile(Settings(semantic_frame_policy="FIXED_MAX"))
    timestamps = plan_video_timestamps(50_000, 55_300, profile)
    assert len(timestamps) == 24
    assert all(50_000 <= t < 55_300 for t in timestamps)


def test_frame_policy_configurable_via_string():
    config = Settings(semantic_frame_policy="ADAPTIVE")
    assert config.semantic_frame_policy == FramePolicy.ADAPTIVE
    config2 = Settings(semantic_frame_policy="FIXED_MAX")
    assert config2.semantic_frame_policy == FramePolicy.FIXED_MAX
