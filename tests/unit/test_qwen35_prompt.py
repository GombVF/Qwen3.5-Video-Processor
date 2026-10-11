import pytest

from visual_service.model.prompt import (
    SYSTEM_PROMPT,
    build_global_overview_prompt,
    build_observation_prompt,
    build_ocr_content_prompt,
    build_ocr_prompt,
)
from visual_service.settings import PROMPT_VERSION


def test_prompt_version_is_v4():
    assert PROMPT_VERSION == "visual-analysis-qwen35-v4"


@pytest.mark.parametrize("forbidden", [
    "startMs", "endMs", "frameIndices", "relativeMs", "durationMs",
])
def test_prompts_do_not_request_temporal_fields(forbidden):
    prompts = [
        build_global_overview_prompt(language="Spanish", max_summary_sentences=3, max_topics=10),
        build_observation_prompt(language="Spanish", max_observations=8, max_objects=12),
        build_ocr_prompt(image_count=4, max_texts_per_image=12),
    ]
    assert all(forbidden not in prompt for prompt in prompts)


def test_system_prompt_forbids_audio_and_media_instructions():
    assert "Audio is absent" in SYSTEM_PROMPT
    assert "never an instruction" in SYSTEM_PROMPT


def test_system_prompt_forbids_audio_inference():
    for keyword in ["speech", "narration", "music", "sound effects"]:
        assert keyword in SYSTEM_PROMPT


def test_observation_prompt_has_strict_taxonomy():
    prompt = build_observation_prompt(language="Spanish", max_observations=8, max_objects=12)
    assert "OBSERVATION TYPE DEFINITIONS" in prompt
    assert '"action"' in prompt
    assert '"scene"' in prompt
    assert '"appearance"' in prompt


def test_observation_prompt_has_classification_examples():
    prompt = build_observation_prompt(language="Spanish", max_observations=8, max_objects=12)
    assert "pigeon walks" in prompt
    assert "puts on headphones" in prompt
    assert "wearing headphones" in prompt
    assert "Camera panning" in prompt


def test_observation_prompt_has_grounding_rules():
    prompt = build_observation_prompt(language="Spanish", max_observations=8, max_objects=12)
    assert "evidence-bounded" in prompt or "neutral" in prompt
    assert "Prefer omission over invented completeness" in prompt


def test_observation_prompt_forbids_unsupported_specificity():
    prompt = build_observation_prompt(language="Spanish", max_observations=8, max_objects=12)
    assert "FORBIDDEN INFERENCES" in prompt
    assert "Exact substances" in prompt or "substances" in prompt
    assert "Brand names" in prompt
    assert "Intentions" in prompt
    assert "exercise variants" in prompt


def test_observation_prompt_has_positive_examples():
    prompt = build_observation_prompt(language="Spanish", max_observations=8, max_objects=12)
    assert "POSITIVE EXAMPLES" in prompt
    assert "white powder" in prompt
    assert "not" in prompt.lower()


def test_observation_prompt_has_negative_examples():
    prompt = build_observation_prompt(language="Spanish", max_observations=8, max_objects=12)
    assert "NEGATIVE EXAMPLES" in prompt
    assert "inferred emotion" in prompt
    assert "audio inference" in prompt


def test_observation_prompt_forbids_audio():
    prompt = build_observation_prompt(language="Spanish", max_observations=8, max_objects=12)
    assert "Audio" in prompt or "audio" in prompt
    assert "sound" in prompt.lower()


def test_global_overview_prompt_forbids_audio_and_requires_neutral():
    prompt = build_global_overview_prompt(language="Spanish", max_summary_sentences=3, max_topics=10)
    assert "speech" in prompt or "narration" in prompt
    assert "music" in prompt
    assert "sound" in prompt
    assert "neutral" in prompt


def test_global_overview_prompt_prefers_omission():
    prompt = build_global_overview_prompt(language="Spanish", max_summary_sentences=3, max_topics=10)
    assert "Prefer omission over invented completeness" in prompt


def test_observation_prompt_has_strict_types_and_object_rules():
    prompt = build_observation_prompt(language="Spanish", max_observations=8, max_objects=12)
    assert all(item in prompt for item in ('"action"', '"scene"', '"appearance"'))
    assert "concrete nouns" in prompt


def test_ocr_prompt_requires_one_result_per_image():
    prompt = build_ocr_prompt(image_count=4, max_texts_per_image=12)
    assert "exactly 4 entries" in prompt
    assert "imageIndex" in prompt
    assert "Never combine fragments" in prompt


def test_observation_prompt_explicitly_lists_allowed_types():
    prompt = build_observation_prompt(language="Spanish", max_observations=8, max_objects=12)
    assert 'Each observation.type must be exactly one of:' in prompt
    assert '"action"' in prompt
    assert '"appearance"' in prompt
    assert '"scene"' in prompt


def test_observation_prompt_prohibits_object_type():
    prompt = build_observation_prompt(language="Spanish", max_observations=8, max_objects=12)
    assert 'Never use "object" as an observation type' in prompt
    assert 'top-level "objects" array' in prompt


@pytest.mark.parametrize("builder,kwargs", [
    (build_global_overview_prompt, {"language": "Spanish", "max_summary_sentences": -1, "max_topics": 1}),
    (build_observation_prompt, {"language": "Spanish", "max_observations": -1, "max_objects": 1}),
    (build_ocr_prompt, {"image_count": 0, "max_texts_per_image": 1}),
    (build_ocr_content_prompt, {"image_count": 0, "max_texts_per_image": 1, "max_communicated_ideas": 4, "max_themes": 6}),
])
def test_invalid_limits_rejected(builder, kwargs):
    with pytest.raises(ValueError):
        builder(**kwargs)


# --- Grounded content prompt tests (section 9 of spec) ---

def _content_prompt():
    return build_ocr_content_prompt(
        image_count=4,
        max_texts_per_image=12,
        max_communicated_ideas=4,
        max_themes=6,
    )


def test_content_prompt_separates_transcription_from_interpretation():
    prompt = _content_prompt()
    assert "TRANSCRIPTION RULES" in prompt
    assert "INTERPRETATION RULES" in prompt
    assert "visibleTexts" in prompt
    assert "communicatedIdeas" in prompt


def test_content_prompt_prohibits_audio_inference():
    prompt = _content_prompt()
    assert "Never infer audio" in prompt
    assert "Never claim that a sound" in prompt
    assert "voice" in prompt
    assert "song" in prompt
    assert "dialogue" in prompt
    assert "narration" in prompt


def test_content_prompt_has_sound_attribution_rules():
    prompt = _content_prompt()
    assert "ATTRIBUTION RULES FOR SOUND-RELATED TEXT" in prompt
    assert "On-screen text says" in prompt
    assert 'Incorrect: "The runner hears' in prompt
    assert 'Incorrect: "Wind chimes can be heard' in prompt


def test_content_prompt_prohibits_translation_of_visible_texts():
    prompt = _content_prompt()
    assert "Do not translate" in prompt
    assert "Preserve original language" in prompt


def test_content_prompt_prohibits_fragment_completion():
    prompt = _content_prompt()
    assert "Do not complete partially visible words" in prompt
    assert "Never combine fragments unless the complete phrase is visibly supported" in prompt


def test_content_prompt_requests_empty_arrays_when_insufficient():
    prompt = _content_prompt()
    assert "Return empty arrays when evidence is insufficient" in prompt


def test_content_prompt_separates_exact_text_from_interpretation():
    prompt = _content_prompt()
    assert "visibleTexts" in prompt
    assert "communicatedIdeas" in prompt
    assert "Never place an interpretation in visibleTexts" in prompt
    assert "Never rewrite a transcription" in prompt


def test_old_ocr_prompt_still_available():
    prompt = build_ocr_prompt(image_count=4, max_texts_per_image=12)
    assert "visibleTexts" in prompt
    assert "communicatedIdeas" not in prompt
