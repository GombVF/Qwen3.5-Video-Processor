import pytest

from visual_service.model.prompt import (
    SYNTHESIS_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    build_global_overview_prompt,
    build_grounded_synthesis_prompt,
    build_observation_prompt,
    build_ocr_content_prompt,
    build_ocr_prompt,
)
from visual_service.settings import PROMPT_VERSION


def test_prompt_version_is_v8():
    assert PROMPT_VERSION == "visual-analysis-qwen35-v8"


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


# --- Grounded synthesis prompt tests ---

def _synthesis_prompt():
    return build_grounded_synthesis_prompt(
        evidence_block="[text-0001] El\n[text-0002] problema",
        evidence_count=2,
        max_communicated_ideas=8,
        max_themes=8,
        max_evidence_references=6,
    )


def test_synthesis_prompt_contains_evidence_block():
    prompt = _synthesis_prompt()
    assert "[text-0001] El" in prompt
    assert "[text-0002] problema" in prompt


def test_synthesis_prompt_prohibits_visual_claims():
    prompt = _synthesis_prompt()
    assert "no access to images" in prompt


def test_synthesis_prompt_prohibits_audio_claims():
    prompt = _synthesis_prompt()
    assert "no access to audio" in prompt


def test_synthesis_prompt_requires_evidence_ids():
    prompt = _synthesis_prompt()
    assert "evidence" in prompt.lower()
    assert '"evidence"' in prompt


def test_synthesis_prompt_states_evidence_may_be_incomplete():
    prompt = _synthesis_prompt()
    assert "incomplete" in prompt.lower()
    assert "OCR errors" in prompt


def test_synthesis_prompt_prohibits_invention():
    prompt = _synthesis_prompt()
    assert "Do not invent" in prompt


def test_synthesis_prompt_prohibits_low_value_narration():
    prompt = _synthesis_prompt()
    assert "The text begins with" in prompt
    assert "The sequence ends with" in prompt
    assert "The word X appears" in prompt
    assert "The phrase translates to" in prompt


def test_synthesis_prompt_requests_global_ideas():
    prompt = _synthesis_prompt()
    assert "video-level" in prompt or "as a whole" in prompt


def test_synthesis_prompt_includes_configured_limits():
    prompt = build_grounded_synthesis_prompt(
        evidence_block="[text-0001] test",
        evidence_count=1,
        max_communicated_ideas=5,
        max_themes=3,
        max_evidence_references=4,
    )
    assert "5" in prompt
    assert "3" in prompt


def test_synthesis_prompt_requests_concise_ideas():
    prompt = _synthesis_prompt()
    assert "25 words" in prompt


def test_synthesis_prompt_requests_concise_themes():
    prompt = _synthesis_prompt()
    assert "5 words" in prompt


def test_synthesis_prompt_limits_evidence_references():
    prompt = _synthesis_prompt()
    assert "6" in prompt
    assert "evidence IDs per item" in prompt


def test_synthesis_prompt_requests_representative_evidence():
    prompt = _synthesis_prompt()
    assert "strongest" in prompt or "representative" in prompt


def test_synthesis_prompt_prohibits_exhaustive_citation():
    prompt = _synthesis_prompt()
    assert "Do not cite every" in prompt


def test_synthesis_prompt_allows_late_evidence():
    prompt = _synthesis_prompt()
    assert "late evidence" in prompt


def test_synthesis_prompt_invalid_evidence_refs_rejected():
    with pytest.raises(ValueError):
        build_grounded_synthesis_prompt(
            evidence_block="[text-0001] test",
            evidence_count=1,
            max_communicated_ideas=8,
            max_themes=8,
            max_evidence_references=-1,
        )


def test_synthesis_prompt_distinguishes_central_from_incidental():
    prompt = _synthesis_prompt()
    assert "incidental environmental text" in prompt
    assert "Do not promote" in prompt


def test_synthesis_prompt_excludes_incidental_examples():
    prompt = _synthesis_prompt()
    for term in ["addresses", "license plates", "codes", "street names", "storefront names"]:
        assert term in prompt


def test_synthesis_prompt_forbids_medium_themes():
    prompt = _synthesis_prompt()
    assert '"Environmental text"' in prompt
    assert '"OCR"' in prompt
    assert '"Captions"' in prompt
    assert '"Visible words"' in prompt
    assert "evidence medium" in prompt


def test_synthesis_prompt_describes_evidence_structure():
    prompt = _synthesis_prompt()
    assert "groups separated by blank lines" in prompt
    assert "close together" in prompt


def test_synthesis_prompt_cross_fragment_composition():
    prompt = _synthesis_prompt()
    assert "distant groups" in prompt


def test_synthesis_prompt_natural_wording():
    prompt = _synthesis_prompt()
    assert "redundant phrasing" in prompt


def test_synthesis_prompt_allows_identity_content():
    prompt = _synthesis_prompt()
    for prohibited_term in ["identity", "divinity", "philosophy", "introspection"]:
        assert prohibited_term not in prompt.split("FORBIDDEN")[1].split("OUTPUT")[0]


def test_synthesis_prompt_allows_sound_text_attribution():
    prompt = _synthesis_prompt()
    assert "On-screen text says" in prompt


def test_synthesis_system_prompt_is_text_only():
    assert "pixels" not in SYNTHESIS_SYSTEM_PROMPT
    assert "images" not in SYNTHESIS_SYSTEM_PROMPT.lower() or "no access to images" in SYNTHESIS_SYSTEM_PROMPT.lower()
    assert "textual-evidence" in SYNTHESIS_SYSTEM_PROMPT


def test_synthesis_system_prompt_prohibits_visual_audio():
    assert "visual content" in SYNTHESIS_SYSTEM_PROMPT.lower() or "visual information" in SYNTHESIS_SYSTEM_PROMPT.lower()
    assert "audio" in SYNTHESIS_SYSTEM_PROMPT.lower()


def test_synthesis_prompt_prohibits_visual_descriptions():
    prompt = _synthesis_prompt()
    assert "Do not describe objects" in prompt or "objects, people, environments" in prompt


def test_synthesis_prompt_sound_attribution():
    prompt = _synthesis_prompt()
    assert "On-screen text says" in prompt
    assert "The runner hears" in prompt


def test_synthesis_prompt_empty_evidence_rejected():
    with pytest.raises(ValueError):
        build_grounded_synthesis_prompt(
            evidence_block="",
            evidence_count=0,
            max_communicated_ideas=8,
            max_themes=8,
        )


def test_synthesis_prompt_invalid_limits_rejected():
    with pytest.raises(ValueError):
        build_grounded_synthesis_prompt(
            evidence_block="[text-0001] test",
            evidence_count=1,
            max_communicated_ideas=-1,
            max_themes=8,
        )


def test_synthesis_prompt_silent_omission():
    prompt = _synthesis_prompt()
    assert "it completely from communicatedIdeas and themes" in prompt


def test_synthesis_prompt_no_mention_omitted():
    prompt = _synthesis_prompt()
    assert "Do not mention, classify, summarize" in prompt
    assert "omitted incidental evidence" in prompt


def test_synthesis_prompt_maxima_are_ceilings():
    prompt = _synthesis_prompt()
    assert "Do not fill the configured maximum" in prompt


def test_synthesis_prompt_1_4_recommendation():
    prompt = _synthesis_prompt()
    assert "1-4" in prompt


def test_synthesis_prompt_minimum_evidence_set():
    prompt = _synthesis_prompt()
    assert "smallest representative evidence set" in prompt


def test_synthesis_prompt_self_referential():
    prompt = _synthesis_prompt()
    assert "self-referential constructions" in prompt


def test_synthesis_prompt_no_expand_names():
    prompt = _synthesis_prompt()
    assert "Do not expand partially visible proper names" in prompt
