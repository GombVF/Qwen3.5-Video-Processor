import json

import pytest

from visual_service.validation.output_validator import (
    ModelOutputInvalid,
    salvage_truncated_output,
    validate_global_overview,
    validate_observations,
    validate_ocr_batch,
    validate_ocr_content_batch,
)


def test_global_overview_validates_and_deduplicates_topics():
    result = validate_global_overview(
        '{"summary":"Una persona cocina.","topics":["cocina","Cocina"]}',
        max_topics=10,
    )
    assert result.output.topics == ["cocina"]


def test_observation_order_is_normalized():
    raw = json.dumps({
        "observations": [
            {"order": 9, "type": "action", "description": "La persona vierte líquido."},
            {"order": 3, "type": "scene", "description": "Hay una cocina."},
        ],
        "objects": ["tazón"],
    })
    result = validate_observations(raw, max_observations=8, max_objects=12)
    assert [item.order for item in result.output.observations] == [1, 2]


def test_invalid_observation_type_is_discarded_not_rejected():
    raw = '{"observations":[{"order":1,"type":"exercise","description":"x"}],"objects":[]}'
    result = validate_observations(raw, max_observations=8, max_objects=12)
    assert len(result.output.observations) == 0
    assert result.is_partial is True
    assert any("exercise" in w for w in result.warnings)


def test_valid_observation_types_accepted():
    for obs_type in ("action", "scene", "appearance"):
        raw = json.dumps({
            "observations": [{"order": 1, "type": obs_type, "description": "test"}],
            "objects": [],
        })
        result = validate_observations(raw, max_observations=8, max_objects=12)
        assert result.output.observations[0].type == obs_type


def test_observation_type_case_insensitive():
    raw = '{"observations":[{"order":1,"type":"ACTION","description":"x"}],"objects":[]}'
    result = validate_observations(raw, max_observations=8, max_objects=12)
    assert result.output.observations[0].type == "action"


@pytest.mark.parametrize("field", ["startMs", "endMs", "frameIndices", "timestamp"])
def test_temporal_fields_are_rejected_recursively(field):
    raw = json.dumps({
        "observations": [{
            "order": 1, "type": "scene", "description": "x", field: 10,
        }],
        "objects": [],
    })
    with pytest.raises(ModelOutputInvalid, match="Temporal field"):
        validate_observations(raw, max_observations=8, max_objects=12)


def test_ocr_canonical_batch():
    raw = json.dumps({"images": [
        {"imageIndex": 0, "visibleTexts": ["HELLO"]},
        {"imageIndex": 1, "visibleTexts": []},
    ]})
    result = validate_ocr_batch(raw, image_count=2, max_texts_per_image=12)
    assert result.output.images[0].visibleTexts == ["HELLO"]


def test_ocr_bare_array_regression_from_v5_is_recovered():
    result = validate_ocr_batch('["HELLO","WORLD"]', image_count=1, max_texts_per_image=12)
    assert result.output.images[0].visibleTexts == ["HELLO", "WORLD"]
    assert any("bare OCR string array" in warning for warning in result.warnings)


def test_ocr_wrong_image_count_rejected():
    raw = '{"images":[{"imageIndex":0,"visibleTexts":[]}]}'
    with pytest.raises(ModelOutputInvalid, match="expected 2"):
        validate_ocr_batch(raw, image_count=2, max_texts_per_image=12)


def test_markdown_fence_is_tolerated_but_reported():
    raw = '```json\n{"summary":"x","topics":[]}\n```'
    result = validate_global_overview(raw, max_topics=10)
    assert any("code fence" in warning for warning in result.warnings)


def test_forward_salvage_never_closes_unfinished_string():
    raw = '{"observations":[{"order":1,"type":"action","description":"complete"}],"objects":["bowl","unfin'
    salvage = salvage_truncated_output(raw)
    assert salvage is not None
    serialized = json.dumps(salvage.data)
    assert "unfin" not in serialized
    assert "complete" in serialized


def test_no_salvage_without_complete_value():
    assert salvage_truncated_output('{"summary":"unfinished') is None


def test_salvage_never_fabricates_fields():
    raw = '{"observations":[{"order":1,"type":"action","description":"done"}],"objects":["x","y'
    salvage = salvage_truncated_output(raw)
    assert salvage is not None
    data = salvage.data
    assert "observations" in data
    obs = data["observations"]
    assert len(obs) == 1
    assert obs[0]["description"] == "done"


def test_salvage_preserves_complete_objects():
    raw = '{"summary":"ok","topics":["a","b","c'
    salvage = salvage_truncated_output(raw)
    assert salvage is not None
    assert salvage.data.get("summary") == "ok"


# --- Item-level semantic validation ---

def test_valid_observation_preserved_when_another_has_type_object():
    raw = json.dumps({
        "observations": [
            {"order": 1, "type": "action", "description": "A person places a container."},
            {"order": 2, "type": "object", "description": "A blue container."},
            {"order": 3, "type": "scene", "description": "An outdoor walkway."},
        ],
        "objects": ["blue container", "table"],
    })
    result = validate_observations(raw, max_observations=8, max_objects=12)
    assert len(result.output.observations) == 2
    assert result.output.observations[0].type == "action"
    assert result.output.observations[0].description == "A person places a container."
    assert result.output.observations[1].type == "scene"
    assert result.output.observations[1].description == "An outdoor walkway."
    assert result.output.objects == ["blue container", "table"]
    assert result.is_partial is True


def test_invalid_types_are_not_reinterpreted():
    raw = json.dumps({
        "observations": [
            {"order": 1, "type": "object", "description": "A blue container."},
        ],
        "objects": [],
    })
    result = validate_observations(raw, max_observations=8, max_objects=12)
    assert len(result.output.observations) == 0
    assert result.is_partial is True
    assert not any("appearance" in w for w in result.warnings)
    assert not any("scene" in w and "map" in w for w in result.warnings)


def test_remaining_order_is_normalized_after_discard():
    raw = json.dumps({
        "observations": [
            {"order": 1, "type": "action", "description": "First."},
            {"order": 2, "type": "object", "description": "Bad."},
            {"order": 3, "type": "scene", "description": "Third."},
        ],
        "objects": [],
    })
    result = validate_observations(raw, max_observations=8, max_objects=12)
    assert [item.order for item in result.output.observations] == [1, 2]


def test_discarding_invalid_observation_marks_partial():
    raw = json.dumps({
        "observations": [
            {"order": 1, "type": "action", "description": "Valid."},
            {"order": 2, "type": "object", "description": "Invalid."},
        ],
        "objects": [],
    })
    result = validate_observations(raw, max_observations=8, max_objects=12)
    assert result.is_partial is True


def test_diagnostics_identify_discarded_entry_and_type():
    raw = json.dumps({
        "observations": [
            {"order": 1, "type": "action", "description": "Valid."},
            {"order": 2, "type": "object", "description": "Invalid."},
        ],
        "objects": [],
    })
    result = validate_observations(raw, max_observations=8, max_objects=12)
    assert any(
        "discarded observation 2" in w and "'object'" in w
        for w in result.warnings
    )


def test_completely_unusable_output_still_fails_safely():
    raw = '{"not_observations": true}'
    with pytest.raises(ModelOutputInvalid):
        validate_observations(raw, max_observations=8, max_objects=12)


def test_objects_preserved_when_observations_partially_discarded():
    raw = json.dumps({
        "observations": [
            {"order": 1, "type": "object", "description": "Bad."},
        ],
        "objects": ["table", "chair"],
    })
    result = validate_observations(raw, max_observations=8, max_objects=12)
    assert result.output.objects == ["table", "chair"]


def test_all_valid_observations_not_partial():
    raw = json.dumps({
        "observations": [
            {"order": 1, "type": "action", "description": "Walks."},
            {"order": 2, "type": "scene", "description": "Room."},
        ],
        "objects": [],
    })
    result = validate_observations(raw, max_observations=8, max_objects=12)
    assert result.is_partial is False
    assert len(result.output.observations) == 2


# --- Grounded content validation tests ---

def _content_kwargs():
    return dict(image_count=2, max_texts_per_image=12, max_communicated_ideas=4, max_themes=6)


def test_ocr_content_batch_validates_all_fields():
    raw = json.dumps({
        "images": [
            {"imageIndex": 0, "visibleTexts": ["HELLO"]},
            {"imageIndex": 1, "visibleTexts": []},
        ],
        "communicatedIdeas": ["On-screen text says hello."],
        "themes": ["greeting"],
    })
    result = validate_ocr_content_batch(raw, **_content_kwargs())
    assert result.output.images[0].visibleTexts == ["HELLO"]
    assert result.output.communicatedIdeas == ["On-screen text says hello."]
    assert result.output.themes == ["greeting"]


def test_valid_visible_text_survives_invalid_communicated_idea():
    raw = json.dumps({
        "images": [
            {"imageIndex": 0, "visibleTexts": ["HELLO"]},
            {"imageIndex": 1, "visibleTexts": ["WORLD"]},
        ],
        "communicatedIdeas": ["Valid idea.", 42, None],
        "themes": ["valid-theme"],
    })
    result = validate_ocr_content_batch(raw, **_content_kwargs())
    assert result.output.images[0].visibleTexts == ["HELLO"]
    assert result.output.images[1].visibleTexts == ["WORLD"]
    assert result.output.communicatedIdeas == ["Valid idea."]
    assert any("non-string" in w.lower() for w in result.warnings)


def test_valid_communicated_ideas_survive_invalid_theme():
    raw = json.dumps({
        "images": [
            {"imageIndex": 0, "visibleTexts": []},
            {"imageIndex": 1, "visibleTexts": []},
        ],
        "communicatedIdeas": ["Good idea."],
        "themes": [123, "valid-theme"],
    })
    result = validate_ocr_content_batch(raw, **_content_kwargs())
    assert result.output.communicatedIdeas == ["Good idea."]
    assert result.output.themes == ["valid-theme"]


def test_non_string_ideas_rejected_individually():
    raw = json.dumps({
        "images": [
            {"imageIndex": 0, "visibleTexts": []},
            {"imageIndex": 1, "visibleTexts": []},
        ],
        "communicatedIdeas": [42, "Valid."],
        "themes": [],
    })
    result = validate_ocr_content_batch(raw, **_content_kwargs())
    assert result.output.communicatedIdeas == ["Valid."]
    assert any("non-string" in w.lower() for w in result.warnings)


def test_non_string_themes_rejected_individually():
    raw = json.dumps({
        "images": [
            {"imageIndex": 0, "visibleTexts": []},
            {"imageIndex": 1, "visibleTexts": []},
        ],
        "communicatedIdeas": [],
        "themes": [True, "motivation"],
    })
    result = validate_ocr_content_batch(raw, **_content_kwargs())
    assert result.output.themes == ["motivation"]


def test_exact_duplicate_ideas_removed():
    raw = json.dumps({
        "images": [
            {"imageIndex": 0, "visibleTexts": []},
            {"imageIndex": 1, "visibleTexts": []},
        ],
        "communicatedIdeas": ["Same idea.", "Same idea."],
        "themes": [],
    })
    result = validate_ocr_content_batch(raw, **_content_kwargs())
    assert result.output.communicatedIdeas == ["Same idea."]


def test_exact_duplicate_themes_removed():
    raw = json.dumps({
        "images": [
            {"imageIndex": 0, "visibleTexts": []},
            {"imageIndex": 1, "visibleTexts": []},
        ],
        "communicatedIdeas": [],
        "themes": ["motivation", "Motivation"],
    })
    result = validate_ocr_content_batch(raw, **_content_kwargs())
    assert result.output.themes == ["motivation"]


def test_idea_list_limits_enforced():
    raw = json.dumps({
        "images": [
            {"imageIndex": 0, "visibleTexts": []},
            {"imageIndex": 1, "visibleTexts": []},
        ],
        "communicatedIdeas": [f"Idea {i}." for i in range(10)],
        "themes": [],
    })
    result = validate_ocr_content_batch(raw, **_content_kwargs())
    assert len(result.output.communicatedIdeas) == 4


def test_theme_list_limits_enforced():
    raw = json.dumps({
        "images": [
            {"imageIndex": 0, "visibleTexts": []},
            {"imageIndex": 1, "visibleTexts": []},
        ],
        "communicatedIdeas": [],
        "themes": [f"theme-{i}" for i in range(10)],
    })
    result = validate_ocr_content_batch(raw, **_content_kwargs())
    assert len(result.output.themes) == 6


def test_salvage_never_manufactures_idea():
    raw = '{"images":[{"imageIndex":0,"visibleTexts":["OK"]}],"communicatedIdeas":["done"],"themes":["x","y'
    salvage = salvage_truncated_output(raw)
    assert salvage is not None
    serialized = json.dumps(salvage.data)
    assert "done" in serialized


def test_salvage_never_manufactures_theme():
    raw = '{"images":[{"imageIndex":0,"visibleTexts":[]}],"communicatedIdeas":[],"themes":["abc'
    salvage = salvage_truncated_output(raw)
    assert salvage is not None
    data = salvage.data
    themes = data.get("themes", [])
    assert all(isinstance(t, str) for t in themes)
    assert "abc" not in json.dumps(themes)


def test_unknown_fields_follow_validation_policy():
    raw = json.dumps({
        "images": [
            {"imageIndex": 0, "visibleTexts": []},
            {"imageIndex": 1, "visibleTexts": []},
        ],
        "communicatedIdeas": [],
        "themes": [],
        "extraField": "should warn",
    })
    result = validate_ocr_content_batch(raw, **_content_kwargs())
    assert any("Unknown field" in w for w in result.warnings)


def test_missing_ideas_becomes_empty_array():
    raw = json.dumps({
        "images": [
            {"imageIndex": 0, "visibleTexts": ["text"]},
            {"imageIndex": 1, "visibleTexts": []},
        ],
        "themes": [],
    })
    result = validate_ocr_content_batch(raw, **_content_kwargs())
    assert result.output.communicatedIdeas == []
    assert result.output.images[0].visibleTexts == ["text"]


def test_missing_themes_becomes_empty_array():
    raw = json.dumps({
        "images": [
            {"imageIndex": 0, "visibleTexts": []},
            {"imageIndex": 1, "visibleTexts": []},
        ],
        "communicatedIdeas": ["idea"],
    })
    result = validate_ocr_content_batch(raw, **_content_kwargs())
    assert result.output.themes == []
    assert result.output.communicatedIdeas == ["idea"]


# --- Synthesis validation tests ---

from visual_service.validation.output_validator import validate_synthesis_output


_EVIDENCE_IDS = frozenset(["text-0001", "text-0002", "text-0003"])


def _synthesis_kwargs():
    return dict(valid_evidence_ids=_EVIDENCE_IDS, max_communicated_ideas=8, max_themes=8, max_evidence_references_per_item=6)


def test_valid_synthesis_output_accepted():
    raw = json.dumps({
        "communicatedIdeas": [
            {"idea": "On-screen text discusses frustration.", "evidence": ["text-0001"]}
        ],
        "themes": [
            {"theme": "frustration", "evidence": ["text-0002"]}
        ],
    })
    result = validate_synthesis_output(raw, **_synthesis_kwargs())
    assert len(result.output.communicatedIdeas) == 1
    assert result.output.communicatedIdeas[0].idea == "On-screen text discusses frustration."
    assert len(result.output.themes) == 1


def test_unknown_evidence_id_rejected():
    raw = json.dumps({
        "communicatedIdeas": [
            {"idea": "An idea.", "evidence": ["text-0001", "text-9999"]}
        ],
        "themes": [],
    })
    result = validate_synthesis_output(raw, **_synthesis_kwargs())
    assert len(result.output.communicatedIdeas) == 1
    assert result.output.communicatedIdeas[0].evidence == ["text-0001"]
    assert any("text-9999" in w for w in result.warnings)


def test_all_evidence_ids_invalid_rejects_item():
    raw = json.dumps({
        "communicatedIdeas": [
            {"idea": "Bad idea.", "evidence": ["text-9999"]}
        ],
        "themes": [],
    })
    result = validate_synthesis_output(raw, **_synthesis_kwargs())
    assert len(result.output.communicatedIdeas) == 0


def test_empty_evidence_list_rejects_item():
    raw = json.dumps({
        "communicatedIdeas": [
            {"idea": "No evidence.", "evidence": []}
        ],
        "themes": [],
    })
    result = validate_synthesis_output(raw, **_synthesis_kwargs())
    assert len(result.output.communicatedIdeas) == 0


def test_empty_idea_string_rejected():
    raw = json.dumps({
        "communicatedIdeas": [
            {"idea": "", "evidence": ["text-0001"]}
        ],
        "themes": [],
    })
    result = validate_synthesis_output(raw, **_synthesis_kwargs())
    assert len(result.output.communicatedIdeas) == 0


def test_duplicate_evidence_ids_in_item_deduped():
    raw = json.dumps({
        "communicatedIdeas": [
            {"idea": "An idea.", "evidence": ["text-0001", "text-0001", "text-0002"]}
        ],
        "themes": [],
    })
    result = validate_synthesis_output(raw, **_synthesis_kwargs())
    assert result.output.communicatedIdeas[0].evidence == ["text-0001", "text-0002"]


def test_duplicate_ideas_deduped():
    raw = json.dumps({
        "communicatedIdeas": [
            {"idea": "Same idea.", "evidence": ["text-0001"]},
            {"idea": "Same idea.", "evidence": ["text-0002"]},
        ],
        "themes": [],
    })
    result = validate_synthesis_output(raw, **_synthesis_kwargs())
    assert len(result.output.communicatedIdeas) == 1


def test_duplicate_themes_deduped_case_insensitive():
    raw = json.dumps({
        "communicatedIdeas": [],
        "themes": [
            {"theme": "Frustration", "evidence": ["text-0001"]},
            {"theme": "frustration", "evidence": ["text-0002"]},
        ],
    })
    result = validate_synthesis_output(raw, **_synthesis_kwargs())
    assert len(result.output.themes) == 1


def test_max_ideas_enforced():
    ideas = [
        {"idea": f"Idea {i}.", "evidence": ["text-0001"]}
        for i in range(10)
    ]
    raw = json.dumps({"communicatedIdeas": ideas, "themes": []})
    result = validate_synthesis_output(raw, **_synthesis_kwargs())
    assert len(result.output.communicatedIdeas) == 8


def test_max_themes_enforced():
    themes = [
        {"theme": f"Theme {i}", "evidence": ["text-0001"]}
        for i in range(10)
    ]
    raw = json.dumps({"communicatedIdeas": [], "themes": themes})
    result = validate_synthesis_output(raw, **_synthesis_kwargs())
    assert len(result.output.themes) == 8


def test_valid_items_survive_invalid_neighbors():
    raw = json.dumps({
        "communicatedIdeas": [
            {"idea": "Good idea.", "evidence": ["text-0001"]},
            {"idea": "Bad idea.", "evidence": ["text-9999"]},
            {"idea": "Another good idea.", "evidence": ["text-0002"]},
        ],
        "themes": [],
    })
    result = validate_synthesis_output(raw, **_synthesis_kwargs())
    assert len(result.output.communicatedIdeas) == 2
    assert result.output.communicatedIdeas[0].idea == "Good idea."
    assert result.output.communicatedIdeas[1].idea == "Another good idea."


def test_non_string_evidence_id_ignored():
    raw = json.dumps({
        "communicatedIdeas": [
            {"idea": "An idea.", "evidence": [1, "text-0001"]}
        ],
        "themes": [],
    })
    result = validate_synthesis_output(raw, **_synthesis_kwargs())
    assert len(result.output.communicatedIdeas) == 1
    assert result.output.communicatedIdeas[0].evidence == ["text-0001"]


def test_partial_evidence_validity_keeps_item():
    raw = json.dumps({
        "communicatedIdeas": [
            {"idea": "Idea.", "evidence": ["text-0001", "text-9999", "text-0002"]}
        ],
        "themes": [],
    })
    result = validate_synthesis_output(raw, **_synthesis_kwargs())
    assert len(result.output.communicatedIdeas) == 1
    assert result.output.communicatedIdeas[0].evidence == ["text-0001", "text-0002"]


def test_synthesis_non_object_idea_rejected():
    raw = json.dumps({
        "communicatedIdeas": ["just a string"],
        "themes": [],
    })
    result = validate_synthesis_output(raw, **_synthesis_kwargs())
    assert len(result.output.communicatedIdeas) == 0


def test_synthesis_truncated_salvage():
    truncated = '{"communicatedIdeas": [{"idea": "Complete idea.", "evidence": ["text-0001"]}], "themes": [{"theme": "tru'
    result = validate_synthesis_output(truncated, **_synthesis_kwargs())
    assert len(result.output.communicatedIdeas) == 1
    assert result.output.communicatedIdeas[0].idea == "Complete idea."


def test_synthesis_no_repair_inference():
    raw = json.dumps({
        "communicatedIdeas": [],
        "themes": [],
    })
    result = validate_synthesis_output(raw, **_synthesis_kwargs())
    assert len(result.output.communicatedIdeas) == 0
    assert len(result.output.themes) == 0
    assert not result.is_partial


def test_synthesis_excessive_evidence_rejects_item():
    evidence_ids = frozenset([f"text-{i:04d}" for i in range(10)])
    raw = json.dumps({
        "communicatedIdeas": [
            {"idea": "Too many refs.", "evidence": [f"text-{i:04d}" for i in range(7)]}
        ],
        "themes": [],
    })
    result = validate_synthesis_output(
        raw, valid_evidence_ids=evidence_ids, max_communicated_ideas=8,
        max_themes=8, max_evidence_references_per_item=6,
    )
    assert len(result.output.communicatedIdeas) == 0
    assert any("Rejected idea" in w and "7 evidence references" in w for w in result.warnings)


def test_synthesis_evidence_at_limit_accepted():
    evidence_ids = frozenset([f"text-{i:04d}" for i in range(10)])
    raw = json.dumps({
        "communicatedIdeas": [
            {"idea": "Exactly at limit.", "evidence": [f"text-{i:04d}" for i in range(6)]}
        ],
        "themes": [],
    })
    result = validate_synthesis_output(
        raw, valid_evidence_ids=evidence_ids, max_communicated_ideas=8,
        max_themes=8, max_evidence_references_per_item=6,
    )
    assert len(result.output.communicatedIdeas) == 1


def test_synthesis_excessive_evidence_preserves_neighbors():
    evidence_ids = frozenset([f"text-{i:04d}" for i in range(10)])
    raw = json.dumps({
        "communicatedIdeas": [
            {"idea": "Good idea.", "evidence": ["text-0001"]},
            {"idea": "Too many refs.", "evidence": [f"text-{i:04d}" for i in range(7)]},
            {"idea": "Another good idea.", "evidence": ["text-0002"]},
        ],
        "themes": [],
    })
    result = validate_synthesis_output(
        raw, valid_evidence_ids=evidence_ids, max_communicated_ideas=8,
        max_themes=8, max_evidence_references_per_item=6,
    )
    assert len(result.output.communicatedIdeas) == 2
    assert result.output.communicatedIdeas[0].idea == "Good idea."
    assert result.output.communicatedIdeas[1].idea == "Another good idea."


def test_synthesis_does_not_rewrite_content():
    raw = json.dumps({
        "communicatedIdeas": [
            {"idea": "Exact wording preserved.", "evidence": ["text-0001"]}
        ],
        "themes": [
            {"theme": "exact theme", "evidence": ["text-0002"]}
        ],
    })
    result = validate_synthesis_output(raw, **_synthesis_kwargs())
    assert result.output.communicatedIdeas[0].idea == "Exact wording preserved."
    assert result.output.themes[0].theme == "exact theme"


def test_synthesis_does_not_manufacture_references():
    raw = json.dumps({
        "communicatedIdeas": [
            {"idea": "An idea.", "evidence": ["text-0001"]}
        ],
        "themes": [],
    })
    result = validate_synthesis_output(raw, **_synthesis_kwargs())
    assert result.output.communicatedIdeas[0].evidence == ["text-0001"]
    all_ev = []
    for item in result.output.communicatedIdeas:
        all_ev.extend(item.evidence)
    for item in result.output.themes:
        all_ev.extend(item.evidence)
    assert all(ev in _EVIDENCE_IDS for ev in all_ev)


def test_environmental_text_rejected_as_theme():
    raw = json.dumps({
        "communicatedIdeas": [],
        "themes": [
            {"theme": "Environmental text", "evidence": ["text-0001"]},
            {"theme": "Identity", "evidence": ["text-0002"]},
        ],
    })
    result = validate_synthesis_output(raw, **_synthesis_kwargs())
    assert len(result.output.themes) == 1
    assert result.output.themes[0].theme == "Identity"


def test_normalized_meta_themes_rejected():
    for meta in ["OCR", "Captions", "visible text", "SIGNAGE"]:
        raw = json.dumps({
            "communicatedIdeas": [],
            "themes": [{"theme": meta, "evidence": ["text-0001"]}],
        })
        result = validate_synthesis_output(raw, **_synthesis_kwargs())
        assert len(result.output.themes) == 0, f"{meta!r} should be blocked"


def test_legitimate_themes_not_rejected():
    for theme in ["California wildlife", "Street art", "Label design"]:
        raw = json.dumps({
            "communicatedIdeas": [],
            "themes": [{"theme": theme, "evidence": ["text-0001"]}],
        })
        result = validate_synthesis_output(raw, **_synthesis_kwargs())
        assert len(result.output.themes) == 1
        assert result.output.themes[0].theme == theme


def test_incidental_idea_phrases_rejected():
    raw = json.dumps({
        "communicatedIdeas": [
            {"idea": "The text mentions incidental environmental text such as CALIFORNIA.", "evidence": ["text-0001"]},
        ],
        "themes": [],
    })
    result = validate_synthesis_output(raw, **_synthesis_kwargs())
    assert len(result.output.communicatedIdeas) == 0


def test_diagnostic_warning_on_rejection():
    raw = json.dumps({
        "communicatedIdeas": [],
        "themes": [{"theme": "Environmental text", "evidence": ["text-0001"]}],
    })
    result = validate_synthesis_output(raw, **_synthesis_kwargs())
    assert any("Rejected meta-theme" in w for w in result.warnings)


def test_valid_ideas_not_rewritten():
    raw = json.dumps({
        "communicatedIdeas": [
            {"idea": "The text poses questions about identity.", "evidence": ["text-0001"]},
        ],
        "themes": [],
    })
    result = validate_synthesis_output(raw, **_synthesis_kwargs())
    assert len(result.output.communicatedIdeas) == 1
    assert result.output.communicatedIdeas[0].idea == "The text poses questions about identity."


def test_no_evidence_invented_by_blocklist():
    raw = json.dumps({
        "communicatedIdeas": [
            {"idea": "Text mentions incidental environmental text.", "evidence": ["text-0001"]},
            {"idea": "Valid idea.", "evidence": ["text-0002"]},
        ],
        "themes": [],
    })
    result = validate_synthesis_output(raw, **_synthesis_kwargs())
    assert len(result.output.communicatedIdeas) == 1
    assert result.output.communicatedIdeas[0].idea == "Valid idea."
    all_ev = []
    for item in result.output.communicatedIdeas:
        all_ev.extend(item.evidence)
    assert all(ev in _EVIDENCE_IDS for ev in all_ev)
