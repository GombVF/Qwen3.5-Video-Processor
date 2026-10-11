import pytest

from visual_service.model.consolidator import (
    ObservationEntry,
    WindowObservationResult,
    consolidate_results,
    _word_similarity,
    _remove_ocr_fragments,
    _consolidate_communicated_ideas,
    _consolidate_themes,
)


def _window(index, observations, objects=()):
    return WindowObservationResult(index, observations, list(objects))


# --- Observation deduplication (Gap 3) ---

def test_overlap_duplicates_removed_and_order_normalized():
    result = _consolidate([
        _window(0, [ObservationEntry(1, "action", "La persona mezcla los ingredientes.")]),
        _window(1, [
            ObservationEntry(1, "action", "La persona mezcla los ingredientes"),
            ObservationEntry(2, "action", "Coloca el tazón sobre la mesa."),
        ]),
    ])
    assert len(result.observations) == 2
    assert [item["order"] for item in result.observations] == [1, 2]


def test_different_observation_types_are_not_merged():
    result = _consolidate([_window(0, [
        ObservationEntry(1, "scene", "Una persona está en una cocina."),
        ObservationEntry(2, "action", "Una persona camina en una cocina."),
    ])])
    assert len(result.observations) == 2


def test_distinct_steps_are_preserved():
    result = _consolidate([_window(0, [
        ObservationEntry(1, "action", "La persona corta las verduras."),
        ObservationEntry(2, "action", "La persona mezcla las verduras en un tazón."),
        ObservationEntry(3, "action", "La persona sirve las verduras en un plato."),
    ])])
    assert len(result.observations) == 3


def test_repeated_running_walking_across_overlap():
    result = _consolidate([
        _window(0, [
            ObservationEntry(1, "action", "Una persona corre por un sendero."),
        ]),
        _window(1, [
            ObservationEntry(1, "action", "Una persona corre por un sendero."),
            ObservationEntry(2, "action", "La persona camina lentamente."),
        ]),
    ])
    assert len(result.observations) == 2
    descriptions = [o["description"] for o in result.observations]
    assert "corre" in descriptions[0]
    assert "camina" in descriptions[1]


def test_repeated_appearance_across_windows():
    result = _consolidate([
        _window(0, [
            ObservationEntry(1, "appearance", "La persona lleva auriculares negros."),
        ]),
        _window(1, [
            ObservationEntry(1, "appearance", "La persona lleva auriculares negros."),
        ]),
    ])
    assert len(result.observations) == 1


def test_similar_appearance_deduplicated_at_overlap():
    result = _consolidate([
        _window(0, [
            ObservationEntry(1, "appearance", "Una persona con barba y collar."),
        ]),
        _window(1, [
            ObservationEntry(1, "appearance", "Una persona con barba y un collar metálico."),
        ]),
    ])
    assert len(result.observations) == 1


def test_genuinely_repeated_later_action_preserved():
    result = _consolidate([
        _window(0, [
            ObservationEntry(1, "action", "La persona levanta la mano derecha."),
        ]),
        _window(5, [
            ObservationEntry(1, "action", "La persona levanta la mano derecha."),
        ]),
    ], nonadjacent_threshold=0.92)
    assert len(result.observations) == 2


def test_nonadjacent_windows_preserve_repeated_actions():
    result = _consolidate([
        _window(0, [
            ObservationEntry(1, "action", "La persona mezcla los ingredientes en el tazón."),
        ]),
        _window(5, [
            ObservationEntry(1, "action", "La persona mezcla los ingredientes en el tazón."),
        ]),
    ])
    assert len(result.observations) == 2


def test_consecutive_order_normalization():
    result = _consolidate([
        _window(0, [ObservationEntry(5, "action", "A")]),
        _window(1, [ObservationEntry(99, "scene", "B")]),
    ])
    assert [o["order"] for o in result.observations] == [1, 2]


def test_dedup_diagnostics_emitted():
    result = _consolidate([
        _window(0, [ObservationEntry(1, "action", "La persona cocina.")]),
        _window(1, [ObservationEntry(1, "action", "La persona cocina.")]),
    ])
    assert any("deduplicated" in d for d in result.diagnostics)


# --- Word similarity ---

def test_word_similarity_identical():
    assert _word_similarity({"hello", "world"}, {"hello", "world"}) == 1.0


def test_word_similarity_disjoint():
    assert _word_similarity({"hello"}, {"world"}) == 0.0


def test_word_similarity_partial_overlap():
    sim = _word_similarity({"la", "persona", "cocina"}, {"la", "persona", "camina"})
    assert 0.4 < sim < 0.7


# --- OCR deduplication (Gap 3+4) ---

def test_ocr_dedup_is_conservative():
    result = _consolidate(
        [_window(0, [])],
        visible_texts=["WHO ARE YOU?", "WHO ARE YOU", "WHO ARE YOU TODAY?"],
    )
    assert result.visible_texts == ["WHO ARE YOU?", "WHO ARE YOU TODAY?"]


def test_objects_and_topics_deduplicate_case_insensitively():
    result = _consolidate(
        [_window(0, [], ["Tazón"]), _window(1, [], ["tazón", "cuchara"])],
        global_topics=["Cocina", "cocina"],
    )
    assert result.objects == ["Tazón", "cuchara"]
    assert result.topics == ["Cocina"]


# --- OCR fragment removal (Gap 4) ---

def test_fragment_removed_when_subsumed_by_nearby_frame():
    result = _consolidate(
        [_window(0, [])],
        visible_texts=["SUBSCRIBE NOW", "SUBSCRIBE", "LIKE AND SHARE"],
        text_frame_sequences=[0, 1, 2],
    )
    assert "SUBSCRIBE NOW" in result.visible_texts
    assert "LIKE AND SHARE" in result.visible_texts
    texts_lower = [t.lower() for t in result.visible_texts]
    assert texts_lower.count("subscribe") == 0 or "subscribe now" in texts_lower


def test_fragment_not_removed_when_from_distant_frame():
    result = _consolidate(
        [_window(0, [])],
        visible_texts=["SUBSCRIBE NOW", "SUBSCRIBE"],
        text_frame_sequences=[0, 50],
        fragment_neighborhood=3,
    )
    assert len(result.visible_texts) == 2


def test_complete_caption_plus_partial_fragments():
    result = _consolidate(
        [_window(0, [])],
        visible_texts=[
            "¡Hola a todos! Bienvenidos",
            "¡Hola a todos!",
            "Bienvenidos",
        ],
        text_frame_sequences=[0, 1, 2],
    )
    assert "¡Hola a todos! Bienvenidos" in result.visible_texts
    assert len(result.visible_texts) == 1


def test_repeated_caption_over_consecutive_frames():
    result = _consolidate(
        [_window(0, [])],
        visible_texts=["HELLO WORLD", "HELLO WORLD", "HELLO WORLD"],
        text_frame_sequences=[0, 1, 2],
    )
    assert result.visible_texts == ["HELLO WORLD"]


def test_different_captions_sharing_words_preserved():
    result = _consolidate(
        [_window(0, [])],
        visible_texts=["HELLO WORLD", "HELLO EVERYONE"],
        text_frame_sequences=[0, 1],
    )
    assert len(result.visible_texts) == 2


def test_unicode_and_emoji_preserved():
    texts = ["¡Hola! 🎉", "Café ☕", "日本語テスト"]
    result = _consolidate(
        [_window(0, [])],
        visible_texts=texts,
        text_frame_sequences=[0, 1, 2],
    )
    assert result.visible_texts == texts


def test_no_invented_reconstructed_strings():
    input_texts = ["HELLO", "WORLD"]
    result = _consolidate(
        [_window(0, [])],
        visible_texts=input_texts,
        text_frame_sequences=[0, 1],
    )
    for text in result.visible_texts:
        assert text in input_texts


def test_fragments_crossing_ocr_batch_boundaries():
    result = _consolidate(
        [_window(0, [])],
        visible_texts=[
            "Full caption text here",
            "Full caption text",
            "caption text here",
        ],
        text_frame_sequences=[3, 4, 5],
    )
    assert "Full caption text here" in result.visible_texts
    remaining = [t for t in result.visible_texts if t != "Full caption text here"]
    assert len(remaining) == 0


def test_remove_ocr_fragments_helper():
    values = ["SUBSCRIBE NOW", "SUBSCRIBE", "DIFFERENT TEXT"]
    frames = [0, 1, 5]
    kept_values, kept_frames = _remove_ocr_fragments(values, frames, 3, [])
    assert "SUBSCRIBE NOW" in kept_values
    assert "DIFFERENT TEXT" in kept_values
    assert len(kept_values) == 2


def test_fragment_removal_diagnostics():
    diagnostics: list[str] = []
    _remove_ocr_fragments(
        ["LONG TEXT HERE", "LONG TEXT"],
        [0, 1],
        3,
        diagnostics,
    )
    assert any("subsumed" in d for d in diagnostics)


# --- Communicated ideas consolidation ---

def test_duplicate_ideas_across_batches_removed():
    diagnostics: list[str] = []
    result = _consolidate_communicated_ideas(
        [
            "On-screen text presents running sounds as motivation to run faster.",
            "On-screen text says running-footstep sounds are used to encourage faster running.",
        ],
        threshold=0.85,
        max_ideas=20,
        diagnostics=diagnostics,
    )
    assert len(result) == 2


def test_exact_duplicate_ideas_removed():
    diagnostics: list[str] = []
    result = _consolidate_communicated_ideas(
        [
            "On-screen text mentions wind chimes.",
            "On-screen text mentions wind chimes.",
        ],
        threshold=0.85,
        max_ideas=20,
        diagnostics=diagnostics,
    )
    assert len(result) == 1
    assert "wind chimes" in result[0]


def test_distinct_ideas_remain_separate():
    diagnostics: list[str] = []
    result = _consolidate_communicated_ideas(
        [
            "On-screen text presents running sounds as motivation to run faster.",
            "On-screen text mentions unexpected wind chimes in the woods.",
        ],
        threshold=0.85,
        max_ideas=20,
        diagnostics=diagnostics,
    )
    assert len(result) == 2
    assert any("running" in i for i in result)
    assert any("wind chimes" in i for i in result)


def test_more_complete_idea_variant_preserved():
    diagnostics: list[str] = []
    result = _consolidate_communicated_ideas(
        [
            "On-screen text mentions running.",
            "On-screen text mentions running as a way to get faster.",
        ],
        threshold=0.85,
        max_ideas=20,
        diagnostics=diagnostics,
    )
    assert len(result) <= 2


def test_no_new_merged_sentence_manufactured():
    ideas = [
        "On-screen text presents running sounds as motivation.",
        "On-screen text mentions unexpected wind chimes.",
    ]
    diagnostics: list[str] = []
    result = _consolidate_communicated_ideas(ideas, 0.85, 20, diagnostics)
    for item in result:
        assert item in ideas


def test_idea_final_limit_respected():
    ideas = [
        "The runner uses footstep sounds for motivation.",
        "Wind chimes appear unexpectedly in the forest.",
        "A philosophical reflection on modern frustration is presented.",
        "The text questions the value of excessive information.",
        "Life and its meaning are explored through personal narrative.",
        "Self-reflection is encouraged through visual poetry.",
        "Technology has changed how we communicate daily.",
        "Nature photography reveals hidden patterns in leaves.",
        "Cooking traditions vary across Mediterranean cultures.",
        "Travel broadens perspective and challenges assumptions.",
        "Scientific discovery often begins with simple curiosity.",
        "Architecture reflects the values of its era.",
        "Fashion trends cycle through decades predictably.",
    ]
    diagnostics: list[str] = []
    result = _consolidate_communicated_ideas(ideas, 0.85, 10, diagnostics)
    assert len(result) == 10
    assert any("Truncated" in d for d in diagnostics)


def test_earliest_idea_occurrence_preserved():
    diagnostics: list[str] = []
    result = _consolidate_communicated_ideas(
        ["First version of idea.", "First version of idea."],
        threshold=0.85,
        max_ideas=20,
        diagnostics=diagnostics,
    )
    assert result == ["First version of idea."]


# --- Theme consolidation ---

def test_themes_deduplicated_case_insensitively():
    diagnostics: list[str] = []
    result = _consolidate_themes(
        ["Motivation", "motivation", "MOTIVATION"],
        max_themes=12,
        diagnostics=diagnostics,
    )
    assert result == ["Motivation"]


def test_theme_order_stable():
    diagnostics: list[str] = []
    result = _consolidate_themes(
        ["frustration", "motivation", "self-reflection"],
        max_themes=12,
        diagnostics=diagnostics,
    )
    assert result == ["frustration", "motivation", "self-reflection"]


def test_theme_final_limit_respected():
    themes = [f"theme-{i}" for i in range(20)]
    diagnostics: list[str] = []
    result = _consolidate_themes(themes, max_themes=5, diagnostics=diagnostics)
    assert len(result) == 5
    assert any("Truncated" in d for d in diagnostics)


def test_visible_text_consolidation_unchanged_with_ideas():
    result = _consolidate(
        [_window(0, [])],
        visible_texts=["WHO ARE YOU?", "WHO ARE YOU", "WHO ARE YOU TODAY?"],
        communicated_ideas=["An idea."],
        ocr_themes=["theme"],
    )
    assert result.visible_texts == ["WHO ARE YOU?", "WHO ARE YOU TODAY?"]
    assert result.communicated_ideas == ["An idea."]
    assert result.themes == ["theme"]


def _consolidate(windows, visible_texts=(), text_frame_sequences=None,
                 communicated_ideas=None, ocr_themes=None, **kwargs):
    defaults = dict(
        global_summary="x",
        global_topics=[],
        visible_texts=list(visible_texts),
        adjacent_threshold=0.78,
        nonadjacent_threshold=0.92,
        visible_text_threshold=0.94,
        text_frame_sequences=text_frame_sequences,
        fragment_neighborhood=3,
        communicated_ideas=communicated_ideas,
        ocr_themes=ocr_themes,
        idea_threshold=0.85,
        max_communicated_ideas=20,
        max_themes=12,
    )
    defaults.update(kwargs)
    return consolidate_results(windows, **defaults)
