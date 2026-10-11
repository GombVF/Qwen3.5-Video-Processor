from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from difflib import SequenceMatcher


@dataclass(frozen=True)
class ObservationEntry:
    order: int
    type: str
    description: str


@dataclass
class WindowObservationResult:
    window_index: int
    observations: list[ObservationEntry]
    objects: list[str]
    is_partial: bool = False
    warnings: list[str] = field(default_factory=list)


@dataclass
class ConsolidatedResult:
    summary: str
    observations: list[dict]
    visible_texts: list[str]
    objects: list[str]
    topics: list[str]
    communicated_ideas: list[str] = field(default_factory=list)
    themes: list[str] = field(default_factory=list)
    window_count: int = 0
    diagnostics: list[str] = field(default_factory=list)


def consolidate_results(
    windows: list[WindowObservationResult],
    *,
    global_summary: str,
    global_topics: list[str],
    visible_texts: list[str],
    adjacent_threshold: float,
    nonadjacent_threshold: float,
    visible_text_threshold: float,
    text_frame_sequences: list[int] | None = None,
    fragment_neighborhood: int = 3,
    communicated_ideas: list[str] | None = None,
    ocr_themes: list[str] | None = None,
    idea_threshold: float = 0.85,
    max_communicated_ideas: int = 20,
    max_themes: int = 12,
) -> ConsolidatedResult:
    diagnostics = [
        f"Window {window.window_index}: partial/salvaged output"
        for window in windows
        if window.is_partial
    ]
    observations = _consolidate_observations(
        windows,
        adjacent_threshold=adjacent_threshold,
        nonadjacent_threshold=nonadjacent_threshold,
        diagnostics=diagnostics,
    )
    texts = _consolidate_visible_texts(
        visible_texts,
        visible_text_threshold,
        diagnostics,
        frame_sequences=text_frame_sequences,
        fragment_neighborhood=fragment_neighborhood,
    )
    objects = _deduplicate_exact([item for window in windows for item in window.objects])
    topics = _deduplicate_exact(global_topics)

    consolidated_ideas = _consolidate_communicated_ideas(
        communicated_ideas or [],
        idea_threshold,
        max_communicated_ideas,
        diagnostics,
    )
    consolidated_themes = _consolidate_themes(
        ocr_themes or [],
        max_themes,
        diagnostics,
    )

    return ConsolidatedResult(
        summary=global_summary.strip(),
        observations=[
            {"order": index, "type": item.type, "description": item.description}
            for index, (item, _) in enumerate(observations, start=1)
        ],
        visible_texts=texts,
        objects=objects,
        topics=topics,
        communicated_ideas=consolidated_ideas,
        themes=consolidated_themes,
        window_count=len(windows),
        diagnostics=diagnostics,
    )


def _consolidate_observations(
    windows: list[WindowObservationResult],
    *,
    adjacent_threshold: float,
    nonadjacent_threshold: float,
    diagnostics: list[str],
) -> list[tuple[ObservationEntry, int]]:
    merged: list[tuple[ObservationEntry, int]] = []
    for window in sorted(windows, key=lambda item: item.window_index):
        for observation in sorted(window.observations, key=lambda item: item.order):
            duplicate = False
            candidate = _normalize(observation.description)
            candidate_words = _tokenize(candidate)
            for existing, source_window in merged:
                if observation.type != existing.type:
                    continue
                window_distance = abs(window.window_index - source_window)
                if window_distance == 0:
                    continue
                if window_distance > 1:
                    continue
                threshold = adjacent_threshold
                existing_norm = _normalize(existing.description)
                char_sim = _similarity(candidate, existing_norm)
                word_sim = _word_similarity(candidate_words, _tokenize(existing_norm))
                combined = max(char_sim, word_sim)
                if combined >= threshold:
                    duplicate = True
                    diagnostics.append(
                        f"Window {window.window_index}: deduplicated {observation.type}: "
                        f"{observation.description[:100]} "
                        f"(char={char_sim:.2f} word={word_sim:.2f} "
                        f"threshold={threshold:.2f})"
                    )
                    break
            if not duplicate:
                merged.append((observation, window.window_index))
    return merged


def _consolidate_visible_texts(
    values: list[str],
    threshold: float,
    diagnostics: list[str],
    *,
    frame_sequences: list[int] | None = None,
    fragment_neighborhood: int = 3,
) -> list[str]:
    if frame_sequences is not None:
        values, frame_sequences = _remove_ocr_fragments(
            values, frame_sequences, fragment_neighborhood, diagnostics,
        )

    result: list[str] = []
    result_frames: list[int] = []
    for idx, value in enumerate(values):
        candidate = value.strip()
        if not candidate:
            continue
        candidate_norm = _normalize_ocr(candidate)
        candidate_frame = frame_sequences[idx] if frame_sequences else idx
        match_index = None
        for ri, existing in enumerate(result):
            existing_norm = _normalize_ocr(existing)
            if candidate_norm == existing_norm:
                match_index = ri
                break
            length_ratio = min(len(candidate_norm), len(existing_norm)) / max(
                len(candidate_norm), len(existing_norm), 1,
            )
            if length_ratio >= 0.85 and _similarity(candidate_norm, existing_norm) >= threshold:
                match_index = ri
                break
        if match_index is None:
            result.append(candidate)
            result_frames.append(candidate_frame)
        elif len(candidate) > len(result[match_index]):
            diagnostics.append(
                f"Replaced near-identical OCR variant with more complete text: {candidate[:100]}"
            )
            result[match_index] = candidate
            result_frames[match_index] = candidate_frame
        else:
            diagnostics.append(f"Deduplicated repeated OCR text: {candidate[:100]}")
    return result


def _remove_ocr_fragments(
    values: list[str],
    frame_sequences: list[int],
    neighborhood: int,
    diagnostics: list[str],
) -> tuple[list[str], list[int]]:
    n = len(values)
    removed: set[int] = set()
    for i in range(n):
        if i in removed:
            continue
        short = values[i].strip()
        if not short:
            removed.add(i)
            continue
        short_norm = _normalize_ocr(short)
        if len(short_norm) < 4:
            continue
        for j in range(n):
            if j == i or j in removed:
                continue
            if abs(frame_sequences[i] - frame_sequences[j]) > neighborhood:
                continue
            long_val = values[j].strip()
            long_norm = _normalize_ocr(long_val)
            if len(short_norm) < len(long_norm) and short_norm in long_norm:
                diagnostics.append(
                    f"Removed OCR fragment '{short[:80]}' subsumed by "
                    f"'{long_val[:80]}' (frames {frame_sequences[i]}->{frame_sequences[j]})"
                )
                removed.add(i)
                break
    kept_values = [values[i] for i in range(n) if i not in removed]
    kept_frames = [frame_sequences[i] for i in range(n) if i not in removed]
    return kept_values, kept_frames


def _deduplicate_exact(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        stripped = value.strip()
        key = _normalize(stripped)
        if stripped and key not in seen:
            seen.add(key)
            result.append(stripped)
    return result


def _normalize(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold().strip()
    return re.sub(r"\s+", " ", normalized)


def _normalize_ocr(value: str) -> str:
    normalized = _normalize(value)
    return re.sub(r"[^\w\s]", "", normalized)


def _tokenize(normalized: str) -> set[str]:
    return set(normalized.split())


def _similarity(left: str, right: str) -> float:
    if left == right:
        return 1.0
    return SequenceMatcher(None, left, right).ratio()


def _word_similarity(left_words: set[str], right_words: set[str]) -> float:
    if not left_words and not right_words:
        return 1.0
    if not left_words or not right_words:
        return 0.0
    intersection = left_words & right_words
    union = left_words | right_words
    return len(intersection) / len(union)


def _consolidate_communicated_ideas(
    ideas: list[str],
    threshold: float,
    max_ideas: int,
    diagnostics: list[str],
) -> list[str]:
    result: list[str] = []
    for idea in ideas:
        stripped = idea.strip()
        if not stripped:
            continue
        candidate_norm = _normalize(stripped)
        candidate_words = _tokenize(candidate_norm)
        match_index = None
        for ri, existing in enumerate(result):
            existing_norm = _normalize(existing)
            if candidate_norm == existing_norm:
                match_index = ri
                break
            char_sim = _similarity(candidate_norm, existing_norm)
            word_sim = _word_similarity(candidate_words, _tokenize(existing_norm))
            combined = max(char_sim, word_sim)
            if combined >= threshold:
                match_index = ri
                break
        if match_index is None:
            result.append(stripped)
        elif len(stripped) > len(result[match_index]):
            diagnostics.append(
                f"Replaced near-duplicate communicated idea with more complete variant: "
                f"{stripped[:100]}"
            )
            result[match_index] = stripped
        else:
            diagnostics.append(
                f"Deduplicated communicated idea: {stripped[:100]}"
            )
    if len(result) > max_ideas:
        diagnostics.append(
            f"Truncated communicated ideas from {len(result)} to limit {max_ideas}"
        )
        result = result[:max_ideas]
    return result


def _consolidate_themes(
    themes: list[str],
    max_themes: int,
    diagnostics: list[str],
) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for theme in themes:
        stripped = theme.strip()
        if not stripped:
            continue
        key = _normalize(stripped)
        if key in seen:
            diagnostics.append(f"Deduplicated theme: {stripped[:80]}")
            continue
        seen.add(key)
        result.append(stripped)
    if len(result) > max_themes:
        diagnostics.append(
            f"Truncated themes from {len(result)} to limit {max_themes}"
        )
        result = result[:max_themes]
    return result
