from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator


class ModelOutputInvalid(Exception):
    pass


class ModelOutputTruncated(ModelOutputInvalid):
    def __init__(self, message: str, raw_output: str = "", generated_tokens: int = 0):
        super().__init__(message)
        self.raw_output = raw_output
        self.generated_tokens = generated_tokens


OBSERVATION_TYPES = frozenset({"action", "scene", "appearance"})
FORBIDDEN_TEMPORAL_KEYS = frozenset({
    "startMs", "endMs", "timestamp", "timestamps", "duration",
    "durationMs", "frame", "frameIndex", "frameIndices", "relativeMs",
})


class ObservationItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    order: int = Field(ge=1)
    type: str
    description: str = Field(min_length=1)

    @field_validator("type")
    @classmethod
    def validate_type(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in OBSERVATION_TYPES:
            raise ValueError(f"Unsupported observation type: {value!r}")
        return normalized

    @field_validator("description")
    @classmethod
    def normalize_description(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("description must not be blank")
        return value


class ObservationOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    observations: list[ObservationItem] = Field(default_factory=list)
    objects: list[str] = Field(default_factory=list)


class GlobalOverviewOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    summary: str = Field(min_length=1)
    topics: list[str] = Field(default_factory=list)


class OcrImageOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    imageIndex: int = Field(ge=0)
    visibleTexts: list[str] = Field(default_factory=list)


class OcrBatchOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    images: list[OcrImageOutput]


class OcrContentBatchOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    images: list[OcrImageOutput]
    communicatedIdeas: list = Field(default_factory=list)
    themes: list = Field(default_factory=list)


class SynthesisIdeaItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    idea: str = Field(min_length=1)
    evidence: list[str] = Field(min_length=1)


class SynthesisThemeItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    theme: str = Field(min_length=1)
    evidence: list[str] = Field(min_length=1)


class SynthesisOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    communicatedIdeas: list[SynthesisIdeaItem] = Field(default_factory=list)
    themes: list[SynthesisThemeItem] = Field(default_factory=list)


@dataclass
class SalvageResult:
    data: dict[str, Any]
    raw_input: str
    discarded_tail: str
    diagnostics: list[str] = field(default_factory=list)


@dataclass
class ValidationResult:
    output: Any
    warnings: list[str] = field(default_factory=list)
    is_partial: bool = False


def strip_thinking_content(raw_text: str) -> tuple[str, bool]:
    """Remove a complete Qwen thinking preamble while preserving provenance."""
    stripped = raw_text.strip()
    match = re.match(r"^<think>.*?</think>\s*", stripped, flags=re.DOTALL)
    if match:
        return stripped[match.end():], True
    return stripped, False


def _parse_json_value(raw_text: str) -> tuple[Any, list[str]]:
    cleaned, had_thinking = strip_thinking_content(raw_text)
    warnings: list[str] = []
    if had_thinking:
        warnings.append("Model emitted thinking content despite non-thinking mode; ignored")

    fence = re.search(r"```(?:json)?\s*(.*?)\s*```", cleaned, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        cleaned = fence.group(1).strip()
        warnings.append("Removed Markdown code fence from model output")

    starts = [pos for pos in (cleaned.find("{"), cleaned.find("[")) if pos >= 0]
    if not starts:
        raise ModelOutputInvalid(f"No JSON value found in model output: {raw_text[:200]}")

    start = min(starts)
    try:
        value, consumed = json.JSONDecoder().raw_decode(cleaned[start:])
    except json.JSONDecodeError as exc:
        raise ModelOutputInvalid(f"Invalid JSON model output: {exc}") from exc

    trailing = cleaned[start + consumed:].strip()
    if trailing:
        warnings.append("Ignored non-JSON text after the first complete JSON value")
    return value, warnings


def _reject_temporal_fields(value: Any, path: str = "$") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if key in FORBIDDEN_TEMPORAL_KEYS:
                raise ModelOutputInvalid(f"Temporal field {key!r} is forbidden at {path}")
            _reject_temporal_fields(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_temporal_fields(child, f"{path}[{index}]")


def _deduplicate_strings(values: list[str], limit: int, field_name: str) -> tuple[list[str], list[str]]:
    warnings: list[str] = []
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, str):
            raise ModelOutputInvalid(f"{field_name} must contain only strings")
        stripped = value.strip()
        if not stripped:
            continue
        key = stripped.casefold()
        if key in seen:
            warnings.append(f"Removed exact duplicate from {field_name}: {stripped[:80]}")
            continue
        seen.add(key)
        result.append(stripped)
    if len(result) > limit:
        warnings.append(f"Truncated {field_name} from {len(result)} to configured limit {limit}")
        result = result[:limit]
    return result, warnings


def validate_global_overview(raw_text: str, *, max_topics: int) -> ValidationResult:
    data, warnings = _parse_json_value(raw_text)
    if not isinstance(data, dict):
        raise ModelOutputInvalid("Global overview must be one JSON object")
    _reject_temporal_fields(data)
    try:
        output = GlobalOverviewOutput.model_validate(data)
    except ValidationError as exc:
        raise ModelOutputInvalid(f"Global overview schema invalid: {exc}") from exc
    output.summary = output.summary.strip()
    output.topics, dedup_warnings = _deduplicate_strings(output.topics, max_topics, "topics")
    warnings.extend(dedup_warnings)
    return ValidationResult(output=output, warnings=warnings)


def validate_observations(
    raw_text: str, *, max_observations: int, max_objects: int,
) -> ValidationResult:
    data, warnings = _parse_json_value(raw_text)
    if not isinstance(data, dict):
        raise ModelOutputInvalid("Observation output must be one JSON object")
    _reject_temporal_fields(data)

    observations_raw = data.get("observations", [])
    if not isinstance(observations_raw, list):
        raise ModelOutputInvalid("observations must be a list")

    structure_data = {**data, "observations": []}
    try:
        output = ObservationOutput.model_validate(structure_data)
    except ValidationError as exc:
        raise ModelOutputInvalid(f"Observation schema invalid: {exc}") from exc

    is_partial = False
    valid_items: list[ObservationItem] = []
    for idx, obs_data in enumerate(observations_raw):
        raw_order = obs_data.get("order", idx + 1) if isinstance(obs_data, dict) else idx + 1
        try:
            item = ObservationItem.model_validate(obs_data)
            valid_items.append(item)
        except ValidationError:
            is_partial = True
            if isinstance(obs_data, dict) and "type" in obs_data:
                obs_type = obs_data["type"]
                warnings.append(
                    f"discarded observation {raw_order}: "
                    f"unsupported observation type {obs_type!r}"
                )
            else:
                warnings.append(
                    f"discarded observation {raw_order}: invalid observation structure"
                )

    deduplicated: list[ObservationItem] = []
    seen: set[tuple[str, str]] = set()
    for item in valid_items:
        key = (item.type, item.description.casefold())
        if key in seen:
            warnings.append(f"Removed exact duplicate observation: {item.description[:80]}")
            continue
        seen.add(key)
        deduplicated.append(item)
    if len(deduplicated) > max_observations:
        warnings.append(f"Truncated observations from {len(deduplicated)} to {max_observations}")
        deduplicated = deduplicated[:max_observations]
    for index, item in enumerate(deduplicated, start=1):
        if item.order != index:
            warnings.append(f"Normalized observation order {item.order} to {index}")
        item.order = index
    output.observations = deduplicated
    output.objects, object_warnings = _deduplicate_strings(output.objects, max_objects, "objects")
    warnings.extend(object_warnings)
    return ValidationResult(output=output, warnings=warnings, is_partial=is_partial)


def _normalize_ocr_shape(value: Any, image_count: int) -> tuple[dict[str, Any], list[str]]:
    warnings: list[str] = []
    if isinstance(value, dict) and "images" in value:
        return value, warnings
    if image_count == 1 and isinstance(value, dict) and "visibleTexts" in value:
        warnings.append("Normalized single-image OCR object to canonical batch shape")
        return {"images": [{"imageIndex": 0, "visibleTexts": value["visibleTexts"]}]}, warnings
    if isinstance(value, list):
        if image_count == 1 and all(isinstance(item, str) for item in value):
            warnings.append("Normalized bare OCR string array to canonical batch shape")
            return {"images": [{"imageIndex": 0, "visibleTexts": value}]}, warnings
        if all(isinstance(item, dict) for item in value):
            warnings.append("Normalized bare OCR image array to canonical batch shape")
            return {"images": value}, warnings
    raise ModelOutputInvalid("OCR output does not match a supported structural shape")


def validate_ocr_batch(
    raw_text: str, *, image_count: int, max_texts_per_image: int,
) -> ValidationResult:
    value, warnings = _parse_json_value(raw_text)
    data, shape_warnings = _normalize_ocr_shape(value, image_count)
    warnings.extend(shape_warnings)
    _reject_temporal_fields(data)
    try:
        output = OcrBatchOutput.model_validate(data)
    except ValidationError as exc:
        raise ModelOutputInvalid(f"OCR schema invalid: {exc}") from exc
    if len(output.images) != image_count:
        raise ModelOutputInvalid(f"OCR returned {len(output.images)} image entries; expected {image_count}")
    for expected_index, image in enumerate(output.images):
        if image.imageIndex != expected_index:
            raise ModelOutputInvalid(
                f"OCR imageIndex {image.imageIndex} does not match position {expected_index}"
            )
        image.visibleTexts, text_warnings = _deduplicate_strings(
            image.visibleTexts, max_texts_per_image,
            f"images[{expected_index}].visibleTexts",
        )
        warnings.extend(text_warnings)
    return ValidationResult(output=output, warnings=warnings)


def _validate_string_list_independently(
    values: list,
    limit: int,
    field_name: str,
) -> tuple[list[str], list[str]]:
    warnings: list[str] = []
    valid: list[str] = []
    for idx, item in enumerate(values):
        if not isinstance(item, str):
            warnings.append(f"Rejected non-string {field_name}[{idx}]: {type(item).__name__}")
            continue
        stripped = item.strip()
        if not stripped:
            continue
        valid.append(stripped)
    deduped, dedup_warnings = _deduplicate_strings(valid, limit, field_name)
    warnings.extend(dedup_warnings)
    return deduped, warnings


def _normalize_ocr_content_shape(
    value: Any,
    image_count: int,
) -> tuple[dict[str, Any], list[str]]:
    warnings: list[str] = []
    if isinstance(value, dict) and "images" in value:
        return value, warnings
    base, base_warnings = _normalize_ocr_shape(value, image_count)
    warnings.extend(base_warnings)
    return base, warnings


def validate_ocr_content_batch(
    raw_text: str,
    *,
    image_count: int,
    max_texts_per_image: int,
    max_communicated_ideas: int,
    max_themes: int,
) -> ValidationResult:
    value, warnings = _parse_json_value(raw_text)
    data, shape_warnings = _normalize_ocr_content_shape(value, image_count)
    warnings.extend(shape_warnings)
    _reject_temporal_fields(data)

    ideas_raw = data.pop("communicatedIdeas", [])
    themes_raw = data.pop("themes", [])
    unknown_extra = set(data.keys()) - {"images"}
    for key in sorted(unknown_extra):
        warnings.append(f"Unknown field in OCR content output: {key!r}")
        data.pop(key)

    try:
        ocr_output = OcrBatchOutput.model_validate(data)
    except ValidationError as exc:
        raise ModelOutputInvalid(f"OCR content schema invalid: {exc}") from exc
    if len(ocr_output.images) != image_count:
        raise ModelOutputInvalid(
            f"OCR returned {len(ocr_output.images)} image entries; expected {image_count}"
        )
    is_partial = False
    for expected_index, image in enumerate(ocr_output.images):
        if image.imageIndex != expected_index:
            raise ModelOutputInvalid(
                f"OCR imageIndex {image.imageIndex} does not match position {expected_index}"
            )
        image.visibleTexts, text_warnings = _deduplicate_strings(
            image.visibleTexts,
            max_texts_per_image,
            f"images[{expected_index}].visibleTexts",
        )
        warnings.extend(text_warnings)

    if not isinstance(ideas_raw, list):
        warnings.append("communicatedIdeas is not a list; using empty")
        ideas_raw = []
        is_partial = True
    if not isinstance(themes_raw, list):
        warnings.append("themes is not a list; using empty")
        themes_raw = []
        is_partial = True

    ideas, idea_warnings = _validate_string_list_independently(
        ideas_raw, max_communicated_ideas, "communicatedIdeas",
    )
    warnings.extend(idea_warnings)
    is_partial = is_partial or len(ideas) < len([
        i for i in ideas_raw if isinstance(i, str) and i.strip()
    ])

    themes, theme_warnings = _validate_string_list_independently(
        themes_raw, max_themes, "themes",
    )
    warnings.extend(theme_warnings)
    is_partial = is_partial or len(themes) < len([
        t for t in themes_raw if isinstance(t, str) and t.strip()
    ])

    content_output = OcrContentBatchOutput(
        images=ocr_output.images,
        communicatedIdeas=ideas,
        themes=themes,
    )
    return ValidationResult(output=content_output, warnings=warnings, is_partial=is_partial)


def validate_synthesis_output(
    raw_text: str,
    *,
    valid_evidence_ids: frozenset[str],
    max_communicated_ideas: int,
    max_themes: int,
) -> ValidationResult:
    try:
        data, warnings = _parse_json_value(raw_text)
    except ModelOutputInvalid:
        salvaged = salvage_truncated_output(raw_text)
        if salvaged is None:
            raise
        data = salvaged.data
        warnings = list(salvaged.diagnostics)
    if not isinstance(data, dict):
        raise ModelOutputInvalid("Synthesis output must be one JSON object")

    ideas_raw = data.get("communicatedIdeas", [])
    themes_raw = data.get("themes", [])

    if not isinstance(ideas_raw, list):
        warnings.append("communicatedIdeas is not a list; using empty")
        ideas_raw = []
    if not isinstance(themes_raw, list):
        warnings.append("themes is not a list; using empty")
        themes_raw = []

    valid_ideas: list[SynthesisIdeaItem] = []
    seen_ideas: set[str] = set()
    for idx, item_data in enumerate(ideas_raw):
        if not isinstance(item_data, dict):
            warnings.append(f"Rejected non-object communicatedIdeas[{idx}]")
            continue
        idea_text = item_data.get("idea", "")
        if not isinstance(idea_text, str) or not idea_text.strip():
            warnings.append(f"Rejected empty idea at communicatedIdeas[{idx}]")
            continue
        evidence = item_data.get("evidence", [])
        if not isinstance(evidence, list) or not evidence:
            warnings.append(f"Rejected idea with empty evidence at communicatedIdeas[{idx}]")
            continue
        valid_ev: list[str] = []
        seen_ev: set[str] = set()
        for ev_id in evidence:
            if not isinstance(ev_id, str):
                warnings.append(f"Ignored non-string evidence ID in communicatedIdeas[{idx}]")
                continue
            if ev_id in seen_ev:
                warnings.append(f"Removed duplicate evidence ID {ev_id} in communicatedIdeas[{idx}]")
                continue
            seen_ev.add(ev_id)
            if ev_id not in valid_evidence_ids:
                warnings.append(f"Rejected unknown evidence ID {ev_id} in communicatedIdeas[{idx}]")
                continue
            valid_ev.append(ev_id)
        if not valid_ev:
            warnings.append(
                f"Rejected idea with no valid evidence IDs at communicatedIdeas[{idx}]"
            )
            continue
        idea_key = idea_text.strip().casefold()
        if idea_key in seen_ideas:
            warnings.append(f"Removed duplicate idea: {idea_text[:80]}")
            continue
        seen_ideas.add(idea_key)
        valid_ideas.append(SynthesisIdeaItem(idea=idea_text.strip(), evidence=valid_ev))

    if len(valid_ideas) > max_communicated_ideas:
        warnings.append(
            f"Truncated communicatedIdeas from {len(valid_ideas)} to {max_communicated_ideas}"
        )
        valid_ideas = valid_ideas[:max_communicated_ideas]

    valid_themes: list[SynthesisThemeItem] = []
    seen_themes: set[str] = set()
    for idx, item_data in enumerate(themes_raw):
        if not isinstance(item_data, dict):
            warnings.append(f"Rejected non-object themes[{idx}]")
            continue
        theme_text = item_data.get("theme", "")
        if not isinstance(theme_text, str) or not theme_text.strip():
            warnings.append(f"Rejected empty theme at themes[{idx}]")
            continue
        evidence = item_data.get("evidence", [])
        if not isinstance(evidence, list) or not evidence:
            warnings.append(f"Rejected theme with empty evidence at themes[{idx}]")
            continue
        valid_ev = []
        seen_ev: set[str] = set()
        for ev_id in evidence:
            if not isinstance(ev_id, str):
                warnings.append(f"Ignored non-string evidence ID in themes[{idx}]")
                continue
            if ev_id in seen_ev:
                warnings.append(f"Removed duplicate evidence ID {ev_id} in themes[{idx}]")
                continue
            seen_ev.add(ev_id)
            if ev_id not in valid_evidence_ids:
                warnings.append(f"Rejected unknown evidence ID {ev_id} in themes[{idx}]")
                continue
            valid_ev.append(ev_id)
        if not valid_ev:
            warnings.append(f"Rejected theme with no valid evidence IDs at themes[{idx}]")
            continue
        theme_key = theme_text.strip().casefold()
        if theme_key in seen_themes:
            warnings.append(f"Removed duplicate theme: {theme_text[:80]}")
            continue
        seen_themes.add(theme_key)
        valid_themes.append(SynthesisThemeItem(theme=theme_text.strip(), evidence=valid_ev))

    if len(valid_themes) > max_themes:
        warnings.append(f"Truncated themes from {len(valid_themes)} to {max_themes}")
        valid_themes = valid_themes[:max_themes]

    output = SynthesisOutput(communicatedIdeas=valid_ideas, themes=valid_themes)
    is_partial = (
        len(valid_ideas) < len([
            i for i in ideas_raw
            if isinstance(i, dict) and isinstance(i.get("idea"), str) and i["idea"].strip()
        ])
        or len(valid_themes) < len([
            t for t in themes_raw
            if isinstance(t, dict) and isinstance(t.get("theme"), str) and t["theme"].strip()
        ])
    )
    return ValidationResult(output=output, warnings=warnings, is_partial=is_partial)


def salvage_truncated_output(raw_text: str) -> SalvageResult | None:
    """Recover only fully generated JSON values using a forward scan."""
    cleaned, _ = strip_thinking_content(raw_text)
    start = cleaned.find("{")
    if start < 0:
        return None
    fragment = cleaned[start:]
    try:
        value, consumed = json.JSONDecoder().raw_decode(fragment)
        if isinstance(value, dict):
            return SalvageResult(value, raw_text, fragment[consumed:], [])
    except json.JSONDecodeError:
        pass

    object_depth = 0
    array_depth = 0
    in_string = False
    escape = False
    candidates: list[tuple[int, str, str]] = []
    for index, char in enumerate(fragment):
        if escape:
            escape = False
            continue
        if in_string:
            if char == "\\":
                escape = True
            elif char == '"':
                in_string = False
                if object_depth == 1 and array_depth == 1:
                    candidates.append((index + 1, "]}", "complete string array element"))
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            object_depth += 1
        elif char == "}":
            object_depth -= 1
            if object_depth == 1 and array_depth == 1:
                candidates.append((index + 1, "]}", "complete object array element"))
        elif char == "[":
            array_depth += 1
        elif char == "]":
            array_depth -= 1
        elif char == "," and object_depth == 1 and array_depth == 0:
            candidates.append((index, "}", "complete top-level field"))

    for position, closing, reason in reversed(candidates):
        candidate = fragment[:position].rstrip().rstrip(",") + closing
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict) and data:
            return SalvageResult(
                data=data,
                raw_input=raw_text,
                discarded_tail=fragment[position:],
                diagnostics=[
                    f"Recovered through last {reason}; discarded incomplete tail "
                    f"starting at character {position}"
                ],
            )
    return None
