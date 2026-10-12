from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class EvidenceItem:
    id: str
    text: str
    frame_sequence: int


@dataclass(frozen=True)
class EvidenceGroupInfo:
    item_count: int
    first_frame: int
    last_frame: int
    span: int
    boundary_reason: str


@dataclass
class EvidenceSet:
    items: list[EvidenceItem]
    evidence_block: str
    id_set: frozenset[str]
    group_count: int = 0
    groups: list[EvidenceGroupInfo] = field(default_factory=list)


def prepare_evidence(
    visible_texts: list[str],
    frame_sequences: list[int] | None,
    *,
    group_gap: int = 2,
    max_group_span: int = 15,
    max_group_items: int = 25,
) -> EvidenceSet:
    if not visible_texts:
        return EvidenceSet(items=[], evidence_block="", id_set=frozenset())

    frames = frame_sequences if frame_sequences is not None else list(range(len(visible_texts)))
    pairs = list(zip(visible_texts, frames))
    pairs.sort(key=lambda p: p[1])

    deduped: list[tuple[str, int]] = []
    for text, frame in pairs:
        if deduped and deduped[-1][0] == text:
            continue
        deduped.append((text, frame))

    items: list[EvidenceItem] = []
    lines: list[str] = []
    groups: list[EvidenceGroupInfo] = []

    group_anchor_frame = deduped[0][1]
    group_first_frame = deduped[0][1]
    group_last_frame = deduped[0][1]
    group_item_count = 0
    group_boundary_reason = "start"

    for index, (text, frame) in enumerate(deduped):
        item_id = f"text-{index + 1:04d}"
        items.append(EvidenceItem(id=item_id, text=text, frame_sequence=frame))

        if index > 0:
            prev_frame = deduped[index - 1][1]
            reason = None
            if (frame - prev_frame) > group_gap:
                reason = "gap"
            elif (frame - group_anchor_frame) > max_group_span:
                reason = "span"
            elif group_item_count >= max_group_items:
                reason = "item_count"

            if reason is not None:
                groups.append(EvidenceGroupInfo(
                    item_count=group_item_count,
                    first_frame=group_first_frame,
                    last_frame=group_last_frame,
                    span=group_last_frame - group_first_frame,
                    boundary_reason=group_boundary_reason,
                ))
                lines.append("")
                group_anchor_frame = frame
                group_first_frame = frame
                group_item_count = 0
                group_boundary_reason = reason

        group_last_frame = frame
        group_item_count += 1
        lines.append(f"[{item_id}] {text}")

    groups.append(EvidenceGroupInfo(
        item_count=group_item_count,
        first_frame=group_first_frame,
        last_frame=group_last_frame,
        span=group_last_frame - group_first_frame,
        boundary_reason=group_boundary_reason,
    ))

    evidence_block = "\n".join(lines)
    id_set = frozenset(item.id for item in items)
    return EvidenceSet(
        items=items,
        evidence_block=evidence_block,
        id_set=id_set,
        group_count=len(groups),
        groups=groups,
    )


def estimate_evidence_tokens(evidence_block: str, processor: Any) -> int:
    token_ids = processor.tokenizer.encode(evidence_block, add_special_tokens=False)
    return len(token_ids)
