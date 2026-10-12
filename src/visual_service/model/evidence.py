from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class EvidenceItem:
    id: str
    text: str
    frame_sequence: int


@dataclass
class EvidenceSet:
    items: list[EvidenceItem]
    evidence_block: str
    id_set: frozenset[str]


def prepare_evidence(
    visible_texts: list[str],
    frame_sequences: list[int] | None,
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
    for index, (text, frame) in enumerate(deduped):
        item_id = f"text-{index + 1:04d}"
        items.append(EvidenceItem(id=item_id, text=text, frame_sequence=frame))
        lines.append(f"[{item_id}] {text}")

    evidence_block = "\n".join(lines)
    id_set = frozenset(item.id for item in items)
    return EvidenceSet(items=items, evidence_block=evidence_block, id_set=id_set)


def estimate_evidence_tokens(evidence_block: str, processor: Any) -> int:
    token_ids = processor.tokenizer.encode(evidence_block, add_special_tokens=False)
    return len(token_ids)
