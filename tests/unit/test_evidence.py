import pytest

from visual_service.model.evidence import EvidenceItem, EvidenceSet, prepare_evidence


class TestPrepareEvidence:

    def test_empty_texts_produce_empty_evidence(self):
        result = prepare_evidence([], None)
        assert result.items == []
        assert result.evidence_block == ""
        assert result.id_set == frozenset()

    def test_single_text_produces_one_item(self):
        result = prepare_evidence(["Hello"], [0])
        assert len(result.items) == 1
        assert result.items[0].id == "text-0001"
        assert result.items[0].text == "Hello"
        assert result.items[0].frame_sequence == 0

    def test_evidence_ids_are_zero_padded(self):
        texts = [f"text{i}" for i in range(12)]
        result = prepare_evidence(texts, list(range(12)))
        assert result.items[0].id == "text-0001"
        assert result.items[9].id == "text-0010"
        assert result.items[11].id == "text-0012"

    def test_evidence_preserves_chronological_order(self):
        texts = ["B", "A", "C"]
        frames = [2, 0, 1]
        result = prepare_evidence(texts, frames)
        assert [item.text for item in result.items] == ["A", "C", "B"]

    def test_exact_consecutive_duplicates_removed(self):
        texts = ["HELLO", "HELLO", "WORLD"]
        frames = [0, 1, 2]
        result = prepare_evidence(texts, frames)
        assert len(result.items) == 2
        assert result.items[0].text == "HELLO"
        assert result.items[1].text == "WORLD"

    def test_nonconsecutive_repetitions_preserved(self):
        texts = ["HELLO", "WORLD", "HELLO"]
        frames = [0, 1, 2]
        result = prepare_evidence(texts, frames)
        assert len(result.items) == 3
        assert result.items[0].text == "HELLO"
        assert result.items[2].text == "HELLO"

    def test_near_duplicates_not_removed(self):
        texts = ["HELLO", "HELLO!"]
        frames = [0, 1]
        result = prepare_evidence(texts, frames)
        assert len(result.items) == 2

    def test_evidence_block_format(self):
        texts = ["El", "problema"]
        frames = [0, 1]
        result = prepare_evidence(texts, frames)
        lines = result.evidence_block.split("\n")
        assert lines[0] == "[text-0001] El"
        assert lines[1] == "[text-0002] problema"

    def test_evidence_id_set_matches_items(self):
        texts = ["A", "B", "C"]
        result = prepare_evidence(texts, [0, 1, 2])
        expected = frozenset(["text-0001", "text-0002", "text-0003"])
        assert result.id_set == expected

    def test_evidence_without_frame_sequences(self):
        texts = ["A", "B", "C"]
        result = prepare_evidence(texts, None)
        assert len(result.items) == 3
        assert result.items[0].frame_sequence == 0
        assert result.items[1].frame_sequence == 1
        assert result.items[2].frame_sequence == 2

    def test_evidence_large_set_ids_sequential(self):
        texts = [f"word{i}" for i in range(100)]
        result = prepare_evidence(texts, list(range(100)))
        assert len(result.items) == 100
        assert result.items[0].id == "text-0001"
        assert result.items[99].id == "text-0100"

    def test_rejected_strings_excluded_by_caller(self):
        texts = ["accepted"]
        result = prepare_evidence(texts, [0])
        assert len(result.items) == 1
        assert result.items[0].text == "accepted"

    def test_evidence_text_not_translated_or_corrected(self):
        texts = ["El problema demasiado"]
        result = prepare_evidence(texts, [0])
        assert result.items[0].text == "El problema demasiado"

    def test_multiple_consecutive_duplicates_across_frames(self):
        texts = ["A", "A", "A", "B", "B", "C"]
        frames = [0, 1, 2, 3, 4, 5]
        result = prepare_evidence(texts, frames)
        assert [item.text for item in result.items] == ["A", "B", "C"]

    def test_stable_sort_preserves_order_within_same_frame(self):
        texts = ["first", "second"]
        frames = [0, 0]
        result = prepare_evidence(texts, frames)
        assert [item.text for item in result.items] == ["first", "second"]
