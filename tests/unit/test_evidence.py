import pytest

from visual_service.model.evidence import EvidenceGroupInfo, EvidenceItem, EvidenceSet, prepare_evidence


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

    def test_evidence_groups_separated_by_blank_lines(self):
        texts = ["A", "B", "C", "D"]
        frames = [0, 1, 5, 6]
        result = prepare_evidence(texts, frames)
        lines = result.evidence_block.split("\n")
        assert "" in lines
        assert result.group_count == 2

    def test_evidence_nearby_frames_in_same_group(self):
        texts = ["A", "B", "C"]
        frames = [0, 1, 2]
        result = prepare_evidence(texts, frames)
        assert "" not in result.evidence_block.split("\n")
        assert result.group_count == 1

    def test_evidence_distant_frames_in_different_groups(self):
        texts = ["A", "B", "C"]
        frames = [0, 10, 20]
        result = prepare_evidence(texts, frames)
        lines = result.evidence_block.split("\n")
        blank_count = sum(1 for line in lines if line == "")
        assert blank_count == 2
        assert result.group_count == 3

    def test_evidence_single_item_group_count(self):
        result = prepare_evidence(["hello"], [0])
        assert result.group_count == 1

    def test_evidence_empty_group_count(self):
        result = prepare_evidence([], None)
        assert result.group_count == 0

    def test_evidence_preserves_incidental_text(self):
        texts = ["CALIFORNIA", "123 Main St", "P22"]
        frames = [0, 1, 2]
        result = prepare_evidence(texts, frames)
        assert len(result.items) == 3
        assert result.items[0].text == "CALIFORNIA"
        assert result.items[1].text == "123 Main St"
        assert result.items[2].text == "P22"

    def test_evidence_custom_group_gap(self):
        texts = ["A", "B", "C"]
        frames = [0, 2, 4]
        result_gap1 = prepare_evidence(texts, frames, group_gap=1)
        result_gap3 = prepare_evidence(texts, frames, group_gap=3)
        assert result_gap1.group_count == 3
        assert result_gap3.group_count == 1

    def test_transitive_chain_split_at_span_limit(self):
        texts = [f"w{i}" for i in range(20)]
        frames = list(range(20))
        result = prepare_evidence(texts, frames, max_group_span=10)
        assert result.group_count >= 2
        for g in result.groups:
            assert g.span <= 10

    def test_max_items_per_group_enforcement(self):
        texts = [f"w{i}" for i in range(30)]
        frames = list(range(30))
        result = prepare_evidence(texts, frames, max_group_items=10)
        assert result.group_count == 3
        for g in result.groups:
            assert g.item_count <= 10

    def test_stable_ids_after_splitting(self):
        texts = [f"w{i}" for i in range(20)]
        frames = list(range(20))
        result = prepare_evidence(texts, frames, max_group_span=5)
        ids = [item.id for item in result.items]
        expected = [f"text-{i+1:04d}" for i in range(20)]
        assert ids == expected

    def test_146_item_chain_cannot_be_one_group(self):
        texts = [f"w{i}" for i in range(146)]
        frames = list(range(146))
        result = prepare_evidence(texts, frames)
        assert result.group_count > 1
        for g in result.groups:
            assert g.item_count <= 25

    def test_group_ordering_deterministic(self):
        texts = [f"w{i}" for i in range(50)]
        frames = list(range(50))
        r1 = prepare_evidence(texts, frames)
        r2 = prepare_evidence(texts, frames)
        assert [(g.item_count, g.first_frame, g.last_frame) for g in r1.groups] == \
               [(g.item_count, g.first_frame, g.last_frame) for g in r2.groups]

    def test_no_evidence_deleted_by_splitting(self):
        texts = [f"w{i}" for i in range(50)]
        frames = list(range(50))
        result = prepare_evidence(texts, frames, max_group_span=10)
        assert len(result.items) == 50

    def test_evidence_group_info_metadata(self):
        texts = ["A", "B", "C", "D"]
        frames = [0, 1, 10, 11]
        result = prepare_evidence(texts, frames)
        assert len(result.groups) == 2
        g0 = result.groups[0]
        assert g0.item_count == 2
        assert g0.first_frame == 0
        assert g0.last_frame == 1
        assert g0.span == 1
        assert g0.boundary_reason == "start"
        g1 = result.groups[1]
        assert g1.item_count == 2
        assert g1.first_frame == 10
        assert g1.last_frame == 11
        assert g1.boundary_reason == "gap"

    def test_group_count_equals_len_groups(self):
        for n in [1, 5, 20, 50]:
            texts = [f"w{i}" for i in range(n)]
            frames = list(range(n))
            result = prepare_evidence(texts, frames)
            assert result.group_count == len(result.groups)

    def test_backward_compat_gap_only(self):
        texts = ["A", "B", "C", "D"]
        frames = [0, 1, 10, 11]
        result = prepare_evidence(texts, frames, max_group_span=9999, max_group_items=9999)
        assert result.group_count == 2
        assert result.groups[0].boundary_reason == "start"
        assert result.groups[1].boundary_reason == "gap"
