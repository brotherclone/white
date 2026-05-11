"""Tests for app/tools/candidate_browser.py — data layer only (no UI)."""

from pathlib import Path

import yaml

from white_api.candidate_browser import (
    CandidateEntry,
    approve_candidate,
    load_all_candidates,
    reject_candidate,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_review(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(data, f, allow_unicode=True, sort_keys=False)


def _make_candidate(
    id: str,
    status: str = "pending",
    rank: int = 1,
    composite: float = 0.5,
    section: str = "",
    pattern_name: str = "test_pattern",
) -> dict:
    return {
        "id": id,
        "midi_file": f"candidates/{id}.mid",
        "rank": rank,
        "section": section,
        "pattern_name": pattern_name,
        "status": status,
        "scores": {"composite": composite, "theory": {}, "chromatic": {}},
    }


# ---------------------------------------------------------------------------
# load_all_candidates
# ---------------------------------------------------------------------------


class TestLoadAllCandidates:
    def test_loads_candidates_from_all_phases(self, tmp_path):
        prod = tmp_path / "song_v1"
        _write_review(
            prod / "chords" / "review.yml",
            {
                "candidates": [
                    _make_candidate("chord_001"),
                    _make_candidate("chord_002"),
                ]
            },
        )
        _write_review(
            prod / "melody" / "review.yml",
            {"candidates": [_make_candidate("mel_001", section="intro")]},
        )
        entries = load_all_candidates(prod)
        assert len(entries) == 3

    def test_phase_filter(self, tmp_path):
        prod = tmp_path / "song_v1"
        _write_review(
            prod / "chords" / "review.yml",
            {"candidates": [_make_candidate("chord_001")]},
        )
        _write_review(
            prod / "drums" / "review.yml",
            {"candidates": [_make_candidate("drum_001")]},
        )
        entries = load_all_candidates(prod, phase_filter="chords")
        assert len(entries) == 1
        assert entries[0].phase == "chords"

    def test_section_filter(self, tmp_path):
        prod = tmp_path / "song_v1"
        _write_review(
            prod / "melody" / "review.yml",
            {
                "candidates": [
                    _make_candidate("mel_001", section="intro"),
                    _make_candidate("mel_002", section="verse"),
                ]
            },
        )
        entries = load_all_candidates(prod, section_filter="intro")
        assert len(entries) == 1
        assert entries[0].section == "intro"

    def test_missing_phase_dir_skipped(self, tmp_path):
        prod = tmp_path / "song_v1"
        _write_review(
            prod / "chords" / "review.yml",
            {"candidates": [_make_candidate("chord_001")]},
        )
        # drums phase dir does not exist — should be silently skipped
        entries = load_all_candidates(prod)
        assert all(e.phase == "chords" for e in entries)

    def test_empty_production_dir_returns_empty(self, tmp_path):
        prod = tmp_path / "empty_song"
        prod.mkdir()
        assert load_all_candidates(prod) == []

    def test_sorted_by_phase_then_section_then_rank(self, tmp_path):
        prod = tmp_path / "song_v1"
        _write_review(
            prod / "chords" / "review.yml",
            {
                "candidates": [
                    _make_candidate("chord_002", rank=2),
                    _make_candidate("chord_001", rank=1),
                ]
            },
        )
        _write_review(
            prod / "drums" / "review.yml",
            {"candidates": [_make_candidate("drum_001")]},
        )
        entries = load_all_candidates(prod)
        phases = [e.phase for e in entries]
        assert phases.index("chords") < phases.index("drums")
        # Within chords, rank 1 before rank 2
        chord_entries = [e for e in entries if e.phase == "chords"]
        assert chord_entries[0].rank == 1
        assert chord_entries[1].rank == 2

    def test_composite_score_parsed(self, tmp_path):
        prod = tmp_path / "song_v1"
        _write_review(
            prod / "chords" / "review.yml",
            {"candidates": [_make_candidate("chord_001", composite=0.7654)]},
        )
        entries = load_all_candidates(prod)
        assert abs(entries[0].composite_score - 0.7654) < 1e-4

    def test_status_read_correctly(self, tmp_path):
        prod = tmp_path / "song_v1"
        _write_review(
            prod / "chords" / "review.yml",
            {
                "candidates": [
                    _make_candidate("chord_001", status="approved"),
                    _make_candidate("chord_002", status="rejected"),
                    _make_candidate("chord_003", status="pending"),
                ]
            },
        )
        entries = load_all_candidates(prod)
        statuses = {e.candidate_id: e.status for e in entries}
        assert statuses["chord_001"] == "approved"
        assert statuses["chord_002"] == "rejected"
        assert statuses["chord_003"] == "pending"

    def test_corrupt_review_yml_skipped(self, tmp_path):
        prod = tmp_path / "song_v1"
        phase_dir = prod / "chords"
        phase_dir.mkdir(parents=True)
        (phase_dir / "review.yml").write_text("{{{{not yaml")
        entries = load_all_candidates(prod)
        assert entries == []


# ---------------------------------------------------------------------------
# approve_candidate / reject_candidate
# ---------------------------------------------------------------------------


class TestApproveCandidate:
    def _make_entry(self, tmp_path: Path, status: str = "pending") -> CandidateEntry:
        review_yml = tmp_path / "chords" / "review.yml"
        _write_review(
            review_yml,
            {"candidates": [_make_candidate("chord_001", status=status)]},
        )
        entries = load_all_candidates(tmp_path)
        return entries[0]

    def test_approve_writes_to_yml(self, tmp_path):
        entry = self._make_entry(tmp_path)
        approve_candidate(entry)
        with open(entry.review_yml) as f:
            data = yaml.safe_load(f)
        c = next(c for c in data["candidates"] if c["id"] == "chord_001")
        assert c["status"] == "approved"

    def test_approve_updates_entry_in_place(self, tmp_path):
        entry = self._make_entry(tmp_path)
        approve_candidate(entry)
        assert entry.status == "approved"

    def test_reject_writes_to_yml(self, tmp_path):
        entry = self._make_entry(tmp_path)
        reject_candidate(entry)
        with open(entry.review_yml) as f:
            data = yaml.safe_load(f)
        c = next(c for c in data["candidates"] if c["id"] == "chord_001")
        assert c["status"] == "rejected"

    def test_reject_updates_entry_in_place(self, tmp_path):
        entry = self._make_entry(tmp_path)
        reject_candidate(entry)
        assert entry.status == "rejected"

    def test_approve_preserves_other_candidates(self, tmp_path):
        review_yml = tmp_path / "chords" / "review.yml"
        _write_review(
            review_yml,
            {
                "candidates": [
                    _make_candidate("chord_001", status="pending"),
                    _make_candidate("chord_002", status="pending"),
                ]
            },
        )
        entries = load_all_candidates(tmp_path)
        approve_candidate(entries[0])

        with open(review_yml) as f:
            data = yaml.safe_load(f)
        statuses = {c["id"]: c["status"] for c in data["candidates"]}
        assert statuses["chord_001"] == "approved"
        assert statuses["chord_002"] == "pending"

    def test_re_approve_already_approved(self, tmp_path):
        entry = self._make_entry(tmp_path, status="approved")
        approve_candidate(entry)  # should not raise
        assert entry.status == "approved"


# ---------------------------------------------------------------------------
# Non-generated entries (generated: false, null scores/rank)
# ---------------------------------------------------------------------------


class TestNonGeneratedEntries:
    def _write_non_generated_review(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(
                {
                    "candidates": [
                        {
                            "id": "hand_chorus_v1",
                            "midi_file": "approved/chorus_v1.mid",
                            "section": "chorus",
                            "label": "chorus_v1",
                            "status": "approved",
                            "generated": False,
                            "scores": None,
                            "rank": None,
                            "notes": "Non-generated part",
                        }
                    ]
                },
                f,
                allow_unicode=True,
                sort_keys=False,
            )

    def test_null_scores_does_not_raise(self, tmp_path):
        prod = tmp_path / "song_v1"
        self._write_non_generated_review(prod / "melody" / "review.yml")
        entries = load_all_candidates(prod, phase_filter="melody")
        assert len(entries) == 1

    def test_null_rank_defaults_to_99(self, tmp_path):
        prod = tmp_path / "song_v1"
        self._write_non_generated_review(prod / "melody" / "review.yml")
        entries = load_all_candidates(prod, phase_filter="melody")
        assert entries[0].rank == 99

    def test_null_scores_composite_defaults_to_zero(self, tmp_path):
        prod = tmp_path / "song_v1"
        self._write_non_generated_review(prod / "melody" / "review.yml")
        entries = load_all_candidates(prod, phase_filter="melody")
        assert entries[0].composite_score == 0.0

    def test_generated_flag_false_for_non_generated(self, tmp_path):
        prod = tmp_path / "song_v1"
        self._write_non_generated_review(prod / "melody" / "review.yml")
        entries = load_all_candidates(prod, phase_filter="melody")
        assert entries[0].generated is False

    def test_generated_flag_true_by_default(self, tmp_path):
        prod = tmp_path / "song_v1"
        _write_review(
            prod / "chords" / "review.yml",
            {"candidates": [_make_candidate("chord_001")]},
        )
        entries = load_all_candidates(prod, phase_filter="chords")
        assert entries[0].generated is True
