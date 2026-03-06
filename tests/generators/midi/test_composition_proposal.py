"""Tests for composition_proposal.py."""

from __future__ import annotations

import textwrap
from pathlib import Path

import yaml

from app.generators.midi.composition_proposal import (
    build_loop_inventory,
    parse_response,
    write_proposal,
)
from app.generators.midi.assembly_manifest import (
    ArrangementSection,
    compute_proposal_drift,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_review(
    path: Path, candidates: list[dict], extra: dict | None = None
) -> None:
    data = {"candidates": candidates, **(extra or {})}
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(data, f)


def _write_proposal_yml(path: Path, sections: list[dict]) -> None:
    data = {"proposed_by": "claude", "proposed_sections": sections}
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(data, f)


# ---------------------------------------------------------------------------
# build_loop_inventory
# ---------------------------------------------------------------------------


class TestBuildLoopInventory:
    def test_approved_loops_collected(self, tmp_path):
        _write_review(
            tmp_path / "chords" / "review.yml",
            [
                {
                    "id": "c01",
                    "label": "verse",
                    "status": "approved",
                    "scores": {"composite": 0.75},
                    "energy": "medium",
                    "notes": "",
                },
                {
                    "id": "c02",
                    "label": "chorus",
                    "status": "rejected",
                    "scores": {"composite": 0.5},
                    "energy": "high",
                    "notes": "",
                },
            ],
        )
        inventory = build_loop_inventory(tmp_path)
        assert "chords" in inventory
        assert len(inventory["chords"]) == 1
        assert inventory["chords"][0]["label"] == "verse"
        assert inventory["chords"][0]["score"] == 0.75

    def test_rejected_loops_excluded(self, tmp_path):
        _write_review(
            tmp_path / "drums" / "review.yml",
            [
                {
                    "id": "d01",
                    "label": "beat",
                    "status": "rejected",
                    "scores": {"composite": 0.9},
                    "energy": "high",
                    "notes": "",
                }
            ],
        )
        inventory = build_loop_inventory(tmp_path)
        assert "drums" not in inventory

    def test_missing_instrument_omitted(self, tmp_path):
        _write_review(
            tmp_path / "bass" / "review.yml",
            [
                {
                    "id": "b01",
                    "label": "groove",
                    "status": "approved",
                    "scores": {"composite": 0.6},
                    "energy": "low",
                    "notes": "driving",
                }
            ],
        )
        inventory = build_loop_inventory(tmp_path)
        assert "bass" in inventory
        assert "chords" not in inventory
        assert "drums" not in inventory

    def test_energy_and_notes_preserved(self, tmp_path):
        _write_review(
            tmp_path / "melody" / "review.yml",
            [
                {
                    "id": "m01",
                    "label": "hook",
                    "status": "approved",
                    "scores": {"composite": 0.8},
                    "energy": "high",
                    "notes": "punchy",
                }
            ],
        )
        inventory = build_loop_inventory(tmp_path)
        loop = inventory["melody"][0]
        assert loop["energy"] == "high"
        assert loop["notes"] == "punchy"


# ---------------------------------------------------------------------------
# parse_response
# ---------------------------------------------------------------------------


class TestParseResponse:
    def test_valid_yaml_block_extracted(self):
        raw = textwrap.dedent(
            """\
            Here is my proposal:

            ```yaml
            sounds_like:
              - Scott Walker
              - Ennio Morricone
            proposed_sections:
              - name: verse
                repeat: 2
                energy_note: sparse
                transition_note: builds into chorus
                loops:
                  chords: verse_chords
                  drums: beat
                  bass: null
                  melody: hook
            ```

            Rationale: The verse repeats twice to establish the drone before release.
        """
        )
        structured, rationale = parse_response(raw)
        assert structured["sounds_like"] == ["Scott Walker", "Ennio Morricone"]
        assert len(structured["proposed_sections"]) == 1
        assert structured["proposed_sections"][0]["name"] == "verse"
        assert "drone" in rationale

    def test_malformed_yaml_returns_empty_structured(self):
        raw = "```yaml\n: invalid: : yaml\n```\n\nRationale: something"
        structured, rationale = parse_response(raw)
        assert structured == {}
        assert "something" in rationale

    def test_no_yaml_block_stores_raw_as_rationale(self):
        raw = "I propose you use a long intro and build slowly."
        structured, rationale = parse_response(raw)
        assert structured == {}
        assert "long intro" in rationale

    def test_sounds_like_in_structured(self):
        raw = textwrap.dedent(
            """\
            ```yaml
            sounds_like:
              - Radiohead
            proposed_sections: []
            ```
            Rationale: minimal
        """
        )
        structured, _ = parse_response(raw)
        assert structured.get("sounds_like") == ["Radiohead"]


# ---------------------------------------------------------------------------
# write_proposal
# ---------------------------------------------------------------------------


class TestWriteProposal:
    def test_file_written_with_required_keys(self, tmp_path):
        proposal_meta = {"color": "Blue", "title": "Test Song"}
        inventory = {
            "chords": [
                {"label": "verse", "bars": 4, "score": 0.7, "energy": "", "notes": ""}
            ]
        }
        structured = {"sounds_like": ["Tom Waits"], "proposed_sections": []}
        rationale = "Go slow and drone."

        out = write_proposal(tmp_path, proposal_meta, inventory, structured, rationale)

        assert out.exists()
        with open(out) as f:
            data = yaml.safe_load(f)

        assert data["proposed_by"] == "claude"
        assert data["color_target"] == "Blue"
        assert data["sounds_like"] == ["Tom Waits"]
        assert data["rationale"] == "Go slow and drone."
        assert "generated" in data
        assert "loop_inventory" in data

    def test_empty_structured_writes_empty_sections(self, tmp_path):
        out = write_proposal(
            tmp_path, {"color": "Red", "title": ""}, {}, {}, "raw fallback"
        )
        with open(out) as f:
            data = yaml.safe_load(f)
        assert data["proposed_sections"] == []
        assert data["sounds_like"] == []


# ---------------------------------------------------------------------------
# compute_proposal_drift
# ---------------------------------------------------------------------------


class TestComputeProposalDrift:
    def _make_section(self, name: str) -> ArrangementSection:
        s = ArrangementSection.__new__(ArrangementSection)
        s.name = name
        s.start = 0.0
        s.end = 10.0
        s.vocals = False
        s.loops = {}
        return s

    def test_sections_added(self, tmp_path):
        _write_proposal_yml(
            tmp_path / "composition_proposal.yml",
            [
                {"name": "verse", "repeat": 1},
            ],
        )
        actual = [self._make_section("verse"), self._make_section("chorus")]
        result = compute_proposal_drift(tmp_path / "composition_proposal.yml", actual)
        assert "chorus" in result["sections_added"]
        assert result["sections_removed"] == []

    def test_sections_removed(self, tmp_path):
        _write_proposal_yml(
            tmp_path / "composition_proposal.yml",
            [
                {"name": "verse", "repeat": 1},
                {"name": "outro", "repeat": 1},
            ],
        )
        actual = [self._make_section("verse")]
        result = compute_proposal_drift(tmp_path / "composition_proposal.yml", actual)
        assert "outro" in result["sections_removed"]
        assert result["sections_added"] == []

    def test_repeat_delta_detected(self, tmp_path):
        _write_proposal_yml(
            tmp_path / "composition_proposal.yml",
            [
                {"name": "verse", "repeat": 3},
            ],
        )
        actual = [self._make_section("verse"), self._make_section("verse")]
        result = compute_proposal_drift(tmp_path / "composition_proposal.yml", actual)
        assert any(
            d["name"] == "verse" and d["proposed"] == 3 and d["actual"] == 2
            for d in result["repeat_deltas"]
        )

    def test_order_changed(self, tmp_path):
        _write_proposal_yml(
            tmp_path / "composition_proposal.yml",
            [
                {"name": "verse", "repeat": 1},
                {"name": "chorus", "repeat": 1},
            ],
        )
        actual = [self._make_section("chorus"), self._make_section("verse")]
        result = compute_proposal_drift(tmp_path / "composition_proposal.yml", actual)
        assert result["order_changed"] is True

    def test_order_unchanged(self, tmp_path):
        _write_proposal_yml(
            tmp_path / "composition_proposal.yml",
            [
                {"name": "verse", "repeat": 1},
                {"name": "chorus", "repeat": 1},
            ],
        )
        actual = [self._make_section("verse"), self._make_section("chorus")]
        result = compute_proposal_drift(tmp_path / "composition_proposal.yml", actual)
        assert result["order_changed"] is False

    def test_missing_proposal_returns_none(self, tmp_path):
        actual = [self._make_section("verse")]
        result = compute_proposal_drift(tmp_path / "nonexistent.yml", actual)
        assert result is None

    def test_empty_proposed_sections_returns_none(self, tmp_path):
        path = tmp_path / "composition_proposal.yml"
        with open(path, "w") as f:
            yaml.dump({"proposed_by": "claude", "proposed_sections": []}, f)
        actual = [self._make_section("verse")]
        result = compute_proposal_drift(path, actual)
        assert result is None
