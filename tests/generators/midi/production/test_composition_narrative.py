"""Tests for composition_narrative.py and narrative_constraints.py."""

from __future__ import annotations

import pytest

from app.generators.midi.production.composition_narrative import load_narrative
from app.structures.music.narrative_constraints import (
    CompositionNarrative,
    NarrativeSection,
    extract_constraints,
    narrative_tag_adjustment,
)

# ---------------------------------------------------------------------------
# NarrativeSection validation
# ---------------------------------------------------------------------------


class TestNarrativeSection:
    def test_valid_controlled_vocabulary(self):
        sec = NarrativeSection(
            arc=0.75,
            register="mid_high",
            texture="full",
            harmonic_complexity="rich",
            rhythm_character="open",
            lead_voice="melody",
        )
        assert sec.register == "mid_high"
        assert sec.rhythm_character == "open"

    def test_invalid_register_raises(self):
        with pytest.raises(Exception):
            NarrativeSection(register="ultra_high")

    def test_invalid_texture_raises(self):
        with pytest.raises(Exception):
            NarrativeSection(texture="totally_empty")

    def test_missing_fields_default_to_none(self):
        sec = NarrativeSection()
        assert sec.register is None
        assert sec.texture is None
        assert sec.harmonic_complexity is None
        assert sec.rhythm_character is None
        assert sec.lead_voice is None

    def test_narrative_text_field(self):
        sec = NarrativeSection(narrative="This section is about grief.")
        assert sec.narrative == "This section is about grief."


# ---------------------------------------------------------------------------
# CompositionNarrative.from_dict
# ---------------------------------------------------------------------------


class TestCompositionNarrativeFromDict:
    def test_roundtrip_minimal(self):
        data = {
            "schema_version": "1",
            "generated_by": "claude",
            "rationale": "This is a grief song.",
            "the_moment": {"section": "chorus_2", "description": "The peak."},
            "sections": {
                "verse": {
                    "arc": 0.30,
                    "register": "low_mid",
                    "texture": "sparse",
                    "harmonic_complexity": "simple",
                    "rhythm_character": "minimal",
                    "lead_voice": "melody",
                    "narrative": "quiet opening",
                },
                "chorus": {
                    "arc": 0.75,
                    "register": "mid",
                    "texture": "full",
                    "harmonic_complexity": "rich",
                    "rhythm_character": "present",
                    "lead_voice": "melody",
                },
            },
        }
        narrative = CompositionNarrative.from_dict(data)
        assert narrative.rationale == "This is a grief song."
        assert narrative.the_moment.section == "chorus_2"
        assert "verse" in narrative.sections
        assert narrative.sections["verse"].arc == pytest.approx(0.30)
        assert narrative.sections["chorus"].texture == "full"

    def test_empty_sections(self):
        narrative = CompositionNarrative.from_dict({})
        assert narrative.sections == {}
        assert narrative.the_moment is None


# ---------------------------------------------------------------------------
# extract_constraints
# ---------------------------------------------------------------------------


class TestExtractConstraints:
    def _make_narrative(self, section_data: dict) -> CompositionNarrative:
        return CompositionNarrative.from_dict({"sections": {"verse": section_data}})

    def test_rhythm_minimal_returns_ghost_prefer(self):
        narrative = self._make_narrative({"rhythm_character": "minimal"})
        c = extract_constraints("verse", narrative)
        assert "ghost_only" in c.get("rhythm_prefer", [])
        assert "dense" in c.get("rhythm_penalise", [])

    def test_rhythm_absent_penalises_dense(self):
        narrative = self._make_narrative({"rhythm_character": "absent"})
        c = extract_constraints("verse", narrative)
        assert "dense" in c.get("rhythm_penalise", [])

    def test_lead_voice_bass_prefers_walking(self):
        narrative = self._make_narrative({"lead_voice": "bass"})
        c = extract_constraints("verse", narrative)
        assert "walking" in c.get("bass_prefer", [])

    def test_lead_voice_melody_prefers_drone(self):
        narrative = self._make_narrative({"lead_voice": "melody"})
        c = extract_constraints("verse", narrative)
        assert "drone" in c.get("bass_prefer", [])
        assert "walking" in c.get("bass_penalise", [])

    def test_lead_voice_none_sets_skip_melody(self):
        narrative = self._make_narrative({"lead_voice": "none"})
        c = extract_constraints("verse", narrative)
        assert c.get("skip_melody") is True

    def test_register_low_prefers_descent(self):
        narrative = self._make_narrative({"register": "low"})
        c = extract_constraints("verse", narrative)
        assert "descent" in c.get("melody_prefer", [])

    def test_register_high_prefers_wide_interval(self):
        narrative = self._make_narrative({"register": "high"})
        c = extract_constraints("verse", narrative)
        assert "wide_interval" in c.get("melody_prefer", [])

    def test_missing_section_returns_empty(self):
        narrative = CompositionNarrative.from_dict({"sections": {}})
        assert extract_constraints("chorus", narrative) == {}


# ---------------------------------------------------------------------------
# narrative_tag_adjustment
# ---------------------------------------------------------------------------


class TestNarrativeTagAdjustment:
    def test_drums_minimal_boosts_ghost(self):
        narrative = CompositionNarrative.from_dict(
            {"sections": {"verse": {"rhythm_character": "minimal"}}}
        )
        c = extract_constraints("verse", narrative)
        adj = narrative_tag_adjustment(c, ["ghost_only", "sparse"], "drums")
        assert adj == pytest.approx(0.10)

    def test_drums_busy_penalises_sparse(self):
        narrative = CompositionNarrative.from_dict(
            {"sections": {"chorus": {"rhythm_character": "busy"}}}
        )
        c = extract_constraints("chorus", narrative)
        adj = narrative_tag_adjustment(c, ["sparse"], "drums")
        assert adj == pytest.approx(-0.05)

    def test_bass_lead_voice_bass_boosts_walking(self):
        narrative = CompositionNarrative.from_dict(
            {"sections": {"bridge": {"lead_voice": "bass"}}}
        )
        c = extract_constraints("bridge", narrative)
        adj = narrative_tag_adjustment(c, ["walking"], "bass")
        assert adj == pytest.approx(0.10)

    def test_melody_low_register_boosts_descent(self):
        narrative = CompositionNarrative.from_dict(
            {"sections": {"verse": {"register": "low"}}}
        )
        c = extract_constraints("verse", narrative)
        adj = narrative_tag_adjustment(c, ["descent"], "melody")
        assert adj == pytest.approx(0.10)

    def test_empty_constraints_returns_zero(self):
        assert narrative_tag_adjustment({}, ["walking", "arpeggiated"], "bass") == 0.0

    def test_empty_tags_returns_zero(self):
        narrative = CompositionNarrative.from_dict(
            {"sections": {"verse": {"rhythm_character": "minimal"}}}
        )
        c = extract_constraints("verse", narrative)
        assert narrative_tag_adjustment(c, [], "drums") == 0.0


# ---------------------------------------------------------------------------
# load_narrative
# ---------------------------------------------------------------------------


class TestLoadNarrative:
    def test_returns_none_when_absent(self, tmp_path):
        assert load_narrative(tmp_path) is None

    def test_loads_valid_narrative(self, tmp_path):
        import yaml

        data = {
            "schema_version": "1",
            "sections": {"verse": {"rhythm_character": "minimal"}},
        }
        (tmp_path / "composition_narrative.yml").write_text(
            yaml.dump(data), encoding="utf-8"
        )
        narrative = load_narrative(tmp_path)
        assert narrative is not None
        assert "verse" in narrative.sections
