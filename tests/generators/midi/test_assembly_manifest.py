"""Tests for assembly_manifest.py — Logic arrangement import."""

from __future__ import annotations

import textwrap

import pytest
import yaml

from app.generators.midi.assembly_manifest import (
    Clip,
    _tc_to_seconds,
    compute_drift,
    derive_sections,
    import_arrangement,
    parse_arrangement,
)
from app.generators.midi.production_plan import (
    PlanSection,
    ProductionPlan,
    load_plan,
    save_plan,
)


# ---------------------------------------------------------------------------
# _tc_to_seconds
# ---------------------------------------------------------------------------


class TestTcToSeconds:
    def test_four_part_with_offset(self):
        # Logic position 01:00:50:00.00 → 1*3600 + 50 = 3650
        assert _tc_to_seconds("01:00:50:00.00") == 3650.0

    def test_four_part_zero(self):
        assert _tc_to_seconds("01:00:00:00.00") == 3600.0

    def test_length_format(self):
        # Lengths use 00: prefix: 00:00:10:00.00 → 10s
        assert _tc_to_seconds("00:00:10:00.00") == 10.0

    def test_length_twenty_seconds(self):
        assert _tc_to_seconds("00:00:20:00.00") == 20.0

    def test_minutes_carry(self):
        # 01:01:10:00.00 → 3600 + 70 = 3670
        assert _tc_to_seconds("01:01:10:00.00") == 3670.0

    def test_frames_ignored(self):
        # Frames field doesn't change result
        assert _tc_to_seconds("01:00:10:12.00") == _tc_to_seconds("01:00:10:00.00")

    def test_three_part(self):
        assert _tc_to_seconds("01:00:30") == 3630.0

    def test_two_part(self):
        assert _tc_to_seconds("01:30") == 90.0

    def test_invalid_returns_zero(self):
        assert _tc_to_seconds("not_a_tc") == 0.0


# ---------------------------------------------------------------------------
# parse_arrangement
# ---------------------------------------------------------------------------


class TestParseArrangement:
    SAMPLE = textwrap.dedent(
        """\
        01:00:00:00.00     intro_arp_up     1     00:00:10:00.00
        01:00:00:00.00     Drums Primitive  2     00:00:10:00.00
        01:00:10:00.00     bass_intro_01    3     00:00:10:00.00
        01:00:20:00.00     verse_hypnotic   1     00:00:10:00.00
        01:00:20:00.00     verse_drums      2     00:00:10:00.00
    """
    )

    def test_clip_count(self):
        clips = parse_arrangement(self.SAMPLE)
        assert len(clips) == 5

    def test_base_offset_stripped(self):
        clips = parse_arrangement(self.SAMPLE)
        assert clips[0].start == 0.0

    def test_second_slot_start(self):
        clips = parse_arrangement(self.SAMPLE)
        bass_clip = next(c for c in clips if c.name == "bass_intro_01")
        assert bass_clip.start == 10.0

    def test_track_numbers(self):
        clips = parse_arrangement(self.SAMPLE)
        tracks = {c.name: c.track for c in clips}
        assert tracks["intro_arp_up"] == 1
        assert tracks["Drums Primitive"] == 2
        assert tracks["bass_intro_01"] == 3

    def test_length_parsed(self):
        clips = parse_arrangement(self.SAMPLE)
        assert clips[0].length == 10.0

    def test_multi_track_same_start(self):
        clips = parse_arrangement(self.SAMPLE)
        slot_zero = [c for c in clips if c.start == 0.0]
        assert len(slot_zero) == 2

    def test_empty_lines_ignored(self):
        text = "\n\n  01:00:00:00.00  intro_x  1  00:00:10:00.00\n\n"
        clips = parse_arrangement(text)
        assert len(clips) == 1

    def test_malformed_lines_skipped(self):
        text = "not a valid line\n01:00:00:00.00  intro_x  1  00:00:10:00.00\n"
        clips = parse_arrangement(text)
        assert len(clips) == 1

    def test_returns_empty_for_empty_input(self):
        assert parse_arrangement("") == []


# ---------------------------------------------------------------------------
# derive_sections
# ---------------------------------------------------------------------------


class TestDeriveSections:
    def _make_clips(self, spec: list[tuple]) -> list[Clip]:
        """spec: list of (start, name, track, length)"""
        return [Clip(start=s, name=n, track=t, length=dur) for s, n, t, dur in spec]

    def test_single_section(self):
        clips = self._make_clips(
            [(0.0, "intro_arp", 1, 10.0), (0.0, "intro_drums", 2, 10.0)]
        )
        sections = derive_sections(clips)
        assert len(sections) == 1
        assert sections[0].name == "Intro"

    def test_boundary_on_prefix_change(self):
        clips = self._make_clips(
            [
                (0.0, "intro_arp", 1, 10.0),
                (10.0, "verse_hypnotic", 1, 10.0),
            ]
        )
        sections = derive_sections(clips)
        assert len(sections) == 2
        assert sections[0].name == "Intro"
        assert sections[1].name == "Verse"

    def test_section_start_end_times(self):
        clips = self._make_clips(
            [
                (0.0, "intro_arp", 1, 10.0),
                (10.0, "verse_x", 1, 10.0),
            ]
        )
        sections = derive_sections(clips)
        assert sections[0].start == 0.0
        assert sections[0].end == 10.0
        assert sections[1].start == 10.0
        assert sections[1].end == 20.0

    def test_loops_populated(self):
        clips = self._make_clips(
            [
                (0.0, "intro_arp", 1, 10.0),
                (0.0, "intro_drums", 2, 10.0),
                (0.0, "intro_bass", 3, 10.0),
            ]
        )
        sections = derive_sections(clips)
        assert sections[0].loops["chords"] == "intro_arp"
        assert sections[0].loops["drums"] == "intro_drums"
        assert sections[0].loops["bass"] == "intro_bass"

    def test_vocals_false_without_melody(self):
        clips = self._make_clips([(0.0, "verse_drums", 2, 10.0)])
        sections = derive_sections(clips)
        assert sections[0].vocals is False

    def test_vocals_false_melody_without_suffix(self):
        clips = self._make_clips([(0.0, "verse_melody_plain", 4, 10.0)])
        sections = derive_sections(clips)
        assert sections[0].vocals is False

    def test_vocals_true_from_gw_suffix(self):
        clips = self._make_clips([(0.0, "verse_hypnotic_gw", 4, 10.0)])
        sections = derive_sections(clips)
        assert sections[0].vocals is True

    def test_vocals_true_from_custom_suffix(self):
        clips = self._make_clips([(0.0, "verse_melody_rb", 4, 10.0)])
        sections = derive_sections(clips, vocalist_suffix="_rb")
        assert sections[0].vocals is True

    def test_vocals_true_from_vocal_in_name(self):
        clips = self._make_clips([(0.0, "verse_vocal_lead", 4, 10.0)])
        sections = derive_sections(clips)
        assert sections[0].vocals is True

    def test_unknown_prefix_section(self):
        clips = self._make_clips([(0.0, "weirdloop_01", 1, 10.0)])
        sections = derive_sections(clips)
        assert sections[0].name == "unknown"

    def test_same_prefix_across_slots_stays_one_section(self):
        clips = self._make_clips(
            [(0.0, "verse_a", 1, 10.0), (10.0, "verse_b", 1, 10.0)]
        )
        sections = derive_sections(clips)
        assert len(sections) == 1
        assert sections[0].name == "Verse"

    def test_empty_clips_returns_empty(self):
        assert derive_sections([]) == []

    def test_bridge_prefix(self):
        clips = self._make_clips([(0.0, "bridge_main", 1, 20.0)])
        sections = derive_sections(clips)
        assert sections[0].name == "Bridge"

    def test_multiple_sections_order_preserved(self):
        clips = self._make_clips(
            [
                (0.0, "intro_x", 1, 10.0),
                (10.0, "verse_x", 1, 10.0),
                (20.0, "bridge_x", 1, 10.0),
                (30.0, "verse_x", 1, 10.0),
            ]
        )
        names = [s.name for s in derive_sections(clips)]
        assert names == ["Intro", "Verse", "Bridge", "Verse"]


# ---------------------------------------------------------------------------
# compute_drift
# ---------------------------------------------------------------------------


def _make_plan(sections_spec: list[tuple]) -> ProductionPlan:
    """Create a minimal ProductionPlan from (name, bars, repeat) tuples."""
    sections = [PlanSection(name=n, bars=b, repeat=r) for n, b, r in sections_spec]
    return ProductionPlan(
        song_slug="test_song",
        generated="2026-01-01T00:00:00+00:00",
        bpm=84,
        time_sig="7/8",
        key="Gb minor",
        color="Black",
        sections=sections,
    )


class TestComputeDrift:
    def test_no_drift(self):
        # At 84 BPM, 7/8: spb = (7 * 0.5) * (60/84) = 2.5s
        # Intro: 4 bars × 3 = 12 bars × 2.5 = 30s
        plan = _make_plan([("Intro", 4, 3)])
        from app.generators.midi.assembly_manifest import ArrangementSection

        actual = [ArrangementSection(name="Intro", start=0.0, end=30.0)]
        drift = compute_drift(plan, actual)
        assert len(drift) == 1
        assert drift[0].drift_seconds == 0.0

    def test_positive_drift(self):
        plan = _make_plan([("Intro", 4, 3)])
        from app.generators.midi.assembly_manifest import ArrangementSection

        actual = [ArrangementSection(name="Intro", start=10.0, end=40.0)]
        drift = compute_drift(plan, actual)
        assert drift[0].drift_seconds == 10.0

    def test_negative_drift(self):
        plan = _make_plan([("Intro", 4, 3), ("Bridge", 4, 1)])
        from app.generators.midi.assembly_manifest import ArrangementSection

        # Bridge should start at 30s, but actually starts at 20s
        actual = [
            ArrangementSection(name="Intro", start=0.0, end=20.0),
            ArrangementSection(name="Bridge", start=20.0, end=30.0),
        ]
        drift = compute_drift(plan, actual)
        assert drift[1].drift_seconds == -10.0

    def test_plan_name_vs_arrangement_name(self):
        plan = _make_plan([("Verse", 4, 3)])
        from app.generators.midi.assembly_manifest import ArrangementSection

        actual = [ArrangementSection(name="Bridge", start=0.0, end=30.0)]
        drift = compute_drift(plan, actual)
        assert drift[0].plan_name == "Verse"
        assert drift[0].arrangement_name == "Bridge"

    def test_vocals_flag_changed(self):
        plan = _make_plan([("Verse", 4, 2)])
        plan.sections[0].vocals = False
        from app.generators.midi.assembly_manifest import ArrangementSection

        actual = [ArrangementSection(name="Verse", start=0.0, end=20.0, vocals=True)]
        drift = compute_drift(plan, actual)
        assert drift[0].vocals_flag_changed is True

    def test_vocals_flag_unchanged(self):
        plan = _make_plan([("Verse", 4, 2)])
        plan.sections[0].vocals = True
        from app.generators.midi.assembly_manifest import ArrangementSection

        actual = [ArrangementSection(name="Verse", start=0.0, end=20.0, vocals=True)]
        drift = compute_drift(plan, actual)
        assert drift[0].vocals_flag_changed is False


# ---------------------------------------------------------------------------
# Integration test — The Archivist's Rebellion
# ---------------------------------------------------------------------------

ARCHIVIST_ARRANGEMENT = textwrap.dedent(
    """\
    01:00:00:00.00     intro_arp_up     1     00:00:10:00.00
    01:00:00:00.00     Drums Primitive  2     00:00:10:00.00
    01:00:10:00.00     Drum_intro_eighths  2  00:00:10:00.00
    01:00:10:00.00     bass_intro_simple   3  00:00:10:00.00
    01:00:20:00.00     intro_eighth_hypnotic  1  00:00:10:00.00
    01:00:20:00.00     bass_intro_02_eighths  2  00:00:10:00.00
    01:00:20:00.00     bass_intro_02_eighths  3  00:00:10:00.00
    01:00:20:00.00     melody_intro_04        4  00:00:10:00.00
    01:00:30:00.00     intro_eighth_hypnotic  1  00:00:10:00.00
    01:00:30:00.00     bass_intro_02_eighths  2  00:00:10:00.00
    01:00:30:00.00     bass_intro_01_bounce   3  00:00:10:00.00
    01:00:30:00.00     melody_intro_04        4  00:00:10:00.00
    01:00:40:00.00     intro_plain            1  00:00:10:00.00
    01:00:40:00.00     Drums_Primitive        2  00:00:10:00.00
    01:00:40:00.00     melody_intro_01        4  00:00:10:00.00
    01:00:50:00.00     bridge_eighth_hypnotic 1  00:00:20:00.00
    01:00:50:00.00     bridge_eighth_hypnotic 2  00:00:20:00.00
    01:00:50:00.00     bridge_eighth_hypnotic 3  00:00:20:00.00
    01:01:10:00.00     verse_2_eighth_hypnotic  1  00:00:10:00.00
    01:01:10:00.00     verse_02_sparse          2  00:00:10:00.00
    01:01:20:00.00     verse_2_eighth_hypnotic  3  00:00:10:00.00
    01:01:20:00.00     verse_2_hypnotic_gw      4  00:00:10:00.00
    01:01:50:00.00     bridge_2_punctuated      1  00:00:10:00.00
    01:01:50:00.00     bridge_02_busy_work      2  00:00:10:00.00
    01:01:50:00.00     bridge_2_punctuated      3  00:00:10:00.00
    01:01:50:00.00     bridge_2_punct_gw        4  00:00:10:00.00
    01:02:00:00.00     verse_2_eighth_hypnotic  1  00:00:10:00.00
    01:02:00:00.00     verse_02_sparse          2  00:00:10:00.00
    01:02:00:00.00     verse_2_eighth_hypnotic  3  00:00:10:00.00
    01:02:00:00.00     verse_2_hypnotic_gw      4  00:00:10:00.00
    01:02:40:00.00     bridge_2_arp_up          1  00:00:10:00.00
    01:02:40:00.00     bridge_02_claps          2  00:00:10:00.00
    01:02:40:00.00     bridge_2_arp_up          3  00:00:10:00.00
    01:02:50:00.00     bridge_2_arp_gw          4  00:00:10:00.00
    01:03:10:00.00     verse_2_plain            1  00:00:10:00.00
    01:03:10:00.00     drum_verse_2_01          2  00:00:10:00.00
    01:03:10:00.00     bass_verse_1_04          3  00:00:10:00.00
    01:03:10:00.00     verse_2_hypnotic_gw      4  00:00:10:00.00
"""
)


class TestArchivistIntegration:
    def test_sections_detected(self):
        clips = parse_arrangement(ARCHIVIST_ARRANGEMENT)
        sections = derive_sections(clips)
        names = [s.name for s in sections]
        assert "Intro" in names
        assert "Bridge" in names
        assert "Verse" in names

    def test_intro_starts_at_zero(self):
        clips = parse_arrangement(ARCHIVIST_ARRANGEMENT)
        sections = derive_sections(clips)
        assert sections[0].name == "Intro"
        assert sections[0].start == 0.0

    def test_bridge_starts_at_50s(self):
        clips = parse_arrangement(ARCHIVIST_ARRANGEMENT)
        sections = derive_sections(clips)
        bridge = next(s for s in sections if s.name == "Bridge")
        assert bridge.start == 50.0

    def test_verse_vocals_from_gw(self):
        clips = parse_arrangement(ARCHIVIST_ARRANGEMENT)
        sections = derive_sections(clips)
        verse_sections = [s for s in sections if s.name == "Verse"]
        # verse_2_hypnotic_gw should trigger vocals=True
        vocal_verses = [s for s in verse_sections if s.vocals]
        assert len(vocal_verses) > 0

    def test_bridge_vocals_from_gw(self):
        clips = parse_arrangement(ARCHIVIST_ARRANGEMENT)
        sections = derive_sections(clips)
        bridge_sections = [s for s in sections if s.name == "Bridge"]
        vocal_bridges = [s for s in bridge_sections if s.vocals]
        assert len(vocal_bridges) > 0

    def test_full_import(self, tmp_path):
        """End-to-end: import updates plan, manifest, and drift report."""
        # Set up a minimal production plan
        plan = _make_plan(
            [
                ("Intro", 4, 3),
                ("Verse", 4, 3),
                ("Bridge", 4, 1),
                ("Verse", 4, 2),
                ("Bridge", 4, 2),
            ]
        )
        plan.song_slug = "test_archivist"
        save_plan(plan, tmp_path)

        (tmp_path / "manifest_bootstrap.yml").write_text(
            "manifest_id: test_archivist\nbpm: 84\ntempo: 7/8\nvocals: false\nstructure: []\nTRT: null\n"
        )

        # Write arrangement file
        arr_file = tmp_path / "arrangement.txt"
        arr_file.write_text(ARCHIVIST_ARRANGEMENT)

        import_arrangement(tmp_path, arr_file)

        # Plan should be updated
        updated_plan = load_plan(tmp_path)
        assert updated_plan is not None
        assert len(updated_plan.sections) == 5

        # Loops should be populated on sections that had clips
        intro_sec = updated_plan.sections[0]
        assert "chords" in intro_sec.loops

        # Drift report written
        assert (tmp_path / "drift_report.yml").exists()

        # Manifest updated
        manifest = yaml.safe_load((tmp_path / "manifest_bootstrap.yml").read_text())
        assert len(manifest["structure"]) == 5

    def test_missing_arrangement_raises(self, tmp_path):
        plan = _make_plan([("Intro", 4, 1)])
        save_plan(plan, tmp_path)
        with pytest.raises(FileNotFoundError, match="Arrangement file not found"):
            import_arrangement(tmp_path, tmp_path / "nonexistent.txt")

    def test_missing_plan_raises(self, tmp_path):
        arr_file = tmp_path / "arr.txt"
        arr_file.write_text(ARCHIVIST_ARRANGEMENT)
        with pytest.raises(FileNotFoundError):
            import_arrangement(tmp_path, arr_file)
