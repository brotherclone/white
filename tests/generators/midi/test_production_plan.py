"""Tests for the production plan generator."""

import io
from pathlib import Path

import mido
import pytest
import yaml

from app.generators.midi.production.production_plan import (
    MANIFEST_BOOTSTRAP_FILENAME,
    PLAN_FILENAME,
    PlanSection,
    ProductionPlan,
    _infer_arc_from_label,
    bootstrap_manifest,
    build_next_section_map,
    derive_bar_count,
    generate_plan,
    generate_plan_mechanical,
    load_plan,
    refresh_plan,
    save_plan,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_chord_midi(bars: int, bpm: int = 120, tpb: int = 480) -> bytes:
    """Create a simple chord MIDI with the given number of bars (4/4)."""
    mid = mido.MidiFile(ticks_per_beat=tpb)
    track = mido.MidiTrack()
    mid.tracks.append(track)
    track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(bpm), time=0))
    bar_ticks = tpb * 4  # 4/4
    total_ticks = bar_ticks * bars
    track.append(mido.Message("note_on", note=60, velocity=80, time=0))
    track.append(mido.Message("note_off", note=60, velocity=0, time=total_ticks))
    track.append(mido.MetaMessage("end_of_track", time=0))
    buf = io.BytesIO()
    mid.save(file=buf)
    return buf.getvalue()


def _write_chord_review(chords_dir: Path, sections: list[dict], bpm: int = 120) -> None:
    """Write a chord review.yml with the given approved sections."""
    chords_dir.mkdir(parents=True, exist_ok=True)
    candidates = []
    for i, s in enumerate(sections):
        cand = {
            "id": f"chord_{i+1:03d}",
            "label": s["label"],
            "status": "approved",
            "chords": [{"name": "C", "notes": ["C4", "E4", "G4"]}]
            * s.get("chord_count", 4),
        }
        if "hr_distribution" in s:
            cand["hr_distribution"] = s["hr_distribution"]
        candidates.append(cand)
    review = {
        "bpm": bpm,
        "time_sig": "4/4",
        "color": "Black",
        "key": "C minor",
        "title": "Test Song",
        "candidates": candidates,
    }
    with open(chords_dir / "review.yml", "w") as f:
        yaml.dump(review, f)


def _write_chord_midi(
    approved_dir: Path, label: str, bars: int, bpm: int = 120
) -> None:
    """Write a chord MIDI file to approved_dir."""
    approved_dir.mkdir(parents=True, exist_ok=True)
    midi_bytes = _make_chord_midi(bars, bpm)
    (approved_dir / f"{label}.mid").write_bytes(midi_bytes)


# ---------------------------------------------------------------------------
# Bar count derivation
# ---------------------------------------------------------------------------


class TestDeriveBarCount:
    def test_chord_midi_source(self, tmp_path):
        prod = tmp_path / "production" / "test_song"
        approved = prod / "chords" / "approved"
        _write_chord_midi(approved, "verse", bars=4)

        bars, source = derive_bar_count("verse", prod, bpm=120, time_sig=(4, 4))
        assert bars == 4
        assert source == "chords"

    def test_hr_distribution_takes_priority(self, tmp_path):
        prod = tmp_path / "production" / "test_song"
        # Write chord MIDI (4 bars) — should be overridden by hr_distribution
        _write_chord_midi(prod / "chords" / "approved", "verse", bars=4)
        # hr_distribution sums to 6 bars — should win
        bars, source = derive_bar_count(
            "verse",
            prod,
            bpm=120,
            time_sig=(4, 4),
            hr_distribution=[1.5, 0.5, 2.0, 2.0],
        )
        assert bars == 6
        assert source == "hr_distribution"

    def test_fallback_to_chord_count(self, tmp_path):
        prod = tmp_path / "production" / "test_song"
        bars, source = derive_bar_count(
            "verse", prod, bpm=120, time_sig=(4, 4), chord_count_fallback=6
        )
        assert bars == 6
        assert source == "chord_count"

    def test_label_normalisation(self, tmp_path):
        prod = tmp_path / "production" / "test_song"
        # File saved as "verse_chorus" but label passed with spaces/caps
        _write_chord_midi(prod / "chords" / "approved", "verse_chorus", bars=4)
        bars, source = derive_bar_count("Verse-Chorus", prod, bpm=120, time_sig=(4, 4))
        assert bars == 4
        assert source == "chords"

    def test_7_8_time_sig(self, tmp_path):
        prod = tmp_path / "production" / "test_song"
        # 7/8: bar = 3.5 beats. Write a MIDI that is exactly 2 bars of 7/8.
        # bar_ticks = 480 * 3.5 = 1680 ticks per bar
        mid = mido.MidiFile(ticks_per_beat=480)
        track = mido.MidiTrack()
        mid.tracks.append(track)
        track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(120), time=0))
        track.append(mido.Message("note_on", note=60, velocity=80, time=0))
        track.append(
            mido.Message("note_off", note=60, velocity=0, time=3360)
        )  # 2 * 1680
        track.append(mido.MetaMessage("end_of_track", time=0))
        approved = prod / "chords" / "approved"
        approved.mkdir(parents=True)
        buf = io.BytesIO()
        mid.save(file=buf)
        (approved / "bridge.mid").write_bytes(buf.getvalue())

        bars, source = derive_bar_count("bridge", prod, bpm=120, time_sig=(7, 8))
        assert bars == 2
        assert source == "chords"


# ---------------------------------------------------------------------------
# Plan I/O (save / load round-trip)
# ---------------------------------------------------------------------------


class TestPlanIO:
    def test_save_and_load_round_trip(self, tmp_path):
        plan = ProductionPlan(
            song_slug="test_song",
            generated="2026-02-18T10:00:00Z",
            bpm=120,
            time_sig="4/4",
            key="C minor",
            color="Black",
            title="Test Song",
            sections=[
                PlanSection(
                    name="verse", bars=4, play_count=2, vocals=True, notes="main"
                ),
                PlanSection(name="chorus", bars=2, play_count=3, vocals=True),
            ],
        )
        save_plan(plan, tmp_path)
        assert (tmp_path / PLAN_FILENAME).exists()

        loaded = load_plan(tmp_path)
        assert loaded is not None
        assert loaded.song_slug == "test_song"
        assert loaded.bpm == 120
        assert len(loaded.sections) == 2
        assert loaded.sections[0].name == "verse"
        assert loaded.sections[0].bars == 4
        assert loaded.sections[0].play_count == 2
        assert loaded.sections[0].vocals is True
        assert loaded.sections[0].notes == "main"
        assert loaded.sections[1].name == "chorus"

    def test_load_returns_none_when_absent(self, tmp_path):
        assert load_plan(tmp_path) is None

    def test_sections_order_preserved(self, tmp_path):
        plan = ProductionPlan(
            song_slug="s",
            generated="",
            bpm=100,
            time_sig="4/4",
            key="",
            color="",
            sections=[
                PlanSection(name="intro", bars=2),
                PlanSection(name="verse", bars=4),
                PlanSection(name="chorus", bars=4),
                PlanSection(name="outro", bars=2),
            ],
        )
        save_plan(plan, tmp_path)
        loaded = load_plan(tmp_path)
        names = [s.name for s in loaded.sections]
        assert names == ["intro", "verse", "chorus", "outro"]


# ---------------------------------------------------------------------------
# Plan generation
# ---------------------------------------------------------------------------


class TestGeneratePlan:
    def _setup(self, tmp_path, sections, bpm=120):
        prod = tmp_path / "production" / "test_song"
        prod.mkdir(parents=True)
        _write_chord_review(prod / "chords", sections, bpm=bpm)
        return prod

    def test_generates_sections_in_order(self, tmp_path):
        sections = [
            {"label": "Intro", "chord_count": 2},
            {"label": "Verse", "chord_count": 4},
            {"label": "Chorus", "chord_count": 4},
        ]
        prod = self._setup(tmp_path, sections)
        plan = generate_plan_mechanical(prod)
        assert [s.name for s in plan.sections] == ["Intro", "Verse", "Chorus"]

    def test_bar_count_from_chord_count_fallback(self, tmp_path):
        prod = self._setup(tmp_path, [{"label": "Verse", "chord_count": 4}])
        plan = generate_plan_mechanical(prod)
        assert plan.sections[0].bars == 4
        assert plan.sections[0]._bar_source == "chord_count"

    def test_bar_count_from_chord_midi(self, tmp_path):
        prod = self._setup(tmp_path, [{"label": "verse", "chord_count": 4}])
        _write_chord_midi(prod / "chords" / "approved", "verse", bars=8)
        plan = generate_plan_mechanical(prod)
        assert plan.sections[0].bars == 8
        assert plan.sections[0]._bar_source == "chords"

    def test_bar_count_from_hr_distribution(self, tmp_path):
        # hr_distribution in review.yml takes priority over chord MIDI
        prod = self._setup(
            tmp_path,
            [
                {
                    "label": "verse",
                    "chord_count": 4,
                    "hr_distribution": [1.5, 0.5, 2.0, 2.0],
                }
            ],
        )
        _write_chord_midi(prod / "chords" / "approved", "verse", bars=4)
        plan = generate_plan_mechanical(prod)
        assert plan.sections[0].bars == 6  # sum([1.5, 0.5, 2.0, 2.0])
        assert plan.sections[0]._bar_source == "hr_distribution"

    def test_deduplicates_repeated_labels(self, tmp_path):
        # Same label approved twice — only the first occurrence should appear
        prod = self._setup(
            tmp_path,
            [
                {"label": "Verse", "chord_count": 4},
                {"label": "Verse", "chord_count": 4},
                {"label": "Chorus", "chord_count": 4},
            ],
        )
        plan = generate_plan_mechanical(prod)
        names = [s.name for s in plan.sections]
        assert names.count("Verse") == 1
        assert len(plan.sections) == 2

    def test_metadata_from_chord_review(self, tmp_path):
        prod = self._setup(tmp_path, [{"label": "Verse", "chord_count": 4}], bpm=88)
        plan = generate_plan_mechanical(prod)
        assert plan.bpm == 88
        assert plan.color == "Black"
        assert plan.time_sig == "4/4"
        assert plan.song_slug == "test_song"

    def test_raises_when_no_approved_sections(self, tmp_path):
        prod = tmp_path / "production" / "test_song"
        prod.mkdir(parents=True)
        chords_dir = prod / "chords"
        chords_dir.mkdir()
        with open(chords_dir / "review.yml", "w") as f:
            yaml.dump({"bpm": 120, "candidates": []}, f)
        with pytest.raises(ValueError, match="No approved"):
            generate_plan_mechanical(prod)

    def test_raises_when_no_chord_review(self, tmp_path):
        prod = tmp_path / "production" / "test_song"
        prod.mkdir(parents=True)
        with pytest.raises(FileNotFoundError):
            generate_plan_mechanical(prod)


# ---------------------------------------------------------------------------
# Refresh
# ---------------------------------------------------------------------------


class TestRefreshPlan:
    def _make_plan(self, tmp_path, sections):
        prod = tmp_path / "production" / "test_song"
        prod.mkdir(parents=True)
        _write_chord_review(prod / "chords", sections)
        plan = generate_plan_mechanical(prod)
        save_plan(plan, prod)
        return prod, plan

    def test_refresh_updates_bar_count(self, tmp_path):
        prod, _ = self._make_plan(tmp_path, [{"label": "verse", "chord_count": 4}])
        # Now write a chord MIDI with different bar count
        _write_chord_midi(prod / "chords" / "approved", "verse", bars=8)

        refreshed = refresh_plan(prod)
        assert refreshed.sections[0].bars == 8
        assert refreshed.sections[0]._bar_source == "chords"

    def test_refresh_preserves_human_edits(self, tmp_path):
        prod, plan = self._make_plan(tmp_path, [{"label": "verse", "chord_count": 4}])
        # Simulate human edits
        plan.sections[0].play_count = 3
        plan.sections[0].vocals = True
        plan.sections[0].notes = "main section"
        save_plan(plan, prod)

        refreshed = refresh_plan(prod)
        assert refreshed.sections[0].play_count == 3
        assert refreshed.sections[0].vocals is True
        assert refreshed.sections[0].notes == "main section"

    def test_refresh_raises_when_no_plan(self, tmp_path):
        prod = tmp_path / "production" / "test_song"
        prod.mkdir(parents=True)
        _write_chord_review(prod / "chords", [{"label": "verse", "chord_count": 4}])
        with pytest.raises(FileNotFoundError):
            refresh_plan(prod)

    def test_refresh_warns_orphaned_sections(self, tmp_path, capsys):
        prod, plan = self._make_plan(tmp_path, [{"label": "verse", "chord_count": 4}])
        # Add a section to the plan that doesn't exist in chord review
        plan.sections.append(PlanSection(name="ghost_section", bars=4))
        save_plan(plan, prod)

        refresh_plan(prod)
        captured = capsys.readouterr()
        assert "ghost_section" in captured.out
        assert "Warning" in captured.out


# ---------------------------------------------------------------------------
# Manifest bootstrap
# ---------------------------------------------------------------------------


class TestBootstrapManifest:
    def _make_plan_file(self, tmp_path, sections):
        prod = tmp_path / "production" / "test_song"
        prod.mkdir(parents=True)
        plan = ProductionPlan(
            song_slug="test_song",
            generated="2026-02-18T10:00:00Z",
            bpm=120,
            time_sig="4/4",
            key="C minor",
            color="Black",
            title="Test Song",
            sections=sections,
        )
        save_plan(plan, prod)
        return prod

    def test_bootstrap_writes_file(self, tmp_path):
        prod = self._make_plan_file(tmp_path, [PlanSection(name="verse", bars=4)])
        out = bootstrap_manifest(prod)
        assert out.exists()
        assert out.name == MANIFEST_BOOTSTRAP_FILENAME

    def test_bootstrap_structure_contains_sections(self, tmp_path):
        sections = [
            PlanSection(name="verse", bars=4, play_count=2),
            PlanSection(name="chorus", bars=2, play_count=3),
        ]
        prod = self._make_plan_file(tmp_path, sections)
        bootstrap_manifest(prod)

        with open(prod / MANIFEST_BOOTSTRAP_FILENAME) as f:
            data = yaml.safe_load(f)

        structure = data["structure"]
        assert len(structure) == 2
        assert structure[0]["section_name"] == "verse"
        assert structure[1]["section_name"] == "chorus"

    def test_bootstrap_timestamps_are_cumulative(self, tmp_path):
        # At 120 BPM, 4/4: 1 bar = 2 seconds
        # verse: 4 bars × 2 repeats = 8 bars = 16 seconds (0:00 → 0:16)
        # chorus: 2 bars × 3 repeats = 6 bars = 12 seconds (0:16 → 0:28)
        sections = [
            PlanSection(name="verse", bars=4, play_count=2),
            PlanSection(name="chorus", bars=2, play_count=3),
        ]
        prod = self._make_plan_file(tmp_path, sections)
        bootstrap_manifest(prod)

        with open(prod / MANIFEST_BOOTSTRAP_FILENAME) as f:
            data = yaml.safe_load(f)

        verse = data["structure"][0]
        chorus = data["structure"][1]
        assert verse["start_time"] == "[00:00.000]"
        assert verse["end_time"] == "[00:16.000]"
        assert chorus["start_time"] == "[00:16.000]"
        assert chorus["end_time"] == "[00:28.000]"

    def test_bootstrap_vocals_flag(self, tmp_path):
        prod = self._make_plan_file(
            tmp_path,
            [PlanSection(name="verse", bars=4, vocals=True)],
        )
        bootstrap_manifest(prod)
        with open(prod / MANIFEST_BOOTSTRAP_FILENAME) as f:
            data = yaml.safe_load(f)
        assert data["vocals"] is True
        assert data["lyrics"] is True

    def test_bootstrap_null_render_fields(self, tmp_path):
        prod = self._make_plan_file(tmp_path, [PlanSection(name="verse", bars=4)])
        bootstrap_manifest(prod)
        with open(prod / MANIFEST_BOOTSTRAP_FILENAME) as f:
            data = yaml.safe_load(f)
        for field in (
            "release_date",
            "album_sequence",
            "main_audio_file",
            "TRT",
            "lrc_file",
        ):
            assert data[field] is None

    def test_bootstrap_raises_when_no_plan(self, tmp_path):
        prod = tmp_path / "production" / "test_song"
        prod.mkdir(parents=True)
        with pytest.raises(FileNotFoundError):
            bootstrap_manifest(prod)


# ---------------------------------------------------------------------------
# next_section map
# ---------------------------------------------------------------------------


class TestBuildNextSectionMap:
    def test_basic_sequence(self):
        plan = ProductionPlan(
            song_slug="s",
            generated="",
            bpm=120,
            time_sig="4/4",
            key="",
            color="",
            sections=[
                PlanSection(name="intro", bars=4),
                PlanSection(name="verse", bars=8),
                PlanSection(name="chorus", bars=4),
                PlanSection(name="outro", bars=4),
            ],
        )
        m = build_next_section_map(plan)
        assert m["intro"] == "verse"
        assert m["verse"] == "chorus"
        assert m["chorus"] == "outro"
        assert m["outro"] is None

    def test_single_section(self):
        plan = ProductionPlan(
            song_slug="s",
            generated="",
            bpm=120,
            time_sig="4/4",
            key="",
            color="",
            sections=[PlanSection(name="verse", bars=4)],
        )
        m = build_next_section_map(plan)
        assert m["verse"] is None

    def test_empty_plan(self):
        plan = ProductionPlan(
            song_slug="s",
            generated="",
            bpm=120,
            time_sig="4/4",
            key="",
            color="",
        )
        assert build_next_section_map(plan) == {}


# ---------------------------------------------------------------------------
# Claude composition proposal
# ---------------------------------------------------------------------------


_MOCK_CLAUDE_RESPONSE = """\
```yaml
proposed_sections:
  - name: Intro
    play_count: 1
    energy_note: sparse and atmospheric
  - name: Verse
    play_count: 2
    energy_note: builds tension
  - name: Chorus
    play_count: 3
    energy_note: full release
  - name: Outro
    play_count: 1
    energy_note: decay and silence
```

Rationale:
The arrangement follows a classic arc — sparse intro, double verse to build,
triple chorus for catharsis, then a single outro to dissolve. The color's axes
demand accumulation before release.
"""


class TestClaudeCompositionProposal:
    def _make_plan(self, sections=None):
        if sections is None:
            sections = [
                PlanSection(name="Intro", bars=2),
                PlanSection(name="Verse", bars=4, vocals=True),
                PlanSection(name="Chorus", bars=4, vocals=True),
                PlanSection(name="Outro", bars=2),
            ]
        return ProductionPlan(
            song_slug="test_song",
            generated="2026-02-18T10:00:00Z",
            bpm=120,
            time_sig="4/4",
            key="C minor",
            color="Black",
            title="Test Song",
            concept="A song about entropy",
            mood=["melancholic", "abstract"],
            genres=["krautrock"],
            sections=sections,
        )

    def test_propose_arrangement_sets_proposed_by(self, monkeypatch, tmp_path):
        from app.generators.midi.production import production_plan as pp

        def _fake_propose(plan):
            from app.generators.midi.production.production_plan import (
                _parse_arrangement_response,
            )

            updated, rationale = _parse_arrangement_response(
                _MOCK_CLAUDE_RESPONSE, plan
            )
            plan.sections = updated
            plan.rationale = rationale
            plan.proposed_by = "claude"
            return plan

        monkeypatch.setattr(pp, "propose_arrangement", _fake_propose)
        prod = tmp_path / "production" / "test_song"
        prod.mkdir(parents=True)
        _write_chord_review(
            prod / "chords",
            [
                {"label": "Intro", "chord_count": 2},
                {"label": "Verse", "chord_count": 4},
                {"label": "Chorus", "chord_count": 4},
                {"label": "Outro", "chord_count": 2},
            ],
        )
        result = generate_plan(prod, use_claude=True)
        assert result.proposed_by == "claude"
        assert result.rationale != ""

    def test_propose_arrangement_sets_reason_on_sections(self):
        from app.generators.midi.production.production_plan import (
            _parse_arrangement_response,
        )

        plan = self._make_plan()
        updated, rationale = _parse_arrangement_response(_MOCK_CLAUDE_RESPONSE, plan)
        assert updated[0].reason == "sparse and atmospheric"
        assert updated[1].reason == "builds tension"
        assert updated[2].reason == "full release"
        assert updated[3].reason == "decay and silence"

    def test_propose_arrangement_sets_play_counts(self):
        from app.generators.midi.production.production_plan import (
            _parse_arrangement_response,
        )

        plan = self._make_plan()
        updated, _ = _parse_arrangement_response(_MOCK_CLAUDE_RESPONSE, plan)
        counts = {s.name: s.play_count for s in updated}
        assert counts["Intro"] == 1
        assert counts["Verse"] == 2
        assert counts["Chorus"] == 3
        assert counts["Outro"] == 1

    def test_generate_plan_mechanical_still_works(self, tmp_path):
        prod = tmp_path / "production" / "test_song"
        prod.mkdir(parents=True)
        _write_chord_review(
            prod / "chords",
            [{"label": "Verse", "chord_count": 4}],
        )
        plan = generate_plan_mechanical(prod)
        assert plan.sections[0].name == "Verse"
        assert plan.proposed_by == ""
        assert plan.rationale == ""

    def test_generate_plan_use_claude_false_skips_claude(self, tmp_path):
        prod = tmp_path / "production" / "test_song"
        prod.mkdir(parents=True)
        _write_chord_review(
            prod / "chords",
            [{"label": "Verse", "chord_count": 4}],
        )
        plan = generate_plan(prod, use_claude=False)
        assert plan.proposed_by == ""
        assert plan.rationale == ""

    def test_generate_plan_falls_back_on_api_error(self, tmp_path, monkeypatch):
        from app.generators.midi.production import production_plan as pp

        def _raise(_plan):
            raise RuntimeError("API unavailable")

        monkeypatch.setattr(pp, "propose_arrangement", _raise)
        prod = tmp_path / "production" / "test_song"
        prod.mkdir(parents=True)
        _write_chord_review(
            prod / "chords",
            [{"label": "Verse", "chord_count": 4}],
        )
        plan = generate_plan(prod, use_claude=True)
        # Fell back to mechanical — no Claude fields
        assert plan.proposed_by == ""
        assert plan.rationale == ""


class TestReasonFieldRoundTrip:
    def test_reason_saved_and_loaded(self, tmp_path):
        plan = ProductionPlan(
            song_slug="s",
            generated="",
            bpm=120,
            time_sig="4/4",
            key="",
            color="",
            sections=[
                PlanSection(
                    name="verse", bars=4, play_count=2, reason="build tension here"
                ),
                PlanSection(name="chorus", bars=4, reason=""),
            ],
        )
        save_plan(plan, tmp_path)
        loaded = load_plan(tmp_path)
        assert loaded.sections[0].reason == "build tension here"
        assert loaded.sections[1].reason == ""

    def test_reason_absent_in_yaml_defaults_to_empty(self, tmp_path):
        # Write plan YAML without reason field (old format)
        plan_path = tmp_path / PLAN_FILENAME
        data = {
            "song_slug": "s",
            "generated": "",
            "sections": [
                {
                    "name": "verse",
                    "bars": 4,
                    "play_count": 1,
                    "vocals": False,
                    "notes": "",
                }
            ],
        }
        with open(plan_path, "w") as f:
            yaml.dump(data, f)
        loaded = load_plan(tmp_path)
        assert loaded.sections[0].reason == ""

    def test_refresh_preserves_reason(self, tmp_path):
        prod = tmp_path / "production" / "test_song"
        prod.mkdir(parents=True)
        _write_chord_review(prod / "chords", [{"label": "verse", "chord_count": 4}])
        plan = generate_plan_mechanical(prod)
        plan.sections[0].reason = "Claude said: slow burn"
        plan.proposed_by = "claude"
        save_plan(plan, prod)

        refreshed = refresh_plan(prod)
        assert refreshed.sections[0].reason == "Claude said: slow burn"
        assert refreshed.proposed_by == "claude"


# ---------------------------------------------------------------------------
# Lyric repeat type
# ---------------------------------------------------------------------------


class TestNormalizeRepeatType:
    def test_valid_values_pass_through(self):
        from app.generators.midi.production.production_plan import (
            _normalize_repeat_type,
        )

        assert _normalize_repeat_type("exact") == "exact"
        assert _normalize_repeat_type("variation") == "variation"
        assert _normalize_repeat_type("fresh") == "fresh"

    def test_uppercase_normalised(self):
        from app.generators.midi.production.production_plan import (
            _normalize_repeat_type,
        )

        assert _normalize_repeat_type("Exact") == "exact"
        assert _normalize_repeat_type("VARIATION") == "variation"

    def test_invalid_value_falls_back_to_fresh(self):
        from app.generators.midi.production.production_plan import (
            _normalize_repeat_type,
        )

        assert _normalize_repeat_type("typo") == "fresh"
        assert (
            _normalize_repeat_type("exact_repeat") == "fresh"
        )  # internal sentinel not valid in YAML

    def test_none_and_empty_return_fresh(self):
        from app.generators.midi.production.production_plan import (
            _normalize_repeat_type,
        )

        assert _normalize_repeat_type(None) == "fresh"
        assert _normalize_repeat_type("") == "fresh"


class TestInferRepeatType:
    def test_chorus_exact(self):
        from app.generators.midi.production.production_plan import _infer_repeat_type

        assert _infer_repeat_type("chorus") == "exact"
        assert _infer_repeat_type("melody_chorus") == "exact"
        assert _infer_repeat_type("refrain") == "exact"
        assert _infer_repeat_type("hook") == "exact"

    def test_verse_variation(self):
        from app.generators.midi.production.production_plan import _infer_repeat_type

        assert _infer_repeat_type("verse") == "variation"
        assert _infer_repeat_type("melody_verse") == "variation"
        assert _infer_repeat_type("pre_chorus") == "variation"
        assert _infer_repeat_type("pre-chorus") == "variation"

    def test_other_fresh(self):
        from app.generators.midi.production.production_plan import _infer_repeat_type

        assert _infer_repeat_type("bridge") == "fresh"
        assert _infer_repeat_type("outro") == "fresh"
        assert _infer_repeat_type("intro") == "fresh"


class TestLyricRepeatTypeRoundTrip:
    def test_exact_saved_and_loaded(self, tmp_path):
        """lyric_repeat_type='exact' persists through save/load."""
        plan = ProductionPlan(
            song_slug="test",
            generated="2026-01-01",
            bpm=120,
            time_sig="4/4",
            key="C major",
            color="Red",
            sections=[PlanSection(name="chorus", bars=4, lyric_repeat_type="exact")],
        )
        save_plan(plan, tmp_path)
        loaded = load_plan(tmp_path)
        assert loaded.sections[0].lyric_repeat_type == "exact"

    def test_fresh_omitted_from_yaml_but_defaults_on_load(self, tmp_path):
        """lyric_repeat_type='fresh' is omitted from YAML but loads as 'fresh'."""
        plan = ProductionPlan(
            song_slug="test",
            generated="2026-01-01",
            bpm=120,
            time_sig="4/4",
            key="C major",
            color="Red",
            sections=[PlanSection(name="bridge", bars=4, lyric_repeat_type="fresh")],
        )
        save_plan(plan, tmp_path)
        raw = (tmp_path / "production_plan.yml").read_text()
        assert "lyric_repeat_type" not in raw
        loaded = load_plan(tmp_path)
        assert loaded.sections[0].lyric_repeat_type == "fresh"

    def test_human_override_survives_refresh(self, tmp_path):
        """A human-set lyric_repeat_type is preserved across --refresh."""
        prod = tmp_path / "production" / "test_song"
        prod.mkdir(parents=True)
        _write_chord_review(prod / "chords", [{"label": "bridge", "chord_count": 4}])
        plan = generate_plan_mechanical(prod)
        # Human overrides bridge from 'fresh' to 'exact'
        plan.sections[0].lyric_repeat_type = "exact"
        save_plan(plan, prod)

        refreshed = refresh_plan(prod)
        assert refreshed.sections[0].lyric_repeat_type == "exact"

    def test_generate_plan_mechanical_populates_repeat_type(self, tmp_path):
        """generate_plan_mechanical infers lyric_repeat_type from section labels."""
        prod = tmp_path / "production" / "test_song"
        prod.mkdir(parents=True)
        _write_chord_review(
            prod / "chords",
            [
                {"label": "chorus", "chord_count": 4},
                {"label": "verse", "chord_count": 4},
                {"label": "bridge", "chord_count": 4},
            ],
        )
        plan = generate_plan_mechanical(prod)
        by_name = {s.name: s for s in plan.sections}
        assert by_name["chorus"].lyric_repeat_type == "exact"
        assert by_name["verse"].lyric_repeat_type == "variation"
        assert by_name["bridge"].lyric_repeat_type == "fresh"


# ---------------------------------------------------------------------------
# Arc tests
# ---------------------------------------------------------------------------


class TestInferArcFromLabel:
    """_infer_arc_from_label maps canonical section labels to expected arc ranges."""

    @pytest.mark.parametrize(
        "label,expected",
        [
            ("intro", 0.15),
            ("outro", 0.15),
            ("bridge", 0.20),
            ("verse", 0.35),
            ("pre_chorus", 0.55),
            ("chorus", 0.75),
            ("refrain", 0.75),
            ("hook", 0.75),
            ("climax", 0.90),
            ("peak", 0.90),
            ("section_x", 0.40),  # unknown → default
        ],
    )
    def test_canonical_labels(self, label, expected):
        assert _infer_arc_from_label(label) == expected

    def test_case_insensitive(self):
        assert _infer_arc_from_label("CHORUS") == _infer_arc_from_label("chorus")

    def test_hyphen_normalised(self):
        assert _infer_arc_from_label("pre-chorus") == 0.55

    def test_verse_2_variant(self):
        assert _infer_arc_from_label("verse_2") == 0.35


class TestArcRoundTrip:
    """arc field survives save/load round-trip."""

    def test_arc_written_and_loaded(self, tmp_path):
        plan = ProductionPlan(
            song_slug="t",
            generated="",
            bpm=120,
            time_sig="4/4",
            key="",
            color="",
            sections=[
                PlanSection(name="verse", bars=4, arc=0.35),
                PlanSection(name="chorus", bars=4, arc=0.75),
            ],
        )
        save_plan(plan, tmp_path)
        loaded = load_plan(tmp_path)
        assert loaded.sections[0].arc == pytest.approx(0.35)
        assert loaded.sections[1].arc == pytest.approx(0.75)

    def test_arc_zero_not_written(self, tmp_path):
        """arc=0.0 should be omitted from YAML (falsy guard)."""
        plan = ProductionPlan(
            song_slug="t",
            generated="",
            bpm=120,
            time_sig="4/4",
            key="",
            color="",
            sections=[PlanSection(name="intro", bars=2, arc=0.0)],
        )
        save_plan(plan, tmp_path)
        import yaml as _yaml

        with open(tmp_path / PLAN_FILENAME) as f:
            raw = _yaml.safe_load(f)
        assert "arc" not in raw["sections"][0]

    def test_arc_missing_in_yaml_loads_as_zero(self, tmp_path):
        """Plans written before arc field added load with arc=0.0."""
        plan = ProductionPlan(
            song_slug="t",
            generated="",
            bpm=120,
            time_sig="4/4",
            key="",
            color="",
            sections=[PlanSection(name="verse", bars=4, arc=0.0)],
        )
        save_plan(plan, tmp_path)
        loaded = load_plan(tmp_path)
        assert loaded.sections[0].arc == pytest.approx(0.0)


class TestArcGenerateAndSync:
    """generate_plan seeds arc; sync_plan_from_arrangement preserves human override."""

    def test_generate_plan_mechanical_seeds_arc(self, tmp_path):
        prod = tmp_path / "production" / "song"
        prod.mkdir(parents=True)
        _write_chord_review(
            prod / "chords",
            [
                {"label": "intro", "chord_count": 4},
                {"label": "chorus", "chord_count": 4},
            ],
        )
        plan = generate_plan_mechanical(prod)
        by_name = {s.name: s for s in plan.sections}
        assert by_name["intro"].arc == pytest.approx(0.15)
        assert by_name["chorus"].arc == pytest.approx(0.75)

    def test_sync_preserves_human_arc_override(self, tmp_path):
        """Human arc edits survive a plan refresh."""
        prod = tmp_path / "production" / "song"
        prod.mkdir(parents=True)
        _write_chord_review(prod / "chords", [{"label": "verse", "chord_count": 4}])
        plan = generate_plan_mechanical(prod)
        # Human overrides arc
        plan.sections[0].arc = 0.99
        save_plan(plan, prod)

        refreshed = refresh_plan(prod)
        assert refreshed.sections[0].arc == pytest.approx(0.99)
