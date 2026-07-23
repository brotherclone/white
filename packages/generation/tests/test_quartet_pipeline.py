"""Tests for quartet_pipeline — soprano extraction, voice generation, disk I/O."""

import io
from pathlib import Path
from unittest.mock import MagicMock, patch

import mido
import pytest
import yaml

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_melody_midi(notes: list[int], bpm: int = 120, tpb: int = 480) -> bytes:
    """Create a melody MIDI with one note per beat (quarter note, 4/4) on channel 0."""
    mid = mido.MidiFile(ticks_per_beat=tpb)
    track = mido.MidiTrack()
    mid.tracks.append(track)
    track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(bpm), time=0))
    for pitch in notes:
        track.append(
            mido.Message("note_on", channel=0, note=pitch, velocity=90, time=0)
        )
        track.append(
            mido.Message("note_off", channel=0, note=pitch, velocity=0, time=tpb)
        )
    track.append(mido.MetaMessage("end_of_track", time=0))
    buf = io.BytesIO()
    mid.save(file=buf)
    return buf.getvalue()


def _write_song_context(prod_dir: Path, bpm: int = 120, color: str = "Red") -> None:
    (prod_dir / "song_context.yml").write_text(
        f"bpm: {bpm}\ncolor: {color}\ntime_sig: 4/4\nkey: C major\n"
    )


def _write_melody_approved(prod_dir: Path, section: str, notes: list[int]) -> Path:
    approved = prod_dir / "melody" / "approved"
    approved.mkdir(parents=True, exist_ok=True)
    path = approved / f"{section}.mid"
    path.write_bytes(_make_melody_midi(notes))
    return path


def _write_chords_review(prod_dir: Path, label: str) -> Path:
    """Write a minimal approved chords/review.yml for `generate_quartet`'s chord lookup."""
    chords_dir = prod_dir / "chords"
    chords_dir.mkdir(parents=True, exist_ok=True)
    review = {
        "time_sig": "4/4",
        "candidates": [
            {
                "label": label,
                "status": "approved",
                "chords": [
                    {"notes": ["C3", "E3", "G3"]},
                    {"notes": ["F3", "A3", "C4"]},
                ],
            }
        ],
    }
    path = chords_dir / "review.yml"
    with open(path, "w") as f:
        yaml.dump(review, f, sort_keys=False, allow_unicode=True, width=float("inf"))
    return path


def _write_melody_review(prod_dir: Path, label: str, section: str) -> Path:
    """Write a minimal melody/review.yml candidate mapping an approved instance
    label back to its underlying generated section (for _melody_section_for_label)."""
    melody_dir = prod_dir / "melody"
    melody_dir.mkdir(parents=True, exist_ok=True)
    review = {
        "candidates": [
            {"label": label, "section": section, "status": "approved"},
        ],
    }
    path = melody_dir / "review.yml"
    with open(path, "w") as f:
        yaml.dump(review, f, sort_keys=False, allow_unicode=True, width=float("inf"))
    return path


# ---------------------------------------------------------------------------
# 1. extract_soprano_notes
# ---------------------------------------------------------------------------


class TestExtractSopranoNotes:
    def test_extracts_correct_notes(self, tmp_path):
        from white_generation.pipelines.quartet_pipeline import extract_soprano_notes

        notes = [60, 62, 64, 65]
        midi_bytes = _make_melody_midi(notes)
        result = extract_soprano_notes(midi_bytes)
        assert result == notes

    def test_empty_midi_returns_empty(self, tmp_path):
        from white_generation.pipelines.quartet_pipeline import extract_soprano_notes

        mid = mido.MidiFile(ticks_per_beat=480)
        track = mido.MidiTrack()
        mid.tracks.append(track)
        track.append(mido.MetaMessage("end_of_track", time=0))
        buf = io.BytesIO()
        mid.save(file=buf)
        assert extract_soprano_notes(buf.getvalue()) == []


# ---------------------------------------------------------------------------
# 2. build_quartet_midi
# ---------------------------------------------------------------------------


class TestBuildQuartetMidi:
    def _note_events(self, notes: list[int], tpb: int = 480):
        return [(i * tpb, pitch, tpb) for i, pitch in enumerate(notes)]

    def test_four_channels_present(self):
        from white_generation.pipelines.quartet_pipeline import build_quartet_midi

        soprano = [60, 62, 64, 65]
        events = self._note_events(soprano)
        alto = [55, 57, 59, 60]
        tenor = [52, 53, 55, 57]
        bass = [48, 48, 47, 48]

        midi_bytes = build_quartet_midi(events, alto, tenor, bass, bpm=120)
        mid = mido.MidiFile(file=io.BytesIO(midi_bytes))

        channels_used = set()
        for track in mid.tracks:
            for msg in track:
                if hasattr(msg, "channel"):
                    channels_used.add(msg.channel)

        assert {0, 1, 2, 3}.issubset(channels_used)

    def test_output_is_valid_midi(self):
        from white_generation.pipelines.quartet_pipeline import build_quartet_midi

        events = self._note_events([60, 62, 64])
        midi_bytes = build_quartet_midi(
            events, [55, 57, 59], [50, 52, 53], [43, 45, 47]
        )
        mid = mido.MidiFile(file=io.BytesIO(midi_bytes))
        assert mid.ticks_per_beat == 480


# ---------------------------------------------------------------------------
# Mock scorer helper
# ---------------------------------------------------------------------------


def _make_mock_scorer():
    """Return a Refractor-shaped mock that returns plausible score dicts."""
    mock = MagicMock()
    mock.prepare_concept.return_value = MagicMock(shape=(768,))

    def _score(midi_bytes, concept_emb=None):
        return {
            "temporal": {"past": 0.6, "present": 0.3, "future": 0.1},
            "spatial": {"thing": 0.5, "place": 0.3, "person": 0.2},
            "ontological": {"imagined": 0.2, "forgotten": 0.3, "known": 0.5},
            "confidence": 0.8,
        }

    mock.score.side_effect = _score
    return mock


# ---------------------------------------------------------------------------
# 2b. _resolve_chord_label
# ---------------------------------------------------------------------------


class TestResolveChordLabel:
    def test_exact_match(self):
        from white_generation.pipelines.quartet_pipeline import _resolve_chord_label

        assert _resolve_chord_label("verse", ["verse", "chorus"]) == "verse"

    def test_prefix_match_for_split_suffix(self):
        from white_generation.pipelines.quartet_pipeline import _resolve_chord_label

        assert _resolve_chord_label("chorus_split", ["verse", "chorus"]) == "chorus"

    def test_prefix_match_for_numbered_and_split_suffix(self):
        from white_generation.pipelines.quartet_pipeline import _resolve_chord_label

        assert _resolve_chord_label("verse_alt_2_split", ["verse", "bridge"]) == "verse"

    def test_prefix_match_for_inst_suffix(self):
        from white_generation.pipelines.quartet_pipeline import _resolve_chord_label

        assert _resolve_chord_label("bridge_inst", ["bridge", "verse"]) == "bridge"

    def test_no_match_returns_none(self):
        from white_generation.pipelines.quartet_pipeline import _resolve_chord_label

        assert _resolve_chord_label("mystery", ["verse", "chorus"]) is None

    def test_prefers_longest_prefix_match(self):
        from white_generation.pipelines.quartet_pipeline import _resolve_chord_label

        assert (
            _resolve_chord_label("verse_alt_split", ["verse", "verse_alt"])
            == "verse_alt"
        )


# ---------------------------------------------------------------------------
# 2b-ii. _melody_section_for_label
# ---------------------------------------------------------------------------


class TestMelodySectionForLabel:
    def test_maps_instance_label_to_section(self, tmp_path):
        from white_generation.pipelines.quartet_pipeline import (
            _melody_section_for_label,
        )

        prod = tmp_path / "production" / "test_song"
        prod.mkdir(parents=True)
        _write_melody_review(prod, label="1", section="double")

        assert _melody_section_for_label(prod, "1") == "double"

    def test_falls_back_to_label_when_no_review_file(self, tmp_path):
        from white_generation.pipelines.quartet_pipeline import (
            _melody_section_for_label,
        )

        prod = tmp_path / "production" / "test_song"
        prod.mkdir(parents=True)

        assert _melody_section_for_label(prod, "verse") == "verse"

    def test_falls_back_to_label_when_no_matching_candidate(self, tmp_path):
        from white_generation.pipelines.quartet_pipeline import (
            _melody_section_for_label,
        )

        prod = tmp_path / "production" / "test_song"
        prod.mkdir(parents=True)
        _write_melody_review(prod, label="2", section="double")

        assert _melody_section_for_label(prod, "1") == "1"


# ---------------------------------------------------------------------------
# 2c. _voice_length_beats / _align_voice_length
# ---------------------------------------------------------------------------


class TestAlignVoiceLength:
    def test_voice_length_beats_matches_note_span(self):
        from white_generation.pipelines.quartet_pipeline import _voice_length_beats

        midi_bytes = _make_melody_midi([60, 62, 64, 65])  # 4 quarter notes = 4 beats
        assert _voice_length_beats(midi_bytes) == 4.0

    def test_pads_short_voice_by_sustaining_last_note(self):
        from white_generation.pipelines.quartet_pipeline import (
            _align_voice_length,
            _parse_voice_notes_with_velocity,
            _voice_length_beats,
        )

        midi_bytes = _make_melody_midi([60, 62])  # 2 beats
        aligned = _align_voice_length(midi_bytes, target_beats=4.0)
        assert _voice_length_beats(aligned) == 4.0
        # Still the same 2 notes — no notes invented, just the last one sustained.
        notes, _ = _parse_voice_notes_with_velocity(aligned)
        assert len(notes) == 2

    def test_trims_long_voice(self):
        from white_generation.pipelines.quartet_pipeline import (
            _align_voice_length,
            _voice_length_beats,
        )

        midi_bytes = _make_melody_midi([60, 62, 64, 65, 67, 69])  # 6 beats
        aligned = _align_voice_length(midi_bytes, target_beats=4.0)
        assert _voice_length_beats(aligned) == 4.0

    def test_already_aligned_voice_unchanged_length(self):
        from white_generation.pipelines.quartet_pipeline import (
            _align_voice_length,
            _voice_length_beats,
        )

        midi_bytes = _make_melody_midi([60, 62, 64, 65])  # 4 beats
        aligned = _align_voice_length(midi_bytes, target_beats=4.0)
        assert _voice_length_beats(aligned) == 4.0

    def test_empty_midi_returned_unchanged(self):
        from white_generation.pipelines.quartet_pipeline import _align_voice_length

        mid = mido.MidiFile(ticks_per_beat=480)
        track = mido.MidiTrack()
        mid.tracks.append(track)
        track.append(mido.MetaMessage("end_of_track", time=0))
        buf = io.BytesIO()
        mid.save(file=buf)
        empty_bytes = buf.getvalue()
        assert _align_voice_length(empty_bytes, target_beats=4.0) == empty_bytes


# ---------------------------------------------------------------------------
# 3. generate_quartet (integration)
# ---------------------------------------------------------------------------


class TestGenerateQuartet:
    def test_returns_top_k_candidates(self, tmp_path):
        from white_generation.pipelines.quartet_pipeline import generate_quartet

        prod = tmp_path / "production" / "test_song"
        prod.mkdir(parents=True)
        _write_song_context(prod)
        _write_melody_approved(prod, "chorus", [60, 62, 64, 65, 67, 65, 64, 62])
        _write_chords_review(prod, "chorus")

        candidates = generate_quartet(
            prod, "chorus", top_k=2, seed=42, scorer=_make_mock_scorer()
        )
        assert len(candidates) == 2

    def test_candidates_have_required_fields(self, tmp_path):
        from white_generation.pipelines.quartet_pipeline import generate_quartet

        prod = tmp_path / "production" / "test_song"
        prod.mkdir(parents=True)
        _write_song_context(prod)
        _write_melody_approved(prod, "verse", [60, 62, 64, 65])
        _write_chords_review(prod, "verse")

        candidates = generate_quartet(
            prod, "verse", top_k=1, seed=0, scorer=_make_mock_scorer()
        )
        c = candidates[0]
        assert "id" in c
        assert "composite_score" in c
        assert "violin_ii_pattern" in c
        assert "viola_pattern" in c
        assert "cello_pattern" in c
        assert "midi_bytes" in c
        assert isinstance(c["midi_bytes"], bytes)

    def test_candidates_sorted_by_score_descending(self, tmp_path):
        from white_generation.pipelines.quartet_pipeline import generate_quartet

        prod = tmp_path / "production" / "test_song"
        prod.mkdir(parents=True)
        _write_song_context(prod)
        _write_melody_approved(prod, "bridge", [60, 62, 64, 65, 64, 62])
        _write_chords_review(prod, "bridge")

        candidates = generate_quartet(
            prod, "bridge", top_k=3, seed=7, scorer=_make_mock_scorer()
        )
        scores = [c["composite_score"] for c in candidates]
        assert scores == sorted(scores, reverse=True)

    def test_missing_midi_raises(self, tmp_path):
        from white_generation.pipelines.quartet_pipeline import generate_quartet

        prod = tmp_path / "production" / "test_song"
        prod.mkdir(parents=True)
        _write_song_context(prod)
        (prod / "melody" / "approved").mkdir(parents=True)

        with pytest.raises(FileNotFoundError):
            generate_quartet(prod, "nonexistent_section", top_k=1)

    def test_no_scorer_composite_equals_counterpoint(self, tmp_path):
        """When Refractor is unavailable, composite score equals counterpoint (no 0.5 bias)."""
        from white_generation.pipelines.quartet_pipeline import generate_quartet

        prod = tmp_path / "production" / "test_song"
        prod.mkdir(parents=True)
        _write_song_context(prod)
        _write_melody_approved(prod, "verse", [60, 62, 64, 65])
        _write_chords_review(prod, "verse")

        with patch.dict("sys.modules", {"white_analysis.refractor": None}):
            candidates = generate_quartet(prod, "verse", top_k=1, seed=0, scorer=None)
        c = candidates[0]
        assert c["composite_score"] == c["scores"]["counterpoint"]
        assert c["scores"]["chromatic"] is None

    def test_unmatched_chord_label_raises_clear_error(self, tmp_path):
        from white_generation.pipelines.quartet_pipeline import generate_quartet

        prod = tmp_path / "production" / "test_song"
        prod.mkdir(parents=True)
        _write_song_context(prod)
        _write_melody_approved(prod, "mystery_section", [60, 62, 64, 65])
        _write_chords_review(prod, "verse")  # unrelated to 'mystery_section'

        with pytest.raises(ValueError, match="No chord data found"):
            generate_quartet(
                prod, "mystery_section", top_k=1, scorer=_make_mock_scorer()
            )

    def test_split_suffix_label_resolves_to_base_chord(self, tmp_path):
        """A melody label with a _split suffix (from melody_auto_split) should still
        resolve to its base chord section instead of raising or grabbing an
        unrelated section's harmony."""
        from white_generation.pipelines.quartet_pipeline import generate_quartet

        prod = tmp_path / "production" / "test_song"
        prod.mkdir(parents=True)
        _write_song_context(prod)
        _write_melody_approved(prod, "verse_split", [60, 62, 64, 65])
        _write_chords_review(prod, "verse")

        candidates = generate_quartet(
            prod, "verse_split", top_k=1, seed=0, scorer=_make_mock_scorer()
        )
        assert len(candidates) == 1

    def test_repeated_section_under_instance_label_resolves_chords(self, tmp_path):
        """Regression: a section generated once ('double') but approved multiple
        times under distinct per-instance labels ('1'..'5' for separate
        occurrences in the arrangement) must still resolve to that section's
        chords via melody/review.yml, not raise 'No chord data found' just
        because the instance label has no string relationship to the chord
        section name."""
        from white_generation.pipelines.quartet_pipeline import generate_quartet

        prod = tmp_path / "production" / "test_song"
        prod.mkdir(parents=True)
        _write_song_context(prod)
        _write_melody_approved(prod, "1", [60, 62, 64, 65])
        _write_melody_review(prod, label="1", section="double")
        _write_chords_review(prod, "double")

        candidates = generate_quartet(
            prod, "1", top_k=1, seed=0, scorer=_make_mock_scorer()
        )
        assert len(candidates) == 1

    def test_all_voices_match_soprano_length(self, tmp_path):
        """Regression: violin_ii/viola/cello must end at the same tick as the
        soprano, not drift short (a dangling gap) or long (an extra measure) from
        their independently-derived harmonic-rhythm durations."""
        from white_generation.pipelines.quartet_pipeline import generate_quartet

        prod = tmp_path / "production" / "test_song"
        prod.mkdir(parents=True)
        _write_song_context(prod)
        _write_melody_approved(prod, "verse", [60, 62, 64, 65, 67, 69, 71, 72])
        _write_chords_review(prod, "verse")

        candidates = generate_quartet(
            prod, "verse", top_k=1, seed=3, scorer=_make_mock_scorer()
        )
        midi_bytes = candidates[0]["midi_bytes"]
        mid = mido.MidiFile(file=io.BytesIO(midi_bytes))

        end_tick_by_channel: dict[int, int] = {}
        for track in mid.tracks:
            abs_tick = 0
            for msg in track:
                abs_tick += msg.time
                if msg.type in ("note_on", "note_off"):
                    end_tick_by_channel[msg.channel] = max(
                        end_tick_by_channel.get(msg.channel, 0), abs_tick
                    )

        assert len(end_tick_by_channel) == 4
        assert len(set(end_tick_by_channel.values())) == 1


# ---------------------------------------------------------------------------
# 4. write_quartet_candidates
# ---------------------------------------------------------------------------


class TestWriteQuartetCandidates:
    def _make_candidates(self, section: str, n: int) -> list[dict]:
        midi = _make_melody_midi([60, 62, 64, 65])
        return [
            {
                "id": f"{section}_quartet_{i + 1:03d}",
                "label": section,
                "section": section,
                "singer": "gabriel",
                "violin_ii_pattern": "conversational",
                "viola_pattern": "stepwise",
                "cello_pattern": "root_pulse",
                "scores": {"counterpoint": 0.9, "chromatic": 0.8, "composite": 0.83},
                "composite_score": 0.83,
                "midi_bytes": midi,
                "status": "pending",
            }
            for i in range(n)
        ]

    def test_midi_files_written(self, tmp_path):
        from white_generation.pipelines.quartet_pipeline import (
            write_quartet_candidates,
        )

        prod = tmp_path / "production" / "test_song"
        prod.mkdir(parents=True)
        _write_song_context(prod)
        candidates = self._make_candidates("chorus", 2)
        cands_dir, review_path = write_quartet_candidates(candidates, prod, "chorus")

        assert cands_dir.exists()
        mid_files = list(cands_dir.glob("*.mid"))
        assert len(mid_files) == 2

    def test_review_yml_written(self, tmp_path):
        from white_generation.pipelines.quartet_pipeline import (
            write_quartet_candidates,
        )

        prod = tmp_path / "production" / "test_song"
        prod.mkdir(parents=True)
        _write_song_context(prod)
        candidates = self._make_candidates("verse", 1)
        _, review_path = write_quartet_candidates(candidates, prod, "verse")

        assert review_path.exists()
        with open(review_path) as f:
            data = yaml.safe_load(f)
        assert data["phase"] == "quartet"
        assert len(data["candidates"]) == 1
        assert data["candidates"][0]["label"] == "verse"
        assert data["candidates"][0]["status"] == "pending"

    def test_appending_second_section_merges(self, tmp_path):
        from white_generation.pipelines.quartet_pipeline import (
            write_quartet_candidates,
        )

        prod = tmp_path / "production" / "test_song"
        prod.mkdir(parents=True)
        _write_song_context(prod)

        write_quartet_candidates(self._make_candidates("verse", 1), prod, "verse")
        write_quartet_candidates(self._make_candidates("chorus", 1), prod, "chorus")

        review_path = prod / "quartet" / "review.yml"
        with open(review_path) as f:
            data = yaml.safe_load(f)
        labels = [c["label"] for c in data["candidates"]]
        assert "verse" in labels
        assert "chorus" in labels

    def test_regenerating_section_replaces_old_entries(self, tmp_path):
        from white_generation.pipelines.quartet_pipeline import (
            write_quartet_candidates,
        )

        prod = tmp_path / "production" / "test_song"
        prod.mkdir(parents=True)
        _write_song_context(prod)

        write_quartet_candidates(self._make_candidates("chorus", 3), prod, "chorus")
        write_quartet_candidates(self._make_candidates("chorus", 1), prod, "chorus")

        review_path = prod / "quartet" / "review.yml"
        with open(review_path) as f:
            data = yaml.safe_load(f)
        chorus_entries = [c for c in data["candidates"] if c["label"] == "chorus"]
        assert len(chorus_entries) == 1  # replaced, not appended

    def test_review_uses_midi_file_key(self, tmp_path):
        """review.yml entries use 'midi_file' so promote_part.py recognises them."""
        from white_generation.pipelines.quartet_pipeline import (
            write_quartet_candidates,
        )

        prod = tmp_path / "production" / "test_song"
        prod.mkdir(parents=True)
        _write_song_context(prod)
        _, review_path = write_quartet_candidates(
            self._make_candidates("verse", 1), prod, "verse"
        )

        with open(review_path) as f:
            data = yaml.safe_load(f)
        cand = data["candidates"][0]
        assert "midi_file" in cand
        assert cand["midi_file"].startswith("candidates/")
        assert "file" not in cand

    def test_stale_midi_files_cleaned_on_regenerate(self, tmp_path):
        """Old MIDI files for a section are removed before writing new candidates."""
        from white_generation.pipelines.quartet_pipeline import (
            write_quartet_candidates,
        )

        prod = tmp_path / "production" / "test_song"
        prod.mkdir(parents=True)
        _write_song_context(prod)

        # Write 3 candidates first
        write_quartet_candidates(self._make_candidates("chorus", 3), prod, "chorus")
        cands_dir = prod / "quartet" / "candidates"
        assert len(list(cands_dir.glob("chorus_quartet_*.mid"))) == 3

        # Regenerate with 1 candidate — stale files should be gone
        write_quartet_candidates(self._make_candidates("chorus", 1), prod, "chorus")
        assert len(list(cands_dir.glob("chorus_quartet_*.mid"))) == 1
