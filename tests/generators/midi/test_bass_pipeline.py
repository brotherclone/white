"""Tests for the bass line generation pipeline."""

import io
from pathlib import Path

import mido
import pytest
import yaml


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_chord_midi(voicings: list[list[int]], bpm: int = 120) -> bytes:
    """Create a minimal chord MIDI file from voicings (one chord per bar)."""
    mid = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack()
    mid.tracks.append(track)
    track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(bpm), time=0))

    bar_ticks = 480 * 4  # 4/4
    for i, notes in enumerate(voicings):
        for note in notes:
            track.append(mido.Message("note_on", note=note, velocity=80, time=0))
        # Hold for one bar
        track.append(
            mido.Message("note_off", note=notes[0], velocity=0, time=bar_ticks)
        )
        for note in notes[1:]:
            track.append(mido.Message("note_off", note=note, velocity=0, time=0))

    track.append(mido.MetaMessage("end_of_track", time=0))
    buf = io.BytesIO()
    mid.save(file=buf)
    return buf.getvalue()


def make_drum_midi(kick_beats: list[float], bpm: int = 120) -> bytes:
    """Create a minimal drum MIDI with kick hits at specified beat positions."""
    mid = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack()
    mid.tracks.append(track)
    track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(bpm), time=0))

    events = []
    for beat in kick_beats:
        abs_tick = int(beat * 480)
        events.append((abs_tick, 36, 100, True))
        events.append((abs_tick + 120, 36, 0, False))

    events.sort(key=lambda e: (e[0], not e[3]))
    prev_tick = 0
    for abs_tick, note, vel, is_on in events:
        delta = abs_tick - prev_tick
        msg_type = "note_on" if is_on else "note_off"
        track.append(
            mido.Message(msg_type, note=note, velocity=vel, time=delta, channel=9)
        )
        prev_tick = abs_tick

    track.append(mido.MetaMessage("end_of_track", time=0))
    buf = io.BytesIO()
    mid.save(file=buf)
    return buf.getvalue()


def setup_production_dir(
    tmp_path: Path,
    voicings: list[list[int]] | None = None,
    kick_beats: list[float] | None = None,
    sections: list[str] | None = None,
    hr_durations: dict | None = None,
) -> Path:
    """Set up a mock production directory with chord/drum/HR data."""
    prod_dir = tmp_path / "production" / "test_song"

    # Chord data
    if voicings is None:
        voicings = [[60, 64, 67], [62, 65, 69]]  # C major, D minor
    if sections is None:
        sections = ["Verse"]

    chords_dir = prod_dir / "chords"
    approved_dir = chords_dir / "approved"
    approved_dir.mkdir(parents=True)

    # Write chord MIDI
    for section in sections:
        label = section.lower().replace(" ", "_")
        midi_path = approved_dir / f"{label}.mid"
        midi_path.write_bytes(make_chord_midi(voicings))

    # Write chord review.yml
    candidates = []
    for section in sections:
        candidates.append(
            {
                "id": f"chord_{section.lower()}_01",
                "label": section,
                "status": "approved",
                "chords": [{"notes": ["C4"]}] * len(voicings),
            }
        )

    review = {
        "bpm": 120,
        "time_sig": "4/4",
        "color": "Red",
        "thread": "",
        "song_proposal": "",
        "candidates": candidates,
    }
    with open(chords_dir / "review.yml", "w") as f:
        yaml.dump(review, f)

    # Drum data (optional)
    if kick_beats is not None:
        drums_dir = prod_dir / "drums"
        drums_approved = drums_dir / "approved"
        drums_approved.mkdir(parents=True)

        for section in sections:
            label = section.lower().replace(" ", "_")
            drum_path = drums_approved / f"{label}.mid"
            drum_path.write_bytes(make_drum_midi(kick_beats))

        drum_candidates = []
        for section in sections:
            drum_candidates.append(
                {
                    "id": f"drum_{section.lower()}_01",
                    "section": section,
                    "label": section.lower().replace(" ", "_"),
                    "status": "approved",
                }
            )
        drum_review = {"candidates": drum_candidates}
        with open(drums_dir / "review.yml", "w") as f:
            yaml.dump(drum_review, f)

    # Harmonic rhythm (optional)
    if hr_durations is not None:
        hr_dir = prod_dir / "harmonic_rhythm"
        hr_dir.mkdir(parents=True)
        hr_candidates = []
        for section in sections:
            label = section.lower().replace(" ", "_")
            if label in hr_durations:
                hr_candidates.append(
                    {
                        "section": section,
                        "status": "approved",
                        "distribution": hr_durations[label],
                    }
                )
        hr_review = {"candidates": hr_candidates}
        with open(hr_dir / "review.yml", "w") as f:
            yaml.dump(hr_review, f)

    return prod_dir


# ---------------------------------------------------------------------------
# 1. Chord root extraction
# ---------------------------------------------------------------------------


class TestChordRootExtraction:

    def test_extract_section_chord_data(self, tmp_path):
        from app.generators.midi.bass_pipeline import extract_section_chord_data

        prod_dir = setup_production_dir(tmp_path)
        chord_data, review = extract_section_chord_data(prod_dir)

        assert "verse" in chord_data
        assert len(chord_data["verse"]) == 2
        assert isinstance(review, dict)

    def test_extract_section_chord_data_missing_dir(self, tmp_path):
        from app.generators.midi.bass_pipeline import extract_section_chord_data

        with pytest.raises(FileNotFoundError):
            extract_section_chord_data(tmp_path / "nonexistent")


# ---------------------------------------------------------------------------
# 2. Kick onset extraction
# ---------------------------------------------------------------------------


class TestKickOnsetExtraction:

    def test_extract_kick_onsets(self, tmp_path):
        from app.generators.midi.bass_pipeline import extract_kick_onsets

        kick_path = tmp_path / "kick.mid"
        kick_path.write_bytes(make_drum_midi([0.0, 1.0, 2.0, 3.0]))
        onsets = extract_kick_onsets(str(kick_path))
        assert len(onsets) == 4
        assert onsets[0] == pytest.approx(0.0)

    def test_read_approved_kick_onsets(self, tmp_path):
        from app.generators.midi.bass_pipeline import read_approved_kick_onsets

        prod_dir = setup_production_dir(tmp_path, kick_beats=[0.0, 2.0])
        kicks = read_approved_kick_onsets(prod_dir)
        assert "verse" in kicks
        assert len(kicks["verse"]) == 2

    def test_read_approved_kick_onsets_no_drums(self, tmp_path):
        from app.generators.midi.bass_pipeline import read_approved_kick_onsets

        prod_dir = setup_production_dir(tmp_path)  # no kick_beats
        kicks = read_approved_kick_onsets(prod_dir)
        assert kicks == {}


# ---------------------------------------------------------------------------
# 3. MIDI generation
# ---------------------------------------------------------------------------


class TestBassMidiGeneration:

    def test_basic_midi_generation(self):
        from app.generators.midi.bass_patterns import TEMPLATES_4_4_ROOT
        from app.generators.midi.bass_pipeline import bass_pattern_to_midi_bytes

        pattern = TEMPLATES_4_4_ROOT[0]  # root_whole
        voicings = [[60, 64, 67]]  # C major

        midi_bytes, resolved = bass_pattern_to_midi_bytes(pattern, voicings)

        assert len(midi_bytes) > 0
        assert len(resolved) > 0

        # Parse and verify
        mid = mido.MidiFile(file=io.BytesIO(midi_bytes))
        notes = [
            msg for msg in mid.tracks[0] if msg.type == "note_on" and msg.velocity > 0
        ]
        assert len(notes) >= 1
        for msg in notes:
            assert msg.channel == 0
            assert 24 <= msg.note <= 60

    def test_midi_with_multiple_chords(self):
        from app.generators.midi.bass_patterns import TEMPLATES_4_4_WALKING
        from app.generators.midi.bass_pipeline import bass_pattern_to_midi_bytes

        pattern = TEMPLATES_4_4_WALKING[0]  # walking_quarter
        voicings = [[60, 64, 67], [62, 65, 69]]  # C major, D minor

        midi_bytes, resolved = bass_pattern_to_midi_bytes(pattern, voicings)

        mid = mido.MidiFile(file=io.BytesIO(midi_bytes))
        notes = [
            msg for msg in mid.tracks[0] if msg.type == "note_on" and msg.velocity > 0
        ]
        # walking_quarter has 4 notes per bar × 2 chords = 8 notes
        assert len(notes) == 8

    def test_midi_with_harmonic_rhythm(self):
        from app.generators.midi.bass_patterns import TEMPLATES_4_4_ROOT
        from app.generators.midi.bass_pipeline import bass_pattern_to_midi_bytes

        pattern = TEMPLATES_4_4_ROOT[1]  # root_half (2 notes per bar)
        voicings = [[60, 64, 67], [62, 65, 69]]
        durations = [2.0, 1.0]  # first chord 2 bars, second chord 1 bar

        midi_bytes, resolved = bass_pattern_to_midi_bytes(
            pattern, voicings, durations=durations
        )

        mid = mido.MidiFile(file=io.BytesIO(midi_bytes))
        notes = [
            msg for msg in mid.tracks[0] if msg.type == "note_on" and msg.velocity > 0
        ]
        # First chord: 2 bars × 2 notes = 4, second: 1 bar × 2 notes = 2
        assert len(notes) == 6

    def test_all_notes_in_bass_register(self):
        from app.generators.midi.bass_patterns import ALL_TEMPLATES
        from app.generators.midi.bass_pipeline import bass_pattern_to_midi_bytes

        # Test with a high voicing to verify register clamping
        high_voicing = [[72, 76, 79]]  # C5, E5, G5

        for tmpl in ALL_TEMPLATES:
            if tmpl.time_sig != (4, 4):
                continue
            midi_bytes, resolved = bass_pattern_to_midi_bytes(tmpl, high_voicing)
            for beat_pos, note in resolved:
                assert (
                    24 <= note <= 60
                ), f"{tmpl.name}: note {note} outside bass register"


# ---------------------------------------------------------------------------
# 4. Composite scoring
# ---------------------------------------------------------------------------


class TestCompositeScoring:

    def test_bass_composite_score(self):
        from app.generators.midi.bass_pipeline import bass_composite_score

        scorer_result = {
            "temporal": {"past": 0.8, "present": 0.1, "future": 0.1},
            "spatial": {"thing": 0.7, "place": 0.2, "person": 0.1},
            "ontological": {"imagined": 0.1, "forgotten": 0.1, "known": 0.8},
            "confidence": 0.9,
        }
        theory_breakdown = {
            "root_adherence": 1.0,
            "kick_alignment": 0.5,
            "voice_leading": 0.8,
        }

        composite, breakdown = bass_composite_score(
            theory=0.77,
            chromatic_match=0.85,
            scorer_result=scorer_result,
            theory_breakdown=theory_breakdown,
        )

        assert 0 <= composite <= 1.0
        assert "composite" in breakdown
        assert "theory" in breakdown
        assert "chromatic" in breakdown
        assert breakdown["theory"]["root_adherence"] == 1.0

    def test_composite_weights(self):
        from app.generators.midi.bass_pipeline import bass_composite_score

        scorer_result = {
            "temporal": {"past": 0.5, "present": 0.3, "future": 0.2},
            "spatial": {"thing": 0.5, "place": 0.3, "person": 0.2},
            "ontological": {"imagined": 0.5, "forgotten": 0.3, "known": 0.2},
            "confidence": 0.5,
        }

        comp_30_70, _ = bass_composite_score(
            1.0, 0.0, scorer_result, {}, theory_weight=0.3, chromatic_weight=0.7
        )
        comp_70_30, _ = bass_composite_score(
            1.0, 0.0, scorer_result, {}, theory_weight=0.7, chromatic_weight=0.3
        )

        # Higher theory weight → higher score when theory=1.0, chromatic=0.0
        assert comp_70_30 > comp_30_70


# ---------------------------------------------------------------------------
# 5. Review YAML
# ---------------------------------------------------------------------------


class TestReviewYaml:

    def test_generate_review_yaml(self):
        from app.generators.midi.bass_pipeline import generate_bass_review_yaml

        sections = [
            {
                "label": "verse",
                "label_display": "Verse",
                "chord_id": "c01",
                "_section_key": "verse",
            },
        ]
        ranked = {
            "verse": [
                {
                    "id": "bass_verse_01",
                    "style": "root",
                    "pattern_name": "root_whole",
                    "energy": "low",
                    "breakdown": {"composite": 0.8, "theory": {}, "chromatic": {}},
                }
            ]
        }

        review = generate_bass_review_yaml(
            "/tmp/prod",
            sections,
            ranked,
            seed=42,
            scoring_weights={"theory": 0.3, "chromatic": 0.7},
            song_info={"bpm": 120, "time_sig": (4, 4), "color_name": "Red"},
        )

        assert review["pipeline"] == "bass-generation"
        assert review["bpm"] == 120
        assert len(review["candidates"]) == 1
        assert review["candidates"][0]["status"] == "pending"
        assert review["candidates"][0]["label"] is None


# ---------------------------------------------------------------------------
# 6. Section reading
# ---------------------------------------------------------------------------


class TestReadApprovedSections:

    def test_reads_approved_sections(self):
        from app.generators.midi.bass_pipeline import read_approved_sections

        review = {
            "candidates": [
                {"label": "Verse", "status": "approved", "id": "c01"},
                {"label": "Chorus", "status": "pending", "id": "c02"},
                {"label": "Bridge", "status": "accepted", "id": "c03"},
            ]
        }

        sections = read_approved_sections(review)
        assert len(sections) == 2  # Verse + Bridge
        labels = [s["label"] for s in sections]
        assert "verse" in labels
        assert "bridge" in labels

    def test_skips_non_approved(self):
        from app.generators.midi.bass_pipeline import read_approved_sections

        review = {
            "candidates": [
                {"label": "Verse", "status": "rejected", "id": "c01"},
            ]
        }
        assert read_approved_sections(review) == []


# ---------------------------------------------------------------------------
# 7. Integration test (mocked ChromaticScorer)
# ---------------------------------------------------------------------------


class TestIntegration:

    def test_end_to_end_with_mock_scorer(self, tmp_path, monkeypatch):
        """Run the full pipeline with a mocked ChromaticScorer."""
        prod_dir = setup_production_dir(
            tmp_path,
            voicings=[[60, 64, 67], [55, 59, 62]],  # C major, G major
            kick_beats=[0.0, 2.0],
            sections=["Verse"],
        )

        class MockScorer:
            def __init__(self, onnx_path=None):
                pass

            def prepare_concept(self, text):
                import numpy as np

                return np.zeros(768)

            def score_batch(self, candidates, concept_emb=None):
                return [
                    {
                        "candidate": c,
                        "temporal": {"past": 0.8, "present": 0.1, "future": 0.1},
                        "spatial": {"thing": 0.7, "place": 0.2, "person": 0.1},
                        "ontological": {
                            "imagined": 0.1,
                            "forgotten": 0.1,
                            "known": 0.8,
                        },
                        "confidence": 0.85,
                    }
                    for c in candidates
                ]

        monkeypatch.setattr(
            "training.chromatic_scorer.ChromaticScorer",
            MockScorer,
        )

        from app.generators.midi.bass_pipeline import run_bass_pipeline

        result = run_bass_pipeline(
            production_dir=str(prod_dir),
            seed=42,
            top_k=3,
        )

        # Verify output
        assert "verse" in result
        assert len(result["verse"]) <= 3

        # Verify files written
        bass_dir = prod_dir / "bass"
        assert bass_dir.exists()
        assert (bass_dir / "review.yml").exists()
        assert (bass_dir / "candidates").exists()
        assert (bass_dir / "approved").exists()

        # Verify MIDI files
        midi_files = list((bass_dir / "candidates").glob("*.mid"))
        assert len(midi_files) > 0

        # Verify review YAML structure
        with open(bass_dir / "review.yml") as f:
            review = yaml.safe_load(f)
        assert review["pipeline"] == "bass-generation"
        assert review["color"] == "Red"
        assert len(review["candidates"]) > 0
        for cand in review["candidates"]:
            assert cand["status"] == "pending"
            assert "scores" in cand

        # Verify all MIDI notes in bass register
        for midi_file in midi_files:
            mid = mido.MidiFile(str(midi_file))
            for msg in mid.tracks[0]:
                if msg.type == "note_on" and msg.velocity > 0:
                    assert (
                        24 <= msg.note <= 60
                    ), f"{midi_file.name}: note {msg.note} out of range"
                    assert (
                        msg.channel == 0
                    ), f"{midi_file.name}: wrong channel {msg.channel}"

    def test_pipeline_with_harmonic_rhythm(self, tmp_path, monkeypatch):
        """Verify pipeline reads harmonic rhythm durations."""
        prod_dir = setup_production_dir(
            tmp_path,
            voicings=[[60, 64, 67], [55, 59, 62]],
            kick_beats=[0.0, 2.0],
            sections=["Verse"],
            hr_durations={"verse": [1.5, 0.5]},
        )

        class MockScorer:
            def __init__(self, onnx_path=None):
                pass

            def prepare_concept(self, text):
                import numpy as np

                return np.zeros(768)

            def score_batch(self, candidates, concept_emb=None):
                return [
                    {
                        "candidate": c,
                        "temporal": {"past": 0.5, "present": 0.3, "future": 0.2},
                        "spatial": {"thing": 0.5, "place": 0.3, "person": 0.2},
                        "ontological": {
                            "imagined": 0.3,
                            "forgotten": 0.3,
                            "known": 0.4,
                        },
                        "confidence": 0.7,
                    }
                    for c in candidates
                ]

        monkeypatch.setattr(
            "training.chromatic_scorer.ChromaticScorer",
            MockScorer,
        )

        from app.generators.midi.bass_pipeline import run_bass_pipeline

        result = run_bass_pipeline(
            production_dir=str(prod_dir),
            seed=42,
            top_k=3,
        )

        assert "verse" in result
        midi_files = list((prod_dir / "bass" / "candidates").glob("*.mid"))
        assert len(midi_files) > 0

    def test_pipeline_without_drums(self, tmp_path, monkeypatch):
        """Pipeline should work without approved drums (kick alignment disabled)."""
        prod_dir = setup_production_dir(
            tmp_path,
            voicings=[[60, 64, 67]],
            sections=["Verse"],
        )

        class MockScorer:
            def __init__(self, onnx_path=None):
                pass

            def prepare_concept(self, text):
                import numpy as np

                return np.zeros(768)

            def score_batch(self, candidates, concept_emb=None):
                return [
                    {
                        "candidate": c,
                        "temporal": {"past": 0.5, "present": 0.3, "future": 0.2},
                        "spatial": {"thing": 0.5, "place": 0.3, "person": 0.2},
                        "ontological": {
                            "imagined": 0.3,
                            "forgotten": 0.3,
                            "known": 0.4,
                        },
                        "confidence": 0.7,
                    }
                    for c in candidates
                ]

        monkeypatch.setattr(
            "training.chromatic_scorer.ChromaticScorer",
            MockScorer,
        )

        from app.generators.midi.bass_pipeline import run_bass_pipeline

        result = run_bass_pipeline(
            production_dir=str(prod_dir),
            seed=42,
            top_k=3,
        )

        assert "verse" in result
