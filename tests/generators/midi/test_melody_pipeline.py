"""Tests for the melody generation pipeline."""

import io
from unittest.mock import MagicMock, patch

import mido
import pytest
import yaml


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def make_chord_midi(notes: list[int], ticks_per_beat: int = 480) -> bytes:
    """Create a minimal chord MIDI file as bytes."""
    mid = mido.MidiFile(ticks_per_beat=ticks_per_beat)
    track = mido.MidiTrack()
    mid.tracks.append(track)
    track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(120), time=0))
    for n in notes:
        track.append(mido.Message("note_on", note=n, velocity=80, time=0))
    for n in notes:
        track.append(
            mido.Message("note_off", note=n, velocity=0, time=ticks_per_beat * 4)
        )
    track.append(mido.MetaMessage("end_of_track", time=0))
    buf = io.BytesIO()
    mid.save(file=buf)
    return buf.getvalue()


@pytest.fixture
def production_dir(tmp_path):
    """Set up a minimal production directory with approved chords."""
    prod = tmp_path / "production" / "test_song"
    chords_approved = prod / "chords" / "approved"
    chords_approved.mkdir(parents=True)

    # Write a chord MIDI file
    chord_bytes = make_chord_midi([48, 52, 55])  # C major
    (chords_approved / "verse").mkdir(exist_ok=True)
    (chords_approved / "verse.mid").write_bytes(chord_bytes)

    # Write chord review YAML
    review = {
        "bpm": 120,
        "time_sig": "4/4",
        "color": "Red",
        "candidates": [
            {
                "id": "chord_verse_01",
                "label": "verse",
                "status": "approved",
            }
        ],
    }
    with open(prod / "chords" / "review.yml", "w") as f:
        yaml.dump(review, f)

    return prod


# ---------------------------------------------------------------------------
# 1. Section reading
# ---------------------------------------------------------------------------


class TestSectionReading:

    def test_read_approved_sections(self, production_dir):
        from app.generators.midi.melody_pipeline import read_approved_sections

        with open(production_dir / "chords" / "review.yml") as f:
            chord_review = yaml.safe_load(f)

        sections = read_approved_sections(chord_review)
        assert len(sections) == 1
        assert sections[0]["label"] == "verse"
        assert sections[0]["chord_id"] == "chord_verse_01"

    def test_read_approved_sections_skips_pending(self):
        from app.generators.midi.melody_pipeline import read_approved_sections

        review = {
            "candidates": [
                {"id": "a", "label": "verse", "status": "pending"},
                {"id": "b", "label": "chorus", "status": "approved"},
            ]
        }
        sections = read_approved_sections(review)
        assert len(sections) == 1
        assert sections[0]["label"] == "chorus"


# ---------------------------------------------------------------------------
# 2. MIDI generation
# ---------------------------------------------------------------------------


class TestMelodyMidiGeneration:

    def test_melody_notes_to_midi_bytes(self):
        from app.generators.midi.melody_pipeline import melody_notes_to_midi_bytes

        notes = [(0.0, 60, 1.0), (1.0, 62, 1.0), (2.0, 64, 2.0)]
        midi_bytes = melody_notes_to_midi_bytes(notes, bpm=120)

        assert isinstance(midi_bytes, bytes)
        assert len(midi_bytes) > 0

        # Verify it's valid MIDI
        mid = mido.MidiFile(file=io.BytesIO(midi_bytes))
        assert len(mid.tracks) == 1

    def test_generate_melody_for_section(self):
        from app.generators.midi.melody_patterns import ALL_TEMPLATES, SINGERS
        from app.generators.midi.melody_pipeline import generate_melody_for_section

        voicings = [[48, 52, 55], [53, 57, 60]]  # C, F
        singer = SINGERS["gabriel"]
        pattern = ALL_TEMPLATES[0]

        midi_bytes, notes = generate_melody_for_section(
            pattern,
            voicings,
            singer,
            bpm=120,
        )
        assert isinstance(midi_bytes, bytes)
        assert len(notes) > 0

        # All notes within singer range
        for _, note, _ in notes:
            assert singer.low <= note <= singer.high

    def test_generate_melody_with_durations(self):
        from app.generators.midi.melody_patterns import ALL_TEMPLATES, SINGERS
        from app.generators.midi.melody_pipeline import generate_melody_for_section

        voicings = [[48, 52, 55], [53, 57, 60]]
        singer = SINGERS["gabriel"]
        durations = [1.0, 2.0]  # 1 bar, 2 bars

        midi_bytes, notes = generate_melody_for_section(
            ALL_TEMPLATES[0],
            voicings,
            singer,
            bpm=120,
            durations=durations,
        )
        assert len(notes) > 0


# ---------------------------------------------------------------------------
# 3. Composite scoring
# ---------------------------------------------------------------------------


class TestCompositeScoring:

    def test_melody_composite_score(self):
        from app.generators.midi.melody_pipeline import melody_composite_score

        scorer_result = {
            "temporal": {"past": 0.8, "present": 0.1, "future": 0.1},
            "spatial": {"thing": 0.7, "place": 0.2, "person": 0.1},
            "ontological": {"imagined": 0.1, "forgotten": 0.1, "known": 0.8},
            "confidence": 0.85,
        }
        theory_breakdown = {
            "singability": 0.7,
            "chord_tone_alignment": 0.8,
            "contour_quality": 0.6,
        }

        composite, breakdown = melody_composite_score(
            theory=0.7,
            chromatic_match=0.8,
            scorer_result=scorer_result,
            theory_breakdown=theory_breakdown,
        )

        expected = 0.3 * 0.7 + 0.7 * 0.8
        assert composite == pytest.approx(expected)
        assert "composite" in breakdown
        assert "theory" in breakdown
        assert "chromatic" in breakdown

    def test_custom_weights(self):
        from app.generators.midi.melody_pipeline import melody_composite_score

        scorer_result = {
            "temporal": {"past": 0.5, "present": 0.25, "future": 0.25},
            "spatial": {"thing": 0.5, "place": 0.25, "person": 0.25},
            "ontological": {"imagined": 0.25, "forgotten": 0.25, "known": 0.5},
            "confidence": 0.5,
        }

        composite, _ = melody_composite_score(
            theory=1.0,
            chromatic_match=0.0,
            scorer_result=scorer_result,
            theory_breakdown={"singability": 1.0},
            theory_weight=0.5,
            chromatic_weight=0.5,
        )
        assert composite == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# 4. Review YAML generation
# ---------------------------------------------------------------------------


class TestReviewYaml:

    def test_generate_melody_review_yaml(self):
        from app.generators.midi.melody_pipeline import generate_melody_review_yaml

        sections = [
            {
                "label": "verse",
                "label_display": "Verse",
                "_section_key": "verse",
                "chord_id": "c01",
            }
        ]
        ranked = {
            "verse": [
                {
                    "id": "melody_verse_01",
                    "contour": "stepwise",
                    "pattern_name": "stepwise_ascend_med",
                    "energy": "medium",
                    "breakdown": {
                        "composite": 0.75,
                        "theory": {"singability": 0.8},
                        "chromatic": {"match": 0.7},
                    },
                }
            ]
        }

        review = generate_melody_review_yaml(
            "/test/prod",
            sections,
            ranked,
            seed=42,
            scoring_weights={"theory": 0.3, "chromatic": 0.7},
            song_info={"bpm": 120, "time_sig": (4, 4), "color_name": "Red"},
            singer_name="Gabriel",
        )

        assert review["pipeline"] == "melody-generation"
        assert review["singer"] == "Gabriel"
        assert len(review["candidates"]) == 1
        assert review["candidates"][0]["label"] is None
        assert review["candidates"][0]["status"] == "pending"
        assert review["candidates"][0]["singer"] == "Gabriel"


# ---------------------------------------------------------------------------
# 5. Integration (mock ChromaticScorer)
# ---------------------------------------------------------------------------


class TestMelodyPipelineIntegration:

    def test_full_pipeline_with_mock_scorer(self, production_dir):
        """Run the full pipeline with a mocked ChromaticScorer."""
        mock_scorer = MagicMock()
        mock_scorer.prepare_concept.return_value = MagicMock(shape=(768,))

        def mock_score_batch(candidates, concept_emb=None):
            results = []
            for c in candidates:
                results.append(
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
                )
            return results

        mock_scorer.score_batch = mock_score_batch

        with patch(
            "training.chromatic_scorer.ChromaticScorer", return_value=mock_scorer
        ):
            from app.generators.midi.melody_pipeline import run_melody_pipeline

            result = run_melody_pipeline(
                production_dir=str(production_dir),
                singer_name="gabriel",
                seed=42,
                top_k=3,
            )

        assert isinstance(result, dict)
        # Check MIDI files were written
        candidates_dir = production_dir / "melody" / "candidates"
        assert candidates_dir.exists()
        midi_files = list(candidates_dir.glob("*.mid"))
        assert len(midi_files) > 0

        # Check review.yml was written
        review_path = production_dir / "melody" / "review.yml"
        assert review_path.exists()

        with open(review_path) as f:
            review = yaml.safe_load(f)
        assert review["pipeline"] == "melody-generation"
        assert review["singer"] == "Gabriel"
        assert len(review["candidates"]) > 0
