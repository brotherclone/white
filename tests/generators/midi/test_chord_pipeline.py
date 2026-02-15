"""Tests for the chord generation pipeline."""

import shutil
from pathlib import Path

import pytest
import yaml


# ---------------------------------------------------------------------------
# 3.1: Chromatic target mapping
# ---------------------------------------------------------------------------


class TestChromaticTargetMapping:

    def test_red_target(self):
        from app.generators.midi.chord_pipeline import get_chromatic_target

        target = get_chromatic_target("Red")
        assert target["temporal"] == [0.8, 0.1, 0.1]  # Past
        assert target["spatial"] == [0.8, 0.1, 0.1]  # Thing
        assert target["ontological"] == [0.1, 0.1, 0.8]  # Known

    def test_green_target(self):
        from app.generators.midi.chord_pipeline import get_chromatic_target

        target = get_chromatic_target("Green")
        assert target["temporal"] == [0.1, 0.8, 0.1]  # Present
        assert target["spatial"] == [0.1, 0.8, 0.1]  # Place
        assert target["ontological"] == [0.1, 0.8, 0.1]  # Forgotten

    def test_indigo_target(self):
        from app.generators.midi.chord_pipeline import get_chromatic_target

        target = get_chromatic_target("Indigo")
        assert target["temporal"] == [0.1, 0.1, 0.8]  # Future
        assert target["ontological"] == [0.1, 0.8, 0.1]  # Forgotten

    def test_black_uniform(self):
        from app.generators.midi.chord_pipeline import get_chromatic_target

        target = get_chromatic_target("Black")
        for dim in ["temporal", "spatial", "ontological"]:
            assert abs(sum(target[dim]) - 1.0) < 0.01

    def test_white_uniform(self):
        from app.generators.midi.chord_pipeline import get_chromatic_target

        target = get_chromatic_target("White")
        for dim in ["temporal", "spatial", "ontological"]:
            assert abs(target[dim][0] - target[dim][1]) < 0.01

    def test_unknown_color_falls_back(self):
        from app.generators.midi.chord_pipeline import get_chromatic_target

        target = get_chromatic_target("Magenta")
        # Should return uniform (White) as fallback
        assert abs(target["temporal"][0] - 1 / 3) < 0.01

    def test_case_insensitive(self):
        from app.generators.midi.chord_pipeline import get_chromatic_target

        t1 = get_chromatic_target("red")
        t2 = get_chromatic_target("RED")
        t3 = get_chromatic_target("Red")
        assert t1 == t2 == t3

    def test_all_targets_sum_to_one(self):
        from app.generators.midi.chord_pipeline import CHROMATIC_TARGETS

        for color, target in CHROMATIC_TARGETS.items():
            for dim in ["temporal", "spatial", "ontological"]:
                assert (
                    abs(sum(target[dim]) - 1.0) < 0.01
                ), f"{color} {dim} doesn't sum to 1"


# ---------------------------------------------------------------------------
# 3.1 continued: Key parsing
# ---------------------------------------------------------------------------


class TestKeyParsing:

    def test_standard_key(self):
        from app.generators.midi.chord_pipeline import parse_key_string

        assert parse_key_string("C major") == ("C", "Major")
        assert parse_key_string("F# minor") == ("F#", "Minor")
        assert parse_key_string("Bb major") == ("Bb", "Major")

    def test_unicode_accidentals(self):
        from app.generators.midi.chord_pipeline import parse_key_string

        assert parse_key_string("A♭ major") == ("Ab", "Major")
        assert parse_key_string("B♭ major") == ("Bb", "Major")

    def test_defaults(self):
        from app.generators.midi.chord_pipeline import parse_key_string

        assert parse_key_string("C") == ("C", "Major")


# ---------------------------------------------------------------------------
# 3.2: Composite scoring
# ---------------------------------------------------------------------------


class TestCompositeScoring:

    def test_basic_composite(self):
        from app.generators.midi.chord_pipeline import composite_score

        comp, breakdown = composite_score(
            theory_score=0.8,
            theory_breakdown={
                "melody": 0.9,
                "voice_leading": 0.8,
                "variety": 0.7,
                "graph_probability": 0.6,
            },
            chromatic_match=0.9,
            scorer_result={
                "temporal": {"past": 0.8, "present": 0.1, "future": 0.1},
                "spatial": {"thing": 0.7, "place": 0.2, "person": 0.1},
                "ontological": {"imagined": 0.1, "forgotten": 0.1, "known": 0.8},
                "confidence": 0.9,
            },
            theory_weight=0.3,
            chromatic_weight=0.7,
        )
        expected = 0.3 * 0.8 + 0.7 * 0.9
        assert abs(comp - expected) < 0.001

    def test_weights_affect_ranking(self):
        from app.generators.midi.chord_pipeline import composite_score

        scorer_result = {
            "temporal": {"past": 0.5, "present": 0.3, "future": 0.2},
            "spatial": {"thing": 0.5, "place": 0.3, "person": 0.2},
            "ontological": {"imagined": 0.5, "forgotten": 0.3, "known": 0.2},
            "confidence": 0.5,
        }
        breakdown_stub = {
            "melody": 0.5,
            "voice_leading": 0.5,
            "variety": 0.5,
            "graph_probability": 0.5,
        }

        # High theory weight
        comp_theory, _ = composite_score(
            0.9, breakdown_stub, 0.3, scorer_result, 0.8, 0.2
        )
        # High chromatic weight
        comp_chroma, _ = composite_score(
            0.9, breakdown_stub, 0.3, scorer_result, 0.2, 0.8
        )

        # Theory-weighted should be higher when theory > chromatic
        assert comp_theory > comp_chroma

    def test_chromatic_match_calculation(self):
        from app.generators.midi.chord_pipeline import compute_chromatic_match

        # Perfect match for Red
        target = {
            "temporal": [0.8, 0.1, 0.1],
            "spatial": [0.8, 0.1, 0.1],
            "ontological": [0.1, 0.1, 0.8],
        }
        scorer_result = {
            "temporal": {"past": 0.8, "present": 0.1, "future": 0.1},
            "spatial": {"thing": 0.8, "place": 0.1, "person": 0.1},
            "ontological": {"imagined": 0.1, "forgotten": 0.1, "known": 0.8},
            "confidence": 1.0,
        }
        match = compute_chromatic_match(scorer_result, target)
        # With perfect alignment and confidence=1.0, should be close to max
        assert match > 0.6

    def test_chromatic_match_poor(self):
        from app.generators.midi.chord_pipeline import compute_chromatic_match

        # Mismatch: target is Red (Past/Thing/Known) but prediction is opposite
        target = {
            "temporal": [0.8, 0.1, 0.1],
            "spatial": [0.8, 0.1, 0.1],
            "ontological": [0.1, 0.1, 0.8],
        }
        scorer_result = {
            "temporal": {"past": 0.1, "present": 0.1, "future": 0.8},
            "spatial": {"thing": 0.1, "place": 0.1, "person": 0.8},
            "ontological": {"imagined": 0.8, "forgotten": 0.1, "known": 0.1},
            "confidence": 1.0,
        }
        match = compute_chromatic_match(scorer_result, target)
        # Poor alignment should score low
        assert match < 0.3


# ---------------------------------------------------------------------------
# 3.3: Integration test (requires chord data + ONNX — skip if unavailable)
# ---------------------------------------------------------------------------

_chord_data = Path(
    "/Volumes/LucidNonsense/White/app/generators/midi/prototype/data/chords.parquet"
)
_has_chord_data = _chord_data.exists()

_onnx_path = Path("/Volumes/LucidNonsense/White/training/data/fusion_model.onnx")
try:
    import onnxruntime  # noqa: F401

    _has_ort = True
except ImportError:
    _has_ort = False
_has_onnx = _onnx_path.exists() and _has_ort

_thread_dir = Path(
    "/Volumes/LucidNonsense/White/shrinkwrapped/white-the-breathing-machine-learns-to-sing"
)
_has_thread = _thread_dir.exists()


@pytest.mark.skipif(
    not (_has_chord_data and _has_onnx and _has_thread),
    reason="Requires chord data, ONNX model, onnxruntime, and shrinkwrapped thread",
)
class TestChordPipelineIntegration:

    def test_full_pipeline(self, tmp_path):
        """Run the full pipeline on a real song proposal with output to temp dir."""
        from app.generators.midi.chord_pipeline import (
            run_chord_pipeline,
        )

        # Use a real thread but output to temp
        # First, create a temp copy of the thread dir structure
        thread_copy = tmp_path / "test-thread"
        shutil.copytree(
            _thread_dir,
            thread_copy,
            ignore=shutil.ignore_patterns("production"),
        )

        # Find first song proposal
        songs = list((thread_copy / "yml").glob("song_proposal_*.yml"))
        assert len(songs) > 0, "No song proposals found"
        song_file = songs[0].name

        results = run_chord_pipeline(
            thread_dir=str(thread_copy),
            song_filename=song_file,
            seed=42,
            num_candidates=20,  # Small for speed
            top_k=3,
            progression_length=4,
        )

        assert len(results) == 3

        # Check MIDI files exist
        from app.generators.midi.chord_pipeline import song_slug

        slug = song_slug(song_file)
        chords_dir = thread_copy / "production" / slug / "chords"
        assert (chords_dir / "review.yml").exists()
        assert (chords_dir / "candidates" / "chord_001.mid").exists()
        assert (chords_dir / "candidates" / "chord_002.mid").exists()
        assert (chords_dir / "candidates" / "chord_003.mid").exists()

        # Check review.yml structure
        with open(chords_dir / "review.yml") as f:
            review = yaml.safe_load(f)
        assert len(review["candidates"]) == 3
        assert review["candidates"][0]["rank"] == 1
        assert review["candidates"][0]["status"] == "pending"
        assert review["candidates"][0]["label"] is None
        assert "composite" in review["candidates"][0]["scores"]


# ---------------------------------------------------------------------------
# 3.4: Promotion command
# ---------------------------------------------------------------------------


class TestPromoteChords:

    def test_promote_approved(self, tmp_path):
        from app.generators.midi.promote_chords import promote_chords

        # Set up directory structure
        chords_dir = tmp_path / "chords"
        candidates_dir = chords_dir / "candidates"
        approved_dir = chords_dir / "approved"
        candidates_dir.mkdir(parents=True)

        # Write dummy MIDI files
        (candidates_dir / "chord_001.mid").write_bytes(b"MIDI1")
        (candidates_dir / "chord_002.mid").write_bytes(b"MIDI2")
        (candidates_dir / "chord_003.mid").write_bytes(b"MIDI3")

        # Write review YAML
        review = {
            "candidates": [
                {
                    "id": "chord_001",
                    "midi_file": "candidates/chord_001.mid",
                    "rank": 1,
                    "label": "verse-candidate",
                    "status": "approved",
                    "notes": "",
                },
                {
                    "id": "chord_002",
                    "midi_file": "candidates/chord_002.mid",
                    "rank": 2,
                    "label": "chorus-candidate",
                    "status": "approved",
                    "notes": "",
                },
                {
                    "id": "chord_003",
                    "midi_file": "candidates/chord_003.mid",
                    "rank": 3,
                    "label": None,
                    "status": "rejected",
                    "notes": "too repetitive",
                },
            ]
        }
        review_path = chords_dir / "review.yml"
        with open(review_path, "w") as f:
            yaml.dump(review, f)

        promote_chords(str(review_path))

        # Check promoted files
        assert (approved_dir / "verse_candidate.mid").exists()
        assert (approved_dir / "chorus_candidate.mid").exists()
        assert not (approved_dir / "unlabeled.mid").exists()  # rejected, not promoted

    def test_promote_multiple_same_label(self, tmp_path):
        from app.generators.midi.promote_chords import promote_chords

        chords_dir = tmp_path / "chords"
        candidates_dir = chords_dir / "candidates"
        candidates_dir.mkdir(parents=True)

        (candidates_dir / "chord_001.mid").write_bytes(b"MIDI1")
        (candidates_dir / "chord_002.mid").write_bytes(b"MIDI2")

        review = {
            "candidates": [
                {
                    "id": "chord_001",
                    "midi_file": "candidates/chord_001.mid",
                    "rank": 1,
                    "label": "verse-candidate",
                    "status": "approved",
                    "notes": "",
                },
                {
                    "id": "chord_002",
                    "midi_file": "candidates/chord_002.mid",
                    "rank": 2,
                    "label": "verse-candidate",
                    "status": "approved",
                    "notes": "alt",
                },
            ]
        }
        review_path = chords_dir / "review.yml"
        with open(review_path, "w") as f:
            yaml.dump(review, f)

        promote_chords(str(review_path))

        approved_dir = chords_dir / "approved"
        assert (approved_dir / "verse_candidate.mid").exists()
        assert (approved_dir / "verse_candidate_2.mid").exists()

    def test_promote_no_approved(self, tmp_path, capsys):
        from app.generators.midi.promote_chords import promote_chords

        chords_dir = tmp_path / "chords"
        chords_dir.mkdir()

        review = {
            "candidates": [
                {
                    "id": "chord_001",
                    "midi_file": "candidates/chord_001.mid",
                    "rank": 1,
                    "label": None,
                    "status": "pending",
                    "notes": "",
                },
            ]
        }
        review_path = chords_dir / "review.yml"
        with open(review_path, "w") as f:
            yaml.dump(review, f)

        promote_chords(str(review_path))

        captured = capsys.readouterr()
        assert "No approved candidates" in captured.out
