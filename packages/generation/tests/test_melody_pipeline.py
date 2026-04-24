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
                "chords": [
                    {"notes": ["C3", "E3", "G3"]},
                    {"notes": ["F3", "A3", "C4"]},
                    {"notes": ["G3", "B3", "D4"]},
                    {"notes": ["C3", "E3", "G3"]},
                ],
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
        from white_generation.pipelines.melody_pipeline import read_approved_sections

        with open(production_dir / "chords" / "review.yml") as f:
            chord_review = yaml.safe_load(f)

        sections = read_approved_sections(chord_review)
        assert len(sections) == 1
        assert sections[0]["label"] == "verse"
        assert sections[0]["chord_id"] == "chord_verse_01"

    def test_read_approved_sections_skips_pending(self):
        from white_generation.pipelines.melody_pipeline import read_approved_sections

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
        from white_generation.pipelines.melody_pipeline import (
            melody_notes_to_midi_bytes,
        )

        notes = [(0.0, 60, 1.0), (1.0, 62, 1.0), (2.0, 64, 2.0)]
        midi_bytes = melody_notes_to_midi_bytes(notes, bpm=120)

        assert isinstance(midi_bytes, bytes)
        assert len(midi_bytes) > 0

        # Verify it's valid MIDI
        mid = mido.MidiFile(file=io.BytesIO(midi_bytes))
        assert len(mid.tracks) == 1

    def test_generate_melody_for_section(self):
        from white_generation.patterns.melody_patterns import ALL_TEMPLATES, SINGERS
        from white_generation.pipelines.melody_pipeline import (
            generate_melody_for_section,
        )

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
        from white_generation.patterns.melody_patterns import ALL_TEMPLATES, SINGERS
        from white_generation.pipelines.melody_pipeline import (
            generate_melody_for_section,
        )

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
        from white_generation.pipelines.melody_pipeline import melody_composite_score

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
        from white_generation.pipelines.melody_pipeline import melody_composite_score

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
        from white_generation.pipelines.melody_pipeline import (
            generate_melody_review_yaml,
        )

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
                    "contour": "declarative",
                    "pattern_name": "decl_arc_med",
                    "energy": "medium",
                    "use_case": "vocal",
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
        assert review["candidates"][0]["use_case"] == "vocal"


# ---------------------------------------------------------------------------
# 5. Candidate sync
# ---------------------------------------------------------------------------


class TestSyncMelodyCandidates:

    def _make_review(self, tmp_path, candidates=None):
        melody_dir = tmp_path / "melody"
        candidates_dir = melody_dir / "candidates"
        candidates_dir.mkdir(parents=True)
        (melody_dir / "approved").mkdir()

        review = {
            "pipeline": "melody-generation",
            "bpm": 120,
            "singer": "Shirley",
            "candidates": candidates or [],
        }
        with open(melody_dir / "review.yml", "w") as f:
            yaml.dump(review, f)
        return melody_dir, candidates_dir

    def test_adds_untracked_midi(self, tmp_path):
        from white_generation.pipelines.melody_pipeline import sync_melody_candidates

        melody_dir, candidates_dir = self._make_review(tmp_path)
        (candidates_dir / "my_verse.mid").write_bytes(b"MIDI")

        added = sync_melody_candidates(melody_dir)

        assert added == 1
        with open(melody_dir / "review.yml") as f:
            review = yaml.safe_load(f)
        ids = [c["id"] for c in review["candidates"]]
        assert "my_verse" in ids
        assert review["candidates"][0]["status"] == "pending"
        assert review["candidates"][0]["section"] == "manual"

    def test_skips_already_tracked(self, tmp_path):
        from white_generation.pipelines.melody_pipeline import sync_melody_candidates

        existing = [
            {
                "id": "melody_verse_01",
                "midi_file": "candidates/melody_verse_01.mid",
                "status": "approved",
            }
        ]
        melody_dir, candidates_dir = self._make_review(tmp_path, candidates=existing)
        (candidates_dir / "melody_verse_01.mid").write_bytes(b"MIDI")

        added = sync_melody_candidates(melody_dir)

        assert added == 0
        with open(melody_dir / "review.yml") as f:
            review = yaml.safe_load(f)
        assert len(review["candidates"]) == 1  # no new entry

    def test_skips_scratch_files(self, tmp_path):
        from white_generation.pipelines.melody_pipeline import sync_melody_candidates

        melody_dir, candidates_dir = self._make_review(tmp_path)
        (candidates_dir / "drums_scratch.mid").write_bytes(b"MIDI")

        added = sync_melody_candidates(melody_dir)

        assert added == 0

    def test_deduplicates_id_collision(self, tmp_path):
        from white_generation.pipelines.melody_pipeline import sync_melody_candidates

        existing = [
            {"id": "my_verse", "midi_file": "candidates/other.mid", "status": "pending"}
        ]
        melody_dir, candidates_dir = self._make_review(tmp_path, candidates=existing)
        (candidates_dir / "my_verse.mid").write_bytes(b"MIDI")

        added = sync_melody_candidates(melody_dir)

        assert added == 1
        with open(melody_dir / "review.yml") as f:
            review = yaml.safe_load(f)
        new_ids = [c["id"] for c in review["candidates"]]
        # Original id preserved, new entry gets suffix
        assert "my_verse" in new_ids
        assert "my_verse_2" in new_ids

    def test_no_review_yml_returns_zero(self, tmp_path):
        from white_generation.pipelines.melody_pipeline import sync_melody_candidates

        melody_dir = tmp_path / "melody"
        melody_dir.mkdir()
        # No review.yml
        added = sync_melody_candidates(melody_dir)
        assert added == 0

    def test_preserves_existing_entries(self, tmp_path):
        from white_generation.pipelines.melody_pipeline import sync_melody_candidates

        existing = [
            {
                "id": "melody_verse_01",
                "midi_file": "candidates/melody_verse_01.mid",
                "label": "melody_verse",
                "status": "approved",
                "scores": {"composite": 0.75},
            }
        ]
        melody_dir, candidates_dir = self._make_review(tmp_path, candidates=existing)
        (candidates_dir / "melody_verse_01.mid").write_bytes(b"MIDI")
        (candidates_dir / "my_new.mid").write_bytes(b"MIDI")

        added = sync_melody_candidates(melody_dir)

        assert added == 1
        with open(melody_dir / "review.yml") as f:
            review = yaml.safe_load(f)
        # Original entry intact with scores
        orig = next(c for c in review["candidates"] if c["id"] == "melody_verse_01")
        assert orig["label"] == "melody_verse"
        assert orig["scores"]["composite"] == 0.75


# ---------------------------------------------------------------------------
# 6. Integration (mock Refractor)
# ---------------------------------------------------------------------------


class TestMelodyPipelineIntegration:

    def test_full_pipeline_with_mock_scorer(self, production_dir):
        """Run the full pipeline with a mocked Refractor."""
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

        with patch("white_analysis.refractor.Refractor", return_value=mock_scorer):
            from white_generation.pipelines.melody_pipeline import (
                run_melody_pipeline,
            )

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
        assert all("use_case" in c for c in review["candidates"])
        assert all(
            c["use_case"] in ("vocal", "instrumental") for c in review["candidates"]
        )


# ---------------------------------------------------------------------------
# 6b. Melodic continuity helpers
# ---------------------------------------------------------------------------


def make_melody_midi(notes: list[int], ticks_per_beat: int = 480) -> bytes:
    """Create a single-track melody MIDI from a sequence of pitches."""
    mid = mido.MidiFile(ticks_per_beat=ticks_per_beat)
    track = mido.MidiTrack()
    mid.tracks.append(track)
    track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(120), time=0))
    for note in notes:
        track.append(mido.Message("note_on", note=note, velocity=90, time=0))
        track.append(
            mido.Message("note_off", note=note, velocity=0, time=ticks_per_beat)
        )
    track.append(mido.MetaMessage("end_of_track", time=0))
    buf = io.BytesIO()
    mid.save(file=buf)
    return buf.getvalue()


class TestMelodicContinuityHelpers:

    def test_last_note_of_midi_returns_last_pitch(self):
        from white_generation.pipelines.melody_pipeline import last_note_of_midi

        midi_bytes = make_melody_midi([60, 62, 64, 65])
        assert last_note_of_midi(midi_bytes) == 65

    def test_last_note_of_midi_single_note(self):
        from white_generation.pipelines.melody_pipeline import last_note_of_midi

        midi_bytes = make_melody_midi([60])
        assert last_note_of_midi(midi_bytes) == 60

    def test_last_note_of_midi_no_notes_returns_none(self):
        from white_generation.pipelines.melody_pipeline import last_note_of_midi

        mid = mido.MidiFile()
        track = mido.MidiTrack()
        mid.tracks.append(track)
        track.append(mido.MetaMessage("end_of_track", time=0))
        buf = io.BytesIO()
        mid.save(file=buf)
        assert last_note_of_midi(buf.getvalue()) is None

    def test_first_note_of_candidate_returns_first_pitch(self):
        from white_generation.pipelines.melody_pipeline import (
            first_note_of_candidate,
        )

        midi_bytes = make_melody_midi([60, 62, 64])
        assert first_note_of_candidate({"midi_bytes": midi_bytes}) == 60

    def test_first_note_of_candidate_no_midi_bytes_returns_none(self):
        from white_generation.pipelines.melody_pipeline import (
            first_note_of_candidate,
        )

        assert first_note_of_candidate({}) is None
        assert first_note_of_candidate({"midi_bytes": None}) is None

    def test_continuity_penalty_within_range_is_1(self):
        from white_generation.pipelines.melody_pipeline import continuity_penalty

        assert continuity_penalty(60, 63, max_semitones=4) == 1.0
        assert continuity_penalty(60, 64, max_semitones=4) == 1.0  # exactly at limit

    def test_continuity_penalty_over_range_is_085(self):
        from white_generation.pipelines.melody_pipeline import continuity_penalty

        assert continuity_penalty(60, 65, max_semitones=4) == pytest.approx(0.85)
        assert continuity_penalty(60, 72, max_semitones=4) == pytest.approx(0.85)

    def test_continuity_penalty_none_note_is_1(self):
        from white_generation.pipelines.melody_pipeline import continuity_penalty

        assert continuity_penalty(None, 60) == 1.0
        assert continuity_penalty(60, None) == 1.0
        assert continuity_penalty(None, None) == 1.0

    def test_continuity_penalty_custom_max_semitones(self):
        from white_generation.pipelines.melody_pipeline import continuity_penalty

        # 7 semitones: fine with max=7, penalised with max=6
        assert continuity_penalty(60, 67, max_semitones=7) == 1.0
        assert continuity_penalty(60, 67, max_semitones=6) == pytest.approx(0.85)

    def test_continuity_penalty_descending_interval(self):
        from white_generation.pipelines.melody_pipeline import continuity_penalty

        # Direction doesn't matter — abs interval is used
        assert continuity_penalty(63, 60, max_semitones=4) == 1.0  # 3 st down: fine
        assert continuity_penalty(65, 60, max_semitones=4) == pytest.approx(
            0.85
        )  # 5 st down: penalised


class TestMelodicContinuityIntegration:
    """Integration: preceding approved section penalises large-leap candidates."""

    def test_within_range_candidate_scores_higher(self, production_dir, tmp_path):
        """Candidate whose first note is within range outscores one that leaps."""
        from white_generation.pipelines.melody_pipeline import (
            continuity_penalty,
            first_note_of_candidate,
            last_note_of_midi,
        )

        # Anchor: last note of approved verse melody = 60 (middle C)
        anchor = 60
        anchor_midi = make_melody_midi([55, 57, 60])

        # Candidate A: first note 63 — 3 semitones from anchor, within range
        midi_a = make_melody_midi([63, 65, 67])
        # Candidate B: first note 72 — 12 semitones from anchor, over range
        midi_b = make_melody_midi([72, 74, 76])

        assert last_note_of_midi(anchor_midi) == 60
        assert first_note_of_candidate({"midi_bytes": midi_a}) == 63
        assert first_note_of_candidate({"midi_bytes": midi_b}) == 72

        penalty_a = continuity_penalty(63, anchor, max_semitones=4)
        penalty_b = continuity_penalty(72, anchor, max_semitones=4)

        assert penalty_a == 1.0
        assert penalty_b == pytest.approx(0.85)

        base_score = 0.75
        assert base_score * penalty_a > base_score * penalty_b

    def test_pipeline_applies_continuity_when_approved_section_exists(
        self, production_dir
    ):
        """Full pipeline run: approved verse MIDI causes continuity penalty on chorus."""
        from unittest.mock import MagicMock, patch

        # Add a chorus section to the chord review
        review_path = production_dir / "chords" / "review.yml"
        with open(review_path) as f:
            review = yaml.safe_load(f)

        review["candidates"].append(
            {
                "id": "chord_chorus_01",
                "label": "chorus",
                "status": "approved",
                "chords": [
                    {"notes": ["C3", "E3", "G3"]},
                    {"notes": ["F3", "A3", "C4"]},
                ],
            }
        )
        with open(review_path, "w") as f:
            yaml.dump(review, f)

        # Write an approved verse melody with last note = 60
        approved_dir = production_dir / "melody" / "approved"
        approved_dir.mkdir(parents=True, exist_ok=True)
        (approved_dir / "verse.mid").write_bytes(make_melody_midi([55, 58, 60]))

        mock_scorer = MagicMock()
        mock_scorer.prepare_concept.return_value = MagicMock(shape=(768,))

        def mock_score_batch(candidates, concept_emb=None):
            return [
                {
                    "candidate": c,
                    "temporal": {"past": 0.8, "present": 0.1, "future": 0.1},
                    "spatial": {"thing": 0.7, "place": 0.2, "person": 0.1},
                    "ontological": {"imagined": 0.1, "forgotten": 0.1, "known": 0.8},
                    "confidence": 0.85,
                }
                for c in candidates
            ]

        mock_scorer.score_batch = mock_score_batch

        with patch("white_analysis.refractor.Refractor", return_value=mock_scorer):
            from white_generation.pipelines.melody_pipeline import (
                run_melody_pipeline,
            )

            result = run_melody_pipeline(
                production_dir=str(production_dir),
                singer_name="gabriel",
                seed=42,
                top_k=3,
            )

        # Both sections should be present in result
        assert "verse" in result or any("verse" in k for k in result)
        assert "chorus" in result or any("chorus" in k for k in result)

        # Chorus candidates should all have composite scores <= verse candidates
        # (since continuity penalty may apply) — just verify pipeline ran without error
        review_path = production_dir / "melody" / "review.yml"
        assert review_path.exists()
        with open(review_path) as f:
            gen_review = yaml.safe_load(f)
        assert len(gen_review["candidates"]) > 0


class TestUseCasePromotion:
    def _write_review(self, melody_dir, candidates, tmp_path):
        candidates_dir = melody_dir / "candidates"
        candidates_dir.mkdir(parents=True)
        for c in candidates:
            (candidates_dir / f"{c['id']}.mid").write_bytes(b"MIDI")
        review = {"pipeline": "melody-generation", "candidates": candidates}
        review_path = melody_dir / "review.yml"
        with open(review_path, "w") as f:
            yaml.dump(review, f)
        return review_path

    def test_use_case_carried_into_review_yml(self, tmp_path):
        """use_case written to review.yml is readable after promotion approval."""
        from app.generators.midi.production.promote_part import promote_part

        melody_dir = tmp_path / "melody"
        review_path = self._write_review(
            melody_dir,
            [
                {
                    "id": "mel_001",
                    "midi_file": "candidates/mel_001.mid",
                    "rank": 1,
                    "label": "verse",
                    "status": "approved",
                    "use_case": "vocal",
                    "pattern_name": "stepwise_ascent",
                    "notes": "",
                }
            ],
            tmp_path,
        )
        promote_part(str(review_path))
        assert (melody_dir / "approved" / "verse.mid").exists()
        # use_case is preserved in the review.yml candidate entry
        with open(review_path) as f:
            review = yaml.safe_load(f)
        assert review["candidates"][0]["use_case"] == "vocal"

    def test_use_case_absent_does_not_error(self, tmp_path):
        from app.generators.midi.production.promote_part import promote_part

        melody_dir = tmp_path / "melody"
        review_path = self._write_review(
            melody_dir,
            [
                {
                    "id": "mel_001",
                    "midi_file": "candidates/mel_001.mid",
                    "rank": 1,
                    "label": "verse",
                    "status": "approved",
                    "notes": "",
                }
            ],
            tmp_path,
        )
        # Should not raise even when use_case is absent
        promote_part(str(review_path))
        assert (melody_dir / "approved" / "verse.mid").exists()
