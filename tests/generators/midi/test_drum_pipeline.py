"""Tests for the drum pattern generation pipeline."""

import io

import mido
import pytest
import yaml


# ---------------------------------------------------------------------------
# 1. Template validation
# ---------------------------------------------------------------------------


class TestDrumPatternTemplates:

    def test_all_templates_have_required_fields(self):
        from app.generators.midi.drum_patterns import ALL_TEMPLATES

        for t in ALL_TEMPLATES:
            assert t.name, "Template missing name"
            assert t.genre_family, f"{t.name}: missing genre_family"
            assert t.energy in (
                "low",
                "medium",
                "high",
            ), f"{t.name}: invalid energy '{t.energy}'"
            assert len(t.time_sig) == 2, f"{t.name}: invalid time_sig"
            assert t.description, f"{t.name}: missing description"
            assert len(t.voices) > 0, f"{t.name}: no voices defined"

    def test_all_templates_have_valid_velocity_levels(self):
        from app.generators.midi.drum_patterns import ALL_TEMPLATES, VELOCITY

        valid_levels = set(VELOCITY.keys())
        for t in ALL_TEMPLATES:
            for voice_name, hits in t.voices.items():
                for beat_pos, vel_level in hits:
                    assert (
                        vel_level in valid_levels
                    ), f"{t.name}/{voice_name}: invalid velocity '{vel_level}'"

    def test_all_templates_beat_positions_within_bar(self):
        from app.generators.midi.drum_patterns import ALL_TEMPLATES

        for t in ALL_TEMPLATES:
            bar_length = t.bar_length_beats()
            for voice_name, hits in t.voices.items():
                for beat_pos, _ in hits:
                    assert (
                        0 <= beat_pos < bar_length
                    ), f"{t.name}/{voice_name}: beat {beat_pos} outside bar length {bar_length}"

    def test_all_templates_use_valid_gm_voices(self):
        from app.generators.midi.drum_patterns import ALL_TEMPLATES, GM_PERCUSSION

        valid_voices = set(GM_PERCUSSION.keys())
        for t in ALL_TEMPLATES:
            for voice_name in t.voices:
                assert (
                    voice_name in valid_voices
                ), f"{t.name}: unknown voice '{voice_name}'"

    def test_unique_template_names(self):
        from app.generators.midi.drum_patterns import ALL_TEMPLATES

        names = [t.name for t in ALL_TEMPLATES]
        assert len(names) == len(
            set(names)
        ), f"Duplicate template names: {[n for n in names if names.count(n) > 1]}"

    def test_4_4_templates_exist_for_core_families(self):
        from app.generators.midi.drum_patterns import ALL_TEMPLATES

        families_4_4 = {t.genre_family for t in ALL_TEMPLATES if t.time_sig == (4, 4)}
        for expected in ["ambient", "electronic", "krautrock", "rock"]:
            assert expected in families_4_4, f"No 4/4 templates for {expected}"

    def test_each_family_has_low_medium_high(self):
        from app.generators.midi.drum_patterns import ALL_TEMPLATES

        # Check 4/4 core families
        for family in ["ambient", "electronic", "krautrock", "rock"]:
            energies = {
                t.energy
                for t in ALL_TEMPLATES
                if t.time_sig == (4, 4) and t.genre_family == family
            }
            for level in ["low", "medium", "high"]:
                assert level in energies, f"{family} 4/4 missing energy={level}"

    def test_7_8_templates_exist(self):
        from app.generators.midi.drum_patterns import ALL_TEMPLATES

        templates_7_8 = [t for t in ALL_TEMPLATES if t.time_sig == (7, 8)]
        assert len(templates_7_8) >= 4, f"Only {len(templates_7_8)} 7/8 templates"

    def test_motorik_pattern_has_kick_every_beat(self):
        from app.generators.midi.drum_patterns import ALL_TEMPLATES

        motorik = [t for t in ALL_TEMPLATES if t.name == "motorik"]
        assert len(motorik) == 1
        pattern = motorik[0]
        kick_beats = [pos for pos, _ in pattern.voices["kick"]]
        assert kick_beats == [0, 1, 2, 3], "Motorik should have kick on every beat"
        assert "snare" in pattern.voices, "Motorik should have snare"
        snare_beats = [pos for pos, _ in pattern.voices["snare"]]
        assert 2 in snare_beats, "Motorik snare should be on beat 3"

    def test_bar_length_beats(self):
        from app.generators.midi.drum_patterns import DrumPattern

        p_4_4 = DrumPattern(
            name="t", genre_family="t", energy="low", time_sig=(4, 4), description="t"
        )
        assert p_4_4.bar_length_beats() == 4.0

        p_7_8 = DrumPattern(
            name="t", genre_family="t", energy="low", time_sig=(7, 8), description="t"
        )
        assert p_7_8.bar_length_beats() == 3.5

        p_3_4 = DrumPattern(
            name="t", genre_family="t", energy="low", time_sig=(3, 4), description="t"
        )
        assert p_3_4.bar_length_beats() == 3.0

    def test_fallback_pattern(self):
        from app.generators.midi.drum_patterns import make_fallback_pattern

        fb = make_fallback_pattern((5, 4))
        assert fb.time_sig == (5, 4)
        assert fb.genre_family == "fallback"
        assert "kick" in fb.voices
        assert len(fb.voices["kick"]) == 1
        assert fb.voices["kick"][0][0] == 0


# ---------------------------------------------------------------------------
# 2. Genre family mapping
# ---------------------------------------------------------------------------


class TestGenreMapping:

    def test_ambient_match(self):
        from app.generators.midi.drum_patterns import map_genres_to_families

        result = map_genres_to_families(["ambient", "drone"])
        assert "ambient" in result

    def test_glitch_ambient_matches_both(self):
        from app.generators.midi.drum_patterns import map_genres_to_families

        result = map_genres_to_families(["glitch ambient"])
        assert "ambient" in result
        assert "electronic" in result  # "glitch" is an electronic keyword

    def test_krautrock_match(self):
        from app.generators.midi.drum_patterns import map_genres_to_families

        result = map_genres_to_families(["krautrock", "motorik"])
        assert "krautrock" in result

    def test_post_classical_matches_classical(self):
        from app.generators.midi.drum_patterns import map_genres_to_families

        result = map_genres_to_families(["post-classical"])
        assert "classical" in result

    def test_no_match_returns_empty(self):
        from app.generators.midi.drum_patterns import map_genres_to_families

        result = map_genres_to_families(["reggaeton", "cumbia"])
        assert result == []

    def test_empty_tags_returns_empty(self):
        from app.generators.midi.drum_patterns import map_genres_to_families

        result = map_genres_to_families([])
        assert result == []

    def test_case_insensitive(self):
        from app.generators.midi.drum_patterns import map_genres_to_families

        result = map_genres_to_families(["AMBIENT", "Electronic"])
        assert "ambient" in result
        assert "electronic" in result


# ---------------------------------------------------------------------------
# 3. Energy scoring
# ---------------------------------------------------------------------------


class TestEnergyScoring:

    def test_exact_match(self):
        from app.generators.midi.drum_patterns import energy_appropriateness

        assert energy_appropriateness("medium", "medium") == 1.0
        assert energy_appropriateness("low", "low") == 1.0
        assert energy_appropriateness("high", "high") == 1.0

    def test_one_level_away(self):
        from app.generators.midi.drum_patterns import energy_appropriateness

        assert energy_appropriateness("low", "medium") == 0.5
        assert energy_appropriateness("medium", "high") == 0.5
        assert energy_appropriateness("high", "medium") == 0.5

    def test_two_levels_away(self):
        from app.generators.midi.drum_patterns import energy_appropriateness

        assert energy_appropriateness("low", "high") == 0.0
        assert energy_appropriateness("high", "low") == 0.0


# ---------------------------------------------------------------------------
# 4. Template selection
# ---------------------------------------------------------------------------


class TestTemplateSelection:

    def test_filters_by_time_sig(self):
        from app.generators.midi.drum_patterns import ALL_TEMPLATES, select_templates

        result = select_templates(ALL_TEMPLATES, (7, 8), ["ambient"], "medium")
        for t in result:
            assert t.time_sig == (7, 8)

    def test_filters_by_genre_family(self):
        from app.generators.midi.drum_patterns import ALL_TEMPLATES, select_templates

        result = select_templates(ALL_TEMPLATES, (4, 4), ["krautrock"], "medium")
        for t in result:
            assert t.genre_family == "krautrock"

    def test_includes_adjacent_energy(self):
        from app.generators.midi.drum_patterns import ALL_TEMPLATES, select_templates

        result = select_templates(ALL_TEMPLATES, (4, 4), ["rock"], "medium")
        energies = {t.energy for t in result}
        assert "medium" in energies
        # Should include at least one adjacent
        assert "low" in energies or "high" in energies

    def test_exact_energy_first(self):
        from app.generators.midi.drum_patterns import ALL_TEMPLATES, select_templates

        result = select_templates(ALL_TEMPLATES, (4, 4), ["rock"], "medium")
        if len(result) >= 2:
            # First result should be exact match or at least not worse than second
            from app.generators.midi.drum_patterns import energy_distance

            d0 = energy_distance(result[0].energy, "medium")
            d1 = energy_distance(result[1].energy, "medium")
            assert d0 <= d1

    def test_no_match_returns_empty(self):
        from app.generators.midi.drum_patterns import ALL_TEMPLATES, select_templates

        result = select_templates(ALL_TEMPLATES, (11, 8), ["ambient"], "medium")
        assert result == []

    def test_multiple_families(self):
        from app.generators.midi.drum_patterns import ALL_TEMPLATES, select_templates

        result = select_templates(
            ALL_TEMPLATES, (4, 4), ["ambient", "electronic"], "medium"
        )
        families = {t.genre_family for t in result}
        assert "ambient" in families
        assert "electronic" in families


# ---------------------------------------------------------------------------
# 5. Drum MIDI generation
# ---------------------------------------------------------------------------


class TestDrumMidiGeneration:

    def test_generates_valid_midi(self):
        from app.generators.midi.drum_patterns import ALL_TEMPLATES
        from app.generators.midi.drum_pipeline import drum_pattern_to_midi_bytes

        pattern = ALL_TEMPLATES[0]
        midi_bytes = drum_pattern_to_midi_bytes(pattern, bpm=120, bar_count=2)
        assert len(midi_bytes) > 0

        # Should be parseable
        mid = mido.MidiFile(file=io.BytesIO(midi_bytes))
        assert len(mid.tracks) >= 1

    def test_uses_channel_10(self):
        from app.generators.midi.drum_patterns import ALL_TEMPLATES
        from app.generators.midi.drum_pipeline import (
            DRUM_CHANNEL,
            drum_pattern_to_midi_bytes,
        )

        pattern = ALL_TEMPLATES[0]
        midi_bytes = drum_pattern_to_midi_bytes(pattern, bpm=120, bar_count=1)
        mid = mido.MidiFile(file=io.BytesIO(midi_bytes))

        for msg in mid.tracks[0]:
            if msg.type in ("note_on", "note_off"):
                assert (
                    msg.channel == DRUM_CHANNEL
                ), f"Expected channel {DRUM_CHANNEL}, got {msg.channel}"

    def test_uses_correct_gm_notes(self):
        from app.generators.midi.drum_patterns import GM_PERCUSSION, DrumPattern
        from app.generators.midi.drum_pipeline import drum_pattern_to_midi_bytes

        pattern = DrumPattern(
            name="test",
            genre_family="test",
            energy="medium",
            time_sig=(4, 4),
            description="test",
            voices={
                "kick": [(0, "normal")],
                "snare": [(1, "accent")],
            },
        )
        midi_bytes = drum_pattern_to_midi_bytes(pattern, bpm=120, bar_count=1)
        mid = mido.MidiFile(file=io.BytesIO(midi_bytes))

        note_ons = [
            msg.note
            for msg in mid.tracks[0]
            if msg.type == "note_on" and msg.velocity > 0
        ]
        assert GM_PERCUSSION["kick"] in note_ons
        assert GM_PERCUSSION["snare"] in note_ons

    def test_velocity_values(self):
        from app.generators.midi.drum_patterns import VELOCITY, DrumPattern
        from app.generators.midi.drum_pipeline import drum_pattern_to_midi_bytes

        pattern = DrumPattern(
            name="test",
            genre_family="test",
            energy="medium",
            time_sig=(4, 4),
            description="test",
            voices={
                "kick": [(0, "accent"), (1, "normal"), (2, "ghost")],
            },
        )
        midi_bytes = drum_pattern_to_midi_bytes(pattern, bpm=120, bar_count=1)
        mid = mido.MidiFile(file=io.BytesIO(midi_bytes))

        velocities = [
            msg.velocity
            for msg in mid.tracks[0]
            if msg.type == "note_on" and msg.velocity > 0
        ]
        assert VELOCITY["accent"] in velocities
        assert VELOCITY["normal"] in velocities
        assert VELOCITY["ghost"] in velocities

    def test_bar_repetition(self):
        from app.generators.midi.drum_patterns import DrumPattern
        from app.generators.midi.drum_pipeline import drum_pattern_to_midi_bytes

        pattern = DrumPattern(
            name="test",
            genre_family="test",
            energy="medium",
            time_sig=(4, 4),
            description="test",
            voices={
                "kick": [(0, "normal")],
            },
        )
        midi_1bar = drum_pattern_to_midi_bytes(pattern, bpm=120, bar_count=1)
        midi_4bar = drum_pattern_to_midi_bytes(pattern, bpm=120, bar_count=4)

        mid_1 = mido.MidiFile(file=io.BytesIO(midi_1bar))
        mid_4 = mido.MidiFile(file=io.BytesIO(midi_4bar))

        notes_1 = [
            msg for msg in mid_1.tracks[0] if msg.type == "note_on" and msg.velocity > 0
        ]
        notes_4 = [
            msg for msg in mid_4.tracks[0] if msg.type == "note_on" and msg.velocity > 0
        ]

        assert len(notes_4) == 4 * len(notes_1)

    def test_correct_tempo(self):
        from app.generators.midi.drum_patterns import ALL_TEMPLATES
        from app.generators.midi.drum_pipeline import drum_pattern_to_midi_bytes

        pattern = ALL_TEMPLATES[0]
        midi_bytes = drum_pattern_to_midi_bytes(pattern, bpm=84, bar_count=1)
        mid = mido.MidiFile(file=io.BytesIO(midi_bytes))

        tempo_msgs = [msg for msg in mid.tracks[0] if msg.type == "set_tempo"]
        assert len(tempo_msgs) == 1
        assert mido.tempo2bpm(tempo_msgs[0].tempo) == pytest.approx(84, abs=0.1)


# ---------------------------------------------------------------------------
# 6. Section reader
# ---------------------------------------------------------------------------


class TestSectionReader:

    def _make_review_dir(self, tmp_path, candidates):
        """Create a mock production dir with chord review.yml."""
        chords_dir = tmp_path / "chords"
        chords_dir.mkdir(parents=True)
        review = {"candidates": candidates, "bpm": 84}
        with open(chords_dir / "review.yml", "w") as f:
            yaml.dump(review, f)
        return tmp_path

    def test_reads_approved_sections(self, tmp_path):
        from app.generators.midi.drum_pipeline import read_approved_sections

        prod_dir = self._make_review_dir(
            tmp_path,
            [
                {
                    "id": "chord_001",
                    "label": "Verse",
                    "status": "approved",
                    "chords": [1, 2, 3, 4],
                },
                {
                    "id": "chord_002",
                    "label": "Chorus",
                    "status": "Accepted",
                    "chords": [1, 2, 3, 4],
                },
                {
                    "id": "chord_003",
                    "label": None,
                    "status": "rejected",
                    "chords": [1, 2],
                },
            ],
        )
        sections = read_approved_sections(prod_dir)
        assert len(sections) == 2
        assert sections[0]["label"] == "verse"
        assert sections[1]["label"] == "chorus"

    def test_bar_count_from_chord_count(self, tmp_path):
        from app.generators.midi.drum_pipeline import read_approved_sections

        prod_dir = self._make_review_dir(
            tmp_path,
            [
                {
                    "id": "chord_001",
                    "label": "Verse",
                    "status": "approved",
                    "chords": [1, 2, 3],
                },
            ],
        )
        sections = read_approved_sections(prod_dir)
        assert sections[0]["bar_count"] == 3

    def test_rejects_no_approved(self, tmp_path):
        from app.generators.midi.drum_pipeline import read_approved_sections

        prod_dir = self._make_review_dir(
            tmp_path,
            [
                {"id": "chord_001", "label": "Verse", "status": "rejected"},
            ],
        )
        sections = read_approved_sections(prod_dir)
        assert sections == []

    def test_skips_unlabeled_approved(self, tmp_path):
        from app.generators.midi.drum_pipeline import read_approved_sections

        prod_dir = self._make_review_dir(
            tmp_path,
            [
                {
                    "id": "chord_001",
                    "label": None,
                    "status": "approved",
                    "chords": [1, 2],
                },
            ],
        )
        sections = read_approved_sections(prod_dir)
        assert sections == []

    def test_missing_review_raises(self, tmp_path):
        from app.generators.midi.drum_pipeline import read_approved_sections

        with pytest.raises(FileNotFoundError):
            read_approved_sections(tmp_path)


# ---------------------------------------------------------------------------
# 7. Composite scoring
# ---------------------------------------------------------------------------


class TestDrumCompositeScore:

    def test_perfect_scores(self):
        from app.generators.midi.drum_pipeline import drum_composite_score

        scorer_result = {
            "temporal": {"past": 0.8, "present": 0.1, "future": 0.1},
            "spatial": {"thing": 0.8, "place": 0.1, "person": 0.1},
            "ontological": {"imagined": 0.1, "forgotten": 0.1, "known": 0.8},
            "confidence": 1.0,
        }
        comp, breakdown = drum_composite_score(1.0, 1.0, scorer_result)
        assert comp == pytest.approx(1.0, abs=0.01)
        assert "energy_appropriateness" in breakdown
        assert "chromatic" in breakdown

    def test_weights_applied(self):
        from app.generators.midi.drum_pipeline import drum_composite_score

        scorer_result = {
            "temporal": {"past": 0.5, "present": 0.3, "future": 0.2},
            "spatial": {"thing": 0.5, "place": 0.3, "person": 0.2},
            "ontological": {"imagined": 0.3, "forgotten": 0.3, "known": 0.4},
            "confidence": 0.5,
        }
        # energy=1.0, chromatic=0.5
        comp, _ = drum_composite_score(
            1.0, 0.5, scorer_result, energy_weight=0.3, chromatic_weight=0.7
        )
        expected = 0.3 * 1.0 + 0.7 * 0.5
        assert comp == pytest.approx(expected, abs=0.01)


# ---------------------------------------------------------------------------
# 8. Integration test (mock scorer)
# ---------------------------------------------------------------------------


class TestDrumPipelineIntegration:

    def test_full_pipeline_with_mock_scorer(self, tmp_path, monkeypatch):
        """Run the pipeline end-to-end with a mocked ChromaticScorer."""
        # Set up mock production directory
        chords_dir = tmp_path / "chords"
        chords_dir.mkdir(parents=True)
        candidates_dir = chords_dir / "candidates"
        candidates_dir.mkdir()
        approved_dir = chords_dir / "approved"
        approved_dir.mkdir()

        # Create a minimal chord review.yml
        review = {
            "song_proposal": "test_song.yml",
            "thread": str(tmp_path),
            "bpm": 120,
            "color": "Red",
            "candidates": [
                {
                    "id": "chord_001",
                    "label": "Verse",
                    "status": "approved",
                    "chords": [
                        {"function": "I", "name": "C", "notes": ["C3", "E3", "G3"]},
                        {"function": "IV", "name": "F", "notes": ["F3", "A3", "C4"]},
                        {"function": "V", "name": "G", "notes": ["G3", "B3", "D4"]},
                        {"function": "I", "name": "C", "notes": ["C3", "E3", "G3"]},
                    ],
                },
                {
                    "id": "chord_002",
                    "label": "Chorus",
                    "status": "approved",
                    "chords": [
                        {"function": "I", "name": "C", "notes": ["C3", "E3", "G3"]},
                        {"function": "V", "name": "G", "notes": ["G3", "B3", "D4"]},
                    ],
                },
            ],
        }
        with open(chords_dir / "review.yml", "w") as f:
            yaml.dump(review, f)

        # Create a mock song proposal
        song_proposal = {
            "key": "C major",
            "bpm": 120,
            "tempo": {"numerator": 4, "denominator": 4},
            "rainbow_color": {"color_name": "Red"},
            "concept": "Test concept for Red",
            "genres": ["rock", "ambient"],
        }
        yml_dir = tmp_path / "yml"
        yml_dir.mkdir()
        with open(yml_dir / "test_song.yml", "w") as f:
            yaml.dump(song_proposal, f)

        # Create mock manifest
        with open(tmp_path / "manifest.yml", "w") as f:
            yaml.dump({"concept": "Test concept for Red"}, f)

        # Mock ChromaticScorer
        import numpy as np

        class MockScorer:
            def __init__(self, **kwargs):
                pass

            def prepare_concept(self, text):
                return np.zeros(768)

            def score(self, midi_bytes, concept_emb=None):
                return {
                    "temporal": {"past": 0.6, "present": 0.2, "future": 0.2},
                    "spatial": {"thing": 0.7, "place": 0.2, "person": 0.1},
                    "ontological": {"imagined": 0.1, "forgotten": 0.1, "known": 0.8},
                    "confidence": 0.75,
                }

            def score_batch(self, candidates, concept_emb=None):
                results = []
                for cand in candidates:
                    result = self.score(cand["midi_bytes"], concept_emb)
                    result["candidate"] = cand
                    results.append(result)
                return results

        # Monkeypatch the scorer at the source module (imported lazily inside run_drum_pipeline)
        monkeypatch.setattr(
            "training.chromatic_scorer.ChromaticScorer",
            MockScorer,
        )

        # Run the pipeline
        from app.generators.midi.drum_pipeline import run_drum_pipeline

        result = run_drum_pipeline(
            production_dir=str(tmp_path),
            thread_dir=str(tmp_path),
            song_filename="test_song.yml",
            seed=42,
            top_k=3,
        )

        # Verify results
        assert "verse" in result
        assert "chorus" in result
        assert len(result["verse"]) > 0
        assert len(result["chorus"]) > 0

        # Verify files were written
        drums_dir = tmp_path / "drums"
        assert drums_dir.exists()
        assert (drums_dir / "review.yml").exists()
        assert (drums_dir / "candidates").exists()

        # Verify review.yml structure
        with open(drums_dir / "review.yml") as f:
            drum_review = yaml.safe_load(f)
        assert drum_review["pipeline"] == "drum-generation"
        assert drum_review["bpm"] == 120
        assert len(drum_review["candidates"]) > 0

        # Verify each candidate has required fields
        for cand in drum_review["candidates"]:
            assert "id" in cand
            assert "midi_file" in cand
            assert "section" in cand
            assert "genre_family" in cand
            assert "pattern_name" in cand
            assert "energy" in cand
            assert "scores" in cand
            assert cand["status"] == "pending"
            assert cand["label"] is None

        # Verify MIDI files exist
        for cand in drum_review["candidates"]:
            midi_path = drums_dir / cand["midi_file"]
            assert midi_path.exists(), f"Missing MIDI: {midi_path}"
