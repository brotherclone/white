"""Tests for the bass line pattern template library."""

# ---------------------------------------------------------------------------
# 1. Template validation
# ---------------------------------------------------------------------------


class TestBassPatternTemplates:

    def test_all_templates_have_required_fields(self):
        from app.generators.midi.bass_patterns import ALL_TEMPLATES

        for t in ALL_TEMPLATES:
            assert t.name, "Template missing name"
            assert t.style, f"{t.name}: missing style"
            assert t.energy in (
                "low",
                "medium",
                "high",
            ), f"{t.name}: invalid energy '{t.energy}'"
            assert len(t.time_sig) == 2, f"{t.name}: invalid time_sig"
            assert t.description, f"{t.name}: missing description"
            assert len(t.notes) > 0, f"{t.name}: no notes defined"

    def test_all_templates_have_valid_velocity_levels(self):
        from app.generators.midi.bass_patterns import ALL_TEMPLATES, VELOCITY

        valid_levels = set(VELOCITY.keys())
        for t in ALL_TEMPLATES:
            for beat_pos, tone_sel, vel_level in t.notes:
                assert (
                    vel_level in valid_levels
                ), f"{t.name}: invalid velocity '{vel_level}'"

    def test_all_templates_have_valid_tone_selections(self):
        from app.generators.midi.bass_patterns import ALL_TEMPLATES

        valid_tones = {
            "root",
            "5th",
            "3rd",
            "octave_up",
            "octave_down",
            "chromatic_approach",
            "passing_tone",
        }
        for t in ALL_TEMPLATES:
            for beat_pos, tone_sel, vel_level in t.notes:
                assert (
                    tone_sel in valid_tones
                ), f"{t.name}: invalid tone_selection '{tone_sel}'"

    def test_all_templates_beat_positions_within_bar(self):
        from app.generators.midi.bass_patterns import ALL_TEMPLATES

        for t in ALL_TEMPLATES:
            bar_length = t.bar_length_beats()
            for beat_pos, _, _ in t.notes:
                assert (
                    0 <= beat_pos < bar_length
                ), f"{t.name}: beat {beat_pos} outside bar length {bar_length}"

    def test_unique_template_names(self):
        from app.generators.midi.bass_patterns import ALL_TEMPLATES

        names = [t.name for t in ALL_TEMPLATES]
        assert len(names) == len(
            set(names)
        ), f"Duplicate template names: {[n for n in names if names.count(n) > 1]}"

    def test_4_4_templates_minimum_count(self):
        from app.generators.midi.bass_patterns import ALL_TEMPLATES

        count_4_4 = len([t for t in ALL_TEMPLATES if t.time_sig == (4, 4)])
        assert count_4_4 >= 12, f"Only {count_4_4} 4/4 templates (need >= 12)"

    def test_4_4_templates_have_multiple_styles(self):
        from app.generators.midi.bass_patterns import ALL_TEMPLATES

        styles_4_4 = {t.style for t in ALL_TEMPLATES if t.time_sig == (4, 4)}
        for expected in ["root", "walking", "arpeggiated", "syncopated"]:
            assert expected in styles_4_4, f"No 4/4 templates for style '{expected}'"

    def test_7_8_templates_exist(self):
        from app.generators.midi.bass_patterns import ALL_TEMPLATES

        count_7_8 = len([t for t in ALL_TEMPLATES if t.time_sig == (7, 8)])
        assert count_7_8 >= 3, f"Only {count_7_8} 7/8 templates (need >= 3)"

    def test_note_durations_parallel_when_present(self):
        from app.generators.midi.bass_patterns import ALL_TEMPLATES

        for t in ALL_TEMPLATES:
            if t.note_durations is not None:
                assert len(t.note_durations) == len(t.notes), (
                    f"{t.name}: note_durations ({len(t.note_durations)}) "
                    f"!= notes ({len(t.notes)})"
                )

    def test_bar_length_beats(self):
        from app.generators.midi.bass_patterns import BassPattern

        p44 = BassPattern("t", "root", "low", (4, 4), "test", [(0, "root", "normal")])
        assert p44.bar_length_beats() == 4.0

        p78 = BassPattern("t", "root", "low", (7, 8), "test", [(0, "root", "normal")])
        assert p78.bar_length_beats() == 3.5


# ---------------------------------------------------------------------------
# 2. Tone resolution
# ---------------------------------------------------------------------------


class TestToneResolution:

    def test_clamp_to_bass_register_high_note(self):
        from app.generators.midi.bass_patterns import clamp_to_bass_register

        # C5 = 72, should become C3 = 48
        result = clamp_to_bass_register(72)
        assert 24 <= result <= 60

    def test_clamp_to_bass_register_low_note(self):
        from app.generators.midi.bass_patterns import clamp_to_bass_register

        # C0 = 12, should become C1 = 24
        result = clamp_to_bass_register(12)
        assert 24 <= result <= 60

    def test_clamp_to_bass_register_in_range(self):
        from app.generators.midi.bass_patterns import clamp_to_bass_register

        assert clamp_to_bass_register(36) == 36  # C2, already in range

    def test_extract_root_from_voicing(self):
        from app.generators.midi.bass_patterns import extract_root

        # C major chord: C4=60, E4=64, G4=67 → root should be in bass register
        root = extract_root([60, 64, 67])
        assert root % 12 == 0  # pitch class C
        assert 24 <= root <= 60

    def test_extract_root_empty_voicing(self):
        from app.generators.midi.bass_patterns import extract_root

        assert extract_root([]) == 36  # fallback C2

    def test_extract_chord_tones_c_major(self):
        from app.generators.midi.bass_patterns import extract_chord_tones

        # C major: C4=60, E4=64, G4=67
        tones = extract_chord_tones([60, 64, 67])
        root_pc = tones["root"] % 12
        assert root_pc == 0  # C
        assert tones["3rd"] % 12 == 4  # E (major 3rd)
        assert tones["5th"] % 12 == 7  # G (perfect 5th)

    def test_extract_chord_tones_a_minor(self):
        from app.generators.midi.bass_patterns import extract_chord_tones

        # A minor: A3=57, C4=60, E4=64
        tones = extract_chord_tones([57, 60, 64])
        root_pc = tones["root"] % 12
        assert root_pc == 9  # A
        assert tones["3rd"] % 12 == 0  # C (minor 3rd = 3 semitones above A)
        assert tones["5th"] % 12 == 4  # E (perfect 5th)

    def test_resolve_tone_root(self):
        from app.generators.midi.bass_patterns import resolve_tone

        note = resolve_tone("root", [60, 64, 67])
        assert note % 12 == 0  # C
        assert 24 <= note <= 60

    def test_resolve_tone_5th(self):
        from app.generators.midi.bass_patterns import resolve_tone

        note = resolve_tone("5th", [60, 64, 67])
        assert note % 12 == 7  # G
        assert 24 <= note <= 60

    def test_resolve_tone_3rd(self):
        from app.generators.midi.bass_patterns import resolve_tone

        note = resolve_tone("3rd", [60, 64, 67])
        assert note % 12 == 4  # E
        assert 24 <= note <= 60

    def test_resolve_tone_octave_up(self):
        from app.generators.midi.bass_patterns import resolve_tone

        root = resolve_tone("root", [36, 40, 43])  # C2
        octave = resolve_tone("octave_up", [36, 40, 43])
        assert octave == root + 12 or octave == root  # may clamp

    def test_resolve_tone_octave_down(self):
        from app.generators.midi.bass_patterns import resolve_tone

        note = resolve_tone("octave_down", [48, 52, 55])  # C3
        assert 24 <= note <= 60
        assert note % 12 == 0  # still C

    def test_resolve_chromatic_approach_with_next(self):
        from app.generators.midi.bass_patterns import resolve_tone

        # Current = C, Next = D (root 38 = D2)
        note = resolve_tone(
            "chromatic_approach", [36, 40, 43], next_voicing=[38, 42, 45]
        )
        # Should be one semitone below next root (D2=38 → C#2=37)
        assert note == 37

    def test_resolve_chromatic_approach_without_next(self):
        from app.generators.midi.bass_patterns import resolve_tone

        # Last chord — falls back to root
        note = resolve_tone("chromatic_approach", [36, 40, 43], next_voicing=None)
        assert note % 12 == 0  # C (root)

    def test_resolve_passing_tone_ascending(self):
        from app.generators.midi.bass_patterns import resolve_tone

        # Current root C2=36, next root E2=40 → passing = root + 2 = 38 (D2)
        note = resolve_tone("passing_tone", [36, 40, 43], next_voicing=[40, 44, 47])
        assert note == 38

    def test_resolve_passing_tone_descending(self):
        from app.generators.midi.bass_patterns import resolve_tone

        # Current root E2=40, next root C2=36 → passing = root - 2 = 38 (D2)
        note = resolve_tone("passing_tone", [40, 44, 47], next_voicing=[36, 40, 43])
        assert note == 38

    def test_resolve_passing_tone_same_root(self):
        from app.generators.midi.bass_patterns import resolve_tone

        # Same root → returns root
        note = resolve_tone("passing_tone", [36, 40, 43], next_voicing=[36, 40, 43])
        assert note % 12 == 0

    def test_resolve_unknown_tone_returns_root(self):
        from app.generators.midi.bass_patterns import resolve_tone

        note = resolve_tone("nonexistent", [36, 40, 43])
        assert note % 12 == 0  # root


# ---------------------------------------------------------------------------
# 3. Template selection
# ---------------------------------------------------------------------------


class TestTemplateSelection:

    def test_select_by_time_sig_and_energy(self):
        from app.generators.midi.bass_patterns import ALL_TEMPLATES, select_templates

        result = select_templates(ALL_TEMPLATES, (4, 4), "medium")
        assert len(result) > 0
        for t in result:
            assert t.time_sig == (4, 4)

    def test_select_includes_adjacent_energy(self):
        from app.generators.midi.bass_patterns import ALL_TEMPLATES, select_templates

        result = select_templates(ALL_TEMPLATES, (4, 4), "low")
        energies = {t.energy for t in result}
        assert "low" in energies
        # Should also include medium (one step away)
        assert "medium" in energies

    def test_select_excludes_two_levels_away(self):
        from app.generators.midi.bass_patterns import ALL_TEMPLATES, select_templates

        result = select_templates(ALL_TEMPLATES, (4, 4), "low")
        energies = {t.energy for t in result}
        assert "high" not in energies

    def test_select_7_8(self):
        from app.generators.midi.bass_patterns import ALL_TEMPLATES, select_templates

        result = select_templates(ALL_TEMPLATES, (7, 8), "medium")
        assert len(result) > 0
        for t in result:
            assert t.time_sig == (7, 8)

    def test_select_unknown_time_sig_returns_empty(self):
        from app.generators.midi.bass_patterns import ALL_TEMPLATES, select_templates

        result = select_templates(ALL_TEMPLATES, (5, 4), "medium")
        assert len(result) == 0

    def test_fallback_pattern(self):
        from app.generators.midi.bass_patterns import make_fallback_pattern

        fb = make_fallback_pattern((5, 4))
        assert fb.time_sig == (5, 4)
        assert fb.style == "root"
        assert len(fb.notes) == 1
        assert fb.notes[0][1] == "root"

    def test_exact_energy_first(self):
        from app.generators.midi.bass_patterns import ALL_TEMPLATES, select_templates

        result = select_templates(ALL_TEMPLATES, (4, 4), "low")
        if len(result) >= 2:
            # First templates should be exact matches
            first_energy = result[0].energy
            assert first_energy == "low"


# ---------------------------------------------------------------------------
# 4. Theory scoring
# ---------------------------------------------------------------------------


class TestTheoryScoring:

    def test_root_adherence_all_root(self):
        from app.generators.midi.bass_patterns import root_adherence

        # All notes on strong beats are root (C = pitch class 0)
        notes = [(0.0, 36), (2.0, 48)]  # C2 on beat 1, C3 on beat 3
        assert root_adherence(notes, 36, (4, 4)) == 1.0

    def test_root_adherence_none_root(self):
        from app.generators.midi.bass_patterns import root_adherence

        # No notes on strong beats are root
        notes = [(0.0, 40), (2.0, 43)]  # E2, G2 — not C
        assert root_adherence(notes, 36, (4, 4)) == 0.0

    def test_root_adherence_partial(self):
        from app.generators.midi.bass_patterns import root_adherence

        # One root, one non-root on strong beats
        notes = [(0.0, 36), (2.0, 40)]  # C2 (root), E2 (not root)
        assert root_adherence(notes, 36, (4, 4)) == 0.5

    def test_root_adherence_off_beat_notes_ignored(self):
        from app.generators.midi.bass_patterns import root_adherence

        # Note on beat 1 (strong) and beat 1.5 (weak)
        notes = [(0.0, 36), (1.5, 40)]  # C2 on strong, E2 on weak
        # Only beat 0 is strong, and it's root
        assert root_adherence(notes, 36, (4, 4)) == 1.0

    def test_kick_alignment_perfect(self):
        from app.generators.midi.bass_patterns import kick_alignment

        bass = [0.0, 2.0]
        kicks = [0.0, 2.0]
        assert kick_alignment(bass, kicks) == 1.0

    def test_kick_alignment_none(self):
        from app.generators.midi.bass_patterns import kick_alignment

        bass = [0.0, 2.0]
        kicks = [1.0, 3.0]  # completely offset
        assert kick_alignment(bass, kicks) == 0.0

    def test_kick_alignment_partial(self):
        from app.generators.midi.bass_patterns import kick_alignment

        bass = [0.0, 1.0, 2.0, 3.0]
        kicks = [0.0, 2.0]  # 2/4 aligned
        assert kick_alignment(bass, kicks) == 0.5

    def test_kick_alignment_empty(self):
        from app.generators.midi.bass_patterns import kick_alignment

        assert kick_alignment([], [0.0]) == 0.0
        assert kick_alignment([0.0], []) == 0.0

    def test_voice_leading_score_unison(self):
        from app.generators.midi.bass_patterns import voice_leading_score

        assert voice_leading_score([0]) == 1.0

    def test_voice_leading_score_large_interval(self):
        from app.generators.midi.bass_patterns import voice_leading_score

        # Octave = 12 semitones → 0.3
        assert voice_leading_score([12]) == 0.3

    def test_voice_leading_score_empty(self):
        from app.generators.midi.bass_patterns import voice_leading_score

        assert voice_leading_score([]) == 1.0

    def test_bass_theory_score_with_kick(self):
        from app.generators.midi.bass_patterns import bass_theory_score

        # All scores = 1.0 → mean = 1.0
        assert bass_theory_score(1.0, 1.0, 1.0) == 1.0

    def test_bass_theory_score_without_kick(self):
        from app.generators.midi.bass_patterns import bass_theory_score

        # No kick → mean of root + voice_leading only
        assert bass_theory_score(1.0, None, 0.5) == 0.75

    def test_voice_leading_score_mixed(self):
        from app.generators.midi.bass_patterns import voice_leading_score

        # semitone (1.0) + perfect 5th (0.5) → mean 0.75
        assert voice_leading_score([1, 7]) == 0.75
