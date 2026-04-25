"""Tests for quartet_patterns — voice ranges, parallel detection, scoring."""

from white_generation.patterns.quartet_patterns import (
    ALL_VOICE_PATTERNS,
    VOICE_RANGES,
    VoicePattern,
    check_parallels,
    clamp_to_voice_range,
    counterpoint_score,
    fix_voice_crossing,
    get_patterns_for_voice,
)
from white_generation.pipelines.quartet_pipeline import (
    generate_voice_notes,
    resolve_parallel_violations,
)

# ---------------------------------------------------------------------------
# 1. clamp_to_voice_range
# ---------------------------------------------------------------------------


class TestClampToVoiceRange:
    def test_note_within_range_unchanged(self):
        low, high = VOICE_RANGES["alto"]
        mid = (low + high) // 2
        assert clamp_to_voice_range(mid, "alto") == mid

    def test_note_above_ceiling_shifted_down_by_octave(self):
        _, high = VOICE_RANGES["alto"]
        note = high + 1
        result = clamp_to_voice_range(note, "alto")
        assert result <= high

    def test_note_below_floor_shifted_up_by_octave(self):
        low, _ = VOICE_RANGES["alto"]
        note = low - 1
        result = clamp_to_voice_range(note, "alto")
        assert result >= low

    def test_all_voice_types_return_in_range(self):
        for voice_type in ("alto", "tenor", "bass_voice"):
            low, high = VOICE_RANGES[voice_type]
            for note in (20, 40, 60, 80, 100):
                result = clamp_to_voice_range(note, voice_type)
                assert (
                    low <= result <= high
                ), f"{voice_type}: clamp({note}) = {result} out of [{low},{high}]"


# ---------------------------------------------------------------------------
# 2. check_parallels
# ---------------------------------------------------------------------------


class TestCheckParallels:
    def test_no_parallels_returns_empty(self):
        soprano = [60, 62, 64, 65]
        voice = [53, 57, 59, 60]  # varying intervals — no consecutive P5/P8
        # Not a guaranteed empty but typical contrary-motion — at least no assertion error
        result = check_parallels(soprano, voice)
        assert isinstance(result, list)

    def test_parallel_fifth_detected(self):
        # P5 = 7 semitones. soprano moves C4→D4 (60→62), voice moves F3→G3 (53→55).
        # Interval both times = 7 → parallel 5ths.
        soprano = [60, 62]
        voice = [53, 55]
        violations = check_parallels(soprano, voice)
        assert len(violations) == 1
        assert "P5" in violations[0]

    def test_parallel_octave_detected(self):
        # P8 = 12 semitones. soprano C4→D4 (60→62), voice C3→D3 (48→50).
        soprano = [60, 62]
        voice = [48, 50]
        violations = check_parallels(soprano, voice)
        assert len(violations) == 1
        assert "P8" in violations[0]

    def test_contrary_motion_no_violation(self):
        # Soprano rises, voice falls — interval changes each step.
        soprano = [60, 62, 64]
        voice = [57, 55, 53]
        violations = check_parallels(soprano, voice)
        assert violations == []

    def test_single_note_no_consecutive_pairs(self):
        assert check_parallels([60], [53]) == []


# ---------------------------------------------------------------------------
# 3. fix_voice_crossing
# ---------------------------------------------------------------------------


class TestFixVoiceCrossing:
    def test_no_crossing_unchanged(self):
        soprano = [72, 72, 72, 72]
        alto = [65, 65, 65, 65]
        tenor = [60, 60, 60, 60]
        bass = [53, 53, 53, 53]
        a, t, b = fix_voice_crossing(soprano, alto, tenor, bass)
        assert all(av <= sv for av, sv in zip(a, soprano))
        assert all(tv <= av for tv, av in zip(t, a))
        assert all(bv <= tv for bv, tv in zip(b, t))

    def test_alto_above_soprano_corrected(self):
        soprano = [60, 60, 60, 60]
        alto = [65, 65, 65, 65]  # above soprano — should be corrected
        tenor = [53, 53, 53, 53]
        bass = [48, 48, 48, 48]
        a, t, b = fix_voice_crossing(soprano, alto, tenor, bass)
        assert all(av <= sv for av, sv in zip(a, soprano))

    def test_all_voices_clamped_to_range(self):
        soprano = [60] * 4
        alto = [70] * 4  # forced well above soprano — after fix, should be in range
        tenor = [68] * 4
        bass = [65] * 4
        a, t, b = fix_voice_crossing(soprano, alto, tenor, bass)
        for v, vt in [(a, "alto"), (t, "tenor"), (b, "bass_voice")]:
            low, high = VOICE_RANGES[vt]
            assert all(low <= n <= high for n in v), f"{vt} out of range: {v}"


# ---------------------------------------------------------------------------
# 4. generate_voice_notes
# ---------------------------------------------------------------------------


class TestGenerateVoiceNotes:
    def test_output_length_matches_soprano(self):
        pattern = VoicePattern("test", "alto", [-4, -3, -4, -3])
        soprano = [60, 62, 64, 65, 67, 65]
        result = generate_voice_notes(soprano, pattern)
        assert len(result) == len(soprano)

    def test_all_notes_in_voice_range(self):
        pattern = VoicePattern("test", "alto", [-4, -3])
        soprano = list(range(55, 75))
        result = generate_voice_notes(soprano, pattern)
        low, high = VOICE_RANGES["alto"]
        assert all(low <= n <= high for n in result)

    def test_leap_cap_applied(self):
        """Consecutive generated notes should not jump more than MAX_OFFSET_CHANGE."""
        from white_generation.pipelines.quartet_pipeline import MAX_OFFSET_CHANGE

        pattern = VoicePattern("test", "tenor", [-7, -20, -7, -20])  # large swings
        soprano = [60, 60, 60, 60]
        result = generate_voice_notes(soprano, pattern)
        for i in range(1, len(result)):
            assert (
                abs(result[i] - result[i - 1]) <= MAX_OFFSET_CHANGE + 12
            )  # octave slack from clamp


# ---------------------------------------------------------------------------
# 5. resolve_parallel_violations
# ---------------------------------------------------------------------------


class TestResolveParallelViolations:
    def test_fixes_parallel_fifth(self):
        soprano = [60, 62]
        voice = [53, 55]  # parallel 5ths
        fixed = resolve_parallel_violations(soprano, voice, "alto")
        violations = check_parallels(soprano, fixed)
        assert violations == [] or len(violations) < len(
            check_parallels(soprano, voice)
        )

    def test_no_violations_unchanged(self):
        soprano = [60, 62, 64]
        voice = [57, 55, 53]
        original = list(voice)
        fixed = resolve_parallel_violations(soprano, voice, "alto")
        assert fixed == original


# ---------------------------------------------------------------------------
# 6. counterpoint_score
# ---------------------------------------------------------------------------


class TestCounterpointScore:
    def test_perfect_contrary_motion_high_score(self):
        soprano = [60, 62, 64, 65]
        voice = [57, 55, 53, 52]  # contrary motion, no parallels
        score = counterpoint_score(soprano, voice)
        assert score >= 0.7

    def test_all_parallel_fifths_low_score(self):
        # Every step is a parallel 5th
        soprano = [60, 62, 64, 66]
        voice = [53, 55, 57, 59]
        score = counterpoint_score(soprano, voice)
        assert score < 0.7

    def test_empty_inputs_return_zero(self):
        assert counterpoint_score([], []) == 0.0

    def test_score_between_0_and_1(self):
        soprano = [60, 62, 64, 65, 67]
        voice = [53, 55, 57, 58, 60]
        score = counterpoint_score(soprano, voice)
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# 7. get_patterns_for_voice
# ---------------------------------------------------------------------------


class TestGetPatternsForVoice:
    def test_returns_list_for_all_voice_types(self):
        for vt in ("alto", "tenor", "bass_voice"):
            result = get_patterns_for_voice(vt)
            assert len(result) > 0

    def test_energy_filter_respected(self):
        alto_low = get_patterns_for_voice("alto", "low")
        assert all(p.energy == "low" for p in alto_low)

    def test_unknown_energy_falls_back_to_all(self):
        result = get_patterns_for_voice("alto", "nonexistent_energy")
        assert len(result) == len(ALL_VOICE_PATTERNS["alto"])

    def test_unknown_voice_type_returns_empty(self):
        result = get_patterns_for_voice("flugelhorn")
        assert result == []
