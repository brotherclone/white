"""Tests for the melody contour pattern template library."""

import pytest

# ---------------------------------------------------------------------------
# 1. Template validation
# ---------------------------------------------------------------------------


class TestMelodyPatternTemplates:

    def test_all_templates_have_required_fields(self):
        from white_generation.patterns.melody_patterns import ALL_TEMPLATES

        for t in ALL_TEMPLATES:
            assert t.name, "Template missing name"
            assert t.contour, f"{t.name}: missing contour"
            assert t.energy in (
                "low",
                "medium",
                "high",
            ), f"{t.name}: invalid energy '{t.energy}'"
            assert len(t.time_sig) == 2, f"{t.name}: invalid time_sig"
            assert t.description, f"{t.name}: missing description"
            assert len(t.intervals) > 0, f"{t.name}: no intervals defined"
            assert len(t.rhythm) > 0, f"{t.name}: no rhythm defined"

    def test_intervals_and_rhythm_same_length(self):
        from white_generation.patterns.melody_patterns import ALL_TEMPLATES

        for t in ALL_TEMPLATES:
            assert len(t.intervals) == len(
                t.rhythm
            ), f"{t.name}: intervals ({len(t.intervals)}) != rhythm ({len(t.rhythm)})"

    def test_first_interval_is_zero(self):
        from white_generation.patterns.melody_patterns import ALL_TEMPLATES

        for t in ALL_TEMPLATES:
            assert (
                t.intervals[0] == 0
            ), f"{t.name}: first interval must be 0, got {t.intervals[0]}"

    def test_rhythm_positions_within_bar(self):
        from white_generation.patterns.melody_patterns import ALL_TEMPLATES

        for t in ALL_TEMPLATES:
            bar_length = t.bar_length_beats()
            for pos in t.rhythm:
                assert (
                    0 <= pos < bar_length
                ), f"{t.name}: rhythm pos {pos} outside bar length {bar_length}"

    def test_rhythm_positions_ascending(self):
        from white_generation.patterns.melody_patterns import ALL_TEMPLATES

        for t in ALL_TEMPLATES:
            for i in range(1, len(t.rhythm)):
                assert (
                    t.rhythm[i] >= t.rhythm[i - 1]
                ), f"{t.name}: rhythm not ascending at index {i}"

    def test_unique_template_names(self):
        from white_generation.patterns.melody_patterns import ALL_TEMPLATES

        names = [t.name for t in ALL_TEMPLATES]
        assert len(names) == len(
            set(names)
        ), f"Duplicate names: {[n for n in names if names.count(n) > 1]}"

    def test_4_4_templates_minimum_count(self):
        from white_generation.patterns.melody_patterns import ALL_TEMPLATES

        count = len([t for t in ALL_TEMPLATES if t.time_sig == (4, 4)])
        assert count >= 12, f"Only {count} 4/4 templates (need >= 12)"

    def test_7_8_templates_minimum_count(self):
        from white_generation.patterns.melody_patterns import ALL_TEMPLATES

        count = len([t for t in ALL_TEMPLATES if t.time_sig == (7, 8)])
        assert count >= 6, f"Only {count} 7/8 templates (need >= 6)"

    def test_4_4_has_required_contour_types(self):
        from white_generation.patterns.melody_patterns import ALL_TEMPLATES

        contours = {t.contour for t in ALL_TEMPLATES if t.time_sig == (4, 4)}
        for required in ("stepwise", "arpeggiated", "repeated", "leap_step"):
            assert required in contours, f"Missing 4/4 contour type: {required}"

    def test_valid_contour_types(self):
        from white_generation.patterns.melody_patterns import ALL_TEMPLATES

        valid = {
            "stepwise",
            "arpeggiated",
            "repeated",
            "leap_step",
            "pentatonic",
            "scalar_run",
            "declarative",
            "call_and_rest",
            "haiku",
            "incantatory",
            "drone_and_step",
            "conversational",
            "anacrusis",
            "dotted",
        }
        for t in ALL_TEMPLATES:
            assert t.contour in valid, f"{t.name}: invalid contour '{t.contour}'"

    def test_durations_length_matches_if_present(self):
        from white_generation.patterns.melody_patterns import ALL_TEMPLATES

        for t in ALL_TEMPLATES:
            if t.durations is not None:
                assert len(t.durations) == len(
                    t.intervals
                ), f"{t.name}: durations ({len(t.durations)}) != intervals ({len(t.intervals)})"


# ---------------------------------------------------------------------------
# 2. Singer ranges
# ---------------------------------------------------------------------------


class TestSingerRanges:

    def test_all_singers_defined(self):
        from white_generation.patterns.melody_patterns import SINGERS

        for name in ("busyayo", "gabriel", "robbie", "shirley", "katherine"):
            assert name in SINGERS, f"Missing singer: {name}"

    def test_singer_ranges_valid(self):
        from white_generation.patterns.melody_patterns import SINGERS

        for key, s in SINGERS.items():
            assert s.low < s.high, f"{key}: low ({s.low}) >= high ({s.high})"
            assert s.low >= 0, f"{key}: low < 0"
            assert s.high <= 127, f"{key}: high > 127"
            assert s.voice_type, f"{key}: missing voice_type"

    def test_singer_mid_calculation(self):
        from white_generation.patterns.melody_patterns import SINGERS

        for key, s in SINGERS.items():
            expected = (s.low + s.high) // 2
            assert s.mid == expected, f"{key}: mid {s.mid} != expected {expected}"

    def test_busyayo_is_baritone(self):
        from white_generation.patterns.melody_patterns import SINGERS

        s = SINGERS["busyayo"]
        assert s.low == 45
        assert s.high == 64
        assert s.voice_type == "baritone"


# ---------------------------------------------------------------------------
# 3. Clamping and inference
# ---------------------------------------------------------------------------


class TestClampAndInfer:

    def test_clamp_within_range_unchanged(self):
        from white_generation.patterns.melody_patterns import (
            SINGERS,
            clamp_to_singer_range,
        )

        singer = SINGERS["gabriel"]
        assert clamp_to_singer_range(55, singer) == 55

    def test_clamp_above_range_octave_down(self):
        from white_generation.patterns.melody_patterns import (
            SINGERS,
            clamp_to_singer_range,
        )

        singer = SINGERS["gabriel"]  # 48-67
        result = clamp_to_singer_range(72, singer)
        assert singer.low <= result <= singer.high

    def test_clamp_below_range_octave_up(self):
        from white_generation.patterns.melody_patterns import (
            SINGERS,
            clamp_to_singer_range,
        )

        singer = SINGERS["gabriel"]  # 48-67
        result = clamp_to_singer_range(36, singer)
        assert singer.low <= result <= singer.high

    def test_clamp_extreme_value(self):
        from white_generation.patterns.melody_patterns import (
            SINGERS,
            clamp_to_singer_range,
        )

        singer = SINGERS["gabriel"]
        result = clamp_to_singer_range(120, singer)
        assert singer.low <= result <= singer.high

    def test_infer_singer_returns_singer_range(self):
        from white_generation.patterns.melody_patterns import (
            SingerRange,
            infer_singer,
        )

        result = infer_singer(48)
        assert isinstance(result, SingerRange)

    def test_infer_singer_different_tonics(self):
        from white_generation.patterns.melody_patterns import infer_singer

        # Just ensure it doesn't crash for various tonics
        for tonic in range(36, 84):
            result = infer_singer(tonic)
            assert result.name


# ---------------------------------------------------------------------------
# 4. Melody resolution
# ---------------------------------------------------------------------------


class TestMelodyResolution:

    def test_resolve_basic_pattern(self):
        from white_generation.patterns.melody_patterns import (
            ALL_TEMPLATES,
            SINGERS,
            resolve_melody_notes,
        )

        pattern = ALL_TEMPLATES[0]
        voicing = [48, 52, 55]  # C major
        singer = SINGERS["gabriel"]

        notes = resolve_melody_notes(pattern, voicing, singer)
        assert len(notes) > 0
        for onset, note, dur in notes:
            assert singer.low <= note <= singer.high, f"Note {note} outside range"
            assert dur > 0, f"Duration {dur} <= 0"

    def test_resolve_all_templates(self):
        from white_generation.patterns.melody_patterns import (
            ALL_TEMPLATES,
            SINGERS,
            resolve_melody_notes,
        )

        voicing = [60, 64, 67]  # C major
        for singer in SINGERS.values():
            for tmpl in ALL_TEMPLATES:
                notes = resolve_melody_notes(tmpl, voicing, singer)
                assert len(notes) > 0, f"No notes for {tmpl.name} with {singer.name}"
                for onset, note, dur in notes:
                    assert (
                        singer.low <= note <= singer.high
                    ), f"{tmpl.name}/{singer.name}: note {note} outside range"

    def test_resolve_with_empty_voicing_uses_fallback(self):
        from white_generation.patterns.melody_patterns import (
            ALL_TEMPLATES,
            SINGERS,
            resolve_melody_notes,
        )

        notes = resolve_melody_notes(ALL_TEMPLATES[0], [], SINGERS["gabriel"])
        assert len(notes) > 0

    def test_phrase_ending_resolves_to_root_or_fifth(self):
        from white_generation.patterns.melody_patterns import (
            ALL_TEMPLATES,
            SINGERS,
            resolve_melody_notes,
        )

        voicing = [48, 52, 55]  # C major: root=C, 5th=G
        singer = SINGERS["gabriel"]

        for tmpl in ALL_TEMPLATES[:5]:
            notes = resolve_melody_notes(tmpl, voicing, singer)
            if notes:
                _, last_note, _ = notes[-1]
                last_pc = last_note % 12
                root_pc = 0  # C
                fifth_pc = 7  # G
                assert last_pc in (
                    root_pc,
                    fifth_pc,
                ), f"{tmpl.name}: last note {last_note} (pc={last_pc}) not root or 5th"


# ---------------------------------------------------------------------------
# 5. Strong-beat chord snap
# ---------------------------------------------------------------------------


class TestStrongBeatSnap:

    def test_snap_to_chord_tone(self):
        from white_generation.patterns.melody_patterns import (
            SINGERS,
            strong_beat_chord_snap,
        )

        singer = SINGERS["gabriel"]
        chord_tones_pc = {0, 4, 7}  # C, E, G
        notes = [(0.0, 49, 1.0)]  # C#3 on beat 0 — should snap to C or E

        result = strong_beat_chord_snap(notes, chord_tones_pc, (4, 4), singer)
        assert result[0][1] % 12 in chord_tones_pc

    def test_no_snap_on_weak_beat(self):
        from white_generation.patterns.melody_patterns import (
            SINGERS,
            strong_beat_chord_snap,
        )

        singer = SINGERS["gabriel"]
        chord_tones_pc = {0, 4, 7}
        notes = [(1.0, 49, 1.0)]  # beat 1 is weak — should not snap

        result = strong_beat_chord_snap(notes, chord_tones_pc, (4, 4), singer)
        assert result[0][1] == 49

    def test_no_snap_if_already_chord_tone(self):
        from white_generation.patterns.melody_patterns import (
            SINGERS,
            strong_beat_chord_snap,
        )

        singer = SINGERS["gabriel"]
        chord_tones_pc = {0, 4, 7}
        notes = [(0.0, 48, 1.0)]  # C3 — already a chord tone

        result = strong_beat_chord_snap(notes, chord_tones_pc, (4, 4), singer)
        assert result[0][1] == 48


# ---------------------------------------------------------------------------
# 6. Template selection
# ---------------------------------------------------------------------------


class TestTemplateSelection:

    def test_select_exact_energy(self):
        from white_generation.patterns.melody_patterns import (
            ALL_TEMPLATES,
            select_templates,
        )

        results = select_templates(ALL_TEMPLATES, (4, 4), "medium")
        assert len(results) > 0
        # First result should be exact energy match
        assert results[0].energy == "medium"

    def test_select_includes_adjacent_energy(self):
        from white_generation.patterns.melody_patterns import (
            ALL_TEMPLATES,
            select_templates,
        )

        results = select_templates(ALL_TEMPLATES, (4, 4), "low")
        energies = {t.energy for t in results}
        # Should include low and medium (adjacent), not high
        assert "low" in energies
        assert "high" not in energies

    def test_select_7_8_templates(self):
        from white_generation.patterns.melody_patterns import (
            ALL_TEMPLATES,
            select_templates,
        )

        results = select_templates(ALL_TEMPLATES, (7, 8), "medium")
        assert len(results) > 0
        for t in results:
            assert t.time_sig == (7, 8)

    def test_select_unsupported_time_sig_empty(self):
        from white_generation.patterns.melody_patterns import (
            ALL_TEMPLATES,
            select_templates,
        )

        results = select_templates(ALL_TEMPLATES, (5, 4), "medium")
        assert len(results) == 0

    def test_fallback_pattern(self):
        from white_generation.patterns.melody_patterns import make_fallback_pattern

        fb = make_fallback_pattern((5, 4))
        assert fb.time_sig == (5, 4)
        assert fb.contour == "repeated"
        assert len(fb.intervals) == 1
        assert fb.intervals[0] == 0


# ---------------------------------------------------------------------------
# 7. Theory scoring
# ---------------------------------------------------------------------------


class TestTheoryScoring:

    def test_singability_basic(self):
        from white_generation.patterns.melody_patterns import (
            SINGERS,
            singability_score,
        )

        singer = SINGERS["gabriel"]
        # Stepwise notes — should score reasonably well
        notes = [(i * 0.5, 55 + i, 0.5) for i in range(8)]
        score = singability_score(notes, singer)
        assert 0.0 <= score <= 1.0

    def test_singability_large_leaps_penalised(self):
        from white_generation.patterns.melody_patterns import (
            SINGERS,
            singability_score,
        )

        singer = SINGERS["gabriel"]
        # Large leaps
        notes = [(0, 48, 1.0), (1, 67, 1.0), (2, 48, 1.0), (3, 67, 1.0)]
        large_score = singability_score(notes, singer)
        # Stepwise
        notes_step = [(0, 55, 1.0), (1, 57, 1.0), (2, 58, 1.0), (3, 60, 1.0)]
        step_score = singability_score(notes_step, singer)
        assert step_score > large_score

    def test_chord_tone_alignment_all_chord_tones(self):
        from white_generation.patterns.melody_patterns import chord_tone_alignment

        chord_tones_pc = {0, 4, 7}  # C, E, G
        notes = [(0.0, 48, 1.0), (2.0, 52, 1.0)]  # C and E on strong beats
        score = chord_tone_alignment(notes, chord_tones_pc, (4, 4))
        assert score == 1.0

    def test_chord_tone_alignment_no_chord_tones(self):
        from white_generation.patterns.melody_patterns import chord_tone_alignment

        chord_tones_pc = {0, 4, 7}
        notes = [(0.0, 49, 1.0), (2.0, 51, 1.0)]  # C# and Eb on strong beats
        score = chord_tone_alignment(notes, chord_tones_pc, (4, 4))
        assert score == 0.0

    def test_contour_quality_basic(self):
        from white_generation.patterns.melody_patterns import contour_quality

        notes = [(i, 55 + i, 1.0) for i in range(6)]
        score = contour_quality(notes)
        assert 0.0 <= score <= 1.0

    def test_contour_quality_repetition_penalised(self):
        from white_generation.patterns.melody_patterns import contour_quality

        # 6 consecutive same pitches
        notes_repeat = [(i, 55, 1.0) for i in range(6)]
        # Varied pitches
        notes_varied = [(i, 55 + i % 3, 1.0) for i in range(6)]

        assert contour_quality(notes_varied) > contour_quality(notes_repeat)

    def test_melody_theory_score_is_mean(self):
        from white_generation.patterns.melody_patterns import melody_theory_score

        assert melody_theory_score(0.6, 0.9, 0.3) == pytest.approx(0.6)
        assert melody_theory_score(1.0, 1.0, 1.0) == pytest.approx(1.0)
        assert melody_theory_score(0.0, 0.0, 0.0) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 8. use_case field and vocal template constraints
# ---------------------------------------------------------------------------


class TestUseCaseField:

    def test_all_templates_have_use_case(self):
        from white_generation.patterns.melody_patterns import ALL_TEMPLATES

        for t in ALL_TEMPLATES:
            assert t.use_case in (
                "vocal",
                "lead",
            ), f"{t.name}: invalid use_case '{t.use_case}'"

    def test_existing_lead_templates_marked_lead(self):
        from white_generation.patterns.melody_patterns import ALL_TEMPLATES

        lead_templates = [t for t in ALL_TEMPLATES if t.use_case == "lead"]
        assert (
            len(lead_templates) >= 12
        ), f"Expected >= 12 lead templates, got {len(lead_templates)}"

    def test_vocal_4_4_minimum_count(self):
        from white_generation.patterns.melody_patterns import ALL_TEMPLATES

        vocal_4_4 = [
            t for t in ALL_TEMPLATES if t.use_case == "vocal" and t.time_sig == (4, 4)
        ]
        assert (
            len(vocal_4_4) >= 30
        ), f"Expected >= 30 vocal 4/4 templates, got {len(vocal_4_4)}"

    def test_six_vocal_archetypes_present(self):
        from white_generation.patterns.melody_patterns import ALL_TEMPLATES

        vocal_4_4_contours = {
            t.contour
            for t in ALL_TEMPLATES
            if t.use_case == "vocal" and t.time_sig == (4, 4)
        }
        for archetype in (
            "declarative",
            "call_and_rest",
            "haiku",
            "incantatory",
            "drone_and_step",
            "conversational",
        ):
            assert (
                archetype in vocal_4_4_contours
            ), f"Missing vocal archetype: {archetype}"

    def test_vocal_templates_max_six_onsets_per_bar_4_4(self):
        from white_generation.patterns.melody_patterns import ALL_TEMPLATES

        for t in ALL_TEMPLATES:
            if t.use_case != "vocal" or t.time_sig != (4, 4):
                continue
            onset_count = len(t.rhythm)
            assert (
                onset_count <= 6
            ), f"{t.name}: vocal 4/4 template has {onset_count} onsets (max 6)"

    def test_vocal_templates_have_rest_gap(self):
        from white_generation.patterns.melody_patterns import ALL_TEMPLATES

        for t in ALL_TEMPLATES:
            if t.use_case != "vocal":
                continue
            if t.durations is None:
                continue
            # A vocal template satisfies the breath-gap requirement if either:
            # (a) there is a gap >= 0.5 beats between consecutive notes, OR
            # (b) the first note starts >= 0.5 beats into the bar (anacrusis opening rest).
            has_rest = t.rhythm[0] >= 0.5
            for i in range(len(t.rhythm) - 1):
                note_end = t.rhythm[i] + t.durations[i]
                next_onset = t.rhythm[i + 1]
                if next_onset - note_end >= 0.5:
                    has_rest = True
                    break
            assert has_rest, f"{t.name}: vocal template has no rest gap >= 0.5 beats"

    def test_vocal_templates_have_held_note(self):
        from white_generation.patterns.melody_patterns import ALL_TEMPLATES

        for t in ALL_TEMPLATES:
            if t.use_case != "vocal":
                continue
            if t.durations is None:
                continue
            has_held = any(d >= 1.5 for d in t.durations)
            assert has_held, f"{t.name}: vocal template has no note >= 1.5 beats"

    def test_select_templates_defaults_to_vocal(self):
        from white_generation.patterns.melody_patterns import (
            ALL_TEMPLATES,
            select_templates,
        )

        results = select_templates(ALL_TEMPLATES, (4, 4), "medium")
        for t in results:
            assert (
                t.use_case == "vocal"
            ), f"select_templates returned lead template: {t.name}"

    def test_select_templates_lead_use_case(self):
        from white_generation.patterns.melody_patterns import (
            ALL_TEMPLATES,
            select_templates,
        )

        results = select_templates(ALL_TEMPLATES, (4, 4), "medium", use_case="lead")
        assert len(results) > 0
        for t in results:
            assert t.use_case == "lead", f"Expected lead template, got: {t.name}"

    def test_vocal_3_4_templates_present(self):
        from white_generation.patterns.melody_patterns import ALL_TEMPLATES

        vocal_3_4 = [
            t for t in ALL_TEMPLATES if t.use_case == "vocal" and t.time_sig == (3, 4)
        ]
        assert (
            len(vocal_3_4) >= 4
        ), f"Expected >= 4 vocal 3/4 templates, got {len(vocal_3_4)}"

    def test_vocal_6_8_templates_present(self):
        from white_generation.patterns.melody_patterns import ALL_TEMPLATES

        vocal_6_8 = [
            t for t in ALL_TEMPLATES if t.use_case == "vocal" and t.time_sig == (6, 8)
        ]
        assert (
            len(vocal_6_8) >= 4
        ), f"Expected >= 4 vocal 6/8 templates, got {len(vocal_6_8)}"

    def test_singability_dense_pattern_scores_lower(self):
        from white_generation.patterns.melody_patterns import (
            SINGERS,
            singability_score,
        )

        singer = SINGERS["gabriel"]
        # Sparse: 3 held notes with rests
        sparse = [(0.0, 55, 1.5), (2.0, 57, 1.5)]
        # Dense: 8 rapid notes, no rests
        dense = [(i * 0.5, 55 + i % 3, 0.4) for i in range(8)]
        sparse_score = singability_score(sparse, singer, (4, 4))
        dense_score = singability_score(dense, singer, (4, 4))
        assert (
            sparse_score > dense_score
        ), f"Sparse ({sparse_score:.3f}) should score higher than dense ({dense_score:.3f})"
