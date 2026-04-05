"""Tests for app/util/phrase_dynamics.py"""

from app.util.phrase_dynamics import (
    DynamicCurve,
    apply_dynamics_curve,
    infer_curve,
    parse_curve,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _note_on(tick, note, vel):
    return (tick, note, vel, True)


def _note_off(tick, note):
    return (tick, note, 0, False)


def _on_velocities(events):
    return [e[2] for e in events if e[3] and e[2] > 0]


# ---------------------------------------------------------------------------
# parse_curve
# ---------------------------------------------------------------------------


class TestParseCurve:
    def test_flat(self):
        assert parse_curve("flat") == DynamicCurve.FLAT

    def test_linear_cresc_aliases(self):
        assert parse_curve("linear_cresc") == DynamicCurve.LINEAR_CRESC
        assert parse_curve("crescendo") == DynamicCurve.LINEAR_CRESC
        assert parse_curve("cresc") == DynamicCurve.LINEAR_CRESC

    def test_linear_dim_aliases(self):
        assert parse_curve("linear_dim") == DynamicCurve.LINEAR_DIM
        assert parse_curve("diminuendo") == DynamicCurve.LINEAR_DIM
        assert parse_curve("dim") == DynamicCurve.LINEAR_DIM

    def test_swell(self):
        assert parse_curve("swell") == DynamicCurve.SWELL

    def test_case_insensitive(self):
        assert parse_curve("FLAT") == DynamicCurve.FLAT
        assert parse_curve("Linear_Cresc") == DynamicCurve.LINEAR_CRESC

    def test_unknown_returns_flat(self):
        assert parse_curve("mystery") == DynamicCurve.FLAT
        assert parse_curve("") == DynamicCurve.FLAT


# ---------------------------------------------------------------------------
# infer_curve
# ---------------------------------------------------------------------------


class TestInferCurve:
    def test_intro_gives_swell(self):
        assert infer_curve("intro") == DynamicCurve.SWELL

    def test_chorus_and_high_give_cresc(self):
        assert infer_curve("chorus") == DynamicCurve.LINEAR_CRESC
        assert infer_curve("high") == DynamicCurve.LINEAR_CRESC

    def test_outro_gives_dim(self):
        assert infer_curve("outro") == DynamicCurve.LINEAR_DIM

    def test_verse_and_bridge_give_flat(self):
        assert infer_curve("verse") == DynamicCurve.FLAT
        assert infer_curve("bridge") == DynamicCurve.FLAT
        assert infer_curve("medium") == DynamicCurve.FLAT
        assert infer_curve("low") == DynamicCurve.FLAT


# ---------------------------------------------------------------------------
# apply_dynamics_curve — FLAT
# ---------------------------------------------------------------------------


class TestFlatCurve:
    def test_flat_returns_same_velocities(self):
        events = [_note_on(0, 60, 90), _note_on(100, 62, 90), _note_on(200, 64, 90)]
        result = apply_dynamics_curve(events, DynamicCurve.FLAT, 60, 110)
        assert _on_velocities(result) == [90, 90, 90]

    def test_flat_does_not_mutate_input(self):
        events = [_note_on(0, 60, 90)]
        original = list(events)
        apply_dynamics_curve(events, DynamicCurve.FLAT, 60, 110)
        assert events == original


# ---------------------------------------------------------------------------
# apply_dynamics_curve — LINEAR_CRESC
# ---------------------------------------------------------------------------


class TestLinearCresc:
    def test_velocities_increase_monotonically(self):
        events = [_note_on(i * 100, 60, 90) for i in range(5)]
        result = apply_dynamics_curve(events, DynamicCurve.LINEAR_CRESC, 50, 110)
        vels = _on_velocities(result)
        assert vels == sorted(vels), f"Expected ascending velocities, got {vels}"

    def test_first_note_at_min_vel(self):
        events = [_note_on(0, 60, 90), _note_on(100, 62, 90), _note_on(200, 64, 90)]
        result = apply_dynamics_curve(events, DynamicCurve.LINEAR_CRESC, 50, 110)
        assert _on_velocities(result)[0] == 50

    def test_last_note_at_max_vel(self):
        events = [_note_on(0, 60, 90), _note_on(100, 62, 90), _note_on(200, 64, 90)]
        result = apply_dynamics_curve(events, DynamicCurve.LINEAR_CRESC, 50, 110)
        assert _on_velocities(result)[-1] == 110

    def test_clamp_respected(self):
        events = [_note_on(i * 10, 60, 90) for i in range(10)]
        result = apply_dynamics_curve(events, DynamicCurve.LINEAR_CRESC, 60, 110)
        for v in _on_velocities(result):
            assert 60 <= v <= 110


# ---------------------------------------------------------------------------
# apply_dynamics_curve — LINEAR_DIM
# ---------------------------------------------------------------------------


class TestLinearDim:
    def test_velocities_decrease_monotonically(self):
        events = [_note_on(i * 100, 60, 90) for i in range(5)]
        result = apply_dynamics_curve(events, DynamicCurve.LINEAR_DIM, 50, 110)
        vels = _on_velocities(result)
        assert vels == sorted(vels, reverse=True), f"Expected descending, got {vels}"

    def test_first_note_at_max_vel(self):
        events = [_note_on(0, 60, 90), _note_on(100, 62, 90), _note_on(200, 64, 90)]
        result = apply_dynamics_curve(events, DynamicCurve.LINEAR_DIM, 50, 110)
        assert _on_velocities(result)[0] == 110

    def test_last_note_at_min_vel(self):
        events = [_note_on(0, 60, 90), _note_on(100, 62, 90), _note_on(200, 64, 90)]
        result = apply_dynamics_curve(events, DynamicCurve.LINEAR_DIM, 50, 110)
        assert _on_velocities(result)[-1] == 50


# ---------------------------------------------------------------------------
# apply_dynamics_curve — SWELL
# ---------------------------------------------------------------------------


class TestSwell:
    def test_midpoint_louder_than_endpoints(self):
        events = [_note_on(i * 100, 60, 90) for i in range(9)]
        result = apply_dynamics_curve(events, DynamicCurve.SWELL, 50, 110)
        vels = _on_velocities(result)
        mid = len(vels) // 2
        assert (
            vels[mid] >= vels[0]
        ), f"Midpoint {vels[mid]} should be >= start {vels[0]}"
        assert (
            vels[mid] >= vels[-1]
        ), f"Midpoint {vels[mid]} should be >= end {vels[-1]}"

    def test_clamp_respected(self):
        events = [_note_on(i * 10, 60, 90) for i in range(20)]
        result = apply_dynamics_curve(events, DynamicCurve.SWELL, 45, 127)
        for v in _on_velocities(result):
            assert 45 <= v <= 127


# ---------------------------------------------------------------------------
# Note-off events pass through unchanged
# ---------------------------------------------------------------------------


class TestNoteOffPassthrough:
    def test_note_offs_unchanged(self):
        events = [
            _note_on(0, 60, 90),
            _note_off(100, 60),
            _note_on(200, 62, 90),
            _note_off(300, 62),
        ]
        for curve in DynamicCurve:
            result = apply_dynamics_curve(events, curve, 50, 110)
            offs = [(e[0], e[1], e[2]) for e in result if not e[3]]
            assert offs == [
                (100, 60, 0),
                (300, 62, 0),
            ], f"Curve {curve} mutated note-offs"

    def test_empty_events_list(self):
        for curve in DynamicCurve:
            result = apply_dynamics_curve([], curve, 50, 110)
            assert result == []

    def test_only_note_offs(self):
        events = [_note_off(100, 60), _note_off(200, 62)]
        for curve in DynamicCurve:
            result = apply_dynamics_curve(events, curve, 50, 110)
            assert result == events
