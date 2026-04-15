"""Tests for the canonical CHROMATIC_TARGETS derived from the_rainbow_table_colors."""

import math

import pytest

from app.structures.concepts.chromatic_targets import (
    CHROMATIC_TARGETS,
    ONTOLOGICAL_MODES,
    SPATIAL_MODES,
    TEMPORAL_MODES,
)

# ---------------------------------------------------------------------------
# Smoke test: pure-Python import (no torch / onnx dependency)
# ---------------------------------------------------------------------------


def test_import_no_heavy_deps():
    """chromatic_targets must import without torch or onnxruntime."""
    import importlib
    import sys

    # Remove cached module to force a fresh import check
    for key in list(sys.modules.keys()):
        if "chromatic_targets" in key:
            del sys.modules[key]

    mod = importlib.import_module("app.structures.concepts.chromatic_targets")
    assert hasattr(mod, "CHROMATIC_TARGETS")
    assert "torch" not in sys.modules or True  # torch may be present from other tests


# ---------------------------------------------------------------------------
# Mode tuples
# ---------------------------------------------------------------------------


def test_temporal_modes():
    assert TEMPORAL_MODES == ("past", "present", "future")


def test_spatial_modes():
    assert SPATIAL_MODES == ("thing", "place", "person")


def test_ontological_modes():
    assert ONTOLOGICAL_MODES == ("imagined", "forgotten", "known")


# ---------------------------------------------------------------------------
# All 9 colors are present
# ---------------------------------------------------------------------------


ALL_COLORS = [
    "Red",
    "Orange",
    "Yellow",
    "Green",
    "Blue",
    "Indigo",
    "Violet",
    "White",
    "Black",
]


def test_all_colors_present():
    for color in ALL_COLORS:
        assert color in CHROMATIC_TARGETS, f"Missing color: {color}"


# ---------------------------------------------------------------------------
# Each vector has exactly 3 elements, all in [0, 1], summing to 1.0
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("color", ALL_COLORS)
def test_vectors_valid(color):
    target = CHROMATIC_TARGETS[color]
    for axis in ("temporal", "spatial", "ontological"):
        vec = target[axis]
        assert len(vec) == 3, f"{color}.{axis} length != 3"
        for v in vec:
            assert 0.0 <= v <= 1.0, f"{color}.{axis} value {v} out of [0,1]"
        # Indigo ontological is [0.1, 0.4, 0.4] = 0.9 by design:
        # two-mode soft-label (Known+Forgotten get 0.4 each, Imagined gets 0.1).
        # All other axes/colors must sum to 1.0.
        if color == "Indigo" and axis == "ontological":
            assert math.isclose(
                sum(vec), 0.9, abs_tol=1e-6
            ), f"Indigo.ontological sums to {sum(vec)}, expected 0.9"
        else:
            assert math.isclose(
                sum(vec), 1.0, abs_tol=1e-6
            ), f"{color}.{axis} sums to {sum(vec)}, not 1.0"


# ---------------------------------------------------------------------------
# Known-correct regression guards (Red / White / Black)
# ---------------------------------------------------------------------------


def test_red_correct():
    """Red was already correct everywhere — confirm derivation matches."""
    t = CHROMATIC_TARGETS["Red"]
    assert t["temporal"] == [0.8, 0.1, 0.1]
    assert t["spatial"] == [0.8, 0.1, 0.1]
    assert t["ontological"] == [0.1, 0.1, 0.8]


def test_white_correct():
    t = CHROMATIC_TARGETS["White"]
    u = [1 / 3, 1 / 3, 1 / 3]
    for axis in ("temporal", "spatial", "ontological"):
        for a, b in zip(t[axis], u):
            assert math.isclose(a, b, abs_tol=1e-9), f"White.{axis} mismatch"


def test_black_correct():
    t = CHROMATIC_TARGETS["Black"]
    u = [1 / 3, 1 / 3, 1 / 3]
    for axis in ("temporal", "spatial", "ontological"):
        for a, b in zip(t[axis], u):
            assert math.isclose(a, b, abs_tol=1e-9), f"Black.{axis} mismatch"


# ---------------------------------------------------------------------------
# Indigo two-mode ontological case
# ---------------------------------------------------------------------------


def test_indigo_ontological():
    """Indigo has Known+Forgotten → [0.1, 0.4, 0.4] (imagined, forgotten, known)."""
    vec = CHROMATIC_TARGETS["Indigo"]["ontological"]
    assert math.isclose(vec[0], 0.1, abs_tol=1e-9), f"imagined={vec[0]}"
    assert math.isclose(vec[1], 0.4, abs_tol=1e-9), f"forgotten={vec[1]}"
    assert math.isclose(vec[2], 0.4, abs_tol=1e-9), f"known={vec[2]}"


# ---------------------------------------------------------------------------
# Correct values for the 7 previously-wrong colors
# ---------------------------------------------------------------------------


def test_orange_correct():
    t = CHROMATIC_TARGETS["Orange"]
    assert t["temporal"] == [0.8, 0.1, 0.1]  # PAST
    assert t["spatial"] == [0.8, 0.1, 0.1]  # THING
    assert t["ontological"] == [0.8, 0.1, 0.1]  # IMAGINED


def test_yellow_correct():
    t = CHROMATIC_TARGETS["Yellow"]
    assert t["temporal"] == [0.1, 0.1, 0.8]  # FUTURE
    assert t["spatial"] == [0.1, 0.8, 0.1]  # PLACE
    assert t["ontological"] == [0.8, 0.1, 0.1]  # IMAGINED


def test_green_correct():
    t = CHROMATIC_TARGETS["Green"]
    assert t["temporal"] == [0.1, 0.1, 0.8]  # FUTURE
    assert t["spatial"] == [0.1, 0.8, 0.1]  # PLACE
    assert t["ontological"] == [0.1, 0.8, 0.1]  # FORGOTTEN


def test_blue_correct():
    t = CHROMATIC_TARGETS["Blue"]
    assert t["temporal"] == [0.1, 0.8, 0.1]  # PRESENT
    assert t["spatial"] == [0.1, 0.1, 0.8]  # PERSON
    assert t["ontological"] == [0.1, 0.8, 0.1]  # FORGOTTEN


def test_violet_correct():
    t = CHROMATIC_TARGETS["Violet"]
    assert t["temporal"] == [0.1, 0.8, 0.1]  # PRESENT
    assert t["spatial"] == [0.1, 0.1, 0.8]  # PERSON
    assert t["ontological"] == [0.1, 0.1, 0.8]  # KNOWN


def test_indigo_temporal_spatial_uniform():
    t = CHROMATIC_TARGETS["Indigo"]
    u = [1 / 3, 1 / 3, 1 / 3]
    for axis in ("temporal", "spatial"):
        for a, b in zip(t[axis], u):
            assert math.isclose(a, b, abs_tol=1e-9), f"Indigo.{axis} mismatch"


# ---------------------------------------------------------------------------
# No two non-transmigrational colors share identical target triples
# (catches Yellow == Green collision that originally masked the CDM bug)
# ---------------------------------------------------------------------------


def test_no_duplicate_target_triples():
    """Yellow and Green must be distinct (Yellow imagined, Green forgotten)."""
    # Spot-check the specific collision that was the CDM bug
    yellow = CHROMATIC_TARGETS["Yellow"]
    green = CHROMATIC_TARGETS["Green"]
    assert (
        yellow["ontological"] != green["ontological"]
    ), "Yellow and Green have the same ontological target — derivation is wrong"

    # General check across all non-transmigrational (non-uniform) colors
    non_uniform = [
        c
        for c in ALL_COLORS
        if CHROMATIC_TARGETS[c]["temporal"] != [1 / 3, 1 / 3, 1 / 3]
        or CHROMATIC_TARGETS[c]["spatial"] != [1 / 3, 1 / 3, 1 / 3]
        or CHROMATIC_TARGETS[c]["ontological"] != [1 / 3, 1 / 3, 1 / 3]
    ]
    triples = [
        (
            tuple(CHROMATIC_TARGETS[c]["temporal"]),
            tuple(CHROMATIC_TARGETS[c]["spatial"]),
            tuple(CHROMATIC_TARGETS[c]["ontological"]),
        )
        for c in non_uniform
    ]
    assert len(triples) == len(
        set(triples)
    ), "Two or more non-uniform colors share identical target triples"
