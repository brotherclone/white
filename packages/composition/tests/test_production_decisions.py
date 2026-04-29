"""Tests for production_decisions.py."""

from __future__ import annotations

from pathlib import Path

import yaml
from white_composition.production_decisions import (
    DECISIONS_FILENAME,
    generate_decisions,
    write_decisions_file,
)

# ---------------------------------------------------------------------------
# Scaffold helpers
# ---------------------------------------------------------------------------


def _write_song_context(production_dir: Path, **overrides) -> None:
    defaults = {
        "schema_version": "1",
        "title": "Test Song",
        "color": "Orange",
        "key": "C major",
        "bpm": 120,
        "time_sig": "4/4",
        "singer": "gabriel",
        "thread": "thread-abc",
        "phases": {},
    }
    defaults.update(overrides)
    (production_dir / "song_context.yml").write_text(
        yaml.dump(defaults, default_flow_style=False, sort_keys=False)
    )


def _write_review(
    production_dir: Path,
    phase: str,
    candidates: list[dict],
    bpm: int = 120,
) -> None:
    phase_dir = production_dir / phase
    phase_dir.mkdir(parents=True, exist_ok=True)
    data = {
        "bpm": bpm,
        "time_sig": "4/4",
        "color": "Orange",
        "candidates": candidates,
    }
    (phase_dir / "review.yml").write_text(
        yaml.dump(data, default_flow_style=False, sort_keys=False)
    )


def _approved_candidate(
    label: str, chromatic_match: float = 0.6, theory_total: float = 0.5
):
    return {
        "id": f"chord_{label}",
        "label": label,
        "status": "approved",
        "scores": {
            "composite": 0.57,
            "theory_total": theory_total,
            "chromatic": {
                "temporal": {"past": 0.5, "present": 0.3, "future": 0.2},
                "spatial": {"thing": 0.5, "place": 0.3, "person": 0.2},
                "ontological": {"imagined": 0.4, "forgotten": 0.3, "known": 0.3},
                "confidence": 0.8,
                "match": chromatic_match,
            },
        },
    }


def _pending_candidate(label: str):
    return {
        "id": f"chord_{label}_b",
        "label": label + "_b",
        "status": "pending",
        "scores": {
            "composite": 0.3,
            "theory_total": 0.2,
            "chromatic": {
                "temporal": {"past": 0.3, "present": 0.4, "future": 0.3},
                "spatial": {"thing": 0.3, "place": 0.4, "person": 0.3},
                "ontological": {"imagined": 0.3, "forgotten": 0.4, "known": 0.3},
                "confidence": 0.4,
                "match": 0.3,
            },
        },
    }


def _write_bar_beat_arrangement(production_dir: Path) -> None:
    """Write a simple bar/beat format arrangement.txt with 2 sections."""
    lines = [
        "1 1 1 1\tverse\t1\t8 0 0 0",
        "1 1 1 1\tdrum_verse_01\t2\t8 0 0 0",
        "1 1 1 1\tbass_verse_01\t3\t8 0 0 0",
        "1 1 1 1\tmelody_verse\t4\t8 0 0 0",
        "9 1 1 1\tchorus\t1\t8 0 0 0",
        "9 1 1 1\tdrum_chorus_01\t2\t8 0 0 0",
        "9 1 1 1\tbass_chorus_01\t3\t8 0 0 0",
        "17 1 1 1\tverse\t1\t8 0 0 0",
        "17 1 1 1\tdrum_verse_01\t2\t8 0 0 0",
        "17 1 1 1\tbass_verse_01\t3\t8 0 0 0",
        "17 1 1 1\tmelody_verse_alt\t4\t8 0 0 0",
        "25 1 1 1\tchorus\t1\t8 0 0 0",
        "25 1 1 1\tdrum_chorus_02\t2\t8 0 0 0",
        "25 1 1 1\tmelody_chorus\t4\t8 0 0 0",
    ]
    (production_dir / "arrangement.txt").write_text("\n".join(lines))


def _write_mix_score(production_dir: Path) -> None:
    melody_dir = production_dir / "melody"
    melody_dir.mkdir(parents=True, exist_ok=True)
    data = {
        "temporal": {"past": 0.6, "present": 0.2, "future": 0.2},
        "spatial": {"thing": 0.5, "place": 0.3, "person": 0.2},
        "ontological": {"imagined": 0.1, "forgotten": 0.2, "known": 0.7},
        "confidence": 0.85,
        "chromatic_match": 0.72,
    }
    (melody_dir / "mix_score.yml").write_text(
        yaml.dump(data, default_flow_style=False, sort_keys=False)
    )


def _write_drift_report(production_dir: Path) -> None:
    data = {
        "overall_pitch_match": 0.82,
        "overall_rhythm_drift": 0.25,
        "total_lyric_edits": 7,
        "total_word_count": 120,
        "sections": [],
    }
    (production_dir / "drift_report.yml").write_text(
        yaml.dump(data, default_flow_style=False, sort_keys=False)
    )


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------


def test_identity_fields(tmp_path):
    _write_song_context(tmp_path, title="My Song", color="Red", key="F# minor", bpm=95)
    decisions = generate_decisions(tmp_path)
    ident = decisions["identity"]
    assert ident["title"] == "My Song"
    assert ident["color"] == "Red"
    assert ident["key"] == "F# minor"
    assert ident["bpm"] == 95
    assert ident["time_sig"] == "4/4"
    assert ident["singer"] == "gabriel"


# ---------------------------------------------------------------------------
# Phase decisions — full set
# ---------------------------------------------------------------------------


def test_phase_decisions_all_phases(tmp_path):
    _write_song_context(tmp_path)
    for phase in ("chords", "drums", "bass", "melody"):
        _write_review(
            tmp_path, phase, [_approved_candidate("a"), _pending_candidate("b")]
        )

    decisions = generate_decisions(tmp_path)
    pd = decisions["phase_decisions"]

    for phase in ("chords", "drums", "bass", "melody"):
        assert pd[phase] is not None, f"Expected decisions for {phase}"
        assert pd[phase]["candidates_generated"] == 2
        assert pd[phase]["approved_count"] == 1
        assert pd[phase]["approved_labels"] == ["a"]
        assert pd[phase]["mean_chromatic_score"] == 0.6
        assert pd[phase]["mean_theory_score"] == 0.5


def test_phase_decisions_multiple_approved(tmp_path):
    _write_song_context(tmp_path)
    candidates = [
        _approved_candidate("verse", chromatic_match=0.4, theory_total=0.3),
        _approved_candidate("chorus", chromatic_match=0.8, theory_total=0.7),
    ]
    _write_review(tmp_path, "chords", candidates)

    pd = generate_decisions(tmp_path)["phase_decisions"]["chords"]
    assert pd["approved_count"] == 2
    assert pd["approved_labels"] == ["verse", "chorus"]
    assert pd["mean_chromatic_score"] == round((0.4 + 0.8) / 2, 4)
    assert pd["mean_theory_score"] == round((0.3 + 0.7) / 2, 4)


def test_phase_decisions_missing_review(tmp_path):
    _write_song_context(tmp_path)
    # Write only chords review, leave others absent
    _write_review(tmp_path, "chords", [_approved_candidate("verse")])

    pd = generate_decisions(tmp_path)["phase_decisions"]
    assert pd["chords"] is not None
    assert pd["drums"] is None
    assert pd["bass"] is None
    assert pd["melody"] is None


def test_phase_decisions_no_approved(tmp_path):
    _write_song_context(tmp_path)
    _write_review(
        tmp_path, "chords", [_pending_candidate("verse"), _pending_candidate("chorus")]
    )

    pd = generate_decisions(tmp_path)["phase_decisions"]["chords"]
    assert pd["candidates_generated"] == 2
    assert pd["approved_count"] == 0
    assert pd["approved_labels"] == []
    assert pd["mean_chromatic_score"] is None
    assert pd["mean_theory_score"] is None


# ---------------------------------------------------------------------------
# Arrangement summary
# ---------------------------------------------------------------------------


def test_arrangement_summary_bar_beat(tmp_path):
    _write_song_context(tmp_path)
    _write_bar_beat_arrangement(tmp_path)

    decisions = generate_decisions(tmp_path)
    arr = decisions["arrangement_summary"]

    assert arr is not None
    assert arr["section_count"] == 2

    sections_by_name = {s["name"]: s for s in arr["sections"]}
    assert "verse" in sections_by_name
    assert "chorus" in sections_by_name

    verse = sections_by_name["verse"]
    assert verse["bars"] == 8
    assert verse["play_count"] == 2
    assert (
        verse["vocals"] is True
    )  # melody_verse and melody_verse_alt both contain "verse"

    chorus = sections_by_name["chorus"]
    assert chorus["bars"] == 8
    assert chorus["play_count"] == 2
    assert chorus["vocals"] is True  # melody_chorus contains "chorus"

    assert arr["total_bars"] == (8 * 2) + (8 * 2)  # 32
    assert arr["total_plays"] == 4


def test_arrangement_summary_no_vocals(tmp_path):
    _write_song_context(tmp_path)
    # Arrangement with no track 4 clips
    lines = [
        "1 1 1 1\tintro\t1\t4 0 0 0",
        "1 1 1 1\tdrum_intro\t2\t4 0 0 0",
        "1 1 1 1\tbass_intro\t3\t4 0 0 0",
    ]
    (tmp_path / "arrangement.txt").write_text("\n".join(lines))

    arr = generate_decisions(tmp_path)["arrangement_summary"]
    assert arr["sections"][0]["vocals"] is False


def test_arrangement_summary_absent(tmp_path):
    _write_song_context(tmp_path)
    # No arrangement.txt
    decisions = generate_decisions(tmp_path)
    assert decisions["arrangement_summary"] is None


# ---------------------------------------------------------------------------
# Mix scores
# ---------------------------------------------------------------------------


def test_mix_scores_present(tmp_path):
    _write_song_context(tmp_path)
    _write_mix_score(tmp_path)

    ms = generate_decisions(tmp_path)["mix_scores"]
    assert ms is not None
    assert ms["confidence"] == 0.85
    assert ms["chromatic_match"] == 0.72
    assert ms["temporal"]["mode"] == "past"
    assert ms["ontological"]["mode"] == "known"


def test_mix_scores_absent(tmp_path):
    _write_song_context(tmp_path)
    assert generate_decisions(tmp_path)["mix_scores"] is None


# ---------------------------------------------------------------------------
# Vocal drift
# ---------------------------------------------------------------------------


def test_vocal_drift_present(tmp_path):
    _write_song_context(tmp_path)
    _write_drift_report(tmp_path)

    vd = generate_decisions(tmp_path)["vocal_drift"]
    assert vd is not None
    assert vd["overall_pitch_match"] == 0.82
    assert vd["overall_rhythm_drift"] == 0.25
    assert vd["total_lyric_edits"] == 7


def test_vocal_drift_absent(tmp_path):
    _write_song_context(tmp_path)
    assert generate_decisions(tmp_path)["vocal_drift"] is None


# ---------------------------------------------------------------------------
# Fully populated production directory
# ---------------------------------------------------------------------------


def test_full_production_dir(tmp_path):
    _write_song_context(tmp_path)
    for phase in ("chords", "drums", "bass", "melody"):
        _write_review(tmp_path, phase, [_approved_candidate(f"{phase}_a")])
    _write_bar_beat_arrangement(tmp_path)
    _write_mix_score(tmp_path)
    _write_drift_report(tmp_path)

    decisions = generate_decisions(tmp_path)

    assert decisions["identity"]["title"] == "Test Song"
    for phase in ("chords", "drums", "bass", "melody"):
        assert decisions["phase_decisions"][phase]["approved_count"] == 1
    assert decisions["arrangement_summary"]["section_count"] == 2
    assert decisions["mix_scores"]["chromatic_match"] == 0.72
    assert decisions["vocal_drift"]["total_lyric_edits"] == 7


# ---------------------------------------------------------------------------
# write_decisions_file
# ---------------------------------------------------------------------------


def test_write_decisions_file(tmp_path):
    _write_song_context(tmp_path)
    decisions = generate_decisions(tmp_path)
    out = write_decisions_file(tmp_path, decisions)

    assert out == tmp_path / DECISIONS_FILENAME
    assert out.exists()

    with open(out) as f:
        loaded = yaml.safe_load(f)

    assert loaded["identity"]["title"] == "Test Song"
    assert "phase_decisions" in loaded
    assert "arrangement_summary" in loaded
    assert "mix_scores" in loaded
    assert "vocal_drift" in loaded


# ---------------------------------------------------------------------------
# Graceful partial — completely empty production dir
# ---------------------------------------------------------------------------


def test_completely_empty_production_dir(tmp_path):
    _write_song_context(tmp_path)
    decisions = generate_decisions(tmp_path)

    assert decisions["identity"]["title"] == "Test Song"
    for phase in ("chords", "drums", "bass", "melody"):
        assert decisions["phase_decisions"][phase] is None
    assert decisions["arrangement_summary"] is None
    assert decisions["mix_scores"] is None
    assert decisions["vocal_drift"] is None
