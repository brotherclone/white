"""Tests for the song evaluator (AirGigs readiness metric)."""

import statistics
from pathlib import Path

import pytest
import yaml

from app.generators.midi.production_plan import PlanSection, ProductionPlan, save_plan
from unittest.mock import MagicMock, patch

from app.generators.midi.song_evaluator import (
    EVALUATION_FILENAME,
    PhaseReport,
    EvaluationReport,
    _chromatic_consistency,
    _collect_flags,
    _compute_arrangement_metrics,
    _compute_composite,
    _compute_lyric_metrics,
    _compute_structural_integrity,
    _count_syllables,
    _rescore_lyrics,
    _determine_readiness,
    _load_phase_report,
    evaluate,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_chord_candidate(
    id: str,
    label,
    status: str = "approved",
    melody: float = 1.0,
    voice_leading: float = 0.9,
    variety: float = 1.0,
    graph_probability: float = 0.0,
    match: float = 0.18,
    confidence: float = 0.14,
) -> dict:
    return {
        "id": id,
        "label": label,
        "status": status,
        "scores": {
            "theory": {
                "melody": melody,
                "voice_leading": voice_leading,
                "variety": variety,
                "graph_probability": graph_probability,
            },
            "chromatic": {"match": match, "confidence": confidence},
        },
    }


def _make_drum_candidate(
    id: str,
    label,
    section: str,
    status: str = "approved",
    energy: float = 1.0,
    match: float = 0.17,
    confidence: float = 0.14,
) -> dict:
    return {
        "id": id,
        "label": label,
        "section": section,
        "status": status,
        "scores": {
            "energy_appropriateness": energy,
            "chromatic": {"match": match, "confidence": confidence},
        },
    }


def _make_bass_candidate(
    id: str,
    label,
    section: str,
    status: str = "approved",
    root: float = 1.0,
    vl: float = 0.9,
    kick: float = 1.0,
    match: float = 0.17,
    confidence: float = 0.14,
) -> dict:
    return {
        "id": id,
        "label": label,
        "section": section,
        "status": status,
        "scores": {
            "theory": {
                "root_adherence": root,
                "voice_leading": vl,
                "kick_alignment": kick,
            },
            "chromatic": {"match": match, "confidence": confidence},
        },
    }


def _make_melody_candidate(
    id: str,
    label,
    section: str,
    status: str = "approved",
    singability: float = 0.7,
    alignment: float = 1.0,
    contour: float = 0.8,
    match: float = 0.17,
    confidence: float = 0.14,
) -> dict:
    return {
        "id": id,
        "label": label,
        "section": section,
        "status": status,
        "scores": {
            "theory": {
                "singability": singability,
                "chord_tone_alignment": alignment,
                "contour_quality": contour,
            },
            "chromatic": {"match": match, "confidence": confidence},
        },
    }


def _write_review(phase_dir: Path, data: dict) -> None:
    phase_dir.mkdir(parents=True, exist_ok=True)
    (phase_dir / "review.yml").write_text(yaml.dump(data))


def _make_plan(
    tmp_path: Path,
    sections: list,
    slug: str = "test_song",
    color: str = "green",
    title: str = "Test Song",
) -> ProductionPlan:
    plan = ProductionPlan(
        song_slug=slug,
        generated="2026-01-01T00:00:00+00:00",
        bpm=120,
        time_sig="4/4",
        key="C major",
        color=color,
        title=title,
        sections=sections,
    )
    save_plan(plan, tmp_path)
    return plan


# ---------------------------------------------------------------------------
# Phase report: missing file
# ---------------------------------------------------------------------------


def test_phase_report_returns_none_when_missing(tmp_path):
    result = _load_phase_report(tmp_path, "chords")
    assert result is None


# ---------------------------------------------------------------------------
# Phase report: chords
# ---------------------------------------------------------------------------


def test_phase_report_chords_basic(tmp_path):
    candidates = [
        _make_chord_candidate(
            "c1", "verse", melody=1.0, voice_leading=0.9, variety=1.0
        ),
        _make_chord_candidate(
            "c2", "chorus", melody=0.8, voice_leading=0.8, variety=0.8
        ),
        _make_chord_candidate("c3", None, status="rejected"),
    ]
    _write_review(tmp_path / "chords", {"candidates": candidates})

    report = _load_phase_report(tmp_path, "chords")

    assert report is not None
    assert report.phase == "chords"
    assert report.complete is True
    assert report.sections_found == 2
    assert report.sections_approved == 2
    assert report.coverage == 1.0
    # theory: mean of (1.0+0.9+1.0)/3 and (0.8+0.8+0.8)/3
    expected_theory = statistics.mean([(1.0 + 0.9 + 1.0) / 3, (0.8 + 0.8 + 0.8) / 3])
    assert abs(report.mean_theory - expected_theory) < 1e-6
    assert abs(report.mean_chromatic_match - 0.18) < 1e-6
    assert report.approved_labels == ["verse", "chorus"]


def test_phase_report_chords_excludes_graph_probability(tmp_path):
    """graph_probability is NOT included in chord theory score."""
    c = _make_chord_candidate(
        "c1", "intro", melody=1.0, voice_leading=1.0, variety=1.0, graph_probability=0.0
    )
    _write_review(tmp_path / "chords", {"candidates": [c]})

    report = _load_phase_report(tmp_path, "chords")
    # Should be (1+1+1)/3 = 1.0, not influenced by graph_probability=0
    assert abs(report.mean_theory - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# Phase report: drums
# ---------------------------------------------------------------------------


def test_phase_report_drums_basic(tmp_path):
    candidates = [
        _make_drum_candidate("d1", "drums_verse", "verse", energy=1.0),
        _make_drum_candidate("d2", "drums_chorus", "chorus", energy=0.8),
    ]
    _write_review(
        tmp_path / "drums",
        {"sections_found": ["verse", "chorus"], "candidates": candidates},
    )

    report = _load_phase_report(tmp_path, "drums")

    assert report is not None
    assert report.sections_found == 2
    assert report.sections_approved == 2
    assert abs(report.mean_theory - 0.9) < 1e-6  # mean of 1.0 and 0.8


def test_phase_report_drums_uses_energy_appropriateness(tmp_path):
    """Drum theory score comes from energy_appropriateness, not a theory sub-dict."""
    c = _make_drum_candidate("d1", "drums_verse", "verse", energy=0.75)
    _write_review(
        tmp_path / "drums",
        {"sections_found": ["verse"], "candidates": [c]},
    )
    report = _load_phase_report(tmp_path, "drums")
    assert abs(report.mean_theory - 0.75) < 1e-6


# ---------------------------------------------------------------------------
# Phase report: bass
# ---------------------------------------------------------------------------


def test_phase_report_bass_basic(tmp_path):
    candidates = [
        _make_bass_candidate("b1", "bass_verse", "verse", root=1.0, vl=0.9, kick=0.8),
    ]
    _write_review(
        tmp_path / "bass",
        {"sections_found": ["verse"], "candidates": candidates},
    )
    report = _load_phase_report(tmp_path, "bass")

    assert report is not None
    expected = (1.0 + 0.9 + 0.8) / 3
    assert abs(report.mean_theory - expected) < 1e-6


def test_phase_report_bass_uses_correct_fields(tmp_path):
    """Bass theory uses root_adherence, voice_leading, kick_alignment."""
    c = _make_bass_candidate("b1", "bass_v", "verse", root=0.6, vl=0.7, kick=0.8)
    _write_review(
        tmp_path / "bass",
        {"sections_found": ["verse"], "candidates": [c]},
    )
    report = _load_phase_report(tmp_path, "bass")
    assert abs(report.mean_theory - (0.6 + 0.7 + 0.8) / 3) < 1e-6


# ---------------------------------------------------------------------------
# Phase report: melody
# ---------------------------------------------------------------------------


def test_phase_report_melody_uses_correct_fields(tmp_path):
    """Melody theory uses singability, chord_tone_alignment, contour_quality."""
    c = _make_melody_candidate(
        "m1", "melody_v", "verse", singability=0.5, alignment=1.0, contour=0.9
    )
    _write_review(
        tmp_path / "melody",
        {"sections_found": ["verse"], "candidates": [c]},
    )
    report = _load_phase_report(tmp_path, "melody")
    assert abs(report.mean_theory - (0.5 + 1.0 + 0.9) / 3) < 1e-6


# ---------------------------------------------------------------------------
# Phase report: coverage
# ---------------------------------------------------------------------------


def test_phase_report_coverage_partial(tmp_path):
    """Coverage < 1.0 when some sections have no approved candidate."""
    candidates = [
        _make_drum_candidate("d1", "drums_verse", "verse"),
        # chorus: no approved candidate
    ]
    _write_review(
        tmp_path / "drums",
        {"sections_found": ["verse", "chorus"], "candidates": candidates},
    )
    report = _load_phase_report(tmp_path, "drums")
    assert report.sections_found == 2
    assert report.sections_approved == 1
    assert abs(report.coverage - 0.5) < 1e-6


# ---------------------------------------------------------------------------
# Chromatic consistency
# ---------------------------------------------------------------------------


def test_chromatic_consistency_single_candidate():
    assert _chromatic_consistency([0.5]) == 1.0


def test_chromatic_consistency_empty():
    assert _chromatic_consistency([]) == 1.0


def test_chromatic_consistency_uniform():
    scores = [0.168, 0.168, 0.168, 0.168]
    result = _chromatic_consistency(scores)
    assert abs(result - 1.0) < 1e-6


def test_chromatic_consistency_varied():
    scores = [0.1, 0.2, 0.3, 0.4]
    expected = 1.0 - statistics.stdev(scores)
    assert abs(_chromatic_consistency(scores) - expected) < 1e-6


# ---------------------------------------------------------------------------
# Arrangement metrics
# ---------------------------------------------------------------------------


def test_arrangement_total_bars_counts_repeat(tmp_path):
    _make_plan(
        tmp_path,
        [
            PlanSection(name="verse", bars=4, repeat=2),
            PlanSection(name="chorus", bars=8, repeat=1),
        ],
    )
    arr = _compute_arrangement_metrics(tmp_path)
    assert arr["total_bars"] == 4 * 2 + 8 * 1  # 16


def test_arrangement_section_variety(tmp_path):
    _make_plan(
        tmp_path,
        [
            PlanSection(name="intro", bars=4, repeat=1),
            PlanSection(name="verse", bars=4, repeat=1),
            PlanSection(name="verse", bars=4, repeat=1),
            PlanSection(name="outro", bars=4, repeat=1),
        ],
    )
    arr = _compute_arrangement_metrics(tmp_path)
    assert arr["unique_sections"] == 3  # intro, verse, outro
    assert abs(arr["section_variety"] - 3 / 4) < 1e-6


def test_arrangement_vocal_coverage(tmp_path):
    _make_plan(
        tmp_path,
        [
            PlanSection(name="intro", bars=4, repeat=1, vocals=False),
            PlanSection(name="verse", bars=4, repeat=2, vocals=True),
        ],
    )
    arr = _compute_arrangement_metrics(tmp_path)
    # total=12, vocal=8
    assert arr["total_bars"] == 12
    assert arr["vocal_bars"] == 8
    assert abs(arr["vocal_coverage"] - 8 / 12) < 1e-6


def test_arrangement_no_plan(tmp_path):
    arr = _compute_arrangement_metrics(tmp_path)
    assert arr["total_bars"] == 0
    assert arr["vocal_coverage"] == 0.0


# ---------------------------------------------------------------------------
# Lyric metrics
# ---------------------------------------------------------------------------


def test_lyric_density_no_file(tmp_path):
    has_lyrics, density = _compute_lyric_metrics(tmp_path, 10)
    assert has_lyrics is False
    assert density == 0.0


def test_lyric_density_basic(tmp_path):
    melody_dir = tmp_path / "melody"
    melody_dir.mkdir()
    # 4 one-syllable words, 4 vocal bars → density = 1.0
    (melody_dir / "lyrics.txt").write_text("# comment\n[verse]\nshe\nknew\neach\nrow\n")

    has_lyrics, density = _compute_lyric_metrics(tmp_path, 4)
    assert has_lyrics is True
    assert density == 1.0


def test_lyric_density_skips_headers_and_comments(tmp_path):
    melody_dir = tmp_path / "melody"
    melody_dir.mkdir()
    text = "# title\n\n[section]\nword one\n"
    (melody_dir / "lyrics.txt").write_text(text)

    has_lyrics, density = _compute_lyric_metrics(tmp_path, 2)
    assert has_lyrics is True
    # "word one" = 2 syllables, 2 bars → density = 1.0
    assert abs(density - 1.0) < 1e-6


def test_lyric_density_zero_vocal_bars(tmp_path):
    melody_dir = tmp_path / "melody"
    melody_dir.mkdir()
    (melody_dir / "lyrics.txt").write_text("hello\n")
    has_lyrics, density = _compute_lyric_metrics(tmp_path, 0)
    assert has_lyrics is True
    assert density == 0.0


# ---------------------------------------------------------------------------
# Syllable counting
# ---------------------------------------------------------------------------


def test_count_syllables_single_words():
    assert _count_syllables("she") == 1
    assert _count_syllables("knew") == 1
    assert _count_syllables("morning") == 2
    assert _count_syllables("angle") == 2
    assert _count_syllables("absence") == 2


def test_count_syllables_hyphenated_word():
    # "rust-patch" = rust(1) + patch(1) = 2
    assert _count_syllables("rust-patch") == 2


def test_count_syllables_dense_line():
    # "the rust-patch caught the last morning" → 1+1+1+1+1+1+2 = 8
    assert _count_syllables("the rust-patch caught the last morning") == 8


# ---------------------------------------------------------------------------
# Structural integrity
# ---------------------------------------------------------------------------


def test_structural_integrity_absent(tmp_path):
    integrity, max_drift, mismatches = _compute_structural_integrity(tmp_path)
    assert integrity == 1.0
    assert max_drift == 0.0
    assert mismatches == 0


def test_structural_integrity_no_drift_no_mismatch(tmp_path):
    sections = [
        {"drift_seconds": 0.0, "name_mismatch": False},
        {"drift_seconds": 0.0, "name_mismatch": False},
    ]
    (tmp_path / "drift_report.yml").write_text(yaml.dump({"sections": sections}))
    integrity, max_drift, mismatches = _compute_structural_integrity(tmp_path)
    assert abs(integrity - 1.0) < 1e-6
    assert max_drift == 0.0
    assert mismatches == 0


def test_structural_integrity_large_drift(tmp_path):
    # 5s drift → drift_score = 1 - 5/120 ≈ 0.958; mismatch_score = 1.0
    sections = [{"drift_seconds": 5.0, "name_mismatch": False}]
    (tmp_path / "drift_report.yml").write_text(yaml.dump({"sections": sections}))
    integrity, max_drift, mismatches = _compute_structural_integrity(tmp_path)
    assert max_drift == 5.0
    expected = 1.0 * 0.8 + (1.0 - 5.0 / 120.0) * 0.2
    assert abs(integrity - expected) < 1e-6


def test_structural_integrity_mismatch(tmp_path):
    sections = [
        {"drift_seconds": 0.0, "name_mismatch": True},
        {"drift_seconds": 0.0, "name_mismatch": False},
    ]
    (tmp_path / "drift_report.yml").write_text(yaml.dump({"sections": sections}))
    integrity, max_drift, mismatches = _compute_structural_integrity(tmp_path)
    assert mismatches == 1
    # mismatch_score = 1 - 1/2 = 0.5; drift_score = 1.0
    expected = 0.5 * 0.8 + 1.0 * 0.2
    assert abs(integrity - expected) < 1e-6


# ---------------------------------------------------------------------------
# Composite score
# ---------------------------------------------------------------------------


def test_composite_score_weights():
    result = _compute_composite(
        chromatic_alignment=1.0,
        theory_quality=0.0,
        production_completeness=0.0,
        structural_integrity=0.0,
        lyric_maturity=0.0,
    )
    assert abs(result - 0.40) < 1e-6

    result = _compute_composite(0.0, 1.0, 0.0, 0.0, 0.0)
    assert abs(result - 0.25) < 1e-6

    result = _compute_composite(0.0, 0.0, 1.0, 0.0, 0.0)
    assert abs(result - 0.20) < 1e-6

    result = _compute_composite(0.0, 0.0, 0.0, 1.0, 0.0)
    assert abs(result - 0.10) < 1e-6

    result = _compute_composite(0.0, 0.0, 0.0, 0.0, 1.0)
    assert abs(result - 0.05) < 1e-6

    # All ones → 1.0
    result = _compute_composite(1.0, 1.0, 1.0, 1.0, 1.0)
    assert abs(result - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# Readiness thresholds
# ---------------------------------------------------------------------------


def test_readiness_ready():
    assert _determine_readiness(0.75) == "ready"
    assert _determine_readiness(1.0) == "ready"


def test_readiness_demo_boundary():
    assert _determine_readiness(0.7499) == "demo"


def test_readiness_demo():
    assert _determine_readiness(0.55) == "demo"
    assert _determine_readiness(0.65) == "demo"


def test_readiness_draft():
    assert _determine_readiness(0.54) == "draft"
    assert _determine_readiness(0.0) == "draft"


def test_readiness_demo_lower_boundary():
    assert _determine_readiness(0.5499) == "draft"


# ---------------------------------------------------------------------------
# Flags
# ---------------------------------------------------------------------------


def _minimal_report(**overrides) -> EvaluationReport:
    """Create a minimal EvaluationReport for flag testing."""
    defaults = dict(
        song_slug="test",
        color="green",
        title="Test",
        evaluated="2026-01-01T00:00:00+00:00",
        phases={
            "chords": PhaseReport("chords", True, 2, 2, 1.0, 0.85, 0.17, 0.14, 1.0, []),
            "drums": PhaseReport("drums", True, 2, 2, 1.0, 1.0, 0.17, 0.14, 1.0, []),
            "bass": PhaseReport("bass", True, 2, 2, 1.0, 0.85, 0.17, 0.14, 1.0, []),
            "melody": PhaseReport("melody", True, 2, 2, 1.0, 0.80, 0.17, 0.14, 1.0, []),
        },
        phases_complete=4,
        total_bars=40,
        unique_sections=4,
        section_variety=1.0,
        vocal_coverage=0.5,
        has_lyrics=True,
        lyric_syllable_density=1.0,
        max_drift_seconds=0.0,
        name_mismatches=0,
        structural_integrity=1.0,
        chromatic_alignment=0.17,
        theory_quality=0.87,
        production_completeness=1.0,
        lyric_maturity=1.0,
        composite_score=0.60,
        airgigs_readiness="demo",
        flags=[],
    )
    defaults.update(overrides)
    return EvaluationReport(**defaults)


def test_flag_incomplete_phase():
    report = _minimal_report(
        phases={
            "chords": PhaseReport("chords", True, 2, 2, 1.0, 0.85, 0.17, 0.14, 1.0, [])
        }
    )
    flags = _collect_flags(report)
    assert "incomplete: missing drums" in flags
    assert "incomplete: missing bass" in flags
    assert "incomplete: missing melody" in flags


def test_flag_sparse_lyrics():
    report = _minimal_report(has_lyrics=True, lyric_syllable_density=0.3)
    flags = _collect_flags(report)
    assert "sparse lyrics" in flags


def test_no_flag_sparse_lyrics_when_no_lyrics():
    report = _minimal_report(has_lyrics=False, lyric_syllable_density=0.0)
    flags = _collect_flags(report)
    assert "sparse lyrics" not in flags


def test_flag_timing_drift():
    report = _minimal_report(max_drift_seconds=3.5)
    flags = _collect_flags(report)
    assert "timing drift 3.5s" in flags


def test_no_flag_timing_drift_below_threshold():
    report = _minimal_report(max_drift_seconds=1.9)
    flags = _collect_flags(report)
    assert not any("timing drift" in f for f in flags)


def test_flag_arrangement_mismatch():
    report = _minimal_report(name_mismatches=2)
    flags = _collect_flags(report)
    assert "arrangement mismatch (2 sections)" in flags


def test_flag_strong_chromatic_alignment():
    report = _minimal_report(chromatic_alignment=0.85)
    flags = _collect_flags(report)
    assert "strong chromatic alignment" in flags


def test_flag_high_theory_quality():
    report = _minimal_report(theory_quality=0.90)
    flags = _collect_flags(report)
    assert "high theory quality" in flags


def test_flag_low_confidence():
    phases = {
        "chords": PhaseReport("chords", True, 2, 2, 1.0, 0.85, 0.17, 0.10, 1.0, []),
    }
    report = _minimal_report(phases=phases)
    flags = _collect_flags(report)
    assert "low chromatic confidence" in flags


# ---------------------------------------------------------------------------
# Integration test: green song
# ---------------------------------------------------------------------------

GREEN_SONG_DIR = Path(
    "shrinkwrapped/white-the-breathing-machine-learns-to-sing/"
    "production/green__last_pollinators_elegy_v1"
)


@pytest.fixture
def green_production_dir():
    """Resolve green song dir relative to repo root."""
    repo_root = Path(__file__).parent.parent.parent.parent
    d = repo_root / GREEN_SONG_DIR
    if not d.exists():
        pytest.skip(f"Green song production dir not found: {d}")
    return d


@pytest.fixture(autouse=False)
def cleanup_evaluation(green_production_dir):
    """Remove song_evaluation.yml after test."""
    yield
    out = green_production_dir / EVALUATION_FILENAME
    if out.exists():
        out.unlink()


def test_evaluate_green_song(green_production_dir, cleanup_evaluation):
    report = evaluate(green_production_dir)

    # All 4 phases should be present and complete
    assert report.phases_complete == 4
    assert set(report.phases.keys()) == {"chords", "drums", "bass", "melody"}
    for phase, pr in report.phases.items():
        assert pr.complete, f"Phase {phase} should be complete"

    # Arrangement
    assert report.total_bars > 0
    assert report.unique_sections > 0
    assert 0.0 < report.vocal_coverage < 1.0

    # Lyrics
    assert report.has_lyrics is True
    assert report.lyric_syllable_density > 0.0

    # Chromatic alignment in expected range
    assert 0.10 <= report.chromatic_alignment <= 0.30

    # Readiness is a valid value
    assert report.airgigs_readiness in ("draft", "demo", "ready")

    # song_evaluation.yml was written
    out = green_production_dir / EVALUATION_FILENAME
    assert out.exists()

    # Verify YAML round-trip
    with open(out) as f:
        data = yaml.safe_load(f)
    assert data["song_slug"] == "green__last_pollinators_elegy_v1"
    assert data["phases_complete"] == 4
    assert data["has_lyrics"] is True
    assert "composite_score" in data
    assert "airgigs_readiness" in data


# ---------------------------------------------------------------------------
# _rescore_lyrics tests
# ---------------------------------------------------------------------------


def _make_mock_scorer(edited_match=0.75, draft_match=0.60):
    """Build a mock ChromaticScorer for lyric text scoring."""
    mock = MagicMock()
    import numpy as np

    mock.prepare_concept.return_value = np.zeros(768, dtype=np.float32)

    def _score_batch(candidates, concept_emb=None):
        return [
            {
                "temporal": {"past": 0.7, "present": 0.2, "future": 0.1},
                "spatial": {"thing": 0.6, "place": 0.3, "person": 0.1},
                "ontological": {"imagined": 0.1, "forgotten": 0.1, "known": 0.8},
                "confidence": 0.05,
                "match": edited_match if i == 0 else draft_match,
            }
            for i, _ in enumerate(candidates)
        ]

    mock.score_batch.side_effect = _score_batch
    return mock


class TestRescoreLyrics:
    def _make_prod_dir(self, tmp_path, lyrics_text=None, draft_text=None):
        from app.generators.midi.production_plan import (
            PlanSection,
            ProductionPlan,
            save_plan,
        )

        prod_dir = tmp_path / "production" / "test_song"
        melody_dir = prod_dir / "melody"
        melody_dir.mkdir(parents=True)

        if lyrics_text is not None:
            (melody_dir / "lyrics.txt").write_text(lyrics_text)
        if draft_text is not None:
            (melody_dir / "lyrics_draft.txt").write_text(draft_text)

        plan = ProductionPlan(
            song_slug="test_song",
            generated="2026-01-01T00:00:00+00:00",
            bpm=120,
            time_sig="4/4",
            key="C major",
            color="Red",
            title="Test Song",
            concept="a red concept",
            sections=[PlanSection(name="verse", bars=4, repeat=1, vocals=True)],
        )
        save_plan(plan, prod_dir)
        return prod_dir

    def test_rescore_lyrics_happy_path(self, tmp_path):
        """Both lyrics.txt and lyrics_draft.txt scored; delta computed."""
        prod_dir = self._make_prod_dir(
            tmp_path,
            lyrics_text="the edited line\nand another",
            draft_text="the original draft\nfirst version",
        )
        mock_scorer = _make_mock_scorer(edited_match=0.75, draft_match=0.60)

        with patch(
            "training.chromatic_scorer.ChromaticScorer", return_value=mock_scorer
        ):
            result = _rescore_lyrics(prod_dir, "a red concept", "Red")

        assert "lyrics_edited_chromatic_match" in result
        assert "lyrics_draft_chromatic_match" in result
        assert "lyrics_chromatic_delta" in result
        assert result["lyrics_chromatic_delta"] == round(
            result["lyrics_edited_chromatic_match"]
            - result["lyrics_draft_chromatic_match"],
            4,
        )

    def test_rescore_lyrics_missing_draft(self, tmp_path):
        """Missing draft → only edited match written, no error."""
        prod_dir = self._make_prod_dir(tmp_path, lyrics_text="the edited line")
        mock_scorer = _make_mock_scorer(edited_match=0.70)

        with patch(
            "training.chromatic_scorer.ChromaticScorer", return_value=mock_scorer
        ):
            result = _rescore_lyrics(prod_dir, "a red concept", "Red")

        assert "lyrics_edited_chromatic_match" in result
        assert "lyrics_draft_chromatic_match" not in result
        assert "lyrics_chromatic_delta" not in result

    def test_rescore_lyrics_missing_lyrics_txt(self, tmp_path):
        """Missing lyrics.txt → empty dict returned, no crash."""
        prod_dir = self._make_prod_dir(tmp_path)  # no lyrics files

        with patch("training.chromatic_scorer.ChromaticScorer"):
            result = _rescore_lyrics(prod_dir, "a red concept", "Red")

        assert result == {}

    def test_rescore_lyrics_merges_into_existing_yml(self, tmp_path):
        """Existing song_evaluation.yml fields are preserved after merge."""
        prod_dir = self._make_prod_dir(
            tmp_path,
            lyrics_text="edited",
            draft_text="draft",
        )
        eval_path = prod_dir / EVALUATION_FILENAME
        existing = {
            "composite_score": 0.72,
            "airgigs_readiness": "demo",
            "phases_complete": 4,
        }
        with open(eval_path, "w") as f:
            yaml.dump(existing, f)

        mock_scorer = _make_mock_scorer(edited_match=0.68, draft_match=0.55)
        with patch(
            "training.chromatic_scorer.ChromaticScorer", return_value=mock_scorer
        ):
            lyric_scores = _rescore_lyrics(prod_dir, "a red concept", "Red")

        # Simulate the CLI merge
        with open(eval_path) as f:
            data = yaml.safe_load(f)
        data.update(lyric_scores)
        with open(eval_path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

        with open(eval_path) as f:
            merged = yaml.safe_load(f)

        assert merged["composite_score"] == 0.72  # preserved
        assert merged["airgigs_readiness"] == "demo"  # preserved
        assert "lyrics_edited_chromatic_match" in merged
        assert "lyrics_draft_chromatic_match" in merged
