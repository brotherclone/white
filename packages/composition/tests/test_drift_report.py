"""Tests for production plan drift report."""

from unittest.mock import patch

import pytest
import yaml

from white_composition.drift_report import (
    REPORT_FILENAME,
    BarDelta,
    DriftReport,
    DriftSummary,
    _arc_correlation,
    _build_bar_deltas,
    _build_drift_summary,
    _expand_proposed,
    _interpolate_to_n,
    _normalize_label,
    _pearson_r,
    compare_plans,
    load_report,
    write_report,
)
from white_composition.production_plan import PlanSection, ProductionPlan

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_plan(sections: list[dict], **kwargs) -> ProductionPlan:
    plan_sections = [
        PlanSection(
            name=s["name"],
            bars=s.get("bars", 8),
            play_count=s.get("play_count", 1),
            arc=s.get("arc", 0.35),
        )
        for s in sections
    ]
    return ProductionPlan(
        song_slug="test_song",
        generated="2026-01-01T00:00:00Z",
        bpm=120,
        time_sig="4/4",
        key="C minor",
        color="Black",
        title="Test Song",
        sections=plan_sections,
        **kwargs,
    )


def _make_arrangement_txt(instances: list[dict], bpm: int = 120) -> str:
    """Generate a bar/beat format arrangement.txt with one clip per instance.

    Each instance dict: {name, track, bars}. Track 1 = chords, track 4 = melody.
    """
    lines = []
    bar = 1
    for inst in instances:
        s_bar = bar
        e_bar = bar + inst["bars"]
        # Format: "1 1 1 1\tname\ttrack\t5 1 1 1"
        start = f"{s_bar} 1 1 1"
        end = f"{e_bar} 1 1 1"
        lines.append(f"{start}\t{inst['name']}\t{inst['track']}\t{end}")
        bar = e_bar
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# _normalize_label
# ---------------------------------------------------------------------------


def test_normalize_label_plain():
    assert _normalize_label("Chorus") == "chorus"


def test_normalize_label_strips_numeric_suffix():
    assert _normalize_label("chorus_02") == "chorus"
    assert _normalize_label("verse_2") == "verse"


def test_normalize_label_strips_v_prefix_suffix():
    assert _normalize_label("bridge_v3") == "bridge"


def test_normalize_label_hyphen_to_underscore():
    assert _normalize_label("pre-chorus") == "pre_chorus"


def test_normalize_label_preserves_multi_word():
    assert _normalize_label("verse_alt") == "verse_alt"


# ---------------------------------------------------------------------------
# _pearson_r
# ---------------------------------------------------------------------------


def test_pearson_r_perfect_positive():
    xs = [1.0, 2.0, 3.0, 4.0]
    ys = [2.0, 4.0, 6.0, 8.0]
    r = _pearson_r(xs, ys)
    assert r == pytest.approx(1.0, abs=1e-6)


def test_pearson_r_perfect_negative():
    xs = [1.0, 2.0, 3.0]
    ys = [3.0, 2.0, 1.0]
    r = _pearson_r(xs, ys)
    assert r == pytest.approx(-1.0, abs=1e-6)


def test_pearson_r_constant_returns_none():
    xs = [1.0, 1.0, 1.0]
    ys = [2.0, 3.0, 4.0]
    assert _pearson_r(xs, ys) is None


def test_pearson_r_too_short():
    assert _pearson_r([1.0], [1.0]) is None
    assert _pearson_r([], []) is None


def test_pearson_r_mismatched_lengths():
    assert _pearson_r([1.0, 2.0], [1.0]) is None


# ---------------------------------------------------------------------------
# _interpolate_to_n
# ---------------------------------------------------------------------------


def test_interpolate_same_length():
    vals = [0.1, 0.5, 0.9]
    assert _interpolate_to_n(vals, 3) == vals


def test_interpolate_upsample_endpoints_preserved():
    vals = [0.0, 1.0]
    result = _interpolate_to_n(vals, 5)
    assert result[0] == pytest.approx(0.0)
    assert result[-1] == pytest.approx(1.0)
    assert len(result) == 5


def test_interpolate_downsample():
    vals = [0.0, 0.25, 0.5, 0.75, 1.0]
    result = _interpolate_to_n(vals, 3)
    assert len(result) == 3
    assert result[0] == pytest.approx(0.0)
    assert result[-1] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# _arc_correlation
# ---------------------------------------------------------------------------


def test_arc_correlation_identical_sequences():
    arc = [0.15, 0.35, 0.75, 0.20]
    r = _arc_correlation(arc, arc)
    assert r == pytest.approx(1.0, abs=1e-3)


def test_arc_correlation_different_lengths():
    proposed = [0.15, 0.35, 0.75, 0.75, 0.20]
    actual = [0.35, 0.75, 0.20]
    r = _arc_correlation(proposed, actual)
    assert r is not None
    assert -1.0 <= r <= 1.0


def test_arc_correlation_too_short():
    assert _arc_correlation([0.5], [0.5, 0.8]) is None
    assert _arc_correlation([0.5, 0.8], [0.5]) is None


def test_arc_correlation_constant_returns_none():
    assert _arc_correlation([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]) is None


# ---------------------------------------------------------------------------
# _expand_proposed
# ---------------------------------------------------------------------------


def test_expand_proposed_respects_play_count():
    plan = _make_plan(
        [
            {"name": "verse", "bars": 8, "play_count": 2, "arc": 0.35},
            {"name": "chorus", "bars": 8, "play_count": 1, "arc": 0.75},
        ]
    )
    expanded = _expand_proposed(plan)
    assert [n for n, _, _ in expanded] == ["verse", "verse", "chorus"]


def test_expand_proposed_bars_preserved():
    plan = _make_plan([{"name": "bridge", "bars": 4, "play_count": 3, "arc": 0.20}])
    expanded = _expand_proposed(plan)
    assert all(bars == 4 for _, _, bars in expanded)
    assert len(expanded) == 3


def test_expand_proposed_arc_preserved():
    plan = _make_plan([{"name": "chorus", "bars": 8, "play_count": 2, "arc": 0.75}])
    expanded = _expand_proposed(plan)
    assert all(arc == pytest.approx(0.75) for _, arc, _ in expanded)


# ---------------------------------------------------------------------------
# _build_drift_summary
# ---------------------------------------------------------------------------


def _actual_insts(names: list[str], bars: int = 8) -> list[dict]:
    return [{"section_name": n, "bars": bars, "has_vocals": False} for n in names]


def test_drift_summary_removed():
    plan = _make_plan(
        [
            {"name": "intro", "play_count": 1},
            {"name": "verse", "play_count": 1},
        ]
    )
    proposed = _expand_proposed(plan)
    actual = _actual_insts(["verse"])
    summary = _build_drift_summary(proposed, actual)
    assert summary.removed == ["intro"]
    assert summary.added == []


def test_drift_summary_added():
    plan = _make_plan([{"name": "verse", "play_count": 1}])
    proposed = _expand_proposed(plan)
    actual = _actual_insts(["verse", "coda"])
    summary = _build_drift_summary(proposed, actual)
    assert summary.added == ["coda"]
    assert summary.removed == []


def test_drift_summary_reordered():
    plan = _make_plan(
        [
            {"name": "verse", "play_count": 1},
            {"name": "chorus", "play_count": 1},
            {"name": "bridge", "play_count": 1},
        ]
    )
    proposed = _expand_proposed(plan)
    # Human swapped chorus and bridge
    actual = _actual_insts(["verse", "bridge", "chorus"])
    summary = _build_drift_summary(proposed, actual)
    assert summary.reordered is True


def test_drift_summary_same_order():
    plan = _make_plan(
        [
            {"name": "verse", "play_count": 2},
            {"name": "chorus", "play_count": 1},
        ]
    )
    proposed = _expand_proposed(plan)
    actual = _actual_insts(["verse", "verse", "chorus"])
    summary = _build_drift_summary(proposed, actual)
    assert summary.reordered is False


def test_drift_summary_clip_suffix_stripped():
    """Clip names like 'chorus_02' should match plan label 'chorus'."""
    plan = _make_plan([{"name": "chorus", "play_count": 1}])
    proposed = _expand_proposed(plan)
    actual = _actual_insts(["chorus_02"])
    summary = _build_drift_summary(proposed, actual)
    assert summary.removed == []
    assert summary.added == []


# ---------------------------------------------------------------------------
# _build_bar_deltas
# ---------------------------------------------------------------------------


def test_bar_deltas_exact_match():
    plan = _make_plan([{"name": "verse", "bars": 8, "play_count": 1}])
    proposed = _expand_proposed(plan)
    actual = _actual_insts(["verse"], bars=8)
    deltas = _build_bar_deltas(proposed, actual)
    assert deltas["verse"].delta == 0
    assert deltas["verse"].proposed == 8


def test_bar_deltas_section_cut():
    plan = _make_plan([{"name": "chorus", "bars": 8, "play_count": 2}])
    proposed = _expand_proposed(plan)
    actual = _actual_insts(["chorus"], bars=8)  # one instance instead of two
    deltas = _build_bar_deltas(proposed, actual)
    assert deltas["chorus"].proposed == 16  # 8 × 2
    assert deltas["chorus"].actual == 8
    assert deltas["chorus"].delta == -8


def test_bar_deltas_section_extended():
    plan = _make_plan([{"name": "bridge", "bars": 4, "play_count": 1}])
    proposed = _expand_proposed(plan)
    actual = _actual_insts(["bridge"], bars=6)
    deltas = _build_bar_deltas(proposed, actual)
    assert deltas["bridge"].delta == 2


def test_bar_deltas_only_shared_labels():
    plan = _make_plan(
        [
            {"name": "intro", "bars": 4, "play_count": 1},
            {"name": "verse", "bars": 8, "play_count": 1},
        ]
    )
    proposed = _expand_proposed(plan)
    # intro removed, coda added
    actual = _actual_insts(["verse", "coda"], bars=8)
    deltas = _build_bar_deltas(proposed, actual)
    assert "intro" not in deltas  # only in proposed
    assert "coda" not in deltas  # only in actual
    assert "verse" in deltas


# ---------------------------------------------------------------------------
# compare_plans (integration)
# ---------------------------------------------------------------------------


def test_compare_plans_full(tmp_path):
    """End-to-end: production_plan.yml + arrangement.txt → DriftReport."""
    plan = _make_plan(
        [
            {"name": "intro", "bars": 4, "play_count": 1, "arc": 0.15},
            {"name": "verse", "bars": 8, "play_count": 2, "arc": 0.35},
            {"name": "chorus", "bars": 8, "play_count": 2, "arc": 0.75},
            {"name": "outro", "bars": 4, "play_count": 1, "arc": 0.15},
        ]
    )

    # Human removed intro/outro, cut chorus to one instance
    instances = [
        {"name": "verse", "track": 1, "bars": 8},
        {"name": "verse", "track": 4, "bars": 8},  # vocals on track 4
        {"name": "chorus_01", "track": 1, "bars": 8},
        {"name": "verse", "track": 1, "bars": 8},
    ]
    arr_text = _make_arrangement_txt(instances)
    arr_path = tmp_path / "arrangement.txt"
    arr_path.write_text(arr_text)

    report = compare_plans(plan, arr_path, use_claude=False)

    assert "intro" in report.drift.removed
    assert "outro" in report.drift.removed
    assert report.drift.added == []
    assert report.energy_arc_correlation is not None
    assert report.summary == ""  # no_claude


def test_compare_plans_reordered(tmp_path):
    plan = _make_plan(
        [
            {"name": "verse", "bars": 8, "play_count": 1, "arc": 0.35},
            {"name": "chorus", "bars": 8, "play_count": 1, "arc": 0.75},
            {"name": "bridge", "bars": 4, "play_count": 1, "arc": 0.20},
        ]
    )
    instances = [
        {"name": "verse", "track": 1, "bars": 8},
        {"name": "bridge", "track": 1, "bars": 4},
        {"name": "chorus", "track": 1, "bars": 8},
    ]
    arr_path = tmp_path / "arrangement.txt"
    arr_path.write_text(_make_arrangement_txt(instances))

    report = compare_plans(plan, arr_path, use_claude=False)
    assert report.drift.reordered is True


# ---------------------------------------------------------------------------
# write_report / load_report
# ---------------------------------------------------------------------------


def _make_report() -> DriftReport:
    return DriftReport(
        generated="2026-01-01T00:00:00Z",
        song_title="Test Song",
        proposed_sections=["verse", "verse", "chorus"],
        actual_sections=["verse", "chorus"],
        drift=DriftSummary(removed=[], added=[], reordered=False),
        bar_deltas={
            "chorus": BarDelta(proposed=8, actual=8, delta=0),
            "verse": BarDelta(proposed=16, actual=8, delta=-8),
        },
        energy_arc_correlation=0.85,
        summary="The human cut one verse repeat.",
    )


def test_write_and_load_round_trip(tmp_path):
    report = _make_report()
    write_report(tmp_path, report)
    loaded = load_report(tmp_path)
    assert loaded is not None
    assert loaded.song_title == report.song_title
    assert loaded.proposed_sections == report.proposed_sections
    assert loaded.actual_sections == report.actual_sections
    assert loaded.drift.removed == []
    assert loaded.drift.reordered is False
    assert loaded.bar_deltas["verse"].delta == -8
    assert loaded.energy_arc_correlation == pytest.approx(0.85)
    assert loaded.summary == "The human cut one verse repeat."


def test_load_report_missing_returns_none(tmp_path):
    assert load_report(tmp_path) is None


def test_write_report_creates_yaml_file(tmp_path):
    report = _make_report()
    out_path = write_report(tmp_path, report)
    assert out_path == tmp_path / REPORT_FILENAME
    assert out_path.exists()
    data = yaml.safe_load(out_path.read_text())
    assert data["song_title"] == "Test Song"
    assert data["drift"]["reordered"] is False
    assert data["bar_deltas"]["verse"]["delta"] == -8


def test_write_report_null_correlation(tmp_path):
    report = _make_report()
    report.energy_arc_correlation = None
    out_path = write_report(tmp_path, report)
    data = yaml.safe_load(out_path.read_text())
    assert data["energy_arc_correlation"] is None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_missing_plan(tmp_path):
    from white_composition.drift_report import main

    with patch(
        "sys.argv", ["drift_report", "--production-dir", str(tmp_path), "--no-claude"]
    ):
        with pytest.raises(SystemExit) as exc:
            main()
    assert exc.value.code == 1


def test_cli_missing_arrangement(tmp_path):
    from white_composition.drift_report import main

    # Write a production plan but no arrangement.txt
    plan = _make_plan([{"name": "verse", "bars": 8}])
    from white_composition.production_plan import save_plan

    save_plan(plan, tmp_path)

    with patch(
        "sys.argv", ["drift_report", "--production-dir", str(tmp_path), "--no-claude"]
    ):
        with pytest.raises(SystemExit) as exc:
            main()
    assert exc.value.code == 1


def test_cli_no_claude_writes_report(tmp_path):
    from white_composition.drift_report import main
    from white_composition.production_plan import save_plan

    plan = _make_plan(
        [
            {"name": "verse", "bars": 8, "play_count": 1},
            {"name": "chorus", "bars": 8, "play_count": 1},
        ]
    )
    save_plan(plan, tmp_path)

    instances = [
        {"name": "verse", "track": 1, "bars": 8},
        {"name": "chorus", "track": 1, "bars": 8},
    ]
    arr_path = tmp_path / "arrangement.txt"
    arr_path.write_text(_make_arrangement_txt(instances))

    with patch(
        "sys.argv", ["drift_report", "--production-dir", str(tmp_path), "--no-claude"]
    ):
        main()

    report = load_report(tmp_path)
    assert report is not None
    assert report.summary == ""
    assert (tmp_path / REPORT_FILENAME).exists()


def test_cli_claude_api_failure_still_writes(tmp_path):
    from white_composition.drift_report import main
    from white_composition.production_plan import save_plan

    plan = _make_plan([{"name": "verse", "bars": 8, "play_count": 1}])
    save_plan(plan, tmp_path)

    instances = [{"name": "verse", "track": 1, "bars": 8}]
    arr_path = tmp_path / "arrangement.txt"
    arr_path.write_text(_make_arrangement_txt(instances))

    with (
        patch("sys.argv", ["drift_report", "--production-dir", str(tmp_path)]),
        patch(
            "white_composition.drift_report._generate_summary",
            side_effect=Exception("API down"),
        ),
    ):
        main()

    report = load_report(tmp_path)
    assert report is not None


def test_cli_custom_arrangement_path(tmp_path):
    from white_composition.drift_report import main
    from white_composition.production_plan import save_plan

    plan = _make_plan([{"name": "verse", "bars": 8, "play_count": 1}])
    save_plan(plan, tmp_path)

    custom_path = tmp_path / "custom_arr.txt"
    custom_path.write_text(
        _make_arrangement_txt([{"name": "verse", "track": 1, "bars": 8}])
    )

    with patch(
        "sys.argv",
        [
            "drift_report",
            "--production-dir",
            str(tmp_path),
            "--arrangement",
            str(custom_path),
            "--no-claude",
        ],
    ):
        main()

    assert (tmp_path / REPORT_FILENAME).exists()
