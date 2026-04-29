"""Tests for app.generators.midi.style_reference and related helpers."""

from __future__ import annotations

from pathlib import Path

import mido
import pytest
import yaml
from white_core.music.style_profile import StyleProfile
from white_generation.patterns.aesthetic_hints import style_profile_tag_adjustment
from white_generation.style_reference import (
    aggregate_profiles,
    artist_slug,
    extract_style_profile,
    load_or_extract_profile,
)

# ---------------------------------------------------------------------------
# artist_slug
# ---------------------------------------------------------------------------


def test_artist_slug_lowercases():
    assert artist_slug("Grouper") == "grouper"


def test_artist_slug_replaces_spaces():
    assert artist_slug("Beach House") == "beach_house"


def test_artist_slug_strips_special_chars():
    assert artist_slug("Low (band)") == "low_band"


# ---------------------------------------------------------------------------
# extract_style_profile
# ---------------------------------------------------------------------------


def _make_midi(notes: list[tuple[int, int, int]], tpb: int = 480) -> Path:
    """Build a minimal MIDI file in memory and return bytes wrapped in a BytesIO-backed Path.

    notes: list of (pitch, start_tick, dur_ticks)
    """
    mid = mido.MidiFile(ticks_per_beat=tpb)
    track = mido.MidiTrack()
    mid.tracks.append(track)

    events: list[tuple[int, str, int, int]] = []
    for pitch, start, dur in notes:
        events.append((start, "on", pitch, 80))
        events.append((start + dur, "off", pitch, 0))
    events.sort()

    prev_tick = 0
    for tick, ev_type, pitch, vel in events:
        delta = tick - prev_tick
        if ev_type == "on":
            track.append(mido.Message("note_on", note=pitch, velocity=vel, time=delta))
        else:
            track.append(mido.Message("note_off", note=pitch, velocity=0, time=delta))
        prev_tick = tick

    return mid


def _save_midi(mid: mido.MidiFile, tmp_path: Path, name: str = "test.mid") -> Path:
    p = tmp_path / name
    mid.save(str(p))
    return p


class TestExtractStyleProfile:
    def test_basic_extraction(self, tmp_path):
        tpb = 480
        mid = _make_midi(
            [(60, 0, tpb), (62, tpb * 2, tpb), (64, tpb * 4, tpb)],
            tpb=tpb,
        )
        f = _save_midi(mid, tmp_path)
        profile = extract_style_profile([f])
        assert profile is not None
        assert profile.note_density > 0
        assert profile.mean_duration_beats > 0
        assert profile.velocity_mean > 0

    def test_empty_midi_returns_none(self, tmp_path):
        mid = mido.MidiFile(ticks_per_beat=480)
        mid.tracks.append(mido.MidiTrack())
        f = _save_midi(mid, tmp_path)
        profile = extract_style_profile([f])
        assert profile is None

    def test_interval_histogram_populated(self, tmp_path):
        tpb = 480
        mid = _make_midi(
            [(60, 0, tpb), (62, tpb, tpb), (64, tpb * 2, tpb)],  # ascending by 2 each
            tpb=tpb,
        )
        f = _save_midi(mid, tmp_path)
        profile = extract_style_profile([f])
        assert profile is not None
        # Intervals should include +2 (major second)
        assert 2 in profile.interval_histogram

    def test_rest_ratio_high_for_sparse_midi(self, tmp_path):
        tpb = 480
        bar_ticks = tpb * 4
        # One note at start, nothing for 8 bars
        mid = _make_midi([(60, 0, tpb)], tpb=tpb)
        # Add a dummy note far out to create many empty bars
        track = mid.tracks[0]
        track.append(mido.Message("note_on", note=62, velocity=80, time=bar_ticks * 8))
        track.append(mido.Message("note_off", note=62, velocity=0, time=tpb))
        f = _save_midi(mid, tmp_path)
        profile = extract_style_profile([f])
        assert profile is not None
        assert profile.rest_ratio > 0.5


# ---------------------------------------------------------------------------
# aggregate_profiles
# ---------------------------------------------------------------------------


class TestAggregateProfiles:
    def test_empty_list_returns_none(self):
        assert aggregate_profiles([]) is None

    def test_single_profile_identity(self):
        p = StyleProfile(artist="test", note_density=2.0, mean_duration_beats=1.0)
        agg = aggregate_profiles([p])
        assert agg is not None
        assert agg.note_density == pytest.approx(2.0)

    def test_averages_numeric_fields(self):
        p1 = StyleProfile(artist="a", note_density=2.0, velocity_mean=60.0)
        p2 = StyleProfile(artist="b", note_density=4.0, velocity_mean=80.0)
        agg = aggregate_profiles([p1, p2])
        assert agg is not None
        assert agg.note_density == pytest.approx(3.0)
        assert agg.velocity_mean == pytest.approx(70.0)

    def test_merges_interval_histograms(self):
        p1 = StyleProfile(artist="a", interval_histogram={2: 1.0})
        p2 = StyleProfile(artist="b", interval_histogram={-2: 1.0})
        agg = aggregate_profiles([p1, p2])
        assert 2 in agg.interval_histogram
        assert -2 in agg.interval_histogram


# ---------------------------------------------------------------------------
# load_or_extract_profile (caching)
# ---------------------------------------------------------------------------


class TestLoadOrExtractProfile:
    def test_returns_none_when_no_midi_files(self, tmp_path):
        (tmp_path / "grouper").mkdir()
        result = load_or_extract_profile("Grouper", tmp_path)
        assert result is None

    def test_extracts_and_caches_profile(self, tmp_path):
        artist_dir = tmp_path / "grouper"
        artist_dir.mkdir()
        tpb = 480
        mid = _make_midi([(60, 0, tpb), (62, tpb * 2, tpb)], tpb=tpb)
        _save_midi(mid, artist_dir, "test.mid")

        profile = load_or_extract_profile("Grouper", tmp_path)
        assert profile is not None

        # Profile cache file created
        cache_path = artist_dir / "profile.yml"
        assert cache_path.exists()

    def test_loads_cached_profile_without_re_extracting(self, tmp_path):
        artist_dir = tmp_path / "grouper"
        artist_dir.mkdir()
        tpb = 480
        mid = _make_midi([(60, 0, tpb)], tpb=tpb)
        _save_midi(mid, artist_dir, "test.mid")

        # First extraction
        p1 = load_or_extract_profile("Grouper", tmp_path)
        assert p1 is not None

        # Mutate cache to detect re-extraction
        cache = artist_dir / "profile.yml"
        data = yaml.safe_load(cache.read_text())
        data["note_density"] = 999.0
        cache.write_text(yaml.dump(data))

        # Make cache newer than MIDI
        import os
        import time

        time.sleep(0.01)
        os.utime(cache, None)

        p2 = load_or_extract_profile("Grouper", tmp_path)
        assert p2 is not None
        assert p2.note_density == pytest.approx(999.0)  # loaded from cache


# ---------------------------------------------------------------------------
# style_profile_tag_adjustment
# ---------------------------------------------------------------------------


class TestStyleProfileTagAdjustment:
    def test_low_density_boosts_sparse_drums(self):
        profile = {"note_density": 1.5, "style_weight": 0.4}
        adj = style_profile_tag_adjustment(profile, ["sparse"], "drums")
        assert adj > 0

    def test_low_density_penalises_dense_drums(self):
        profile = {"note_density": 1.0, "style_weight": 0.4}
        adj = style_profile_tag_adjustment(profile, ["dense"], "drums")
        assert adj < 0

    def test_long_duration_boosts_pedal_bass(self):
        profile = {"mean_duration_beats": 2.0, "style_weight": 0.4}
        adj = style_profile_tag_adjustment(profile, ["pedal"], "bass")
        assert adj > 0

    def test_high_rest_boosts_sparse_melody(self):
        profile = {"rest_ratio": 0.7, "style_weight": 0.4}
        adj = style_profile_tag_adjustment(profile, ["sparse"], "melody")
        assert adj > 0

    def test_missing_profile_returns_zero(self):
        assert style_profile_tag_adjustment(None, ["sparse"], "drums") == pytest.approx(
            0.0
        )
        assert style_profile_tag_adjustment({}, ["sparse"], "drums") == pytest.approx(
            0.0
        )

    def test_style_weight_scales_adjustment(self):
        profile_high = {"note_density": 1.5, "style_weight": 0.8}
        profile_low = {"note_density": 1.5, "style_weight": 0.2}
        adj_high = style_profile_tag_adjustment(profile_high, ["sparse"], "drums")
        adj_low = style_profile_tag_adjustment(profile_low, ["sparse"], "drums")
        assert adj_high > adj_low
