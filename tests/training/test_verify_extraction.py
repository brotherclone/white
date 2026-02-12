"""Tests for training data verification tool."""

import mido
import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import logging

from io import BytesIO
from pathlib import Path
from training.verify_extraction import (
    extract_segments,
    format_coverage_report,
    format_fidelity_report,
    generate_coverage_report,
    load_media_rows,
    select_segments,
    verify_audio_fidelity,
    verify_midi_fidelity,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# --- Helpers ---


def make_audio_bytes(duration_s: float = 1.0, sr: int = 44100) -> bytes:
    """Generate synthetic audio waveform bytes (440Hz sine)."""
    t = np.linspace(0, duration_s, int(sr * duration_s), dtype=np.float32)
    audio = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    return audio.tobytes()


def make_midi_bytes() -> bytes:
    """Generate a minimal valid MIDI file as bytes."""
    mid = mido.MidiFile()
    track = mido.MidiTrack()
    mid.tracks.append(track)
    track.append(mido.Message("note_on", note=60, velocity=64, time=0))
    track.append(mido.Message("note_off", note=60, velocity=0, time=480))
    buf = BytesIO()
    mid.save(file=buf)
    return buf.getvalue()


def make_metadata_df(n: int = 20, colors: list[str] | None = None) -> pl.DataFrame:
    """Create a synthetic metadata DataFrame."""
    if colors is None:
        colors = ["Red", "Orange", "Yellow", "Blue"]

    rows = []
    for i in range(n):
        color = colors[i % len(colors)]
        song_id = f"{(i % 8) + 1:02d}_{(i % 12) + 1:02d}"
        rows.append(
            {
                "segment_id": f"{song_id}_seg_{i:04d}_track_01",
                "song_id": song_id,
                "rainbow_color": color,
                "has_audio": i % 5 != 0,  # 80% audio
                "has_midi": i % 3 == 0,  # 33% MIDI
                "lyric_text": f"Some lyrics {i}" if i % 4 != 0 else None,
                "start_seconds": float(i),
                "end_seconds": float(i + 1),
                "duration_seconds": 1.0,
                "structure_section": "Verse",
                "content_type": "audio_only",
                "track_description": "vocals",
            }
        )
    return pl.DataFrame(rows)


def make_media_parquet(tmp_path: Path, metadata: pl.DataFrame) -> Path:
    """Create a synthetic media parquet from metadata."""
    rows = []
    for row in metadata.iter_rows(named=True):
        audio = make_audio_bytes(duration_s=1.0) if row["has_audio"] else None
        midi = make_midi_bytes() if row["has_midi"] else None
        rows.append(
            {
                "segment_id": row["segment_id"],
                "audio_waveform": audio,
                "audio_sample_rate": 44100 if audio else None,
                "midi_binary": midi,
            }
        )

    # Write as parquet with pyarrow for binary column support
    table = pa.table(
        {
            "segment_id": [r["segment_id"] for r in rows],
            "audio_waveform": [r["audio_waveform"] for r in rows],
            "audio_sample_rate": [r["audio_sample_rate"] for r in rows],
            "midi_binary": [r["midi_binary"] for r in rows],
        }
    )
    path = tmp_path / "test_media.parquet"
    pq.write_table(table, path, row_group_size=10)
    return path


# --- select_segments ---


class TestSelectSegments:
    def test_no_filters(self):
        df = make_metadata_df(20)
        result = select_segments(df)
        assert len(result) == 20

    def test_filter_by_color(self):
        df = make_metadata_df(20)
        result = select_segments(df, color="Red")
        assert result["rainbow_color"].to_list() == ["Red"] * len(result)
        assert len(result) > 0

    def test_filter_by_song(self):
        df = make_metadata_df(20)
        result = select_segments(df, song="01_01")
        assert all(s == "01_01" for s in result["song_id"].to_list())

    def test_random_n(self):
        df = make_metadata_df(100)
        result = select_segments(df, random_n=5)
        assert len(result) == 5

    def test_color_and_random(self):
        df = make_metadata_df(100)
        result = select_segments(df, color="Red", random_n=3)
        assert len(result) <= 3
        assert all(c == "Red" for c in result["rainbow_color"].to_list())

    def test_empty_result_for_missing_color(self):
        df = make_metadata_df(20)
        result = select_segments(df, color="Violet")
        assert len(result) == 0


# --- generate_coverage_report ---


class TestCoverageReport:
    def test_basic_report(self):
        df = make_metadata_df(20)
        report = generate_coverage_report(df)
        assert report["total_segments"] == 20
        assert "by_color" in report
        assert report["unique_songs"] > 0

    def test_color_stats(self):
        df = make_metadata_df(20, colors=["Red", "Blue"])
        report = generate_coverage_report(df)
        red_stat = next(c for c in report["by_color"] if c["color"] == "Red")
        assert red_stat["segments"] == 10
        assert red_stat["audio_pct"] > 0

    def test_missing_colors_show_zero(self):
        df = make_metadata_df(20, colors=["Red"])
        report = generate_coverage_report(df)
        green_stat = next(c for c in report["by_color"] if c["color"] == "Green")
        assert green_stat["segments"] == 0

    def test_unlabeled_detection(self):
        rows = [
            {
                "segment_id": f"seg_{i}",
                "song_id": "01_01",
                "rainbow_color": "UNKNOWN" if i < 5 else "Red",
                "has_audio": True,
                "has_midi": False,
                "lyric_text": "test",
                "start_seconds": 0.0,
                "end_seconds": 10.0,
                "duration_seconds": 10.0,
                "structure_section": "Verse",
                "content_type": "audio_only",
                "track_description": "vocals",
            }
            for i in range(10)
        ]
        df = pl.DataFrame(rows)
        report = generate_coverage_report(df)
        unlabeled = next(
            (c for c in report["by_color"] if c["color"] == "UNLABELED"), None
        )
        assert unlabeled is not None
        assert unlabeled["segments"] == 5

    def test_empty_metadata(self):
        df = pl.DataFrame(
            schema={
                "segment_id": pl.Utf8,
                "song_id": pl.Utf8,
                "rainbow_color": pl.Utf8,
                "has_audio": pl.Boolean,
                "has_midi": pl.Boolean,
                "lyric_text": pl.Utf8,
            }
        )
        report = generate_coverage_report(df)
        assert report["total_segments"] == 0


class TestFormatCoverageReport:
    def test_contains_key_sections(self):
        report = {
            "total_segments": 100,
            "unique_songs": 10,
            "total_audio": 85,
            "total_midi": 43,
            "audio_pct": 85.0,
            "midi_pct": 43.0,
            "by_color": [
                {
                    "color": "Red",
                    "segments": 50,
                    "audio_pct": 90.0,
                    "midi_pct": 50.0,
                    "text_pct": 100.0,
                },
            ],
            "zero_segment_songs": [],
            "low_segment_songs": [],
        }
        text = format_coverage_report(report)
        assert "TRAINING DATA COVERAGE REPORT" in text
        assert "Red" in text
        assert "85.0%" in text

    def test_shows_warnings(self):
        report = {
            "total_segments": 10,
            "unique_songs": 1,
            "total_audio": 10,
            "total_midi": 0,
            "audio_pct": 100.0,
            "midi_pct": 0.0,
            "by_color": [],
            "zero_segment_songs": ["missing_01"],
            "low_segment_songs": [{"song_id": "low_01", "segment_count": 1}],
        }
        text = format_coverage_report(report)
        assert "missing_01" in text
        assert "low_01" in text


# --- extract_segments ---


class TestExtractSegments:
    def test_extracts_audio_files(self, tmp_path):
        metadata = make_metadata_df(10)
        media_path = make_media_parquet(tmp_path, metadata)
        output = tmp_path / "output"

        stats = extract_segments(metadata, media_path, output, random_n=3)
        assert stats["extracted_audio"] > 0
        wav_files = list(output.glob("*.wav"))
        assert len(wav_files) > 0

    def test_extracts_midi_files(self, tmp_path):
        metadata = make_metadata_df(10)
        media_path = make_media_parquet(tmp_path, metadata)
        output = tmp_path / "output"

        stats = extract_segments(metadata, media_path, output, random_n=10)
        assert stats["extracted_midi"] > 0
        mid_files = list(output.glob("*.mid"))
        assert len(mid_files) > 0

    def test_writes_sidecar_json(self, tmp_path):
        metadata = make_metadata_df(10)
        media_path = make_media_parquet(tmp_path, metadata)
        output = tmp_path / "output"

        extract_segments(metadata, media_path, output, random_n=3)
        json_files = list(output.glob("*.json"))
        assert len(json_files) > 0

        import json

        with open(json_files[0]) as f:
            sidecar = json.load(f)
        assert "segment_id" in sidecar
        assert "rainbow_color" in sidecar

    def test_filter_by_color(self, tmp_path):
        metadata = make_metadata_df(20)
        media_path = make_media_parquet(tmp_path, metadata)
        output = tmp_path / "output"

        stats = extract_segments(metadata, media_path, output, color="Red")
        logger.info(f"Extracted {stats['extracted_audio']} audio files")
        # All extracted files should be Red
        json_files = list(output.glob("*.json"))
        import json

        for jf in json_files:
            with open(jf) as f:
                data = json.load(f)
            assert data["rainbow_color"] == "Red"

    def test_empty_selection(self, tmp_path):
        metadata = make_metadata_df(10)
        media_path = make_media_parquet(tmp_path, metadata)
        output = tmp_path / "output"

        stats = extract_segments(metadata, media_path, output, color="Violet")
        assert stats["extracted_audio"] == 0
        assert stats["extracted_midi"] == 0

    def test_wav_is_playable(self, tmp_path):
        """Verify extracted WAV files can be read back."""
        metadata = make_metadata_df(10)
        media_path = make_media_parquet(tmp_path, metadata)
        output = tmp_path / "output"

        extract_segments(metadata, media_path, output, random_n=3)
        wav_files = list(output.glob("*.wav"))
        assert len(wav_files) > 0

        # Read back with soundfile
        import soundfile as sf

        audio, sr = sf.read(str(wav_files[0]))
        assert sr == 44100
        assert len(audio) > 0


# --- verify_audio_fidelity ---


class TestAudioFidelity:
    def test_all_pass(self, tmp_path):
        metadata = make_metadata_df(20)
        media_path = make_media_parquet(tmp_path, metadata)

        results = verify_audio_fidelity(metadata, media_path, random_n=5, row_group=0)
        assert results["checked"] > 0
        assert results["passed"] == results["checked"]
        assert results["failures"] == []

    def test_detects_no_audio_segments(self, tmp_path):
        # All segments have has_audio=False
        rows = [
            {
                "segment_id": f"seg_{i}",
                "song_id": "01_01",
                "rainbow_color": "Red",
                "has_audio": False,
                "has_midi": False,
                "lyric_text": None,
                "start_seconds": 0.0,
                "end_seconds": 10.0,
                "duration_seconds": 10.0,
                "structure_section": "Verse",
                "content_type": "empty",
                "track_description": "vocals",
            }
            for i in range(5)
        ]
        metadata = pl.DataFrame(rows)
        media_path = make_media_parquet(tmp_path, metadata)

        results = verify_audio_fidelity(metadata, media_path, random_n=5, row_group=0)
        assert results["checked"] == 0


# --- verify_midi_fidelity ---


class TestMidiFidelity:
    def test_all_pass(self, tmp_path):
        metadata = make_metadata_df(20)
        media_path = make_media_parquet(tmp_path, metadata)

        results = verify_midi_fidelity(metadata, media_path, random_n=5, row_group=0)
        assert results["checked"] > 0
        assert results["passed"] == results["checked"]

    def test_detects_empty_midi(self, tmp_path):
        """MIDI with no note events should fail."""
        # Create metadata with has_midi=True
        rows = [
            {
                "segment_id": "seg_0",
                "song_id": "01_01",
                "rainbow_color": "Red",
                "has_audio": False,
                "has_midi": True,
                "lyric_text": None,
                "start_seconds": 0.0,
                "end_seconds": 10.0,
                "duration_seconds": 10.0,
                "structure_section": "Verse",
                "content_type": "midi_only",
                "track_description": "guitar",
            }
        ]
        metadata = pl.DataFrame(rows)

        # Create media with empty MIDI (no note events)
        empty_mid = mido.MidiFile()
        empty_mid.tracks.append(mido.MidiTrack())
        buf = BytesIO()
        empty_mid.save(file=buf)

        table = pa.table(
            {
                "segment_id": ["seg_0"],
                "audio_waveform": [None],
                "audio_sample_rate": [None],
                "midi_binary": [buf.getvalue()],
            }
        )
        media_path = tmp_path / "test_media.parquet"
        pq.write_table(table, media_path)

        results = verify_midi_fidelity(metadata, media_path, random_n=1, row_group=0)
        assert results["checked"] == 1
        assert results["passed"] == 0
        assert any("no note_on" in f for f in results["failures"])


# --- format_fidelity_report ---


class TestFormatFidelityReport:
    def test_format_clean(self):
        audio = {"checked": 10, "passed": 10, "failures": []}
        midi = {"checked": 5, "passed": 5, "failures": []}
        text = format_fidelity_report(audio, midi)
        assert "10/10" in text
        assert "5/5" in text

    def test_format_with_failures(self):
        audio = {
            "checked": 10,
            "passed": 8,
            "failures": ["seg_1: too loud", "seg_2: empty"],
        }
        midi = {"checked": 5, "passed": 5, "failures": []}
        text = format_fidelity_report(audio, midi)
        assert "8/10" in text
        assert "too loud" in text


# --- load_media_rows ---


class TestLoadMediaRows:
    def test_loads_requested_ids(self, tmp_path):
        metadata = make_metadata_df(10)
        media_path = make_media_parquet(tmp_path, metadata)
        all_ids = metadata["segment_id"].to_list()

        result = load_media_rows(media_path, all_ids[:3])
        assert len(result) == 3
        assert set(result["segment_id"].to_list()).issubset(set(all_ids))

    def test_empty_request(self, tmp_path):
        metadata = make_metadata_df(10)
        media_path = make_media_parquet(tmp_path, metadata)

        result = load_media_rows(media_path, [])
        assert len(result) == 0

    def test_missing_ids_ignored(self, tmp_path):
        metadata = make_metadata_df(10)
        media_path = make_media_parquet(tmp_path, metadata)

        result = load_media_rows(media_path, ["nonexistent_seg"])
        assert len(result) == 0
