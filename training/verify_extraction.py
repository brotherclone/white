"""Training data verification tool.

Extracts, inspects, and validates training segments from parquet files.
Use this before committing to a multi-hour RunPod training run.

Usage:
    # Coverage report only (fast, no binary reads)
    python -m training.verify_extraction

    # Extract 10 random segments as playable WAV/MIDI
    python -m training.verify_extraction --extract --random 10

    # Extract all Green album segments
    python -m training.verify_extraction --extract --color Green

    # Extract segments from a specific song
    python -m training.verify_extraction --extract --song 05_01

    # Run fidelity checks on 20 random segments
    python -m training.verify_extraction --fidelity --random 20

    # Full verification (coverage + extract 5 + fidelity 10)
    python -m training.verify_extraction --all
"""

import argparse
import json
import logging
import sys
from io import BytesIO
from pathlib import Path
from typing import Optional

import mido
import numpy as np
import polars as pl
import pyarrow.parquet as pq
import soundfile as sf

logger = logging.getLogger(__name__)

# Default paths
DEFAULT_METADATA_PARQUET = (
    Path(__file__).parent / "data" / "training_segments_metadata.parquet"
)
DEFAULT_MEDIA_PARQUET = (
    Path(__file__).parent / "data" / "training_segments_media.parquet"
)
DEFAULT_OUTPUT_DIR = Path(__file__).parent / "verification_output"

# Album color display order
COLOR_ORDER = ["Red", "Orange", "Yellow", "Green", "Blue", "Indigo", "Violet", "White"]


# ---------------------------------------------------------------------------
# 1. Segment Extraction
# ---------------------------------------------------------------------------


def load_metadata(parquet_path: Path) -> pl.DataFrame:
    """Load metadata parquet (no binary columns)."""
    return pl.read_parquet(parquet_path)


def load_media_rows(
    parquet_path: Path,
    segment_ids: list[str],
) -> pl.DataFrame:
    """Load specific rows from media parquet by segment_id.

    The media parquet is ~60GB (10 row groups of ~6GB each). To avoid scanning
    the entire file, we first read only the segment_id column from each row group
    to locate matches, then only read binary data from row groups that have them.
    """
    target_ids = set(segment_ids)
    found_tables = []
    pf = pq.ParquetFile(parquet_path)

    for i in range(pf.metadata.num_row_groups):
        # Fast: read only segment_id from this row group
        id_table = pf.read_row_group(i, columns=["segment_id"])
        ids_in_group = set(id_table.column("segment_id").to_pylist())
        matches = target_ids & ids_in_group

        if matches:
            logger.info(f"Reading row group {i} ({len(matches)} matches)...")
            # Slow: read binary data from this row group
            table = pf.read_row_group(
                i,
                columns=[
                    "segment_id",
                    "audio_waveform",
                    "audio_sample_rate",
                    "midi_binary",
                ],
            )
            df = pl.from_arrow(table).filter(pl.col("segment_id").is_in(list(matches)))
            found_tables.append(df)
            target_ids -= matches

        if not target_ids:
            break

    if not found_tables:
        return pl.DataFrame(
            schema={
                "segment_id": pl.Utf8,
                "audio_waveform": pl.Binary,
                "audio_sample_rate": pl.Int32,
                "midi_binary": pl.Binary,
            }
        )

    return pl.concat(found_tables)


def get_segment_ids_in_row_group(parquet_path: Path, row_group: int = 0) -> list[str]:
    """Get all segment_ids from a specific row group (fast, no binary read)."""
    pf = pq.ParquetFile(parquet_path)
    id_table = pf.read_row_group(row_group, columns=["segment_id"])
    return id_table.column("segment_id").to_pylist()


def select_segments(
    metadata: pl.DataFrame,
    color: Optional[str] = None,
    song: Optional[str] = None,
    random_n: Optional[int] = None,
) -> pl.DataFrame:
    """Filter and optionally sample segments from metadata."""
    df = metadata

    if color:
        df = df.filter(pl.col("rainbow_color") == color)
        if len(df) == 0:
            logger.warning(f"No segments found for color '{color}'")

    if song:
        df = df.filter(pl.col("song_id") == song)
        if len(df) == 0:
            logger.warning(f"No segments found for song '{song}'")

    if random_n and len(df) > random_n:
        df = df.sample(n=random_n, seed=42)

    return df


def extract_segments(
    metadata: pl.DataFrame,
    media_parquet: Path,
    output_dir: Path,
    color: Optional[str] = None,
    song: Optional[str] = None,
    random_n: Optional[int] = None,
) -> dict:
    """Extract segments as playable WAV/MIDI files.

    Returns dict with extraction stats.
    """
    selected = select_segments(metadata, color=color, song=song, random_n=random_n)
    if len(selected) == 0:
        return {"extracted_audio": 0, "extracted_midi": 0, "errors": []}

    segment_ids = selected["segment_id"].to_list()
    media = load_media_rows(media_parquet, segment_ids)

    # Join metadata with media on segment_id
    joined = selected.join(media, on="segment_id", how="left")

    output_dir.mkdir(parents=True, exist_ok=True)
    stats = {"extracted_audio": 0, "extracted_midi": 0, "errors": [], "files": []}

    for row in joined.iter_rows(named=True):
        seg_id = row["segment_id"]
        rc = row.get("rainbow_color", "unknown") or "unknown"
        section = row.get("structure_section", "") or "unknown"
        # Clean section name for filename
        section_clean = section.replace(" ", "_").replace("/", "-")[:30]

        base_name = f"{rc}_{seg_id}_{section_clean}"

        # Extract audio
        audio_bytes = row.get("audio_waveform")
        sample_rate = row.get("audio_sample_rate")
        if audio_bytes is not None and sample_rate is not None:
            try:
                audio_array = np.frombuffer(audio_bytes, dtype=np.float32)
                wav_path = output_dir / f"{base_name}.wav"
                sf.write(str(wav_path), audio_array, sample_rate)
                stats["extracted_audio"] += 1
                stats["files"].append(str(wav_path.name))
            except Exception as e:
                stats["errors"].append(f"Audio extraction failed for {seg_id}: {e}")

        # Extract MIDI
        midi_bytes = row.get("midi_binary")
        if midi_bytes is not None:
            try:
                mid_path = output_dir / f"{base_name}.mid"
                with open(mid_path, "wb") as f:
                    f.write(midi_bytes)
                stats["extracted_midi"] += 1
                stats["files"].append(str(mid_path.name))
            except Exception as e:
                stats["errors"].append(f"MIDI extraction failed for {seg_id}: {e}")

        # Write sidecar metadata JSON
        sidecar = {
            "segment_id": seg_id,
            "song_id": row.get("song_id"),
            "rainbow_color": rc,
            "structure_section": section,
            "start_seconds": row.get("start_seconds"),
            "end_seconds": row.get("end_seconds"),
            "duration_seconds": row.get("duration_seconds"),
            "has_audio": audio_bytes is not None,
            "has_midi": midi_bytes is not None,
            "lyric_text": row.get("lyric_text"),
            "track_description": row.get("track_description"),
            "content_type": row.get("content_type"),
        }
        json_path = output_dir / f"{base_name}.json"
        with open(json_path, "w") as f:
            json.dump(sidecar, f, indent=2, default=str)

    return stats


# ---------------------------------------------------------------------------
# 2. Coverage Report
# ---------------------------------------------------------------------------


def generate_coverage_report(metadata: pl.DataFrame) -> dict:
    """Generate modality coverage report by album color.

    Returns dict with per-color stats and anomalies.
    """
    total = len(metadata)

    # Per-color breakdown
    color_stats = []
    for color in COLOR_ORDER:
        color_df = metadata.filter(pl.col("rainbow_color") == color)
        n = len(color_df)
        if n == 0:
            color_stats.append(
                {
                    "color": color,
                    "segments": 0,
                    "audio_pct": 0.0,
                    "midi_pct": 0.0,
                    "text_pct": 0.0,
                }
            )
            continue

        audio_count = color_df.filter(pl.col("has_audio")).height
        midi_count = color_df.filter(pl.col("has_midi")).height

        # Text presence: lyric_text is non-null and non-empty
        if "lyric_text" in color_df.columns:
            text_count = color_df.filter(
                pl.col("lyric_text").is_not_null() & (pl.col("lyric_text") != "")
            ).height
        else:
            text_count = 0

        color_stats.append(
            {
                "color": color,
                "segments": n,
                "audio_pct": round(100 * audio_count / n, 1),
                "midi_pct": round(100 * midi_count / n, 1),
                "text_pct": round(100 * text_count / n, 1),
            }
        )

    # Handle unlabeled segments
    labeled_colors = set(COLOR_ORDER)
    unlabeled = metadata.filter(
        ~pl.col("rainbow_color").is_in(labeled_colors)
        | pl.col("rainbow_color").is_null()
    )
    if len(unlabeled) > 0:
        un = len(unlabeled)
        un_audio = unlabeled.filter(pl.col("has_audio")).height
        un_midi = unlabeled.filter(pl.col("has_midi")).height
        if "lyric_text" in unlabeled.columns:
            un_text = unlabeled.filter(
                pl.col("lyric_text").is_not_null() & (pl.col("lyric_text") != "")
            ).height
        else:
            un_text = 0
        color_stats.append(
            {
                "color": "UNLABELED",
                "segments": un,
                "audio_pct": round(100 * un_audio / un, 1),
                "midi_pct": round(100 * un_midi / un, 1),
                "text_pct": round(100 * un_text / un, 1),
            }
        )

    # Per-song segment counts for anomaly detection
    song_counts = (
        metadata.group_by("song_id")
        .agg(pl.len().alias("segment_count"))
        .sort("song_id")
    )
    zero_segment_songs = song_counts.filter(pl.col("segment_count") == 0)[
        "song_id"
    ].to_list()

    # Anomalies: songs with unusually few segments
    low_segment_songs = song_counts.filter(pl.col("segment_count") < 3).to_dicts()

    # Overall totals
    total_audio = metadata.filter(pl.col("has_audio")).height
    total_midi = metadata.filter(pl.col("has_midi")).height

    return {
        "total_segments": total,
        "total_audio": total_audio,
        "total_midi": total_midi,
        "audio_pct": round(100 * total_audio / total, 1) if total > 0 else 0,
        "midi_pct": round(100 * total_midi / total, 1) if total > 0 else 0,
        "by_color": color_stats,
        "zero_segment_songs": zero_segment_songs,
        "low_segment_songs": low_segment_songs,
        "unique_songs": metadata["song_id"].n_unique(),
    }


def format_coverage_report(report: dict) -> str:
    """Format coverage report as readable text."""
    lines = []
    lines.append("=" * 70)
    lines.append("TRAINING DATA COVERAGE REPORT")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"Total segments: {report['total_segments']}")
    lines.append(f"Unique songs:   {report['unique_songs']}")
    lines.append(
        f"Audio coverage: {report['total_audio']}/{report['total_segments']} ({report['audio_pct']}%)"
    )
    lines.append(
        f"MIDI coverage:  {report['total_midi']}/{report['total_segments']} ({report['midi_pct']}%)"
    )
    lines.append("")

    # Color table
    lines.append(
        f"{'Color':<12} {'Segments':>8} {'Audio %':>8} {'MIDI %':>8} {'Text %':>8}"
    )
    lines.append("-" * 50)
    for cs in report["by_color"]:
        lines.append(
            f"{cs['color']:<12} {cs['segments']:>8} {cs['audio_pct']:>7.1f}% {cs['midi_pct']:>7.1f}% {cs['text_pct']:>7.1f}%"
        )
    lines.append("")

    # Anomalies
    if report["zero_segment_songs"]:
        lines.append(
            f"WARNING: {len(report['zero_segment_songs'])} songs with 0 segments:"
        )
        for s in report["zero_segment_songs"]:
            lines.append(f"  - {s}")
        lines.append("")

    if report["low_segment_songs"]:
        lines.append(
            f"NOTE: {len(report['low_segment_songs'])} songs with < 3 segments:"
        )
        for s in report["low_segment_songs"]:
            lines.append(f"  - {s['song_id']}: {s['segment_count']} segments")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 3. Fidelity Verification
# ---------------------------------------------------------------------------


def verify_audio_fidelity(
    metadata: pl.DataFrame,
    media_parquet: Path,
    random_n: int = 10,
    row_group: int = 0,
) -> dict:
    """Verify audio data integrity for a sample of segments.

    Checks:
    - Audio bytes decode to valid float32 array
    - Sample rate is reasonable (8000-96000 Hz)
    - Decoded duration is close to expected duration

    Constrains sample to a single row group for speed (each row group is ~6GB).
    """
    # Get segment IDs in the target row group (fast, no binary read)
    rg_ids = set(get_segment_ids_in_row_group(media_parquet, row_group))

    # Sample from segments that have audio AND are in our row group
    audio_segments = metadata.filter(
        pl.col("has_audio") & pl.col("segment_id").is_in(list(rg_ids))
    )
    if len(audio_segments) == 0:
        return {"checked": 0, "passed": 0, "failures": []}

    sample = audio_segments.sample(n=min(random_n, len(audio_segments)), seed=42)
    segment_ids = sample["segment_id"].to_list()
    media = load_media_rows(media_parquet, segment_ids)
    joined = sample.join(media, on="segment_id", how="left")

    results = {"checked": 0, "passed": 0, "failures": []}

    for row in joined.iter_rows(named=True):
        seg_id = row["segment_id"]
        results["checked"] += 1

        audio_bytes = row.get("audio_waveform")
        sample_rate = row.get("audio_sample_rate")
        expected_duration = row.get("duration_seconds")

        if audio_bytes is None:
            results["failures"].append(
                f"{seg_id}: audio_waveform is null despite has_audio=True"
            )
            continue

        if sample_rate is None:
            results["failures"].append(f"{seg_id}: audio_sample_rate is null")
            continue

        # Check sample rate reasonable
        if not (8000 <= sample_rate <= 96000):
            results["failures"].append(f"{seg_id}: unusual sample rate {sample_rate}")
            continue

        # Decode audio
        try:
            audio_array = np.frombuffer(audio_bytes, dtype=np.float32)
        except Exception as e:
            results["failures"].append(f"{seg_id}: failed to decode audio bytes: {e}")
            continue

        if len(audio_array) == 0:
            results["failures"].append(f"{seg_id}: decoded audio is empty")
            continue

        # Check duration
        decoded_duration = len(audio_array) / sample_rate
        if expected_duration is not None:
            duration_diff = abs(decoded_duration - expected_duration)
            if duration_diff > 0.5:
                results["failures"].append(
                    f"{seg_id}: duration mismatch - decoded {decoded_duration:.2f}s vs expected {expected_duration:.2f}s (diff {duration_diff:.2f}s)"
                )
                continue

        # Check for NaN/Inf
        if np.any(np.isnan(audio_array)) or np.any(np.isinf(audio_array)):
            results["failures"].append(f"{seg_id}: audio contains NaN or Inf values")
            continue

        # Check amplitude range (should be roughly -1 to 1 for normalized audio)
        max_amp = np.max(np.abs(audio_array))
        if max_amp > 2.0:
            results["failures"].append(
                f"{seg_id}: audio amplitude out of range (max={max_amp:.2f})"
            )
            continue

        results["passed"] += 1

    return results


def verify_midi_fidelity(
    metadata: pl.DataFrame,
    media_parquet: Path,
    random_n: int = 10,
    row_group: int = 0,
) -> dict:
    """Verify MIDI data integrity for a sample of segments.

    Checks:
    - MIDI bytes parse as valid MIDI file
    - At least one note_on event exists
    - Notes have reasonable pitch values (0-127)

    Constrains sample to a single row group for speed.
    """
    rg_ids = set(get_segment_ids_in_row_group(media_parquet, row_group))

    midi_segments = metadata.filter(
        pl.col("has_midi") & pl.col("segment_id").is_in(list(rg_ids))
    )
    if len(midi_segments) == 0:
        return {"checked": 0, "passed": 0, "failures": []}

    sample = midi_segments.sample(n=min(random_n, len(midi_segments)), seed=42)
    segment_ids = sample["segment_id"].to_list()
    media = load_media_rows(media_parquet, segment_ids)
    joined = sample.join(media, on="segment_id", how="left")

    results = {"checked": 0, "passed": 0, "failures": []}

    for row in joined.iter_rows(named=True):
        seg_id = row["segment_id"]
        results["checked"] += 1

        midi_bytes = row.get("midi_binary")
        if midi_bytes is None:
            results["failures"].append(
                f"{seg_id}: midi_binary is null despite has_midi=True"
            )
            continue

        # Parse MIDI
        try:
            mid = mido.MidiFile(file=BytesIO(midi_bytes))
        except Exception as e:
            results["failures"].append(f"{seg_id}: failed to parse MIDI: {e}")
            continue

        # Check for note_on events
        note_count = 0
        for track in mid.tracks:
            for msg in track:
                if msg.type == "note_on" and msg.velocity > 0:
                    note_count += 1

        if note_count == 0:
            results["failures"].append(f"{seg_id}: MIDI has no note_on events")
            continue

        # Check pitch range
        pitches = set()
        for track in mid.tracks:
            for msg in track:
                if msg.type == "note_on":
                    pitches.add(msg.note)

        if pitches and (min(pitches) < 0 or max(pitches) > 127):
            results["failures"].append(
                f"{seg_id}: MIDI pitches out of range [{min(pitches)}, {max(pitches)}]"
            )
            continue

        results["passed"] += 1

    return results


def format_fidelity_report(audio_results: dict, midi_results: dict) -> str:
    """Format fidelity check results as readable text."""
    lines = []
    lines.append("=" * 70)
    lines.append("FIDELITY VERIFICATION REPORT")
    lines.append("=" * 70)
    lines.append("")

    # Audio
    lines.append(
        f"Audio checks: {audio_results['passed']}/{audio_results['checked']} passed"
    )
    if audio_results["failures"]:
        lines.append("  Failures:")
        for f in audio_results["failures"]:
            lines.append(f"    - {f}")
    lines.append("")

    # MIDI
    lines.append(
        f"MIDI checks:  {midi_results['passed']}/{midi_results['checked']} passed"
    )
    if midi_results["failures"]:
        lines.append("  Failures:")
        for f in midi_results["failures"]:
            lines.append(f"    - {f}")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 4. Main Entry Point
# ---------------------------------------------------------------------------


def run_verification(
    metadata_parquet: Path = DEFAULT_METADATA_PARQUET,
    media_parquet: Path = DEFAULT_MEDIA_PARQUET,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    extract: bool = False,
    fidelity: bool = False,
    color: Optional[str] = None,
    song: Optional[str] = None,
    random_n: Optional[int] = None,
) -> dict:
    """Run verification checks and return combined results.

    Returns dict with pass/fail status and details.
    """
    results = {"passed": True, "checks": {}}

    # Always run coverage report
    logger.info("Loading metadata...")
    metadata = load_metadata(metadata_parquet)
    logger.info(f"Loaded {len(metadata)} segments from metadata parquet")

    logger.info("Generating coverage report...")
    coverage = generate_coverage_report(metadata)
    report_text = format_coverage_report(coverage)
    print(report_text)
    results["checks"]["coverage"] = coverage

    # Flag if any colors have 0 segments
    missing_colors = [
        c["color"]
        for c in coverage["by_color"]
        if c["segments"] == 0 and c["color"] != "UNLABELED"
    ]
    if missing_colors:
        logger.warning(f"Missing colors: {', '.join(missing_colors)}")
        results["passed"] = False
        results["checks"]["missing_colors"] = missing_colors

    # Extraction
    if extract:
        n = random_n or 5
        logger.info(f"Extracting segments (color={color}, song={song}, n={n})...")
        extract_stats = extract_segments(
            metadata,
            media_parquet,
            output_dir,
            color=color,
            song=song,
            random_n=n,
        )
        print(
            f"\nExtracted: {extract_stats['extracted_audio']} audio, {extract_stats['extracted_midi']} MIDI"
        )
        if extract_stats["errors"]:
            print(f"Errors: {len(extract_stats['errors'])}")
            for e in extract_stats["errors"]:
                print(f"  - {e}")
            results["passed"] = False
        print(f"Output directory: {output_dir}")
        results["checks"]["extraction"] = extract_stats

    # Fidelity
    if fidelity:
        n = random_n or 10
        logger.info(f"Running audio fidelity checks (n={n})...")
        audio_results = verify_audio_fidelity(metadata, media_parquet, random_n=n)

        logger.info(f"Running MIDI fidelity checks (n={n})...")
        midi_results = verify_midi_fidelity(metadata, media_parquet, random_n=n)

        fidelity_text = format_fidelity_report(audio_results, midi_results)
        print(fidelity_text)

        if audio_results["failures"] or midi_results["failures"]:
            results["passed"] = False
        results["checks"]["audio_fidelity"] = audio_results
        results["checks"]["midi_fidelity"] = midi_results

    # Final summary
    print("=" * 70)
    if results["passed"]:
        print("RESULT: PASS - All checks passed")
    else:
        print("RESULT: FAIL - Issues detected (see above)")
    print("=" * 70)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Verify training data extraction quality",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--metadata-parquet",
        type=Path,
        default=DEFAULT_METADATA_PARQUET,
        help="Path to metadata parquet",
    )
    parser.add_argument(
        "--media-parquet",
        type=Path,
        default=DEFAULT_MEDIA_PARQUET,
        help="Path to media parquet",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for extracted files",
    )
    parser.add_argument(
        "--extract", action="store_true", help="Extract segments as WAV/MIDI"
    )
    parser.add_argument(
        "--fidelity", action="store_true", help="Run fidelity verification checks"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all checks (coverage + extract 5 + fidelity 10)",
    )
    parser.add_argument("--color", type=str, help="Filter by album color (e.g., Green)")
    parser.add_argument("--song", type=str, help="Filter by song ID (e.g., 05_01)")
    parser.add_argument(
        "--random",
        type=int,
        dest="random_n",
        help="Number of random segments to select",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    if args.all:
        args.extract = True
        args.fidelity = True
        if args.random_n is None:
            args.random_n = 10

    results = run_verification(
        metadata_parquet=args.metadata_parquet,
        media_parquet=args.media_parquet,
        output_dir=args.output_dir,
        extract=args.extract,
        fidelity=args.fidelity,
        color=args.color,
        song=args.song,
        random_n=args.random_n,
    )

    sys.exit(0 if results["passed"] else 1)


if __name__ == "__main__":
    main()
