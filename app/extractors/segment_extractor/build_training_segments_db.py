"""Build training segments database from timestamp extraction pipeline.

This module creates a parquet database with ONE ROW PER TRACK PER SEGMENT.
Each row represents a single audio track's segment at a specific timestamp,
only included if the track has non-silent audio or MIDI notes present.

Architecture:
- Each segment timestamp creates multiple rows (one per active track)
- Silence detection filters out tracks not playing in that segment
- MIDI presence detection for tracks with MIDI files
- All rows inherit full manifest metadata (concept, rebracketing, etc.)

Example: Song with guitar, vocals, fiddle and 10 lyric segments:
  - Guitar: 10 rows (playing throughout)
  - Vocals: 7 rows (silent during intro, active after)
  - Fiddle: 1 row (only plays in final segment)
"""

import logging
import os
from pathlib import Path
from typing import List, Optional, Dict

import numpy as np
import polars as pl
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from app.util.audio_io import load_audio
from app.util.manifest_loader import load_manifest
from app.util.timestamp_audio_extractor import (
    create_segment_specs_from_lrc,
    create_segment_specs_from_structure,
    duration_to_seconds,
)

load_dotenv()
logger = logging.getLogger(__name__)


def is_segment_silent(
    audio_file_path: str,
    start_seconds: float,
    end_seconds: float,
    threshold_db: float = -40.0,
) -> bool:
    """Check if an audio segment is silent (below threshold).

    Args:
        audio_file_path: Path to audio file
        start_seconds: Segment start time
        end_seconds: Segment end time
        threshold_db: Silence threshold in dB (default -40dB)

    Returns:
        True if segment is silent, False if it has audio content
    """
    try:
        # Load audio segment
        audio, sr = load_audio(audio_file_path, sr=None)

        # Calculate sample indices
        start_sample = int(start_seconds * sr)
        end_sample = int(end_seconds * sr)

        # Extract segment
        if audio.ndim == 1:
            segment = audio[start_sample:end_sample]
        else:
            segment = audio[start_sample:end_sample, :]

        if len(segment) == 0:
            return True

        # Calculate RMS energy
        rms = np.sqrt(np.mean(segment**2))

        # Convert to dB
        if rms > 0:
            db = 20 * np.log10(rms)
        else:
            db = -np.inf

        return db < threshold_db

    except Exception as e:
        logger.warning(f"Failed to check silence for {audio_file_path}: {e}")
        return True  # Assume silent if we can't read


def has_midi_notes_in_segment(
    midi_file_path: str, start_seconds: float, end_seconds: float
) -> bool:
    """Check if MIDI file has notes in the given time range.

    Args:
        midi_file_path: Path to MIDI file
        start_seconds: Segment start time
        end_seconds: Segment end time

    Returns:
        True if MIDI has notes in this segment, False otherwise
    """
    try:
        from mido import MidiFile

        midi = MidiFile(midi_file_path)

        # Get tempo for tick conversion
        tempo = 500000  # Default
        ticks_per_beat = midi.ticks_per_beat

        for track in midi.tracks:
            for msg in track:
                if msg.type == "set_tempo":
                    tempo = msg.tempo
                    break

        # Calculate ticks per second
        ticks_per_second = ticks_per_beat * 1000000 / tempo
        start_tick = int(start_seconds * ticks_per_second)
        end_tick = int(end_seconds * ticks_per_second)

        # Check for note_on messages in range
        for track in midi.tracks:
            absolute_tick = 0
            for msg in track:
                absolute_tick += msg.time
                if start_tick <= absolute_tick <= end_tick:
                    if msg.type == "note_on" and msg.velocity > 0:
                        return True

        return False

    except Exception as e:
        logger.warning(f"Failed to check MIDI notes for {midi_file_path}: {e}")
        return False


class BuildTrainingSegmentsDB(BaseModel):
    """Build per-track-per-segment training database."""

    output_dir: Path = Field(
        default_factory=lambda: Path(os.getenv("MANIFEST_PATH", "."))
        / "../training/data"
    )
    segments_output_dir: Path = Field(
        default_factory=lambda: Path(os.getenv("MANIFEST_PATH", "."))
        / "../training/output/track_segments"
    )
    staged_material_dir: Path = Field(
        default_factory=lambda: Path(os.getenv("MANIFEST_PATH", "."))
        / "../staged_raw_material"
    )
    track_filter: Optional[str] = Field(
        default=None,
        description="Optional filter pattern (e.g., '08_*' for Violet album)",
    )
    silence_threshold_db: float = Field(
        default=-40.0, description="Silence detection threshold in dB"
    )
    skip_silence_detection: bool = Field(
        default=False, description="Skip silence detection for faster processing"
    )

    def __init__(self, **data):
        super().__init__(**data)
        if not os.getenv("MANIFEST_PATH"):
            raise ValueError("MANIFEST_PATH environment variable not set")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.segments_output_dir.mkdir(parents=True, exist_ok=True)

    def find_track_directories(self) -> List[Path]:
        """Find all track directories to process."""
        manifest_path = Path(os.getenv("MANIFEST_PATH"))
        track_dirs = [d for d in manifest_path.iterdir() if d.is_dir()]

        if self.track_filter:
            import fnmatch

            track_dirs = [
                d for d in track_dirs if fnmatch.fnmatch(d.name, self.track_filter)
            ]

        return sorted(track_dirs)

    def process_track_directory(self, track_dir: Path) -> List[Dict]:
        """Process a single track directory and generate per-track-per-segment rows.

        Args:
            track_dir: Path to track directory (e.g., staged_raw_material/01_01)

        Returns:
            List of row dictionaries (one per track per segment)
        """
        rows = []

        # Load manifest
        yml_files = list(track_dir.glob("*.yml"))
        if not yml_files:
            logger.warning(f"No manifest found in {track_dir}")
            return rows

        manifest_path = yml_files[0]
        track_id = manifest_path.stem

        try:
            manifest = load_manifest(str(manifest_path))
        except Exception as e:
            logger.error(f"Failed to load manifest {manifest_path}: {e}")
            return rows

        # Find main audio (for structure reference)
        main_audio_path = track_dir / manifest.main_audio_file
        if not main_audio_path.exists():
            logger.warning(f"Main audio not found: {main_audio_path}")
            return rows

        # Find LRC file
        lrc_path = (
            track_dir / manifest.lrc_file
            if manifest.lrc_file
            else track_dir / f"{track_id}.lrc"
        )

        if lrc_path.exists():
            # Create segment specs from LRC + structure
            segment_specs = create_segment_specs_from_lrc(
                str(lrc_path),
                str(main_audio_path),
                manifest=manifest,
                max_segment_length=30.0,
                structure_threshold=2.0,
                overlap_seconds=2.0,
            )
        elif manifest.structure:
            # Fallback: use structure sections for segmentation (instrumental tracks)
            logger.info(
                f"No LRC file for {track_id}, falling back to structure-based segmentation "
                f"({len(manifest.structure)} sections)"
            )
            segment_specs = create_segment_specs_from_structure(
                str(main_audio_path),
                manifest=manifest,
                max_segment_length=30.0,
                overlap_seconds=2.0,
            )
        else:
            logger.warning(
                f"No LRC file and no structure data for {track_id}, skipping"
            )
            return rows

        logger.info(
            f"Processing {track_id}: {len(segment_specs)} segments × {len(manifest.audio_tracks)} tracks"
        )

        # For each segment, process each individual audio track
        for seg_idx, seg_spec in enumerate(segment_specs):
            segment_id_base = f"{track_id}_seg_{seg_idx+1:04d}"

            # Get structure section for this segment
            current_structure_section = None
            if manifest.structure:
                seg_midpoint = (seg_spec.start_seconds + seg_spec.end_seconds) / 2
                for section in manifest.structure:
                    section_start = duration_to_seconds(section.start_time)
                    section_end = duration_to_seconds(section.end_time)
                    if section_start <= seg_midpoint <= section_end:
                        current_structure_section = section.section_name
                        break

            # Process each audio track
            for track in manifest.audio_tracks:
                track_num = track.id

                # Skip tracks without audio files
                if not track.audio_file:
                    logger.debug(f"Track {track_num} has no audio_file, skipping")
                    continue

                audio_file_path = track_dir / track.audio_file

                if not audio_file_path.exists():
                    logger.debug(f"Audio file not found: {audio_file_path}")
                    continue

                # Check if track has audio content in this segment
                if self.skip_silence_detection:
                    has_audio = True  # Assume all tracks have audio
                else:
                    has_audio = not is_segment_silent(
                        str(audio_file_path),
                        seg_spec.start_seconds,
                        seg_spec.end_seconds,
                        threshold_db=self.silence_threshold_db,
                    )

                # Check for MIDI content
                has_midi = False
                midi_file = None
                if hasattr(track, "midi_file") and track.midi_file:
                    midi_path = track_dir / track.midi_file
                    if midi_path.exists():
                        midi_file = str(midi_path)
                        has_midi = has_midi_notes_in_segment(
                            midi_file, seg_spec.start_seconds, seg_spec.end_seconds
                        )
                elif hasattr(track, "midi_group_file") and track.midi_group_file:
                    midi_path = track_dir / track.midi_group_file
                    if midi_path.exists():
                        midi_file = str(midi_path)
                        has_midi = has_midi_notes_in_segment(
                            midi_file, seg_spec.start_seconds, seg_spec.end_seconds
                        )

                # Skip this track-segment if completely silent and no MIDI (unless skipping detection)
                if not self.skip_silence_detection and not has_audio and not has_midi:
                    continue

                # Create segment output path
                segment_audio_file = (
                    self.segments_output_dir
                    / track_id
                    / f"{segment_id_base}_track_{track_num:02d}.wav"
                )

                # Create row
                row = {
                    # Segment identifiers
                    "segment_id": f"{segment_id_base}_track_{track_num:02d}",
                    "segment_index": seg_idx,
                    "track_id": track_id,
                    "track_number": track_num,
                    # Track info
                    "track_description": (
                        str(track.description) if track.description else None
                    ),
                    "track_group": (
                        str(getattr(track, "group", None))
                        if getattr(track, "group", None)
                        else None
                    ),
                    "track_player": (
                        str(getattr(track, "player", None))
                        if getattr(track, "player", None)
                        else None
                    ),
                    # Audio/MIDI paths
                    "source_audio_file": str(audio_file_path),
                    "segment_audio_file": str(segment_audio_file),
                    "midi_file": str(midi_file) if midi_file else None,
                    # Timing
                    "start_seconds": seg_spec.start_seconds,
                    "end_seconds": seg_spec.end_seconds,
                    "duration_seconds": seg_spec.end_seconds - seg_spec.start_seconds,
                    # Content flags
                    "has_audio": has_audio,
                    "has_midi": has_midi,
                    # Lyric/structure context
                    "lyric_text": str(seg_spec.text) if seg_spec.text else "",
                    "structure_section": (
                        str(current_structure_section)
                        if current_structure_section
                        else None
                    ),
                    "segment_type": str(seg_spec.segment_type),
                    # Adjustments
                    "original_start_seconds": seg_spec.metadata.get(
                        "original_start", seg_spec.start_seconds
                    ),
                    "original_end_seconds": seg_spec.metadata.get(
                        "original_end", seg_spec.end_seconds
                    ),
                    "has_structure_adjustments": len(
                        seg_spec.metadata.get("adjustments", [])
                    )
                    > 0,
                    "structure_adjustments": (
                        " | ".join(seg_spec.metadata.get("adjustments", []))
                        if seg_spec.metadata.get("adjustments")
                        else ""
                    ),
                    "is_sub_segment": bool("sub_segment" in seg_spec.metadata),
                    "sub_segment_info": (
                        str(seg_spec.metadata.get("sub_segment"))
                        if seg_spec.metadata.get("sub_segment")
                        else None
                    ),
                    # LRC reference
                    "lrc_line_number": (
                        int(seg_spec.metadata.get("lrc_line"))
                        if seg_spec.metadata.get("lrc_line")
                        else None
                    ),
                }

                rows.append(row)

        logger.info(f"  Generated {len(rows)} track-segment rows for {track_id}")
        return rows

    def build_parquet(self) -> Path:
        """Build the complete training segments database.

        Returns:
            Path to generated parquet file
        """
        all_rows = []

        # Process each track directory
        track_dirs = self.find_track_directories()
        logger.info(f"Found {len(track_dirs)} track directories to process")

        for track_dir in track_dirs:
            logger.info(f"Processing {track_dir.name}...")
            rows = self.process_track_directory(track_dir)
            all_rows.extend(rows)

        if not all_rows:
            raise ValueError("No track-segment rows generated")

        # Create DataFrame with increased schema inference to handle mixed types
        df = pl.DataFrame(all_rows, infer_schema_length=10000)

        # Add computed features
        df = df.with_columns(
            [
                # Lyric features
                pl.col("lyric_text").str.len_chars().alias("lyric_char_count"),
                pl.col("lyric_text")
                .str.split(" ")
                .list.len()
                .alias("lyric_word_count"),
                # Adjustment features
                (pl.col("start_seconds") - pl.col("original_start_seconds")).alias(
                    "start_adjustment_seconds"
                ),
                (pl.col("end_seconds") - pl.col("original_end_seconds")).alias(
                    "end_adjustment_seconds"
                ),
                # Content type
                pl.when(pl.col("has_audio") & pl.col("has_midi"))
                .then(pl.lit("audio_midi"))
                .when(pl.col("has_audio"))
                .then(pl.lit("audio_only"))
                .when(pl.col("has_midi"))
                .then(pl.lit("midi_only"))
                .otherwise(pl.lit("empty"))
                .alias("content_type"),
            ]
        )

        # Save to parquet
        output_path = self.output_dir / "training_segments.parquet"
        df.write_parquet(output_path, compression="snappy")

        logger.info(f"✅ Training segments database saved: {output_path}")
        logger.info(f"   Rows: {len(df):,}")
        logger.info(f"   Columns: {len(df.columns)}")
        logger.info(f"   Size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")

        self.print_summary(df)

        return output_path

    def print_summary(self, df: pl.DataFrame):
        """Print summary statistics."""
        logger.info("\n" + "=" * 70)
        logger.info("Training Segments Database Summary (Per-Track-Per-Segment)")
        logger.info("=" * 70)

        logger.info(f"Total track-segment rows: {len(df):,}")
        # Handle both old and new column names for backwards compatibility
        song_col = "song_id" if "song_id" in df.columns else "track_id"
        logger.info(f"Unique songs: {df[song_col].n_unique()}")
        logger.info(f"Unique segments (timestamps): {df['segment_index'].n_unique()}")
        logger.info(
            f"Average tracks per segment: {len(df) / df['segment_index'].n_unique():.1f}"
        )

        logger.info("\nContent Type Distribution:")
        for row in (
            df.group_by("content_type")
            .agg(pl.len().alias("count"))
            .sort("count", descending=True)
            .iter_rows(named=True)
        ):
            logger.info(f"  {row['content_type']}: {row['count']:,}")

        logger.info("\nTop 10 Track Descriptions:")
        for row in (
            df.group_by("track_description")
            .agg(pl.len().alias("count"))
            .sort("count", descending=True)
            .head(10)
            .iter_rows(named=True)
        ):
            logger.info(f"  {row['track_description']}: {row['count']:,} segments")

        logger.info("\nDuration Statistics:")
        logger.info(f"  Mean: {df['duration_seconds'].mean():.2f}s")
        logger.info(f"  Median: {df['duration_seconds'].median():.2f}s")
        logger.info(f"  Min: {df['duration_seconds'].min():.2f}s")
        logger.info(f"  Max: {df['duration_seconds'].max():.2f}s")

        logger.info("=" * 70)


def join_with_manifest_db(
    segments_df: pl.DataFrame, manifest_db_path: Path
) -> pl.DataFrame:
    """Join segment data with track-level manifest metadata.

    This creates the final training dataset where each track-segment row
    has all the manifest metadata (concept, rebracketing, mood, etc.)

    Args:
        segments_df: Per-track-per-segment DataFrame
        manifest_db_path: Path to base_manifest_db.parquet

    Returns:
        Joined DataFrame ready for training
    """
    manifest_df = pl.read_parquet(manifest_db_path)

    # Join on track_id AND track_number (track_id in manifest)
    # Note: manifest has 'track_id' column that's actually the individual track identifier
    # We need to match our segments (track_id + track_number) to manifest rows

    # First, create a composite key in segments
    segments_with_key = segments_df.with_columns(
        [
            (pl.col("track_id") + "_" + pl.col("track_number").cast(pl.Utf8)).alias(
                "manifest_track_key"
            )
        ]
    )

    # Create matching key in manifest (id column is like "01_01", track_id column is the track number)
    manifest_with_key = manifest_df.with_columns(
        [
            (pl.col("id") + "_" + pl.col("track_id").cast(pl.Utf8)).alias(
                "manifest_track_key"
            )
        ]
    )

    # Join
    joined = segments_with_key.join(
        manifest_with_key, on="manifest_track_key", how="left"
    )

    logger.info(
        f"✅ Joined with manifest DB: {len(joined):,} rows, {len(joined.columns)} columns"
    )

    # Clean up duplicate and confusing columns
    joined = clean_joined_schema(joined)

    logger.info(f"   After cleanup: {len(joined.columns)} columns")

    return joined


def clean_joined_schema(df: pl.DataFrame) -> pl.DataFrame:
    """Remove duplicate columns and rename confusing ones after join.

    Duplicates to remove:
    - description (same as track_description)
    - player (same as track_player)
    - group (same as track_group)
    - midi_file_right (same as midi_file)
    - track_id_right (redundant with track_number)
    - id (duplicate of track_id)

    Renames for clarity:
    - track_id -> song_id (it's the song identifier like "01_01")
    """
    # Remove duplicate columns
    columns_to_drop = [
        "description",  # duplicate of track_description
        "player",  # duplicate of track_player
        "group",  # duplicate of track_group
        "midi_file_right",  # duplicate of midi_file
        "track_id_right",  # redundant with track_number
        "id",  # duplicate of track_id (song ID)
        "audio_file",  # manifest path, we have source_audio_file
    ]

    # Drop columns that exist
    existing_drops = [c for c in columns_to_drop if c in df.columns]
    df = df.drop(existing_drops)

    # Rename for clarity
    df = df.rename(
        {
            "track_id": "song_id",  # Make it clear this is song ID like "01_01"
        }
    )

    return df


def embed_audio_and_midi_binaries(
    df: pl.DataFrame, staged_material_dir: Path, sample_rate: int = 44100
) -> pl.DataFrame:
    """Extract and embed audio/MIDI segments as binary data in the dataframe.

    This creates a self-contained training dataset with embedded binaries,
    perfect for transferring to training VMs like RunPod.

    Args:
        df: Training segments dataframe
        staged_material_dir: Path to staged_raw_material directory
        sample_rate: Target sample rate for audio (default: 44100)

    Returns:
        DataFrame with embedded audio_waveform and midi_binary columns
    """
    import numpy as np
    from app.util.audio_io import load_audio

    logger.info("\n" + "=" * 70)
    logger.info("Embedding Audio and MIDI Binaries")
    logger.info("=" * 70)

    audio_waveforms = []
    audio_sample_rates = []
    midi_binaries = []

    total_rows = len(df)
    logger.info(f"Processing {total_rows:,} track-segment rows...")

    for i, row in enumerate(df.iter_rows(named=True), 1):
        if i % 100 == 0:
            logger.info(f"  Progress: {i:,} / {total_rows:,} ({i/total_rows*100:.1f}%)")

        # Extract audio segment
        audio_waveform = None
        sr = None

        if row["has_audio"] and row["source_audio_file"]:
            try:
                audio_file = Path(row["source_audio_file"])
                if audio_file.exists():
                    # Load and extract segment
                    audio, sr = load_audio(str(audio_file), sr=sample_rate)
                    start_sample = int(row["start_seconds"] * sr)
                    end_sample = int(row["end_seconds"] * sr)
                    segment = audio[start_sample:end_sample]

                    # Convert to bytes (float32)
                    audio_waveform = segment.astype(np.float32).tobytes()
                else:
                    logger.warning(f"Audio file not found: {audio_file}")
            except Exception as e:
                logger.warning(f"Failed to load audio for row {i}: {e}")

        audio_waveforms.append(audio_waveform)
        audio_sample_rates.append(sr if sr else sample_rate)

        # Load MIDI file as binary
        midi_binary = None

        if row["has_midi"] and row["midi_file"]:
            try:
                midi_file = Path(row["midi_file"])
                if midi_file.exists():
                    with open(midi_file, "rb") as f:
                        midi_binary = f.read()
                else:
                    logger.debug(f"MIDI file not found: {midi_file}")
            except Exception as e:
                logger.warning(f"Failed to load MIDI for row {i}: {e}")

        midi_binaries.append(midi_binary)

    # Add binary columns to dataframe
    df = df.with_columns(
        [
            pl.Series("audio_waveform", audio_waveforms, dtype=pl.Binary),
            pl.Series("audio_sample_rate", audio_sample_rates, dtype=pl.Int32),
            pl.Series("midi_binary", midi_binaries, dtype=pl.Binary),
        ]
    )

    # Calculate sizes
    audio_count = sum(1 for w in audio_waveforms if w is not None)
    midi_count = sum(1 for m in midi_binaries if m is not None)

    logger.info("\n✅ Embedded binaries:")
    logger.info(f"   Audio segments: {audio_count:,} / {total_rows:,}")
    logger.info(f"   MIDI files: {midi_count:,} / {total_rows:,}")

    return df


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Build per-track-per-segment training database"
    )
    parser.add_argument(
        "--track-filter",
        type=str,
        help='Filter pattern (e.g., "08_*" for Violet album)',
    )
    parser.add_argument(
        "--silence-threshold",
        type=float,
        default=-40.0,
        help="Silence threshold in dB (default: -40.0)",
    )
    parser.add_argument(
        "--skip-silence-detection",
        action="store_true",
        help="Skip silence detection for faster processing (include all tracks)",
    )
    parser.add_argument(
        "--join-manifest",
        action="store_true",
        help="Create joined parquet with full manifest metadata",
    )
    parser.add_argument(
        "--embed-binaries",
        action="store_true",
        help="Embed audio waveforms and MIDI files as binary data (creates self-contained dataset)",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=44100,
        help="Sample rate for embedded audio (default: 44100 Hz)",
    )

    args = parser.parse_args()

    builder = BuildTrainingSegmentsDB(
        track_filter=args.track_filter,
        silence_threshold_db=args.silence_threshold,
        skip_silence_detection=args.skip_silence_detection,
    )

    # Build segment database
    segments_path = builder.build_parquet()
    print(f"\n✅ Training segments database: {segments_path}")

    # Optional: create joined database
    if args.join_manifest:
        manifest_path = builder.output_dir / "base_manifest_db.parquet"
        if manifest_path.exists():
            segments_df = pl.read_parquet(segments_path)
            joined_df = join_with_manifest_db(segments_df, manifest_path)

            joined_path = builder.output_dir / "training_segments_metadata.parquet"
            joined_df.write_parquet(joined_path, compression="snappy")

            print(f"✅ Full training data: {joined_path}")
            print(f"   Rows: {len(joined_df):,}, Columns: {len(joined_df.columns)}")

            # Optional: embed binaries for self-contained dataset
            if args.embed_binaries:
                print("\n🔄 Embedding audio and MIDI binaries...")
                embedded_df = embed_audio_and_midi_binaries(
                    joined_df, builder.staged_material_dir, sample_rate=args.sample_rate
                )

                embedded_path = builder.output_dir / "training_segments_media.parquet"

                print(
                    "\n📝 Writing parquet file (this may take several minutes for large datasets)..."
                )
                try:
                    # Use zstd compression for better binary data compression
                    # and write with row_group_size to handle large files better
                    embedded_df.write_parquet(
                        embedded_path,
                        compression="zstd",
                        compression_level=3,  # Balance between speed and compression
                        row_group_size=1000,  # Write in chunks for better memory handling
                    )

                    file_size_gb = embedded_path.stat().st_size / (1024**3)
                    print(f"\n✅ Self-contained training data: {embedded_path}")
                    print(
                        f"   Rows: {len(embedded_df):,}, Columns: {len(embedded_df.columns)}"
                    )
                    print(f"   Size: {file_size_gb:.2f} GB")
                    print(
                        "\n💡 This file contains embedded audio/MIDI and is ready to transfer to RunPod!"
                    )
                except Exception as e:
                    print(f"\n❌ Error writing parquet file: {e}")
                    print("   File may be incomplete or corrupted")
                    raise
        else:
            print(f"⚠️  Manifest DB not found: {manifest_path}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    main()
