#!/usr/bin/env python3
"""Example script demonstrating the timestamp audio extraction pipeline.

This script shows how to:
1. Process a single track directory
2. Process multiple tracks with custom configuration
3. Generate and review metadata

Usage:
    python examples/timestamp_extraction_example.py
"""

import json
import logging
from pathlib import Path

from app.util.timestamp_pipeline import (
    PipelineConfig,
    process_staged_raw_material,
    process_track_directory,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


def example_single_track():
    """Example: Process a single track."""
    logger.info("=" * 60)
    logger.info("Example 1: Processing Single Track")
    logger.info("=" * 60)

    # Configure pipeline
    config = PipelineConfig(
        max_segment_length_seconds=30.0,
        structure_alignment_threshold_seconds=2.0,
        overlap_seconds=2.0,
        extract_midi=True,
        output_metadata=True,
        sample_rate=None,  # Keep original sample rate
    )

    # Process track 01_01 (The Conjurer's Thread)
    result = process_track_directory(
        track_dir="/Volumes/LucidNonsense/White/staged_raw_material/01_01",
        output_dir="/Volumes/LucidNonsense/White/training/output/example_segments",
        config=config,
    )

    if result["success"]:
        logger.info(f"✅ Successfully processed track {result['track_id']}")
        logger.info(f"   Segments extracted: {result['segments_extracted']}")
        logger.info(f"   Total duration: {result['total_duration_seconds']:.1f}s")
        logger.info(f"   Structure adjustments: {result['structure_adjusted_count']}")
        logger.info(f"   Sub-segments (split): {result['sub_segment_count']}")
        logger.info(f"   Output: {result['output_directory']}")

        # Show sample metadata
        if result["metadata_file"]:
            with open(result["metadata_file"], "r") as f:
                metadata = json.load(f)

            logger.info("\n   First segment metadata:")
            logger.info(f"     ID: {metadata[0]['segment_id']}")
            logger.info(f"     Duration: {metadata[0]['duration_seconds']:.3f}s")
            logger.info(f"     Lyric: \"{metadata[0]['lyric_text']}\"")
            logger.info(f"     Type: {metadata[0]['segment_type']}")
            if metadata[0]["structure_adjustments"]:
                logger.info(f"     Adjustments: {metadata[0]['structure_adjustments']}")
    else:
        logger.error(f"❌ Failed: {result.get('error')}")


def example_batch_processing():
    """Example: Process multiple tracks from the Violet album."""
    logger.info("\n" + "=" * 60)
    logger.info("Example 2: Batch Processing Violet Album Tracks")
    logger.info("=" * 60)

    # More aggressive configuration for longer instrumental sections
    config = PipelineConfig(
        max_segment_length_seconds=45.0,  # Longer segments
        structure_alignment_threshold_seconds=3.0,  # More aggressive alignment
        overlap_seconds=3.0,
        extract_midi=True,
        output_metadata=True,
        sample_rate=44100,  # Resample to 44.1kHz
    )

    # Process all Violet album tracks (08_*)
    result = process_staged_raw_material(
        staged_dir="staged_raw_material",
        output_dir="output/violet_segments",
        config=config,
        track_filter="08_*",
    )

    if result.get("success", True):
        logger.info("✅ Batch processing complete")
        logger.info(f"   Tracks processed: {result['total_tracks_processed']}")
        logger.info(f"   Successful: {result['successful_tracks']}")
        logger.info(f"   Failed: {result['failed_tracks']}")
        logger.info(f"   Total segments: {result['total_segments_extracted']}")
        logger.info(f"   Total duration: {result['total_duration_seconds']:.1f}s")

        if result["failed_tracks"] > 0:
            logger.warning("\n   Failed tracks:")
            for fail in result["failed_track_details"]:
                logger.warning(f"     - {fail['directory']}: {fail['error']}")
    else:
        logger.error(f"❌ Failed: {result.get('error')}")


def example_custom_processing():
    """Example: Custom processing with specific tracks and settings."""
    logger.info("\n" + "=" * 60)
    logger.info("Example 3: Custom Processing Configuration")
    logger.info("=" * 60)

    # Minimal segments for testing
    config = PipelineConfig(
        max_segment_length_seconds=10.0,  # Very short segments
        structure_alignment_threshold_seconds=0.5,  # Tight alignment
        overlap_seconds=1.0,  # Minimal overlap
        extract_midi=False,  # Skip MIDI for speed
        output_metadata=True,
        sample_rate=22050,  # Lower sample rate for faster processing
    )

    # Process just the first track
    result = process_track_directory(
        track_dir="staged_raw_material/01_01",
        output_dir="output/test_segments",
        config=config,
    )

    if result["success"]:
        logger.info("✅ Test processing complete")
        logger.info(f"   Segments: {result['segments_extracted']}")
        logger.info("   Note: With 10s max length, many segments were split")
    else:
        logger.error(f"❌ Failed: {result.get('error')}")


def review_output_structure():
    """Review the output directory structure."""
    logger.info("\n" + "=" * 60)
    logger.info("Output Directory Structure")
    logger.info("=" * 60)

    output_base = Path("output/example_segments")

    if not output_base.exists():
        logger.warning("No output directory found. Run example_single_track() first.")
        return

    # Show directory tree
    for track_dir in sorted(output_base.iterdir()):
        if track_dir.is_dir():
            logger.info(f"\n{track_dir.name}/")

            # List audio segments
            wav_files = sorted(track_dir.glob("*.wav"))
            logger.info(f"  Audio segments: {len(wav_files)}")
            for wav in wav_files[:3]:  # Show first 3
                logger.info(f"    - {wav.name}")
            if len(wav_files) > 3:
                logger.info(f"    ... and {len(wav_files) - 3} more")

            # Check for MIDI directory
            midi_dir = track_dir / "midi"
            if midi_dir.exists():
                midi_files = sorted(midi_dir.glob("*.mid"))
                logger.info(f"  MIDI segments: {len(midi_files)}")

            # Check for metadata
            metadata_file = track_dir / f"{track_dir.name}_segments_metadata.json"
            if metadata_file.exists():
                logger.info(f"  Metadata: {metadata_file.name}")


def main():
    """Run all examples."""
    logger.info("Timestamp Audio Extraction Pipeline Examples")
    logger.info("=" * 60)

    try:
        # Example 1: Single track
        example_single_track()

        # Example 2: Batch processing
        # Uncomment to run (processes multiple tracks):
        # example_batch_processing()

        # Example 3: Custom configuration
        example_custom_processing()

        # Review output
        review_output_structure()

        logger.info("\n" + "=" * 60)
        logger.info("All examples completed successfully!")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Error running examples: {e}", exc_info=True)
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
