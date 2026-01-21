"""Integrated timestamp audio extraction pipeline.

This module provides the complete end-to-end pipeline for extracting training-ready
audio segments from Logic Pro exports using LRC timestamps, manifest structure data,
and MIDI files.

Workflow:
1. Load manifest (structure, track info)
2. Parse LRC file (lyrics with timestamps)
3. Create segment specifications with structure-aware boundaries
4. Extract audio segments with max length constraints
5. Extract corresponding MIDI segments
6. Generate metadata for each segment
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

from app.util.manifest_loader import load_manifest
from app.util.midi_segment_utils import segment_midi_with_audio
from app.util.timestamp_audio_extractor import (
    create_segment_specs_from_lrc,
    extract_audio_segment,
)

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for the timestamp extraction pipeline."""

    max_segment_length_seconds: float = 30.0
    structure_alignment_threshold_seconds: float = 2.0
    overlap_seconds: float = 2.0
    extract_midi: bool = True
    output_metadata: bool = True
    sample_rate: Optional[int] = None  # None = keep original


@dataclass
class SegmentMetadata:
    """Metadata for an extracted segment."""

    segment_id: str
    track_id: str
    audio_file: str
    midi_files: List[str]
    start_seconds: float
    end_seconds: float
    duration_seconds: float
    lyric_text: str
    segment_type: str  # "lyric", "structure", or "combined"
    structure_adjustments: List[str]
    original_start: float
    original_end: float
    lrc_line_number: Optional[int]
    is_sub_segment: bool
    sub_segment_info: Optional[str]


def process_track_directory(
    track_dir: str, output_dir: str, config: Optional[PipelineConfig] = None
) -> Dict[str, any]:
    """Process a single track directory through the complete pipeline.

    Expected directory structure:
        track_dir/
            track_id.yml          - Manifest file
            track_id.lrc          - LRC lyrics file
            track_id_main.wav     - Main audio file
            track_id_##_*.wav     - Individual track audio files
            *.mid                 - MIDI files (optional)

    Args:
        track_dir: Path to track directory
        output_dir: Output directory for extracted segments
        config: Pipeline configuration (uses defaults if None)

    Returns:
        Dictionary with processing results and statistics
    """
    if config is None:
        config = PipelineConfig()

    track_dir = Path(track_dir)
    output_dir = Path(output_dir)

    # Find manifest file
    yml_files = list(track_dir.glob("*.yml"))
    if not yml_files:
        logger.error(f"No manifest (.yml) file found in {track_dir}")
        return {"success": False, "error": "No manifest file found"}

    manifest_path = yml_files[0]
    track_id = manifest_path.stem

    logger.info(f"Processing track {track_id} from {track_dir}")

    # Load manifest
    try:
        manifest = load_manifest(str(manifest_path))
    except Exception as e:
        logger.error(f"Failed to load manifest {manifest_path}: {e}")
        return {"success": False, "error": f"Failed to load manifest: {e}"}

    # Find LRC file
    lrc_path = (
        track_dir / manifest.lrc_file
        if manifest.lrc_file
        else track_dir / f"{track_id}.lrc"
    )
    if not lrc_path.exists():
        logger.error(f"LRC file not found: {lrc_path}")
        return {"success": False, "error": f"LRC file not found: {lrc_path}"}

    # Find main audio file
    audio_path = track_dir / manifest.main_audio_file
    if not audio_path.exists():
        logger.error(f"Main audio file not found: {audio_path}")
        return {"success": False, "error": f"Main audio file not found: {audio_path}"}

    # Create segment specifications
    logger.info(f"Creating segment specifications from {lrc_path}")
    segment_specs = create_segment_specs_from_lrc(
        str(lrc_path),
        str(audio_path),
        manifest=manifest,
        max_segment_length=config.max_segment_length_seconds,
        structure_threshold=config.structure_alignment_threshold_seconds,
        overlap_seconds=config.overlap_seconds,
    )

    if not segment_specs:
        logger.warning(f"No segments created for track {track_id}")
        return {
            "success": True,
            "segments_extracted": 0,
            "warning": "No segments created",
        }

    logger.info(f"Created {len(segment_specs)} segment specifications")

    # Create output directory
    track_output_dir = output_dir / track_id
    track_output_dir.mkdir(parents=True, exist_ok=True)

    # Extract segments
    extracted_segments = []
    metadata_list = []

    for i, spec in enumerate(segment_specs):
        segment_id = f"{track_id}_seg_{i+1:04d}"

        # Extract audio segment
        audio_output_path = track_output_dir / f"{segment_id}.wav"
        audio_success = extract_audio_segment(
            spec.source_file,
            spec.start_seconds,
            spec.end_seconds,
            str(audio_output_path),
            sample_rate=config.sample_rate,
        )

        if not audio_success:
            logger.warning(f"Failed to extract audio segment {segment_id}")
            continue

        # Extract MIDI segments if configured
        midi_files = []
        if config.extract_midi and manifest.audio_tracks:
            midi_output_dir = track_output_dir / "midi"
            midi_output_dir.mkdir(exist_ok=True)

            midi_files = segment_midi_with_audio(
                str(audio_path),
                spec.start_seconds,
                spec.end_seconds,
                str(midi_output_dir),
                manifest_tracks=manifest.audio_tracks,
                output_prefix=segment_id,
            )

        # Create metadata
        metadata = SegmentMetadata(
            segment_id=segment_id,
            track_id=track_id,
            audio_file=str(audio_output_path),
            midi_files=[str(f) for f in midi_files],
            start_seconds=spec.start_seconds,
            end_seconds=spec.end_seconds,
            duration_seconds=spec.end_seconds - spec.start_seconds,
            lyric_text=spec.text,
            segment_type=spec.segment_type,
            structure_adjustments=spec.metadata.get("adjustments", []),
            original_start=spec.metadata.get("original_start", spec.start_seconds),
            original_end=spec.metadata.get("original_end", spec.end_seconds),
            lrc_line_number=spec.metadata.get("lrc_line"),
            is_sub_segment="sub_segment" in spec.metadata,
            sub_segment_info=spec.metadata.get("sub_segment"),
        )

        extracted_segments.append(audio_output_path)
        metadata_list.append(metadata)

    # Write metadata file if configured
    if config.output_metadata:
        metadata_path = track_output_dir / f"{track_id}_segments_metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(
                [asdict(m) for m in metadata_list], f, indent=2, ensure_ascii=False
            )
        logger.info(f"Wrote metadata to {metadata_path}")

    # Generate summary
    total_duration = sum(m.duration_seconds for m in metadata_list)
    structure_adjusted = sum(1 for m in metadata_list if m.structure_adjustments)
    sub_segments = sum(1 for m in metadata_list if m.is_sub_segment)

    result = {
        "success": True,
        "track_id": track_id,
        "segments_extracted": len(extracted_segments),
        "total_duration_seconds": total_duration,
        "structure_adjusted_count": structure_adjusted,
        "sub_segment_count": sub_segments,
        "output_directory": str(track_output_dir),
        "metadata_file": str(metadata_path) if config.output_metadata else None,
    }

    logger.info(
        f"Completed processing track {track_id}: "
        f"{len(extracted_segments)} segments, "
        f"{total_duration:.1f}s total, "
        f"{structure_adjusted} structure-adjusted, "
        f"{sub_segments} sub-segments"
    )

    return result


def process_multiple_tracks(
    track_dirs: List[str], output_dir: str, config: Optional[PipelineConfig] = None
) -> Dict[str, any]:
    """Process multiple track directories through the pipeline.

    Args:
        track_dirs: List of track directory paths
        output_dir: Base output directory
        config: Pipeline configuration

    Returns:
        Dictionary with aggregate results and per-track details
    """
    if config is None:
        config = PipelineConfig()

    results = []
    total_segments = 0
    total_duration = 0.0
    failed_tracks = []

    for track_dir in track_dirs:
        logger.info(f"Processing track directory: {track_dir}")
        result = process_track_directory(track_dir, output_dir, config)

        if result["success"]:
            total_segments += result.get("segments_extracted", 0)
            total_duration += result.get("total_duration_seconds", 0.0)
        else:
            failed_tracks.append(
                {"directory": track_dir, "error": result.get("error", "Unknown error")}
            )

        results.append(result)

    summary = {
        "total_tracks_processed": len(track_dirs),
        "successful_tracks": len([r for r in results if r["success"]]),
        "failed_tracks": len(failed_tracks),
        "total_segments_extracted": total_segments,
        "total_duration_seconds": total_duration,
        "failed_track_details": failed_tracks,
        "per_track_results": results,
    }

    logger.info(
        f"Batch processing complete: "
        f"{summary['successful_tracks']}/{len(track_dirs)} tracks successful, "
        f"{total_segments} total segments, "
        f"{total_duration:.1f}s total duration"
    )

    return summary


def process_staged_raw_material(
    staged_dir: str,
    output_dir: str,
    config: Optional[PipelineConfig] = None,
    track_filter: Optional[str] = None,
) -> Dict[str, any]:
    """Process all tracks in staged_raw_material directory.

    Args:
        staged_dir: Path to staged_raw_material directory
        output_dir: Output directory for processed segments
        config: Pipeline configuration
        track_filter: Optional filter pattern (e.g., "08_*" for violet album tracks)

    Returns:
        Dictionary with processing summary
    """
    staged_path = Path(staged_dir)

    if not staged_path.exists():
        logger.error(f"Staged directory not found: {staged_dir}")
        return {"success": False, "error": "Staged directory not found"}

    # Find all track directories
    track_dirs = [d for d in staged_path.iterdir() if d.is_dir()]

    # Apply filter if provided
    if track_filter:
        import fnmatch

        track_dirs = [d for d in track_dirs if fnmatch.fnmatch(d.name, track_filter)]

    logger.info(f"Found {len(track_dirs)} track directories to process")

    if not track_dirs:
        return {"success": False, "error": "No track directories found"}

    # Process all tracks
    return process_multiple_tracks([str(d) for d in track_dirs], output_dir, config)
