"""Audio segment extraction utilities for timestamp-based pipeline.

This module provides functionality to extract audio segments based on:
- LRC timestamps with lyrics
- Manifest structure boundaries (verse, chorus, bridge, etc.)
- Maximum segment length constraints
- MIDI file segmentation aligned with audio
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import soundfile as sf

from app.structures.manifests.manifest import Manifest
from app.structures.manifests.manifest_song_structure import ManifestSongStructure
from app.structures.music.core.duration import Duration
from app.util.audio_io import load_audio
from app.util.lrc_utils import load_lrc

logger = logging.getLogger(__name__)


@dataclass
class AudioSegmentSpec:
    """Specification for an audio segment to extract."""

    start_seconds: float
    end_seconds: float
    text: str
    source_file: str
    segment_type: str  # "lyric", "structure", or "combined"
    metadata: (
        dict  # Additional metadata like structure section, original timestamps, etc.
    )


def duration_to_seconds(duration: Duration | str) -> float:
    """Convert a Duration object or string to seconds.

    Args:
        duration: Duration object or string in [MM:SS.mmm] format

    Returns:
        Total seconds as float
    """
    if isinstance(duration, str):
        duration = Duration.model_validate(duration)
    return duration.minutes * 60 + duration.seconds


def find_nearest_structure_boundary(
    timestamp_seconds: float,
    structure: List[ManifestSongStructure],
    threshold_seconds: float = 2.0,
) -> Optional[Tuple[float, str]]:
    """Find the nearest structure boundary within threshold distance.

    Args:
        timestamp_seconds: Timestamp to check in seconds
        structure: List of structure sections from manifest
        threshold_seconds: Maximum distance to consider (default 2.0 seconds)

    Returns:
        Tuple of (boundary_time_seconds, section_name) if found within threshold,
        otherwise None
    """
    closest_boundary = None
    closest_distance = float("inf")
    closest_section = None

    for section in structure:
        start_sec = duration_to_seconds(section.start_time)
        end_sec = duration_to_seconds(section.end_time)

        # Check distance to start boundary
        dist_to_start = abs(timestamp_seconds - start_sec)
        if dist_to_start < closest_distance and dist_to_start <= threshold_seconds:
            closest_distance = dist_to_start
            closest_boundary = start_sec
            closest_section = section.section_name

        # Check distance to end boundary
        dist_to_end = abs(timestamp_seconds - end_sec)
        if dist_to_end < closest_distance and dist_to_end <= threshold_seconds:
            closest_distance = dist_to_end
            closest_boundary = end_sec
            closest_section = section.section_name

    if closest_boundary is not None:
        return (closest_boundary, closest_section)
    return None


def adjust_segment_to_structure(
    start_seconds: float,
    end_seconds: float,
    structure: List[ManifestSongStructure],
    threshold_seconds: float = 2.0,
) -> Tuple[float, float, dict]:
    """Adjust segment boundaries to align with structure sections when close.

    This implements the structure-aware segmentation requirement where segments
    are extended to structure boundaries when within threshold distance.

    Args:
        start_seconds: Original start time in seconds
        end_seconds: Original end time in seconds
        structure: List of structure sections from manifest
        threshold_seconds: Maximum distance to snap to boundary (default 2.0s)

    Returns:
        Tuple of (adjusted_start, adjusted_end, metadata_dict)
    """
    adjusted_start = start_seconds
    adjusted_end = end_seconds
    metadata = {
        "original_start": start_seconds,
        "original_end": end_seconds,
        "adjustments": [],
    }

    # Check if start is near a structure boundary
    start_boundary = find_nearest_structure_boundary(
        start_seconds, structure, threshold_seconds
    )
    if start_boundary:
        boundary_time, section_name = start_boundary
        adjusted_start = boundary_time
        metadata["adjustments"].append(
            f"Start aligned to {section_name} boundary at {boundary_time:.3f}s"
        )

    # Check if end is near a structure boundary
    end_boundary = find_nearest_structure_boundary(
        end_seconds, structure, threshold_seconds
    )
    if end_boundary:
        boundary_time, section_name = end_boundary
        adjusted_end = boundary_time
        metadata["adjustments"].append(
            f"End aligned to {section_name} boundary at {boundary_time:.3f}s"
        )

    return adjusted_start, adjusted_end, metadata


def split_long_segment(
    start_seconds: float,
    end_seconds: float,
    max_length_seconds: float,
    overlap_seconds: float = 2.0,
) -> List[Tuple[float, float]]:
    """Split a long segment into multiple overlapping sub-segments.

    Args:
        start_seconds: Segment start time
        end_seconds: Segment end time
        max_length_seconds: Maximum length for each sub-segment
        overlap_seconds: Overlap duration between sub-segments (default 2.0s)

    Returns:
        List of (start, end) tuples for each sub-segment
    """
    duration = end_seconds - start_seconds

    if duration <= max_length_seconds:
        return [(start_seconds, end_seconds)]

    segments = []
    current_start = start_seconds
    step_size = max_length_seconds - overlap_seconds

    while current_start < end_seconds:
        current_end = min(current_start + max_length_seconds, end_seconds)
        segments.append((current_start, current_end))

        if current_end >= end_seconds:
            break

        current_start += step_size

    return segments


def create_segment_specs_from_lrc(
    lrc_file_path: str,
    audio_file_path: str,
    manifest: Optional[Manifest] = None,
    max_segment_length: Optional[float] = 30.0,
    structure_threshold: float = 2.0,
    overlap_seconds: float = 2.0,
) -> List[AudioSegmentSpec]:
    """Create audio segment specifications from LRC file with optional structure alignment.

    Args:
        lrc_file_path: Path to LRC file
        audio_file_path: Path to corresponding audio file
        manifest: Optional Manifest object with structure data
        max_segment_length: Maximum segment length in seconds (default 30.0)
        structure_threshold: Distance threshold for structure alignment (default 2.0s)
        overlap_seconds: Overlap for split segments (default 2.0s)

    Returns:
        List of AudioSegmentSpec objects
    """
    # Parse LRC file
    lrc_entries = load_lrc(lrc_file_path)
    if not lrc_entries:
        logger.warning(f"No entries found in LRC file: {lrc_file_path}")
        return []

    # Get audio duration for final segment
    try:
        info = sf.info(audio_file_path)
        audio_duration = info.duration
    except Exception as e:
        logger.error(f"Failed to get audio info from {audio_file_path}: {e}")
        audio_duration = None

    segment_specs = []

    for i, entry in enumerate(lrc_entries):
        start_sec = entry["start_time"]
        end_sec = entry.get("end_time")

        # If end_sec is None and we have audio duration, use it
        if end_sec is None and audio_duration is not None:
            end_sec = audio_duration

        # Skip if we don't have an end time
        if end_sec is None:
            continue

        # Adjust to structure boundaries if manifest provided
        metadata = {
            "lrc_line": entry.get("line_number"),
            "original_timestamp": entry.get("timestamp_raw"),
        }

        if manifest and manifest.structure:
            adjusted_start, adjusted_end, adjust_meta = adjust_segment_to_structure(
                start_sec, end_sec, manifest.structure, structure_threshold
            )
            metadata.update(adjust_meta)
            segment_type = "combined"
        else:
            adjusted_start, adjusted_end = start_sec, end_sec
            segment_type = "lyric"

        # Apply maximum length constraint
        if max_segment_length and (adjusted_end - adjusted_start) > max_segment_length:
            # Split into sub-segments
            sub_segments = split_long_segment(
                adjusted_start, adjusted_end, max_segment_length, overlap_seconds
            )

            for j, (sub_start, sub_end) in enumerate(sub_segments):
                sub_metadata = metadata.copy()
                sub_metadata["sub_segment"] = f"{j+1}/{len(sub_segments)}"
                sub_metadata["split_reason"] = "max_length_exceeded"

                if len(sub_segments) > 1:
                    logger.info(
                        f"Split segment {i+1} into {len(sub_segments)} parts "
                        f"(original {adjusted_end - adjusted_start:.1f}s > {max_segment_length}s)"
                    )

                segment_specs.append(
                    AudioSegmentSpec(
                        start_seconds=sub_start,
                        end_seconds=sub_end,
                        text=entry["text"],
                        source_file=audio_file_path,
                        segment_type=segment_type,
                        metadata=sub_metadata,
                    )
                )
        else:
            # Single segment
            segment_specs.append(
                AudioSegmentSpec(
                    start_seconds=adjusted_start,
                    end_seconds=adjusted_end,
                    text=entry["text"],
                    source_file=audio_file_path,
                    segment_type=segment_type,
                    metadata=metadata,
                )
            )

    return segment_specs


def extract_audio_segment(
    audio_file_path: str,
    start_seconds: float,
    end_seconds: float,
    output_path: str,
    sample_rate: Optional[int] = None,
) -> bool:
    """Extract a segment of audio and save to file.

    Args:
        audio_file_path: Source audio file
        start_seconds: Start time in seconds
        end_seconds: End time in seconds
        output_path: Output file path
        sample_rate: Optional target sample rate (None = keep original)

    Returns:
        True if successful, False otherwise
    """
    try:
        # Load audio
        audio, sr = load_audio(audio_file_path, sr=sample_rate)

        # Calculate sample indices
        start_sample = int(start_seconds * sr)
        end_sample = int(end_seconds * sr)

        # Extract segment
        if audio.ndim == 1:
            segment = audio[start_sample:end_sample]
        else:
            segment = audio[start_sample:end_sample, :]

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Write segment
        sf.write(output_path, segment, sr, subtype="PCM_16")

        logger.info(
            f"Extracted segment {start_seconds:.3f}s - {end_seconds:.3f}s "
            f"({end_seconds - start_seconds:.3f}s) to {output_path}"
        )
        return True

    except Exception as e:
        logger.error(f"Failed to extract segment from {audio_file_path}: {e}")
        return False


def extract_all_segments(
    segment_specs: List[AudioSegmentSpec],
    output_dir: str,
    filename_prefix: str = "segment",
) -> List[str]:
    """Extract all audio segments defined by specs.

    Args:
        segment_specs: List of AudioSegmentSpec objects
        output_dir: Output directory for segments
        filename_prefix: Prefix for output filenames

    Returns:
        List of paths to successfully extracted segments
    """
    extracted_paths = []

    for i, spec in enumerate(segment_specs):
        output_filename = f"{filename_prefix}_{i+1:04d}.wav"
        output_path = os.path.join(output_dir, output_filename)

        success = extract_audio_segment(
            spec.source_file, spec.start_seconds, spec.end_seconds, output_path
        )

        if success:
            extracted_paths.append(output_path)

    logger.info(
        f"Extracted {len(extracted_paths)}/{len(segment_specs)} segments to {output_dir}"
    )

    return extracted_paths
