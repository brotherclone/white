"""MIDI file segmentation utilities for timestamp-based pipeline.

This module provides functionality to segment MIDI files based on timestamps,
aligning them with audio segments for training data preparation.
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional

logger = logging.getLogger(__name__)

try:
    import mido
    from mido import MidiFile, MidiTrack, MetaMessage

    MIDO_AVAILABLE = True
except ImportError:
    MIDO_AVAILABLE = False
    logger.warning(
        "mido library not available. MIDI segmentation features will be disabled. "
        "Install with: pip install mido"
    )


def segment_midi_file(
    midi_file_path: str,
    start_seconds: float,
    end_seconds: float,
    output_path: str,
    preserve_tempo: bool = True,
) -> bool:
    """Segment a MIDI file to a specific time range.

    The MIDI events are time-shifted so the segment starts at time 0:00,
    while preserving relative timing between events.

    Args:
        midi_file_path: Path to source MIDI file
        start_seconds: Start time in seconds
        end_seconds: End time in seconds
        output_path: Output file path
        preserve_tempo: Whether to preserve tempo meta messages (default True)

    Returns:
        True if successful, False otherwise
    """
    if not MIDO_AVAILABLE:
        logger.error("mido library not available for MIDI segmentation")
        return False

    try:
        # Load MIDI file
        midi = MidiFile(midi_file_path)

        # Create new MIDI file for segment
        segment_midi = MidiFile(type=midi.type, ticks_per_beat=midi.ticks_per_beat)

        # Convert seconds to ticks
        tempo = 500000  # Default tempo (120 BPM)
        ticks_per_beat = midi.ticks_per_beat

        # Extract tempo from first track if available
        for track in midi.tracks:
            for msg in track:
                if msg.type == "set_tempo":
                    tempo = msg.tempo
                    break
            if tempo != 500000:
                break

        # Calculate ticks per second
        ticks_per_second = ticks_per_beat * 1000000 / tempo

        start_tick = int(start_seconds * ticks_per_second)
        end_tick = int(end_seconds * ticks_per_second)

        # Process each track
        for track in midi.tracks:
            new_track = MidiTrack()
            absolute_tick = 0
            delta_tick_accumulator = 0

            # Track whether we've added the track name
            track_named = False

            for msg in track:
                absolute_tick += msg.time
                delta_tick_accumulator += msg.time

                # Check if message is within segment range
                if start_tick <= absolute_tick <= end_tick:
                    # Adjust time relative to segment start
                    adjusted_tick = absolute_tick - start_tick

                    # For the first message in range, use the delta from segment start
                    if len(new_track) == 0:
                        delta_time = adjusted_tick
                    else:
                        delta_time = msg.time

                    # Copy message with adjusted timing
                    if msg.is_meta:
                        # Preserve important meta messages
                        if (
                            msg.type in ["track_name", "instrument_name"]
                            and not track_named
                        ):
                            new_track.append(msg.copy(time=delta_time))
                            track_named = True
                        elif (
                            msg.type == "set_tempo"
                            and preserve_tempo
                            and len(new_track) == 0
                        ):
                            new_track.append(msg.copy(time=delta_time))
                        elif msg.type == "time_signature" and len(new_track) == 0:
                            new_track.append(msg.copy(time=delta_time))
                        elif msg.type == "key_signature" and len(new_track) == 0:
                            new_track.append(msg.copy(time=delta_time))
                        elif msg.type == "end_of_track":
                            # Don't add yet, will add at the end
                            pass
                    else:
                        # Regular MIDI message (note_on, note_off, control_change, etc.)
                        new_track.append(msg.copy(time=delta_time))

            # Add end_of_track message
            if len(new_track) > 0:
                new_track.append(MetaMessage("end_of_track", time=0))
                segment_midi.tracks.append(new_track)

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Save segmented MIDI
        segment_midi.save(output_path)

        logger.info(
            f"Segmented MIDI {start_seconds:.3f}s - {end_seconds:.3f}s "
            f"to {output_path}"
        )
        return True

    except Exception as e:
        logger.error(f"Failed to segment MIDI file {midi_file_path}: {e}")
        return False


def find_matching_midi_files(audio_file_path: str, manifest_tracks: List) -> List[str]:
    """Find MIDI files that correspond to an audio file based on manifest.

    Args:
        audio_file_path: Path to audio file
        manifest_tracks: List of ManifestTrack objects from manifest

    Returns:
        List of MIDI file paths that correspond to the audio file
    """
    audio_filename = os.path.basename(audio_file_path)
    audio_dir = os.path.dirname(audio_file_path)

    midi_files = []

    for track in manifest_tracks:
        # Check if this track's audio file matches
        if hasattr(track, "audio_file") and track.audio_file == audio_filename:
            # Check for midi_file or midi_group_file
            if hasattr(track, "midi_file") and track.midi_file:
                midi_path = os.path.join(audio_dir, track.midi_file)
                if os.path.exists(midi_path):
                    midi_files.append(midi_path)

            if hasattr(track, "midi_group_file") and track.midi_group_file:
                midi_path = os.path.join(audio_dir, track.midi_group_file)
                if os.path.exists(midi_path) and midi_path not in midi_files:
                    midi_files.append(midi_path)

    return midi_files


def segment_midi_with_audio(
    audio_file_path: str,
    start_seconds: float,
    end_seconds: float,
    output_dir: str,
    manifest_tracks: Optional[List] = None,
    output_prefix: str = "segment",
) -> List[str]:
    """Segment all MIDI files associated with an audio file.

    This is a convenience function that finds matching MIDI files and segments
    them to match the audio segment timing.

    Args:
        audio_file_path: Path to audio file
        start_seconds: Segment start time
        end_seconds: Segment end time
        output_dir: Output directory for MIDI segments
        manifest_tracks: Optional list of ManifestTrack objects
        output_prefix: Prefix for output filenames

    Returns:
        List of paths to segmented MIDI files
    """
    if not MIDO_AVAILABLE:
        logger.warning("MIDI segmentation skipped - mido library not available")
        return []

    # Find associated MIDI files
    if manifest_tracks:
        midi_files = find_matching_midi_files(audio_file_path, manifest_tracks)
    else:
        # Fallback: look for MIDI files with similar names
        audio_base = os.path.splitext(os.path.basename(audio_file_path))[0]
        audio_dir = os.path.dirname(audio_file_path)
        midi_files = [
            os.path.join(audio_dir, f)
            for f in os.listdir(audio_dir)
            if f.endswith(".mid") and audio_base in f
        ]

    if not midi_files:
        logger.debug(f"No MIDI files found for audio file {audio_file_path}")
        return []

    # Segment each MIDI file
    segmented_paths = []
    for i, midi_path in enumerate(midi_files):
        midi_basename = os.path.splitext(os.path.basename(midi_path))[0]
        output_filename = f"{output_prefix}_{midi_basename}.mid"
        output_path = os.path.join(output_dir, output_filename)

        success = segment_midi_file(midi_path, start_seconds, end_seconds, output_path)

        if success:
            segmented_paths.append(output_path)

    return segmented_paths


def extract_midi_note_density(midi_file_path: str) -> Optional[float]:
    """Calculate note density (notes per second) for a MIDI file.

    This can be useful for filtering segments or analyzing training data.

    Args:
        midi_file_path: Path to MIDI file

    Returns:
        Notes per second, or None if calculation fails
    """
    if not MIDO_AVAILABLE:
        return None

    try:
        midi = MidiFile(midi_file_path)
        total_notes = 0
        total_time = 0.0

        for track in midi.tracks:
            for msg in track:
                if msg.type == "note_on" and msg.velocity > 0:
                    total_notes += 1
                total_time += mido.tick2second(
                    msg.time, midi.ticks_per_beat, 500000  # default tempo
                )

        if total_time > 0:
            return total_notes / total_time

        return 0.0

    except Exception as e:
        logger.error(f"Failed to calculate note density for {midi_file_path}: {e}")
        return None
