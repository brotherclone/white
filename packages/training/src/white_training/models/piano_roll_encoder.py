"""
Piano roll CNN encoder for MIDI embedding extraction.

Converts raw MIDI binary (from training parquet) into a piano roll matrix
[128 pitch x T time steps], then encodes via CNN to a 512-dim embedding.

Used as the MIDI modality encoder in the multimodal fusion model (Phase 3).
"""

import io
import logging
from typing import Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("torch not available — PianoRollEncoder CNN disabled")

try:
    import mido

    MIDO_AVAILABLE = True
except ImportError:
    MIDO_AVAILABLE = False
    logger.warning("mido not available — MIDI piano roll conversion disabled")


# --- Piano Roll Conversion (numpy + mido only, no torch) ---


def midi_bytes_to_piano_roll(
    midi_bytes: bytes,
    time_steps: int = 256,
    time_resolution_ms: float = 100.0,
    velocity_normalize: bool = True,
) -> np.ndarray:
    """Convert raw MIDI file bytes to a piano roll matrix.

    Args:
        midi_bytes: Raw .mid file content as bytes.
        time_steps: Number of time columns in output matrix.
        time_resolution_ms: Milliseconds per time step.
        velocity_normalize: If True, normalize velocity to [0, 1].
            If False, use binary (note on = 1.0, off = 0.0).

    Returns:
        Piano roll matrix of shape [128, time_steps], dtype float32.
        Pitch axis is full MIDI range (0-127).
        Values are velocity (0-1) or binary presence.
    """
    if not MIDO_AVAILABLE:
        raise RuntimeError("mido is required for MIDI piano roll conversion")

    mid = mido.MidiFile(file=io.BytesIO(midi_bytes))

    # Collect all note events with absolute time in seconds
    note_events = []

    for track in mid.tracks:
        abs_time_ticks = 0
        tempo = 500000  # default 120 BPM

        for msg in track:
            abs_time_ticks += msg.time

            if msg.type == "set_tempo":
                tempo = msg.tempo

            if msg.type == "note_on" and msg.velocity > 0:
                time_sec = mido.tick2second(abs_time_ticks, mid.ticks_per_beat, tempo)
                note_events.append(("on", msg.note, msg.velocity, time_sec))
            elif msg.type == "note_off" or (
                msg.type == "note_on" and msg.velocity == 0
            ):
                time_sec = mido.tick2second(abs_time_ticks, mid.ticks_per_beat, tempo)
                note_events.append(("off", msg.note, 0, time_sec))

    # Build piano roll
    piano_roll = np.zeros((128, time_steps), dtype=np.float32)
    max_time_sec = time_steps * time_resolution_ms / 1000.0

    # Track active notes: pitch -> (start_step, velocity)
    active_notes = {}

    # Sort events by time for correct processing
    note_events.sort(key=lambda e: e[3])

    for event_type, pitch, velocity, time_sec in note_events:
        if time_sec >= max_time_sec:
            break

        step = int(time_sec / (time_resolution_ms / 1000.0))
        step = min(step, time_steps - 1)

        if event_type == "on":
            active_notes[pitch] = (step, velocity)
        elif event_type == "off" and pitch in active_notes:
            start_step, vel = active_notes.pop(pitch)
            end_step = min(step, time_steps - 1)

            if velocity_normalize:
                value = vel / 127.0
            else:
                value = 1.0

            piano_roll[pitch, start_step : end_step + 1] = value

    # Close any still-active notes (sustain to end of roll)
    for pitch, (start_step, vel) in active_notes.items():
        if velocity_normalize:
            value = vel / 127.0
        else:
            value = 1.0
        piano_roll[pitch, start_step:] = value

    return piano_roll


# --- CNN Encoder (requires torch) ---

if TORCH_AVAILABLE:

    class PianoRollEncoder(nn.Module):
        """Lightweight CNN that encodes a piano roll matrix into a fixed-dim embedding.

        Input:  [batch, 1, 128, time_steps]  (like a grayscale image)
        Output: [batch, output_dim]
        """

        def __init__(self, output_dim: int = 512, time_steps: int = 256):
            super().__init__()
            self.output_dim = output_dim
            self.time_steps = time_steps

            self.conv = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(2),  # [batch, 32, 64, T/2]
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2),  # [batch, 64, 32, T/4]
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((4, 4)),  # [batch, 128, 4, 4]
            )
            self.fc = nn.Sequential(
                nn.Linear(128 * 4 * 4, output_dim),
                nn.ReLU(),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Encode piano roll to embedding.

            Args:
                x: Piano roll tensor [batch, 1, 128, time_steps]

            Returns:
                Embedding tensor [batch, output_dim]
            """
            features = self.conv(x)
            features = features.view(features.size(0), -1)
            return self.fc(features)

    def batch_midi_to_piano_rolls(
        midi_bytes_list: list[bytes],
        time_steps: int = 256,
        time_resolution_ms: float = 100.0,
        velocity_normalize: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert a batch of MIDI bytes to piano roll tensors.

        Args:
            midi_bytes_list: List of raw MIDI file bytes.
            time_steps: Time steps per piano roll.
            time_resolution_ms: Milliseconds per step.
            velocity_normalize: Normalize velocity to [0, 1].

        Returns:
            Tuple of:
                - piano_rolls: [batch, 1, 128, time_steps] float32 tensor
                - valid_mask: [batch] boolean tensor (True if conversion succeeded)
        """
        rolls = []
        valid = []

        for midi_bytes in midi_bytes_list:
            if midi_bytes is None or len(midi_bytes) == 0:
                rolls.append(np.zeros((128, time_steps), dtype=np.float32))
                valid.append(False)
                continue

            try:
                roll = midi_bytes_to_piano_roll(
                    midi_bytes,
                    time_steps=time_steps,
                    time_resolution_ms=time_resolution_ms,
                    velocity_normalize=velocity_normalize,
                )
                rolls.append(roll)
                valid.append(True)
            except Exception as e:
                logger.warning(f"Failed to convert MIDI to piano roll: {e}")
                rolls.append(np.zeros((128, time_steps), dtype=np.float32))
                valid.append(False)

        stacked = np.stack(rolls)[:, np.newaxis, :, :]
        return torch.from_numpy(stacked), torch.tensor(valid, dtype=torch.bool)

else:

    class PianoRollEncoder:  # type: ignore[no-redef]
        """Stub — torch is not installed in this environment."""

        def __init__(self, *args, **kwargs):
            raise RuntimeError("torch is required for PianoRollEncoder")

    def batch_midi_to_piano_rolls(*args, **kwargs):  # type: ignore[no-redef]
        raise RuntimeError("torch is required for batch_midi_to_piano_rolls")


if __name__ == "__main__":
    print("Testing PianoRollEncoder...\n")

    # Test 1: Piano roll conversion (mido only — works locally)
    if MIDO_AVAILABLE:
        print("=== MIDI bytes to piano roll ===")
        mid = mido.MidiFile()
        track = mido.MidiTrack()
        mid.tracks.append(track)
        track.append(mido.Message("note_on", note=60, velocity=100, time=0))
        track.append(mido.Message("note_off", note=60, velocity=0, time=480))
        track.append(mido.Message("note_on", note=64, velocity=80, time=0))
        track.append(mido.Message("note_off", note=64, velocity=0, time=480))
        track.append(mido.MetaMessage("end_of_track", time=0))

        buf = io.BytesIO()
        mid.save(file=buf)
        midi_bytes = buf.getvalue()

        roll = midi_bytes_to_piano_roll(midi_bytes, time_steps=256)
        print(f"Piano roll shape: {roll.shape}")
        print(f"Non-zero pixels:  {np.count_nonzero(roll)}")
        print(f"Active pitches:   {np.where(roll.any(axis=1))[0].tolist()}")
        assert roll.shape == (128, 256)
        assert roll[60].max() > 0, "Note C4 (60) should be active"
        assert roll[64].max() > 0, "Note E4 (64) should be active"

        # Velocity check
        expected_vel = 100 / 127.0
        assert (
            abs(roll[60].max() - expected_vel) < 0.01
        ), f"C4 velocity should be ~{expected_vel:.3f}, got {roll[60].max():.3f}"

        # Binary mode
        roll_bin = midi_bytes_to_piano_roll(midi_bytes, velocity_normalize=False)
        assert roll_bin[60].max() == 1.0, "Binary mode should be 1.0"
        print("PASS")
    else:
        print("SKIP — mido not installed")

    # Test 2: CNN + batch (torch required)
    if TORCH_AVAILABLE:
        print("\n=== CNN forward pass ===")
        encoder = PianoRollEncoder(output_dim=512, time_steps=256)
        dummy = torch.randn(4, 1, 128, 256)
        out = encoder(dummy)
        print(f"Input:  {dummy.shape}")
        print(f"Output: {out.shape}")
        assert out.shape == (4, 512)

        total_params = sum(p.numel() for p in encoder.parameters())
        print(f"Params: {total_params:,}")
        print("PASS")

        if MIDO_AVAILABLE:
            print("\n=== Batch conversion ===")
            rolls, mask = batch_midi_to_piano_rolls(
                [midi_bytes, midi_bytes, None, midi_bytes]
            )
            print(f"Batch shape: {rolls.shape}")
            print(f"Valid mask:   {mask.tolist()}")
            assert rolls.shape == (4, 1, 128, 256)
            assert mask.tolist() == [True, True, False, True]
            print("PASS")

            print("\n=== End-to-end: MIDI bytes -> embedding ===")
            emb = encoder(rolls)
            print(f"Embedding: {emb.shape}")
            assert emb.shape == (4, 512)
            print("PASS")
    else:
        print("\nSKIP — torch not installed (CNN tests run on Modal)")

    print("\nDone!")
