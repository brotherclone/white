import os
import numpy as np

def has_significant_audio(audio_chunk, threshold_db=-20):
    if len(audio_chunk) == 0:
        return False

    samples = audio_chunk.get_array_of_samples()
    if len(samples) == 0:
        return False

    rms = compute_rms(samples)

    # Convert RMS to decibels (with reference to maximum possible amplitude)
    max_possible_amplitude = 32767  # For 16-bit audio
    db_level = 20 * np.log10(rms / max_possible_amplitude) if rms > 0 else -float('inf')

    # Check if the dB level exceeds the threshold
    return db_level > threshold_db

def compute_rms(samples):
    samples = np.asarray(samples)
    samples = samples[np.isfinite(samples)]
    if samples.size == 0:
        print("Warning: All audio samples are NaN or inf.")
        return 0.0
    mean_square = np.mean(samples ** 2)
    if not np.isfinite(mean_square) or mean_square <= 0:
        print("Warning: mean_square is not positive or is NaN after filtering.")
        return 0.0
    return np.sqrt(mean_square)

def get_microseconds_per_beat(bpm):
    if bpm <= 0:
        raise ValueError("BPM must be a positive number.")
    return 60000000 / bpm  # Convert BPM to microseconds per beat

def audio_to_byes(file_name, audio_dir)-> bytes | None:
    audio_path = os.path.join(audio_dir, file_name)
    try:
        with open(audio_path, "rb") as f:
           return f.read()
    except Exception as e:
        print (f"✗ Failed to read audio file '{audio_path}': {e}")
        return None

def split_midi_file_by_segment(tempo:int, midi_file_path: str, segment_duration: float)-> str:
    return ""

def midi_to_bytes(midi_file_path) -> bytes | None:
    pass