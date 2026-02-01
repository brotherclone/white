import io
import logging
import os
import random
import warnings
from fractions import Fraction
from typing import List, Optional, Union, Tuple

import numpy as np
import soundfile as sf
from dotenv import load_dotenv
from scipy import signal
from scipy.fft import fft, fftfreq, ifft
from scipy.signal import resample_poly

from app.structures.artifacts.audio_artifact_file import AudioChainArtifactFile
from app.structures.concepts.rainbow_table_color import (
    the_rainbow_table_colors,
)
from app.structures.enums.chain_artifact_file_type import ChainArtifactFileType
from app.structures.enums.noise_type import NoiseType

load_dotenv()
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


def _get_thread_artifact_base_path(thread_id: str | None = None) -> str:
    """Get base path for artifacts.

    NOTE: Returns ONLY the base path. The thread_id is added by
    the artifact's file_path property in base_artifact.py.
    """
    base = _get_agent_work_product_base_path()
    if not thread_id:
        logger.warning("No thread_id provided - using base directory only")
    return base


def _get_agent_work_product_base_path() -> str:
    """Return a safe base path for agent work products; fall back to cwd if env var missing."""
    bp = os.getenv("AGENT_WORK_PRODUCT_BASE_PATH")
    if not bp:
        logger.warning(
            "AGENT_WORK_PRODUCT_BASE_PATH not set; falling back to current working directory"
        )
        bp = os.getcwd()
    return bp


def load_audio(
    path_or_bytes: Union[str, bytes], sr: Optional[int] = None
) -> Tuple[np.ndarray, int]:
    """
    Load audio using soundfile. Accepts a file path or raw bytes and returns (audio, sample_rate).
    - audio: float32 numpy array in range [-1.0, 1.0]. Shape is (n_samples,) for mono or (n_samples, n_channels).
    - sr: if provided and differs from file sample rate, audio is resampled to this rate.

    Uses scipy.signal.resample_poly for high-quality resampling.
    """
    # open as file-like for bytes, or pass path directly
    src = (
        io.BytesIO(path_or_bytes)
        if isinstance(path_or_bytes, (bytes, bytearray))
        else path_or_bytes
    )

    data, file_sr = sf.read(src, dtype="float32", always_2d=False)

    # Ensure float32
    data = data.astype(np.float32)

    # If no resampling requested or already matches, return
    if sr is None or sr == file_sr:
        return data, file_sr

    # Compute rational approximation for resample_poly
    frac = Fraction(sr, file_sr).limit_denominator(1000)
    up, down = frac.numerator, frac.denominator

    # Resample mono or each channel separately for multi-channel
    if data.ndim == 1:
        resampled = resample_poly(data, up, down)
    else:
        # resample_poly operates on 1D arrays; process channels individually
        resampled_channels = [
            resample_poly(data[:, ch], up, down) for ch in range(data.shape[1])
        ]
        # stack back into shape (n_samples, n_channels)
        resampled = np.stack(resampled_channels, axis=1)

    # Cast and clip to safe range
    resampled = np.asarray(resampled, dtype=np.float32)
    if resampled.size:
        max_abs = np.max(np.abs(resampled))
        if max_abs > 1.0:
            resampled = resampled / max_abs

    return resampled, sr


def generate_speech_like_noise(
    duration_seconds: float, sample_rate: int = 44100
) -> bytes:
    samples = int(duration_seconds * sample_rate)
    t = np.linspace(0, duration_seconds, samples)
    fundamental = 120 + np.random.uniform(-20, 20)
    formant1 = 700 + np.random.uniform(-100, 100)
    formant2 = 1200 + np.random.uniform(-200, 200)
    formant3 = 2400 + np.random.uniform(-300, 300)
    speech_like = (
        0.3 * np.sin(2 * np.pi * fundamental * t)
        + 0.25 * np.sin(2 * np.pi * fundamental * 2 * t)
        + 0.2 * np.sin(2 * np.pi * formant1 * t)
        + 0.15 * np.sin(2 * np.pi * formant2 * t)
        + 0.1 * np.sin(2 * np.pi * formant3 * t)
        + 0.05 * np.random.randn(samples)
    )
    envelope_freq = 0.5 + np.random.uniform(-0.2, 0.3)
    envelope = 0.5 + 0.5 * np.abs(np.sin(2 * np.pi * envelope_freq * t)) ** 0.7
    for _ in range(int(duration_seconds * 2)):
        pause_start = np.random.randint(0, samples - int(0.1 * sample_rate))
        pause_length = int(np.random.uniform(0.05, 0.15) * sample_rate)
        envelope[pause_start : pause_start + pause_length] *= 0.1
    speech_like *= envelope
    speech_like = np.clip(speech_like / np.max(np.abs(speech_like)), -1.0, 1.0)
    return (speech_like * 32767).astype(np.int16).tobytes()


def generate_noise(
    duration_seconds: float,
    noise_type: NoiseType,
    mix_level: float = 0.25,
    sample_rate: int = 44100,
    freq_low: int = 300,
    freq_high: int = 3400,
) -> bytes:
    samples = int(duration_seconds * sample_rate)
    if noise_type == NoiseType.WHITE:
        working_noise = np.random.normal(0, 1, samples)
    elif noise_type == NoiseType.BROWN:
        working_noise = np.cumsum(np.random.normal(0, 1, samples))
        working_noise -= np.mean(working_noise)
    elif noise_type == NoiseType.PINK:
        working_noise = np.random.randn(samples)
        frequencies = fftfreq(samples, 1 / sample_rate)
        fft_noise = fft(working_noise)
        pink_filter = 1 / np.sqrt(np.abs(frequencies) + 1e-10)
        pink_filter[0] = 1
        fft_noise *= pink_filter
        working_noise = np.real(ifft(fft_noise))
    else:
        working_noise = np.zeros(samples)
    nyquist = sample_rate / 2
    low, high = freq_low / nyquist, freq_high / nyquist
    b, a = signal.butter(4, [low, high], btype="band")
    filtered_noise = signal.filtfilt(b, a, working_noise)
    filtered_noise = _safe_normalize_and_clip(filtered_noise)
    filtered_noise *= mix_level
    return (filtered_noise * 32767).astype(np.int16).tobytes()


def pitch_shift_audio_bytes(
    input_audio: bytes, cents: float = 50, sample_rate: int = 44100
) -> bytes:
    import librosa

    audio_array = (
        np.frombuffer(input_audio, dtype=np.int16).astype(np.float32) / 32767.0
    )
    shifted_audio = librosa.effects.pitch_shift(
        audio_array, sr=sample_rate, n_steps=cents / 100.0
    )
    shifted_audio = np.clip(shifted_audio, -1.0, 1.0)
    return (shifted_audio * 32767).astype(np.int16).tobytes()


def micro_stutter_audio_bytes(
    input_audio: bytes,
    stutter_probability: float = 0.1,
    stutter_length_ms: int = 50,
    sample_rate: int = 44100,
) -> bytes:
    audio_array = np.frombuffer(input_audio, dtype=np.int16)
    output_audio = []
    window_samples = int(0.1 * sample_rate)
    stutter_samples = int(stutter_length_ms / 1000.0 * sample_rate)
    for i in range(0, len(audio_array), window_samples):
        window = audio_array[i : i + window_samples]
        output_audio.extend(window)
        if np.random.random() < stutter_probability and len(window) >= stutter_samples:
            start_idx = np.random.randint(0, len(window) - stutter_samples)
            output_audio.extend(window[start_idx : start_idx + stutter_samples])
    return np.array(output_audio, dtype=np.int16).tobytes()


def gate_audio_bytes(
    wav_bytes: bytes,
    start_sec: float | None = None,
    end_sec: float | None = None,
    sample_rate: Optional[int] = None,
    gate_probability: Optional[float] = None,
    gate_length_ms: int = 100,
) -> bytes:
    """
    Zero-out audio between start_sec and end_sec, or perform probabilistic gating
    when gate_probability is provided. Accepts either WAV container bytes or raw
    PCM int16 bytes. Returns the same format as input.

    Parameters:
    - wav_bytes: input bytes (WAV file bytes or raw PCM int16)
    - start_sec, end_sec: explicit gate window (seconds)
    - sample_rate: required when input is raw PCM bytes (defaults to 44100)
    - gate_probability: if provided, gate is applied with this probability
    - gate_length_ms: gate window length in milliseconds when choosing random window
    """

    had_wav = True
    try:
        # Try to read as WAV file
        with io.BytesIO(wav_bytes) as src:
            data, file_sr = sf.read(src, dtype="float32", always_2d=False)
    except Exception:
        # Not a WAV container: treat as raw PCM int16
        had_wav = False
        file_sr = sample_rate or 44100
        pcm = np.frombuffer(wav_bytes, dtype=np.int16)
        # convert to float32 in -1.0..1.0
        data = pcm.astype(np.float32) / 32767.0

    # If gate_probability is provided, decide whether to gate and pick a random window
    if gate_probability is not None:
        if random.random() >= gate_probability:
            return wav_bytes  # no gating this time
        duration_seconds = data.shape[0] / file_sr
        gate_dur_sec = gate_length_ms / 1000.0
        max_start = max(0.0, duration_seconds - gate_dur_sec)
        start_sec = (
            start_sec if start_sec is not None else random.uniform(0.0, max_start)
        )
        end_sec = start_sec + gate_dur_sec if end_sec is None else end_sec

    # If an explicit window not provided, return original bytes
    if start_sec is None or end_sec is None:
        return wav_bytes
    start_idx = max(0, int(round(start_sec * file_sr)))
    end_idx = max(0, int(round(end_sec * file_sr)))
    end_idx = min(end_idx, data.shape[0])
    data = data.copy()
    # Zero the slice (handle mono or multi-channel)
    if data.ndim == 1:
        data[start_idx:end_idx] = 0.0
    else:
        data[start_idx:end_idx, :] = 0.0
    if had_wav:
        out_buf = io.BytesIO()
        sf.write(out_buf, data, file_sr, format="WAV")
        return out_buf.getvalue()
    else:
        int_data = np.clip(data, -1.0, 1.0)
        return (int_data * 32767).astype(np.int16).tobytes()


def bit_crush_audio_bytes(input_audio: bytes, intensity: float = 0.5) -> bytes:
    original_bits = 16
    target_bits = max(1, int(original_bits - intensity * (original_bits - 1)))
    audio_array = np.frombuffer(input_audio, dtype=np.int16)
    max_val = 2 ** (target_bits - 1) - 1
    min_val = -(2 ** (target_bits - 1))
    if max_val <= 0:
        return np.zeros_like(audio_array).tobytes()
    audio_crushed = np.round(audio_array.astype(np.float32) / 32767.0 * max_val)
    audio_crushed = np.clip(audio_crushed, min_val, max_val)
    return (audio_crushed / max_val * 32767.0).astype(np.int16).tobytes()


def apply_speech_hallucination_processing(
    input_audio: bytes, hallucination_intensity: float = 0.5, sample_rate: int = 44100
) -> bytes:
    audio = input_audio
    noise = generate_noise(
        duration_seconds=len(audio) / (2 * sample_rate),
        noise_type=NoiseType.PINK,
        mix_level=0.15 * hallucination_intensity,
    )
    audio_array = np.frombuffer(audio, dtype=np.int16).astype(np.float32)
    noise_array = np.frombuffer(noise, dtype=np.int16).astype(np.float32)
    min_len = min(len(audio_array), len(noise_array))
    mixed_audio = np.clip(
        audio_array[:min_len] + noise_array[:min_len], -32767, 32767
    ).astype(np.int16)
    audio = mixed_audio.tobytes()
    if hallucination_intensity > 0.3:
        cents = np.random.uniform(-100, 100) * hallucination_intensity
        audio = pitch_shift_audio_bytes(audio, cents, sample_rate)
    if hallucination_intensity > 0.4:
        audio = micro_stutter_audio_bytes(
            audio,
            stutter_probability=0.05 * hallucination_intensity,
            sample_rate=sample_rate,
        )
    if hallucination_intensity > 0.2:
        audio = gate_audio_bytes(
            audio,
            gate_probability=0.03 * hallucination_intensity,
            sample_rate=sample_rate,
        )
    if hallucination_intensity > 0.1:
        audio = bit_crush_audio_bytes(audio, 0.3 * hallucination_intensity)
    return audio


def find_wav_files(root_dir: str, prefix: str | None) -> List[str]:
    return [
        os.path.join(dirpath, file_name)
        for dirpath, _, filenames in os.walk(root_dir)
        for file_name in filenames
        if file_name.lower().endswith(".wav")
        and (prefix is None or file_name.startswith(prefix))
    ]


def find_wav_files_prioritized(
    directory: str,
    prefix: Optional[str] = None,
    priority_keywords: Optional[List[str]] = None,
) -> List[str]:
    if not directory:
        return []
    directory = str(directory)
    if not os.path.exists(directory):
        return []

    wav_files: List[str] = []
    for dirpath, _, filenames in os.walk(directory, followlinks=True):
        for fname in filenames:
            if fname.lower().endswith(".wav"):
                wav_files.append(os.path.join(dirpath, fname))

    # stable sort by basename (case-insensitive), then path for determinism
    wav_files.sort(key=lambda p: (os.path.basename(p).lower(), p))

    # normalize prefix (if it contains path parts, use basename)
    if prefix is not None:
        p_lower = os.path.basename(prefix).lower()

        def _matches_prefix(path: str) -> bool:
            b = os.path.basename(path).lower()
            name_no_ext = os.path.splitext(b)[0]
            return b.startswith(p_lower) or name_no_ext.startswith(p_lower)

        filtered = [p for p in wav_files if _matches_prefix(p)]
    else:
        filtered = wav_files

    if priority_keywords is None:
        priority_keywords = ["vocal", "vox"]
    pk = [k.lower() for k in priority_keywords if k]

    # preserve ordering: priority items first, then the rest
    priority = [
        p for p in filtered if any(k in os.path.basename(p).lower() for k in pk)
    ]
    non_priority = [p for p in filtered if p not in priority]

    return priority + non_priority


def extract_non_silent_segments(
    audio: np.ndarray, sr: int, min_duration: float, top_db: int = 30
) -> List[np.ndarray]:
    import librosa

    intervals = librosa.effects.split(audio, top_db=top_db)
    min_samples = int(min_duration * sr)
    return [audio[start:end] for start, end in intervals if end - start >= min_samples]


def select_random_segment_audio(
    root_dir: str, min_duration: float, num_segments: int, output_dir: str
) -> None:
    wav_files = find_wav_files(root_dir, None)
    random.shuffle(wav_files)
    per_file_segments = {}
    from app.util.audio_io import load_audio

    for wav_path in wav_files:
        try:
            audio, sr = load_audio(wav_path, sr=None)
        except Exception as e:
            logger.error(f"Failed to load `{wav_path}`: {e}")
            continue
        segments = extract_non_silent_segments(audio, sr, min_duration)
        if segments:
            random.shuffle(segments)
            per_file_segments[wav_path] = [(seg, sr) for seg in segments]
    if not per_file_segments:
        logger.info("No non-silent segments found in any file.")
        return
    os.makedirs(output_dir, exist_ok=True)
    written = 0
    file_list = list(per_file_segments.keys())
    while written < num_segments and any(per_file_segments.values()):
        for wav_path in file_list:
            if written >= num_segments:
                break
            seg_list = per_file_segments.get(wav_path)
            if not seg_list:
                continue
            seg, sr = seg_list.pop()
            seg_arr = np.asarray(seg, dtype=np.float32)
            max_abs = np.max(np.abs(seg_arr)) if seg_arr.size else 0.0
            if max_abs > 1.0:
                seg_arr = seg_arr / max_abs
            base = os.path.splitext(os.path.basename(wav_path))[0]
            out_path = os.path.join(output_dir, f"segment_{written + 1}_{base}.wav")
            try:
                sf.write(out_path, seg_arr, sr, subtype="PCM_16")
                logger.info(
                    f"Wrote `{out_path}` (sr={sr}, samples={len(seg_arr)}) from `{wav_path}`"
                )
                written += 1
            except Exception as e:
                logger.error(f"Failed to write `{out_path}`: {e}")
    logger.info(f"Extracted {written} segments to `{output_dir}`.")


def get_audio_segments_as_chain_artifacts(
    min_duration: float,
    num_segments: int,
    rainbow_color_mnemonic_character_value: str,
    thread_id: str | None = None,
) -> list:
    rainbow_color = the_rainbow_table_colors[rainbow_color_mnemonic_character_value]
    wav_files = find_wav_files_prioritized(
        os.getenv("MANIFEST_PATH"), rainbow_color.file_prefix
    )
    # Don't shuffle - keep vocal files prioritized at the front of the list
    logger.info(
        f"Found {len(wav_files)} total files for prefix '{rainbow_color.file_prefix}'"
    )
    vocal_files = [
        f
        for f in wav_files
        if "vocal" in os.path.basename(f).lower()
        or "vox" in os.path.basename(f).lower()
    ]
    logger.info(f"Found {len(vocal_files)} vocal files, will be processed first")

    all_segments = []
    from app.util.audio_io import load_audio

    for wav_path in wav_files:
        is_vocal = (
            "vocal" in os.path.basename(wav_path).lower()
            or "vox" in os.path.basename(wav_path).lower()
        )
        logger.info(
            f"Processing {'[VOCAL]' if is_vocal else '[INSTRUMENT]'} file: {os.path.basename(wav_path)}"
        )
        audio, sr = load_audio(wav_path, sr=None)
        segments = extract_non_silent_segments(audio, sr, min_duration)
        if not segments:
            continue
        random.shuffle(segments)
        all_segments.extend([(seg, sr, wav_path) for seg in segments])
        if len(all_segments) >= max(num_segments * 5, num_segments + 10):
            break
    if not all_segments:
        return []
    random.shuffle(all_segments)
    selected = all_segments[:num_segments]
    artifacts = []
    for idx, (seg, sr, wav_path) in enumerate(selected, start=1):
        artifact = AudioChainArtifactFile(
            thread_id=thread_id or "UNKNOWN_THREAD_ID",
            audio_bytes=seg.tobytes(),
            artifact_name=f"segment_{idx}",
            base_path=_get_thread_artifact_base_path(thread_id),
            rainbow_color_mnemonic_character_value=rainbow_color_mnemonic_character_value,
            chain_artifact_file_type=ChainArtifactFileType.AUDIO,
            artifact_path=wav_path,
            duration=len(seg) / sr,
            sample_rate=sr,
            channels=1,
            bit_depth=16,
        )
        dest_path = artifact.get_artifact_path()
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        logger.info(
            f"Writing audio segment to `{dest_path}` (sr={sr}, samples={len(seg)})"
        )
        seg_arr = np.asarray(seg, dtype=np.float32)
        max_abs = np.max(np.abs(seg_arr)) if seg_arr.size else 0.0
        if max_abs > 1.0:
            seg_arr = seg_arr / max_abs
        sf.write(dest_path, seg_arr, sr, subtype="PCM_16")
        artifacts.append(artifact)
    return artifacts


def create_random_audio_mosaic(
    root_dir: str,
    slice_duration_ms: int,
    target_length_sec: int | float,
    output_path: str,
) -> None:
    wav_files = find_wav_files(root_dir, None)
    random.shuffle(wav_files)
    segments = []
    total_samples = 0
    sample_rate = None
    slice_samples = None
    while total_samples < int(target_length_sec * 44100):
        if not wav_files:
            break
        wav_path = random.choice(wav_files)
        audio, sr = load_audio(wav_path, sr=None)
        if sample_rate is None:
            sample_rate = sr
            slice_samples = int(slice_duration_ms / 1000 * sample_rate)
        if len(audio) < slice_samples:
            continue
        start = random.randint(0, len(audio) - slice_samples)
        segments.append(audio[start : start + slice_samples])
        total_samples += slice_samples
    if segments:
        mosaic = np.concatenate(segments)[: int(target_length_sec * sample_rate)]
        sf.write(output_path, mosaic, sample_rate)
        print(f"Saved mosaic audio to {output_path}")
    else:
        print("No suitable segments found.")


def create_audio_mosaic_chain_artifact(
    segments: List[AudioChainArtifactFile],
    slice_duration_ms: int,
    target_length_sec: int | float,
    thread_id: str | None = None,
) -> AudioChainArtifactFile | None:
    if not segments:
        raise ValueError("`segments` must contain at least one AudioChainArtifactFile")
    wav_files = [
        seg.get_artifact_path()
        for seg in segments
        if os.path.exists(seg.get_artifact_path())
    ]
    if not wav_files:
        raise FileNotFoundError("No segment files found.")
    # Prefer the recorded sample rate on the artifact; if missing, use soundfile
    sample_rate = segments[0].sample_rate or sf.info(wav_files[0]).samplerate
    random_slice_duration_ms = random.randint(
        slice_duration_ms - 50, slice_duration_ms + 50
    )
    slice_samples = int(random_slice_duration_ms / 1000 * sample_rate)
    target_samples = int(target_length_sec * sample_rate)
    random.shuffle(wav_files)
    collected_segments = []
    total_samples = 0
    while total_samples < target_samples and wav_files:
        wav_path = random.choice(wav_files)
        audio, sr = load_audio(wav_path, sr=sample_rate)
        if len(audio) < slice_samples:
            continue
        start = random.randint(0, len(audio) - slice_samples)
        collected_segments.append(audio[start : start + slice_samples])
        total_samples += slice_samples
    if not collected_segments:
        logger.info("No suitable slices could be extracted to build a mosaic")
        return None
    mosaic = np.concatenate(collected_segments)[:target_samples]
    artifact = AudioChainArtifactFile(
        thread_id=thread_id or "UNKNOWN_THREAD_ID",
        audio_bytes=mosaic.tobytes(),
        base_path=_get_thread_artifact_base_path(thread_id),
        rainbow_color_mnemonic_character_value=getattr(
            segments[0], "rainbow_color_mnemonic_character_value", None
        ),
        chain_artifact_file_type=ChainArtifactFileType.AUDIO,
        artifact_name="mosiac",
        duration=len(mosaic) / sample_rate,
        sample_rate=sample_rate,
        channels=1,
        bit_depth=16,
    )
    out_dir = artifact.get_artifact_path(False)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, artifact.file_name)
    sf.write(out_path, mosaic, sample_rate, subtype="PCM_16")
    logger.info(
        f"Writing audio mosaic to `{out_path}` (sr={sample_rate}, samples={len(mosaic)})"
    )
    return artifact


def blend_with_noise(
    input_path: str, blend: float, blended_artifact: AudioChainArtifactFile
):
    from app.util.audio_io import load_audio

    audio, sr = load_audio(input_path, sr=None)
    duration_seconds = len(audio) / sr
    noise = (
        np.frombuffer(
            generate_speech_like_noise(duration_seconds, sr), dtype=np.int16
        ).astype(np.float32)
        / 32767.0
    )
    if len(noise) != len(audio):
        noise = (
            noise[: len(audio)]
            if len(noise) > len(audio)
            else np.pad(noise, (0, len(audio) - len(noise)), mode="constant")
        )
    blended_audio = np.clip((1 - blend) * audio + blend * noise, -1.0, 1.0)
    out_path = blended_artifact.get_artifact_path(True)
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    # write as 16-bit PCM for consistency with other writers
    sf.write(out_path, blended_audio, sr, subtype="PCM_16")


def create_blended_audio_chain_artifact(
    mosaic: AudioChainArtifactFile, blend: float, thread_id: str
) -> AudioChainArtifactFile:

    is_vocal = (
        "vocal" in mosaic.artifact_name.lower() or "vox" in mosaic.artifact_name.lower()
    )

    if is_vocal:
        blend = 0.0

    artifact = AudioChainArtifactFile(
        thread_id=thread_id or "UNKNOWN_THREAD_ID",
        base_path=_get_thread_artifact_base_path(thread_id),
        audio_bytes=mosaic.audio_bytes,
        rainbow_color_mnemonic_character_value=mosaic.rainbow_color_mnemonic_character_value,
        chain_artifact_file_type=ChainArtifactFileType.AUDIO,
        artifact_name="blended",
        duration=mosaic.duration,
        sample_rate=mosaic.sample_rate,
        channels=mosaic.channels,
    )

    blend_with_noise(mosaic.get_artifact_path(with_file_name=True), blend, artifact)
    logger.info(
        f"Writing audio mosaic to `{artifact.get_artifact_path(True)}` (sr={artifact.sample_rate})"
    )
    return artifact


def _safe_normalize_and_clip(x: np.ndarray) -> np.ndarray:
    """Replace NaNs/infs, then normalize and clip safely to [-1, 1]."""
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    if x.size == 0:
        return x
    max_abs = np.max(np.abs(x))
    if max_abs > 0.0:
        return np.clip(x / max_abs, -1.0, 1.0)
    return np.zeros_like(x)


if __name__ == "__main__":
    a = get_audio_segments_as_chain_artifacts(
        0.25, 3, the_rainbow_table_colors["Z"], "mock_101"
    )
    m = create_audio_mosaic_chain_artifact(a, 500, 5, thread_id="mock_101")
    b = create_blended_audio_chain_artifact(m, 0.5, thread_id="mock_101")
