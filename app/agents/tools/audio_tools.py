import numpy as np
import librosa
import warnings
import logging
import os
import random
import soundfile as sf

from typing import List
from dotenv import load_dotenv
from scipy import signal
from scipy.io import wavfile
from scipy.fft import fft, ifft, fftfreq

from app.agents.enums.chain_artifact_file_type import ChainArtifactFileType
from app.agents.enums.noise_type import NoiseType
from app.agents.models.audio_chain_artifact_file import AudioChainArtifactFile
from app.structures.concepts.rainbow_table_color import RainbowTableColor, the_rainbow_table_colors

load_dotenv()
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)


def generate_speech_like_noise(duration_seconds: float, sample_rate: int = 44100) -> bytes:
    """
    Generate noise that mimics the spectral and temporal characteristics of human speech.
    This includes formant structures, pauses, and a speech-like envelope.
    1. Formants: Simulate resonant frequencies typical in human speech.
    2. Envelope: Apply an amplitude envelope that mimics the rise and fall of speech.
    3. Pauses: Introduce random pauses to simulate natural speech patterns.
    :param duration_seconds:
    :param sample_rate:
    :return:
    """
    samples = int(duration_seconds * sample_rate)
    t = np.linspace(0, duration_seconds, samples)
    fundamental = 120 + np.random.uniform(-20, 20)
    formant1 = 700 + np.random.uniform(-100, 100)  # First formant (vowels)
    formant2 = 1200 + np.random.uniform(-200, 200)  # Second formant (vowels)
    formant3 = 2400 + np.random.uniform(-300, 300)  # Third formant (consonants)
    speech_like = (
            0.3 * np.sin(2 * np.pi * fundamental * t) +  # Fundamental
            0.25 * np.sin(2 * np.pi * fundamental * 2 * t) +  # First harmonic
            0.2 * np.sin(2 * np.pi * formant1 * t) +  # First formant
            0.15 * np.sin(2 * np.pi * formant2 * t) +  # Second formant
            0.1 * np.sin(2 * np.pi * formant3 * t) +  # Third formant
            0.05 * np.random.randn(samples)  # Noise component
    )
    envelope_freq = 0.5 + np.random.uniform(-0.2, 0.3)  # Speech cadence
    envelope = 0.5 + 0.5 * np.abs(np.sin(2 * np.pi * envelope_freq * t)) ** 0.7
    for i in range(int(duration_seconds * 2)):  # ~2 pauses per second
        pause_start = np.random.randint(0, samples - int(0.1 * sample_rate))
        pause_length = int(np.random.uniform(0.05, 0.15) * sample_rate)
        envelope[pause_start:pause_start + pause_length] *= 0.1
    speech_like *= envelope
    speech_like = speech_like / np.max(np.abs(speech_like))
    speech_like = np.clip(speech_like, -1.0, 1.0)
    return (speech_like * 32767).astype(np.int16).tobytes()


def generate_noise(duration_seconds: float, noise_type: NoiseType,
                   mix_level: float = 0.25, sample_rate: int = 44100,
                   freq_low: int = 300, freq_high: int = 3400) -> bytes:
    """
    Generate different types of noise (white, pink, brown) filtered to speech frequencies.
    :param duration_seconds:
    :param noise_type:
    :param mix_level:
    :param sample_rate:
    :param freq_low:
    :param freq_high:
    :return:
    """
    samples = int(duration_seconds * sample_rate)
    working_noise = np.zeros(samples)
    if noise_type == NoiseType.WHITE:
        working_noise = np.random.normal(0, 1, samples)
    elif noise_type == NoiseType.BROWN:
        white_noise = np.random.normal(0, 1, samples)
        working_noise = np.cumsum(white_noise)
        working_noise = working_noise - np.mean(working_noise)
    elif noise_type == NoiseType.PINK:
        working_noise = np.random.randn(samples)
        frequencies = fftfreq(samples, 1 / sample_rate)
        fft_noise = fft(working_noise)
        pink_filter = 1 / np.sqrt(np.abs(frequencies) + 1e-10)
        pink_filter[0] = 1
        fft_noise *= pink_filter
        working_noise = np.real(ifft(fft_noise))
    nyquist = sample_rate / 2
    low = freq_low / nyquist
    high = freq_high / nyquist
    b, a = signal.butter(4, [low, high], btype='band')
    filtered_noise = signal.filtfilt(b, a, working_noise)
    filtered_noise = filtered_noise / np.max(np.abs(filtered_noise))
    filtered_noise *= mix_level
    audio_data = (filtered_noise * 32767).astype(np.int16)
    return audio_data.tobytes()


def pitch_shift_audio_bytes(input_audio: bytes, cents: float = 50, sample_rate: int = 44100) -> bytes:
    """
    Apply micro pitch shifting to audio bytes.
    1. Convert bytes to numpy array.
    2. Use librosa to apply pitch shift.
    3. Convert back to bytes.
    4. Ensure no clipping occurs.
    5. Return processed audio bytes.
    :param input_audio:
    :param cents:
    :param sample_rate:
    :return:
    """
    audio_array = np.frombuffer(input_audio, dtype=np.int16).astype(np.float32) / 32767.0
    shifted_audio = librosa.effects.pitch_shift(audio_array, sr=sample_rate, n_steps=cents / 100.0)
    shifted_audio = np.clip(shifted_audio, -1.0, 1.0)
    shifted_audio = (shifted_audio * 32767).astype(np.int16)
    return shifted_audio.tobytes()


def micro_stutter_audio_bytes(input_audio: bytes, stutter_probability: float = 0.1,
                              stutter_length_ms: int = 50, sample_rate: int = 44100) -> bytes:
    """
    Introduce micro-stutters by repeating small segments of audio.
    :param input_audio:
    :param stutter_probability:
    :param stutter_length_ms:
    :param sample_rate:
    :return:
    """
    audio_array = np.frombuffer(input_audio, dtype=np.int16)
    output_audio = []
    window_samples = int(0.1 * sample_rate)
    stutter_samples = int(stutter_length_ms / 1000.0 * sample_rate)
    for i in range(0, len(audio_array), window_samples):
        window = audio_array[i:i + window_samples]
        output_audio.extend(window)
        if np.random.random() < stutter_probability and len(window) >= stutter_samples:
            start_idx = np.random.randint(0, len(window) - stutter_samples)
            stutter_segment = window[start_idx:start_idx + stutter_samples]
            output_audio.extend(stutter_segment)
    return np.array(output_audio, dtype=np.int16).tobytes()


def gate_audio_bytes(input_audio: bytes, gate_probability: float = 0.05,
                     gate_length_ms: int = 20, sample_rate: int = 44100) -> bytes:
    """
    Apply random gates (brief silences) to the audio.
    :param input_audio:
    :param gate_probability:
    :param gate_length_ms:
    :param sample_rate:
    :return:
    """
    audio_array = np.frombuffer(input_audio, dtype=np.int16).copy()
    window_samples = int(0.05 * sample_rate)
    gate_samples = int(gate_length_ms / 1000.0 * sample_rate)
    for i in range(0, len(audio_array) - gate_samples, window_samples):
        if np.random.random() < gate_probability:
            audio_array[i:i + gate_samples] = 0
    return audio_array.tobytes()


def bit_crush_audio_bytes(input_audio: bytes, intensity: float = 0.5) -> bytes:
    """
    Reduce the bit depth of the audio to create a bit-crushed effect.
    :param input_audio:
    :param intensity:
    :return:
    """
    original_bits = 16
    min_bits = 1
    target_bits = int(original_bits - intensity * (original_bits - min_bits))
    if target_bits < 1:
        target_bits = 1
    audio_array = np.frombuffer(input_audio, dtype=np.int16)
    max_val = 2 ** (target_bits - 1) - 1
    min_val = -2 ** (target_bits - 1)
    audio_crushed = np.round(audio_array / 32767 * max_val)
    audio_crushed = np.clip(audio_crushed, min_val, max_val)
    audio_crushed = (audio_crushed / max_val * 32767).astype(np.int16)
    return audio_crushed.tobytes()


def apply_speech_hallucination_processing(input_audio: bytes,
                                          hallucination_intensity: float = 0.5,
                                          sample_rate: int = 44100) -> bytes:
    """
    Apply a series of audio effects to simulate "hallucination" in speech-like audio.
    1. Add speech-range noise.
    2. Apply micro pitch shifting.
    3. Add micro-stutters.
    4. Apply random gates.
    5. Apply bit crushing.
    6. Return processed audio bytes.
    :param input_audio:
    :param hallucination_intensity:
    :param sample_rate:
    :return:
    """

    audio = input_audio
    # 1. Add speech-range noise
    noise = generate_noise(
        duration_seconds=len(audio) / (2 * sample_rate),  # 16-bit = 2 bytes per sample
        noise_type=NoiseType.PINK,
        mix_level=0.15 * hallucination_intensity
    )
    # Mix noise with original audio
    audio_array = np.frombuffer(audio, dtype=np.int16).astype(np.float32)
    noise_array = np.frombuffer(noise, dtype=np.int16).astype(np.float32)
    # Ensure same length
    min_len = min(len(audio_array), len(noise_array))
    mixed_audio = audio_array[:min_len] + noise_array[:min_len]
    mixed_audio = np.clip(mixed_audio, -32767, 32767).astype(np.int16)
    audio = mixed_audio.tobytes()
    # 2. Apply micro pitch shifting
    if hallucination_intensity > 0.3:
        cents = np.random.uniform(-100, 100) * hallucination_intensity
        audio = pitch_shift_audio_bytes(audio, cents, sample_rate)
    # 3. Add micro-stutters
    if hallucination_intensity > 0.4:
        audio = micro_stutter_audio_bytes(
            audio,
            stutter_probability=0.05 * hallucination_intensity,
            sample_rate=sample_rate
        )
    # 4. Apply random gates
    if hallucination_intensity > 0.2:
        audio = gate_audio_bytes(
            audio,
            gate_probability=0.03 * hallucination_intensity,
            sample_rate=sample_rate
        )
    # 5. Apply bit crushing
    if hallucination_intensity > 0.1:
        crush_intensity = 0.3 * hallucination_intensity
        audio = bit_crush_audio_bytes(audio, crush_intensity)
    return audio

def save_wav_from_bytes(filename: str, audio_bytes: bytes, sample_rate: int = 44100):
    """
    Save audio bytes as a WAV file.
    1. Convert bytes to numpy array.
    2. Use scipy.io.wavfile to write the WAV file.
    3. Ensure correct sample rate and format.
    4. Save to specified filename.
    :param filename:
    :param audio_bytes:
    :param sample_rate:
    :return:
    """
    audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
    wavfile.write(filename, sample_rate, audio_array)

def find_wav_files(root_dir: str, prefix: str | None)-> List[str]:
    """
    Recursively find all .wav files in the given directory.
    1. Walk through the directory tree.
    2. Check file extensions.
    3. Collect and return full paths.
    4. Return list of .wav file paths.
    :param prefix:
    :param root_dir:
    :return:
    """
    wav_files: List[str] = []
    for dirpath, _, filenames in os.walk(root_dir):
        for file_name in filenames:
            if file_name.lower().endswith('.wav') and (prefix is None or file_name.startswith(prefix)):
                wav_files.append(os.path.join(dirpath, file_name))
    return wav_files

def extract_non_silent_segments(audio:np.ndarray, sr:int, min_duration:float, top_db:int=30)->List[np.ndarray] :
    """
    Extract non-silent segments from audio using librosa.
    1. Use librosa.effects.split to find non-silent intervals.
    2. Filter segments by minimum duration.
    3. Return list of audio segments.
    :param audio:
    :param sr:
    :param min_duration:
    :param top_db:
    :return:
    """
    intervals = librosa.effects.split(audio, top_db=top_db)
    segments = []
    min_samples = int(min_duration * sr)
    for start, end in intervals:
        if end - start >= min_samples:
            segments.append(audio[start:end])
    return segments

def select_random_segment_audio(root_dir:str, min_duration:float, num_segments:int, output_dir:str) -> None:
    """
    Select random non-silent segments from .wav files in root_dir and save to output_dir.
    1. Find all .wav files.
    2. Shuffle the list for randomness.
    3. For each file, extract non-silent segments.
    4. Save segments until num_segments is reached.
    5. Ensure output directory exists.
    6. Return when done.
    :param root_dir:
    :param min_duration:
    :param num_segments:
    :param output_dir:
    :return:
    """
    wav_files = find_wav_files(root_dir, None)
    random.shuffle(wav_files)
    per_file_segments: dict[str, list[tuple[np.ndarray, int]]] = {}
    for wav_path in wav_files:
        try:
            audio, sr = librosa.load(wav_path, sr=None)
        except Exception as e:
            logging.error(f"Failed to load `{wav_path}`: {e}")
            continue
        segments = extract_non_silent_segments(audio, sr, min_duration)
        if not segments:
            continue
        random.shuffle(segments)
        per_file_segments[wav_path] = [(seg, sr) for seg in segments]
    if not per_file_segments:
        logging.info("No non-silent segments found in any file.")
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
                sf.write(out_path, seg_arr, sr, subtype='PCM_16')
                logging.info(f"Wrote `{out_path}` (sr={sr}, samples={len(seg_arr)}) from `{wav_path}`")
                written += 1
            except Exception as e:
                logging.error(f"Failed to write `{out_path}`: {e}")
                continue
    logging.info(f"Extracted {written} segments to `{output_dir}`.")

def get_audio_segments_as_chain_artifacts(min_duration: float,
                                          num_segments: int,
                                          rainbow_color: RainbowTableColor,
                                          thread_id: str = "t") -> list:
    wav_files = find_wav_files(os.getenv("MANIFEST_PATH"), rainbow_color.file_prefix)
    random.shuffle(wav_files)
    all_segments: list[tuple[np.ndarray, int, str]] = []
    for wav_path in wav_files:
        audio, sr = librosa.load(wav_path, sr=None)
        segments = extract_non_silent_segments(audio, sr, min_duration)
        if not segments:
            continue
        random.shuffle(segments)
        for seg in segments:
            all_segments.append((seg, sr, wav_path))
        if len(all_segments) >= max(num_segments * 5, num_segments + 10):
            break

    if not all_segments:
        return []

    random.shuffle(all_segments)
    selected = all_segments[:num_segments]

    artifacts: list[AudioChainArtifactFile] = []
    for idx, (seg, sr, wav_path) in enumerate(selected, start=1):
        file_name = f"{thread_id}_segment_{idx}.wav"
        artifact = AudioChainArtifactFile(
            base_path=os.getenv("AGENT_WORK_PRODUCT_BASE_PATH"),
            rainbow_color=rainbow_color,
            chain_artifact_file_type=ChainArtifactFileType.AUDIO,
            file_name=file_name,
            artifact_path=wav_path,
            duration=len(seg) / sr,
            sample_rate=sr,
            channels=1,
            bit_depth=16
        )
        dest_path = artifact.get_artifact_path()
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        logging.info(f"Writing audio segment to ` {dest_path} ` (sr={sr}, samples={len(seg)})")
        seg_arr = np.asarray(seg, dtype=np.float32)
        max_abs = np.max(np.abs(seg_arr)) if seg_arr.size else 0.0
        if max_abs > 1.0:
            seg_arr = seg_arr / max_abs
        sf.write(dest_path, seg_arr, sr, subtype='PCM_16')
        artifacts.append(artifact)

    return artifacts




def create_random_audio_mosaic(root_dir:str, slice_duration_ms:int, target_length_sec:int|float, output_path:str) -> None:
    """
    Create an audio mosaic by randomly slicing segments from .wav files in root_dir.
    1. Find all .wav files.
    2. Shuffle the list for randomness.
    3. Randomly select segments of slice_duration_ms.
    4. Concatenate segments until target_length_sec is reached.
    5. Save the mosaic to output_path.
    6. Ensure no clipping occurs.
    7. Return when done.
    :param root_dir:
    :param slice_duration_ms:
    :param target_length_sec:
    :param output_path:
    :return:
    """
    wav_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for file_name in filenames:
            if file_name.lower().endswith('.wav'):
                wav_files.append(os.path.join(dirpath, file_name))
    random.shuffle(wav_files)
    segments = []
    total_samples = 0
    sample_rate = None
    slice_samples = None
    while total_samples < int(target_length_sec * 44100):
        if not wav_files:
            break
        wav_path = random.choice(wav_files)
        audio, sr = librosa.load(wav_path, sr=None)
        if sample_rate is None:
            sample_rate = sr
            slice_samples = int(slice_duration_ms / 1000 * sample_rate)
        if len(audio) < slice_samples:
            continue
        start = random.randint(0, len(audio) - slice_samples)
        segment = audio[start:start + slice_samples]
        segments.append(segment)
        total_samples += slice_samples
    if segments:
        mosaic = np.concatenate(segments)[:int(target_length_sec * sample_rate)]
        sf.write(output_path, mosaic, sample_rate)
        print(f"Saved mosaic audio to {output_path}")
    else:
        print("No suitable segments found.")

def create_audio_mosaic_chain_artifact():
    pass

def blend_with_noise(input_path: str, blend: float, output_dir: str) -> str:
    """
    Blend the input audio with generated speech-like noise.
    1. Load the input audio file.
    2. Generate speech-like noise of the same duration.
    3. Blend the two audios based on the blend factor.
    4. Save the blended audio to output_dir.
    5. Ensure no clipping occurs.
    6. Return the path to the blended audio file.
    :param input_path:
    :param blend:
    :param output_dir:
    :return:
    """
    audio, sr = librosa.load(input_path, sr=None)
    duration_seconds = len(audio) / sr
    noise = np.frombuffer(generate_speech_like_noise(duration_seconds, sr), dtype=np.int16).astype(np.float32) / 32767.0
    if len(noise) > len(audio):
        noise = noise[:len(audio)]
    elif len(noise) < len(audio):
        noise = np.pad(noise, (0, len(audio) - len(noise)), mode='constant')
    blended = (1 - blend) * audio + blend * noise
    blended = np.clip(blended, -1.0, 1.0)
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, os.path.basename(input_path).replace('.wav', '_blended.wav'))
    sf.write(out_path, blended, sr)
    return out_path

def create_blended_audio_chain_artifact():
    pass

if __name__ == "__main__":
    # select_random_segment_audio(
    #     root_dir="/Volumes/LucidNonsense/White/staged_raw_material",
    #     min_duration=1.0,
    #     num_segments=5,
    #     output_dir="/Volumes/LucidNonsense/White/app/agents/work_products/black_work_products/audio_segments"
    # )
    # create_random_audio_mosaic(
    #     root_dir='/Volumes/LucidNonsense/White/app/agents/work_products/black_work_products/audio_segments',
    #     slice_duration_ms=50,
    #     target_length_sec=10,
    #     output_path='/Volumes/LucidNonsense/White/app/agents/work_products/black_work_products/audio_mosaics/mosaic.wav'
    # )
    # blended_path = blend_with_noise(
    #     input_path='/Volumes/LucidNonsense/White/app/agents/work_products/black_work_products/audio_mosaics/mosaic.wav',
    #     blend=0.3,
    #     output_dir='/Volumes/LucidNonsense/White/app/agents/work_products/black_work_products/blended_audios'
    # )
    a = get_audio_segments_as_chain_artifacts(4.0, 3,the_rainbow_table_colors['Z'], '123443')
    print(a)