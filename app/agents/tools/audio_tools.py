import numpy as np
import sounddevice as sd
import librosa
import warnings
import logging
import assemblyai as aai
import os

from dotenv import load_dotenv
from scipy import signal
from scipy.io import wavfile
from scipy.fft import fft, ifft, fftfreq
from app.agents.enums.noise_type import NoiseType



load_dotenv()
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)


def generate_speech_like_noise(duration_seconds: float, sample_rate: int = 44100) -> bytes:
    """
    Generate more speech-like noise that AssemblyAI will be forced to transcribe.
    Creates formant-like patterns that sound vaguely speech-ish.
    """
    samples = int(duration_seconds * sample_rate)
    t = np.linspace(0, duration_seconds, samples)

    # Create speech-like formant structure
    fundamental = 120 + np.random.uniform(-20, 20)  # Vary fundamental frequency

    # Multiple formants at typical speech frequencies
    formant1 = 700 + np.random.uniform(-100, 100)  # First formant (vowels)
    formant2 = 1200 + np.random.uniform(-200, 200)  # Second formant (vowels)
    formant3 = 2400 + np.random.uniform(-300, 300)  # Third formant (consonants)

    # Generate speech-like signal with formants
    speech_like = (
            0.3 * np.sin(2 * np.pi * fundamental * t) +  # Fundamental
            0.25 * np.sin(2 * np.pi * fundamental * 2 * t) +  # First harmonic
            0.2 * np.sin(2 * np.pi * formant1 * t) +  # First formant
            0.15 * np.sin(2 * np.pi * formant2 * t) +  # Second formant
            0.1 * np.sin(2 * np.pi * formant3 * t) +  # Third formant
            0.05 * np.random.randn(samples)  # Noise component
    )

    # Add speech-like envelope (pauses and emphasis)
    envelope_freq = 0.5 + np.random.uniform(-0.2, 0.3)  # Speech cadence
    envelope = 0.5 + 0.5 * np.abs(np.sin(2 * np.pi * envelope_freq * t)) ** 0.7

    # Add random pauses like speech
    for i in range(int(duration_seconds * 2)):  # ~2 pauses per second
        pause_start = np.random.randint(0, samples - int(0.1 * sample_rate))
        pause_length = int(np.random.uniform(0.05, 0.15) * sample_rate)
        envelope[pause_start:pause_start + pause_length] *= 0.1

    speech_like *= envelope

    # Normalize and convert to int16
    speech_like = speech_like / np.max(np.abs(speech_like))
    speech_like = np.clip(speech_like, -1.0, 1.0)

    return (speech_like * 32767).astype(np.int16).tobytes()


def generate_speech_range_noise(duration_seconds: float, noise_type: NoiseType,
                                mix_level: float = 0.25, sample_rate: int = 44100,
                                freq_low: int = 300, freq_high: int = 3400) -> bytes:
    """Generate noise specifically targeting speech frequency ranges."""
    samples = int(duration_seconds * sample_rate)

    # Generate base noise
    if noise_type == NoiseType.WHITE:
        working_noise = np.random.normal(0, 1, samples)
    elif noise_type == NoiseType.BROWN:
        white_noise = np.random.normal(0, 1, samples)
        working_noise = np.cumsum(white_noise)
        working_noise = working_noise - np.mean(working_noise)
    elif noise_type == NoiseType.PINK:
        # Improved pink noise generation
        working_noise = np.random.randn(samples)
        # Apply 1/f filtering in frequency domain
        frequencies = fftfreq(samples, 1 / sample_rate)
        fft_noise = fft(working_noise)
        # Pink noise has 1/f power spectrum
        pink_filter = 1 / np.sqrt(np.abs(frequencies) + 1e-10)
        pink_filter[0] = 1  # DC component
        fft_noise *= pink_filter
        working_noise = np.real(ifft(fft_noise))

    # Apply bandpass filter to target speech frequencies
    nyquist = sample_rate / 2
    low = freq_low / nyquist
    high = freq_high / nyquist

    # Design Butterworth bandpass filter
    b, a = signal.butter(4, [low, high], btype='band')
    filtered_noise = signal.filtfilt(b, a, working_noise)

    # Normalize and apply mix level
    filtered_noise = filtered_noise / np.max(np.abs(filtered_noise))
    filtered_noise *= mix_level

    audio_data = (filtered_noise * 32767).astype(np.int16)
    return audio_data.tobytes()


def pitch_shift_audio(input_audio: bytes, cents: float = 50, sample_rate: int = 44100) -> bytes:
    """Apply pitch shifting to audio data."""
    # Convert bytes to float array
    audio_array = np.frombuffer(input_audio, dtype=np.int16).astype(np.float32) / 32767.0

    # Use librosa for high-quality pitch shifting
    shifted_audio = librosa.effects.pitch_shift(audio_array, sr=sample_rate, n_steps=cents / 100.0)

    # Convert back to int16
    shifted_audio = np.clip(shifted_audio, -1.0, 1.0)
    shifted_audio = (shifted_audio * 32767).astype(np.int16)

    return shifted_audio.tobytes()


def micro_stutter_audio(input_audio: bytes, stutter_probability: float = 0.1,
                        stutter_length_ms: int = 50, sample_rate: int = 44100) -> bytes:
    """Apply random micro-stutters to create phonetic ambiguity."""
    audio_array = np.frombuffer(input_audio, dtype=np.int16)
    output_audio = []

    # Process in 100ms windows
    window_samples = int(0.1 * sample_rate)
    stutter_samples = int(stutter_length_ms / 1000.0 * sample_rate)

    for i in range(0, len(audio_array), window_samples):
        window = audio_array[i:i + window_samples]
        output_audio.extend(window)

        # Random stutter
        if np.random.random() < stutter_probability and len(window) >= stutter_samples:
            # Repeat a small segment from the window
            start_idx = np.random.randint(0, len(window) - stutter_samples)
            stutter_segment = window[start_idx:start_idx + stutter_samples]
            output_audio.extend(stutter_segment)

    return np.array(output_audio, dtype=np.int16).tobytes()


def gate_audio(input_audio: bytes, gate_probability: float = 0.05,
               gate_length_ms: int = 20, sample_rate: int = 44100) -> bytes:
    """Apply random audio gates that cut off consonants/vowels unpredictably."""
    audio_array = np.frombuffer(input_audio, dtype=np.int16).copy()  # Make writable

    # Process in 50ms windows
    window_samples = int(0.05 * sample_rate)
    gate_samples = int(gate_length_ms / 1000.0 * sample_rate)

    for i in range(0, len(audio_array) - gate_samples, window_samples):
        if np.random.random() < gate_probability:
            audio_array[i:i + gate_samples] = 0

    return audio_array.tobytes()


def crush_audio(input_audio: bytes, intensity: float = 0.5) -> bytes:
    """Apply bit crushing with enhanced control for phonetic artifact creation."""
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
    """Apply a complete speech-to-text hallucination processing chain."""
    audio = input_audio

    # 1. Add speech-range noise
    noise = generate_speech_range_noise(
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
        audio = pitch_shift_audio(audio, cents, sample_rate)

    # 3. Add micro-stutters
    if hallucination_intensity > 0.4:
        audio = micro_stutter_audio(
            audio,
            stutter_probability=0.05 * hallucination_intensity,
            sample_rate=sample_rate
        )

    # 4. Apply random gates
    if hallucination_intensity > 0.2:
        audio = gate_audio(
            audio,
            gate_probability=0.03 * hallucination_intensity,
            sample_rate=sample_rate
        )

    # 5. Apply bit crushing
    if hallucination_intensity > 0.1:
        crush_intensity = 0.3 * hallucination_intensity
        audio = crush_audio(audio, crush_intensity)

    return audio


def save_wav(filename: str, audio_bytes: bytes, sample_rate: int = 44100):
    """Save audio bytes as a WAV file."""
    audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
    wavfile.write(filename, sample_rate, audio_array)


def conjurers_evp(evp_duration: int, working_path: str, file_name: str) -> str | None:
    """
    Generate Electronic Voice Phenomena (EVP) using systematic audio hallucination
    and force AssemblyAI to transcribe it with aggressive settings.
    """
    # Check for API key
    api_key = os.getenv("ASSEMBLYAI_API_KEY")
    if not api_key:
        logging.error("ASSEMBLYAI_API_KEY not found in environment variables!")
        return None

    aai.settings.api_key = api_key

    # Generate more speech-like base audio that AssemblyAI will want to transcribe
    logging.info(f"Generating {evp_duration}s of speech-like noise for EVP...")
    base_evp_noise = generate_speech_like_noise(evp_duration)

    # Apply hallucination processing
    logging.info("Applying systematic hallucination processing...")
    hallucinated_evp = apply_speech_hallucination_processing(
        base_evp_noise,
        hallucination_intensity=0.6  # Reduced intensity so it's still "speech-like"
    )

    # Save the audio file
    full_path = f"{working_path}/{file_name}"
    save_wav(full_path, hallucinated_evp)
    logging.info(f"Saved hallucinated EVP audio to {full_path}")

    # Configure AssemblyAI for maximum hallucination
    aai_config = aai.TranscriptionConfig(
        # Basic model settings
        speech_model=aai.SpeechModel.universal,  # Most general model

        # Disable smart features that might filter out our chaos
        filter_profanity=False,
        format_text=False,  # Don't clean up the output
        punctuate=False,  # Don't add smart punctuation

        # Language detection - let it guess wrong
        language_code=None,  # Let it auto-detect (might get it wrong)
        language_detection=True,

        # Advanced settings for more chaos
        disfluencies=True,  # Include "uh", "um", etc.

        speaker_labels=False,
        speakers_expected=None,  # Let it guess
        sentiment_analysis=False,  # Might hallucinate emotions
        entity_detection=True,  # Might hallucinate entities

    )

    logging.info("Starting aggressive AssemblyAI transcription...")
    transcriber = aai.Transcriber(config=aai_config)
    evp_transcription = transcriber.transcribe(full_path)

    if evp_transcription.status == 'error':
        logging.error(f"Transcription failed: {evp_transcription.error}")
        raise RuntimeError(f"Transcription failed: {evp_transcription.error}")

    if evp_transcription.status == 'completed':
        # Log detailed results
        logging.info(f"Transcription completed!")
        logging.info(f"Text: {evp_transcription.text}")
        logging.info(f"Confidence: {evp_transcription.confidence}")

        # Check for additional hallucinated content
        if hasattr(evp_transcription, 'sentiment_analysis_results') and evp_transcription.sentiment_analysis_results:
            logging.info(f"Sentiment: {evp_transcription.sentiment_analysis_results}")

        if hasattr(evp_transcription, 'entities') and evp_transcription.entities:
            logging.info(f"Detected entities: {[e.text for e in evp_transcription.entities]}")

        if hasattr(evp_transcription, 'summary') and evp_transcription.summary:
            logging.info(f"Summary: {evp_transcription.summary}")

        if hasattr(evp_transcription, 'utterances') and evp_transcription.utterances:
            logging.info(f"Utterances: {len(evp_transcription.utterances)} detected")
            for utterance in evp_transcription.utterances[:3]:  # First 3
                logging.info(f"  Utterance: '{utterance.text}' (confidence: {utterance.confidence})")

        # Return the main text, or fallback to utterances if main text is empty
        if evp_transcription.text and evp_transcription.text.strip():
            return evp_transcription.text
        elif hasattr(evp_transcription, 'utterances') and evp_transcription.utterances:
            # Combine utterances if main text is empty
            utterance_texts = [u.text for u in evp_transcription.utterances if u.text.strip()]
            if utterance_texts:
                combined_text = " ".join(utterance_texts)
                logging.info(f"Using combined utterances: {combined_text}")
                return combined_text

    logging.warning("No transcription text generated - AssemblyAI too conservative!")
    return None


def try_speech_like_generation(duration: int = 5) -> str:
    """Test the speech-like noise generation and play it back."""
    logging.info(f"Generating {duration}s of speech-like test audio...")

    # Generate speech-like noise
    speech_like = generate_speech_like_noise(duration)

    # Save for inspection
    save_wav("/tmp/speech_like_test.wav", speech_like)

    # Play it back
    audio_array = np.frombuffer(speech_like, dtype=np.int16)
    logging.info("Playing speech-like test audio...")
    sd.play(audio_array, samplerate=44100)
    sd.wait()

    return f"Generated and played {duration}s of speech-like noise. Saved to /tmp/speech_like_test.wav"


def main():
    """Main test function - no async needed!"""
    logging.info("Starting Black Agent EVP test with enhanced speech-like generation...")

    # Test 1: Generate and listen to speech-like noise
    test_result = try_speech_like_generation(3)
    logging.info(test_result)

    # Test 2: Full EVP pipeline
    evp_text = conjurers_evp(8, "/tmp", "black_agent_evp.wav")
    if evp_text:
        logging.info(f"SUCCESS! Generated EVP Text: '{evp_text}'")
        logging.info(f"Text length: {len(evp_text)} characters")
    else:
        logging.warning("Still no text generated - need to make audio even more speech-like!")


if __name__ == "__main__":
    main()