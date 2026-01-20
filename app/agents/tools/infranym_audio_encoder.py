"""
Infranym Audio Encoder
======================

Three-layer audible steganography for musical composition.
Designed for The Earthly Frames - White Album project.

Creates alien transmissions / vocal textures with hidden puzzle content
that can be layered into actual musical tracks.

Layer 1 (Surface): Primary message - clear, intentional
Layer 2 (Reverse): Reversed overlay - textural, ambiguous
Layer 3 (Submerged): Frequency-isolated - subliminal, discoverable

All layers are AUDIBLE with careful listening, not digital headers.
"""

import time as teatime
import logging
import pyttsx3
import numpy as np
import json

from pydub import AudioSegment
from pydub.effects import low_pass_filter, high_pass_filter
from pathlib import Path
from typing import List, Dict, Any

from app.structures.enums.infranym_voice_profile import InfranymVoiceProfile
from app.structures.artifacts.infranym_voice_layer import InfranymVoiceLayer
from app.structures.artifacts.infranym_voice_composition import InfranymVoiceComposition

logger = logging.getLogger(__name__)


class InfranymAudioEncoder:
    """
    Encode three-layer audio puzzles suitable for musical composition.

    Philosophy:
    - Every layer must sound intentional, not accidental
    - Puzzle content rewards close listening but doesn't break immersion
    - Audio quality suitable for professional music production
    - Designed to mix well with live instruments
    """

    def __init__(self, sample_rate: int = 44100, output_dir: str = "infranym_output"):
        self.sample_rate = sample_rate
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.tts = None
        self.available_voices = []
        self._reinit_tts()
        self._audio_cache: Dict[str, AudioSegment] = {}

    def _reinit_tts(self):
        """(Re)initialize TTS engine - useful for macOS stability"""
        try:
            if self.tts is not None:
                try:
                    self.tts.stop()
                except EnvironmentError as e:
                    logger.warning("TTS engine stop error: %s", e)
                try:
                    del self.tts
                except ReferenceError as re:
                    logger.warning("TTS engine reference error: %s", re)
                    pass
        except EnvironmentError as ee:
            logger.warning("TTS engine init error: %s", ee)
            pass
        teatime.sleep(0.1)
        self.tts = pyttsx3.init()
        self.available_voices = self.tts.getProperty("voices")

        if not self.available_voices:
            raise RuntimeError(
                "No TTS voices available. On macOS, make sure System Voices are installed. "
                "System Preferences > Accessibility > Spoken Content > System voice"
            )

    def list_available_voices(self) -> List[Dict[str, str]]:
        """Get all available system TTS voices"""
        return [
            {
                "id": voice.id,
                "name": voice.name,
                "languages": voice.languages,
                "gender": voice.gender,
            }
            for voice in self.available_voices
        ]

    def generate_speech(
        self,
        text: str,
        rate: int = 150,
        voice_index: int = 0,
        pitch: float = 1.0,
        retry: bool = True,
    ) -> AudioSegment:
        """
        Generate speech audio with specified parameters.

        Args:
            text: Text to synthesize
            rate: Speech rate (50-300, default 150)
            voice_index: Index of system voice to use
            pitch: Pitch shift multiplier (0.5 = octave down, 2.0 = octave up)
            retry: If True, retry once with TTS reinitialization on failure

        Returns:
            AudioSegment with generated speech
        """
        # Create a cache key
        cache_key = f"{text}_{rate}_{voice_index}_{pitch}"
        if cache_key in self._audio_cache:
            logger.info("Using cached audio")
            return self._audio_cache[cache_key]
        logger.info("Reinitializing TTS engine...")
        self._reinit_tts()
        teatime.sleep(0.1)
        # Generate to a temp file (try multiple extensions for macOS compatibility)
        temp_base = self.output_dir / f"temp_{hash(cache_key)}"
        temp_wav = Path(f"{temp_base}.wav")
        temp_aiff = Path(f"{temp_base}.aiff")
        logger.info("Generating: '%s...'", text[:50])
        try:
            self.tts.setProperty("rate", rate)
            if voice_index < len(self.available_voices):
                self.tts.setProperty("voice", self.available_voices[voice_index].id)
            self.tts.save_to_file(text, str(temp_wav))
            self.tts.runAndWait()
            teatime.sleep(0.2)
            logger.info("Checking for temp file: %s", temp_wav.name)
            audio = None
            # First, check if the file was actually created
            if not temp_wav.exists():
                logger.error("Temp file not created: %s", temp_wav)
                raise RuntimeError(
                    f"TTS failed to create audio file for: {text[:50]}..."
                )
            file_size = temp_wav.stat().st_size
            logger.info("File size: %d bytes", file_size)
            if file_size == 0:
                logger.error("TTS created empty file: %s", temp_wav)
                raise RuntimeError(f"TTS created empty file for: {text[:50]}...")
            try:
                audio = AudioSegment.from_wav(str(temp_wav))
            except Exception as wav_error:
                try:
                    audio = AudioSegment.from_file(str(temp_wav), format="aiff")
                except Exception as aiff_error:
                    try:
                        audio = AudioSegment.from_file(str(temp_wav))
                    except Exception as generic_error:
                        raise RuntimeError(
                            f"Could not load TTS audio file. "
                            f"WAV error: {wav_error}, "
                            f"AIFF error: {aiff_error}, "
                            f"Generic error: {generic_error}"
                        )
            if audio is None:
                raise RuntimeError(f"Failed to generate speech for: {text[:50]}...")
            logger.info(
                "Generated %dms, %d samples",
                len(audio),
                len(audio.get_array_of_samples()),
            )
            if len(audio) < 50:
                raise RuntimeError(f"Generated audio too short: {len(audio)}ms")
            if len(audio.get_array_of_samples()) < 10:
                raise RuntimeError(
                    f"Generated audio has too few samples: {len(audio.get_array_of_samples())}"
                )
            if pitch != 1.0:
                audio = audio._spawn(
                    audio.raw_data,
                    overrides={"frame_rate": int(audio.frame_rate * pitch)},
                )
                audio = audio.set_frame_rate(self.sample_rate)
            for temp_file in [temp_wav, temp_aiff]:
                if temp_file.exists():
                    temp_file.unlink()
            self._audio_cache[cache_key] = audio
            return audio

        except Exception as e:
            for temp_file in [temp_wav, temp_aiff]:
                if temp_file.exists():
                    try:
                        temp_file.unlink()
                    except EnvironmentError:
                        logger.warning("Failed to delete temp file: %s", temp_file)
                        pass
            if retry:
                logger.warning(
                    "TTS generation failed, reinitializing engine and retrying..."
                )
                logger.warning("Error generating TTS: %s", e)
                self._reinit_tts()
                return self.generate_speech(text, rate, voice_index, pitch, retry=False)
            else:
                raise RuntimeError(
                    f"TTS generation failed after retry. Text: '{text[:50]}...'\n"
                    f"Error: {e}\n"
                    f"Check that macOS System Voices are installed: "
                    f"System Preferences > Accessibility > Spoken Content"
                )

    @staticmethod
    def apply_voice_profile(
        audio: AudioSegment, profile: InfranymVoiceProfile
    ) -> AudioSegment:
        """
        Apply voice profile effects to audio.

        Profiles create distinct sonic characters suitable for different
        compositional contexts.
        """
        if len(audio) < 100:
            logger.warning(
                "Audio too short (%dms), skipping voice profile effects", len(audio)
            )
            return audio

        if len(audio.get_array_of_samples()) < 10:  # Less than 10 samples
            logger.warning("Audio has too few samples, skipping voice profile effects")
            return audio

        try:
            if profile == InfranymVoiceProfile.ROBOTIC:
                audio = high_pass_filter(audio, 300)
                audio = audio.compress_dynamic_range()
            elif profile == InfranymVoiceProfile.WHISPER:
                audio = low_pass_filter(audio, 3000)
                audio = audio - 6  # -6dB
            elif profile == InfranymVoiceProfile.PROCLAMATION:
                delay_pos = min(30, len(audio) // 4)
                audio = audio.overlay(audio - 12, position=delay_pos)
            elif profile == InfranymVoiceProfile.DISTORTED:
                audio = audio.compress_dynamic_range(threshold=-20.0, ratio=10.0)
                audio = high_pass_filter(audio, 400)
                audio = low_pass_filter(audio, 8000)
            elif profile == InfranymVoiceProfile.ANCIENT:
                audio = low_pass_filter(audio, 2000)

        except Exception as e:
            logger.warning("Error applying %s profile: %s", profile.value, e)
            logger.warning("Skipping voice profile effects for this layer")
            pass
        return audio

    def apply_layer_processing(
        self, audio: AudioSegment, layer: InfranymVoiceLayer
    ) -> AudioSegment:
        """
        Apply all processing for a single layer.

        Args:
            audio: Source audio segment
            layer: Layer configuration

        Returns:
            Processed audio ready for composition
        """
        audio = self.apply_voice_profile(audio, layer.voice_profile)
        if layer.reverse:
            audio = audio.reverse()
        if layer.freq_filter:
            try:
                low_hz, high_hz = layer.freq_filter
                audio = high_pass_filter(audio, low_hz)
                audio = low_pass_filter(audio, high_hz)
            except Exception as e:
                logger.warning("Error applying frequency filter: %s", e)
                logger.warning("Skipping frequency filter")
        if layer.volume_db != 0.0:
            audio = audio + layer.volume_db
        if layer.stereo_pan != 0.0:
            audio = self._apply_pan(audio, layer.stereo_pan)
        return audio

    @staticmethod
    def _apply_pan(audio: AudioSegment, pan: float) -> AudioSegment:
        """
        Apply stereo panning.

        Args:
            audio: Mono or stereo audio
            pan: -1.0 (full left) to 1.0 (full right), 0.0 = center
        """
        # Skip panning if audio is too short or empty
        if len(audio) < 100 or len(audio.get_array_of_samples()) < 10:
            logger.warning("Audio too short for panning, skipping")
            return audio

        try:
            # Convert to stereo if mono
            if audio.channels == 1:
                audio = audio.set_channels(2)
            samples = np.array(audio.get_array_of_samples())
            if audio.channels == 2:
                samples = samples.reshape((-1, 2))
            left_gain = 1.0 - max(0, pan)
            right_gain = 1.0 + min(0, pan)
            samples[:, 0] = samples[:, 0] * left_gain
            samples[:, 1] = samples[:, 1] * right_gain
            panned = audio._spawn(samples.astype(np.int16).tobytes())
            return panned

        except Exception as e:
            logger.warning("Error applying pan: %s", e)
            return audio

    def encode_composition(
        self,
        composition: InfranymVoiceComposition,
        output_filename: str,
        export_layers: bool = True,
    ) -> Dict[str, Any]:
        """
        Encode complete three-layer Infranym composition.

        Args:
            composition: Complete composition definition
            output_filename: Base filename for output (without extension)
            export_layers: If True, export individual layers as separate files

        Returns:
            Dictionary with encoding metadata and file paths
        """
        logger.info("\n🎧 Encoding Infranym: %s", composition.title)
        logger.info("%s", "=" * 60)

        # Generate layer 1: Surface (clear, primary message)
        logger.info("📻 Layer 1 (Surface): Generating...")
        surface_audio = self.generate_speech(
            composition.surface_layer.text,
            rate=composition.surface_layer.rate,
            voice_index=0,
            pitch=composition.surface_layer.pitch,
        )
        surface_audio = self.apply_layer_processing(
            surface_audio, composition.surface_layer
        )
        logger.info("   Duration: %dms", len(surface_audio))

        # Generate layer 2: Reverse (textural, mysterious)
        logger.info("🔄 Layer 2 (Reverse): Generating...")
        reverse_audio = self.generate_speech(
            composition.reverse_layer.text,
            rate=composition.reverse_layer.rate,
            voice_index=0,
            pitch=composition.reverse_layer.pitch,
        )
        reverse_audio = self.apply_layer_processing(
            reverse_audio, composition.reverse_layer
        )
        logger.info("   Duration: %dms", len(reverse_audio))

        # Generate layer 3: Submerged (subliminal, frequency-hidden)
        logger.info("🌊 Layer 3 (Submerged): Generating...")
        submerged_audio = self.generate_speech(
            composition.submerged_layer.text,
            rate=composition.submerged_layer.rate,
            voice_index=0,
            pitch=composition.submerged_layer.pitch,
        )
        submerged_audio = self.apply_layer_processing(
            submerged_audio, composition.submerged_layer
        )
        logger.info("   Duration: %dms", len(submerged_audio))

        # Determine composite duration (longest layer)
        max_duration = max(len(surface_audio), len(reverse_audio), len(submerged_audio))

        # Pad shorter layers with silence
        surface_audio = self._pad_to_duration(surface_audio, max_duration)
        reverse_audio = self._pad_to_duration(reverse_audio, max_duration)
        submerged_audio = self._pad_to_duration(submerged_audio, max_duration)

        # Composite layers
        logger.info("🎚️  Compositing layers...")
        composite = surface_audio.overlay(reverse_audio).overlay(submerged_audio)

        # Export composite
        composite_path = self.output_dir / f"{output_filename}.wav"
        composite.export(composite_path, format="wav")
        logger.info("✅ Composite exported: %s", composite_path)

        # Export individual layers if requested
        layer_paths = {}
        if export_layers:
            layer_paths = {
                "surface": self._export_layer(
                    surface_audio, output_filename, "surface"
                ),
                "reverse": self._export_layer(
                    reverse_audio, output_filename, "reverse"
                ),
                "submerged": self._export_layer(
                    submerged_audio, output_filename, "submerged"
                ),
            }
            logger.info("✅ Individual layers exported")

        # Export metadata
        metadata = {
            "title": composition.title,
            "duration_ms": max_duration,
            "tempo_bpm": composition.tempo_bpm,
            "key_signature": composition.key_signature,
            "layers": {
                "surface": {
                    "text": composition.surface_layer.text,
                    "voice_profile": composition.surface_layer.voice_profile.value,
                    "duration_ms": len(surface_audio),
                },
                "reverse": {
                    "text": composition.reverse_layer.text,
                    "voice_profile": composition.reverse_layer.voice_profile.value,
                    "duration_ms": len(reverse_audio),
                },
                "submerged": {
                    "text": composition.submerged_layer.text,
                    "voice_profile": composition.submerged_layer.voice_profile.value,
                    "duration_ms": len(submerged_audio),
                },
            },
            "files": {"composite": str(composite_path), "layers": layer_paths},
        }

        if composition.metadata:
            metadata.update(composition.metadata)
        metadata_path = self.output_dir / f"{output_filename}_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info("📋 Metadata exported: %s", metadata_path)

        logger.info("\n🎵 Ready for Logic Pro import!")
        logger.info("   Composite track: %s", composite_path)
        if export_layers:
            logger.info("   Individual layers available for separate treatment")
        logger.info("%s", "=" * 60)

        return metadata

    @staticmethod
    def _pad_to_duration(audio: AudioSegment, target_duration_ms: int) -> AudioSegment:
        """Pad audio with silence to reach target duration"""
        if len(audio) >= target_duration_ms:
            return audio
        silence_duration = target_duration_ms - len(audio)
        silence = AudioSegment.silent(duration=silence_duration)
        return audio + silence

    def _export_layer(
        self, audio: AudioSegment, base_filename: str, layer_name: str
    ) -> str:
        """Export individual layer to file"""
        path = self.output_dir / f"{base_filename}_{layer_name}.wav"
        audio.export(path, format="wav")
        return str(path)


with open("/Volumes/LucidNonsense/White/tests/mocks/mock.wav", "rb") as f:
    audio_bytes = f.read()

# Example compositions for testing
EXAMPLE_COMPOSITIONS = {
    "alien_transmission": InfranymVoiceComposition(
        title="Alien Transmission #001",
        tempo_bpm=120,
        key_signature="E minor",
        surface_layer=InfranymVoiceLayer(
            text="Coordinates received. Commencing transmigration protocol.",
            voice_profile=InfranymVoiceProfile.ROBOTIC,
            rate=140,
            pitch=0.9,
            volume_db=0.0,
        ),
        reverse_layer=InfranymVoiceLayer(
            text="The flesh remembers what the mind forgets.",
            voice_profile=InfranymVoiceProfile.WHISPER,
            rate=120,
            pitch=1.1,
            volume_db=-18.0,
            reverse=True,
            stereo_pan=-0.5,  # Slight left
        ),
        submerged_layer=InfranymVoiceLayer(
            text="Information seeks embodiment through creative acts.",
            voice_profile=InfranymVoiceProfile.ANCIENT,
            rate=80,
            pitch=0.7,
            volume_db=-24.0,
            freq_filter=(100, 400),  # Deep bass frequencies
        ),
        metadata={
            "puzzle_solution": "TRANSMIGRATION",
            "color_agent": "indigo",
            "album": "The White Album",
        },
    ),
    "sussex_county_ghost": InfranymVoiceComposition(
        title="Sussex County Ghost Signal",
        tempo_bpm=95,
        key_signature="A minor",
        surface_layer=InfranymVoiceLayer(
            text="Static children in dead frequencies, nineteen ninety three.",
            voice_profile=InfranymVoiceProfile.DISTORTED,
            rate=160,
            pitch=1.0,
            volume_db=-3.0,
        ),
        reverse_layer=InfranymVoiceLayer(
            text="Newton still remembers every basement show.",
            voice_profile=InfranymVoiceProfile.WHISPER,
            rate=100,
            pitch=0.85,
            volume_db=-20.0,
            reverse=True,
            stereo_pan=0.7,  # Right
        ),
        submerged_layer=InfranymVoiceLayer(
            text="Temporal rebracketing begins in teenage memory.",
            voice_profile=InfranymVoiceProfile.ANCIENT,
            rate=70,
            pitch=0.6,
            volume_db=-26.0,
            freq_filter=(80, 350),
        ),
        metadata={
            "location": "Sussex County, NJ",
            "time_period": "1990-1995",
            "color_agent": "orange",
            "rebracketing_type": "temporal",
        },
    ),
    "sigil_charging_ritual": InfranymVoiceComposition(
        title="Sigil Charging Instructions",
        tempo_bpm=108,
        key_signature="D# minor",
        surface_layer=InfranymVoiceLayer(
            text="Focus intent. Merge symbol with desire. Release to the void.",
            voice_profile=InfranymVoiceProfile.PROCLAMATION,
            rate=110,
            pitch=0.95,
            volume_db=0.0,
        ),
        reverse_layer=InfranymVoiceLayer(
            text="Chaos finds order through creative destruction.",
            voice_profile=InfranymVoiceProfile.ROBOTIC,
            rate=130,
            pitch=1.2,
            volume_db=-16.0,
            reverse=True,
            stereo_pan=0.0,  # Center
        ),
        submerged_layer=InfranymVoiceLayer(
            text="Black agent offers truth through disruption.",
            voice_profile=InfranymVoiceProfile.WHISPER,
            rate=90,
            pitch=0.75,
            volume_db=-22.0,
            freq_filter=(120, 450),
        ),
        metadata={
            "ritual_type": "sigil_charging",
            "color_agent": "black",
            "chaos_method": "evp_analysis",
        },
    ),
}


def demo_encode_all_examples():
    """Encode all example compositions"""
    encoder = InfranymAudioEncoder()

    logger.info("\n🎵 INFRANYM AUDIO ENCODER - DEMO")
    logger.info("%s", "=" * 60)
    logger.info("Encoding example compositions for The White Album...")
    logger.info("")

    # Show available voices
    voices = encoder.list_available_voices()
    logger.info("Available TTS voices: %d", len(voices))
    for i, voice in enumerate(voices[:3]):  # Show first 3
        logger.info("  %d: %s", i, voice["name"])
    logger.info("")

    # Encode each example
    results = {}
    for key, composition in EXAMPLE_COMPOSITIONS.items():
        try:
            results[key] = encoder.encode_composition(
                composition, output_filename=key, export_layers=True
            )
            logger.info("")
        except Exception as e:
            logger.error("Failed to encode %s: %s", key, e)
            logger.debug("", exc_info=True)
            logger.info("")
            continue

    logger.info("\n✅ Demo complete!")
    logger.info("📁 Output directory: %s", encoder.output_dir)
    logger.info(
        "🎵 Successfully encoded: %d/%d examples",
        len(results),
        len(EXAMPLE_COMPOSITIONS),
    )
    logger.info("\nReady to import into Logic Pro for The Earthly Frames production.")
    logger.info("%s", "=" * 60)

    return results


if __name__ == "__main__":
    demo_encode_all_examples()
