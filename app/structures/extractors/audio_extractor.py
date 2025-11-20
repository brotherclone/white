import os
import numpy as np
import pandas as pd
import librosa

from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv

from app.structures.music.rainbow_table.raindbow_audio_feature import (
    RainbowAudioFeature,
)
from app.util.manifest_loader import load_manifest


class AudioExtractor:
    """Standalone audio feature extractor"""

    def __init__(self, manifest_id: str, sample_rate: int = None):
        load_dotenv()
        self.manifest_id = manifest_id
        self.sample_rate = sample_rate
        self.manifest_path = os.path.join(
            os.environ["MANIFEST_PATH"], manifest_id, f"{manifest_id}.yml"
        )

        if os.path.exists(self.manifest_path):
            self.manifest = load_manifest(self.manifest_path)
        else:
            self.manifest = None

    def extract_segment_features(
        self, audio_path: str, start_time: float, end_time: float
    ) -> Dict[str, Any]:
        """Extract audio features for a specific time segment - this is the method ManifestExtractor expects"""
        try:
            rainbow_features = self.load_audio_segment(audio_path, start_time, end_time)
            # Convert RainbowAudioFeature to dictionary format expected by ManifestExtractor
            return {
                "rms_energy": getattr(rainbow_features, "rms_energy", 0.0),
                "spectral_centroid": getattr(
                    rainbow_features, "spectral_centroid", 0.0
                ),
                "attack_time": getattr(rainbow_features, "attack_time", 0.0),
                "decay_profile": getattr(rainbow_features, "decay_profile", []),
            }
        except Exception as e:
            print(f"Error in extract_segment_features: {e}")
            return {
                "rms_energy": 0.0,
                "spectral_centroid": 0.0,
                "attack_time": 0.0,
                "decay_profile": [],
            }

    @staticmethod
    def _analyze_silence(
        segment: np.ndarray, sr: int, threshold_db: float
    ) -> Dict[str, Any]:
        """Analyze silence patterns in an audio segment - returns dict, not RainbowAudioFeature"""
        if len(segment) == 0:
            return {
                "is_mostly_silence": True,
                "non_silence_ratio": 0.0,
                "silence_gaps": [],
                "non_silence_regions": [],
                "peak_amplitude": 0.0,
                "rms_energy": 0.0,
            }

        # Convert to dB
        rms = librosa.feature.rms(y=segment, hop_length=sr // 20)[0]  # 50ms frames
        rms_db = librosa.amplitude_to_db(rms, ref=np.max)

        # Find non-silent frames
        non_silent_frames = rms_db > threshold_db
        non_silence_ratio = np.sum(non_silent_frames) / len(non_silent_frames)

        # Convert frame indices back to time
        hop_length = sr // 20
        frame_times = librosa.frames_to_time(
            np.arange(len(non_silent_frames)), sr=sr, hop_length=hop_length
        )

        # Find transitions
        transitions = np.diff(non_silent_frames.astype(int))
        silence_starts = np.where(transitions == -1)[0] + 1
        silence_ends = np.where(transitions == 1)[0] + 1
        non_silence_starts = np.where(transitions == 1)[0] + 1
        non_silence_ends = np.where(transitions == -1)[0] + 1

        # Handle edge cases
        if non_silent_frames[0]:
            non_silence_starts = np.concatenate([[0], non_silence_starts])
        if non_silent_frames[-1]:
            non_silence_ends = np.concatenate(
                [non_silence_ends, [len(non_silent_frames)]]
            )

        if not non_silent_frames[0]:
            silence_starts = np.concatenate([[0], silence_starts])
        if not non_silent_frames[-1]:
            silence_ends = np.concatenate([silence_ends, [len(non_silent_frames)]])

        # Convert to time ranges and calculate gap durations
        silence_gap_durations = []  # List of gap durations (floats)
        non_silence_time_ranges = []  # List of (start, end) tuples

        for start, end in zip(silence_starts, silence_ends):
            if start < len(frame_times) and end <= len(frame_times):
                gap_start = frame_times[start]
                gap_end = frame_times[min(end - 1, len(frame_times) - 1)]
                gap_duration = float(gap_end - gap_start)
                silence_gap_durations.append(gap_duration)

        for start, end in zip(non_silence_starts, non_silence_ends):
            if start < len(frame_times) and end <= len(frame_times):
                range_start = float(frame_times[start])
                range_end = float(frame_times[min(end - 1, len(frame_times) - 1)])
                non_silence_time_ranges.append((range_start, range_end))

        return {
            "is_mostly_silence": non_silence_ratio < 0.1,
            "non_silence_ratio": float(non_silence_ratio),
            "silence_gaps": silence_gap_durations,  # Now list of floats (durations)
            "non_silence_regions": non_silence_time_ranges,  # List of (start, end) tuples
            "peak_amplitude": float(np.max(np.abs(segment))),
            "rms_energy": float(np.sqrt(np.mean(segment**2))),
        }

    @staticmethod
    def _create_silence_features(
        audio_path: str,
        start_time: float,
        end_time: float,
        segment: np.ndarray,
        silence_analysis: Dict[str, Any],
    ) -> RainbowAudioFeature:
        """Create features for a mostly-silent segment"""
        duration = end_time - start_time
        return RainbowAudioFeature(
            audio_file_path=audio_path,
            segment_start_time=start_time,
            segment_end_time=end_time,
            duration=duration,
            duration_samples=len(segment),
            is_mostly_silence=True,
            rms_energy=silence_analysis.get("rms_energy", 0.0),
            spectral_centroid=0.0,
            zero_crossing_rate=0.0,
            tempo=0.0,
            mfcc=np.array([]),
            chroma=np.array([]),
            spectral_contrast=np.array([]),
            onset_frames=np.array([]),
            onset_strength=np.array([]),
            harmonic_ratio=0.0,
            attack_time=0.0,
            decay_profile=np.array([]),
            silence_confidence=1.0 - silence_analysis.get("non_silence_ratio", 0.0),
            non_silence_ratio=silence_analysis.get("non_silence_ratio", 0.0),
            silence_gaps=silence_analysis.get("silence_gaps", []),
            non_silence_regions=silence_analysis.get("non_silence_regions", []),
            peak_amplitude=silence_analysis.get("peak_amplitude", 0.0),
        )

    @staticmethod
    def _create_audio_features(
        audio_path: str,
        start_time: float,
        end_time: float,
        segment: np.ndarray,
        sr: int,
        silence_analysis: Dict[str, Any],
    ) -> RainbowAudioFeature:
        """Create full audio features for non-silent segments"""
        duration = end_time - start_time

        # Calculate basic features with error handling
        try:
            rms_energy = float(librosa.feature.rms(y=segment).mean())
        except Exception:
            rms_energy = 0.0

        try:
            spectral_centroid = float(
                librosa.feature.spectral_centroid(y=segment, sr=sr).mean()
            )
        except Exception:
            spectral_centroid = 0.0

        try:
            zero_crossing_rate = float(
                librosa.feature.zero_crossing_rate(segment).mean()
            )
        except Exception:
            zero_crossing_rate = 0.0

        try:
            tempo = (
                float(librosa.feature.rhythm.tempo(y=segment, sr=sr)[0])
                if len(segment) > sr
                else 0
            )
        except Exception:
            tempo = 0.0

        try:
            mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=13).T
        except Exception:
            mfcc = np.array([])

        # Handle chroma with tuning issues - suppress warnings
        try:
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                chroma = librosa.feature.chroma_stft(y=segment, sr=sr).T
        except Exception:
            chroma = np.array([])

        try:
            spectral_contrast = librosa.feature.spectral_contrast(y=segment, sr=sr).T
        except Exception:
            spectral_contrast = np.array([])

        try:
            onset_frames = librosa.onset.onset_detect(y=segment, sr=sr)
        except Exception:
            onset_frames = np.array([])

        try:
            onset_strength = librosa.onset.onset_strength(y=segment, sr=sr)
        except Exception:
            onset_strength = np.array([])

        return RainbowAudioFeature(
            audio_file_path=audio_path,
            segment_start_time=start_time,
            segment_end_time=end_time,
            duration=duration,
            duration_samples=len(segment),
            is_mostly_silence=False,
            rms_energy=rms_energy,
            spectral_centroid=spectral_centroid,
            zero_crossing_rate=zero_crossing_rate,
            tempo=tempo,
            mfcc=mfcc,
            chroma=chroma,
            spectral_contrast=spectral_contrast,
            onset_frames=onset_frames,
            onset_strength=onset_strength,
            harmonic_ratio=AudioExtractor._calculate_harmonic_ratio(segment),
            attack_time=AudioExtractor._calculate_attack_time(segment, sr),
            decay_profile=AudioExtractor._calculate_decay_profile(segment, sr),
            non_silence_ratio=silence_analysis.get("non_silence_ratio", 0.0),
            silence_gaps=silence_analysis.get("silence_gaps", []),
            non_silence_regions=silence_analysis.get("non_silence_regions", []),
            peak_amplitude=silence_analysis.get("peak_amplitude", 0.0),
        )

    @staticmethod
    def _empty_audio_features() -> RainbowAudioFeature:
        """Return empty audio features structure"""
        return RainbowAudioFeature(
            audio_file_path="",  # Changed from None to empty string
            segment_start_time=0.0,
            segment_end_time=0.0,
            duration=0.0,
            duration_samples=0,
            rms_energy=0.0,
            spectral_centroid=0.0,
            zero_crossing_rate=0.0,
            tempo=0.0,
            mfcc=np.array([]),
            chroma=np.array([]),
            spectral_contrast=np.array([]),
            onset_frames=np.array([]),
            onset_strength=np.array([]),
            harmonic_ratio=0.0,
            attack_time=0.0,
            decay_profile=np.array([]),
            is_mostly_silence=None,
            silence_analysis=None,
            non_silence_ratio=None,
            silence_gaps=None,
            non_silence_regions=None,
            peak_amplitude=None,
            silence_confidence=None,
        )

    @staticmethod
    def _calculate_harmonic_ratio(segment: np.ndarray) -> float:
        """Calculate ratio of harmonic to percussive content"""
        try:
            # Ensure segment is a numpy array
            segment = np.asarray(segment)
            harmonic, percussive = librosa.effects.hpss(segment)
            harmonic_energy = float(np.sum(harmonic**2))
            percussive_energy = float(np.sum(percussive**2))
            total_energy = harmonic_energy + percussive_energy
            return float(harmonic_energy / total_energy) if total_energy > 0 else 0.0
        except Exception as e:
            print(f"Error calculating harmonic ratio: {e}")
            return 0.0

    @staticmethod
    def _calculate_attack_time(segment: np.ndarray, sr: int) -> float:
        """Calculate attack time - how quickly the segment starts"""
        if len(segment) < sr // 10:  # Less than 0.1 seconds
            return 0.0

        # Find the time to reach 90% of peak amplitude
        peak_amp = np.max(np.abs(segment))
        if peak_amp == 0:
            return 0.0

        threshold = 0.9 * peak_amp
        attack_samples = np.where(np.abs(segment) >= threshold)[0]

        if len(attack_samples) > 0:
            return float(attack_samples[0] / sr)
        return 0.0

    @staticmethod
    def _calculate_decay_profile(segment: np.ndarray, sr: int) -> np.ndarray:
        """Calculate how the segment decays over time"""
        if len(segment) < sr // 10:
            return np.array([])

        # Calculate RMS in overlapping windows
        hop_length = sr // 20  # 50ms windows
        frame_length = sr // 10  # 100ms frames

        try:
            rms_frames = librosa.feature.rms(
                y=segment, frame_length=frame_length, hop_length=hop_length
            )[0]
            return rms_frames
        except Exception as e:
            print(f"Error calculating decay profile: {e}")
            return np.array([])

    @staticmethod
    def load_audio_segment(
        audio_path: str,
        start_time: float,
        end_time: float,
        sample_rate: int = None,
        silence_threshold_db: float = -40.0,
        min_non_silence_ratio: float = 0.1,
    ) -> RainbowAudioFeature:
        """Standalone method to extract audio features for a temporal segment with silence detection"""
        try:
            # Load the full audio file using soundfile-backed helper
            from app.util.audio_io import load_audio

            y, sr = load_audio(audio_path, sr=sample_rate)

            # Convert time to sample indices
            start_sample = int(start_time * sr)
            end_sample = int(end_time * sr)

            # Extract the segment
            segment = y[start_sample:end_sample]

            if len(segment) == 0:
                return AudioExtractor._empty_audio_features()

            # Analyze silence content
            silence_analysis = AudioExtractor._analyze_silence(
                segment, sr, silence_threshold_db
            )

            # If segment is mostly silence, mark it appropriately
            if silence_analysis["non_silence_ratio"] < min_non_silence_ratio:
                audio_features = AudioExtractor._create_silence_features(
                    audio_path, start_time, end_time, segment, silence_analysis
                )
            else:
                # Process as normal audio segment
                audio_features = AudioExtractor._create_audio_features(
                    audio_path, start_time, end_time, segment, sr, silence_analysis
                )

            return audio_features

        except Exception as e:
            print(f"Error loading audio segment {start_time:.3f}s-{end_time:.3f}s: {e}")
            return AudioExtractor._empty_audio_features()

    @staticmethod
    def _analyze_silence(
        segment: np.ndarray, sr: int, threshold_db: float
    ) -> Dict[str, Any]:
        """Analyze silence patterns in an audio segment - returns dict, not RainbowAudioFeature"""
        if len(segment) == 0:
            return {
                "is_mostly_silence": True,
                "non_silence_ratio": 0.0,
                "silence_gaps": [],
                "non_silence_regions": [],
                "peak_amplitude": 0.0,
                "rms_energy": 0.0,
            }

        # Convert to dB
        rms = librosa.feature.rms(y=segment, hop_length=sr // 20)[0]  # 50ms frames
        rms_db = librosa.amplitude_to_db(rms, ref=np.max)

        # Find non-silent frames
        non_silent_frames = rms_db > threshold_db
        non_silence_ratio = np.sum(non_silent_frames) / len(non_silent_frames)

        # Convert frame indices back to time
        hop_length = sr // 20
        frame_times = librosa.frames_to_time(
            np.arange(len(non_silent_frames)), sr=sr, hop_length=hop_length
        )

        # Find transitions
        transitions = np.diff(non_silent_frames.astype(int))
        silence_starts = np.where(transitions == -1)[0] + 1
        silence_ends = np.where(transitions == 1)[0] + 1
        non_silence_starts = np.where(transitions == 1)[0] + 1
        non_silence_ends = np.where(transitions == -1)[0] + 1

        # Handle edge cases
        if non_silent_frames[0]:
            non_silence_starts = np.concatenate([[0], non_silence_starts])
        if non_silent_frames[-1]:
            non_silence_ends = np.concatenate(
                [non_silence_ends, [len(non_silent_frames)]]
            )

        if not non_silent_frames[0]:
            silence_starts = np.concatenate([[0], silence_starts])
        if not non_silent_frames[-1]:
            silence_ends = np.concatenate([silence_ends, [len(non_silent_frames)]])

        # Convert to time ranges and calculate gap durations
        silence_gap_durations = []  # List of gap durations (floats)
        non_silence_time_ranges = []  # List of (start, end) tuples

        for start, end in zip(silence_starts, silence_ends):
            if start < len(frame_times) and end <= len(frame_times):
                gap_start = frame_times[start]
                gap_end = frame_times[min(end - 1, len(frame_times) - 1)]
                gap_duration = float(gap_end - gap_start)
                silence_gap_durations.append(gap_duration)

        for start, end in zip(non_silence_starts, non_silence_ends):
            if start < len(frame_times) and end <= len(frame_times):
                range_start = float(frame_times[start])
                range_end = float(frame_times[min(end - 1, len(frame_times) - 1)])
                non_silence_time_ranges.append((range_start, range_end))

        return {
            "is_mostly_silence": non_silence_ratio < 0.1,
            "non_silence_ratio": float(non_silence_ratio),
            "silence_gaps": silence_gap_durations,  # Now list of floats (durations)
            "non_silence_regions": non_silence_time_ranges,  # List of (start, end) tuples
            "peak_amplitude": float(np.max(np.abs(segment))),
            "rms_energy": float(np.sqrt(np.mean(segment**2))),
        }

    @staticmethod
    def _create_silence_features(
        audio_path: str,
        start_time: float,
        end_time: float,
        segment: np.ndarray,
        silence_analysis: Dict[str, Any],
    ) -> RainbowAudioFeature:
        """Create features for a mostly-silent segment"""
        duration = end_time - start_time
        return RainbowAudioFeature(
            audio_file_path=audio_path,
            segment_start_time=start_time,
            segment_end_time=end_time,
            duration=duration,
            duration_samples=len(segment),
            is_mostly_silence=True,
            rms_energy=silence_analysis.get("rms_energy", 0.0),
            spectral_centroid=0.0,
            zero_crossing_rate=0.0,
            tempo=0.0,
            mfcc=np.array([]),
            chroma=np.array([]),
            spectral_contrast=np.array([]),
            onset_frames=np.array([]),
            onset_strength=np.array([]),
            harmonic_ratio=0.0,
            attack_time=0.0,
            decay_profile=np.array([]),
            silence_confidence=1.0 - silence_analysis.get("non_silence_ratio", 0.0),
            non_silence_ratio=silence_analysis.get("non_silence_ratio", 0.0),
            silence_gaps=silence_analysis.get("silence_gaps", []),
            non_silence_regions=silence_analysis.get("non_silence_regions", []),
            peak_amplitude=silence_analysis.get("peak_amplitude", 0.0),
        )

    @staticmethod
    def _create_audio_features(
        audio_path: str,
        start_time: float,
        end_time: float,
        segment: np.ndarray,
        sr: int,
        silence_analysis: Dict[str, Any],
    ) -> RainbowAudioFeature:
        """Create full audio features for non-silent segments"""
        duration = end_time - start_time

        # Calculate basic features with error handling
        try:
            rms_energy = float(librosa.feature.rms(y=segment).mean())
        except Exception:
            rms_energy = 0.0

        try:
            spectral_centroid = float(
                librosa.feature.spectral_centroid(y=segment, sr=sr).mean()
            )
        except Exception:
            spectral_centroid = 0.0

        try:
            zero_crossing_rate = float(
                librosa.feature.zero_crossing_rate(segment).mean()
            )
        except Exception:
            zero_crossing_rate = 0.0

        try:
            tempo = (
                float(librosa.feature.rhythm.tempo(y=segment, sr=sr)[0])
                if len(segment) > sr
                else 0
            )
        except Exception:
            tempo = 0.0

        try:
            mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=13).T
        except Exception:
            mfcc = np.array([])

        # Handle chroma with tuning issues - suppress warnings
        try:
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                chroma = librosa.feature.chroma_stft(y=segment, sr=sr).T
        except Exception:
            chroma = np.array([])

        try:
            spectral_contrast = librosa.feature.spectral_contrast(y=segment, sr=sr).T
        except Exception:
            spectral_contrast = np.array([])

        try:
            onset_frames = librosa.onset.onset_detect(y=segment, sr=sr)
        except Exception:
            onset_frames = np.array([])

        try:
            onset_strength = librosa.onset.onset_strength(y=segment, sr=sr)
        except Exception:
            onset_strength = np.array([])

        return RainbowAudioFeature(
            audio_file_path=audio_path,
            segment_start_time=start_time,
            segment_end_time=end_time,
            duration=duration,
            duration_samples=len(segment),
            is_mostly_silence=False,
            rms_energy=rms_energy,
            spectral_centroid=spectral_centroid,
            zero_crossing_rate=zero_crossing_rate,
            tempo=tempo,
            mfcc=mfcc,
            chroma=chroma,
            spectral_contrast=spectral_contrast,
            onset_frames=onset_frames,
            onset_strength=onset_strength,
            harmonic_ratio=AudioExtractor._calculate_harmonic_ratio(segment),
            attack_time=AudioExtractor._calculate_attack_time(segment, sr),
            decay_profile=AudioExtractor._calculate_decay_profile(segment, sr),
            non_silence_ratio=silence_analysis.get("non_silence_ratio", 0.0),
            silence_gaps=silence_analysis.get("silence_gaps", []),
            non_silence_regions=silence_analysis.get("non_silence_regions", []),
            peak_amplitude=silence_analysis.get("peak_amplitude", 0.0),
        )

    @staticmethod
    def _empty_audio_features() -> RainbowAudioFeature:
        """Return empty audio features structure"""
        return RainbowAudioFeature(
            audio_file_path="",  # Changed from None to empty string
            segment_start_time=0.0,
            segment_end_time=0.0,
            duration=0.0,
            duration_samples=0,
            rms_energy=0.0,
            spectral_centroid=0.0,
            zero_crossing_rate=0.0,
            tempo=0.0,
            mfcc=np.array([]),
            chroma=np.array([]),
            spectral_contrast=np.array([]),
            onset_frames=np.array([]),
            onset_strength=np.array([]),
            harmonic_ratio=0.0,
            attack_time=0.0,
            decay_profile=np.array([]),
            is_mostly_silence=None,
            silence_analysis=None,
            non_silence_ratio=None,
            silence_gaps=None,
            non_silence_regions=None,
            peak_amplitude=None,
            silence_confidence=None,
        )

    @staticmethod
    def _calculate_harmonic_ratio(segment: np.ndarray) -> float:
        """Calculate ratio of harmonic to percussive content"""
        try:
            # Ensure segment is a numpy array
            segment = np.asarray(segment)
            harmonic, percussive = librosa.effects.hpss(segment)
            harmonic_energy = float(np.sum(harmonic**2))
            percussive_energy = float(np.sum(percussive**2))
            total_energy = harmonic_energy + percussive_energy
            return float(harmonic_energy / total_energy) if total_energy > 0 else 0.0
        except Exception as e:
            print(f"Error calculating harmonic ratio: {e}")
            return 0.0

    @staticmethod
    def _calculate_attack_time(segment: np.ndarray, sr: int) -> float:
        """Calculate attack time - how quickly the segment starts"""
        if len(segment) < sr // 10:  # Less than 0.1 seconds
            return 0.0

        # Find the time to reach 90% of peak amplitude
        peak_amp = np.max(np.abs(segment))
        if peak_amp == 0:
            return 0.0

        threshold = 0.9 * peak_amp
        attack_samples = np.where(np.abs(segment) >= threshold)[0]

        if len(attack_samples) > 0:
            return float(attack_samples[0] / sr)
        return 0.0

    @staticmethod
    def _calculate_decay_profile(segment: np.ndarray, sr: int) -> np.ndarray:
        """Calculate how the segment decays over time"""
        if len(segment) < sr // 10:
            return np.array([])

        # Calculate RMS in overlapping windows
        hop_length = sr // 20  # 50ms windows
        frame_length = sr // 10  # 100ms frames

        try:
            rms_frames = librosa.feature.rms(
                y=segment, frame_length=frame_length, hop_length=hop_length
            )[0]
            return rms_frames
        except Exception as e:
            print(f"Error calculating decay profile: {e}")
            return np.array([])

    def load_raw_audio_segment(self, raw_audio_segment_row: pd.Series) -> np.ndarray:
        """Load raw audio for a specific segment from the training data"""
        audio_features = raw_audio_segment_row["audio_features"]
        audio_path = audio_features.audio_file_path
        start_time = audio_features.segment_start_time
        end_time = audio_features.segment_end_time

        if not audio_path or not Path(audio_path).exists():
            return np.array([])

        # Prefer calling librosa.load (tests patch this). Import the module at
        # call-time to ensure any test patching of `librosa.load` on the
        # sys.modules object is respected.
        try:
            import importlib

            librosa_mod = importlib.import_module("librosa")
            y, sr = librosa_mod.load(audio_path, sr=self.sample_rate)
        except Exception:
            from app.util.audio_io import load_audio

            y, sr = load_audio(audio_path, sr=self.sample_rate)
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)

        return y[start_sample:end_sample]


if __name__ == "__main__":
    # Set up a sample rate and manifest id
    sample_rate = 22050
    manifest_id = "01_01"
    audio_extractor = AudioExtractor(sample_rate=sample_rate, manifest_id=manifest_id)
    print("Loaded manifest:", audio_extractor.manifest)

    # Example audio file (update path as needed)
    example_audio_path = (
        "/Volumes/LucidNonsense/White/staged_raw_material/01_01/01_01_02.wav"
    )
    if os.path.exists(example_audio_path):
        # Extract features from a segment (first 2 seconds)
        features = audio_extractor.load_audio_segment(
            audio_path=example_audio_path, start_time=0.0, end_time=2.0
        )
        print("\nAudio segment features:")
        print(features)

        # Simulate a DataFrame row for load_raw_audio_segment
        segment_row = pd.Series({"audio_features": features})
        raw_segment = audio_extractor.load_raw_audio_segment(segment_row)
        print("\nRaw audio segment shape:", raw_segment.shape)
    else:
        print(f"Example audio file not found: {example_audio_path}")
