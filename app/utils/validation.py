import json
import pandas as pd
from typing import Dict, Optional, Any, Union
from pydantic import BaseModel

class ValidationResult(BaseModel):
    """Results of validating a training sample"""
    is_valid: bool = True
    errors: list[str] = []
    warnings: list[str] = []
    sample_id: Optional[str] = None

class ValidationSummary(BaseModel):
    """Summary of validation results across multiple samples"""
    total_samples: int = 0
    valid_samples: int = 0
    samples_with_errors: int = 0
    samples_with_warnings: int = 0
    error_counts: Dict[str, int] = {}
    warning_counts: Dict[str, int] = {}
    results: list[ValidationResult] = []

class TrainingSampleValidator:
    """Validates training samples for completeness and consistency"""

    def __init__(self):
        self.summary = ValidationSummary()

    def validate_sample(self, sample: Any) -> ValidationResult:
        """Validate a single training sample"""
        result = ValidationResult(sample_id=f"{sample.song_title}_{sample.song_segment_sequence}")
        self._validate_metadata(sample, result)
        self._validate_audio(sample, result)
        self._validate_midi(sample, result)
        self._validate_lyrics(sample, result)
        if len(result.errors) > 0:
            result.is_valid = False

        return result

    def _validate_metadata(self, sample: Any, result: ValidationResult) -> None:
        """Validate metadata fields"""
        required_fields = [
            "song_title", "song_bpm", "song_key",
            "song_segment_name", "song_segment_duration"
        ]

        for field in required_fields:
            if not getattr(sample, field, None):
                result.errors.append(f"Missing required field: {field}")

        if sample.song_bpm and not self._is_numeric(sample.song_bpm):
            result.errors.append("BPM must be a numeric value")

        try:
            start_time = float(sample.song_segment_start_time)
            end_time = float(sample.song_segment_end_time)
            duration = float(sample.song_segment_duration)

            if abs((end_time - start_time) - duration) > 0.1:  # Allow small rounding differences
                result.errors.append("Segment duration doesn't match start/end time difference")
        except (ValueError, TypeError):
            result.errors.append("Invalid segment timing values")

    @staticmethod
    def _validate_audio(sample: Any, result: ValidationResult) -> None:
        """Validate audio data consistency"""
        # Check if audio filenames match binary data presence
        if sample.song_segment_main_audio_file_name and not sample.song_segment_main_audio_binary_data:
            result.errors.append("Main audio filename exists but binary data is missing")

        if sample.song_segment_track_audio_file_name and not sample.song_segment_track_audio_binary_data:
            result.errors.append("Track audio filename exists but binary data is missing")

        # If segment should have audio, check that it does
        has_audio_file = bool(sample.song_segment_main_audio_file_name or sample.song_segment_track_audio_file_name)
        has_audio_data = bool(sample.song_segment_main_audio_binary_data or sample.song_segment_track_audio_binary_data)

        if not has_audio_file and not has_audio_data:
            result.warnings.append("Sample contains no audio data")
    @staticmethod
    def _validate_midi(sample: Any, result: ValidationResult) -> None:
        """Validate MIDI data consistency"""
        has_midi_file = bool(sample.song_segment_track_midi_file_name)
        has_midi_data = bool(sample.song_segment_track_midi_binary_data)
        has_midi_notes = bool(sample.song_segment_track_midi_data)

        if has_midi_file and not (has_midi_data or has_midi_notes):
            result.errors.append("MIDI filename exists but no MIDI data or notes are present")

        if has_midi_notes:
            try:
                midi_notes = json.loads(sample.song_segment_track_midi_data)
                if not isinstance(midi_notes, list):
                    result.errors.append("MIDI notes data is not properly formatted as a list")
            except json.JSONDecodeError:
                result.errors.append("MIDI notes data is not valid JSON")

    @staticmethod
    def _validate_lyrics(sample: Any, result: ValidationResult) -> None:
        """Validate lyrics data consistency"""
        has_lyrics = bool(sample.song_segment_lyrics_text)
        has_lrc = bool(sample.song_segment_lyrics_lrc)

        if has_lrc and not has_lyrics:
            result.warnings.append("LRC data exists but plain lyrics are missing")

        # Only warn about missing lyrics for sections that typically have them
        if sample.song_has_lyrics and not (has_lyrics or has_lrc):
            # Check if section name suggests it should have lyrics
            lyric_sections = ["verse", "chorus", "bridge", "hook", "vocal"]
            section_name = sample.song_segment_name.lower() if sample.song_segment_name else ""

            if any(ls in section_name for ls in lyric_sections):
                result.warnings.append(f"Section '{sample.song_segment_name}' suggests vocals but no lyrics data found")
            else:
                # This is likely an instrumental section, so no warning needed
                pass
    @staticmethod
    def _is_numeric(value: Any) -> bool:
        """Check if a value is numeric or can be converted to a number"""
        if isinstance(value, (int, float)):
            return True
        if isinstance(value, str):
            try:
                float(value)
                return True
            except (ValueError, TypeError):
                return False
        return False

    def validate_dataframe(self, df: pd.DataFrame) -> ValidationSummary:
        """Validate all samples in a dataframe"""
        self.summary = ValidationSummary()
        self.summary.total_samples = len(df)

        for _, row in df.iterrows():
            result = self.validate_sample(row)
            self.summary.results.append(result)

            if not result.is_valid:
                self.summary.samples_with_errors += 1

            if result.warnings:
                self.summary.samples_with_warnings += 1

            # Count error types
            for error in result.errors:
                error_type = error.split(":")[0] if ":" in error else error
                self.summary.error_counts[error_type] = self.summary.error_counts.get(error_type, 0) + 1

            # Count warning types
            for warning in result.warnings:
                warning_type = warning.split(":")[0] if ":" in warning else warning
                self.summary.warning_counts[warning_type] = self.summary.warning_counts.get(warning_type, 0) + 1

        self.summary.valid_samples = self.summary.total_samples - self.summary.samples_with_errors
        return self.summary

    def validate_parquet_file(self, file_path: str) -> ValidationSummary:
        """Validate all samples in a parquet file"""
        df = pd.read_parquet(file_path)
        return self.validate_dataframe(df)

    def print_summary(self) -> None:
        """Print a human-readable summary of validation results"""
        print(f"\n=== Validation Summary ===")
        print(f"Total samples: {self.summary.total_samples}")
        print(f"Valid samples: {self.summary.valid_samples} ({self.summary.valid_samples/self.summary.total_samples*100:.1f}%)")
        print(f"Samples with errors: {self.summary.samples_with_errors}")
        print(f"Samples with warnings: {self.summary.samples_with_warnings}")

        if self.summary.error_counts:
            print("\nError types:")
            for error_type, count in sorted(self.summary.error_counts.items(), key=lambda x: x[1], reverse=True):
                print(f"  - {error_type}: {count}")

        if self.summary.warning_counts:
            print("\nWarning types:")
            for warning_type, count in sorted(self.summary.warning_counts.items(), key=lambda x: x[1], reverse=True):
                print(f"  - {warning_type}: {count}")

# ToDo: Validate reference_id matches the directory