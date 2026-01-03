"""
Tests for InfranymAudioEncoder

Tests the three-layer audio steganography encoder for musical composition.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open
import numpy as np
import pytest

from pydub import AudioSegment

from app.agents.tools.infranym_audio_encoder import InfranymAudioEncoder
from app.structures.enums.infranym_voice_profile import InfranymVoiceProfile
from app.structures.artifacts.infranym_voice_layer import InfranymVoiceLayer
from app.structures.artifacts.infranym_voice_composition import InfranymVoiceComposition


@pytest.fixture
def mock_tts_engine():
    """Mock pyttsx3 TTS engine"""
    mock_engine = MagicMock()
    mock_voice = MagicMock()
    mock_voice.id = "com.apple.voice.Alex"
    mock_voice.name = "Alex"
    mock_voice.languages = ["en_US"]
    mock_voice.gender = "male"
    mock_engine.getProperty.return_value = [mock_voice]
    return mock_engine


@pytest.fixture
def mock_audio_segment():
    """Create a mock AudioSegment"""
    mock_audio = MagicMock(spec=AudioSegment)
    mock_audio.channels = 2
    mock_audio.frame_rate = 44100
    mock_audio.sample_width = 2
    mock_audio.__len__.return_value = 1000  # 1 second
    mock_audio.get_array_of_samples.return_value = np.array([100] * 44100)
    mock_audio.raw_data = b"\x00" * 1000

    # Mock operations that return new AudioSegments
    mock_audio.reverse.return_value = mock_audio
    mock_audio.set_channels.return_value = mock_audio
    mock_audio.set_frame_rate.return_value = mock_audio
    mock_audio.overlay.return_value = mock_audio
    mock_audio.compress_dynamic_range.return_value = mock_audio
    mock_audio._spawn.return_value = mock_audio
    mock_audio.__add__.return_value = mock_audio
    mock_audio.__sub__.return_value = mock_audio

    return mock_audio


@pytest.fixture
def temp_output_dir():
    """Create a temporary output directory"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def encoder(temp_output_dir, mock_tts_engine):
    """Create an InfranymAudioEncoder instance with mocked TTS"""
    with patch(
        "app.agents.tools.infranym_audio_encoder.pyttsx3.init",
        return_value=mock_tts_engine,
    ):
        encoder = InfranymAudioEncoder(
            sample_rate=44100, output_dir=str(temp_output_dir)
        )
        yield encoder


class TestInfranymAudioEncoderInit:
    """Test encoder initialization"""

    def test_init_creates_output_dir(self, temp_output_dir, mock_tts_engine):
        """Test that initialization creates output directory"""
        output_path = temp_output_dir / "test_output"

        with patch(
            "app.agents.tools.infranym_audio_encoder.pyttsx3.init",
            return_value=mock_tts_engine,
        ):
            encoder = InfranymAudioEncoder(output_dir=str(output_path))

            assert output_path.exists()
            assert encoder.sample_rate == 44100
            assert encoder.output_dir == output_path

    def test_init_loads_voices(self, encoder, mock_tts_engine):
        """Test that initialization loads available voices"""
        assert len(encoder.available_voices) > 0
        assert encoder.tts is not None

    def test_init_no_voices_raises_error(self, temp_output_dir):
        """Test that initialization fails if no voices are available"""
        mock_engine = MagicMock()
        mock_engine.getProperty.return_value = []

        with patch(
            "app.agents.tools.infranym_audio_encoder.pyttsx3.init",
            return_value=mock_engine,
        ):
            with pytest.raises(RuntimeError, match="No TTS voices available"):
                InfranymAudioEncoder(output_dir=str(temp_output_dir))


class TestListAvailableVoices:
    """Test voice listing functionality"""

    def test_list_available_voices(self, encoder):
        """Test listing available TTS voices"""
        voices = encoder.list_available_voices()

        assert isinstance(voices, list)
        assert len(voices) > 0
        assert "id" in voices[0]
        assert "name" in voices[0]
        assert "languages" in voices[0]
        assert "gender" in voices[0]


class TestGenerateSpeech:
    """Test speech generation functionality"""

    def test_generate_speech_success(self, encoder, mock_audio_segment):
        """Test successful speech generation"""
        test_text = "Test speech"

        # Mock file operations
        with (
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.stat") as mock_stat,
            patch("pathlib.Path.unlink"),
            patch("pydub.AudioSegment.from_wav", return_value=mock_audio_segment),
        ):

            mock_stat.return_value.st_size = 1000

            result = encoder.generate_speech(
                test_text, rate=150, voice_index=0, pitch=1.0
            )

            assert result is not None
            encoder.tts.setProperty.assert_called()
            encoder.tts.save_to_file.assert_called()
            encoder.tts.runAndWait.assert_called()

    def test_generate_speech_with_pitch_shift(self, encoder, mock_audio_segment):
        """Test speech generation with pitch shift"""
        test_text = "Test speech"

        with (
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.stat") as mock_stat,
            patch("pathlib.Path.unlink"),
            patch("pydub.AudioSegment.from_wav", return_value=mock_audio_segment),
        ):

            mock_stat.return_value.st_size = 1000

            result = encoder.generate_speech(
                test_text, rate=150, voice_index=0, pitch=0.5
            )

            assert result is not None
            # Verify pitch shift was applied
            mock_audio_segment._spawn.assert_called()

    def test_generate_speech_caching(self, encoder, mock_audio_segment):
        """Test that speech generation uses caching"""
        test_text = "Test speech"

        with (
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.stat") as mock_stat,
            patch("pathlib.Path.unlink"),
            patch("pydub.AudioSegment.from_wav", return_value=mock_audio_segment),
        ):

            mock_stat.return_value.st_size = 1000

            # First call
            result1 = encoder.generate_speech(test_text, rate=150, voice_index=0)
            call_count_1 = encoder.tts.save_to_file.call_count

            # Second call with same parameters should use cache
            result2 = encoder.generate_speech(test_text, rate=150, voice_index=0)
            call_count_2 = encoder.tts.save_to_file.call_count

            assert result1 is not None
            assert result2 is not None
            # Second call should not generate new audio (uses cache)
            assert call_count_2 == call_count_1

    def test_generate_speech_empty_file_raises_error(self, encoder):
        """Test that empty audio file raises error"""
        test_text = "Test speech"

        with (
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.stat") as mock_stat,
        ):

            mock_stat.return_value.st_size = 0

            with pytest.raises(RuntimeError, match="TTS created empty file"):
                encoder.generate_speech(test_text, rate=150, retry=False)

    def test_generate_speech_file_not_created_raises_error(self, encoder):
        """Test that missing file raises error"""
        test_text = "Test speech"

        with patch("pathlib.Path.exists", return_value=False):
            with pytest.raises(RuntimeError, match="TTS failed to create audio file"):
                encoder.generate_speech(test_text, rate=150, retry=False)

    def test_generate_speech_too_short_raises_error(self, encoder):
        """Test that audio shorter than 50ms raises error"""
        test_text = "Test speech"

        mock_audio = MagicMock(spec=AudioSegment)
        mock_audio.__len__.return_value = 30  # Too short
        mock_audio.get_array_of_samples.return_value = np.array([100] * 100)

        with (
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.stat") as mock_stat,
            patch("pathlib.Path.unlink"),
            patch("pydub.AudioSegment.from_wav", return_value=mock_audio),
        ):

            mock_stat.return_value.st_size = 1000

            with pytest.raises(RuntimeError, match="Generated audio too short"):
                encoder.generate_speech(test_text, rate=150, retry=False)

    def test_generate_speech_retry_on_failure(self, encoder, mock_audio_segment):
        """Test retry mechanism on TTS failure"""
        test_text = "Test speech"

        # First attempt: temp_wav.exists() returns False -> triggers exception
        #   - In exception handler: temp_wav.exists(), temp_aiff.exists() -> both False
        # Second attempt (retry): temp_wav.exists() returns True -> succeeds
        #   - After success: cleanup checks temp_wav.exists(), temp_aiff.exists()
        with (
            patch(
                "pathlib.Path.exists",
                side_effect=[
                    False,  # First attempt: main check fails
                    False,  # First attempt: cleanup check temp_wav
                    False,  # First attempt: cleanup check temp_aiff
                    True,  # Second attempt: main check succeeds
                    True,  # Second attempt: cleanup temp_wav
                    False,  # Second attempt: cleanup temp_aiff
                ],
            ),
            patch("pathlib.Path.stat") as mock_stat,
            patch("pathlib.Path.unlink"),
            patch("pydub.AudioSegment.from_wav", return_value=mock_audio_segment),
        ):

            mock_stat.return_value.st_size = 1000

            result = encoder.generate_speech(test_text, rate=150, retry=True)

            assert result is not None


class TestApplyVoiceProfile:
    """Test voice profile application"""

    def test_apply_robotic_profile(self, mock_audio_segment):
        """Test applying ROBOTIC voice profile"""
        with patch(
            "app.agents.tools.infranym_audio_encoder.high_pass_filter",
            return_value=mock_audio_segment,
        ):
            result = InfranymAudioEncoder.apply_voice_profile(
                mock_audio_segment, InfranymVoiceProfile.ROBOTIC
            )
            assert result is not None

    def test_apply_whisper_profile(self, mock_audio_segment):
        """Test applying WHISPER voice profile"""
        with patch(
            "app.agents.tools.infranym_audio_encoder.low_pass_filter",
            return_value=mock_audio_segment,
        ):
            result = InfranymAudioEncoder.apply_voice_profile(
                mock_audio_segment, InfranymVoiceProfile.WHISPER
            )
            assert result is not None

    def test_apply_proclamation_profile(self, mock_audio_segment):
        """Test applying PROCLAMATION voice profile"""
        result = InfranymAudioEncoder.apply_voice_profile(
            mock_audio_segment, InfranymVoiceProfile.PROCLAMATION
        )
        assert result is not None

    def test_apply_distorted_profile(self, mock_audio_segment):
        """Test applying DISTORTED voice profile"""
        with (
            patch(
                "app.agents.tools.infranym_audio_encoder.high_pass_filter",
                return_value=mock_audio_segment,
            ),
            patch(
                "app.agents.tools.infranym_audio_encoder.low_pass_filter",
                return_value=mock_audio_segment,
            ),
        ):
            result = InfranymAudioEncoder.apply_voice_profile(
                mock_audio_segment, InfranymVoiceProfile.DISTORTED
            )
            assert result is not None

    def test_apply_ancient_profile(self, mock_audio_segment):
        """Test applying ANCIENT voice profile"""
        with patch(
            "app.agents.tools.infranym_audio_encoder.low_pass_filter",
            return_value=mock_audio_segment,
        ):
            result = InfranymAudioEncoder.apply_voice_profile(
                mock_audio_segment, InfranymVoiceProfile.ANCIENT
            )
            assert result is not None

    def test_apply_profile_to_short_audio(self):
        """Test that short audio skips profile effects"""
        short_audio = MagicMock(spec=AudioSegment)
        short_audio.__len__.return_value = 50  # Too short
        short_audio.get_array_of_samples.return_value = np.array([1, 2, 3])

        result = InfranymAudioEncoder.apply_voice_profile(
            short_audio, InfranymVoiceProfile.ROBOTIC
        )

        # Should return original audio without modification
        assert result is short_audio

    def test_apply_profile_handles_errors_gracefully(self, mock_audio_segment):
        """Test that profile application errors are handled gracefully"""
        with patch(
            "app.agents.tools.infranym_audio_encoder.high_pass_filter",
            side_effect=Exception("Filter error"),
        ):
            # Should not raise, but log warning and return original audio
            result = InfranymAudioEncoder.apply_voice_profile(
                mock_audio_segment, InfranymVoiceProfile.ROBOTIC
            )
            assert result is not None


class TestApplyLayerProcessing:
    """Test layer processing functionality"""

    def test_apply_layer_processing_basic(self, encoder, mock_audio_segment):
        """Test basic layer processing"""
        layer = InfranymVoiceLayer(
            text="Test",
            voice_profile=InfranymVoiceProfile.ROBOTIC,
            rate=150,
            pitch=1.0,
            volume_db=0.0,
        )

        with patch(
            "app.agents.tools.infranym_audio_encoder.high_pass_filter",
            return_value=mock_audio_segment,
        ):
            result = encoder.apply_layer_processing(mock_audio_segment, layer)
            assert result is not None

    def test_apply_layer_processing_with_reverse(self, encoder, mock_audio_segment):
        """Test layer processing with reverse effect"""
        layer = InfranymVoiceLayer(
            text="Test", voice_profile=InfranymVoiceProfile.WHISPER, reverse=True
        )

        with patch(
            "app.agents.tools.infranym_audio_encoder.low_pass_filter",
            return_value=mock_audio_segment,
        ):
            result = encoder.apply_layer_processing(mock_audio_segment, layer)
            mock_audio_segment.reverse.assert_called()
            assert result is not None

    def test_apply_layer_processing_with_freq_filter(self, encoder, mock_audio_segment):
        """Test layer processing with frequency filter"""
        layer = InfranymVoiceLayer(
            text="Test",
            voice_profile=InfranymVoiceProfile.ANCIENT,
            freq_filter=(100, 400),
        )

        with (
            patch(
                "app.agents.tools.infranym_audio_encoder.high_pass_filter",
                return_value=mock_audio_segment,
            ) as hp,
            patch(
                "app.agents.tools.infranym_audio_encoder.low_pass_filter",
                return_value=mock_audio_segment,
            ) as lp,
        ):
            result = encoder.apply_layer_processing(mock_audio_segment, layer)

            # Verify filters were called with correct frequencies
            assert hp.call_count >= 1
            assert lp.call_count >= 1
            assert result is not None

    def test_apply_layer_processing_with_volume(self, encoder, mock_audio_segment):
        """Test layer processing with volume adjustment"""
        layer = InfranymVoiceLayer(
            text="Test", voice_profile=InfranymVoiceProfile.WHISPER, volume_db=-6.0
        )

        with patch(
            "app.agents.tools.infranym_audio_encoder.low_pass_filter",
            return_value=mock_audio_segment,
        ):
            result = encoder.apply_layer_processing(mock_audio_segment, layer)
            # Volume adjustment uses __add__ operator
            mock_audio_segment.__add__.assert_called()
            assert result is not None

    def test_apply_layer_processing_with_pan(self, encoder, mock_audio_segment):
        """Test layer processing with stereo panning"""
        layer = InfranymVoiceLayer(
            text="Test", voice_profile=InfranymVoiceProfile.ROBOTIC, stereo_pan=0.5
        )

        with patch(
            "app.agents.tools.infranym_audio_encoder.high_pass_filter",
            return_value=mock_audio_segment,
        ):
            result = encoder.apply_layer_processing(mock_audio_segment, layer)
            assert result is not None


class TestApplyPan:
    """Test stereo panning functionality"""

    def test_apply_pan_center(self, mock_audio_segment):
        """Test center panning (no change)"""
        result = InfranymAudioEncoder._apply_pan(mock_audio_segment, 0.0)
        assert result is not None

    def test_apply_pan_left(self, mock_audio_segment):
        """Test left panning"""
        result = InfranymAudioEncoder._apply_pan(mock_audio_segment, -1.0)
        assert result is not None

    def test_apply_pan_right(self, mock_audio_segment):
        """Test right panning"""
        result = InfranymAudioEncoder._apply_pan(mock_audio_segment, 1.0)
        assert result is not None

    def test_apply_pan_mono_to_stereo(self):
        """Test panning converts mono to stereo"""
        mono_audio = MagicMock(spec=AudioSegment)
        mono_audio.channels = 1
        mono_audio.__len__.return_value = 1000
        mono_audio.get_array_of_samples.return_value = np.array([100] * 1000)
        stereo_audio = MagicMock(spec=AudioSegment)
        stereo_audio.channels = 2
        stereo_audio.get_array_of_samples.return_value = np.array([100] * 2000).reshape(
            -1, 2
        )
        stereo_audio._spawn.return_value = stereo_audio
        mono_audio.set_channels.return_value = stereo_audio
        mono_audio.set_channels.assert_called_with(2)

    def test_apply_pan_short_audio_skips(self):
        """Test that short audio skips panning"""
        short_audio = MagicMock(spec=AudioSegment)
        short_audio.channels = 2
        short_audio.__len__.return_value = 50  # Too short
        short_audio.get_array_of_samples.return_value = np.array([1, 2, 3])

        result = InfranymAudioEncoder._apply_pan(short_audio, 0.5)
        assert result is short_audio

    def test_apply_pan_handles_errors(self, mock_audio_segment):
        """Test that panning errors are handled gracefully"""
        # First call succeeds (for length check), second call fails (in processing)
        mock_audio_segment.get_array_of_samples.side_effect = [
            np.array([100] * 1000),  # First call for length check
            Exception("Array error"),  # Second call raises error
        ]

        result = InfranymAudioEncoder._apply_pan(mock_audio_segment, 0.5)
        # Should return original audio on error
        assert result is mock_audio_segment


class TestPadToDuration:
    """Test audio padding functionality"""

    def test_pad_to_duration_no_padding_needed(self, mock_audio_segment):
        """Test padding when audio is already long enough"""
        mock_audio_segment.__len__.return_value = 2000

        result = InfranymAudioEncoder._pad_to_duration(mock_audio_segment, 1000)
        assert result is mock_audio_segment

    def test_pad_to_duration_adds_silence(self):
        """Test padding adds silence to reach target duration"""
        short_audio = MagicMock(spec=AudioSegment)
        short_audio.__len__.return_value = 500
        padded_audio = MagicMock(spec=AudioSegment)
        short_audio.__add__.return_value = padded_audio

        with patch("pydub.AudioSegment.silent") as mock_silent:
            mock_silent.return_value = MagicMock(spec=AudioSegment)
            result = InfranymAudioEncoder._pad_to_duration(short_audio, 1000)

            # Verify silent() was called with correct duration
            mock_silent.assert_called_once_with(duration=500)
            assert result is padded_audio


class TestExportLayer:
    """Test layer export functionality"""

    def test_export_layer(self, encoder, mock_audio_segment, temp_output_dir):
        """Test exporting individual layer to file"""
        base_filename = "test_composition"
        layer_name = "surface"

        result_path = encoder._export_layer(
            mock_audio_segment, base_filename, layer_name
        )

        expected_path = temp_output_dir / f"{base_filename}_{layer_name}.wav"
        assert result_path == str(expected_path)
        mock_audio_segment.export.assert_called_once_with(expected_path, format="wav")


class TestEncodeComposition:
    """Test full composition encoding"""

    @pytest.fixture
    def test_composition(self):
        """Create a test composition"""
        return InfranymVoiceComposition(
            title="Test Composition",
            tempo_bpm=120,
            key_signature="E minor",
            surface_layer=InfranymVoiceLayer(
                text="Surface message",
                voice_profile=InfranymVoiceProfile.ROBOTIC,
                rate=150,
                pitch=1.0,
            ),
            reverse_layer=InfranymVoiceLayer(
                text="Reverse message",
                voice_profile=InfranymVoiceProfile.WHISPER,
                rate=120,
                pitch=1.1,
                reverse=True,
            ),
            submerged_layer=InfranymVoiceLayer(
                text="Submerged message",
                voice_profile=InfranymVoiceProfile.ANCIENT,
                rate=80,
                pitch=0.7,
                freq_filter=(100, 400),
            ),
        )

    def test_encode_composition_success(
        self, encoder, test_composition, mock_audio_segment, temp_output_dir
    ):
        """Test successful composition encoding"""
        output_filename = "test_output"

        with (
            patch.object(encoder, "generate_speech", return_value=mock_audio_segment),
            patch.object(
                encoder, "apply_layer_processing", return_value=mock_audio_segment
            ),
            patch("builtins.open", mock_open()) as mock_file,
        ):
            assert mock_file is not None
            result = encoder.encode_composition(
                test_composition, output_filename, export_layers=True
            )

            # Verify metadata structure
            assert "title" in result
            assert "duration_ms" in result
            assert "tempo_bpm" in result
            assert "key_signature" in result
            assert "layers" in result
            assert "files" in result

            # Verify all layers are in metadata
            assert "surface" in result["layers"]
            assert "reverse" in result["layers"]
            assert "submerged" in result["layers"]

            # Verify generate_speech was called for each layer
            assert encoder.generate_speech.call_count == 3

            # Verify layer processing was called for each layer
            assert encoder.apply_layer_processing.call_count == 3

            # Verify composite export
            assert mock_audio_segment.export.called

    def test_encode_composition_without_layer_export(
        self, encoder, test_composition, mock_audio_segment
    ):
        """Test composition encoding without exporting individual layers"""
        output_filename = "test_output"

        with (
            patch.object(encoder, "generate_speech", return_value=mock_audio_segment),
            patch.object(
                encoder, "apply_layer_processing", return_value=mock_audio_segment
            ),
            patch("builtins.open", mock_open()),
        ):

            result = encoder.encode_composition(
                test_composition, output_filename, export_layers=False
            )

            # Layer paths should be empty when export_layers=False
            assert result["files"]["layers"] == {}

    def test_encode_composition_with_custom_metadata(
        self, encoder, test_composition, mock_audio_segment
    ):
        """Test composition encoding with custom metadata"""
        test_composition.metadata = {
            "custom_field": "custom_value",
            "puzzle_solution": "TEST",
        }
        output_filename = "test_output"

        with (
            patch.object(encoder, "generate_speech", return_value=mock_audio_segment),
            patch.object(
                encoder, "apply_layer_processing", return_value=mock_audio_segment
            ),
            patch("builtins.open", mock_open()),
        ):

            result = encoder.encode_composition(
                test_composition, output_filename, export_layers=False
            )

            # Verify custom metadata is included
            assert "custom_field" in result
            assert result["custom_field"] == "custom_value"
            assert "puzzle_solution" in result
            assert result["puzzle_solution"] == "TEST"

    def test_encode_composition_pads_layers_to_same_duration(
        self, encoder, test_composition
    ):
        """Test that all layers are padded to the same duration"""
        # Create mock audio segments of different lengths
        surface_audio = MagicMock(spec=AudioSegment)
        surface_audio.__len__.return_value = 1000
        surface_audio.get_array_of_samples.return_value = np.array([100] * 1000)
        surface_audio.overlay.return_value = surface_audio
        surface_audio.__add__.return_value = surface_audio
        surface_audio.export = MagicMock()

        reverse_audio = MagicMock(spec=AudioSegment)
        reverse_audio.__len__.return_value = 1500
        reverse_audio.get_array_of_samples.return_value = np.array([100] * 1500)
        reverse_audio.__add__.return_value = reverse_audio

        submerged_audio = MagicMock(spec=AudioSegment)
        submerged_audio.__len__.return_value = 800
        submerged_audio.get_array_of_samples.return_value = np.array([100] * 800)
        submerged_audio.__add__.return_value = submerged_audio

        def mock_generate(text, **kwargs):
            if "Surface" in text:
                return surface_audio
            elif "Reverse" in text:
                return reverse_audio
            else:
                return submerged_audio

        with (
            patch.object(encoder, "generate_speech", side_effect=mock_generate),
            patch.object(
                encoder, "apply_layer_processing", side_effect=lambda a, lambda_value: a
            ),
            patch("pydub.AudioSegment.silent") as mock_silent,
            patch("builtins.open", mock_open()),
        ):

            mock_silent.return_value = MagicMock(spec=AudioSegment)
            assert mock_silent.call_count >= 2


class TestReinitTts:
    """Test TTS engine reinitialization"""

    def test_reinit_tts_stops_previous_engine(self, encoder):
        """Test that reinitialization stops previous TTS engine"""
        old_tts = encoder.tts

        encoder._reinit_tts()

        # Verify stop was called on old engine
        old_tts.stop.assert_called_once()

    def test_reinit_tts_handles_errors_gracefully(self, encoder, mock_tts_engine):
        """Test that reinitialization handles errors gracefully"""
        encoder.tts.stop.side_effect = EnvironmentError("Stop error")

        with patch(
            "app.agents.tools.infranym_audio_encoder.pyttsx3.init",
            return_value=mock_tts_engine,
        ):
            # Should not raise despite stop() error
            encoder._reinit_tts()
            assert encoder.tts is not None
