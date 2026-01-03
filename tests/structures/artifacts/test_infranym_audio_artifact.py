"""
Tests for InfranymAudioArtifact

Tests the infranym audio artifact class which generates three-layer audio compositions
with steganographic puzzle content.
"""

import pytest
import tempfile

from unittest.mock import Mock

from app.structures.artifacts.infranym_audio_artifact import InfranymAudioArtifact
from app.structures.enums.chain_artifact_type import ChainArtifactType
from app.structures.enums.infranym_voice_profile import InfranymVoiceProfile
from app.structures.artifacts.infranym_voice_layer import InfranymVoiceLayer
from app.structures.artifacts.infranym_voice_composition import InfranymVoiceComposition


@pytest.fixture
def mock_audio_bytes():
    """Mock audio bytes for testing"""
    with open("/Volumes/LucidNonsense/White/tests/mocks/mock.wav", "rb") as f:
        return f.read()


@pytest.fixture
def temp_base_path():
    """Create temporary base path for artifacts"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


class TestInfranymAudioArtifactInit:
    """Test initialization of InfranymAudioArtifact"""

    def test_init_basic(self, mock_audio_bytes, temp_base_path):
        """Test basic initialization with required fields"""
        artifact = InfranymAudioArtifact(
            thread_id="test-thread-123",
            base_path=temp_base_path,
            audio_bytes=mock_audio_bytes,
            secret_word="TEST",
        )

        assert artifact.thread_id == "test-thread-123"
        assert artifact.chain_artifact_type == ChainArtifactType.INFRANYM_AUDIO
        assert artifact.encoder is not None
        assert artifact.surface_layer is not None
        assert artifact.reverse_layer is not None
        assert artifact.submerged_layer is not None
        assert artifact.composition is not None

    def test_init_with_secret_word(self, mock_audio_bytes, temp_base_path):
        """Test initialization with custom secret word"""
        artifact = InfranymAudioArtifact(
            thread_id="test-thread",
            base_path=temp_base_path,
            audio_bytes=mock_audio_bytes,
            secret_word="TRANSMIGRATION",
        )

        assert artifact.secret_word == "TRANSMIGRATION"
        assert artifact.surface_layer.text == "TRANSMIGRATION"
        assert artifact.reverse_layer.text == "TRANSMIGRATION"
        assert artifact.submerged_layer.text == "TRANSMIGRATION"

    def test_init_with_bpm_and_key(self, mock_audio_bytes, temp_base_path):
        """Test initialization with custom BPM and key"""
        artifact = InfranymAudioArtifact(
            thread_id="test-thread",
            base_path=temp_base_path,
            audio_bytes=mock_audio_bytes,
            secret_word="TEST",
            bpm=120,
            key="E minor",
        )

        assert artifact.bpm == 120
        assert artifact.key == "E minor"
        assert artifact.composition.tempo_bpm == 120
        assert artifact.composition.key_signature == "E minor"

    def test_init_with_custom_title(self, mock_audio_bytes, temp_base_path):
        """Test initialization with custom title"""
        artifact = InfranymAudioArtifact(
            thread_id="test-thread",
            base_path=temp_base_path,
            audio_bytes=mock_audio_bytes,
            secret_word="TEST",
            title="Alien Transmission #001",
        )

        assert artifact.title == "Alien Transmission #001"

    def test_init_defaults(self, mock_audio_bytes, temp_base_path):
        """Test default values are set correctly"""
        artifact = InfranymAudioArtifact(
            thread_id="test-thread",
            base_path=temp_base_path,
            audio_bytes=mock_audio_bytes,
            secret_word="DEFAULT",
        )

        assert artifact.title == "Infranym Audio"
        assert artifact.secret_word == "DEFAULT"
        assert artifact.bpm == 100
        assert artifact.key is None


class TestRandomSynthVoice:
    """Test random synth voice generation"""

    def test_get_random_synth_voice_surface(self, mock_audio_bytes, temp_base_path):
        """Test surface layer generation (not reversed, not filtered)"""
        artifact = InfranymAudioArtifact(
            thread_id="test-thread",
            base_path=temp_base_path,
            audio_bytes=mock_audio_bytes,
            secret_word="TEST",
        )

        # Surface layer should not be reversed and not have frequency filter
        assert artifact.surface_layer.reverse is False
        assert artifact.surface_layer.freq_filter is None
        assert artifact.surface_layer.text == "TEST"

    def test_get_random_synth_voice_reverse(self, mock_audio_bytes, temp_base_path):
        """Test reverse layer generation (reversed, not filtered)"""
        artifact = InfranymAudioArtifact(
            thread_id="test-thread",
            base_path=temp_base_path,
            audio_bytes=mock_audio_bytes,
            secret_word="TEST",
        )

        # Reverse layer should be reversed and not have frequency filter
        assert artifact.reverse_layer.reverse is True
        assert artifact.reverse_layer.freq_filter is None
        assert artifact.reverse_layer.text == "TEST"

    def test_get_random_synth_voice_submerged(self, mock_audio_bytes, temp_base_path):
        """Test submerged layer generation (not reversed, filtered)"""
        artifact = InfranymAudioArtifact(
            thread_id="test-thread",
            base_path=temp_base_path,
            audio_bytes=mock_audio_bytes,
            secret_word="TEST",
        )

        # Submerged layer should not be reversed but have frequency filter
        assert artifact.submerged_layer.reverse is False
        assert artifact.submerged_layer.freq_filter is not None
        assert len(artifact.submerged_layer.freq_filter) == 2
        assert artifact.submerged_layer.text == "TEST"

    def test_random_voice_parameters_in_valid_ranges(
        self, mock_audio_bytes, temp_base_path
    ):
        """Test that random voice parameters are within valid ranges"""
        artifact = InfranymAudioArtifact(
            thread_id="test-thread",
            base_path=temp_base_path,
            audio_bytes=mock_audio_bytes,
            secret_word="TEST",
        )

        for layer in [
            artifact.surface_layer,
            artifact.reverse_layer,
            artifact.submerged_layer,
        ]:
            # Rate should be between 50 and 150
            assert 50 <= layer.rate <= 150

            # Pitch should be between 0.0 and 0.99
            assert 0.0 <= layer.pitch <= 0.99

            # Volume should be between -20.0 and 0.0
            assert -20.0 <= layer.volume_db <= 0.0

            # Pan should be between -1.0 and 1.0
            assert -1.0 <= layer.stereo_pan <= 1.0

            # Voice profile should be valid
            assert isinstance(layer.voice_profile, InfranymVoiceProfile)

        # Submerged layer should have frequency filter
        if artifact.submerged_layer.freq_filter:
            top_freq, bottom_freq = artifact.submerged_layer.freq_filter
            assert 200.0 <= top_freq <= 1000.0
            assert 20.0 <= bottom_freq <= 201.0


class TestGenerateComposition:
    """Test composition generation"""

    def test_generate_composition_structure(self, mock_audio_bytes, temp_base_path):
        """Test that composition is generated with correct structure"""
        artifact = InfranymAudioArtifact(
            thread_id="test-thread-123",
            base_path=temp_base_path,
            audio_bytes=mock_audio_bytes,
            secret_word="PUZZLE",
        )

        composition = artifact.composition

        assert isinstance(composition, InfranymVoiceComposition)
        assert composition.title == "test-thread-123_audio_infranym"
        assert composition.surface_layer is artifact.surface_layer
        assert composition.reverse_layer is artifact.reverse_layer
        assert composition.submerged_layer is artifact.submerged_layer

    def test_generate_composition_metadata(self, mock_audio_bytes, temp_base_path):
        """Test that composition metadata is set correctly"""
        artifact = InfranymAudioArtifact(
            thread_id="test-thread",
            base_path=temp_base_path,
            audio_bytes=mock_audio_bytes,
            secret_word="MYSTERY",
        )

        metadata = artifact.composition.metadata

        assert metadata is not None
        assert metadata["puzzle_solution"] == "MYSTERY"
        assert metadata["color_agent"] == "Indigo"
        assert metadata["album"] == "Untitled White Album"

    def test_generate_composition_with_bpm_and_key(
        self, mock_audio_bytes, temp_base_path
    ):
        """Test composition respects BPM and key signature"""
        artifact = InfranymAudioArtifact(
            thread_id="test-thread",
            base_path=temp_base_path,
            audio_bytes=mock_audio_bytes,
            secret_word="TEST",
            bpm=140,
            key="D# minor",
        )

        assert artifact.composition.tempo_bpm == 140
        assert artifact.composition.key_signature == "D# minor"


class TestFlatten:
    """Test flatten method"""

    def test_flatten_returns_dict(self, mock_audio_bytes, temp_base_path):
        """Test that flatten returns a dictionary"""
        artifact = InfranymAudioArtifact(
            thread_id="test-thread",
            base_path=temp_base_path,
            audio_bytes=mock_audio_bytes,
            secret_word="TEST",
        )

        flattened = artifact.flatten()

        assert isinstance(flattened, dict)

    def test_flatten_contains_all_fields(self, mock_audio_bytes, temp_base_path):
        """Test that flatten includes all expected fields"""
        artifact = InfranymAudioArtifact(
            thread_id="test-thread",
            base_path=temp_base_path,
            audio_bytes=mock_audio_bytes,
            secret_word="TEST",
            bpm=120,
            key="A minor",
            title="Test Audio",
        )

        flattened = artifact.flatten()

        # Check InfranymAudioArtifact fields
        assert "secret_word" in flattened
        assert flattened["secret_word"] == "TEST"
        assert "bpm" in flattened
        assert flattened["bpm"] == 120
        assert "key" in flattened
        assert flattened["key"] == "A minor"
        assert "title" in flattened
        assert flattened["title"] == "Test Audio"
        assert "composition" in flattened
        assert "surface_layer" in flattened
        assert "reverse_layer" in flattened
        assert "submerged_layer" in flattened
        assert "metadata" in flattened

        # Check parent class fields
        assert "thread_id" in flattened
        assert flattened["thread_id"] == "test-thread"
        assert "chain_artifact_type" in flattened

    def test_flatten_composition_is_dict(self, mock_audio_bytes, temp_base_path):
        """Test that composition is serialized as dict in flatten"""
        artifact = InfranymAudioArtifact(
            thread_id="test-thread",
            base_path=temp_base_path,
            audio_bytes=mock_audio_bytes,
            secret_word="TEST",
        )

        flattened = artifact.flatten()

        assert isinstance(flattened["composition"], dict)
        assert isinstance(flattened["surface_layer"], dict)
        assert isinstance(flattened["reverse_layer"], dict)
        assert isinstance(flattened["submerged_layer"], dict)


class TestForPrompt:
    """Test for_prompt method"""

    def test_for_prompt_returns_string(self, mock_audio_bytes, temp_base_path):
        """Test that for_prompt returns a string"""
        artifact = InfranymAudioArtifact(
            thread_id="test-thread",
            base_path=temp_base_path,
            audio_bytes=mock_audio_bytes,
            secret_word="TEST",
        )

        prompt = artifact.for_prompt()

        assert isinstance(prompt, str)

    def test_for_prompt_includes_title(self, mock_audio_bytes, temp_base_path):
        """Test that for_prompt includes the title"""
        artifact = InfranymAudioArtifact(
            thread_id="test-thread",
            base_path=temp_base_path,
            audio_bytes=mock_audio_bytes,
            secret_word="TEST",
            title="Alien Transmission",
        )

        prompt = artifact.for_prompt()

        assert "Alien Transmission" in prompt
        assert "Audio:" in prompt

    def test_for_prompt_with_default_title(self, mock_audio_bytes, temp_base_path):
        """Test for_prompt with default title"""
        artifact = InfranymAudioArtifact(
            thread_id="test-thread",
            base_path=temp_base_path,
            audio_bytes=mock_audio_bytes,
            secret_word="TEST",
        )

        prompt = artifact.for_prompt()

        assert "Infranym Audio" in prompt


class TestSaveFile:
    """Test save_file method"""

    def test_save_file_calls_encoder(self, mock_audio_bytes, temp_base_path):
        """Test that save_file calls the encoder's encode_composition method"""
        artifact = InfranymAudioArtifact(
            thread_id="test-thread",
            base_path=temp_base_path,
            audio_bytes=mock_audio_bytes,
            secret_word="TEST",
        )

        # Mock the encoder's encode_composition method
        artifact.encoder.encode_composition = Mock()

        artifact.save_file()

        # Verify encoder was called
        artifact.encoder.encode_composition.assert_called_once()

    def test_save_file_uses_correct_composition(self, mock_audio_bytes, temp_base_path):
        """Test that save_file passes the correct composition to encoder"""
        artifact = InfranymAudioArtifact(
            thread_id="test-thread",
            base_path=temp_base_path,
            audio_bytes=mock_audio_bytes,
            secret_word="TEST",
        )

        artifact.encoder.encode_composition = Mock()
        artifact.save_file()

        call_args = artifact.encoder.encode_composition.call_args
        assert call_args[0][0] == artifact.composition
        assert call_args[1]["export_layers"] is True

    def test_save_file_uses_artifact_name_as_filename(
        self, mock_audio_bytes, temp_base_path
    ):
        """Test that save_file uses artifact_name as filename stem"""
        artifact = InfranymAudioArtifact(
            thread_id="test-thread",
            base_path=temp_base_path,
            audio_bytes=mock_audio_bytes,
            secret_word="TEST",
            artifact_name="custom_audio",
        )

        artifact.encoder.encode_composition = Mock()
        artifact.save_file()

        call_args = artifact.encoder.encode_composition.call_args
        assert call_args[1]["output_filename"] == "custom_audio"


class TestAttributeMutation:
    """Test attribute mutations"""

    def test_secret_word_can_be_updated(self, mock_audio_bytes, temp_base_path):
        """Test that secret_word can be changed after initialization"""
        artifact = InfranymAudioArtifact(
            thread_id="test-thread",
            base_path=temp_base_path,
            audio_bytes=mock_audio_bytes,
            secret_word="ORIGINAL",
        )

        artifact.secret_word = "MODIFIED"
        assert artifact.secret_word == "MODIFIED"

    def test_bpm_can_be_updated(self, mock_audio_bytes, temp_base_path):
        """Test that BPM can be changed after initialization"""
        artifact = InfranymAudioArtifact(
            thread_id="test-thread",
            base_path=temp_base_path,
            audio_bytes=mock_audio_bytes,
            secret_word="TEST",
            bpm=100,
        )

        artifact.bpm = 140
        assert artifact.bpm == 140

    def test_key_can_be_updated(self, mock_audio_bytes, temp_base_path):
        """Test that key signature can be changed after initialization"""
        artifact = InfranymAudioArtifact(
            thread_id="test-thread",
            base_path=temp_base_path,
            audio_bytes=mock_audio_bytes,
            secret_word="TEST",
            key="C major",
        )

        artifact.key = "E minor"
        assert artifact.key == "E minor"

    def test_title_can_be_updated(self, mock_audio_bytes, temp_base_path):
        """Test that title can be changed after initialization"""
        artifact = InfranymAudioArtifact(
            thread_id="test-thread",
            base_path=temp_base_path,
            audio_bytes=mock_audio_bytes,
            secret_word="TEST",
            title="Original Title",
        )

        artifact.title = "New Title"
        assert artifact.title == "New Title"


class TestLayerProperties:
    """Test properties of generated layers"""

    def test_surface_layer_is_infranym_voice_layer(
        self, mock_audio_bytes, temp_base_path
    ):
        """Test that surface layer is an InfranymVoiceLayer instance"""
        artifact = InfranymAudioArtifact(
            thread_id="test-thread",
            base_path=temp_base_path,
            audio_bytes=mock_audio_bytes,
            secret_word="TEST",
        )

        assert isinstance(artifact.surface_layer, InfranymVoiceLayer)

    def test_reverse_layer_is_infranym_voice_layer(
        self, mock_audio_bytes, temp_base_path
    ):
        """Test that reverse layer is an InfranymVoiceLayer instance"""
        artifact = InfranymAudioArtifact(
            thread_id="test-thread",
            base_path=temp_base_path,
            audio_bytes=mock_audio_bytes,
            secret_word="TEST",
        )

        assert isinstance(artifact.reverse_layer, InfranymVoiceLayer)

    def test_submerged_layer_is_infranym_voice_layer(
        self, mock_audio_bytes, temp_base_path
    ):
        """Test that submerged layer is an InfranymVoiceLayer instance"""
        artifact = InfranymAudioArtifact(
            thread_id="test-thread",
            base_path=temp_base_path,
            audio_bytes=mock_audio_bytes,
            secret_word="TEST",
        )

        assert isinstance(artifact.submerged_layer, InfranymVoiceLayer)

    def test_all_layers_use_same_text(self, mock_audio_bytes, temp_base_path):
        """Test that all layers use the same secret word as text"""
        artifact = InfranymAudioArtifact(
            thread_id="test-thread",
            base_path=temp_base_path,
            audio_bytes=mock_audio_bytes,
            secret_word="UNIFIED",
        )

        assert artifact.surface_layer.text == "UNIFIED"
        assert artifact.reverse_layer.text == "UNIFIED"
        assert artifact.submerged_layer.text == "UNIFIED"


class TestEncoderInitialization:
    """Test encoder initialization"""

    def test_encoder_is_initialized(self, mock_audio_bytes, temp_base_path):
        """Test that encoder is initialized during construction"""
        artifact = InfranymAudioArtifact(
            thread_id="test-thread",
            base_path=temp_base_path,
            audio_bytes=mock_audio_bytes,
            secret_word="TEST",
        )

        assert artifact.encoder is not None

    def test_encoder_output_dir_is_artifact_path(
        self, mock_audio_bytes, temp_base_path
    ):
        """Test that encoder output directory is set to artifact path"""
        artifact = InfranymAudioArtifact(
            thread_id="test-thread",
            base_path=temp_base_path,
            audio_bytes=mock_audio_bytes,
            secret_word="TEST",
        )

        # The encoder's output_dir should be set to the artifact directory
        artifact_path = artifact.get_artifact_path(with_file_name=False)
        assert str(artifact.encoder.output_dir) == str(artifact_path)
