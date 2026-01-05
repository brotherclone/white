import pytest
from pydantic import ValidationError

from app.structures.artifacts.infranym_voice_layer import InfranymVoiceLayer
from app.structures.enums.infranym_voice_profile import InfranymVoiceProfile


def test_create_valid_model_with_defaults():
    layer = InfranymVoiceLayer(text="Hello world")
    assert isinstance(layer, InfranymVoiceLayer)
    assert layer.text == "Hello world"
    assert layer.voice_profile == InfranymVoiceProfile.PROCLAMATION
    assert layer.rate == 150
    assert layer.pitch == 1.0
    assert layer.volume_db == 0.0
    assert layer.reverse is False
    assert layer.stereo_pan == 0.0
    assert layer.freq_filter is None


def test_create_valid_model_with_all_fields():
    layer = InfranymVoiceLayer(
        text="Test speech",
        voice_profile=InfranymVoiceProfile.WHISPER,
        rate=200,
        pitch=1.5,
        volume_db=-3.0,
        reverse=True,
        stereo_pan=-0.5,
        freq_filter=(100, 5000),
    )
    assert layer.text == "Test speech"
    assert layer.voice_profile == InfranymVoiceProfile.WHISPER
    assert layer.rate == 200
    assert layer.pitch == 1.5
    assert layer.volume_db == -3.0
    assert layer.reverse is True
    assert layer.stereo_pan == -0.5
    assert layer.freq_filter == (100, 5000)


def test_text_field_required():
    with pytest.raises(ValidationError) as exc_info:
        InfranymVoiceLayer()
    assert "text" in str(exc_info.value)


def test_text_min_length():
    with pytest.raises(ValidationError) as exc_info:
        InfranymVoiceLayer(text="")
    assert "text" in str(exc_info.value)


def test_text_max_length():
    with pytest.raises(ValidationError) as exc_info:
        InfranymVoiceLayer(text="x" * 1001)
    assert "text" in str(exc_info.value)


def test_rate_min_boundary():
    with pytest.raises(ValidationError) as exc_info:
        InfranymVoiceLayer(text="test", rate=49)
    assert "rate" in str(exc_info.value)


def test_rate_max_boundary():
    with pytest.raises(ValidationError) as exc_info:
        InfranymVoiceLayer(text="test", rate=301)
    assert "rate" in str(exc_info.value)


def test_rate_valid_boundaries():
    layer_min = InfranymVoiceLayer(text="test", rate=50)
    assert layer_min.rate == 50
    layer_max = InfranymVoiceLayer(text="test", rate=300)
    assert layer_max.rate == 300


def test_invalid_voice_profile():
    with pytest.raises(ValidationError):
        InfranymVoiceLayer(text="test", voice_profile="invalid_profile")


def test_all_voice_profiles():
    for profile in InfranymVoiceProfile:
        layer = InfranymVoiceLayer(text="test", voice_profile=profile)
        assert layer.voice_profile == profile


def test_reverse_field_boolean():
    layer_true = InfranymVoiceLayer(text="test", reverse=True)
    assert layer_true.reverse is True
    layer_false = InfranymVoiceLayer(text="test", reverse=False)
    assert layer_false.reverse is False


def test_stereo_pan_range():
    layer_left = InfranymVoiceLayer(text="test", stereo_pan=-1.0)
    assert layer_left.stereo_pan == -1.0
    layer_center = InfranymVoiceLayer(text="test", stereo_pan=0.0)
    assert layer_center.stereo_pan == 0.0
    layer_right = InfranymVoiceLayer(text="test", stereo_pan=1.0)
    assert layer_right.stereo_pan == 1.0


def test_freq_filter_tuple():
    layer = InfranymVoiceLayer(text="test", freq_filter=(200, 8000))
    assert layer.freq_filter == (200, 8000)
    assert isinstance(layer.freq_filter, tuple)


def test_model_dump():
    layer = InfranymVoiceLayer(
        text="test",
        voice_profile=InfranymVoiceProfile.ANCIENT,
        rate=100,
    )
    data = layer.model_dump(exclude_none=True, exclude_unset=True)
    assert "text" in data
    assert "voice_profile" in data
    assert "rate" in data


def test_field_descriptions():
    fields = getattr(InfranymVoiceLayer, "model_fields", None)
    assert fields is not None
    assert "text" in fields
    assert "voice_profile" in fields
    assert fields["text"].description == "Text to be synthesized into speech"


def test_wrong_type_for_text():
    with pytest.raises(ValidationError):
        InfranymVoiceLayer(text=123)


def test_wrong_type_for_rate():
    with pytest.raises(ValidationError):
        InfranymVoiceLayer(text="test", rate="fast")


def test_wrong_type_for_pitch():
    with pytest.raises(ValidationError):
        InfranymVoiceLayer(text="test", pitch="high")


def test_reverse_coercion():
    # Pydantic coerces truthy strings to True
    layer = InfranymVoiceLayer(text="test", reverse="yes")
    assert layer.reverse is True
    assert isinstance(layer.reverse, bool)
