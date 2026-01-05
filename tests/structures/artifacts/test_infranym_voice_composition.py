import pytest
from pydantic import ValidationError

from app.structures.artifacts.infranym_voice_composition import (
    InfranymVoiceComposition,
)
from app.structures.artifacts.infranym_voice_layer import InfranymVoiceLayer
from app.structures.enums.infranym_voice_profile import InfranymVoiceProfile


def test_create_valid_composition_minimal():
    surface = InfranymVoiceLayer(text="Surface layer")
    reverse = InfranymVoiceLayer(text="Reverse layer")
    submerged = InfranymVoiceLayer(text="Submerged layer")

    composition = InfranymVoiceComposition(
        surface_layer=surface,
        reverse_layer=reverse,
        submerged_layer=submerged,
        title="Test Composition",
    )

    assert isinstance(composition, InfranymVoiceComposition)
    assert composition.surface_layer == surface
    assert composition.reverse_layer == reverse
    assert composition.submerged_layer == submerged
    assert composition.title == "Test Composition"
    assert composition.tempo_bpm is None
    assert composition.key_signature is None
    assert composition.metadata is None


def test_create_valid_composition_with_all_fields():
    surface = InfranymVoiceLayer(
        text="Surface", voice_profile=InfranymVoiceProfile.PROCLAMATION
    )
    reverse = InfranymVoiceLayer(
        text="Reverse", voice_profile=InfranymVoiceProfile.ROBOTIC, reverse=True
    )
    submerged = InfranymVoiceLayer(
        text="Submerged", voice_profile=InfranymVoiceProfile.WHISPER, volume_db=-6.0
    )

    metadata = {"artist": "Test Artist", "year": 2025}

    composition = InfranymVoiceComposition(
        surface_layer=surface,
        reverse_layer=reverse,
        submerged_layer=submerged,
        title="Complete Composition",
        tempo_bpm=120,
        key_signature="Am",
        metadata=metadata,
    )

    assert composition.title == "Complete Composition"
    assert composition.tempo_bpm == 120
    assert composition.key_signature == "Am"
    assert composition.metadata == metadata
    assert composition.surface_layer.voice_profile == InfranymVoiceProfile.PROCLAMATION
    assert composition.reverse_layer.reverse is True
    assert composition.submerged_layer.volume_db == -6.0


def test_missing_surface_layer():
    reverse = InfranymVoiceLayer(text="Reverse")
    submerged = InfranymVoiceLayer(text="Submerged")

    with pytest.raises(ValidationError) as exc_info:
        InfranymVoiceComposition(
            reverse_layer=reverse, submerged_layer=submerged, title="Missing Surface"
        )
    assert "surface_layer" in str(exc_info.value)


def test_missing_reverse_layer():
    surface = InfranymVoiceLayer(text="Surface")
    submerged = InfranymVoiceLayer(text="Submerged")

    with pytest.raises(ValidationError) as exc_info:
        InfranymVoiceComposition(
            surface_layer=surface, submerged_layer=submerged, title="Missing Reverse"
        )
    assert "reverse_layer" in str(exc_info.value)


def test_missing_submerged_layer():
    surface = InfranymVoiceLayer(text="Surface")
    reverse = InfranymVoiceLayer(text="Reverse")

    with pytest.raises(ValidationError) as exc_info:
        InfranymVoiceComposition(
            surface_layer=surface, reverse_layer=reverse, title="Missing Submerged"
        )
    assert "submerged_layer" in str(exc_info.value)


def test_missing_title():
    surface = InfranymVoiceLayer(text="Surface")
    reverse = InfranymVoiceLayer(text="Reverse")
    submerged = InfranymVoiceLayer(text="Submerged")

    with pytest.raises(ValidationError) as exc_info:
        InfranymVoiceComposition(
            surface_layer=surface, reverse_layer=reverse, submerged_layer=submerged
        )
    assert "title" in str(exc_info.value)


def test_all_fields_missing():
    with pytest.raises(ValidationError) as exc_info:
        InfranymVoiceComposition()
    error_str = str(exc_info.value)
    assert "surface_layer" in error_str
    assert "reverse_layer" in error_str
    assert "submerged_layer" in error_str
    assert "title" in error_str


def test_invalid_layer_type_for_surface():
    reverse = InfranymVoiceLayer(text="Reverse")
    submerged = InfranymVoiceLayer(text="Submerged")

    with pytest.raises(ValidationError):
        InfranymVoiceComposition(
            surface_layer="not a layer",
            reverse_layer=reverse,
            submerged_layer=submerged,
            title="Invalid Surface",
        )


def test_invalid_title_type():
    surface = InfranymVoiceLayer(text="Surface")
    reverse = InfranymVoiceLayer(text="Reverse")
    submerged = InfranymVoiceLayer(text="Submerged")

    with pytest.raises(ValidationError):
        InfranymVoiceComposition(
            surface_layer=surface,
            reverse_layer=reverse,
            submerged_layer=submerged,
            title=123,
        )


def test_tempo_bpm_coercion():
    surface = InfranymVoiceLayer(text="Surface")
    reverse = InfranymVoiceLayer(text="Reverse")
    submerged = InfranymVoiceLayer(text="Submerged")

    # Pydantic coerces string to int
    comp = InfranymVoiceComposition(
        surface_layer=surface,
        reverse_layer=reverse,
        submerged_layer=submerged,
        title="Test",
        tempo_bpm="120",
    )
    assert comp.tempo_bpm == 120
    assert isinstance(comp.tempo_bpm, int)


def test_tempo_bpm_values():
    surface = InfranymVoiceLayer(text="Surface")
    reverse = InfranymVoiceLayer(text="Reverse")
    submerged = InfranymVoiceLayer(text="Submerged")

    comp_slow = InfranymVoiceComposition(
        surface_layer=surface,
        reverse_layer=reverse,
        submerged_layer=submerged,
        title="Slow",
        tempo_bpm=60,
    )
    assert comp_slow.tempo_bpm == 60

    comp_fast = InfranymVoiceComposition(
        surface_layer=surface,
        reverse_layer=reverse,
        submerged_layer=submerged,
        title="Fast",
        tempo_bpm=180,
    )
    assert comp_fast.tempo_bpm == 180


def test_key_signature_values():
    surface = InfranymVoiceLayer(text="Surface")
    reverse = InfranymVoiceLayer(text="Reverse")
    submerged = InfranymVoiceLayer(text="Submerged")

    for key in ["C", "Am", "F#m", "Bb", "G"]:
        comp = InfranymVoiceComposition(
            surface_layer=surface,
            reverse_layer=reverse,
            submerged_layer=submerged,
            title=f"Key {key}",
            key_signature=key,
        )
        assert comp.key_signature == key


def test_metadata_dict():
    surface = InfranymVoiceLayer(text="Surface")
    reverse = InfranymVoiceLayer(text="Reverse")
    submerged = InfranymVoiceLayer(text="Submerged")

    metadata = {
        "artist": "Test Artist",
        "album": "Test Album",
        "year": 2025,
        "tags": ["experimental", "ambient"],
    }

    comp = InfranymVoiceComposition(
        surface_layer=surface,
        reverse_layer=reverse,
        submerged_layer=submerged,
        title="With Metadata",
        metadata=metadata,
    )

    assert comp.metadata == metadata
    assert comp.metadata["artist"] == "Test Artist"
    assert comp.metadata["year"] == 2025


def test_model_dump():
    surface = InfranymVoiceLayer(text="Surface")
    reverse = InfranymVoiceLayer(text="Reverse")
    submerged = InfranymVoiceLayer(text="Submerged")

    comp = InfranymVoiceComposition(
        surface_layer=surface,
        reverse_layer=reverse,
        submerged_layer=submerged,
        title="Test",
        tempo_bpm=120,
    )

    data = comp.model_dump(exclude_none=True, exclude_unset=True)
    assert "surface_layer" in data
    assert "reverse_layer" in data
    assert "submerged_layer" in data
    assert "title" in data
    assert "tempo_bpm" in data


def test_field_descriptions():
    fields = getattr(InfranymVoiceComposition, "model_fields", None)
    assert fields is not None
    assert "surface_layer" in fields
    assert "reverse_layer" in fields
    assert "submerged_layer" in fields
    assert "title" in fields
    assert fields["surface_layer"].description == "Surface layer of the composition"
    assert fields["reverse_layer"].description == "Reverse layer of the composition"
    assert fields["submerged_layer"].description == "Submerged layer of the composition"
    assert fields["title"].description == "Title of the composition"


def test_layers_are_independent():
    surface = InfranymVoiceLayer(
        text="Surface", voice_profile=InfranymVoiceProfile.PROCLAMATION
    )
    reverse = InfranymVoiceLayer(
        text="Reverse", voice_profile=InfranymVoiceProfile.ROBOTIC
    )
    submerged = InfranymVoiceLayer(
        text="Submerged", voice_profile=InfranymVoiceProfile.WHISPER
    )

    comp = InfranymVoiceComposition(
        surface_layer=surface,
        reverse_layer=reverse,
        submerged_layer=submerged,
        title="Independent Layers",
    )

    assert comp.surface_layer.voice_profile != comp.reverse_layer.voice_profile
    assert comp.reverse_layer.voice_profile != comp.submerged_layer.voice_profile
    assert comp.surface_layer.text == "Surface"
    assert comp.reverse_layer.text == "Reverse"
    assert comp.submerged_layer.text == "Submerged"


def test_nested_layer_validation():
    surface = InfranymVoiceLayer(text="Surface")
    reverse = InfranymVoiceLayer(text="Reverse")

    # Invalid submerged layer (empty text)
    with pytest.raises(ValidationError):
        InfranymVoiceComposition(
            surface_layer=surface,
            reverse_layer=reverse,
            submerged_layer=InfranymVoiceLayer(text=""),
            title="Invalid Submerged",
        )


def test_complex_composition():
    surface = InfranymVoiceLayer(
        text="The surface speaks clearly",
        voice_profile=InfranymVoiceProfile.PROCLAMATION,
        rate=150,
        pitch=1.0,
        volume_db=0.0,
        stereo_pan=0.0,
    )

    reverse = InfranymVoiceLayer(
        text="Backward flows the mystery",
        voice_profile=InfranymVoiceProfile.WHISPER,
        rate=120,
        pitch=0.8,
        volume_db=-3.0,
        reverse=True,
        stereo_pan=-0.7,
    )

    submerged = InfranymVoiceLayer(
        text="Deep beneath, ancient voices echo",
        voice_profile=InfranymVoiceProfile.ANCIENT,
        rate=80,
        pitch=0.6,
        volume_db=-6.0,
        stereo_pan=0.5,
        freq_filter=(100, 2000),
    )

    comp = InfranymVoiceComposition(
        surface_layer=surface,
        reverse_layer=reverse,
        submerged_layer=submerged,
        title="Triadic Infranym",
        tempo_bpm=90,
        key_signature="Dm",
        metadata={
            "ritual_type": "invocation",
            "moon_phase": "waning_crescent",
            "duration_seconds": 180,
        },
    )

    assert comp.title == "Triadic Infranym"
    assert comp.surface_layer.rate == 150
    assert comp.reverse_layer.reverse is True
    assert comp.submerged_layer.freq_filter == (100, 2000)
    assert comp.tempo_bpm == 90
    assert comp.key_signature == "Dm"
    assert comp.metadata["ritual_type"] == "invocation"
