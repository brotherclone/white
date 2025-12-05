from pydantic import BaseModel, ValidationError
import pytest

from app.structures.concepts.extinction_musical_parameters import (
    ExtinctionMusicalParameters,
)


def test_inheritance():
    assert issubclass(ExtinctionMusicalParameters, BaseModel)


def test_extinction_musical_parameters_required_fields():
    """Test creating ExtinctionMusicalParameters with required fields only."""
    params = ExtinctionMusicalParameters(
        bpm=80,
        key="D minor",
        note_density="sparse → moderate over time",
    )

    assert params.bpm == 80
    assert params.key == "D minor"
    assert params.note_density == "sparse → moderate over time"
    assert params.time_signature == "4/4"
    assert params.silence_ratio == 0.3
    assert params.dissonance_level == "moderate"
    assert params.harmonic_progression is None
    assert params.primary_instruments == []
    assert params.texture_description == "sparse, with long silences between phrases"
    assert params.structure == "parallel_duet"
    assert params.movement_count == 1
    assert params.ecological_sound_source is None
    assert params.human_sound_source is None


def test_extinction_musical_parameters_all_fields():
    """Test creating ExtinctionMusicalParameters with all fields populated."""
    params = ExtinctionMusicalParameters(
        bpm=60,
        key="A atonal",
        time_signature="5/4",
        note_density="sparse → dense",
        silence_ratio=0.5,
        dissonance_level="dissonant",
        harmonic_progression="descending chromatic",
        primary_instruments=["cello", "contrabass", "prepared piano"],
        texture_description="layered drones with punctuated silence",
        structure="interleaved_narrative",
        movement_count=3,
        ecological_sound_source="whale song recordings",
        human_sound_source="breathing, heartbeat",
    )

    assert params.bpm == 60
    assert params.key == "A atonal"
    assert params.time_signature == "5/4"
    assert params.silence_ratio == 0.5
    assert params.dissonance_level == "dissonant"
    assert params.harmonic_progression == "descending chromatic"
    assert len(params.primary_instruments) == 3
    assert params.ecological_sound_source == "whale song recordings"
    assert params.human_sound_source == "breathing, heartbeat"


def test_bpm_validation():
    """Test BPM validation constraints (40-200)."""
    # Valid BPM values
    params_low = ExtinctionMusicalParameters(bpm=40, key="C", note_density="sparse")
    assert params_low.bpm == 40

    params_high = ExtinctionMusicalParameters(bpm=200, key="C", note_density="dense")
    assert params_high.bpm == 200

    # Invalid BPM: too low
    with pytest.raises(ValidationError):
        ExtinctionMusicalParameters(bpm=39, key="C", note_density="sparse")

    # Invalid BPM: too high
    with pytest.raises(ValidationError):
        ExtinctionMusicalParameters(bpm=201, key="C", note_density="dense")


def test_silence_ratio_validation():
    """Test silence_ratio validation constraints (0.0-0.8)."""
    # Valid silence ratios
    params_min = ExtinctionMusicalParameters(
        bpm=80, key="D", note_density="dense", silence_ratio=0.0
    )
    assert params_min.silence_ratio == 0.0

    params_max = ExtinctionMusicalParameters(
        bpm=80, key="D", note_density="sparse", silence_ratio=0.8
    )
    assert params_max.silence_ratio == 0.8

    # Invalid: too low
    with pytest.raises(ValidationError):
        ExtinctionMusicalParameters(
            bpm=80, key="D", note_density="sparse", silence_ratio=-0.1
        )

    # Invalid: too high
    with pytest.raises(ValidationError):
        ExtinctionMusicalParameters(
            bpm=80, key="D", note_density="sparse", silence_ratio=0.81
        )


def test_dissonance_level_literal():
    """Test that dissonance_level only accepts valid literal values."""
    # Valid values
    for level in ["consonant", "moderate", "dissonant", "microtonal"]:
        params = ExtinctionMusicalParameters(
            bpm=80, key="E", note_density="moderate", dissonance_level=level
        )
        assert params.dissonance_level == level

    # Invalid value
    with pytest.raises(ValidationError):
        ExtinctionMusicalParameters(
            bpm=80, key="E", note_density="moderate", dissonance_level="invalid"
        )


def test_to_artifact_dict():
    """Test to_artifact_dict method returns model_dump."""
    params = ExtinctionMusicalParameters(
        bpm=90,
        key="F# minor",
        note_density="moderate",
        silence_ratio=0.4,
        dissonance_level="moderate",
        primary_instruments=["violin", "electronics"],
        ecological_sound_source="forest ambience",
    )

    artifact_dict = params.to_artifact_dict()

    assert artifact_dict["bpm"] == 90
    assert artifact_dict["key"] == "F# minor"
    assert artifact_dict["note_density"] == "moderate"
    assert artifact_dict["silence_ratio"] == 0.4
    assert artifact_dict["dissonance_level"] == "moderate"
    assert artifact_dict["primary_instruments"] == ["violin", "electronics"]
    assert artifact_dict["ecological_sound_source"] == "forest ambience"


def test_default_factory_independence():
    """Test that default factory creates independent lists for each instance."""
    params1 = ExtinctionMusicalParameters(bpm=70, key="G", note_density="sparse")
    params2 = ExtinctionMusicalParameters(bpm=110, key="B", note_density="dense")

    # Modify params1's list
    params1.primary_instruments.append("cello")

    # Verify params2 is not affected
    assert len(params2.primary_instruments) == 0
