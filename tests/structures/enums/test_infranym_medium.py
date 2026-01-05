"""Tests for InfranymMedium enum."""

from app.structures.enums.infranym_medium import InfranymMedium


def test_infranym_medium_values():
    """Test that InfranymMedium enum has correct values"""
    assert InfranymMedium.MIDI == "midi"
    assert InfranymMedium.AUDIO == "audio"
    assert InfranymMedium.TEXT == "text"
    assert InfranymMedium.IMAGE == "image"


def test_infranym_medium_members():
    """Test that all expected members exist"""
    expected_members = {"MIDI", "AUDIO", "TEXT", "IMAGE"}
    actual_members = {member.name for member in InfranymMedium}
    assert actual_members == expected_members


def test_infranym_medium_from_value():
    """Test creating enum from value"""
    assert InfranymMedium("midi") == InfranymMedium.MIDI
    assert InfranymMedium("audio") == InfranymMedium.AUDIO
    assert InfranymMedium("text") == InfranymMedium.TEXT
    assert InfranymMedium("image") == InfranymMedium.IMAGE


def test_infranym_medium_string_representation():
    """Test string representation"""
    # str() gives the full enum name, .value gives the value
    assert str(InfranymMedium.MIDI) == "InfranymMedium.MIDI"
    assert InfranymMedium.AUDIO.value == "audio"


def test_infranym_medium_is_string_enum():
    """Test that InfranymMedium values are strings"""
    assert isinstance(InfranymMedium.MIDI.value, str)
    assert isinstance(InfranymMedium.AUDIO.value, str)


def test_infranym_medium_iteration():
    """Test iterating over enum members"""
    mediums = list(InfranymMedium)
    assert len(mediums) == 4
    assert InfranymMedium.MIDI in mediums
    assert InfranymMedium.AUDIO in mediums
    assert InfranymMedium.TEXT in mediums
    assert InfranymMedium.IMAGE in mediums


def test_infranym_medium_equality():
    """Test enum equality"""
    assert InfranymMedium.MIDI == InfranymMedium.MIDI
    assert InfranymMedium.MIDI != InfranymMedium.AUDIO
