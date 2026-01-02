import enum
import pytest

from app.structures.enums.infranym_voice_profile import InfranymVoiceProfile

EXPECTED = {
    "ROBOTIC": "robotic",
    "WHISPER": "whisper",
    "PROCLAMATION": "proclaim",
    "DISTORTED": "distorted",
    "ANCIENT": "ancient",
}


def test_members_and_values():
    assert set(InfranymVoiceProfile.__members__.keys()) == set(EXPECTED.keys())
    for name, expected_value in EXPECTED.items():
        member = getattr(InfranymVoiceProfile, name)
        assert member.value == expected_value
        assert isinstance(member.value, str)


def test_members_are_str_and_enum_and_compare_to_value():
    for member in InfranymVoiceProfile:
        assert isinstance(member, InfranymVoiceProfile)
        assert isinstance(member, str)
        assert isinstance(member, enum.Enum)
        assert member == member.value


@pytest.mark.parametrize(
    "value,member",
    [
        ("robotic", InfranymVoiceProfile.ROBOTIC),
        ("whisper", InfranymVoiceProfile.WHISPER),
        ("proclaim", InfranymVoiceProfile.PROCLAMATION),
        ("distorted", InfranymVoiceProfile.DISTORTED),
        ("ancient", InfranymVoiceProfile.ANCIENT),
    ],
)
def test_lookup_by_value(value, member):
    assert InfranymVoiceProfile(value) is member


def test_lookup_by_name():
    assert InfranymVoiceProfile["ROBOTIC"] is InfranymVoiceProfile.ROBOTIC
    assert InfranymVoiceProfile["DISTORTED"] is InfranymVoiceProfile.DISTORTED


def test_invalid_value_raises_value_error():
    with pytest.raises(ValueError):
        InfranymVoiceProfile("snaggle puss")


def test_values_are_unique():
    values = [m.value for m in InfranymVoiceProfile]
    assert len(values) == len(set(values))


def test_enum_members_are_enum_instances():
    assert isinstance(InfranymVoiceProfile.PROCLAMATION, enum.Enum)
    assert isinstance(InfranymVoiceProfile.ROBOTIC, enum.Enum)
