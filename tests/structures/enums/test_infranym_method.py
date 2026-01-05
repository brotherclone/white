import enum
import pytest


from app.structures.enums.infranym_method import InfranymMethod

EXPECTED = {
    "NOTE_CIPHER": "note_cipher",
    "MORSE_DURATION": "morse_duration",
    "BACKMASK_WHISPER": "backmask_whisper",
    "STENOGRAPH_SPECTROGRAM": "stenograph_spectrogram",
    "RIDDLE_POEM": "riddle_poem",
    "ACROSTIC_LYRICS": "acrostic_lyrics",
    "LSB_STEGANOGRAPHY": "lsb_steganography",
    "ANTI_SIGIL": "anti_sigil",
    "DEFAULT": "default",
}


def test_members_and_values():
    assert set(InfranymMethod.__members__.keys()) == set(EXPECTED.keys())
    for name, expected_value in EXPECTED.items():
        member = getattr(InfranymMethod, name)
        assert member.value == expected_value
        assert isinstance(member.value, str)


def test_members_are_str_and_enum_and_compare_to_value():
    for member in InfranymMethod:
        assert isinstance(member, InfranymMethod)
        assert isinstance(member, str)
        assert isinstance(member, enum.Enum)
        assert member == member.value


@pytest.mark.parametrize(
    "value,member",
    [
        ("note_cipher", InfranymMethod.NOTE_CIPHER),
        ("morse_duration", InfranymMethod.MORSE_DURATION),
        ("backmask_whisper", InfranymMethod.BACKMASK_WHISPER),
        ("stenograph_spectrogram", InfranymMethod.STENOGRAPH_SPECTROGRAM),
        ("riddle_poem", InfranymMethod.RIDDLE_POEM),
        ("acrostic_lyrics", InfranymMethod.ACROSTIC_LYRICS),
        ("lsb_steganography", InfranymMethod.LSB_STEGANOGRAPHY),
        ("anti_sigil", InfranymMethod.ANTI_SIGIL),
        ("default", InfranymMethod.DEFAULT),
    ],
)
def test_lookup_by_value(value, member):
    assert InfranymMethod(value) is member


def test_lookup_by_name():
    assert InfranymMethod["ANTI_SIGIL"] is InfranymMethod.ANTI_SIGIL
    assert InfranymMethod["DEFAULT"] is InfranymMethod.DEFAULT


def test_invalid_value_raises_value_error():
    with pytest.raises(ValueError):
        InfranymMethod("dancing")


def test_values_are_unique():
    values = [m.value for m in InfranymMethod]
    assert len(values) == len(set(values))


def test_enum_members_are_enum_instances():
    assert isinstance(InfranymMethod.RIDDLE_POEM, enum.Enum)
    assert isinstance(InfranymMethod.BACKMASK_WHISPER, enum.Enum)
