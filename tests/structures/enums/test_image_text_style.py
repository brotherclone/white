import enum
import pytest

from app.structures.enums.image_text_style import ImageTextStyle

EXPECTED = {
    "CLEAN": "clean",
    "DEFAULT": "default",
    "GLITCH": "glitch",
    "STATIC": "static",
}


def test_members_and_values():
    assert set(ImageTextStyle.__members__.keys()) == set(EXPECTED.keys())
    for name, expected_value in EXPECTED.items():
        member = getattr(ImageTextStyle, name)
        assert member.value == expected_value
        assert isinstance(member.value, str)


def test_members_are_str_and_enum_and_compare_to_value():
    for member in ImageTextStyle:
        assert isinstance(member, ImageTextStyle)
        assert isinstance(member, str)
        assert isinstance(member, enum.Enum)
        assert member == member.value


@pytest.mark.parametrize(
    "value,member",
    [
        ("default", ImageTextStyle.DEFAULT),
        ("glitch", ImageTextStyle.GLITCH),
        ("static", ImageTextStyle.STATIC),
        ("clean", ImageTextStyle.CLEAN),
    ],
)
def test_lookup_by_value(value, member):
    assert ImageTextStyle(value) is member


def test_lookup_by_name():
    assert ImageTextStyle["GLITCH"] is ImageTextStyle.GLITCH
    assert ImageTextStyle["DEFAULT"] is ImageTextStyle.DEFAULT


def test_invalid_value_raises_value_error():
    with pytest.raises(ValueError):
        ImageTextStyle("comic sans")


def test_values_are_unique():
    values = [m.value for m in ImageTextStyle]
    assert len(values) == len(set(values))


def test_enum_members_are_enum_instances():
    assert isinstance(ImageTextStyle.STATIC, enum.Enum)
    assert isinstance(ImageTextStyle.CLEAN, enum.Enum)
