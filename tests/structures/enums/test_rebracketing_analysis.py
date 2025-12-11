import enum
import pytest

from app.structures.enums.rebracketing_analysis_type import RebracketingAnalysisType

EXPECTED = {
    "CAUSAL": "causal",
    "SPATIAL": "spatial",
    "PERCEPTUAL": "perceptual",
    "EXPERIENTIAL": "experiential",
    "TEMPORAL": "temporal",
    "BOUNDARY": "boundary",
    "NONE": "none",
}


def test_members_and_values():
    assert set(RebracketingAnalysisType.__members__.keys()) == set(EXPECTED.keys())
    for name, expected_value in EXPECTED.items():
        member = getattr(RebracketingAnalysisType, name)
        assert member.value == expected_value
        assert isinstance(member.value, str)


def test_members_are_str_and_enum_and_compare_to_value():
    for member in RebracketingAnalysisType:
        assert isinstance(member, RebracketingAnalysisType)
        assert isinstance(member, str)
        assert isinstance(member, enum.Enum)
        assert member == member.value


@pytest.mark.parametrize(
    "value,member",
    [
        ("causal", RebracketingAnalysisType.CAUSAL),
        ("spatial", RebracketingAnalysisType.SPATIAL),
        ("perceptual", RebracketingAnalysisType.PERCEPTUAL),
        ("experiential", RebracketingAnalysisType.EXPERIENTIAL),
        ("temporal", RebracketingAnalysisType.TEMPORAL),
        ("boundary", RebracketingAnalysisType.BOUNDARY),
        ("none", RebracketingAnalysisType.NONE),
    ],
)
def test_lookup_by_value(value, member):
    assert RebracketingAnalysisType(value) is member


def test_lookup_by_name():
    assert RebracketingAnalysisType["BOUNDARY"] is RebracketingAnalysisType.BOUNDARY
    assert (
        RebracketingAnalysisType["EXPERIENTIAL"]
        is RebracketingAnalysisType.EXPERIENTIAL
    )


def test_invalid_value_raises_value_error():
    with pytest.raises(ValueError):
        RebracketingAnalysisType("pronounced")


def test_values_are_unique():
    values = [m.value for m in RebracketingAnalysisType]
    assert len(values) == len(set(values))


def test_enum_members_are_enum_instances():
    assert isinstance(RebracketingAnalysisType.CAUSAL, enum.Enum)
    assert isinstance(RebracketingAnalysisType.NONE, enum.Enum)
