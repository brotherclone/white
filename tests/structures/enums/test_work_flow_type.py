import enum

import pytest

from app.structures.enums.work_flow_type import WorkflowType

EXPECTED = {
    "SINGLE_AGENT": "single_agent",
    "CHAIN": "chain",
    "PARALLEL": "parallel",
    "FULL_SPECTRUM": "full_spectrum",
}


def test_members_and_values():
    assert set(WorkflowType.__members__.keys()) == set(EXPECTED.keys())
    for name, expected_value in EXPECTED.items():
        member = getattr(WorkflowType, name)
        assert member.value == expected_value
        assert isinstance(member.value, str)


def test_members_are_str_and_enum_and_compare_to_value():
    for member in WorkflowType:
        assert isinstance(member, WorkflowType)
        assert isinstance(member, str)
        assert isinstance(member, enum.Enum)
        assert member == member.value


@pytest.mark.parametrize(
    "value,member",
    [
        ("single_agent", WorkflowType.SINGLE_AGENT),
        ("chain", WorkflowType.CHAIN),
        ("parallel", WorkflowType.PARALLEL),
        ("full_spectrum", WorkflowType.FULL_SPECTRUM),
    ],
)
def test_lookup_by_value(value, member):
    assert WorkflowType(value) is member


def test_lookup_by_name():
    assert WorkflowType["SINGLE_AGENT"] is WorkflowType.SINGLE_AGENT
    assert WorkflowType["CHAIN"] is WorkflowType.CHAIN


def test_invalid_value_raises_value_error():
    with pytest.raises(ValueError):
        WorkflowType("unknown")


def test_values_are_unique():
    values = [m.value for m in WorkflowType]
    assert len(values) == len(set(values))


def test_enum_members_are_enum_instances():
    assert isinstance(WorkflowType.SINGLE_AGENT, enum.Enum)
    assert isinstance(WorkflowType.FULL_SPECTRUM, enum.Enum)
