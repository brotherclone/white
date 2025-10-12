import pytest
from app.agents.enums.work_flow_type import WorkflowType


def test_members_and_values():
    expected = {
        "SINGLE_AGENT": "single_agent",
        "CHAIN": "chain",
        "PARALLEL": "parallel",
        "FULL_SPECTRUM": "full_spectrum",
    }
    for name, value in expected.items():
        member = WorkflowType[name]
        assert member.value == value


def test_lookup_by_value():
    assert WorkflowType("chain") is WorkflowType.CHAIN
    assert WorkflowType("single_agent") is WorkflowType.SINGLE_AGENT


def test_invalid_value_raises_value_error():
    with pytest.raises(ValueError):
        WorkflowType("not_a_workflow")

