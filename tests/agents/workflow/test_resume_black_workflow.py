import types
import pytest

from app.agents.workflow import resume_black_workflow as rbw
from app.structures.enums.sigil_state import SigilState
from app.agents.states.black_agent_state import BlackAgentState


def test_check_todoist_tasks_complete_all_complete(monkeypatch):
    class FakeAPI:
        def get_task(self, task_id):
            return types.SimpleNamespace(is_completed=True)

    monkeypatch.setattr(rbw, "get_api_client", lambda: FakeAPI())

    tasks = [{"task_id": "1"}, {"task_id": "2"}]
    assert rbw.check_todoist_tasks_complete(tasks) is True


def test_check_todoist_tasks_complete_not_complete(monkeypatch):
    class FakeAPI:
        def get_task(self, task_id):
            return types.SimpleNamespace(is_completed=False)

    monkeypatch.setattr(rbw, "get_api_client", lambda: FakeAPI())

    tasks = [{"task_id": "1"}]
    assert rbw.check_todoist_tasks_complete(tasks) is False


def test_check_todoist_tasks_complete_error(monkeypatch):
    class FakeAPI:
        def get_task(self, task_id):
            raise RuntimeError("api error")

    monkeypatch.setattr(rbw, "get_api_client", lambda: FakeAPI())

    tasks = [{"task_id": "1"}]
    assert rbw.check_todoist_tasks_complete(tasks) is False


def test_update_sigil_state_to_charged():
    class Artifact:
        def __init__(self, _type, activation_state, wish=None):
            self.type = _type
            self.activation_state = activation_state
            self.wish = wish

    artifacts = [
        Artifact("sigil", SigilState.CREATED, wish="wish-1"),
        Artifact("other", None),
    ]

    state = BlackAgentState(artifacts=artifacts)
    updated = rbw.update_sigil_state_to_charged(state)

    # sigil artifact should be CHARGED
    assert updated.artifacts[0].activation_state == SigilState.CHARGED
    # non-sigil should be unchanged
    assert updated.artifacts[1].type == "other"


def test_resume_black_agent_workflow_success(monkeypatch):
    # initial state: no pending tasks, empty artifacts
    initial_values = {"pending_human_tasks": [], "artifacts": []}
    final_values = {"counter_proposal": {"title": "Done"}}

    class FakeCompiledWorkflow:
        def __init__(self, initial, final):
            self._initial = initial
            self._final = final
            self._calls = 0

        def get_state(self, config):
            self._calls += 1
            if self._calls == 1:
                return types.SimpleNamespace(values=self._initial, next=True)
            return types.SimpleNamespace(values=self._final, next=None)

        def invoke(self, arg, config=None):
            return None

    fake_compiled = FakeCompiledWorkflow(initial_values, final_values)
    fake_agent = types.SimpleNamespace(_compiled_workflow=fake_compiled)

    monkeypatch.setattr(rbw, "BlackAgent", lambda: fake_agent)
    # Let sigil update be no-op for this test (it only needs to accept the BlackAgentState)
    monkeypatch.setattr(rbw, "update_sigil_state_to_charged", lambda s: s)

    black_config = {"configurable": {"thread_id": "t1"}}
    result = rbw.resume_black_agent_workflow(black_config, verify_tasks=True)

    assert result.get("counter_proposal", {}).get("title") == "Done"


def test_resume_black_agent_workflow_fails_when_tasks_incomplete(monkeypatch):
    initial_values = {"pending_human_tasks": [{"task_id": "1"}], "artifacts": []}

    class FakeCompiledWorkflow:
        def __init__(self, initial):
            self._initial = initial

        def get_state(self, config):
            return types.SimpleNamespace(values=self._initial, next=True)

        def invoke(self, arg, config=None):
            return None

    fake_compiled = FakeCompiledWorkflow(initial_values)
    fake_agent = types.SimpleNamespace(_compiled_workflow=fake_compiled)

    monkeypatch.setattr(rbw, "BlackAgent", lambda: fake_agent)
    # Simulate not all Todoist tasks complete
    monkeypatch.setattr(rbw, "check_todoist_tasks_complete", lambda tasks: False)
    monkeypatch.setattr(rbw, "update_sigil_state_to_charged", lambda s: s)

    black_config = {"configurable": {"thread_id": "t1"}}
    with pytest.raises(ValueError):
        rbw.resume_black_agent_workflow(black_config, verify_tasks=True)