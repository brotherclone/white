"""Tests for WhiteAgent auto-chord-kickoff feature."""

import logging

from unittest.mock import MagicMock, patch

from app.agents.white_agent import WhiteAgent
from app.agents.states.white_agent_state import MainAgentState
from app.structures.manifests.song_proposal import SongProposal, SongProposalIteration


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_state(thread_id="test_thread_001", iterations=None):
    if iterations is None:
        iterations = [
            SongProposalIteration(
                iteration_id="red_abc123",
                bpm=120,
                tempo="4/4",
                key="C Major",
                rainbow_color="red",
                title="Test Song",
                mood=["energetic"],
                genres=["rock"],
                concept="This is a substantive philosophical exploration of the interplay between light and sound, examining how chromatic resonance manifests in creative work through rhythmic practice and harmonic representation.",
            )
        ]
    return MainAgentState(
        thread_id=thread_id,
        song_proposals=SongProposal(iterations=iterations),
        artifacts=[],
        ready_for_red=False,
        ready_for_orange=False,
        ready_for_yellow=False,
        ready_for_green=False,
        ready_for_blue=False,
        ready_for_indigo=False,
        ready_for_violet=False,
        ready_for_white=False,
        run_finished=False,
        enabled_agents=[],
    )


# ---------------------------------------------------------------------------
# 1. _invoke_chord_pipeline_safe — error swallowing
# ---------------------------------------------------------------------------


class TestInvokeChordPipelineSafe:

    def test_swallows_system_exit(self, caplog):
        agent = WhiteAgent()
        with patch(
            "app.generators.midi.pipelines.chord_pipeline.run_chord_pipeline",
            side_effect=SystemExit(1),
        ):
            with caplog.at_level(logging.WARNING):
                agent._invoke_chord_pipeline_safe("/tmp/thread", "song.yml")

        assert any(
            "exited" in r.message.lower()
            for r in caplog.records
            if r.levelno == logging.WARNING
        )

    def test_swallows_generic_exception(self, caplog):
        agent = WhiteAgent()
        with patch(
            "app.generators.midi.pipelines.chord_pipeline.run_chord_pipeline",
            side_effect=RuntimeError("boom"),
        ):
            with caplog.at_level(logging.WARNING):
                agent._invoke_chord_pipeline_safe("/tmp/thread", "song.yml")

        assert any(
            "failed" in r.message.lower()
            for r in caplog.records
            if r.levelno == logging.WARNING
        )

    def test_success_logs_info(self, caplog):
        agent = WhiteAgent()
        with patch(
            "app.generators.midi.pipelines.chord_pipeline.run_chord_pipeline",
            return_value=None,
        ):
            with caplog.at_level(logging.INFO):
                agent._invoke_chord_pipeline_safe("/tmp/thread", "song.yml")

        assert any(
            "chord pipeline complete" in r.message.lower()
            for r in caplog.records
            if r.levelno == logging.INFO
        )

    def test_passes_correct_args(self):
        agent = WhiteAgent()
        with patch(
            "app.generators.midi.pipelines.chord_pipeline.run_chord_pipeline",
        ) as mock_run:
            agent._invoke_chord_pipeline_safe("/my/thread", "song_red_abc.yml")

        mock_run.assert_called_once_with(
            thread_dir="/my/thread",
            song_filename="song_red_abc.yml",
            seed=42,
            num_candidates=200,
            top_k=10,
        )


# ---------------------------------------------------------------------------
# 2. finalize_song_proposal — auto chord kickoff integration
# ---------------------------------------------------------------------------


class TestFinalizeWithChordKickoff:

    def _patched_agent(self):
        """WhiteAgent with save_all_proposals and _save_meta_analysis stubbed out."""
        agent = WhiteAgent()
        return agent

    def test_calls_safe_wrapper_when_flag_true(self, monkeypatch):
        monkeypatch.setenv("MOCK_MODE", "false")
        agent = self._patched_agent()
        agent._auto_chord_generation = True
        state = _make_state()

        with (
            patch.object(WhiteAgent, "save_all_proposals"),
            patch.object(WhiteAgent, "_invoke_chord_pipeline_safe") as mock_invoke,
        ):
            agent.finalize_song_proposal(state)

        mock_invoke.assert_called_once()
        call_kwargs = mock_invoke.call_args
        filename_arg = (
            call_kwargs.args[1]
            if call_kwargs.args
            else call_kwargs.kwargs.get("song_filename", "")
        )
        assert filename_arg == "song_proposal_red_red_abc123.yml"

    def test_does_not_call_wrapper_when_flag_false(self):
        agent = self._patched_agent()
        agent._auto_chord_generation = False
        state = _make_state()

        with (
            patch.object(WhiteAgent, "save_all_proposals"),
            patch.object(WhiteAgent, "_invoke_chord_pipeline_safe") as mock_invoke,
        ):
            agent.finalize_song_proposal(state)

        mock_invoke.assert_not_called()

    def test_skips_when_mock_mode(self, monkeypatch):
        monkeypatch.setenv("MOCK_MODE", "true")
        agent = self._patched_agent()
        agent._auto_chord_generation = True
        state = _make_state()

        with (
            patch.object(WhiteAgent, "save_all_proposals"),
            patch.object(WhiteAgent, "_invoke_chord_pipeline_safe") as mock_invoke,
        ):
            agent.finalize_song_proposal(state)

        mock_invoke.assert_not_called()

    def test_skips_when_empty_iterations(self, monkeypatch):
        monkeypatch.setenv("MOCK_MODE", "false")
        agent = self._patched_agent()
        agent._auto_chord_generation = True
        state = _make_state(iterations=[])

        with (
            patch.object(WhiteAgent, "save_all_proposals"),
            patch.object(WhiteAgent, "_invoke_chord_pipeline_safe") as mock_invoke,
        ):
            agent.finalize_song_proposal(state)

        mock_invoke.assert_not_called()

    def test_run_finished_remains_true_when_chord_gen_raises(self, monkeypatch):
        monkeypatch.setenv("MOCK_MODE", "false")
        agent = self._patched_agent()
        agent._auto_chord_generation = True
        state = _make_state()

        with (
            patch.object(WhiteAgent, "save_all_proposals"),
            patch(
                "app.generators.midi.pipelines.chord_pipeline.run_chord_pipeline",
                side_effect=RuntimeError("oops"),
            ),
        ):
            result = agent.finalize_song_proposal(state)

        assert result.run_finished is True


# ---------------------------------------------------------------------------
# 3. start_workflow — flag wiring
# ---------------------------------------------------------------------------


class TestStartWorkflowFlag:

    def _run_workflow(self, agent):
        """Run start_workflow with a stubbed compiled graph."""
        state = _make_state()
        mock_workflow = MagicMock()
        mock_workflow.invoke.return_value = state.__dict__
        with patch.object(WhiteAgent, "build_workflow", return_value=mock_workflow):
            agent.start_workflow(user_input=None)

    def test_env_var_true_sets_flag(self, monkeypatch):
        monkeypatch.setenv("AUTO_CHORD_GENERATION", "true")
        agent = WhiteAgent()
        self._run_workflow(agent)
        assert agent._auto_chord_generation is True

    def test_explicit_false_suppresses_even_with_env(self, monkeypatch):
        monkeypatch.setenv("AUTO_CHORD_GENERATION", "true")
        agent = WhiteAgent()
        state = _make_state()
        mock_workflow = MagicMock()
        mock_workflow.invoke.return_value = state.__dict__
        with patch.object(WhiteAgent, "build_workflow", return_value=mock_workflow):
            agent.start_workflow(user_input=None, auto_chord_generation=False)
        # explicit False wins over env var
        assert agent._auto_chord_generation is False

    def test_explicit_true_sets_flag(self, monkeypatch):
        monkeypatch.delenv("AUTO_CHORD_GENERATION", raising=False)
        agent = WhiteAgent()
        state = _make_state()
        mock_workflow = MagicMock()
        mock_workflow.invoke.return_value = state.__dict__
        with patch.object(WhiteAgent, "build_workflow", return_value=mock_workflow):
            agent.start_workflow(user_input=None, auto_chord_generation=True)
        assert agent._auto_chord_generation is True
