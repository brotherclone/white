"""Tests for WhiteAgent auto-chord-kickoff feature."""

import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

from white_core.manifests.song_proposal import SongProposal, SongProposalIteration

from app.agents.states.white_agent_state import MainAgentState
from app.agents.white_agent import WhiteAgent, _is_white_proposal

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
                is_final=True,
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
            patch("app.generators.midi.production.init_production.init_production"),
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
            patch("app.generators.midi.production.init_production.init_production"),
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


# ---------------------------------------------------------------------------
# 4. Multi-final-iteration kickoff
# ---------------------------------------------------------------------------


def _make_iteration(color, iteration_id=None, is_final=True):
    if iteration_id is None:
        iteration_id = f"{color}_001"
    return SongProposalIteration(
        iteration_id=iteration_id,
        bpm=120,
        tempo="4/4",
        key="C Major",
        rainbow_color=color,
        title=f"{color} Song",
        mood=["contemplative"],
        genres=["experimental"],
        concept="A substantive philosophical concept exploring chromatic resonance and temporal displacement through iterative compositional cycles and harmonic convergence.",
        is_final=is_final,
    )


class TestMultiFinalKickoff:

    def _patched_finalize(self, agent, state, mock_invoke, mock_init):
        with (
            patch.object(WhiteAgent, "save_all_proposals"),
            patch.object(WhiteAgent, "_invoke_chord_pipeline_safe", mock_invoke),
            patch(
                "app.generators.midi.production.init_production.init_production",
                mock_init,
            ),
        ):
            return agent.finalize_song_proposal(state)

    def test_chord_pipeline_called_for_each_final_iteration(self, monkeypatch):
        """init and chord gen run once per is_final=True iteration."""
        monkeypatch.setenv("MOCK_MODE", "false")
        agent = WhiteAgent()
        agent._auto_chord_generation = True

        state = _make_state(
            iterations=[
                _make_iteration("red", "red_001", is_final=True),
                _make_iteration("white", "white_001", is_final=True),
            ]
        )

        mock_invoke = MagicMock()
        mock_init = MagicMock(return_value=Path("/tmp/song_context.yml"))

        self._patched_finalize(agent, state, mock_invoke, mock_init)

        assert mock_invoke.call_count == 2
        assert mock_init.call_count == 2

    def test_white_is_processed_last(self, monkeypatch):
        """White iteration is always passed to chord pipeline last."""
        monkeypatch.setenv("MOCK_MODE", "false")
        agent = WhiteAgent()
        agent._auto_chord_generation = True

        state = _make_state(
            iterations=[
                _make_iteration("white", "white_001", is_final=True),
                _make_iteration("red", "red_001", is_final=True),
            ]
        )

        call_order = []
        mock_invoke = MagicMock(side_effect=lambda td, fn: call_order.append(fn))
        mock_init = MagicMock(return_value=Path("/tmp/song_context.yml"))

        self._patched_finalize(agent, state, mock_invoke, mock_init)

        assert len(call_order) == 2
        assert "white" in call_order[-1].lower()

    def test_non_final_iterations_skipped(self, monkeypatch):
        """Iterations with is_final=False are not kicked off."""
        monkeypatch.setenv("MOCK_MODE", "false")
        agent = WhiteAgent()
        agent._auto_chord_generation = True

        state = _make_state(
            iterations=[
                _make_iteration("red", "red_001", is_final=False),
                _make_iteration("red", "red_002", is_final=True),
            ]
        )

        mock_invoke = MagicMock()
        mock_init = MagicMock(return_value=Path("/tmp/song_context.yml"))

        self._patched_finalize(agent, state, mock_invoke, mock_init)

        assert mock_invoke.call_count == 1
        assert "red_002" in mock_invoke.call_args.args[1]


# ---------------------------------------------------------------------------
# 5. _launch_review_browser
# ---------------------------------------------------------------------------


class TestLaunchReviewBrowser:

    def test_no_subprocess_when_ports_open(self):
        """When both ports are already listening, no Popen calls are made."""
        agent = WhiteAgent()
        dirs = [Path("/tmp/prod/red_001")]

        with (
            patch(
                "app.agents.white_agent.WhiteAgent._launch_review_browser",
                wraps=agent._launch_review_browser,
            ),
            patch("socket.socket") as mock_socket_cls,
            patch("subprocess.Popen") as mock_popen,
            patch("webbrowser.open") as mock_browser,
            patch("time.sleep"),
        ):
            mock_sock = MagicMock()
            mock_sock.__enter__ = lambda s: s
            mock_sock.__exit__ = MagicMock(return_value=False)
            mock_sock.connect_ex.return_value = 0  # both ports open
            mock_socket_cls.return_value = mock_sock

            agent._launch_review_browser(dirs)

        mock_popen.assert_not_called()
        mock_browser.assert_called_once()
        url = mock_browser.call_args.args[0]
        assert "production-dir=" in url
        assert "phase=chords" in url

    def test_url_encodes_production_dir(self):
        """The production-dir query param is URL-encoded (spaces/slashes safe)."""
        agent = WhiteAgent()
        dirs = [Path("/tmp/prod/my song")]

        with (
            patch("socket.socket") as mock_socket_cls,
            patch("subprocess.Popen"),
            patch("webbrowser.open") as mock_browser,
            patch("time.sleep"),
        ):
            mock_sock = MagicMock()
            mock_sock.__enter__ = lambda s: s
            mock_sock.__exit__ = MagicMock(return_value=False)
            mock_sock.connect_ex.return_value = 0
            mock_socket_cls.return_value = mock_sock

            agent._launch_review_browser(dirs)

        url = mock_browser.call_args.args[0]
        assert "my%20song" in url

    def test_empty_dirs_skips_launch(self, caplog):
        """An empty dirs list logs a warning and returns without calling Popen."""
        agent = WhiteAgent()
        with (
            patch("subprocess.Popen") as mock_popen,
            patch("webbrowser.open") as mock_browser,
            caplog.at_level(logging.WARNING),
        ):
            agent._launch_review_browser([])

        mock_popen.assert_not_called()
        mock_browser.assert_not_called()
        assert any(
            "no production directories" in r.message.lower() for r in caplog.records
        )

    def test_swallows_launch_exception(self, caplog):
        """If Popen raises, a warning is logged and no exception propagates."""
        agent = WhiteAgent()
        dirs = [Path("/tmp/prod/red_001")]

        with (
            patch("socket.socket") as mock_socket_cls,
            patch(
                "subprocess.Popen", side_effect=FileNotFoundError("python not found")
            ),
            patch("webbrowser.open"),
            caplog.at_level(logging.WARNING),
        ):
            mock_sock = MagicMock()
            mock_sock.__enter__ = lambda s: s
            mock_sock.__exit__ = MagicMock(return_value=False)
            mock_sock.connect_ex.return_value = 1  # port closed → triggers Popen
            mock_socket_cls.return_value = mock_sock

            agent._launch_review_browser(dirs)  # must not raise

        assert any("auto-launch failed" in r.message.lower() for r in caplog.records)

    def test_subprocess_launched_when_port_8000_closed(self):
        """candidate_server Popen is called when port 8000 is not listening."""
        agent = WhiteAgent()
        dirs = [Path("/tmp/prod/red_001")]

        def port_side_effect(*args, **kwargs):
            return 0  # always "open" for simplicity after first check

        with (
            patch("socket.socket") as mock_socket_cls,
            patch("subprocess.Popen") as mock_popen,
            patch("webbrowser.open"),
            patch("time.sleep"),
        ):
            # First connect_ex call (port 8000 check) returns 1 (closed),
            # subsequent calls return 0
            mock_sock = MagicMock()
            mock_sock.__enter__ = lambda s: s
            mock_sock.__exit__ = MagicMock(return_value=False)
            mock_sock.connect_ex.side_effect = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            mock_socket_cls.return_value = mock_sock

            agent._launch_review_browser(dirs)

        assert mock_popen.call_count >= 1
        launched_cmds = [c.args[0] for c in mock_popen.call_args_list]
        assert any("candidate_server" in str(cmd) for cmd in launched_cmds)


# ---------------------------------------------------------------------------
# 6. _is_white_proposal helper
# ---------------------------------------------------------------------------


class TestIsWhiteProposal:

    def test_string_white(self):
        it = _make_iteration("white", "white_001")
        assert _is_white_proposal(it) is True

    def test_string_red(self):
        it = _make_iteration("red", "red_001")
        assert _is_white_proposal(it) is False

    def test_rainbow_table_color_white(self):
        from white_core.concepts.rainbow_table_color import the_rainbow_table_colors

        it = _make_iteration("white", "white_001")
        it.rainbow_color = the_rainbow_table_colors["A"]
        assert _is_white_proposal(it) is True

    def test_rainbow_table_color_red(self):
        from white_core.concepts.rainbow_table_color import the_rainbow_table_colors

        it = _make_iteration("red", "red_001")
        it.rainbow_color = the_rainbow_table_colors["R"]
        assert _is_white_proposal(it) is False
