import os
import pytest

from app.agents.tools.speech_tools import (
    transcription_from_speech_to_text,
    evp_speech_to_text,
)
from app.structures.concepts.rainbow_table_color import the_rainbow_table_colors


def test_evp_returns_none_when_no_api_key(monkeypatch):
    monkeypatch.delenv("ASSEMBLYAI_API_KEY", raising=False)
    res = evp_speech_to_text("/tmp", "audio.wav")
    assert res is None


def test_evp_returns_none_on_transcription_error(monkeypatch):
    """Test that transcription errors return None instead of raising"""
    monkeypatch.setenv("ASSEMBLYAI_API_KEY", "testkey")
    monkeypatch.setattr("os.path.exists", lambda x: True)

    class FakeTranscription:
        status = "error"
        error = "boom error"

    class FakeTranscriber:
        def __init__(self, config):
            pass

        def transcribe(self, path):
            return FakeTranscription()

    monkeypatch.setattr(
        "app.agents.tools.speech_tools.aai.Transcriber", FakeTranscriber
    )
    result = evp_speech_to_text("/tmp", "audio.wav")
    assert result is None


def test_evp_returns_text_when_completed(monkeypatch):
    monkeypatch.setenv("ASSEMBLYAI_API_KEY", "testkey")
    monkeypatch.setattr("os.path.exists", lambda x: True)

    class FakeTranscription:
        status = "completed"
        text = "Hello world"
        confidence = 0.9
        sentiment_analysis_results = None
        entities = None
        summary = None
        utterances = None

    class FakeTranscriber:
        def __init__(self, config):
            pass

        def transcribe(self, path):
            return FakeTranscription()

    monkeypatch.setattr(
        "app.agents.tools.speech_tools.aai.Transcriber", FakeTranscriber
    )
    out = evp_speech_to_text("/tmp", "audio.wav")
    assert out == "Hello world"


def test_evp_uses_utterances_when_no_text(monkeypatch):
    monkeypatch.setenv("ASSEMBLYAI_API_KEY", "testkey")
    monkeypatch.setattr("os.path.exists", lambda x: True)

    class FakeUtterance:
        def __init__(self, text, confidence=0.8):
            self.text = text
            self.confidence = confidence

    class FakeTranscription:
        status = "completed"
        text = ""  # empty
        confidence = 0.0
        sentiment_analysis_results = None
        entities = None
        summary = None
        utterances = [FakeUtterance("first"), FakeUtterance("second")]

    class FakeTranscriber:
        def __init__(self, config):
            pass

        def transcribe(self, path):
            return FakeTranscription()

    monkeypatch.setattr(
        "app.agents.tools.speech_tools.aai.Transcriber", FakeTranscriber
    )
    out = evp_speech_to_text("/tmp", "audio.wav")
    assert out == "first second"


@pytest.mark.skipif(
    os.environ.get("BLOCK_MODE", "").lower() in {"1", "true", "yes"},
    reason="Skipping test because BLOCK_MODE is enabled",
)
def test_chain_artifact_file_from_speech_to_text_creates_text_file(monkeypatch):
    """Test that speech-to-text returns the transcript string"""
    monkeypatch.setattr(
        "app.agents.tools.speech_tools.evp_speech_to_text",
        lambda wp, fn: "transcript body",
    )

    class FakeAudio:
        file_name = "audio.wav"
        rainbow_color = the_rainbow_table_colors["Z"]
        base_path = "/tmp/base"
        artifact_name = "artifactX"

        def get_artifact_path(self, with_file_name=False):
            return "/tmp/base" if not with_file_name else "/tmp/base/audio.wav"

    audio = FakeAudio()
    txt = transcription_from_speech_to_text(audio)
    assert isinstance(txt, str)
    assert txt == "transcript body"


@pytest.mark.skipif(
    os.environ.get("BLOCK_MODE", "").lower() in {"1", "true", "yes"},
    reason="Skipping test because BLOCK_MODE is enabled",
)
def test_chain_artifact_file_from_speech_to_text_returns_placeholder_when_no_transcript(
    monkeypatch,
):
    """Test that None transcript results in placeholder text, not None"""
    monkeypatch.setattr(
        "app.agents.tools.speech_tools.evp_speech_to_text", lambda wp, fn: None
    )

    class FakeAudio:
        file_name = "audio.wav"
        rainbow_color = the_rainbow_table_colors["Z"]
        base_path = "/tmp/base"
        artifact_name = "artifactX"

        def get_artifact_path(self, with_file_name=False):
            return "/tmp/base" if not with_file_name else "/tmp/base/audio.wav"

    audio = FakeAudio()
    out = transcription_from_speech_to_text(audio)
    assert out is not None
    assert isinstance(out, str)
    assert out == "[EVP: No discernible speech detected]"
