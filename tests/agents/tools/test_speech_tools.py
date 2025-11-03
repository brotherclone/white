from unittest.mock import patch

from app.agents.tools.speech_tools import evp_speech_to_text, chain_artifact_file_from_speech_to_text
from app.structures.concepts.rainbow_table_color import the_rainbow_table_colors
from app.structures.enums.chain_artifact_file_type import ChainArtifactFileType
from app.structures.artifacts.text_chain_artifact_file import TextChainArtifactFile


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

    monkeypatch.setattr("app.agents.tools.speech_tools.aai.Transcriber", FakeTranscriber)
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

    monkeypatch.setattr("app.agents.tools.speech_tools.aai.Transcriber", FakeTranscriber)
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

    monkeypatch.setattr("app.agents.tools.speech_tools.aai.Transcriber", FakeTranscriber)
    out = evp_speech_to_text("/tmp", "audio.wav")
    assert out == "first second"


@patch("app.agents.tools.speech_tools.save_artifact_file_to_md")  # ✅ ADD: Mock the save function
def test_chain_artifact_file_from_speech_to_text_creates_text_file(mock_save, monkeypatch):
    monkeypatch.setattr("app.agents.tools.speech_tools.evp_speech_to_text", lambda wp, fn: "transcript body")

    class FakeAudio:
        file_name = "audio.wav"
        rainbow_color = the_rainbow_table_colors['Z']
        base_path = "/tmp/base"
        artifact_name = "artifactX"

        def get_artifact_path(self, with_file_name=False):
            return "/tmp/base" if not with_file_name else "/tmp/base/audio.wav"

    audio = FakeAudio()
    txt = chain_artifact_file_from_speech_to_text(audio, thread_id="thread-1")
    assert isinstance(txt, TextChainArtifactFile)
    assert txt.text_content == "transcript body"
    assert txt.thread_id == "thread-1"
    assert txt.chain_artifact_file_type is ChainArtifactFileType.MARKDOWN
    assert txt.artifact_name.endswith("_transcript")
    mock_save.assert_called_once_with(txt)


@patch("app.agents.tools.speech_tools.save_artifact_file_to_md")  # ✅ ADD: Mock the save function
def test_chain_artifact_file_from_speech_to_text_returns_placeholder_when_no_transcript(mock_save, monkeypatch):
    """Test that None transcript results in placeholder text, not None artifact"""
    monkeypatch.setattr("app.agents.tools.speech_tools.evp_speech_to_text", lambda wp, fn: None)

    class FakeAudio:
        file_name = "audio.wav"
        rainbow_color = the_rainbow_table_colors['Z']
        base_path = "/tmp/base"
        artifact_name = "artifactX"

        def get_artifact_path(self, with_file_name=False):
            return "/tmp/base" if not with_file_name else "/tmp/base/audio.wav"

    audio = FakeAudio()
    out = chain_artifact_file_from_speech_to_text(audio, thread_id="t")
    assert out is not None
    assert isinstance(out, TextChainArtifactFile)
    assert out.text_content == "[EVP: No discernible speech detected]"
    mock_save.assert_called_once_with(out)