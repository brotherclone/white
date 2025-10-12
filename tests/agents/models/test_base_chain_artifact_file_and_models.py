import os
import pytest
from app.agents.models.audio_chain_artifact_file import AudioChainArtifactFile
from app.agents.models.text_chain_artifact_file import TextChainArtifactFile
from app.agents.models.image_chain_artifact_file import ImageChainArtifactFile
from app.agents.models.evp_artifact import EVPArtifact
from app.agents.models.sigil_artifact import SigilArtifact
from app.agents.enums.chain_artifact_file_type import ChainArtifactFileType
from app.agents.enums.sigil_type import SigilType
from app.agents.enums.sigil_state import SigilState
from app.structures.concepts.rainbow_table_color import get_rainbow_table_color


def test_get_artifact_path_with_color_and_enum_type(tmp_path):
    base = str(tmp_path)
    color = get_rainbow_table_color("R")
    ac = AudioChainArtifactFile(base_path=base, rainbow_color=color, chain_artifact_file_type=ChainArtifactFileType.AUDIO, file_name="track.wav")
    p = ac.get_artifact_path()
    expected = os.path.join(base, color.color_name.lower(), ChainArtifactFileType.AUDIO.value, "track.wav")
    assert p == expected


def test_get_artifact_path_without_file_name_and_with_file_flag(tmp_path):
    base = str(tmp_path)
    color = get_rainbow_table_color("R")
    ac = AudioChainArtifactFile(base_path=base, rainbow_color=color, chain_artifact_file_type=ChainArtifactFileType.AUDIO)
    p = ac.get_artifact_path(with_file_name=False)
    expected = os.path.join(base, color.color_name.lower(), ChainArtifactFileType.AUDIO.value)
    assert p == expected


def test_get_artifact_path_missing_color_and_string_type(tmp_path):
    base = str(tmp_path)
    # construct without validation so we can supply a plain string for chain_artifact_file_type
    ac = AudioChainArtifactFile.model_construct(
        base_path=base, rainbow_color=None, chain_artifact_file_type="audio_folder", file_name=None
    )
    p = ac.get_artifact_path()
    expected = os.path.join(base, "unknown", "audio_folder", ".unknown")
    assert p == expected


def test_get_artifact_path_uses_env_when_base_empty(monkeypatch, tmp_path):
    env_base = str(tmp_path / "envbase")
    monkeypatch.setenv("AGENT_WORK_PRODUCT_BASE_PATH", env_base)
    ac = AudioChainArtifactFile(base_path="", rainbow_color=None, chain_artifact_file_type=ChainArtifactFileType.JSON, file_name="data.json")
    p = ac.get_artifact_path()
    expected = os.path.join(env_base, "unknown", ChainArtifactFileType.JSON.value, "data.json")
    assert p == expected


def test_text_image_audio_model_defaults():
    text = TextChainArtifactFile(base_path="/tmp", chain_artifact_file_type=ChainArtifactFileType.MARKDOWN, file_name="doc.md", text_content="hello")
    assert text.text_content == "hello"
    img = ImageChainArtifactFile(base_path="/tmp", chain_artifact_file_type=ChainArtifactFileType.PNG, file_name="img.png")
    assert img.file_name == "img.png"
    audio = AudioChainArtifactFile(base_path="/tmp", chain_artifact_file_type=ChainArtifactFileType.AUDIO, file_name="a.wav")
    assert audio.sample_rate == 44100


def test_evp_and_sigil_artifact_instantiation(tmp_path):
    base = str(tmp_path)
    # build simple component files
    audio1 = AudioChainArtifactFile(base_path=base, chain_artifact_file_type=ChainArtifactFileType.AUDIO, file_name="a1.wav")
    audio2 = AudioChainArtifactFile(base_path=base, chain_artifact_file_type=ChainArtifactFileType.AUDIO, file_name="a2.wav")
    transcript = TextChainArtifactFile(base_path=base, chain_artifact_file_type=ChainArtifactFileType.JSON, file_name="t.json", text_content="transcript")

    evp = EVPArtifact(
        chain_artifact_type="evp",
        files=[audio1, audio2],
        audio_segments=[audio1, audio2],
        transcript=transcript,
        audio_mosiac=audio1,
        noise_blended_audio=audio2,
        thread_id="thread-1"
    )
    assert evp.thread_id == "thread-1"
    # clean_temp_files is a placeholder, should be callable and return None
    assert evp.clean_temp_files() is None

    sig = SigilArtifact(
        chain_artifact_type="sigil",
        files=[],
        thread_id="t1",
        wish="be better",
        statement_of_intent="improve",
        sigil_type=SigilType.WORD_METHOD,
        glyph_description="simple glyph",
        activation_state=SigilState.CREATED,
        charging_instructions="do the thing"
    )
    assert sig.chain_artifact_type == "sigil"
    assert sig.activation_state is SigilState.CREATED
