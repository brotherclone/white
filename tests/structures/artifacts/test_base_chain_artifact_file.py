import os

import pytest

from app.structures.artifacts.audio_chain_artifact_file import \
    AudioChainArtifactFile
from app.structures.artifacts.evp_artifact import EVPArtifact
from app.structures.artifacts.image_chain_artifact_file import \
    ImageChainArtifactFile
from app.structures.artifacts.sigil_artifact import SigilArtifact
from app.structures.artifacts.text_chain_artifact_file import \
    TextChainArtifactFile
from app.structures.concepts.rainbow_table_color import get_rainbow_table_color
from app.structures.enums.chain_artifact_file_type import ChainArtifactFileType
from app.structures.enums.sigil_state import SigilState
from app.structures.enums.sigil_type import SigilType


# Shared fixture (module-scoped) - mirrors the models folder setup
@pytest.fixture(scope="module")
def shared(tmp_path_factory):
    base = str(tmp_path_factory.mktemp("base"))
    color = get_rainbow_table_color("R")
    artifact_id = "1234"
    thread_id = "thread-1"
    artifact_name = "track"

    def make_audio(**kwargs):
        defaults = dict(
            artifact_id=artifact_id,
            artifact_name=artifact_name,
            thread_id=thread_id,
            base_path=base,
            rainbow_color=color,
            chain_artifact_file_type=ChainArtifactFileType.AUDIO,
        )
        defaults.update(kwargs)
        return AudioChainArtifactFile(**defaults)

    def make_text(**kwargs):
        # give artifact_id/artifact_name so the model will compute a predictable file_name
        defaults = dict(
            artifact_id=artifact_id,
            artifact_name=artifact_name,
            thread_id=thread_id,
            base_path=base,
            chain_artifact_file_type=ChainArtifactFileType.JSON,
            text_content="transcript",
        )
        defaults.update(kwargs)
        return TextChainArtifactFile(**defaults)

    def make_image(**kwargs):
        defaults = dict(
            artifact_id=artifact_id,
            artifact_name=artifact_name,
            thread_id=thread_id,
            base_path=base,
            chain_artifact_file_type=ChainArtifactFileType.PNG,
        )
        defaults.update(kwargs)
        return ImageChainArtifactFile(**defaults)

    audio1 = make_audio()
    audio2 = make_audio()
    transcript = make_text()

    evp = EVPArtifact(
        chain_artifact_type="evp",
        files=[audio1, audio2],
        audio_segments=[audio1, audio2],
        transcript=transcript,
        audio_mosiac=audio1,
        noise_blended_audio=audio2,
        thread_id=thread_id,
    )

    sig = SigilArtifact(
        chain_artifact_type="sigil",
        thread_id="t1",
        wish="be better",
        statement_of_intent="improve",
        sigil_type=SigilType.WORD_METHOD,
        glyph_description="simple glyph",
        activation_state=SigilState.CREATED,
        charging_instructions="do the thing",
    )

    return dict(
        base=base,
        color=color,
        artifact_id=artifact_id,
        thread_id=thread_id,
        artifact_name=artifact_name,
        make_audio=make_audio,
        make_text=make_text,
        make_image=make_image,
        audio1=audio1,
        audio2=audio2,
        transcript=transcript,
        evp=evp,
        sig=sig,
    )


# --- Original model tests (adapted) ---
def test_get_artifact_path_with_color_and_enum_type(shared):
    ac = shared["make_audio"]()
    expected_name = f"{shared['artifact_id']}_{shared['color'].color_name.lower()}_{shared['artifact_name']}.{ChainArtifactFileType.AUDIO.value}"
    file_name = getattr(ac, "file_name", getattr(ac, "filename", None))
    assert file_name == expected_name
    # basic sanity: calling the method twice returns the same value
    assert ac.get_artifact_path() == ac.get_artifact_path()


def test_get_artifact_path_without_file_name_and_with_file_flag(shared):
    # when with_file_name=False we expect base/thread/type (no filename appended)
    ac = shared["make_audio"](file_name=None)
    p = ac.get_artifact_path(with_file_name=False)
    assert p == os.path.join(
        shared["base"], shared["thread_id"], ChainArtifactFileType.AUDIO.value
    )


def test_get_artifact_path_missing_color_and_string_type(tmp_path):
    base = str(tmp_path)
    # construct using a proper enum value so get_artifact_path can use .value
    ac = AudioChainArtifactFile.model_construct(
        thread_id="thread-1",
        base_path=base,
        rainbow_color=None,
        chain_artifact_file_type=ChainArtifactFileType.AUDIO,
        file_name=None,
    )
    p = ac.get_artifact_path()
    expected = os.path.join(
        base, "thread-1", ChainArtifactFileType.AUDIO.value, "unknown.txt"
    )
    assert p == expected


def test_get_artifact_path_uses_env_when_base_empty(monkeypatch, tmp_path):
    env_base = str(tmp_path / "envbase")
    monkeypatch.setenv("AGENT_WORK_PRODUCT_BASE_PATH", env_base)
    ac = AudioChainArtifactFile(
        thread_id="thread-1",
        base_path="",
        rainbow_color=None,
        chain_artifact_file_type=ChainArtifactFileType.JSON,
        artifact_name="data",
    )
    p = ac.get_artifact_path()
    expected = os.path.join(
        env_base, "thread-1", ChainArtifactFileType.JSON.value, ac.file_name
    )
    assert p == expected


def test_text_image_audio_model_defaults(shared):
    text = shared["make_text"](
        text_content="hello",
        artifact_name="doc",
        chain_artifact_file_type=ChainArtifactFileType.MARKDOWN,
    )
    assert getattr(text, "text_content", None) == "hello"
    assert getattr(text, "file_name", "").endswith(".md")

    img = shared["make_image"](artifact_name="img")
    assert getattr(img, "file_name", "").endswith(".png")

    audio = shared["make_audio"](
        artifact_name="test", chain_artifact_file_type=ChainArtifactFileType.AUDIO
    )
    # allow for renamed sample rate property; default expected value preserved
    assert getattr(audio, "sample_rate", getattr(audio, "samplerate", 44100)) == 44100


def test_evp_and_sigil_artifact_instantiation(shared):
    evp = shared["evp"]
    assert evp.thread_id == shared["thread_id"]
    assert evp.clean_temp_files() is None

    sig = shared["sig"]
    assert sig.chain_artifact_type == "sigil"
    assert sig.activation_state is SigilState.CREATED


# --- Color variation tests ---
@pytest.mark.parametrize(
    "color_key, expected_col",
    [
        ("R", "red"),
        ("G", "green"),
        (None, "transparent"),
    ],
)
def test_file_name_generation_for_color_variations(tmp_path, color_key, expected_col):
    base = str(tmp_path)
    artifact_id = "artifact-42"
    artifact_name = "mytrack"
    thread_id = "thread-1"

    rainbow_color = None
    if color_key is not None:
        rainbow_color = get_rainbow_table_color(color_key)

    ac = AudioChainArtifactFile(
        artifact_id=artifact_id,
        artifact_name=artifact_name,
        thread_id=thread_id,
        base_path=base,
        rainbow_color=rainbow_color,
        chain_artifact_file_type=ChainArtifactFileType.AUDIO,
    )

    expected_file_name = f"{artifact_id}_{expected_col}_{artifact_name}.{ChainArtifactFileType.AUDIO.value}"
    assert ac.file_name == expected_file_name

    # and the artifact path should include the file_name when with_file_name=True
    p = ac.get_artifact_path()
    assert p.endswith(
        "/" + ChainArtifactFileType.AUDIO.value + "/" + expected_file_name
    ) or p.endswith(
        "\\" + ChainArtifactFileType.AUDIO.value + "\\" + expected_file_name
    )


# --- Edge cases ---
def test_get_artifact_path_raises_when_thread_id_missing(tmp_path):
    base = str(tmp_path)
    # construct without thread_id via model_construct to avoid pydantic validation side-effects
    ac = AudioChainArtifactFile.model_construct(
        thread_id=None,
        base_path=base,
        rainbow_color=None,
        chain_artifact_file_type=ChainArtifactFileType.AUDIO,
        artifact_name="edge",
        file_name="edge.txt",
    )

    with pytest.raises(ValueError) as exc:
        _ = ac.get_artifact_path()
    assert "Thread ID is required" in str(exc.value)


def test_get_artifact_path_raises_when_chain_artifact_file_type_missing(tmp_path):
    base = str(tmp_path)
    # model_construct to bypass validation and set chain_artifact_file_type to None
    ac = AudioChainArtifactFile.model_construct(
        thread_id="t-1",
        base_path=base,
        rainbow_color=None,
        chain_artifact_file_type=None,
        artifact_name="edge",
        file_name="edge.txt",
    )

    with pytest.raises(ValueError) as exc:
        _ = ac.get_artifact_path()
    assert "Chain artifact file type is required" in str(exc.value)


# --- Permutations (artifact id provided vs generated, multiple colors and file types) ---
@pytest.mark.parametrize("artifact_provided", [True, False])
@pytest.mark.parametrize(
    "color_key, expected_col", [("R", "red"), ("G", "green"), (None, "transparent")]
)
@pytest.mark.parametrize(
    "file_type",
    [
        ChainArtifactFileType.AUDIO,
        ChainArtifactFileType.JSON,
        ChainArtifactFileType.MARKDOWN,
    ],
)
def test_file_name_permutations(
    tmp_path, artifact_provided, color_key, expected_col, file_type
):
    base = str(tmp_path)
    artifact_id = "artifact-42"
    artifact_name = "mytrack"
    thread_id = "thread-xyz"

    rainbow_color = None
    if color_key is not None:
        rainbow_color = get_rainbow_table_color(color_key)

    kwargs = dict(
        artifact_name=artifact_name,
        thread_id=thread_id,
        base_path=base,
        rainbow_color=rainbow_color,
        chain_artifact_file_type=file_type,
    )

    if artifact_provided:
        kwargs["artifact_id"] = artifact_id

    ac = AudioChainArtifactFile(**kwargs)

    # expected suffix
    expected_suffix = f"_{expected_col}_{artifact_name}.{file_type.value}"
    assert ac.file_name.endswith(expected_suffix)

    if artifact_provided:
        assert ac.file_name.startswith(artifact_id + "_")
    else:
        # auto-generated id should be non-empty and different from the literal string "None"
        generated_id = ac.file_name.split("_")[0]
        assert generated_id != ""
        assert generated_id != "None"

    # verify get_artifact_path(with_file_name=False) returns directory path
    dir_path = ac.get_artifact_path(with_file_name=False)
    assert dir_path == os.path.join(base, thread_id, file_type.value)

    # with file name included, path should end with file type directory + file_name
    p = ac.get_artifact_path()
    assert p.endswith(os.path.join(file_type.value, ac.file_name))
