from pathlib import Path

import yaml
from hypothesis import given
from hypothesis import strategies as st

from white_core.artifacts.audio_artifact_file import AudioChainArtifactFile
from white_core.artifacts.evp_artifact import EVPArtifact
from white_core.artifacts.sigil_artifact import SigilArtifact
from white_core.concepts.rainbow_table_color import RainbowColorModes


def test_evp_artifact_mock():
    path = (
        Path(__file__).resolve().parents[3]
        / "tests"
        / "mocks"
        / "black_evp_artifact_mock.yml"
    )
    with path.open("r") as f:
        data = yaml.safe_load(f)

    # Load mock audio bytes
    mock_audio_path = (
        Path(__file__).resolve().parents[3] / "tests" / "mocks" / "mock.wav"
    )
    with open(mock_audio_path, "rb") as f:
        audio_bytes = f.read()

    evp = EVPArtifact(**data)

    # Only mosaic is set now - no segments or blended
    evp.audio_mosiac = AudioChainArtifactFile(
        thread_id="test_thread_id",
        base_path="/chain_artifacts/",
        artifact_name="evp_mosaic",
        sample_rate=44100,
        duration=8.0,
        audio_bytes=audio_bytes,
        channels=2,
    )

    assert isinstance(evp, EVPArtifact)
    assert isinstance(evp.audio_mosiac, AudioChainArtifactFile)


def test_sigil_artifact_mock():
    path = (
        Path(__file__).resolve().parents[3]
        / "tests"
        / "mocks"
        / "black_sigil_artifact_mock.yml"
    )
    with path.open("r") as f:
        data = yaml.safe_load(f)
    sig = SigilArtifact(**data)
    assert isinstance(sig, SigilArtifact)
    assert (
        sig.wish
        == "I will encode liberation frequencies that bypass the Demiurge's surveillance grid."
    )


# current/transitory/transcendental are unused below — the mock data and
# constructed artifact never depend on them, so loading + validating on every
# hypothesis-generated example was pure overhead (and had started tripping the
# too_slow health check). Load once at module scope and reuse it.
_EVP_MOCK_PATH = (
    Path(__file__).resolve().parents[3]
    / "tests"
    / "mocks"
    / "black_evp_artifact_mock.yml"
)
with _EVP_MOCK_PATH.open("r") as _f:
    _EVP_ARTIFACT = EVPArtifact(**yaml.safe_load(_f))

_SIGIL_MOCK_PATH = (
    Path(__file__).resolve().parents[3]
    / "tests"
    / "mocks"
    / "black_sigil_artifact_mock.yml"
)
with _SIGIL_MOCK_PATH.open("r") as _f:
    _SIGIL_ARTIFACT = SigilArtifact(**yaml.safe_load(_f))


@given(
    current=st.sampled_from([m.value for m in RainbowColorModes]),
    transitory=st.sampled_from([m.value for m in RainbowColorModes]),
    transcendental=st.sampled_from([m.value for m in RainbowColorModes]),
)
def test_evp_transmigrational_mode_properties(current, transitory, transcendental):
    """Test that EVPArtifact can be created with various transmigrational modes"""
    assert isinstance(_EVP_ARTIFACT, EVPArtifact)
    assert (
        _EVP_ARTIFACT.transcript
        == "This is a test EVP transcript with mysterious voices"
    )


@given(
    current=st.sampled_from([m.value for m in RainbowColorModes]),
    transitory=st.sampled_from([m.value for m in RainbowColorModes]),
    transcendental=st.sampled_from([m.value for m in RainbowColorModes]),
)
def test_sigil_transmigrational_mode_properties(current, transitory, transcendental):
    """Test that SigilArtifact can be created with various transmigrational modes"""
    assert isinstance(_SIGIL_ARTIFACT, SigilArtifact)
    assert (
        _SIGIL_ARTIFACT.wish
        == "I will encode liberation frequencies that bypass the Demiurge's surveillance grid."
    )
