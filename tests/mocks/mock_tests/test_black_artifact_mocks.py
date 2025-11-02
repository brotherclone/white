from pathlib import Path
import yaml

from app.structures.artifacts.evp_artifact import EVPArtifact
from app.structures.artifacts.sigil_artifact import SigilArtifact
from app.structures.artifacts.audio_chain_artifact_file import AudioChainArtifactFile
from app.structures.artifacts.text_chain_artifact_file import TextChainArtifactFile
from app.structures.concepts.rainbow_table_color import RainbowColorModes
from copy import deepcopy
from hypothesis import given, strategies as st


def test_evp_artifact_mock():
    path = Path(__file__).resolve().parents[3] / "tests" /  "mocks" / "black_evp_artifact_mock.yml"
    with path.open("r") as f:
        data = yaml.safe_load(f)
    evp = EVPArtifact(**data)
    assert isinstance(evp, EVPArtifact)
    assert isinstance(evp.audio_segments[0], AudioChainArtifactFile)
    assert isinstance(evp.transcript, TextChainArtifactFile)
    assert isinstance(evp.audio_mosiac, AudioChainArtifactFile)
    assert isinstance(evp.noise_blended_audio, AudioChainArtifactFile)



def test_sigil_artifact_mock():
    path = Path(__file__).resolve().parents[3] / "tests" /  "mocks" / "black_evp_artifact_mock.yml"
    with path.open("r") as f:
        data = yaml.safe_load(f)
    sig = SigilArtifact(**data)
    assert isinstance(sig, SigilArtifact)
    assert isinstance(sig.artifact_report, TextChainArtifactFile)


@given(
    current=st.sampled_from([m.value for m in RainbowColorModes]),
    transitory=st.sampled_from([m.value for m in RainbowColorModes]),
    transcendental=st.sampled_from([m.value for m in RainbowColorModes]),
)
def test_evp_transmigrational_mode_properties(current, transitory, transcendental):
    path = Path(__file__).resolve().parents[3] / "tests" /  "mocks" / "black_evp_artifact_mock.yml"
    with path.open("r") as f:
        base = yaml.safe_load(f)

    candidate = deepcopy(base)
    tm = {
        "current_mode": current,
        "transitory_mode": transitory,
        "transcendental_mode": transcendental,
    }
    candidate["audio_segments"][0]["rainbow_color"]["transmigrational_mode"] = tm
    candidate["transcript"]["rainbow_color"]["transmigrational_mode"] = tm
    evp = EVPArtifact(**candidate)
    assert isinstance(evp, EVPArtifact)


@given(
    current=st.sampled_from([m.value for m in RainbowColorModes]),
    transitory=st.sampled_from([m.value for m in RainbowColorModes]),
    transcendental=st.sampled_from([m.value for m in RainbowColorModes]),
)
def test_sigil_transmigrational_mode_properties(current, transitory, transcendental):
    path = Path(__file__).resolve().parents[3] / "tests" / "mocks" / "black_evp_artifact_mock.yml"
    with path.open("r") as f:
        base = yaml.safe_load(f)

    candidate = deepcopy(base)
    tm = {
        "current_mode": current,
        "transitory_mode": transitory,
        "transcendental_mode": transcendental,
    }

    # Prefer setting on artifact_report; fall back to transcript; create artifact_report as last resort.
    if isinstance(candidate.get("artifact_report"), dict):
        candidate["artifact_report"].setdefault("rainbow_color", {})["transmigrational_mode"] = tm
    elif isinstance(candidate.get("transcript"), dict):
        candidate["transcript"].setdefault("rainbow_color", {})["transmigrational_mode"] = tm
    else:
        candidate["artifact_report"] = {"rainbow_color": {"transmigrational_mode": tm}}

    # Provide minimal defaults for required SigilArtifact fields when the mock lacks them.
    sigil_defaults = {
        "wish": "seek clarity",
        "statement_of_intent": "to explore the sigil",
        "sigil_type": "pictorial",
        "glyph_description": "geometric abstraction",
        "activation_state": "created",
        "charging_instructions": "focus and breathe",
    }
    for key, val in sigil_defaults.items():
        candidate.setdefault(key, val)

    sig = SigilArtifact(**candidate)
    assert isinstance(sig, SigilArtifact)