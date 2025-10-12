import pytest
from app.agents.enums.chain_artifact_type import ChainArtifactType

def test_members_and_values():
    expected = {
        "DOCUMENT": "doc",
        "AUDIO_MOSIAC": "audio_mos",
        "RANDOM_AUDIO_BY_COLOR_SEGMENT": "col_audio_",
        "NOISE_MIXED_AUDIO": "noise_mixed",
    }
    for name, value in expected.items():
        member = ChainArtifactType[name]
        assert member.value == value
        assert isinstance(member.value, str)

@pytest.mark.parametrize("value,member", [
    ("doc", ChainArtifactType.DOCUMENT),
    ("audio_mos", ChainArtifactType.AUDIO_MOSIAC),
    ("col_audio_", ChainArtifactType.RANDOM_AUDIO_BY_COLOR_SEGMENT),
    ("noise_mixed", ChainArtifactType.NOISE_MIXED_AUDIO),
])
def test_lookup_by_value(value, member):
    assert ChainArtifactType(value) is member

def test_invalid_value_raises_value_error():
    with pytest.raises(ValueError):
        ChainArtifactType("invalid")

def test_values_are_unique():
    values = [m.value for m in ChainArtifactType]
    assert len(values) == len(set(values))

