import enum

import pytest

from app.structures.enums.chain_artifact_type import ChainArtifactType

EXPECTED = {
    "EVP_ARTIFACT": "evp_artifact",
    "INSTRUCTIONS_TO_HUMAN": "instructions_to_human",
    "SIGIL": "sigil_description",
    "BOOK": "book",
    "NEWSPAPER_ARTICLE": "newspaper_article",
    "SYMBOLIC_OBJECT": "symbolic_object",
    "PROPOSAL": "proposal",
    "GAME_RUN": "game_run",
    "CHARACTER_SHEET": "character_sheet",
    "CHARACTER_PORTRAIT": "character_portrait",
    "ARBITRARYS_SURVEY": "arbitrary_survey",
    "LAST_HUMAN": "last_human",
    "LAST_HUMAN_SPECIES_EXTINCTION_NARRATIVE": "last_human_species_extinction_narrative",
    "SPECIES_EXTINCTION": "species_extinction",
    "UNKNOWN": "unknown",
}


def test_members_and_values():
    assert set(ChainArtifactType.__members__.keys()) == set(EXPECTED.keys())
    for name, expected_value in EXPECTED.items():
        member = getattr(ChainArtifactType, name)
        assert member.value == expected_value
        assert isinstance(member.value, str)


def test_members_are_str_and_enum_and_compare_to_value():
    for member in ChainArtifactType:
        assert isinstance(member, ChainArtifactType)
        assert isinstance(member, str)
        assert isinstance(member, enum.Enum)
        assert member == member.value


@pytest.mark.parametrize(
    "value,member",
    [
        ("evp_artifact", ChainArtifactType.EVP_ARTIFACT),
        ("instructions_to_human", ChainArtifactType.INSTRUCTIONS_TO_HUMAN),
        ("sigil_description", ChainArtifactType.SIGIL),
        ("book", ChainArtifactType.BOOK),
        ("newspaper_article", ChainArtifactType.NEWSPAPER_ARTICLE),
        ("symbolic_object", ChainArtifactType.SYMBOLIC_OBJECT),
        ("proposal", ChainArtifactType.PROPOSAL),
        ("game_run", ChainArtifactType.GAME_RUN),
        ("character_sheet", ChainArtifactType.CHARACTER_SHEET),
        ("character_portrait", ChainArtifactType.CHARACTER_PORTRAIT),
        ("arbitrary_survey", ChainArtifactType.ARBITRARYS_SURVEY),
        ("last_human", ChainArtifactType.LAST_HUMAN),
        (
            "last_human_species_extinction_narrative",
            ChainArtifactType.LAST_HUMAN_SPECIES_EXTINCTION_NARRATIVE,
        ),
        ("species_extinction", ChainArtifactType.SPECIES_EXTINCTION),
        ("unknown", ChainArtifactType.UNKNOWN),
    ],
)
def test_lookup_by_value(value, member):
    assert ChainArtifactType(value) is member


def test_lookup_by_name():
    assert ChainArtifactType["PROPOSAL"] is ChainArtifactType.PROPOSAL
    assert ChainArtifactType["SYMBOLIC_OBJECT"] is ChainArtifactType.SYMBOLIC_OBJECT
    assert ChainArtifactType["UNKNOWN"] is ChainArtifactType.UNKNOWN
    assert ChainArtifactType["ARBITRARYS_SURVEY"] is ChainArtifactType.ARBITRARYS_SURVEY


def test_invalid_value_raises_value_error():
    with pytest.raises(ValueError):
        ChainArtifactType("pickled")


def test_values_are_unique():
    values = [m.value for m in ChainArtifactType]
    assert len(values) == len(set(values))


def test_enum_members_are_enum_instances():
    assert isinstance(ChainArtifactType.NEWSPAPER_ARTICLE, enum.Enum)
    assert isinstance(ChainArtifactType.INSTRUCTIONS_TO_HUMAN, enum.Enum)
    assert isinstance(ChainArtifactType.GAME_RUN, enum.Enum)
