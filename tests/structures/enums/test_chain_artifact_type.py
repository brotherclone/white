import enum

import pytest

from app.structures.enums.chain_artifact_type import ChainArtifactType

EXPECTED = {
    "EVP_ARTIFACT": "evp_artifact",
    "INSTRUCTIONS_TO_HUMAN": "instructions_to_human",
    "SIGIL": "sigil",
    "SIGIL_DESCRIPTION": "sigil_description",
    "BOOK": "book",
    "NEWSPAPER_ARTICLE": "newspaper_article",
    "SYMBOLIC_OBJECT": "symbolic_object",
    "PROPOSAL": "proposal",
    "RESCUE_DECISION": "rescue_decision",
    "GAME_RUN": "game_run",
    "CHARACTER_SHEET": "character_sheet",
    "CHARACTER_PORTRAIT": "character_portrait",
    "ARBITRARYS_SURVEY": "arbitrary_survey",
    "LAST_HUMAN": "last_human",
    "LAST_HUMAN_SPECIES_EXTINCTION_NARRATIVE": "last_human_species_extinction_narrative",
    "SPECIES_EXTINCTION": "species_extinction",
    "QUANTUM_TAPE_LABEL": "quantum_tape_label",
    "ALTERNATE_TIMELINE": "alternate_timeline",
    "INFRANYM_MIDI": "infranym_midi",
    "INFRANYM_AUDIO": "infranym_audio",
    "INFRANYM_IMAGE": "infranym_image",
    "INFRANYM_TEXT": "infranym_text",
    "CIRCLE_JERK_INTERVIEW": "circle_jerk_interview",
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
        ("sigil", ChainArtifactType.SIGIL),
        ("sigil_description", ChainArtifactType.SIGIL_DESCRIPTION),
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
        ("quantum_tape_label", ChainArtifactType.QUANTUM_TAPE_LABEL),
        ("alternate_timeline", ChainArtifactType.ALTERNATE_TIMELINE),
        ("infranym_midi", ChainArtifactType.INFRANYM_MIDI),
        ("infranym_audio", ChainArtifactType.INFRANYM_AUDIO),
        ("infranym_image", ChainArtifactType.INFRANYM_IMAGE),
        ("infranym_text", ChainArtifactType.INFRANYM_TEXT),
        ("circle_jerk_interview", ChainArtifactType.CIRCLE_JERK_INTERVIEW),
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
    assert ChainArtifactType["EVP_ARTIFACT"] is ChainArtifactType.EVP_ARTIFACT
    assert ChainArtifactType["SIGIL"] is ChainArtifactType.SIGIL
    assert ChainArtifactType["NEWSPAPER_ARTICLE"] is ChainArtifactType.NEWSPAPER_ARTICLE
    assert ChainArtifactType["RESCUE_DECISION"] is ChainArtifactType.RESCUE_DECISION
    assert ChainArtifactType["GAME_RUN"] is ChainArtifactType.GAME_RUN
    assert ChainArtifactType["CHARACTER_SHEET"] is ChainArtifactType.CHARACTER_SHEET
    assert (
        ChainArtifactType["CHARACTER_PORTRAIT"] is ChainArtifactType.CHARACTER_PORTRAIT
    )
    assert (
        ChainArtifactType["QUANTUM_TAPE_LABEL"] is ChainArtifactType.QUANTUM_TAPE_LABEL
    )
    assert (
        ChainArtifactType["ALTERNATE_TIMELINE"] is ChainArtifactType.ALTERNATE_TIMELINE
    )
    assert ChainArtifactType["INFRANYM_MIDI"] is ChainArtifactType.INFRANYM_MIDI
    assert ChainArtifactType["INFRANYM_AUDIO"] is ChainArtifactType.INFRANYM_AUDIO


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
    assert isinstance(ChainArtifactType.UNKNOWN, enum.Enum)
    assert isinstance(ChainArtifactType.ARBITRARYS_SURVEY, enum.Enum)
    assert isinstance(ChainArtifactType.EVP_ARTIFACT, enum.Enum)
    assert isinstance(ChainArtifactType.SIGIL, enum.Enum)
    assert isinstance(ChainArtifactType.CHARACTER_SHEET, enum.Enum)
    assert isinstance(ChainArtifactType.CHARACTER_PORTRAIT, enum.Enum)
    assert isinstance(ChainArtifactType.QUANTUM_TAPE_LABEL, enum.Enum)
    assert isinstance(ChainArtifactType.ALTERNATE_TIMELINE, enum.Enum)
    assert isinstance(ChainArtifactType.INFRANYM_MIDI, enum.Enum)
    assert isinstance(ChainArtifactType.INFRANYM_AUDIO, enum.Enum)
    assert isinstance(ChainArtifactType.INFRANYM_IMAGE, enum.Enum)
    assert isinstance(ChainArtifactType.INFRANYM_TEXT, enum.Enum)
    assert isinstance(ChainArtifactType.CIRCLE_JERK_INTERVIEW, enum.Enum)
