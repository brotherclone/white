import pytest
from pydantic import ValidationError

from white_core.manifests.song_proposal import SongProposalIteration


def valid_iteration_data(**overrides):
    """Return a minimal valid iteration payload; allow overrides for targeted tests."""
    base = {
        "iteration_id": "test_iter_1",
        "bpm": 88,
        "tempo": {"numerator": 4, "denominator": 4},
        "key": "C Major",
        "rainbow_color": {
            "color_name": "Indigo",
            "hex_value": 4915330,
            "mnemonic_character_value": "I",
            "temporal_mode": "Future",
            "ontological_mode": ["Imagined"],
            "objectional_mode": "Person",
        },
        "title": "A Valid Title",
        "mood": ["yearning"],
        "genres": ["ambient"],
        "concept": "X" * 150,  # >= 100 chars to satisfy validator
    }
    base.update(overrides)
    return base


def test_normalize_flat_and_mode():
    it = SongProposalIteration(**valid_iteration_data(key="Bb Major"))
    assert it.key == "A# major"

    it2 = SongProposalIteration(**valid_iteration_data(key="Eb"))
    assert it2.key == "D#"

    it3 = SongProposalIteration(**valid_iteration_data(key="C mode: maj"))
    assert it3.key == "C major"


def test_normalize_modal_key_strings():
    it = SongProposalIteration(**valid_iteration_data(key="D dorian"))
    assert it.key == "D dorian"

    it2 = SongProposalIteration(**valid_iteration_data(key="E phrygian"))
    assert it2.key == "E phrygian"

    it3 = SongProposalIteration(**valid_iteration_data(key="G mixolydian"))
    assert it3.key == "G mixolydian"


def test_key_signature_object_serializes_to_string():
    """KeySignature objects must round-trip to a clean string, not a nested dict."""
    from white_core.music.core.key_signature import KeySignature

    # "D dorian" has no accidental so the round-trip is lossless
    ks = KeySignature.model_validate("D dorian")
    it = SongProposalIteration(**valid_iteration_data(key=ks))
    assert isinstance(it.key, str)
    assert it.key == "D dorian"

    dumped = it.model_dump(mode="json")
    assert isinstance(dumped["key"], str)
    assert dumped["key"] == "D dorian"


def test_non_note_key_remains_unchanged():
    raw = "Mode Major"
    it = SongProposalIteration(**valid_iteration_data(key=raw))
    assert it.key == raw


def test_title_not_empty_validator():
    with pytest.raises(ValueError):
        SongProposalIteration(**valid_iteration_data(title="   "))


def test_concept_substantive_validator():
    with pytest.raises(ValueError):
        SongProposalIteration(**valid_iteration_data(concept="Too short concept"))


def test_mood_and_genres_type_validators():
    with pytest.raises(ValidationError):
        SongProposalIteration(**valid_iteration_data(mood="not-a-list"))

    with pytest.raises(ValidationError):
        SongProposalIteration(**valid_iteration_data(genres="also-not-a-list"))
