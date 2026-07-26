import pytest
from pydantic import ValidationError

from white_core.manifests.song_proposal import (
    SongProposalIteration,
    resolve_supersession_chains,
)


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


def test_resolve_supersession_chains_clears_pivoted_seed():
    """A White seed pivoted to Black by a counter-proposal must lose is_final,
    even though color and title change mid-chain."""
    seed = SongProposalIteration(
        **valid_iteration_data(
            iteration_id="seed_v1",
            iteration_number=1,
            rainbow_color="White",
            is_final=True,
        )
    )
    pivot = SongProposalIteration(
        **valid_iteration_data(
            iteration_id="pivot_v2",
            iteration_number=2,
            rainbow_color="Black",
            is_final=True,
        )
    )

    resolved, superseded_ids = resolve_supersession_chains([seed, pivot])

    assert resolved == [seed, pivot]
    assert seed.is_final is False
    assert pivot.is_final is True
    assert id(seed) in superseded_ids
    assert id(pivot) not in superseded_ids


def test_resolve_supersession_chains_drops_exact_duplicates():
    seed = SongProposalIteration(
        **valid_iteration_data(
            iteration_id="seed_v1", iteration_number=1, is_final=True
        )
    )
    dup_a = SongProposalIteration(
        **valid_iteration_data(iteration_id="dup_v2", iteration_number=2, is_final=True)
    )
    dup_b = SongProposalIteration(
        **valid_iteration_data(iteration_id="dup_v2", iteration_number=2, is_final=True)
    )
    final = SongProposalIteration(
        **valid_iteration_data(
            iteration_id="final_v3", iteration_number=3, is_final=True
        )
    )

    resolved, _ = resolve_supersession_chains([seed, dup_a, dup_b, final])

    assert [it.iteration_id for it in resolved] == ["seed_v1", "dup_v2", "final_v3"]
    assert seed.is_final is False
    assert dup_a.is_final is False
    assert final.is_final is True


def test_resolve_supersession_chains_leaves_independent_iterations_untouched():
    """Non-consecutive iteration_number, or None, starts an independent chain
    of length one and must not be cleared."""
    orange = SongProposalIteration(
        **valid_iteration_data(
            iteration_id="orange_v1",
            iteration_number=1,
            rainbow_color="Orange",
            is_final=True,
        )
    )
    yellow = SongProposalIteration(
        **valid_iteration_data(
            iteration_id="yellow_v1",
            iteration_number=None,
            rainbow_color="Yellow",
            is_final=True,
        )
    )

    resolve_supersession_chains([orange, yellow])

    assert orange.is_final is True
    assert yellow.is_final is True
