from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from app.structures.manifests.song_proposal import SongProposal, SongProposalIteration


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


def test_save_all_proposals_writes_yaml(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_WORK_PRODUCT_BASE_PATH", str(tmp_path))
    iteration_data = valid_iteration_data(iteration_id="write_test_1")
    iteration = SongProposalIteration(**iteration_data)
    proposal = SongProposal(iterations=[iteration])
    proposal.save_all_proposals()
    out_dir = Path(tmp_path) / "default_thread"
    out_file = out_dir / f"{iteration.iteration_id}.yml"
    assert out_file.exists()
    with out_file.open("r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f)
    assert "iterations" in loaded
    assert loaded["iterations"][0]["iteration_id"] == "write_test_1"
    assert loaded["iterations"][0]["key"] == proposal.iterations[0].key
