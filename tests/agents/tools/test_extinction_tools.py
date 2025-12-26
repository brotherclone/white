"""Unit tests for extinction tools."""

import json
import os
import pytest
from unittest.mock import mock_open, patch, MagicMock

from app.agents.tools.extinction_tools import (
    load_green_corpus,
    get_random_species,
    parse_species_entry,
)
from app.structures.artifacts.species_extinction_artifact import (
    SpeciesExtinctionArtifact,
)
from app.structures.enums.extinction_cause import ExtinctionCause


@pytest.fixture
def sample_species_entry():
    """Sample species entry for testing"""
    return {
        "species": {
            "scientific_name": "Testus extinctus",
            "common_name": "Test Species",
            "taxonomic_group": "Mammalia",
            "iucn_status": "Extinct",
            "extinction_year": 2020,
            "population_trajectory": [
                {"year": 2000, "count": 1000, "count_uncertainty": 100},
                {"year": 2010, "count": 500, "count_uncertainty": 50},
                {"year": 2020, "count": 0, "count_uncertainty": 0},
            ],
            "habitat": "Test habitat",
            "endemic": True,
            "range_km2": 100.0,
            "primary_cause": "habitat_loss",
            "secondary_causes": ["climate_change"],
            "anthropogenic_factors": ["deforestation"],
            "cascade_effects": ["ecosystem disruption"],
            "ecosystem_role": "Test role",
            "symbolic_resonance": "Test resonance",
            "human_parallel_hints": "Test hints",
            "narrative_potential_score": 8,
            "symbolic_weight": 7,
            "size_category": "medium",
            "lifespan_years": 10,
            "movement_pattern": "sedentary",
        }
    }


@pytest.fixture
def sample_corpus(sample_species_entry):
    """Sample corpus for testing"""
    return {
        "entries": [sample_species_entry, sample_species_entry],
        "random": MagicMock(),
    }


def test_load_green_corpus_success():
    """Test successful loading of green corpus"""
    mock_corpus_data = {"entries": [], "metadata": {}}
    mock_file_content = json.dumps(mock_corpus_data)

    with patch("builtins.open", mock_open(read_data=mock_file_content)):
        with patch.dict(os.environ, {"GREEN_CORPUS_DIR": "/fake/path/corpus.json"}):
            result = load_green_corpus()

    assert result == mock_corpus_data
    assert "entries" in result


def test_load_green_corpus_custom_path():
    """Test loading corpus with custom path"""
    mock_corpus_data = {"entries": []}
    mock_file_content = json.dumps(mock_corpus_data)

    with patch("builtins.open", mock_open(read_data=mock_file_content)):
        result = load_green_corpus("/custom/path/corpus.json")

    assert result == mock_corpus_data


def test_parse_species_entry(sample_species_entry):
    """Test parsing a species entry into SpeciesExtinctionArtifact"""
    result = parse_species_entry(sample_species_entry)

    assert isinstance(result, SpeciesExtinctionArtifact)
    assert result.scientific_name == "Testus extinctus"
    assert result.common_name == "Test Species"
    assert result.taxonomic_group == "Mammalia"
    assert result.extinction_year == 2020
    assert result.primary_cause == ExtinctionCause.HABITAT_LOSS
    assert len(result.population_trajectory) == 3
    assert result.endemic is True
    assert result.narrative_potential_score == 8


def test_parse_species_entry_population_trajectory(sample_species_entry):
    """Test that population trajectory is correctly parsed"""
    result = parse_species_entry(sample_species_entry)

    assert len(result.population_trajectory) == 3
    assert result.population_trajectory[0].year == 2000
    assert result.population_trajectory[0].count == 1000
    assert result.population_trajectory[2].count == 0


def test_parse_species_entry_optional_fields(sample_species_entry):
    """Test parsing with optional fields"""
    # Remove optional fields
    del sample_species_entry["species"]["range_km2"]
    del sample_species_entry["species"]["lifespan_years"]

    result = parse_species_entry(sample_species_entry)

    assert result.range_km2 is None
    assert result.lifespan_years is None


def test_get_random_species(sample_corpus, sample_species_entry):
    """Test getting a random species from corpus"""
    # Mock the random index
    sample_corpus["random"].randint.return_value = 0

    with patch("app.agents.tools.extinction_tools.corpus", sample_corpus):
        result = get_random_species(sample_corpus)

    assert isinstance(result, SpeciesExtinctionArtifact)
    assert result.scientific_name == "Testus extinctus"
    sample_corpus["random"].randint.assert_called_once_with(0, 1)


def test_parse_species_entry_extinction_cause_enum():
    """Test that ExtinctionCause enum is correctly parsed"""
    entry = {
        "species": {
            "scientific_name": "Test species",
            "common_name": "Test",
            "taxonomic_group": "Test",
            "iucn_status": "Extinct",
            "extinction_year": 2020,
            "population_trajectory": [],
            "habitat": "Test",
            "endemic": False,
            "primary_cause": "overhunting",
            "secondary_causes": [],
            "anthropogenic_factors": [],
            "cascade_effects": [],
            "ecosystem_role": "Test",
            "symbolic_resonance": "Test",
            "human_parallel_hints": "Test",
            "narrative_potential_score": 5,
            "symbolic_weight": 5,
            "size_category": "small",
        }
    }

    result = parse_species_entry(entry)
    assert result.primary_cause == ExtinctionCause.OVERHUNTING
