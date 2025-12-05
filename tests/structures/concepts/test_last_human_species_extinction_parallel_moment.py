from pydantic import BaseModel, ValidationError
import pytest

from app.structures.concepts.last_human_species_extinction_parallel_moment import (
    LastHumanSpeciesExtinctionParallelMoment,
)


def test_inheritance():
    assert issubclass(LastHumanSpeciesExtinctionParallelMoment, BaseModel)


def test_parallel_moment_creation():
    """Test creating a LastHumanSpeciesExtinctionParallelMoment with all required fields."""
    moment = LastHumanSpeciesExtinctionParallelMoment(
        species_moment="Population declined below 100 individuals",
        human_moment="Family forced to leave ancestral fishing village",
        thematic_connection="Both losing their last viable habitat",
        timestamp_relative="Six months before extinction",
    )

    assert moment.species_moment == "Population declined below 100 individuals"
    assert moment.human_moment == "Family forced to leave ancestral fishing village"
    assert moment.thematic_connection == "Both losing their last viable habitat"
    assert moment.timestamp_relative == "Six months before extinction"


def test_parallel_moment_multiple_instances():
    """Test creating multiple parallel moments with different data."""
    moment1 = LastHumanSpeciesExtinctionParallelMoment(
        species_moment="Last breeding pair observed",
        human_moment="Final harvest from dying land",
        thematic_connection="Terminal reproduction failure",
        timestamp_relative="One year before extinction",
    )

    moment2 = LastHumanSpeciesExtinctionParallelMoment(
        species_moment="Habitat fragmented by development",
        human_moment="Community divided by climate migration",
        thematic_connection="Forced separation from necessary resources",
        timestamp_relative="Five years before extinction",
    )

    assert moment1.species_moment != moment2.species_moment
    assert moment1.thematic_connection != moment2.thematic_connection
    assert moment1.timestamp_relative != moment2.timestamp_relative


def test_parallel_moment_required_fields():
    """Test that all fields are required."""
    # Missing species_moment
    with pytest.raises(ValidationError):
        LastHumanSpeciesExtinctionParallelMoment(
            human_moment="Test human moment",
            thematic_connection="Test connection",
            timestamp_relative="Test time",
        )

    # Missing human_moment
    with pytest.raises(ValidationError):
        LastHumanSpeciesExtinctionParallelMoment(
            species_moment="Test species moment",
            thematic_connection="Test connection",
            timestamp_relative="Test time",
        )

    # Missing thematic_connection
    with pytest.raises(ValidationError):
        LastHumanSpeciesExtinctionParallelMoment(
            species_moment="Test species moment",
            human_moment="Test human moment",
            timestamp_relative="Test time",
        )

    # Missing timestamp_relative
    with pytest.raises(ValidationError):
        LastHumanSpeciesExtinctionParallelMoment(
            species_moment="Test species moment",
            human_moment="Test human moment",
            thematic_connection="Test connection",
        )


def test_parallel_moment_field_types():
    """Test that all fields are strings."""
    moment = LastHumanSpeciesExtinctionParallelMoment(
        species_moment="Ocean temperature increased 2°C",
        human_moment="Water wells dried up completely",
        thematic_connection="Environmental thresholds crossed",
        timestamp_relative="Three years before extinction",
    )

    assert isinstance(moment.species_moment, str)
    assert isinstance(moment.human_moment, str)
    assert isinstance(moment.thematic_connection, str)
    assert isinstance(moment.timestamp_relative, str)


def test_parallel_moment_model_dump():
    """Test model_dump returns correct dictionary."""
    moment = LastHumanSpeciesExtinctionParallelMoment(
        species_moment="Last confirmed sighting",
        human_moment="Last family member left",
        thematic_connection="Final presence documented",
        timestamp_relative="Extinction day",
    )

    dumped = moment.model_dump()

    assert dumped["species_moment"] == "Last confirmed sighting"
    assert dumped["human_moment"] == "Last family member left"
    assert dumped["thematic_connection"] == "Final presence documented"
    assert dumped["timestamp_relative"] == "Extinction day"


def test_parallel_moment_narrative_coherence():
    """Test creating moments that form a coherent narrative arc."""
    moments = [
        LastHumanSpeciesExtinctionParallelMoment(
            species_moment="Healthy population of 10,000",
            human_moment="Thriving fishing community of 500 families",
            thematic_connection="Abundance and stability",
            timestamp_relative="Twenty years before extinction",
        ),
        LastHumanSpeciesExtinctionParallelMoment(
            species_moment="Population declined to 1,000",
            human_moment="Half the families forced to migrate",
            thematic_connection="Collapse accelerating",
            timestamp_relative="Ten years before extinction",
        ),
        LastHumanSpeciesExtinctionParallelMoment(
            species_moment="Final individual dies",
            human_moment="Last elder passes away in displacement camp",
            thematic_connection="End of continuity",
            timestamp_relative="Extinction",
        ),
    ]

    assert len(moments) == 3
    for moment in moments:
        assert isinstance(moment, LastHumanSpeciesExtinctionParallelMoment)
