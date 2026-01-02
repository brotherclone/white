from pydantic import BaseModel

from app.structures.concepts.facet_evolution import FacetEvolution
from app.structures.enums.white_facet import WhiteFacet


def test_facet_evolution():
    f = FacetEvolution(initial_facet=WhiteFacet.TECHNICAL)
    assert f.initial_facet == WhiteFacet.TECHNICAL
    assert f.current_refraction_angle == "initial"
    assert isinstance(f, BaseModel)


def test_initial_values():
    """Test default initial values."""
    f = FacetEvolution(initial_facet=WhiteFacet.RELATIONAL)
    assert f.initial_facet == WhiteFacet.RELATIONAL
    assert f.initial_metadata == {}
    assert f.evolution_history == []
    assert f.current_refraction_angle == "initial"


def test_with_initial_metadata():
    """Test creating with initial metadata."""
    metadata = {
        "agent": "red_agent",
        "context": "initial processing",
        "timestamp": "2025-01-01",
    }
    f = FacetEvolution(
        initial_facet=WhiteFacet.PHENOMENOLOGICAL, initial_metadata=metadata
    )
    assert f.initial_metadata == metadata
    assert f.initial_metadata["agent"] == "red_agent"
    assert f.initial_metadata["context"] == "initial processing"


def test_with_evolution_history():
    """Test creating with evolution history."""
    history = [
        {"facet": "technical", "agent": "red_agent", "refraction_angle": "45_degrees"},
        {"facet": "empirical", "agent": "blue_agent", "refraction_angle": "90_degrees"},
    ]
    f = FacetEvolution(initial_facet=WhiteFacet.TECHNICAL, evolution_history=history)
    assert len(f.evolution_history) == 2
    assert f.evolution_history[0]["agent"] == "red_agent"
    assert f.evolution_history[1]["agent"] == "blue_agent"


def test_with_custom_refraction_angle():
    """Test creating with custom refraction angle."""
    f = FacetEvolution(
        initial_facet=WhiteFacet.COMPARATIVE, current_refraction_angle="prism_shift_127"
    )
    assert f.current_refraction_angle == "prism_shift_127"


def test_all_facet_types():
    """Test creating with each facet type."""
    facets = [
        WhiteFacet.CATEGORICAL,
        WhiteFacet.RELATIONAL,
        WhiteFacet.PROCEDURAL,
        WhiteFacet.COMPARATIVE,
        WhiteFacet.ARCHETYPAL,
        WhiteFacet.TECHNICAL,
        WhiteFacet.PHENOMENOLOGICAL,
    ]
    for facet in facets:
        f = FacetEvolution(initial_facet=facet)
        assert f.initial_facet == facet
        assert f.current_refraction_angle == "initial"


def test_full_evolution():
    """Test creating a fully populated evolution."""
    f = FacetEvolution(
        initial_facet=WhiteFacet.TECHNICAL,
        initial_metadata={
            "starting_agent": "red_agent",
            "inception_point": "artifact_001",
        },
        evolution_history=[
            {
                "stage": 1,
                "agent": "orange_agent",
                "facet_shift": "technical_to_empirical",
                "refraction_angle": "30_degrees",
            },
            {
                "stage": 2,
                "agent": "yellow_agent",
                "facet_shift": "empirical_to_contemplative",
                "refraction_angle": "60_degrees",
            },
            {
                "stage": 3,
                "agent": "green_agent",
                "facet_shift": "contemplative_to_systemic",
                "refraction_angle": "90_degrees",
            },
        ],
        current_refraction_angle="full_spectrum_integration",
    )
    assert f.initial_facet == WhiteFacet.TECHNICAL
    assert len(f.initial_metadata) == 2
    assert len(f.evolution_history) == 3
    assert f.current_refraction_angle == "full_spectrum_integration"
    assert f.evolution_history[2]["stage"] == 3
