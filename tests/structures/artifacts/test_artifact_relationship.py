from pydantic import BaseModel
from app.structures.artifacts.artifact_relationship import ArtifactRelationship


def test_artifact_relationship():
    r = ArtifactRelationship(artifact_id="mock_01")
    assert r.artifact_id == "mock_01"
    assert isinstance(r, BaseModel)


def test_initial_values():
    r = ArtifactRelationship(artifact_id="mock_01")
    assert r.resonant_agents == []
    assert r.entangled_with == []
    assert r.temporal_depth == {}
    assert r.semantic_tags == []
    assert r.model_config is not None


def test_with_resonant_agents():
    """Test creating relationship with resonant agents."""
    r = ArtifactRelationship(
        artifact_id="mock_01",
        resonant_agents=["red_agent", "blue_agent", "green_agent"],
    )
    assert len(r.resonant_agents) == 3
    assert "red_agent" in r.resonant_agents
    assert "blue_agent" in r.resonant_agents
    assert "green_agent" in r.resonant_agents


def test_with_entangled_artifacts():
    """Test creating relationship with entangled artifacts."""
    r = ArtifactRelationship(
        artifact_id="artifact_a", entangled_with=["artifact_b", "artifact_c"]
    )
    assert len(r.entangled_with) == 2
    assert "artifact_b" in r.entangled_with
    assert "artifact_c" in r.entangled_with


def test_with_temporal_depth():
    """Test creating relationship with temporal depth mappings."""
    r = ArtifactRelationship(
        artifact_id="mock_01",
        temporal_depth={
            "red": "Surface meaning",
            "violet": "Deep esoteric meaning",
            "white": "Unified spectrum meaning",
        },
    )
    assert len(r.temporal_depth) == 3
    assert r.temporal_depth["red"] == "Surface meaning"
    assert r.temporal_depth["violet"] == "Deep esoteric meaning"
    assert r.temporal_depth["white"] == "Unified spectrum meaning"


def test_with_semantic_tags():
    """Test creating relationship with semantic tags."""
    r = ArtifactRelationship(
        artifact_id="mock_01",
        semantic_tags=["liminal", "transformation", "rebirth", "threshold"],
    )
    assert len(r.semantic_tags) == 4
    assert "liminal" in r.semantic_tags
    assert "transformation" in r.semantic_tags
    assert "rebirth" in r.semantic_tags
    assert "threshold" in r.semantic_tags


def test_full_relationship():
    """Test creating a fully populated artifact relationship."""
    r = ArtifactRelationship(
        artifact_id="full_artifact",
        resonant_agents=["orange_agent", "indigo_agent"],
        entangled_with=["artifact_x", "artifact_y", "artifact_z"],
        temporal_depth={
            "orange": "Mythic resonance",
            "indigo": "Creative transformation",
            "white": "Full spectrum integration",
        },
        semantic_tags=["mythos", "gnosis", "creation", "destruction"],
    )
    assert r.artifact_id == "full_artifact"
    assert len(r.resonant_agents) == 2
    assert len(r.entangled_with) == 3
    assert len(r.temporal_depth) == 3
    assert len(r.semantic_tags) == 4


def test_empty_artifact_id_allowed():
    """Test that empty string artifact_id is technically allowed by the model."""
    r = ArtifactRelationship(artifact_id="")
    assert r.artifact_id == ""
    assert r.resonant_agents == []
