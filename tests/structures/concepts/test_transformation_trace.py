from pydantic import BaseModel

from app.structures.concepts.transformation_trace import TransformationTrace


def test_transformation_trace():
    t = TransformationTrace(agent_name="Mock", iteration_id="mock_01")
    assert t.agent_name == "Mock"
    assert t.iteration_id == "mock_01"
    assert isinstance(t, BaseModel)


def test_initial_values():
    """Test default initial values."""
    t = TransformationTrace(agent_name="TestAgent", iteration_id="iter_001")
    assert t.agent_name == "TestAgent"
    assert t.iteration_id == "iter_001"
    assert t.boundaries_shifted == []
    assert t.patterns_revealed == []
    assert t.semantic_resonances == {}
    assert t.content_excerpt is None
    assert t.artifact_count == 0


def test_with_boundaries_shifted():
    """Test creating with boundaries shifted."""
    t = TransformationTrace(
        agent_name="red_agent",
        iteration_id="iter_001",
        boundaries_shifted=[
            "TIME/SPACE boundary shifted - past events reframed as future potentials",
            "SELF/OTHER boundary dissolved",
            "FACT/FICTION boundary blurred",
        ],
    )
    assert len(t.boundaries_shifted) == 3
    assert "TIME/SPACE boundary shifted" in t.boundaries_shifted[0]
    assert "SELF/OTHER boundary dissolved" in t.boundaries_shifted[1]


def test_with_patterns_revealed():
    """Test creating with patterns revealed."""
    t = TransformationTrace(
        agent_name="orange_agent",
        iteration_id="iter_002",
        patterns_revealed=[
            "Mythic resonance: The journey pattern",
            "Archetypal structure: Death and rebirth",
            "Symbolic network: Water as transformation",
        ],
    )
    assert len(t.patterns_revealed) == 3
    assert "Mythic resonance" in t.patterns_revealed[0]
    assert "Archetypal structure" in t.patterns_revealed[1]
    assert "Symbolic network" in t.patterns_revealed[2]


def test_with_semantic_resonances():
    """Test creating with semantic resonances."""
    t = TransformationTrace(
        agent_name="indigo_agent",
        iteration_id="iter_003",
        semantic_resonances={
            "temporal": "past/present/future collapse into eternal now",
            "spatial": "inside/outside inversion",
            "identity": "observer becomes observed",
            "causality": "effect precedes cause",
        },
    )
    assert len(t.semantic_resonances) == 4
    assert (
        t.semantic_resonances["temporal"]
        == "past/present/future collapse into eternal now"
    )
    assert t.semantic_resonances["spatial"] == "inside/outside inversion"
    assert t.semantic_resonances["identity"] == "observer becomes observed"


def test_full_transformation_trace():
    """Test creating a fully populated transformation trace."""
    t = TransformationTrace(
        agent_name="violet_agent",
        iteration_id="final_iter",
        boundaries_shifted=[
            "KNOWN/UNKNOWN boundary erased",
            "SACRED/PROFANE boundary inverted",
        ],
        patterns_revealed=[
            "Fractal self-similarity across scales",
            "Strange loop: container contains itself",
        ],
        semantic_resonances={
            "ontological": "being emerges from non-being",
            "epistemological": "knowledge creates ignorance",
            "phenomenological": "experience precedes existence",
        },
    )
    assert t.agent_name == "violet_agent"
    assert t.iteration_id == "final_iter"
    assert len(t.boundaries_shifted) == 2
    assert len(t.patterns_revealed) == 2
    assert len(t.semantic_resonances) == 3


def test_empty_agent_name_allowed():
    """Test that empty agent name is allowed."""
    t = TransformationTrace(agent_name="", iteration_id="test")
    assert t.agent_name == ""
    assert t.iteration_id == "test"


def test_complex_semantic_resonances():
    """Test with complex nested semantic resonances."""
    t = TransformationTrace(
        agent_name="white_agent",
        iteration_id="integration",
        semantic_resonances={
            "spectrum_integration": {
                "red": "embodied presence",
                "orange": "mythic depth",
                "yellow": "aesthetic form",
            },
            "cognitive_shift": "analytical to holistic",
            "temporal_aspect": ["synchronic", "diachronic", "atemporal"],
        },
    )
    assert "spectrum_integration" in t.semantic_resonances
    assert t.semantic_resonances["cognitive_shift"] == "analytical to holistic"


def test_content_excerpt_and_artifact_count():
    """Test the Phase 3 additions: content_excerpt and artifact_count."""
    t = TransformationTrace(
        agent_name="black",
        iteration_id="iter_001",
        boundaries_shifted=["CHAOS → ORDER"],
        patterns_revealed=["Structured chaos"],
        content_excerpt="ThreadKeepr reveals the boundary between conscious ritual and unconscious manifestation...",
        artifact_count=3,
    )
    assert (
        t.content_excerpt
        == "ThreadKeepr reveals the boundary between conscious ritual and unconscious manifestation..."
    )
    assert t.artifact_count == 3


def test_content_excerpt_optional():
    """Test that content_excerpt is optional."""
    t = TransformationTrace(
        agent_name="red",
        iteration_id="iter_002",
        artifact_count=5,
    )
    assert t.content_excerpt is None
    assert t.artifact_count == 5
