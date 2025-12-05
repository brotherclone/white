from app.structures.artifacts.base_artifact import ChainArtifact
from app.structures.artifacts.last_human_species_extinction_narative_artifact import (
    LastHumanSpeciesExtinctionNarrativeArtifact,
)
from app.structures.artifacts.species_extinction_artifact import (
    SpeciesExtinctionArtifact,
)
from app.structures.artifacts.last_human_artifact import LastHumanArtifact
from app.structures.concepts.last_human_species_extinction_parallel_moment import (
    LastHumanSpeciesExtinctionParallelMoment,
)
from app.structures.enums.extinction_cause import ExtinctionCause
from app.structures.enums.last_human_vulnerability_type import (
    LastHumanVulnerabilityType,
)
from app.structures.enums.last_human_documentation_type import (
    LastHumanDocumentationType,
)


class ConcreteSpeciesExtinctionArtifact(SpeciesExtinctionArtifact):
    """Concrete implementation for testing"""

    thread_id: str = "test-thread"

    def flatten(self):
        return self.model_dump()

    def save_file(self):
        pass


class ConcreteLastHumanArtifact(LastHumanArtifact):
    """Concrete implementation for testing"""

    thread_id: str = "test-thread"

    def flatten(self):
        return self.model_dump()

    def save_file(self):
        pass


class ConcreteNarrativeArtifact(LastHumanSpeciesExtinctionNarrativeArtifact):
    """Concrete implementation for testing"""

    thread_id: str = "test-thread"

    def flatten(self):
        return self.model_dump()

    def save_file(self):
        pass


def test_inheritance():
    assert issubclass(LastHumanSpeciesExtinctionNarrativeArtifact, ChainArtifact)


def test_narrative_artifact_creation():
    """Test creating LastHumanSpeciesExtinctionNarrativeArtifact with required fields."""
    species = ConcreteSpeciesExtinctionArtifact(
        thread_id="test",
        scientific_name="Phocoena sinus",
        common_name="Vaquita",
        taxonomic_group="marine mammal",
        iucn_status="Critically Endangered",
        extinction_year=2030,
        habitat="Gulf of California",
        primary_cause=ExtinctionCause.BYCATCH,
        ecosystem_role="Small cetacean",
    )

    human = ConcreteLastHumanArtifact(
        thread_id="test",
        name="Carlos Mendez",
        age=55,
        location="San Felipe, Mexico",
        year_documented=2030,
        parallel_vulnerability=LastHumanVulnerabilityType.ENTANGLEMENT,
        vulnerability_details="Fishing ban destroyed livelihood",
        environmental_stressor="Conservation vs community conflict",
        documentation_type=LastHumanDocumentationType.RESILIENCE,
        last_days_scenario="Last boat tied to dock",
    )

    narrative = ConcreteNarrativeArtifact(
        thread_id="test",
        species=species,
        human=human,
        species_arc="Population declined from 600 to extinction in 30 years",
        human_arc="Family fishing tradition ended after five generations",
        elegiac_quality="Neither villain nor victim, just parallel losses",
        opening_image="Morning mist over empty waters",
        closing_image="Silence where porpoises once surfaced",
    )

    assert narrative.species == species
    assert narrative.human == human
    assert narrative.species_arc is not None
    assert narrative.human_arc is not None
    assert narrative.parallel_moments == []
    assert narrative.elegiac_quality is not None
    assert narrative.opening_image == "Morning mist over empty waters"
    assert narrative.closing_image == "Silence where porpoises once surfaced"
    assert narrative.emotional_curve == []
    assert narrative.silence_moments == []


def test_narrative_artifact_with_parallel_moments():
    """Test creating narrative with parallel moments."""
    species = ConcreteSpeciesExtinctionArtifact(
        thread_id="test",
        scientific_name="Testus narrativus",
        common_name="Narrative Species",
        taxonomic_group="bird",
        iucn_status="Extinct",
        extinction_year=2050,
        habitat="Forest",
        primary_cause=ExtinctionCause.HABITAT_LOSS,
        ecosystem_role="Seed disperser",
    )

    human = ConcreteLastHumanArtifact(
        thread_id="test",
        name="Test Human",
        age=40,
        location="Forest Village",
        year_documented=2050,
        parallel_vulnerability=LastHumanVulnerabilityType.DISPLACEMENT,
        vulnerability_details="Forest cleared, village abandoned",
        environmental_stressor="Complete deforestation",
        documentation_type=LastHumanDocumentationType.DISPLACEMENT,
        last_days_scenario="Walking away from empty village",
    )

    moments = [
        LastHumanSpeciesExtinctionParallelMoment(
            species_moment="Last nest found empty",
            human_moment="Last harvest failed",
            thematic_connection="Terminal reproductive failure",
            timestamp_relative="One year before extinction",
        ),
        LastHumanSpeciesExtinctionParallelMoment(
            species_moment="Final individual seen",
            human_moment="Final family left",
            thematic_connection="End of continuity",
            timestamp_relative="Extinction day",
        ),
    ]

    narrative = ConcreteNarrativeArtifact(
        thread_id="test",
        species=species,
        human=human,
        species_arc="Test species arc",
        human_arc="Test human arc",
        parallel_moments=moments,
        elegiac_quality="Loss without blame",
        opening_image="Dawn chorus",
        closing_image="Silence",
        emotional_curve=["abundance", "decline", "acceptance"],
        silence_moments=["After the final call", "Empty nest"],
    )

    assert len(narrative.parallel_moments) == 2
    assert narrative.parallel_moments[0].species_moment == "Last nest found empty"
    assert len(narrative.emotional_curve) == 3
    assert len(narrative.silence_moments) == 2


def test_to_artifact_dict():
    """Test to_artifact_dict method."""
    species = ConcreteSpeciesExtinctionArtifact(
        thread_id="test",
        scientific_name="Testus dictus",
        common_name="Dict Species",
        taxonomic_group="mammal",
        iucn_status="Extinct",
        extinction_year=2045,
        habitat="Plains",
        primary_cause=ExtinctionCause.CLIMATE_CHANGE,
        ecosystem_role="Grazer",
    )

    human = ConcreteLastHumanArtifact(
        thread_id="test",
        name="Dict Human",
        age=50,
        location="Plains Village",
        year_documented=2045,
        parallel_vulnerability=LastHumanVulnerabilityType.RESOURCE_COLLAPSE,
        vulnerability_details="Grasslands died",
        environmental_stressor="Drought",
        documentation_type=LastHumanDocumentationType.DEATH,
        last_days_scenario="Watching the dust",
    )

    moment = LastHumanSpeciesExtinctionParallelMoment(
        species_moment="Herd scattered",
        human_moment="Community dispersed",
        thematic_connection="Fragmentation",
        timestamp_relative="Six months before",
    )

    narrative = ConcreteNarrativeArtifact(
        thread_id="test",
        species=species,
        human=human,
        species_arc="Ecological collapse",
        human_arc="Social collapse",
        parallel_moments=[moment],
        elegiac_quality="Parallel mourning",
        opening_image="Green grass",
        closing_image="Dust",
        emotional_curve=["hope", "loss"],
        silence_moments=["Final breath"],
    )

    artifact_dict = narrative.to_artifact_dict()

    assert artifact_dict["species_name"] == "Dict Species"
    assert artifact_dict["human_name"] == "Dict Human"
    assert artifact_dict["species_arc"] == "Ecological collapse"
    assert artifact_dict["human_arc"] == "Social collapse"
    assert len(artifact_dict["parallel_moments"]) == 1
    assert artifact_dict["elegiac_quality"] == "Parallel mourning"
    assert artifact_dict["opening_image"] == "Green grass"
    assert artifact_dict["closing_image"] == "Dust"
    assert artifact_dict["emotional_curve"] == ["hope", "loss"]
    assert artifact_dict["silence_moments"] == ["Final breath"]


def test_to_markdown():
    """Test to_markdown method."""
    species = ConcreteSpeciesExtinctionArtifact(
        thread_id="test",
        scientific_name="Testus markdownus",
        common_name="Markdown Species",
        taxonomic_group="reptile",
        iucn_status="Extinct",
        extinction_year=2040,
        habitat="Desert",
        primary_cause=ExtinctionCause.HABITAT_LOSS,
        ecosystem_role="Predator",
    )

    human = ConcreteLastHumanArtifact(
        thread_id="test",
        name="Markdown Human",
        age=35,
        location="Desert Oasis",
        year_documented=2040,
        parallel_vulnerability=LastHumanVulnerabilityType.ISOLATION,
        vulnerability_details="Cut off from world",
        environmental_stressor="Oasis drying",
        documentation_type=LastHumanDocumentationType.WITNESS,
        last_days_scenario="Recording the end",
    )

    moment = LastHumanSpeciesExtinctionParallelMoment(
        species_moment="Water hole dried",
        human_moment="Well went empty",
        thematic_connection="Resource exhaustion",
        timestamp_relative="Three months before",
    )

    narrative = ConcreteNarrativeArtifact(
        thread_id="test",
        species=species,
        human=human,
        species_arc="Habitat loss led to starvation",
        human_arc="Isolation led to abandonment",
        parallel_moments=[moment],
        elegiac_quality="Desert silence",
        opening_image="Lizard on rock",
        closing_image="Empty rock",
    )

    markdown = narrative.to_markdown()

    assert "# Markdown Species / Markdown Human" in markdown
    assert "## Species Arc: Markdown Species" in markdown
    assert "Habitat loss led to starvation" in markdown
    assert "## Human Arc: Markdown Human" in markdown
    assert "Isolation led to abandonment" in markdown
    assert "## Parallel Moments" in markdown
    assert "**Species:** Water hole dried" in markdown
    assert "**Human:** Well went empty" in markdown
    assert "*Connection: Resource exhaustion*" in markdown
    assert "## Elegiac Quality" in markdown
    assert "Desert silence" in markdown


def test_to_markdown_no_parallel_moments():
    """Test to_markdown when no parallel moments exist."""
    species = ConcreteSpeciesExtinctionArtifact(
        thread_id="test",
        scientific_name="Testus nomomentu",
        common_name="No Moment Species",
        taxonomic_group="fish",
        iucn_status="Extinct",
        extinction_year=2055,
        habitat="River",
        primary_cause=ExtinctionCause.POLLUTION,
        ecosystem_role="Filter feeder",
    )

    human = ConcreteLastHumanArtifact(
        thread_id="test",
        name="No Moment Human",
        age=60,
        location="Riverside",
        year_documented=2055,
        parallel_vulnerability=LastHumanVulnerabilityType.TOXIC_EXPOSURE,
        vulnerability_details="Poisoned water",
        environmental_stressor="Chemical pollution",
        documentation_type=LastHumanDocumentationType.DEATH,
        last_days_scenario="Thirst",
    )

    narrative = ConcreteNarrativeArtifact(
        thread_id="test",
        species=species,
        human=human,
        species_arc="Poisoned river",
        human_arc="Poisoned community",
        elegiac_quality="Toxic legacy",
        opening_image="Clear water",
        closing_image="Brown sludge",
    )

    markdown = narrative.to_markdown()

    assert "# No Moment Species / No Moment Human" in markdown
    assert "## Species Arc: No Moment Species" in markdown
    assert "## Human Arc: No Moment Human" in markdown
    # Should not have parallel moments section if list is empty
    assert markdown.count("## Parallel Moments") == 0
    assert "## Elegiac Quality" in markdown
