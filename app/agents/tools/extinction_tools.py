import json
import os
from dotenv import load_dotenv

from app.structures.artifacts.species_extinction_artifact import (
    SpeciesExtinctionArtifact,
)
from app.structures.concepts.population_data import PopulationData
from app.structures.enums.extinction_cause import ExtinctionCause

load_dotenv()


def load_green_corpus(corpus_path: str = os.getenv("GREEN_CORPUS_DIR")):
    """Load the extinction corpus from JSON"""
    with open(corpus_path) as f:
        data = json.load(f)
    return data


def get_random_species(species: dict) -> SpeciesExtinctionArtifact:
    """Get a random species from the corpus"""
    c = len(species["entries"])
    index = species["random"].randint(0, c - 1)
    return parse_species_entry(corpus["entries"][index])


def parse_species_entry(entry_dict: dict) -> SpeciesExtinctionArtifact:
    """Parse a corpus entry into a SpeciesExtinction model"""
    species_data = entry_dict["species"]
    population_trajectory = [
        PopulationData(**pop_data) for pop_data in species_data["population_trajectory"]
    ]
    primary_cause = ExtinctionCause(species_data["primary_cause"])
    species = SpeciesExtinctionArtifact(
        scientific_name=species_data["scientific_name"],
        common_name=species_data["common_name"],
        taxonomic_group=species_data["taxonomic_group"],
        iucn_status=species_data["iucn_status"],
        extinction_year=species_data["extinction_year"],
        population_trajectory=population_trajectory,
        habitat=species_data["habitat"],
        endemic=species_data["endemic"],
        range_km2=species_data.get("range_km2"),
        primary_cause=primary_cause,
        secondary_causes=species_data["secondary_causes"],
        anthropogenic_factors=species_data["anthropogenic_factors"],
        cascade_effects=species_data["cascade_effects"],
        ecosystem_role=species_data["ecosystem_role"],
        symbolic_resonance=species_data["symbolic_resonance"],
        human_parallel_hints=species_data["human_parallel_hints"],
        narrative_potential_score=species_data["narrative_potential_score"],
        symbolic_weight=species_data["symbolic_weight"],
        size_category=species_data["size_category"],
        lifespan_years=species_data.get("lifespan_years"),
        movement_pattern=species_data.get("movement_pattern"),
    )
    return species


if __name__ == "__main__":
    corpus = load_green_corpus()
    print(len(corpus["entries"]))
