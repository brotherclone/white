"""
Blue Album Biographical Timeline Tool
PRESENT + PERSON + FORGOTTEN methodology
For LangGraph agent use in exploring alternate timelines and quantum biographical possibilities
"""

import yaml
import random
from typing import Dict, List, Optional, Any
from pathlib import Path
import logging


def load_biographical_data(file_path: str = "/Volumes/LucidNonsense/White/app/reference/biographical/biographical_reference.yml") -> Dict[str, Any]:
    """Load biographical timeline data from YAML file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        logging.error(f"Biographical data file not found: {file_path}")
        return {"years": {}, "quantum_analysis_prompts": {}, "song_inspiration_templates": {}}
    except yaml.YAMLError as e:
        logging.error(f"Error parsing YAML file: {e}")
        return {"years": {}, "quantum_analysis_prompts": {}, "song_inspiration_templates": {}}


def get_year_analysis(year: int, biographical_data: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Generate quantum biographical analysis for a specific year.

    Args:
        year: Year to analyze (e.g., 1993, 2001)
        biographical_data: Preloaded biographical data (optional)

    Returns:
        Dictionary containing:
        - year_data: Historical and personal context
        - what_if_scenarios: Global and personal alternate possibilities
        - cascade_analysis: How changes might ripple through time
        - song_inspiration: Blue album creative concepts
        - quantum_metrics: Rebracketing and choice-point analysis
    """

    if biographical_data is None:
        biographical_data = load_biographical_data()

    year_str = str(year)

    if year_str not in biographical_data.get("years", {}):
        return {
            "error": f"No biographical data found for year {year}",
            "suggestion": f"Add {year} to biographical_timeline.yml with world_events and personal_context"
        }

    year_data = biographical_data["years"][year_str]
    prompts = biographical_data.get("quantum_analysis_prompts", {})
    song_templates = biographical_data.get("song_inspiration_templates", {})

    # Generate what-if scenarios
    what_if_scenarios = generate_what_if_scenarios(year_data, prompts)

    # Analyze cascade effects
    cascade_analysis = analyze_cascade_effects(year_data, what_if_scenarios)

    # Generate song inspiration
    song_inspiration = generate_song_inspiration(year, year_data, song_templates)

    # Calculate quantum biographical metrics
    quantum_metrics = calculate_quantum_metrics(year_data)

    return {
        "year": year,
        "year_data": year_data,
        "what_if_scenarios": what_if_scenarios,
        "cascade_analysis": cascade_analysis,
        "song_inspiration": song_inspiration,
        "quantum_metrics": quantum_metrics,
        "blue_album_themes": {
            "mode": "PRESENT + PERSON + FORGOTTEN",
            "concept": "How we 'tape over' our narratives with revised versions",
            "focus": "Identity collapse and false narrative construction"
        }
    }


def generate_what_if_scenarios(year_data: Dict, prompts: Dict) -> Dict[str, List[str]]:
    """Generate alternate timeline scenarios based on year data and prompt templates."""

    world_events = year_data.get("world_events", {})
    personal_context = year_data.get("personal_context", {})
    choice_points = personal_context.get("choice_points", [])

    # Global what-ifs based on actual world events
    global_what_ifs = []
    for category in ["major", "cultural", "technology"]:
        events = world_events.get(category, [])
        for event in events[:2]:  # Limit to avoid overwhelming output
            global_what_ifs.append(f"What if '{event}' had unfolded differently?")

    # Personal what-ifs based on choice points
    personal_what_ifs = []
    for choice_point in choice_points:
        personal_what_ifs.append(f"What if you had chosen differently at: {choice_point}")

    # Add template-based prompts
    template_global = prompts.get("global_what_ifs", [])
    template_personal = prompts.get("personal_what_ifs", [])

    # Mix template prompts with specific ones
    global_what_ifs.extend(random.sample(template_global, min(2, len(template_global))))
    personal_what_ifs.extend(random.sample(template_personal, min(2, len(template_personal))))

    return {
        "global_what_ifs": global_what_ifs,
        "personal_what_ifs": personal_what_ifs
    }


def analyze_cascade_effects(year_data: Dict, what_if_scenarios: Dict) -> Dict[str, Any]:
    """Analyze how alternate choices might cascade through time."""

    personal_context = year_data.get("personal_context", {})
    influences = personal_context.get("influences", [])

    cascade_categories = {
        "creative_development": [
            "How might your artistic voice have evolved differently?",
            "What genres or mediums might you have explored instead?",
            "How would your creative collaborations have changed?"
        ],
        "identity_formation": [
            "What aspects of personality might have developed differently?",
            "How would your worldview and values have shifted?",
            "What 'forgotten' versions of yourself exist in collapsed timelines?"
        ],
        "relationship_networks": [
            "What different social/professional circles might you have joined?",
            "How would key relationships have formed or not formed?",
            "What mentorship or influence patterns might have emerged?"
        ],
        "skill_acquisition": [
            "What different expertise would you have developed?",
            "How would your learning path have diverged?",
            "What capabilities would you have instead of current ones?"
        ]
    }

    # Calculate "rebracketing intensity" - how much revision potential exists
    choice_points = personal_context.get("choice_points", [])
    rebracketing_intensity = len(choice_points) / 5.0  # Normalize to 0-1 scale

    return {
        "cascade_categories": cascade_categories,
        "rebracketing_intensity": min(rebracketing_intensity, 1.0),
        "temporal_malleability": "high" if rebracketing_intensity > 0.6 else "medium" if rebracketing_intensity > 0.3 else "low",
        "narrative_revision_potential": f"This period shows {rebracketing_intensity:.1%} revision potential"
    }


def generate_song_inspiration(year: int, year_data: Dict, song_templates: Dict) -> List[Dict[str, str]]:
    """Generate Blue album song concepts based on biographical analysis."""

    personal_context = year_data.get("personal_context", {})
    emotional_landscape = personal_context.get("emotional_landscape", "")

    inspirations = []

    # Generate songs based on templates
    for template_name, template_data in song_templates.items():
        song_concept = {
            "title": f'"{year} ({template_name.title().replace("_", " ")})"',
            "concept": template_data.get("concept", "").format(year=year),
            "musical_approach": template_data.get("musical_approach", ""),
            "lyrical_approach": template_data.get("lyrical_approach", ""),
            "emotional_source": emotional_landscape
        }
        inspirations.append(song_concept)

    # Add year-specific concept
    year_specific = {
        "title": f'"Frequency {year}"',
        "concept": f"Accessing the specific temporal frequency of {year} through creative collaboration",
        "musical_approach": "Layer period-appropriate sounds with modern production techniques",
        "lyrical_approach": "Present-tense narration that bleeds between then and now",
        "emotional_source": emotional_landscape
    }
    inspirations.append(year_specific)

    return inspirations


def calculate_quantum_metrics(year_data: Dict) -> Dict[str, Any]:
    """Calculate metrics for quantum biographical analysis."""

    personal_context = year_data.get("personal_context", {})
    choice_points = personal_context.get("choice_points", [])
    influences = personal_context.get("influences", [])

    metrics = {
        "choice_point_density": len(choice_points),
        "influence_complexity": len(influences),
        "narrative_malleability": len(choice_points) * 0.2,  # How much revision potential
        "temporal_significance": "high" if len(choice_points) > 3 else "medium" if len(choice_points) > 1 else "low",
        "forgotten_self_potential": f"{len(choice_points) * 2} alternate timeline branches identified"
    }

    # Blue album specific metrics
    metrics["taped_over_coefficient"] = min(metrics["narrative_malleability"], 1.0)
    metrics["identity_collapse_risk"] = "high" if metrics["choice_point_density"] > 4 else "moderate"

    return metrics


def explore_alternate_timeline(year: int, choice_point_index: int = 0) -> Dict[str, Any]:
    """
    Deep dive into a specific alternate timeline branch.

    Args:
        year: Year to explore
        choice_point_index: Which choice point to explore alternate for

    Returns:
        Detailed alternate timeline analysis
    """

    biographical_data = load_biographical_data()
    year_analysis = get_year_analysis(year, biographical_data)

    if "error" in year_analysis:
        return year_analysis

    year_data = year_analysis["year_data"]
    choice_points = year_data.get("personal_context", {}).get("choice_points", [])

    if choice_point_index >= len(choice_points):
        return {
            "error": f"Choice point index {choice_point_index} not found",
            "available_choice_points": choice_points
        }

    selected_choice = choice_points[choice_point_index]

    # Generate detailed alternate timeline
    alternate_timeline = {
        "original_choice_point": selected_choice,
        "alternate_decision": f"Alternative path from: {selected_choice}",
        "immediate_consequences": [
            "Different social circles and relationships",
            "Alternative skill development path",
            "Shifted creative/professional trajectory"
        ],
        "long_term_ripples": [
            "Different geographic locations over time",
            "Alternative creative works and collaborations",
            "Different worldview and value development",
            "Alternative present-day circumstances"
        ],
        "emotional_archaeology": "What feelings and experiences exist in this forgotten timeline?",
        "creative_potential": "How might this alternate self's creative work differ?",
        "blue_album_relevance": f"Perfect material for exploring 'taped over' narratives and forgotten identity possibilities"
    }

    return {
        "year": year,
        "choice_point": selected_choice,
        "alternate_timeline": alternate_timeline,
        "song_concept": {
            "title": f'"The Other {year}"',
            "concept": f"A song comparing the actual {year} with the alternate timeline where {selected_choice} went differently",
            "approach": "Dual-channel audio with 'actual' timeline in left ear, 'alternate' timeline in right ear"
        }
    }


# Tool registration for LangGraph
@tool
def blue_biographical_analysis(year: int) -> str:
    """
    Analyze a specific year using Blue album methodology for quantum biographical exploration.
    Explores PRESENT + PERSON + FORGOTTEN themes through alternate timeline analysis.

    Args:
        year: Year to analyze (must exist in biographical_timeline.yml)

    Returns:
        JSON string containing timeline analysis, what-if scenarios, and song inspiration
    """
    import json
    result = get_year_analysis(year)
    return json.dumps(result, indent=2)


@tool
def explore_choice_point(year: int, choice_index: int = 0) -> str:
    """
    Deep dive into a specific alternate timeline branch from a choice point.

    Args:
        year: Year containing the choice point
        choice_index: Index of the choice point to explore (default 0)

    Returns:
        JSON string containing detailed alternate timeline analysis
    """
    import json
    result = explore_alternate_timeline(year, choice_index)
    return json.dumps(result, indent=2)


# Example usage:
if __name__ == "__main__":
    # Test the tool
    analysis_1993 = get_year_analysis(1993)
    print("=== 1993 Biographical Analysis ===")
    print(
        f"Choice points identified: {len(analysis_1993.get('year_data', {}).get('personal_context', {}).get('choice_points', []))}")
    print(f"Quantum metrics: {analysis_1993.get('quantum_metrics', {})}")

    # Example alternate timeline exploration
    if analysis_1993.get('year_data', {}).get('personal_context', {}).get('choice_points'):
        alternate = explore_alternate_timeline(1993, 0)
        print("\n=== Alternate Timeline Analysis ===")
        print(f"Original choice: {alternate.get('choice_point', 'N/A')}")
        print(f"Song concept: {alternate.get('song_concept', {}).get('title', 'N/A')}")