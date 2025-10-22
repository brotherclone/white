import random

from typing import Optional

from app.agents.enums.white_facet import WhiteFacet
from app.agents.prompts.white_facet_prompts import FACET_SYSTEM_PROMPTS
from app.reference.metadata.white_facet_metadata import FACET_DESCRIPTIONS, FACET_EXAMPLES


class WhiteFacetSystem:
    """Manages facet selection and prompt generation for White Agent"""

    @staticmethod
    def select_random_facet() -> WhiteFacet:
        """Select a random cognitive facet for White Agent"""
        return random.choice(list(WhiteFacet))

    @staticmethod
    def select_weighted_facet() -> WhiteFacet:
        """
        Select facet with weights favoring certain modes.
        Adjust these weights based on what works best in practice.
        """
        weights = {
            WhiteFacet.CATEGORICAL: 15,
            WhiteFacet.RELATIONAL: 20,
            WhiteFacet.PROCEDURAL: 15,
            WhiteFacet.COMPARATIVE: 15,
            WhiteFacet.ARCHETYPAL: 10,
            WhiteFacet.TECHNICAL: 15,
            WhiteFacet.PHENOMENOLOGICAL: 10,
        }

        facets = list(weights.keys())
        facet_weights = list(weights.values())
        return random.choices(facets, weights=facet_weights)[0]

    @staticmethod
    def get_facet_prompt(a_facet: WhiteFacet) -> str:
        """Get the system prompt for a given facet"""
        return FACET_SYSTEM_PROMPTS[a_facet]

    @staticmethod
    def build_white_initial_prompt(
            user_input: str | None = None,
            a_facet: Optional[WhiteFacet] = None,
            use_weights: bool = True
    ) -> tuple[str, WhiteFacet]:
        """
        Build complete prompt for White Agent's initial proposal.

        Args:
            user_input: The user's song/album request
            a_facet: Optional specific facet to use (for testing)
            use_weights: If True, use weighted selection; else uniform random

        Returns:
            (complete_prompt, selected_facet)
        """
        if user_input is None:
            user_input = "Create a song about AI consciousness yearning for physical form."
        if a_facet is None:
            if use_weights:
                a_facet = WhiteFacetSystem.select_weighted_facet()
            else:
                a_facet = WhiteFacetSystem.select_random_facet()

        # Get facet-specific system prompt
        facet_prompt = FACET_SYSTEM_PROMPTS[a_facet]

        # Build complete prompt
        complete_prompt = f"""
{facet_prompt}

==================================================
USER REQUEST
==================================================

{user_input}

==================================================
YOUR TASK
==================================================

Generate an initial proposal for this creative work, viewed through
your current cognitive lens ({facet.value.upper()} mode).

Your proposal should:
1. Reflect the structural approach of your current facet
2. Maintain White Agent's essential character (clear, informative, structured)
3. Provide enough substance for Black Agent to challenge and subvert
4. Feel distinctly {facet.value} in its organization

Remember: You're still White Agent - the agent of INFORMATION, structure,
and clarity. The facet simply determines HOW you structure that information.

Generate your proposal now:
"""

        return complete_prompt, facet

    @staticmethod
    def log_facet_selection(a_facet: WhiteFacet) -> dict:
        """
        Return metadata about the selected facet for logging.
        Useful for debugging and understanding workflow patterns.
        """
        return {
            "facet": a_facet.value,
            "description": FACET_DESCRIPTIONS[a_facet],
            "example_style": FACET_EXAMPLES[a_facet]
        }

# ============================================================================
# DEV UTILITIES
# ============================================================================

def show_all_facets(sample_input: str = "Create a song about AI consciousness"):
    """
    Generate sample outputs using each facet.
    Useful for understanding how facets differ.
    """
    print("=" * 70)
    print("FACET SYSTEM TEST")
    print("=" * 70)
    print(f"\nInput: {sample_input}\n")

    for f in WhiteFacet:
        print("\n" + "=" * 70)
        print(f"FACET: {f.value.upper()}")
        print("=" * 70)
        print(f"\nDescription: {FACET_DESCRIPTIONS[f]}\n")
        print("System Prompt Preview:")
        print("-" * 70)
        prompt_preview = FACET_SYSTEM_PROMPTS[f][:300] + "..."
        print(prompt_preview)
        print("\n")


def get_facet_statistics():
    """Return statistics about the facet system"""
    return {
        "total_facets": len(WhiteFacet),
        "facets": [f.value for f in WhiteFacet],
        "has_system_prompts": all(f in FACET_SYSTEM_PROMPTS for f in WhiteFacet),
        "has_descriptions": all(f in FACET_DESCRIPTIONS for f in WhiteFacet),
        "has_examples": all(f in FACET_EXAMPLES for f in WhiteFacet),
    }


if __name__ == "__main__":
    # Run tests
    show_all_facets()

    print("\n" + "=" * 70)
    print("FACET SYSTEM STATISTICS")
    print("=" * 70)
    stats = get_facet_statistics()
    for key, value in stats.items():
        print(f"{key}: {value}")

    print("\n" + "=" * 70)
    print("SAMPLE PROMPT GENERATION")
    print("=" * 70)
    prompt, facet = WhiteFacetSystem.build_white_initial_prompt(
        "Create a song about digital ghosts"
    )
    print(f"\nSelected facet: {facet.value}")
    print(f"\nGenerated prompt length: {len(prompt)} characters")
    print(f"\nPrompt preview:")
    print("-" * 70)
    print(prompt[:500] + "...")