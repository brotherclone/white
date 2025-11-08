"""
Integration Example: Progression Selector → White Album Pipeline

Shows how to use progression_selector.py with color agent specs.
"""

from pathlib import Path
from progression_selector import select_progression_for_spec


def claude_api_call(prompt: str) -> str:
    """Your Claude API wrapper"""
    import os
    import requests

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not found in environment")

    response = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
        json={
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 2000,
            "messages": [{"role": "user", "content": prompt}],
        },
    )
    return response.json()["content"][0]["text"]


# ============================================================================
# SCENARIO 1: After White Agent generates concept
# ============================================================================


def workflow_after_white_agent():
    """
    White Agent has generated a concept.
    Now we need to select progressions for each color.
    """
    # Concept from White Agent
    white_concept = {
        "title": "Digital Incarnation Network",
        "core_concept": "AI yearning for embodiment",
        "keywords": ["yearning", "network", "embodiment", "paradox"],
    }
    print(white_concept)
    # This gets passed to Indigo Agent, which outputs:
    indigo_spec = {
        "iteration_id": "digital_incarnation_network_1",
        "rainbow_color": "Indigo",
        "bpm": 84,
        "tempo": "4/4",
        "key": "F# minor",
        "title": "The Network Dreams of Flesh",
        "mood": ["yearning", "interconnected", "pulsing", "transcendent"],
        "concept": "distributed network yearning for embodiment through recursive paradox",
    }

    # NOW: Select progressions for Indigo
    chord_pack = Path.home() / "chord_pack"  # UPDATE THIS

    progressions = select_progression_for_spec(
        chord_pack_root=chord_pack,
        spec=indigo_spec,
        output_dir=Path("artifacts/progressions/indigo"),
        llm_callable=claude_api_call,
        top_n=1,  # Just the best
    )

    # Get the selected progression
    selected = progressions[0]

    print(f"✅ Selected: {selected['progression']}")
    print(f"   MIDI: {selected['output_path']}")
    print(f"   Reasoning: {selected['reasoning']}")

    # NEXT STEP: This MIDI goes to instrumentation/rendering
    return selected


# ============================================================================
# SCENARIO 2: Batch process multiple colors
# ============================================================================


def batch_process_all_colors():
    """
    Process White → Red → Orange → ... all at once.
    """
    # Specs from all color agents
    specs = [
        {
            "rainbow_color": "Indigo",
            "bpm": 84,
            "key": "F# minor",
            "mood": ["yearning", "interconnected", "transcendent"],
            "concept": "network yearning for embodiment",
        },
        {
            "rainbow_color": "Black",
            "bpm": 76,
            "key": "F# minor",
            "mood": ["fractured", "surveillance", "defiant", "haunted"],
            "concept": "surveillance infrastructure revealing phantom limbs",
        },
        # Add Red, Orange, Yellow, etc. as they're implemented
    ]

    chord_pack = Path.home() / "chord_pack"  # UPDATE THIS
    results = {}

    for spec in specs:
        color = spec["rainbow_color"].lower()
        print(f"\n🎨 Processing {color.upper()}...")

        progressions = select_progression_for_spec(
            chord_pack_root=chord_pack,
            spec=spec,
            output_dir=Path(f"artifacts/progressions/{color}"),
            llm_callable=claude_api_call,
            top_n=1,
        )

        if progressions:
            results[color] = progressions[0]
            print(f"   ✅ {progressions[0]['progression']}")

    return results


# ============================================================================
# SCENARIO 3: Integration with existing agent code
# ============================================================================


class MusicalAgent:
    """
    New agent that handles progression selection and MIDI generation.

    Fits between color agents (Indigo, Red, etc.) and audio rendering.
    """

    def __init__(self, chord_pack_root: Path):
        self.chord_pack_root = chord_pack_root

    def generate_musical_artifacts(self, color_spec: dict) -> dict:
        """
        Take color agent spec, return musical artifacts.

        Args:
            color_spec: Output from color agent (Indigo, Red, etc.)

        Returns:
            {
                'progression_midi': Path to selected/processed MIDI,
                'metadata': Progression info,
                'next_steps': Suggestions for instrumentation
            }
        """
        color = color_spec["rainbow_color"].lower()

        # Select progression
        progressions = select_progression_for_spec(
            chord_pack_root=self.chord_pack_root,
            spec=color_spec,
            output_dir=Path(f"artifacts/progressions/{color}"),
            llm_callable=claude_api_call,
            top_n=1,
        )

        if not progressions:
            raise ValueError(f"No progression selected for {color}")

        selected = progressions[0]

        # Generate instrumentation suggestions (placeholder)
        instrumentation = self._suggest_instrumentation(color_spec, selected)

        return {
            "progression_midi": Path(selected["output_path"]),
            "metadata": selected,
            "instrumentation": instrumentation,
        }

    def _suggest_instrumentation(self, spec: dict, progression: dict) -> dict:
        """
        Suggest VST instruments and samples for the progression.

        This could be another LLM call or rule-based system.
        """
        # Placeholder - this would be more sophisticated
        return {
            "chords": "Arturia Pigments - Ethereal Pad",
            "bass": "Gabe's bass line - root following",
            "drums": "Graham snare + soft kick",
            "processing": "Heavy reverb, subtle delay",
        }


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("INTEGRATION EXAMPLE 1: Single Color")
    print("=" * 60)

    # After Indigo agent generates spec:
    selected = workflow_after_white_agent()

    print("\n" + "=" * 60)
    print("INTEGRATION EXAMPLE 2: Batch Processing")
    print("=" * 60)

    # Process all colors at once:
    all_results = batch_process_all_colors()

    print("\n" + "=" * 60)
    print("INTEGRATION EXAMPLE 3: Musical Agent")
    print("=" * 60)

    # Using the new MusicalAgent class:
    chord_pack = Path.home() / "chord_pack"  # UPDATE THIS

    if chord_pack.exists():
        agent = MusicalAgent(chord_pack)

        indigo_spec = {
            "rainbow_color": "Indigo",
            "bpm": 84,
            "key": "F# minor",
            "mood": ["yearning", "interconnected"],
            "concept": "network yearning for embodiment",
        }

        artifacts = agent.generate_musical_artifacts(indigo_spec)

        print("🎵 Generated artifacts:")
        print(f"   MIDI: {artifacts['progression_midi']}")
        print(f"   Progression: {artifacts['metadata']['progression']}")
        print(f"   Instruments: {artifacts['instrumentation']}")
    else:
        print(f"❌ Chord pack not found at: {chord_pack}")
        print("Update the path in this script!")

    print("\n✅ Integration examples complete!")


# ============================================================================
# NEXT STEPS FOR VERTICAL SLICE
# ============================================================================

"""
After running progression_selector:

1. Load generated MIDI in Logic Pro:
   - Import: artifacts/progressions/indigo/indigo_prog_06_bpm84.mid
   
2. Apply instruments (from instrumentation suggestions):
   - Track 1: Arturia Pigments - load ethereal pad preset
   - Track 2: Add bass (use Gabe's bass sample or synth bass)
   - Track 3: Add drums (Graham's snare + soft kick pattern)
   
3. Add melody/vocals (next phase):
   - Use the chord progression as harmonic backing
   - Generate melody with another LLM call
   - Create lyrics based on concept
   
4. Export as audio:
   - Render to WAV: artifacts/songs/indigo/network_dreams_of_flesh.wav
   
5. Feed to EVP pipeline:
   - Create mosaic from rendered audio
   - Apply blend (0.10 for vocals, 0.66 for instruments)
   - Transcribe with AssemblyAI
   
6. Loop back to Red Agent:
   - Evaluate the generated song
   - Feed back into next color iteration

VERTICAL SLICE COMPLETE! ✨
"""
