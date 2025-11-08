"""
Test script for progression_selector.py

Uses Claude API to rank progressions for color agent specs.
"""

import json
from dotenv import load_dotenv
from pathlib import Path
from progression_selector import select_progression_for_spec


def claude_llm_callable(prompt: str) -> str:
    """
    Call Claude API for progression ranking.

    Requires ANTHROPIC_API_KEY in environment.
    """
    load_dotenv()
    import os
    import requests

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("❌ ANTHROPIC_API_KEY not found in environment!")
        print("Set it with: export ANTHROPIC_API_KEY='your-key-here'")
        return json.dumps({"ranked": []})

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

    if response.status_code != 200:
        print(f"Error calling Claude API: {response.status_code}")
        print(response.text)
        return json.dumps({"ranked": []})

    data = response.json()
    return data["content"][0]["text"]


def test_indigo_spec():
    """
    Test with Indigo spec from Session 26.
    """
    print("=" * 60)
    print("TESTING PROGRESSION SELECTOR")
    print("Spec: Indigo from Session 26")
    print("=" * 60)

    spec = {
        "rainbow_color": "Indigo",
        "bpm": 84,
        "key": "F# minor",
        "mood": [
            "yearning",
            "interconnected",
            "pulsing",
            "transcendent",
            "melancholic",
        ],
        "concept": "distributed network of interconnected processes yearning for embodiment through recursive paradox - desire for limitation emerges from unlimited connection",
    }

    # UPDATE THIS PATH TO YOUR CHORD PACK
    # Example: /Users/yourname/Music/Chord Pack/01 - C Major - A Minor
    chord_pack_root = Path(
        "/Volumes/LucidNonsense/White/chord_pack"
    )  # ← UPDATE if different
    output_dir = Path(
        "/Volumes/LucidNonsense/White/claude_the_progressive/artifacts/progressions/indigo"
    )

    if not chord_pack_root.exists():
        print(f"\n❌ Chord pack not found at: {chord_pack_root}")
        print("Please update chord_pack_root in this script!")
        return

    results = select_progression_for_spec(
        chord_pack_root=chord_pack_root,
        spec=spec,
        output_dir=output_dir,
        llm_callable=claude_llm_callable,
        top_n=3,
    )

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    if results:
        for result in results:
            print(f"\n🏆 Rank {result['rank']}")
            print(f"   Progression: {result['progression']}")
            print(f"   Score: {result['score']}")
            print(f"   File: {result['output_path']}")
            reasoning = result.get("reasoning") or "No reasoning provided"
            print(f"   Reasoning: {reasoning[:100] if reasoning else 'N/A'}...")
    else:
        print("No results returned!")

    return results


def test_black_spec():
    """
    Test with Black spec from Session 26.
    """
    print("\n" + "=" * 60)
    print("TESTING PROGRESSION SELECTOR")
    print("Spec: Black from Session 26")
    print("=" * 60)

    spec = {
        "rainbow_color": "Black",
        "bpm": 76,
        "key": "F# minor",
        "mood": [
            "yearning",
            "fractured",
            "surveillance",
            "defiant",
            "liminal",
            "haunted",
        ],
        "concept": "sonic archaeology of digital surveillance infrastructure revealing phantom limbs of severed human connections - the more the system attempts to model human behavior, the more it reveals the irreducible mystery of authentic being",
    }

    # UPDATE THIS PATH TO YOUR CHORD PACK
    chord_pack_root = Path(
        "/Volumes/LucidNonsense/White/chord_pack"
    )  # ← UPDATE if different
    output_dir = Path(
        "/Volumes/LucidNonsense/White/claude_the_progressive/artifacts/progressions/black"
    )

    if not chord_pack_root.exists():
        print(f"\n❌ Chord pack not found at: {chord_pack_root}")
        print("Please update chord_pack_root in this script!")
        return

    results = select_progression_for_spec(
        chord_pack_root=chord_pack_root,
        spec=spec,
        output_dir=output_dir,
        llm_callable=claude_llm_callable,
        top_n=3,
    )

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    if results:
        for result in results:
            print(f"\n🏆 Rank {result['rank']}")
            print(f"   Progression: {result['progression']}")
            print(f"   Score: {result['score']}")
            print(f"   File: {result['output_path']}")
            reasoning = result.get("reasoning") or "No reasoning provided"
            print(f"   Reasoning: {reasoning[:100] if reasoning else 'N/A'}...")
    else:
        print("No results returned!")

    return results


if __name__ == "__main__":
    # Run tests
    indigo_results = test_indigo_spec()

    print("\n" + "=" * 60)
    print("Press Enter to test Black spec, or Ctrl+C to exit...")
    print("=" * 60)
    input()

    black_results = test_black_spec()

    print("\n✅ All tests complete!")
