"""
Progression Selector for White Album Project

Navigates curated chord pack, uses LLM to rank progressions by mood,
applies tempo from color agent specs.
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import mido


# ============================================================================
# KEY MAPPING: Map musical keys to chord pack folder names
# ============================================================================

KEY_TO_FOLDER = {
    # Major keys (map to folder number)
    "C Major": "01 - C Major - A Minor",
    "C# Major": "02 - C# Major - A# Minor",
    "Db Major": "02 - C# Major - A# Minor",
    "D Major": "03 - D Major - B Minor",
    "D# Major": "04 - D# Major - C Minor",
    "Eb Major": "04 - D# Major - C Minor",
    "E Major": "05 - E Major - C# Minor",
    "F Major": "06 - F Major - D Minor",
    "F# Major": "07 - F# Major - D# Minor",
    "Gb Major": "07 - F# Major - D# Minor",
    "G Major": "08 - G Major - E Minor",
    "G# Major": "09 - G# Major - F Minor",
    "Ab Major": "09 - G# Major - F Minor",
    "A Major": "10 - A Major - F# Minor",
    "A# Major": "11 - A# Major - G Minor",
    "Bb Major": "11 - A# Major - G Minor",
    "B Major": "12 - B Major - G# Minor",
    # Minor keys (map to same folders via relative major)
    "A minor": "01 - C Major - A Minor",
    "A Minor": "01 - C Major - A Minor",
    "A# minor": "02 - C# Major - A# Minor",
    "A# Minor": "02 - C# Major - A# Minor",
    "Bb minor": "02 - C# Major - A# Minor",
    "Bb Minor": "02 - C# Major - A# Minor",
    "B minor": "03 - D Major - B Minor",
    "B Minor": "03 - D Major - B Minor",
    "C minor": "04 - D# Major - C Minor",
    "C Minor": "04 - D# Major - C Minor",
    "C# minor": "05 - E Major - C# Minor",
    "C# Minor": "05 - E Major - C# Minor",
    "Db minor": "05 - E Major - C# Minor",
    "Db Minor": "05 - E Major - C# Minor",
    "D minor": "06 - F Major - D Minor",
    "D Minor": "06 - F Major - D Minor",
    "D# minor": "07 - F# Major - D# Minor",
    "D# Minor": "07 - F# Major - D# Minor",
    "Eb minor": "07 - F# Major - D# Minor",
    "Eb Minor": "07 - F# Major - D# Minor",
    "E minor": "08 - G Major - E Minor",
    "E Minor": "08 - G Major - E Minor",
    "F minor": "09 - G# Major - F Minor",
    "F Minor": "09 - G# Major - F Minor",
    "F# minor": "10 - A Major - F# Minor",
    "F# Minor": "10 - A Major - F# Minor",
    "Gb minor": "10 - A Major - F# Minor",
    "Gb Minor": "10 - A Major - F# Minor",
    "G minor": "11 - A# Major - G Minor",
    "G Minor": "11 - A# Major - G Minor",
    "G# minor": "12 - B Major - G# Minor",
    "G# Minor": "12 - B Major - G# Minor",
    "Ab minor": "12 - B Major - G# Minor",
    "Ab Minor": "12 - B Major - G# Minor",
}


# ============================================================================
# PROGRESSION PARSING
# ============================================================================


def parse_progression_filename(filename: str) -> Optional[Dict]:
    """
    Parse progression filename to extract metadata.

    Example: "Minor Prog 06 (im11-VImaj9-IIImaj9-ivm9-im11-VImaj9-IIImaj9-bIImaj9).mid"

    Returns:
        {
            'mode': 'Minor',
            'number': '06',
            'progression': 'im11-VImaj9-IIImaj9-ivm9-im11-VImaj9-IIImaj9-bIImaj9',
            'filename': 'Minor Prog 06 (...).mid'
        }
    """
    # Match pattern: (Major|Minor) Prog NN (chord-progression).mid
    match = re.match(r"(Major|Minor) Prog (\d+) \((.*?)\)\.mid", filename)

    if not match:
        return None

    return {
        "mode": match.group(1),
        "number": match.group(2),
        "progression": match.group(3),
        "filename": filename,
    }


def analyze_progression_features(progression_str: str) -> Dict:
    """
    Analyze progression string for musical features.

    Features detected:
    - Borrowed chords (bII, bIII, bVI, bVII)
    - Extensions (maj9, m11, add9, 13, etc.)
    - Altered chords (V7alt, dim7, b5)
    - Circular structure (starts/ends on i/I)
    """
    chords = progression_str.split("-")

    features = {
        "has_borrowed_chords": False,
        "borrowed_count": 0,
        "has_extensions": False,
        "extension_types": set(),
        "has_altered": False,
        "is_circular": False,
        "chord_count": len(chords),
        "borrowed_chords": [],
    }

    # Check for borrowed chords
    for chord in chords:
        if chord.startswith("b"):
            features["has_borrowed_chords"] = True
            features["borrowed_count"] += 1
            features["borrowed_chords"].append(chord)

    # Check for extensions
    extension_patterns = [
        "maj9",
        "maj7",
        "m11",
        "m9",
        "m7",
        "add9",
        "add11",
        "6",
        "13",
        "9",
        "7",
    ]
    for pattern in extension_patterns:
        if pattern in progression_str:
            features["has_extensions"] = True
            features["extension_types"].add(pattern)

    # Check for altered chords
    altered_patterns = ["alt", "dim", "b5", "#5", "b9", "#9"]
    for pattern in altered_patterns:
        if pattern in progression_str:
            features["has_altered"] = True

    # Check if circular (starts and ends on tonic)
    first_chord = chords[0].lower()
    last_chord = chords[-1].lower()
    if (first_chord.startswith("i") and last_chord.startswith("i")) or (
        first_chord.startswith("i") and not first_chord.startswith("ii")
    ):
        features["is_circular"] = True

    return features


# ============================================================================
# NAVIGATION
# ============================================================================


def find_key_folder(chord_pack_root: Path, key: str) -> Optional[Path]:
    """
    Find the folder for a given key.

    Args:
        chord_pack_root: Path to chord pack root directory
        key: Musical key (e.g., "F# minor", "C Major")

    Returns:
        Path to key folder, or None if not found
    """
    folder_name = KEY_TO_FOLDER.get(key)

    if not folder_name:
        # Try case-insensitive match
        for k, v in KEY_TO_FOLDER.items():
            if k.lower() == key.lower():
                folder_name = v
                break

    if not folder_name:
        return None

    key_folder = chord_pack_root / folder_name

    if not key_folder.exists():
        return None

    return key_folder


def select_progression_complexity(mood: List[str]) -> str:
    """
    Determine whether to use diatonic or advanced progressions based on mood.

    Advanced mood indicators:
    - yearning, transcendent, ethereal, liminal, haunted
    - fractured, defiant, surveillance

    Diatonic mood indicators:
    - simple, direct, pure, clear
    """
    advanced_keywords = [
        "yearning",
        "transcendent",
        "ethereal",
        "liminal",
        "haunted",
        "fractured",
        "defiant",
        "surveillance",
        "melancholic",
        "introspective",
        "complex",
        "sophisticated",
        "nuanced",
    ]

    mood_lower = [m.lower() for m in mood]

    for keyword in advanced_keywords:
        if keyword in mood_lower:
            return "advanced"

    return "diatonic"


def get_progression_folder(
    key_folder: Path, mode: str, complexity: str
) -> Optional[Path]:
    """
    Navigate to the appropriate progression folder.

    Args:
        key_folder: Path to key folder (e.g., "10 - A Major - F# Minor")
        mode: "Major" or "Minor"
        complexity: "diatonic" or "advanced"

    Returns:
        Path to progression folder
    """
    progressions_root = key_folder / "4 Progressions"

    if not progressions_root.exists():
        return None

    if complexity == "advanced":
        complexity_folder = progressions_root / "2 Advanced Progressions"
    else:
        complexity_folder = progressions_root / "1 Diatonic Triads"

    if not complexity_folder.exists():
        return None

    mode_folder = complexity_folder / f"{mode} Progressions"

    if not mode_folder.exists():
        return None

    return mode_folder


# ============================================================================
# LLM RANKING
# ============================================================================


def create_ranking_prompt(
    progressions: List[Dict], mood: List[str], concept: str
) -> str:
    """
    Create LLM prompt for ranking progressions by mood fit.
    """
    prompt = f"""You are a music theorist and composer analyzing chord progressions for emotional/conceptual fit.

TASK: Rank these chord progressions for the following mood and concept:

MOOD: {', '.join(mood)}
CONCEPT: {concept}

PROGRESSIONS TO RANK:
"""

    for i, prog in enumerate(progressions, 1):
        prompt += f"\n{i}. {prog['mode']} Prog {prog['number']}: {prog['progression']}"

    prompt += """

MUSICAL CONSIDERATIONS:
- Borrowed chords (bII, bIII, bVI, bVII) = yearning, transcendence, modal color
- Circular progressions (start/end on i/I) = interconnection, cyclical nature
- Extensions (maj9, m11, add9, 13) = ethereal, complex, sophisticated quality
- Altered chords (V7alt, dim7, b5) = tension, fractured, defiant, unsettling
- Modal interchange (mixing major/minor quality) = liminal, haunted, ambiguous
- Chromatic mediants (#vi, bIII) = dramatic shifts, surveillance, otherworldly

Return ONLY valid JSON (no markdown, no code blocks) in this exact format:
{
  "ranked": [
    {
      "number": "06",
      "progression": "im11-VImaj9-IIImaj9-ivm9-im11-VImaj9-IIImaj9-bIImaj9",
      "score": 95,
      "reasoning": "Perfect circular structure with im11 returning. Cascade of major chords (VImaj9, IIImaj9) creates transcendence. Terminal bIImaj9 is Neapolitan borrowed chord = yearning quality. Extensions (m11, maj9) = ethereal sophistication."
    }
  ]
}

Rank ALL progressions from best to worst fit. Each must have number, progression, score (0-100), and detailed reasoning."""

    return prompt


def llm_rank_progressions(
    candidates: List[Path], mood: List[str], concept: str, llm_callable: callable
) -> List[Tuple[Path, Dict]]:
    """
    Use LLM to rank progression candidates by mood fit.

    Args:
        candidates: List of MIDI file paths
        mood: List of mood descriptors
        concept: Conceptual description
        llm_callable: Function to call LLM (takes prompt, returns response)

    Returns:
        List of (path, ranking_info) tuples, sorted by rank
    """
    # Parse all filenames
    parsed = []
    for candidate in candidates:
        info = parse_progression_filename(candidate.name)
        if info:
            parsed.append((candidate, info))

    if not parsed:
        return []

    # Extract just the parsed info for prompt
    progression_infos = [info for _, info in parsed]

    # Create prompt
    prompt = create_ranking_prompt(progression_infos, mood, concept)

    # Call LLM
    response = llm_callable(prompt)

    # Parse JSON response
    try:
        # Strip markdown code blocks if present
        response_text = response.strip()
        if response_text.startswith("```"):
            # Remove markdown code blocks
            lines = response_text.split("\n")
            response_text = "\n".join(lines[1:-1]) if len(lines) > 2 else response_text
            if response_text.startswith("json"):
                response_text = response_text[4:].strip()

        ranking_data = json.loads(response_text)
    except json.JSONDecodeError as e:
        print(f"Failed to parse LLM response as JSON: {e}")
        print(f"Response: {response[:500]}")
        return [(path, info) for path, info in parsed]  # Return unranked

    # Map rankings back to file paths
    ranked_results = []
    for ranked_item in ranking_data["ranked"]:
        # Find matching progression
        for path, info in parsed:
            if info["number"] == ranked_item["number"]:
                ranked_results.append(
                    (
                        path,
                        {
                            **info,
                            "score": ranked_item.get("score", 0),
                            "reasoning": ranked_item.get("reasoning", ""),
                        },
                    )
                )
                break

    return ranked_results


# ============================================================================
# TEMPO APPLICATION
# ============================================================================


def set_midi_tempo(midi: mido.MidiFile, bpm: int) -> mido.MidiFile:
    """
    Set tempo for a MIDI file.

    Args:
        midi: MidiFile object
        bpm: Beats per minute

    Returns:
        Modified MidiFile
    """
    # Calculate microseconds per beat
    microseconds_per_beat = int(60_000_000 / bpm)

    # Find or create tempo track
    tempo_set = False
    for track in midi.tracks:
        for i, msg in enumerate(track):
            if msg.type == "set_tempo":
                # Replace existing tempo
                track[i] = mido.MetaMessage(
                    "set_tempo", tempo=microseconds_per_beat, time=msg.time
                )
                tempo_set = True
                break
        if tempo_set:
            break

    # If no tempo found, add to first track
    if not tempo_set and len(midi.tracks) > 0:
        midi.tracks[0].insert(
            0, mido.MetaMessage("set_tempo", tempo=microseconds_per_beat, time=0)
        )

    return midi


def apply_tempo_to_progression(midi_path: Path, bpm: int, output_path: Path) -> Path:
    """
    Load MIDI, set tempo, save to output.

    Args:
        midi_path: Input MIDI file
        bpm: Beats per minute from color spec
        output_path: Where to save modified MIDI

    Returns:
        Path to output file
    """
    midi = mido.MidiFile(midi_path)
    midi = set_midi_tempo(midi, bpm)

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    midi.save(output_path)
    return output_path


# ============================================================================
# MAIN SELECTION WORKFLOW
# ============================================================================


def select_progression_for_spec(
    chord_pack_root: Path,
    spec: Dict,
    output_dir: Path,
    llm_callable: callable,
    top_n: int = 3,
) -> List[Dict]:
    """
    Complete workflow: find, rank, and apply tempo to progressions.

    Args:
        chord_pack_root: Path to chord pack root
        spec: Color agent spec with keys: 'key', 'bpm', 'mood', 'concept', 'rainbow_color'
        output_dir: Where to save processed MIDI files
        llm_callable: Function to call LLM for ranking
        top_n: Number of top progressions to return

    Returns:
        List of dictionaries with selected progression info
    """
    # Extract spec info
    key = spec["key"]
    bpm = spec["bpm"]
    mood = spec["mood"]
    concept = spec.get("concept", "")
    color = spec.get("rainbow_color", "unknown")

    # Determine mode from key
    mode = "Minor" if "minor" in key.lower() else "Major"

    print(f"🎵 Selecting progression for {key} ({mode})")
    print(f"   Mood: {mood}")
    print(f"   BPM: {bpm}")

    # Step 1: Find key folder
    key_folder = find_key_folder(chord_pack_root, key)
    if not key_folder:
        print(f"❌ Key folder not found for: {key}")
        return []

    print(f"✅ Found key folder: {key_folder.name}")

    # Step 2: Determine complexity
    complexity = select_progression_complexity(mood)
    print(f"   Complexity: {complexity}")

    # Step 3: Get progression folder
    prog_folder = get_progression_folder(key_folder, mode, complexity)
    if not prog_folder:
        print("❌ Progression folder not found")
        return []

    print(f"✅ Progression folder: {prog_folder.relative_to(chord_pack_root)}")

    # Step 4: Load all progression files
    candidates = list(prog_folder.glob("*.mid"))
    print(f"   Found {len(candidates)} candidate progressions")

    if not candidates:
        return []

    # Step 5: Rank with LLM
    print("🤖 Ranking progressions with LLM...")
    ranked = llm_rank_progressions(candidates, mood, concept, llm_callable)

    if not ranked:
        print("❌ No progressions ranked")
        return []

    # Step 6: Take top N
    top_progressions = ranked[:top_n]

    results = []
    for i, (path, info) in enumerate(top_progressions, 1):
        print(f"\n{'='*60}")
        print(f"🏆 Rank {i}: {info['mode']} Prog {info['number']}")
        print(f"   Progression: {info['progression']}")
        print(f"   Score: {info.get('score', 'N/A')}")
        print(f"   Reasoning: {info.get('reasoning', 'N/A')}")

        # Apply tempo
        output_filename = f"{color.lower()}_prog_{info['number']}_bpm{bpm}.mid"
        output_path = output_dir / output_filename

        print(f"⚙️  Applying tempo {bpm} BPM...")
        processed_path = apply_tempo_to_progression(path, bpm, output_path)

        print(f"✅ Saved: {processed_path}")

        results.append(
            {
                "rank": i,
                "original_path": str(path),
                "output_path": str(processed_path),
                "mode": info["mode"],
                "number": info["number"],
                "progression": info["progression"],
                "score": info.get("score"),
                "reasoning": info.get("reasoning"),
                "bpm": bpm,
                "key": key,
            }
        )

    return results


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Example color spec (from Session 26 Indigo)
    example_spec = {
        "rainbow_color": "Indigo",
        "bpm": 84,
        "key": "F# minor",
        "mood": ["yearning", "interconnected", "pulsing", "transcendent"],
        "concept": "distributed network of interconnected processes yearning for embodiment",
    }

    # Mock LLM callable for testing
    def mock_llm(prompt):
        # In real usage, this would call Claude/GPT
        return json.dumps(
            {
                "ranked": [
                    {
                        "number": "06",
                        "progression": "im11-VImaj9-IIImaj9-ivm9-im11-VImaj9-IIImaj9-bIImaj9",
                        "score": 95,
                        "reasoning": "Circular structure with transcendent borrowed chord",
                    }
                ]
            }
        )

    # Run selection
    chord_pack_root = Path("/path/to/chord/pack")  # UPDATE THIS
    output_dir = Path("./output/progressions")

    if chord_pack_root.exists():
        results = select_progression_for_spec(
            chord_pack_root, example_spec, output_dir, mock_llm, top_n=3
        )

        print("\n" + "=" * 60)
        print("SELECTION COMPLETE")
        print("=" * 60)
        for result in results:
            print(
                f"{result['rank']}. {result['progression']} → {result['output_path']}"
            )
    else:
        print(f"Chord pack not found at: {chord_pack_root}")
