"""
Chord progression generator using brute-force + scoring approach.
"""

import polars as pl
import pickle
import random
from pathlib import Path
from typing import List, Dict, Optional, Tuple


class ChordProgressionGenerator:
    """
    Generate chord progressions using brute-force sampling and scoring.
    """

    def __init__(
        self,
        data_dir: str = "/Volumes/LucidNonsense/White/app/generators/midi/prototype/data",
    ):
        """Load the chord database and graphs."""
        data_dir = Path(data_dir)

        print("Loading chord database...")
        self.chords_df = pl.read_parquet(data_dir / "chords.parquet")
        self.progressions_df = pl.read_parquet(data_dir / "progressions.parquet")

        print("Loading graphs...")
        with open(data_dir / "chord_transition_graph.pkl", "rb") as f:
            self.chord_graph = pickle.load(f)
        with open(data_dir / "function_transition_graph.pkl", "rb") as f:
            self.function_graph = pickle.load(f)

        print(
            f"✓ Loaded {len(self.chords_df)} chords and {len(self.progressions_df)} progression chords"
        )
        print(
            f"✓ Loaded graphs with {self.chord_graph.number_of_nodes()} chord nodes and {self.function_graph.number_of_nodes()} function nodes"
        )

    def get_chords_in_key(self, key_root: str, mode: str = "Major") -> pl.DataFrame:
        """Get all chords in a specific key."""
        return self.chords_df.filter(
            (pl.col("key_root") == key_root) & (pl.col("mode_in_key") == mode)
        )

    def get_chord_by_function(
        self, key_root: str, mode: str, function: str, category: Optional[str] = None
    ) -> pl.DataFrame:
        """Get chords by roman numeral function."""
        query = (
            (pl.col("key_root") == key_root)
            & (pl.col("mode_in_key") == mode)
            & (pl.col("function") == function)
        )

        if category:
            query = query & (pl.col("category") == category)

        return self.chords_df.filter(query)

    def sample_chord_by_function(
        self, key_root: str, mode: str, function: str, category: Optional[str] = None
    ) -> Optional[Dict]:
        """Sample a random chord by function."""
        candidates = self.get_chord_by_function(key_root, mode, function, category)

        if len(candidates) == 0:
            return None

        return candidates.sample(1).to_dicts()[0]

    def sample_next_chord_from_graph(
        self, current_chord: Dict, key_root: str, mode: str
    ) -> Optional[Dict]:
        """
        Sample next chord using the function transition graph.
        Uses weighted sampling based on transition probabilities.
        """
        # Build node label for current chord
        current_key = f"{key_root}_{mode}"
        current_func = current_chord.get("function")

        if not current_func:
            # If no function, just sample randomly
            return self.get_chords_in_key(key_root, mode).sample(1).to_dicts()[0]

        current_node = f"{current_key}:{current_func}"

        # Check if node exists in graph
        if current_node not in self.function_graph:
            return None

        # Get neighbors with probabilities
        neighbors = list(self.function_graph.neighbors(current_node))
        if not neighbors:
            return None

        # Get transition probabilities
        probabilities = [
            self.function_graph[current_node][neighbor].get("probability", 0.0)
            for neighbor in neighbors
        ]

        # Weighted random choice
        next_node = random.choices(neighbors, weights=probabilities)[0]

        # Extract function from node label
        next_func = next_node.split(":")[1]

        # Sample a chord with this function
        return self.sample_chord_by_function(key_root, mode, next_func)

    def score_progression_melody(self, progression: List[Dict]) -> float:
        """
        Score melodic movement in the progression (highest notes).
        Prefers stepwise motion over large leaps.
        """
        if len(progression) < 2:
            return 0.0

        score = 0.0
        for i in range(len(progression) - 1):
            curr_top = max(progression[i]["midi_notes"])
            next_top = max(progression[i + 1]["midi_notes"])
            interval = abs(next_top - curr_top)

            # Reward stepwise motion
            if interval <= 2:
                score += 1.0
            # Neutral for thirds/fourths
            elif interval <= 5:
                score += 0.5
            # Penalize large leaps
            else:
                score -= 0.2

        return score / (len(progression) - 1)

    def score_progression_voice_leading(self, progression: List[Dict]) -> float:
        """
        Score voice leading quality.
        Rewards small total voice movement between chords.
        """
        if len(progression) < 2:
            return 0.0

        total_movement = 0
        transitions = 0

        for i in range(len(progression) - 1):
            curr_notes = sorted(progression[i]["midi_notes"])
            next_notes = sorted(progression[i + 1]["midi_notes"])

            # Simple heuristic: sum of minimum distances between note sets
            movement = sum(min(abs(c - n) for n in next_notes) for c in curr_notes)
            total_movement += movement
            transitions += 1

        avg_movement = total_movement / transitions if transitions > 0 else 0

        # Normalize: lower movement = higher score
        # Excellent voice leading: ~3 semitones average
        # Poor voice leading: >12 semitones average
        score = max(0, 1.0 - (avg_movement - 3) / 12)
        return score

    def score_progression_variety(self, progression: List[Dict]) -> float:
        """Score chord variety. Penalizes too much repetition."""
        if len(progression) < 2:
            return 0.0

        unique_chords = len(set(c["chord_id"] for c in progression))
        return unique_chords / len(progression)

    def score_progression_graph_probability(
        self, progression: List[Dict], key_root: str, mode: str
    ) -> float:
        """
        Score based on transition probabilities from the function graph.
        Higher score for common progressions from the corpus.
        """
        if len(progression) < 2:
            return 0.0

        key = f"{key_root}_{mode}"
        total_prob = 0.0
        transitions = 0

        for i in range(len(progression) - 1):
            func_from = progression[i].get("function")
            func_to = progression[i + 1].get("function")

            if not func_from or not func_to:
                continue

            node_from = f"{key}:{func_from}"
            node_to = f"{key}:{func_to}"

            if (
                node_from in self.function_graph
                and node_to in self.function_graph.neighbors(node_from)
            ):
                prob = self.function_graph[node_from][node_to].get("probability", 0.0)
                total_prob += prob
                transitions += 1

        return total_prob / transitions if transitions > 0 else 0.0

    def score_progression(
        self,
        progression: List[Dict],
        key_root: str,
        mode: str,
        weights: Optional[Dict[str, float]] = None,
    ) -> float:
        """
        Composite scoring function.
        Combine multiple scoring metrics with configurable weights.
        """
        if not weights:
            weights = {
                "melody": 0.25,
                "voice_leading": 0.3,
                "variety": 0.15,
                "graph_probability": 0.3,
            }

        scores = {
            "melody": self.score_progression_melody(progression),
            "voice_leading": self.score_progression_voice_leading(progression),
            "variety": self.score_progression_variety(progression),
            "graph_probability": self.score_progression_graph_probability(
                progression, key_root, mode
            ),
        }

        # Weighted sum
        total_score = sum(scores[metric] * weight for metric, weight in weights.items())

        return total_score, scores

    def generate_progression_random(
        self,
        key_root: str,
        mode: str = "Major",
        length: int = 4,
        category: Optional[str] = None,
    ) -> List[Dict]:
        """Generate a random chord progression."""
        progression = []

        for _ in range(length):
            candidates = self.get_chords_in_key(key_root, mode)

            if category:
                candidates = candidates.filter(pl.col("category") == category)

            if len(candidates) == 0:
                break

            chord = candidates.sample(1).to_dicts()[0]
            progression.append(chord)

        return progression

    def generate_progression_graph_guided(
        self,
        key_root: str,
        mode: str = "Major",
        length: int = 4,
        start_function: Optional[str] = None,
    ) -> List[Dict]:
        """
        Generate a progression guided by the function transition graph.
        Follows common progressions learned from the corpus.
        """
        progression = []

        # Start chord
        if start_function:
            chord = self.sample_chord_by_function(key_root, mode, start_function)
        else:
            # Default to I chord
            chord = self.sample_chord_by_function(key_root, mode, "I")

        if not chord:
            return []

        progression.append(chord)

        # Generate subsequent chords using graph
        for _ in range(length - 1):
            next_chord = self.sample_next_chord_from_graph(
                progression[-1], key_root, mode
            )

            if next_chord:
                progression.append(next_chord)
            else:
                # Fallback to random if graph fails
                random_chord = (
                    self.get_chords_in_key(key_root, mode).sample(1).to_dicts()[0]
                )
                progression.append(random_chord)

        return progression

    def generate_progression_brute_force(
        self,
        key_root: str,
        mode: str = "Major",
        length: int = 4,
        num_candidates: int = 1000,
        top_k: int = 10,
        weights: Optional[Dict[str, float]] = None,
        use_graph: bool = True,
    ) -> List[Tuple[float, List[Dict], Dict[str, float]]]:
        """
        Brute-force generate many progressions and return top-scoring ones.

        Args:
            key_root: Root of the key (e.g., 'C')
            mode: 'Major' or 'Minor'
            length: Number of chords in progression
            num_candidates: Number of random progressions to generate
            top_k: Number of top progressions to return
            weights: Scoring weights
            use_graph: Use graph-guided generation vs pure random

        Returns:
            List of (score, progression, score_breakdown) tuples
        """
        candidates = []

        for _ in range(num_candidates):
            # Generate progression
            if use_graph:
                prog = self.generate_progression_graph_guided(key_root, mode, length)
            else:
                prog = self.generate_progression_random(key_root, mode, length)

            if len(prog) < length:
                continue

            # Score it
            total_score, score_breakdown = self.score_progression(
                prog, key_root, mode, weights
            )
            candidates.append((total_score, prog, score_breakdown))

        # Sort by score and return top K
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[:top_k]

    def progression_to_midi_notes(self, progression: List[Dict]) -> List[List[int]]:
        """Convert progression to list of MIDI note lists."""
        return [chord["midi_notes"] for chord in progression]

    def print_progression(self, progression: List[Dict]):
        """Pretty print a chord progression."""
        for i, chord in enumerate(progression):
            func = chord.get("function", "?")
            name = chord["chord_name"]
            notes = chord["note_names"]
            print(f"  {i+1}. {func:5s} | {name:15s} | Notes: {notes}")


def example_usage():
    """Example usage of the generator."""
    # Load generator
    gen = ChordProgressionGenerator()

    print("\n" + "=" * 70)
    print("Example 1: Random Progression in C Major")
    print("=" * 70)
    prog = gen.generate_progression_random("C", "Major", length=4, category="triad")
    gen.print_progression(prog)

    print("\n" + "=" * 70)
    print("Example 2: Graph-Guided Progression in C Major")
    print("=" * 70)
    prog = gen.generate_progression_graph_guided(
        "C", "Major", length=8, start_function="I"
    )
    gen.print_progression(prog)

    print("\n" + "=" * 70)
    print("Example 3: Brute-Force + Scoring (Top 3 progressions)")
    print("=" * 70)
    results = gen.generate_progression_brute_force(
        "C", "Major", length=4, num_candidates=1000, top_k=3, use_graph=True
    )

    for rank, (score, prog, breakdown) in enumerate(results, 1):
        print(f"\n--- Rank {rank}: Score = {score:.3f} ---")
        print(f"Score breakdown: {breakdown}")
        gen.print_progression(prog)

    print("\n" + "=" * 70)
    print("Example 4: Brute-Force with Custom Scoring Weights")
    print("=" * 70)
    # Prefer graph-probability and voice leading
    custom_weights = {
        "melody": 0.1,
        "voice_leading": 0.4,
        "variety": 0.1,
        "graph_probability": 0.4,
    }

    results = gen.generate_progression_brute_force(
        "G",
        "Major",
        length=6,
        num_candidates=500,
        top_k=2,
        weights=custom_weights,
        use_graph=True,
    )

    for rank, (score, prog, breakdown) in enumerate(results, 1):
        print(f"\n--- Rank {rank}: Score = {score:.3f} ---")
        print(f"Score breakdown: {breakdown}")
        gen.print_progression(prog)


if __name__ == "__main__":
    example_usage()
