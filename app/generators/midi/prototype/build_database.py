"""
Build Polars/Parquet database from MIDI chord library.
"""

import polars as pl
import networkx as nx
from pathlib import Path
from typing import List, Dict
import pickle

from .midi_parser import parse_all_chords, parse_all_progressions


def build_chords_dataframe(chords_data: List[Dict]) -> pl.DataFrame:
    """
    Convert parsed chord data into a Polars DataFrame.
    """
    # Flatten the chord data for columnar storage
    records = []

    for chord in chords_data:
        record = {
            "chord_id": f"{chord['key_root']}_{chord.get('mode_in_key', 'Unknown')}_{chord.get('function', 'unknown')}_{chord['chord_name']}",
            "key_root": chord["key_root"],
            "key_quality": chord["key_quality"],
            "relative_key": chord["relative_key"],
            "mode_in_key": chord.get("mode_in_key"),
            "function": chord.get("function"),
            "chord_name": chord["chord_name"],
            "quality": chord["quality"],
            "category": chord["category"],
            "root_note": chord["root_note"],
            "bass_note": chord["bass_note"],
            "num_notes": chord["num_notes"],
            "midi_notes": chord["midi_notes"],
            "intervals": chord["intervals"],
            "note_names": chord["note_names"],
            "duration_seconds": chord["duration_seconds"],
            "ticks_per_beat": chord["ticks_per_beat"],
            "source_file": chord["source_file"],
        }
        records.append(record)

    return pl.DataFrame(records)


def build_progressions_dataframe(progressions_data: List[Dict]) -> pl.DataFrame:
    """
    Convert parsed progression data into a Polars DataFrame.
    Flattens progressions into individual chord entries with position.
    """
    records = []

    for prog in progressions_data:
        progression_id = f"{prog['key_root']}_{prog.get('mode_in_key', 'Unknown')}_{prog['chord_name']}"

        for position, chord in enumerate(prog["chords"]):
            record = {
                "progression_id": progression_id,
                "progression_name": prog["chord_name"],
                "key_root": prog["key_root"],
                "key_quality": prog["key_quality"],
                "mode_in_key": prog.get("mode_in_key"),
                "position": position,
                "total_chords": prog["num_chords"],
                "root_note": chord["root_note"],
                "bass_note": chord["bass_note"],
                "quality": chord["quality"],
                "midi_notes": chord["midi_notes"],
                "intervals": chord["intervals"],
                "note_names": chord["note_names"],
                "start_time_ticks": chord["start_time_ticks"],
                "duration_ticks": chord["duration_ticks"],
                "source_file": prog["source_file"],
            }
            records.append(record)

    return pl.DataFrame(records)


def build_chord_transition_graph(progressions_df: pl.DataFrame) -> nx.DiGraph:
    """
    Build a directed graph of chord transitions from progressions.
    Edges are weighted by frequency of transitions.
    """
    G = nx.DiGraph()

    # Group by progression_id to process each progression
    for progression_id in progressions_df["progression_id"].unique():
        prog_chords = progressions_df.filter(
            pl.col("progression_id") == progression_id
        ).sort("position")

        chords = prog_chords.select(["root_note", "quality", "position"]).to_dicts()

        # Add nodes and edges
        for i, chord in enumerate(chords):
            chord_label = f"{chord['root_note']}_{chord['quality']}"

            # Add node with metadata
            if not G.has_node(chord_label):
                G.add_node(
                    chord_label, root_note=chord["root_note"], quality=chord["quality"]
                )

            # Add edge to next chord
            if i < len(chords) - 1:
                next_chord = chords[i + 1]
                next_label = f"{next_chord['root_note']}_{next_chord['quality']}"

                if G.has_edge(chord_label, next_label):
                    G[chord_label][next_label]["weight"] += 1
                else:
                    G.add_edge(chord_label, next_label, weight=1)

    # Normalize weights to probabilities
    for node in G.nodes():
        total_weight = sum(
            G[node][neighbor]["weight"] for neighbor in G.neighbors(node)
        )
        if total_weight > 0:
            for neighbor in G.neighbors(node):
                G[node][neighbor]["probability"] = (
                    G[node][neighbor]["weight"] / total_weight
                )

    return G


def build_function_transition_graph(
    progressions_df: pl.DataFrame, chords_df: pl.DataFrame
) -> nx.DiGraph:
    """
    Build a graph of roman numeral function transitions (I -> IV -> V, etc.).
    This is more useful for music theory-based generation.
    """
    G = nx.DiGraph()

    # We need to map chords in progressions to their functions
    # This requires matching progression chords to the chords_df by root_note and quality

    # Create a mapping: (key, root_note, quality) -> function
    function_map = {}
    for chord in chords_df.to_dicts():
        if chord["function"]:
            key = f"{chord['key_root']}_{chord['mode_in_key']}"
            lookup = (key, chord["root_note"], chord["quality"])
            function_map[lookup] = chord["function"]

    # Build graph from progressions
    for progression_id in progressions_df["progression_id"].unique():
        prog_chords = progressions_df.filter(
            pl.col("progression_id") == progression_id
        ).sort("position")

        chords = prog_chords.select(
            ["key_root", "mode_in_key", "root_note", "quality", "position"]
        ).to_dicts()

        # Map to functions
        functions = []
        for chord in chords:
            key = f"{chord['key_root']}_{chord['mode_in_key']}"
            lookup = (key, chord["root_note"], chord["quality"])
            func = function_map.get(lookup, "unknown")
            functions.append((key, func))

        # Build transitions
        for i in range(len(functions) - 1):
            key_from, func_from = functions[i]
            key_to, func_to = functions[i + 1]

            # Only build transitions within same key
            if key_from == key_to and func_from != "unknown" and func_to != "unknown":
                node_from = f"{key_from}:{func_from}"
                node_to = f"{key_to}:{func_to}"

                if not G.has_node(node_from):
                    G.add_node(node_from, key=key_from, function=func_from)
                if not G.has_node(node_to):
                    G.add_node(node_to, key=key_to, function=func_to)

                if G.has_edge(node_from, node_to):
                    G[node_from][node_to]["weight"] += 1
                else:
                    G.add_edge(node_from, node_to, weight=1)

    # Normalize to probabilities
    for node in G.nodes():
        total_weight = sum(
            G[node][neighbor]["weight"] for neighbor in G.neighbors(node)
        )
        if total_weight > 0:
            for neighbor in G.neighbors(node):
                G[node][neighbor]["probability"] = (
                    G[node][neighbor]["weight"] / total_weight
                )

    return G


def compute_statistics(chords_df: pl.DataFrame, progressions_df: pl.DataFrame) -> Dict:
    """
    Compute useful statistics about the chord library.
    """
    stats = {
        "total_chords": len(chords_df),
        "total_progressions": progressions_df["progression_id"].n_unique(),
        "total_progression_chords": len(progressions_df),
        "keys": chords_df["key_root"].unique().to_list(),
        "qualities": chords_df["quality"].value_counts().to_dict(),
        "categories": chords_df["category"].value_counts().to_dict(),
        "functions": chords_df["function"].value_counts().to_dict(),
    }
    return stats


def main(
    chords_dir: str = "/Volumes/LucidNonsense/White/chords",
    output_dir: str = "/Volumes/LucidNonsense/White/app/generators/midi/prototype/data",
):
    """
    Main pipeline to build the chord database.
    """
    chords_dir = Path(chords_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("🎵 Parsing chord MIDI files...")
    chords_data = parse_all_chords(chords_dir, exclude_progressions=True)
    print(f"   Found {len(chords_data)} chords")

    print("🎼 Parsing progression MIDI files...")
    progressions_data = parse_all_progressions(chords_dir)
    print(f"   Found {len(progressions_data)} progressions")

    print("📊 Building Polars DataFrames...")
    chords_df = build_chords_dataframe(chords_data)
    progressions_df = build_progressions_dataframe(progressions_data)

    print("🕸️  Building chord transition graph...")
    chord_graph = build_chord_transition_graph(progressions_df)
    print(
        f"   Graph has {chord_graph.number_of_nodes()} nodes and {chord_graph.number_of_edges()} edges"
    )

    print("🕸️  Building function transition graph...")
    function_graph = build_function_transition_graph(progressions_df, chords_df)
    print(
        f"   Graph has {function_graph.number_of_nodes()} nodes and {function_graph.number_of_edges()} edges"
    )

    print("📈 Computing statistics...")
    stats = compute_statistics(chords_df, progressions_df)

    print("\n📦 Saving to Parquet...")
    chords_df.write_parquet(output_dir / "chords.parquet")
    progressions_df.write_parquet(output_dir / "progressions.parquet")
    print(f"   Saved chords.parquet ({len(chords_df)} rows)")
    print(f"   Saved progressions.parquet ({len(progressions_df)} rows)")

    print("💾 Saving graphs...")
    with open(output_dir / "chord_transition_graph.pkl", "wb") as f:
        pickle.dump(chord_graph, f)
    with open(output_dir / "function_transition_graph.pkl", "wb") as f:
        pickle.dump(function_graph, f)

    print("📊 Saving statistics...")
    with open(output_dir / "stats.pkl", "wb") as f:
        pickle.dump(stats, f)

    print("\n✅ Database build complete!")
    print("\n📊 Statistics:")
    print(f"   Total chords: {stats['total_chords']}")
    print(f"   Total progressions: {stats['total_progressions']}")
    print(f"   Keys: {len(stats['keys'])}")
    print(f"   Top qualities: {list(stats['qualities'].items())[:5]}")
    print(f"   Categories: {stats['categories']}")

    return chords_df, progressions_df, chord_graph, function_graph, stats


if __name__ == "__main__":
    main()
