#!/usr/bin/env python3
"""
Evolutionary pattern breeding for drum, bass, and melody patterns.

Uses tournament selection, elitism, component-aware crossover, and mutation
to produce novel pattern variants scored by the Refractor.

Public API:
    breed_drum_patterns(concept_emb, seed_patterns, ...) → list[DrumPattern]
    breed_bass_patterns(concept_emb, chord_progression, seed_patterns, ...) → list[BassPattern]
    breed_melody_patterns(concept_emb, chord_progression, seed_patterns, ...) → list[MelodyPattern]
"""

from __future__ import annotations

import copy
import io
import random
from dataclasses import replace
from typing import Any

import mido
import numpy as np
from white_core.enums.bass_chord_tone import BassChordTone

from app.generators.midi.patterns.bass_patterns import (
    BassPattern,
    clamp_to_bass_register,
    resolve_tone,
)
from app.generators.midi.patterns.drum_patterns import (
    VELOCITY,
    DrumPattern,
)
from app.generators.midi.patterns.melody_patterns import MelodyPattern

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MUTATION_PROB = 0.15
_TOURNAMENT_K = 3
_ELITISM_N = 2
_VELOCITY_LEVELS = ("ghost", "normal", "accent")
_BASS_CHORD_TONES = list(BassChordTone)
_EVOLVED_TAG = "evolved"

# Minimal BPM for MIDI generation during fitness evaluation
_SCORE_BPM = 120
_SCORE_BAR_COUNT = 2


# ---------------------------------------------------------------------------
# MIDI serialisation helpers (used for fitness scoring)
# ---------------------------------------------------------------------------


def _drum_to_midi(pattern: DrumPattern) -> bytes:
    """Minimal drum MIDI for fitness scoring."""
    # circular import: drum_pipeline imports from pattern_evolution via evolve flag
    from app.generators.midi.pipelines.drum_pipeline import (
        drum_pattern_to_midi_bytes,
    )  # circular import

    return drum_pattern_to_midi_bytes(
        pattern, bpm=_SCORE_BPM, bar_count=_SCORE_BAR_COUNT
    )


def _bass_to_midi(pattern: BassPattern, chord_progression: list[dict]) -> bytes:
    """Minimal bass MIDI for fitness scoring."""
    tpb = 480
    mid = mido.MidiFile(ticks_per_beat=tpb)
    track = mido.MidiTrack()
    mid.tracks.append(track)
    track.append(
        mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(_SCORE_BPM), time=0)
    )

    chord = (
        chord_progression[0]
        if chord_progression
        else {"root": 36, "notes": [36, 43, 48]}
    )
    voicing = chord.get("notes", [chord.get("root", 36)])

    events: list[tuple[int, int, bool]] = []  # (tick, note, is_on)
    for beat_pos, tone, vel_level in pattern.notes:
        try:
            note = resolve_tone(tone, voicing)
        except Exception:
            note = clamp_to_bass_register(voicing[0] if voicing else 36)
        on_tick = int(beat_pos * tpb)
        off_tick = on_tick + int(0.5 * tpb)
        vel = VELOCITY.get(vel_level, 80)
        events.append((on_tick, note, vel, True))
        events.append((off_tick, note, 0, False))

    events.sort(key=lambda e: (e[0], not e[3]))
    prev = 0
    for tick, note, vel, is_on in events:
        delta = tick - prev
        msg = "note_on" if is_on else "note_off"
        track.append(mido.Message(msg, note=note, velocity=vel, time=delta))
        prev = tick

    out = io.BytesIO()
    mid.save(file=out)
    return out.getvalue()


def _melody_to_midi(pattern: MelodyPattern, chord_progression: list[dict]) -> bytes:
    """Minimal melody MIDI for fitness scoring."""
    tpb = 480
    mid = mido.MidiFile(ticks_per_beat=tpb)
    track = mido.MidiTrack()
    mid.tracks.append(track)
    track.append(
        mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(_SCORE_BPM), time=0)
    )

    chord = (
        chord_progression[0]
        if chord_progression
        else {"root": 60, "notes": [60, 64, 67]}
    )
    root = chord.get("root", 60)
    pitch = root

    durations = pattern.durations or []
    events: list[tuple[int, int, int, bool]] = []

    for i, (onset, interval) in enumerate(zip(pattern.rhythm, pattern.intervals)):
        if i > 0:
            pitch += interval
        pitch = max(48, min(84, pitch))  # keep in vocal range
        dur = durations[i] if i < len(durations) else 0.5
        on_tick = int(onset * tpb)
        off_tick = on_tick + int(dur * tpb)
        events.append((on_tick, pitch, 90, True))
        events.append((off_tick, pitch, 0, False))

    events.sort(key=lambda e: (e[0], not e[3]))
    prev = 0
    for tick, note, vel, is_on in events:
        delta = tick - prev
        msg = "note_on" if is_on else "note_off"
        track.append(mido.Message(msg, note=note, velocity=vel, time=delta))
        prev = tick

    out = io.BytesIO()
    mid.save(file=out)
    return out.getvalue()


# ---------------------------------------------------------------------------
# Fitness scoring
# ---------------------------------------------------------------------------


def _score_population(
    midi_list: list[bytes],
    concept_emb: np.ndarray,
) -> list[float]:
    """Score a list of MIDI byte blobs with the Refractor. Returns fitness values."""
    from training.refractor import Refractor

    scorer = Refractor()
    candidates = [{"midi_bytes": mb} for mb in midi_list]
    results = scorer.score_batch(candidates, concept_emb=concept_emb)

    # Map results back by position
    scored: dict[int, float] = {}
    for r in results:
        midi_key = id(r["candidate"]["midi_bytes"])
        for i, c in enumerate(candidates):
            if id(c["midi_bytes"]) == midi_key:
                # Use mean of all dimension scores as fitness
                s = r.get("scores", {})
                fitness = float(
                    np.mean(
                        [
                            s.get("temporal", 0.0),
                            s.get("spatial", 0.0),
                            s.get("ontological", 0.0),
                        ]
                    )
                )
                scored[i] = fitness
                break

    return [scored.get(i, 0.0) for i in range(len(midi_list))]


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


def _tournament_select(
    population: list[Any], fitness: list[float], k: int = _TOURNAMENT_K
) -> Any:
    """Tournament selection: pick k random candidates, return the fittest."""
    indices = random.sample(range(len(population)), min(k, len(population)))
    best = max(indices, key=lambda i: fitness[i])
    return population[best]


def _elite_indices(fitness: list[float], n: int = _ELITISM_N) -> list[int]:
    """Return indices of the top-n fittest individuals."""
    return sorted(range(len(fitness)), key=lambda i: fitness[i], reverse=True)[:n]


# ---------------------------------------------------------------------------
# Drum crossover and mutation
# ---------------------------------------------------------------------------


def _crossover_drums(parent_a: DrumPattern, parent_b: DrumPattern) -> DrumPattern:
    """Swap whole voice rows between parents to produce a child."""
    voices_a = parent_a.voices
    voices_b = parent_b.voices
    all_voices = set(voices_a) | set(voices_b)
    child_voices: dict[str, list] = {}
    for voice in all_voices:
        # Randomly inherit voice from either parent
        if voice in voices_a and voice in voices_b:
            child_voices[voice] = copy.deepcopy(
                voices_a[voice] if random.random() < 0.5 else voices_b[voice]
            )
        elif voice in voices_a:
            child_voices[voice] = copy.deepcopy(voices_a[voice])
        else:
            child_voices[voice] = copy.deepcopy(voices_b[voice])

    tags = _merge_tags(parent_a.tags, parent_b.tags)
    return replace(
        parent_a,
        name=f"evolved_{parent_a.name}x{parent_b.name}",
        voices=child_voices,
        tags=tags,
    )


def _mutate_drum(pattern: DrumPattern) -> DrumPattern:
    """Randomly flip one cell or randomise one velocity in a random voice."""
    if not pattern.voices or random.random() > _MUTATION_PROB:
        return pattern

    voices = copy.deepcopy(pattern.voices)
    voice_name = random.choice(list(voices.keys()))
    voice = voices[voice_name]
    if not voice:
        return pattern

    idx = random.randrange(len(voice))
    beat_pos, vel = voice[idx]
    # Either flip velocity or shift beat position slightly
    if random.random() < 0.5:
        vel = random.choice(_VELOCITY_LEVELS)
    else:
        beat_pos = max(0.0, beat_pos + random.choice([-0.25, 0.25]))
    voice[idx] = (beat_pos, vel)
    voices[voice_name] = voice
    return replace(pattern, voices=voices)


# ---------------------------------------------------------------------------
# Bass crossover and mutation
# ---------------------------------------------------------------------------


def _crossover_bass(parent_a: BassPattern, parent_b: BassPattern) -> BassPattern:
    """Bar-boundary splice: take first half of A's notes, second half of B's."""
    notes_a = list(parent_a.notes)
    notes_b = list(parent_b.notes)
    split_a = max(1, len(notes_a) // 2)
    split_b = len(notes_b) - max(1, len(notes_b) // 2)
    child_notes = notes_a[:split_a] + notes_b[split_b:]
    if not child_notes:
        child_notes = notes_a or notes_b or []

    durs_a = parent_a.note_durations or []
    durs_b = parent_b.note_durations or []
    child_durs: list[float] | None = None
    if durs_a and durs_b:
        child_durs = durs_a[:split_a] + durs_b[split_b:]
    elif durs_a:
        child_durs = durs_a[: len(child_notes)]
    elif durs_b:
        child_durs = durs_b[: len(child_notes)]

    tags = _merge_tags(parent_a.tags, parent_b.tags)
    return replace(
        parent_a,
        name=f"evolved_{parent_a.name}x{parent_b.name}",
        notes=child_notes,
        note_durations=child_durs if child_durs else None,
        tags=tags,
    )


def _mutate_bass(pattern: BassPattern) -> BassPattern:
    """Shift one note's chord tone to an adjacent tone."""
    if not pattern.notes or random.random() > _MUTATION_PROB:
        return pattern

    notes = list(pattern.notes)
    idx = random.randrange(len(notes))
    beat_pos, tone, vel = notes[idx]
    # Shift to adjacent chord tone
    current_index = _BASS_CHORD_TONES.index(tone) if tone in _BASS_CHORD_TONES else 0
    shift = random.choice([-1, 1])
    new_index = max(0, min(len(_BASS_CHORD_TONES) - 1, current_index + shift))
    notes[idx] = (beat_pos, _BASS_CHORD_TONES[new_index], vel)
    return replace(pattern, notes=notes)


# ---------------------------------------------------------------------------
# Melody crossover and mutation
# ---------------------------------------------------------------------------


def _crossover_melody(
    parent_a: MelodyPattern, parent_b: MelodyPattern
) -> MelodyPattern:
    """Bar-boundary splice on interval sequences."""
    ivs_a = list(parent_a.intervals)
    ivs_b = list(parent_b.intervals)
    rhy_a = list(parent_a.rhythm)
    rhy_b = list(parent_b.rhythm)

    split_a = max(1, len(ivs_a) // 2)
    split_b = len(ivs_b) - max(1, len(ivs_b) // 2)

    child_ivs = ivs_a[:split_a] + ivs_b[split_b:]
    child_rhy = rhy_a[:split_a] + rhy_b[split_b:]
    # Ensure first interval is 0 (starting pitch anchor)
    if child_ivs:
        child_ivs[0] = 0

    child_durs: list[float] | None = None
    durs_a = parent_a.durations or []
    durs_b = parent_b.durations or []
    if durs_a and durs_b:
        child_durs = durs_a[:split_a] + durs_b[split_b:]
    elif durs_a:
        child_durs = durs_a[: len(child_ivs)]
    elif durs_b:
        child_durs = durs_b[: len(child_ivs)]

    tags = _merge_tags(parent_a.tags, parent_b.tags)
    return replace(
        parent_a,
        name=f"evolved_{parent_a.name}x{parent_b.name}",
        intervals=child_ivs,
        rhythm=child_rhy,
        durations=child_durs if child_durs else None,
        tags=tags,
    )


def _mutate_melody(pattern: MelodyPattern) -> MelodyPattern:
    """Shift one interval by ±1 semitone, or one onset by one subdivision."""
    if not pattern.intervals or random.random() > _MUTATION_PROB:
        return pattern

    intervals = list(pattern.intervals)
    rhythm = list(pattern.rhythm)

    if random.random() < 0.5 and len(intervals) > 1:
        # Mutate a non-first interval (first must stay 0)
        idx = random.randint(1, len(intervals) - 1)
        intervals[idx] = intervals[idx] + random.choice([-1, 1])
    elif rhythm:
        idx = random.randrange(len(rhythm))
        rhythm[idx] = max(0.0, rhythm[idx] + random.choice([-0.25, 0.25]))
        rhythm.sort()  # Keep onsets ordered

    return replace(pattern, intervals=intervals, rhythm=rhythm)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _merge_tags(tags_a: list[str], tags_b: list[str]) -> list[str]:
    """Merge parent tags, add evolved tag."""
    merged = list(dict.fromkeys(tags_a + tags_b))  # deduplicate, preserve order
    if _EVOLVED_TAG not in merged:
        merged.append(_EVOLVED_TAG)
    return merged


# ---------------------------------------------------------------------------
# Main breeding loops
# ---------------------------------------------------------------------------


def breed_drum_patterns(
    concept_emb: np.ndarray,
    seed_patterns: list[DrumPattern],
    generations: int = 8,
    population_size: int = 30,
    top_n: int = 5,
) -> list[DrumPattern]:
    """Breed novel DrumPatterns via evolutionary crossover and selection.

    Args:
        concept_emb: Concept embedding from Refractor.prepare_concept().
        seed_patterns: Initial pool of DrumPattern instances.
        generations: Number of evolutionary generations.
        population_size: Target population size per generation.
        top_n: Number of best offspring to return.

    Returns:
        List of up to top_n evolved DrumPattern instances, ordered by fitness.
    """
    if not seed_patterns:
        return []

    # Seed population: repeat/sample seed patterns to reach population_size
    rng = random.Random()
    population = list(seed_patterns)
    while len(population) < population_size:
        population.append(copy.deepcopy(rng.choice(seed_patterns)))

    for _gen in range(generations):
        # Score
        midi_list = [_drum_to_midi(p) for p in population]
        fitness = _score_population(midi_list, concept_emb)

        # Elitism
        elite_idx = _elite_indices(fitness, _ELITISM_N)
        next_gen = [copy.deepcopy(population[i]) for i in elite_idx]

        # Fill rest with tournament-selected crossover children
        while len(next_gen) < len(population):
            pa = _tournament_select(population, fitness)
            pb = _tournament_select(population, fitness)
            child = _crossover_drums(pa, pb)
            child = _mutate_drum(child)
            next_gen.append(child)

        population = next_gen

    # Final scoring and selection
    midi_list = [_drum_to_midi(p) for p in population]
    fitness = _score_population(midi_list, concept_emb)
    sorted_pop = [
        p
        for _, _, p in sorted(
            zip(fitness, range(len(population)), population),
            key=lambda x: x[0],
            reverse=True,
        )
    ]
    return sorted_pop[:top_n]


def breed_bass_patterns(
    concept_emb: np.ndarray,
    chord_progression: list[dict],
    seed_patterns: list[BassPattern],
    generations: int = 8,
    population_size: int = 30,
    top_n: int = 5,
) -> list[BassPattern]:
    """Breed novel BassPatterns via evolutionary crossover and selection.

    Args:
        concept_emb: Concept embedding from Refractor.prepare_concept().
        chord_progression: List of chord dicts with at least 'root' and 'notes'.
        seed_patterns: Initial pool of BassPattern instances.
        generations: Number of evolutionary generations.
        population_size: Target population size per generation.
        top_n: Number of best offspring to return.

    Returns:
        List of up to top_n evolved BassPattern instances, ordered by fitness.
    """
    if not seed_patterns:
        return []

    population = list(seed_patterns)
    while len(population) < population_size:
        population.append(copy.deepcopy(random.choice(seed_patterns)))

    for _gen in range(generations):
        midi_list = []
        for p in population:
            try:
                midi_list.append(_bass_to_midi(p, chord_progression))
            except Exception:
                midi_list.append(b"")
        fitness = _score_population(midi_list, concept_emb)

        elite_idx = _elite_indices(fitness, _ELITISM_N)
        next_gen = [copy.deepcopy(population[i]) for i in elite_idx]

        while len(next_gen) < len(population):
            pa = _tournament_select(population, fitness)
            pb = _tournament_select(population, fitness)
            child = _crossover_bass(pa, pb)
            child = _mutate_bass(child)
            next_gen.append(child)

        population = next_gen

    midi_list = []
    for p in population:
        try:
            midi_list.append(_bass_to_midi(p, chord_progression))
        except Exception:
            midi_list.append(b"")
    fitness = _score_population(midi_list, concept_emb)
    sorted_pop = [
        p
        for _, _, p in sorted(
            zip(fitness, range(len(population)), population),
            key=lambda x: x[0],
            reverse=True,
        )
    ]
    return sorted_pop[:top_n]


def breed_melody_patterns(
    concept_emb: np.ndarray,
    chord_progression: list[dict],
    seed_patterns: list[MelodyPattern],
    generations: int = 8,
    population_size: int = 30,
    top_n: int = 5,
) -> list[MelodyPattern]:
    """Breed novel MelodyPatterns via evolutionary crossover and selection.

    Args:
        concept_emb: Concept embedding from Refractor.prepare_concept().
        chord_progression: List of chord dicts with at least 'root' and 'notes'.
        seed_patterns: Initial pool of MelodyPattern instances.
        generations: Number of evolutionary generations.
        population_size: Target population size per generation.
        top_n: Number of best offspring to return.

    Returns:
        List of up to top_n evolved MelodyPattern instances, ordered by fitness.
    """
    if not seed_patterns:
        return []

    population = list(seed_patterns)
    while len(population) < population_size:
        population.append(copy.deepcopy(random.choice(seed_patterns)))

    for _gen in range(generations):
        midi_list = []
        for p in population:
            try:
                midi_list.append(_melody_to_midi(p, chord_progression))
            except Exception:
                midi_list.append(b"")
        fitness = _score_population(midi_list, concept_emb)

        elite_idx = _elite_indices(fitness, _ELITISM_N)
        next_gen = [copy.deepcopy(population[i]) for i in elite_idx]

        while len(next_gen) < len(population):
            pa = _tournament_select(population, fitness)
            pb = _tournament_select(population, fitness)
            child = _crossover_melody(pa, pb)
            child = _mutate_melody(child)
            next_gen.append(child)

        population = next_gen

    midi_list = []
    for p in population:
        try:
            midi_list.append(_melody_to_midi(p, chord_progression))
        except Exception:
            midi_list.append(b"")
    fitness = _score_population(midi_list, concept_emb)
    sorted_pop = [
        p
        for _, _, p in sorted(
            zip(fitness, range(len(population)), population),
            key=lambda x: x[0],
            reverse=True,
        )
    ]
    return sorted_pop[:top_n]
