#!/usr/bin/env python3
"""
Melody generation pipeline for the Music Production Pipeline.

Reads approved chords, harmonic rhythm, and song proposal metadata. Generates
melody candidates from contour templates within singer vocal range constraints,
scores with theory + Refractor composite, writes top candidates as MIDI
files with a review YAML.

Pipeline position: chords → drums → harmonic rhythm → strums → bass → MELODY

Usage:
    python -m app.generators.midi.pipelines.melody_pipeline \
        --production-dir shrink_wrapped/.../production/black__sequential_dissolution_v2 \
        --singer gabriel --seed 42 --top-k 5
"""

import argparse
import io
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import mido
import numpy as np
import yaml
from white_composition.init_production import load_song_context

from white_generation.patterns.aesthetic_hints import (
    aesthetic_tag_adjustment,
    arc_tag_adjustment,
    style_profile_tag_adjustment,
)
from white_generation.patterns.melody_patterns import (
    ALL_TEMPLATES,
    MELODY_CHANNEL,
    SINGERS,
    VELOCITY,
    MelodyPattern,
    SingerRange,
    chord_tone_alignment,
    contour_quality,
    infer_singer,
    make_fallback_pattern,
    melody_theory_score,
    resolve_melody_notes,
    select_templates,
    singability_score,
)
from white_generation.patterns.strum_patterns import (
    read_approved_harmonic_rhythm,
)
from white_generation.pipelines.chord_pipeline import (
    _to_python,
    compute_chromatic_match,
    get_chromatic_target,
    load_song_proposal,
)
from white_generation.util.diversity_tracker import (
    diversity_factor,
    find_album_dir,
    load_registry,
)
from white_generation.util.phrase_dynamics import (
    DynamicCurve,
    apply_dynamics_curve,
    infer_curve,
    parse_curve,
)

# ---------------------------------------------------------------------------
# Melodic continuity helpers
# ---------------------------------------------------------------------------


def last_note_of_midi(midi_bytes: bytes) -> Optional[int]:
    """Return MIDI pitch of the last note-on event, or None if no notes found."""
    mid = mido.MidiFile(file=io.BytesIO(midi_bytes))
    last_pitch = None
    for track in mid.tracks:
        for msg in track:
            if msg.type == "note_on" and msg.velocity > 0:
                last_pitch = msg.note
    return last_pitch


def first_note_of_candidate(candidate: dict) -> Optional[int]:
    """Return MIDI pitch of the first note-on event in candidate's MIDI bytes."""
    midi_bytes = candidate.get("midi_bytes")
    if not midi_bytes:
        return None
    mid = mido.MidiFile(file=io.BytesIO(midi_bytes))
    for track in mid.tracks:
        for msg in track:
            if msg.type == "note_on" and msg.velocity > 0:
                return msg.note
    return None


def continuity_penalty(
    first_note: Optional[int],
    last_note: Optional[int],
    max_semitones: int = 4,
) -> float:
    """Return 0.85 if interval exceeds max_semitones, else 1.0.

    Returns 1.0 (no penalty) when either note is None.
    """
    if first_note is None or last_note is None:
        return 1.0
    return 0.85 if abs(first_note - last_note) > max_semitones else 1.0


# ---------------------------------------------------------------------------
# Read approved sections and chord data (reuse from bass pipeline pattern)
# ---------------------------------------------------------------------------


def read_approved_sections(chord_review: dict) -> list[dict]:
    """Extract approved sections from chord review YAML."""
    sections = []
    for candidate in chord_review.get("candidates", []):
        status = str(candidate.get("status", "")).lower()
        if status not in ("approved", "accepted"):
            continue
        label = candidate.get("label")
        if not label:
            continue
        sections.append(
            {
                "label": label.lower().replace("-", "_").replace(" ", "_"),
                "label_display": label,
                "chord_id": candidate.get("id", "unknown"),
            }
        )
    return sections


def extract_section_chord_data(
    production_dir: Path,
) -> tuple[dict[str, list[list[int]]], dict]:
    """Read approved chord voicings and review metadata.

    Voicings are read directly from the chord review.yml ``chords`` field so
    the count always matches ``hr_distribution`` — regardless of how many strum
    events were baked into the MIDI file.
    """
    from white_generation.pipelines.bass_pipeline import (
        extract_section_chord_data as _bass_extract,
    )

    return _bass_extract(production_dir)


# ---------------------------------------------------------------------------
# Melody MIDI generation
# ---------------------------------------------------------------------------


def melody_notes_to_midi_bytes(
    resolved_notes: list[tuple[float, int, float]],
    bpm: int = 120,
    ticks_per_beat: int = 480,
    curve: DynamicCurve = DynamicCurve.FLAT,
) -> bytes:
    """Convert resolved melody notes to MIDI bytes.

    Args:
        resolved_notes: List of (onset_beat, midi_note, duration_beats).
        bpm: Beats per minute.
        ticks_per_beat: MIDI resolution.

    Returns:
        MIDI file as bytes.
    """
    mid = mido.MidiFile(ticks_per_beat=ticks_per_beat)
    track = mido.MidiTrack()
    mid.tracks.append(track)

    tempo = mido.bpm2tempo(bpm)
    track.append(mido.MetaMessage("set_tempo", tempo=tempo, time=0))

    # Build events
    events = []
    for onset, note, dur in resolved_notes:
        on_tick = int(onset * ticks_per_beat)
        off_tick = int((onset + dur) * ticks_per_beat)
        velocity = VELOCITY["normal"]
        events.append((on_tick, note, velocity, True))
        events.append((off_tick, note, 0, False))

    # Apply phrase-level dynamics before sort
    events = apply_dynamics_curve(events, curve, min_vel=60, max_vel=127)

    events.sort(key=lambda e: (e[0], not e[3], e[1]))

    prev_tick = 0
    for abs_tick, note, velocity, is_on in events:
        delta = abs_tick - prev_tick
        msg_type = "note_on" if is_on else "note_off"
        track.append(
            mido.Message(
                msg_type,
                note=note,
                velocity=velocity,
                time=delta,
                channel=MELODY_CHANNEL,
            )
        )
        prev_tick = abs_tick

    track.append(mido.MetaMessage("end_of_track", time=0))

    buf = io.BytesIO()
    mid.save(file=buf)
    return buf.getvalue()


def generate_melody_for_section(
    pattern: MelodyPattern,
    voicings: list[list[int]],
    singer: SingerRange,
    bpm: int = 120,
    durations: list[float] | None = None,
    curve: DynamicCurve = DynamicCurve.FLAT,
) -> tuple[bytes, list[tuple[float, int, float]]]:
    """Generate melody MIDI for an entire section from a pattern and chord voicings.

    Applies the pattern to each chord in the section, accumulating the melody
    over the full duration.

    Returns:
        Tuple of (midi_bytes, all_resolved_notes).
    """
    bar_beats = pattern.bar_length_beats()
    all_notes: list[tuple[float, int, float]] = []
    offset_beats = 0.0

    for chord_idx, voicing in enumerate(voicings):
        if durations is not None and chord_idx < len(durations):
            chord_dur_beats = durations[chord_idx] * bar_beats
        else:
            chord_dur_beats = bar_beats

        next_voicing = (
            voicings[chord_idx + 1] if chord_idx + 1 < len(voicings) else None
        )

        # Resolve the pattern for this chord
        notes = resolve_melody_notes(pattern, voicing, singer, next_voicing)

        # Repeat pattern if chord duration > 1 bar
        pattern_dur = bar_beats
        repeat_offset = 0.0
        while repeat_offset < chord_dur_beats:
            for onset, note, dur in notes:
                abs_onset = offset_beats + repeat_offset + onset
                # Don't exceed chord boundary
                if repeat_offset + onset >= chord_dur_beats:
                    break
                # Clamp duration to chord boundary
                max_dur = (offset_beats + chord_dur_beats) - abs_onset
                clamped_dur = min(dur, max_dur)
                if clamped_dur > 0:
                    all_notes.append((abs_onset, note, clamped_dur))
            repeat_offset += pattern_dur

        offset_beats += chord_dur_beats

    midi_bytes = melody_notes_to_midi_bytes(all_notes, bpm=bpm, curve=curve)
    return midi_bytes, all_notes


# ---------------------------------------------------------------------------
# Composite scoring
# ---------------------------------------------------------------------------


def melody_composite_score(
    theory: float,
    chromatic_match: float,
    scorer_result: dict,
    theory_breakdown: dict,
    theory_weight: float = 0.3,
    chromatic_weight: float = 0.7,
) -> tuple[float, dict]:
    """Compute weighted composite score for a melody candidate."""
    composite = theory_weight * theory + chromatic_weight * chromatic_match
    breakdown = {
        "composite": round(composite, 4),
        "theory": {k: round(v, 4) for k, v in theory_breakdown.items()},
        "chromatic": {
            "temporal": {m: round(v, 4) for m, v in scorer_result["temporal"].items()},
            "spatial": {m: round(v, 4) for m, v in scorer_result["spatial"].items()},
            "ontological": {
                m: round(v, 4) for m, v in scorer_result["ontological"].items()
            },
            "confidence": round(scorer_result["confidence"], 4),
            "match": round(chromatic_match, 4),
        },
    }
    return composite, breakdown


# ---------------------------------------------------------------------------
# Review YAML generation
# ---------------------------------------------------------------------------


def generate_melody_review_yaml(
    production_dir: str,
    sections: list[dict],
    ranked_by_section: dict[str, list[dict]],
    seed: int,
    scoring_weights: dict,
    song_info: dict,
    singer_name: str,
) -> dict:
    """Generate the review YAML structure for melody candidates."""
    all_candidates = []
    global_rank = 0
    for section in sections:
        section_key = section["_section_key"]
        for item in ranked_by_section.get(section_key, []):
            global_rank += 1
            all_candidates.append(
                {
                    "id": item["id"],
                    "midi_file": f"candidates/{item['id']}.mid",
                    "rank": global_rank,
                    "section": section["label_display"],
                    "chord_source": section.get("chord_id", ""),
                    "contour": item["contour"],
                    "pattern_name": item["pattern_name"],
                    "energy": item["energy"],
                    "use_case": item.get("use_case", "vocal"),
                    "singer": singer_name,
                    "scores": _to_python(item["breakdown"]),
                    "label": None,
                    "status": "pending",
                    "notes": "",
                }
            )

    return {
        "production_dir": str(production_dir),
        "pipeline": "melody-generation",
        "bpm": song_info.get("bpm", 120),
        "time_sig": f"{song_info['time_sig'][0]}/{song_info['time_sig'][1]}",
        "color": song_info.get("color_name", ""),
        "singer": singer_name,
        "generated": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "scoring_weights": scoring_weights,
        "sections_found": [s["label_display"] for s in sections],
        "candidates": all_candidates,
    }


# ---------------------------------------------------------------------------
# Candidate sync (manual MIDI injection / rollback support)
# ---------------------------------------------------------------------------


def sync_melody_candidates(melody_dir: Path) -> int:
    """Scan candidates/ for .mid files not tracked in review.yml and add stubs.

    Safe to run after dropping in a manually edited or hand-composed MIDI.
    Does NOT wipe existing candidates or regenerate anything.

    Returns the number of new entries added.
    """
    review_path = melody_dir / "review.yml"
    candidates_dir = melody_dir / "candidates"

    if not review_path.exists():
        print(f"ERROR: No review.yml found at {review_path}")
        print("Run the melody pipeline first to create a review.yml base.")
        return 0

    with open(review_path) as f:
        review = yaml.safe_load(f) or {}

    existing_files = {
        Path(c["midi_file"]).name
        for c in review.get("candidates", [])
        if c.get("midi_file")
    }
    existing_ids = {c["id"] for c in review.get("candidates", []) if c.get("id")}

    if not candidates_dir.exists():
        print(f"No candidates/ directory at {candidates_dir}")
        return 0

    new_files = sorted(
        f
        for f in candidates_dir.glob("*.mid")
        if f.name not in existing_files and not f.name.endswith("_scratch.mid")
    )

    if not new_files:
        print("All candidate files are already tracked in review.yml")
        return 0

    singer = review.get("singer", "")
    added = 0
    for midi_file in new_files:
        stub_id = midi_file.stem
        if stub_id in existing_ids:
            i = 2
            while f"{stub_id}_{i}" in existing_ids:
                i += 1
            stub_id = f"{stub_id}_{i}"

        stub = {
            "id": stub_id,
            "midi_file": f"candidates/{midi_file.name}",
            "rank": None,
            "section": "manual",
            "chord_source": "manual",
            "contour": "manual",
            "pattern_name": "manual",
            "energy": "unknown",
            "singer": singer,
            "scores": None,
            "label": None,
            "status": "pending",
            "notes": "manually added — set label and status: approved to promote",
        }
        review.setdefault("candidates", []).append(stub)
        existing_ids.add(stub_id)
        print(f"  + {midi_file.name}  →  id: {stub_id}")
        added += 1

    with open(review_path, "w") as f:
        yaml.dump(
            review,
            f,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
        )

    print(f"\nAdded {added} entries to review.yml")
    print(f"Edit {review_path}")
    print("Set label: <name> and status: approved, then run promote_part")
    return added


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

DEFAULT_ENERGY = {
    "intro": "low",
    "verse": "medium",
    "chorus": "high",
    "bridge": "low",
    "outro": "medium",
}


def run_melody_pipeline(
    production_dir: str,
    thread_dir: Optional[str] = None,
    song_filename: Optional[str] = None,
    singer_name: Optional[str] = None,
    seed: int = 42,
    top_k: int = 5,
    theory_weight: float = 0.3,
    chromatic_weight: float = 0.7,
    onnx_path: Optional[str] = None,
    use_case: str = "vocal",
    evolve: bool = False,
    evolve_generations: int = 8,
    evolve_population: int = 30,
):
    """Run the melody generation pipeline end-to-end."""
    np.random.seed(seed)

    prod_path = Path(production_dir)
    if not prod_path.exists():
        print(f"ERROR: Production directory not found: {prod_path}")
        sys.exit(1)

    # --- 1. Read approved chords ---
    print("=" * 60)
    print("MELODY GENERATION PIPELINE")
    print("=" * 60)

    chord_data, chord_review = extract_section_chord_data(prod_path)

    sections = read_approved_sections(chord_review)
    if not sections:
        print("ERROR: No approved chord sections found in chords/review.yml")
        sys.exit(1)

    print(f"Sections: {', '.join(s['label_display'] for s in sections)}")

    # --- 2. Load song info ---
    song_info = {
        "bpm": chord_review.get("bpm", 120),
        "color_name": chord_review.get("color", "White"),
        "concept": "",
        "time_sig": (4, 4),
    }

    thread_from_review = chord_review.get("thread", "")
    song_from_review = chord_review.get("song_proposal", "")
    t_dir = thread_dir or thread_from_review
    s_file = song_filename or song_from_review

    if t_dir and s_file:
        try:
            song_info = load_song_proposal(Path(t_dir), s_file)
        except Exception as e:
            print(f"  Warning: Could not load song proposal: {e}")

    time_sig_str = chord_review.get("time_sig")
    if time_sig_str and "/" in str(time_sig_str):
        parts = str(time_sig_str).split("/")
        song_info["time_sig"] = (int(parts[0]), int(parts[1]))

    time_sig = tuple(song_info["time_sig"])
    bpm = song_info["bpm"]

    print(f"BPM:   {bpm}")
    print(f"Time:  {time_sig[0]}/{time_sig[1]}")
    print(f"Color: {song_info['color_name']}")

    # Melodic continuity config
    continuity_semitones: int = int(song_info.get("melodic_continuity_semitones", 4))

    # --- 3. Determine singer ---
    if singer_name:
        singer_key = singer_name.lower().strip()
        if singer_key not in SINGERS:
            print(
                f"ERROR: Unknown singer '{singer_name}'. Available: {', '.join(SINGERS.keys())}"
            )
            sys.exit(1)
        singer = SINGERS[singer_key]
    else:
        # Try to get singer from song proposal
        proposal_singer = song_info.get("singer", "")
        if proposal_singer and proposal_singer.lower() in SINGERS:
            singer = SINGERS[proposal_singer.lower()]
        else:
            # Infer from key
            key_str = song_info.get("key", "")
            if key_str:
                from white_generation.pipelines.chord_pipeline import (
                    parse_key_string,
                )

                key_info = parse_key_string(key_str)
                tonic_midi = key_info.get("tonic_midi", 60)
            else:
                tonic_midi = 60
            singer = infer_singer(tonic_midi)

    print(
        f"Singer: {singer.name} ({singer.voice_type}, MIDI {singer.low}-{singer.high})"
    )

    # --- 4. Read harmonic rhythm ---
    hr_durations = read_approved_harmonic_rhythm(prod_path)
    if hr_durations:
        print(f"Harmonic rhythm loaded for: {', '.join(hr_durations.keys())}")
    else:
        print("  No approved harmonic rhythm — using 1 bar per chord")

    # --- 5. Load Refractor ---
    print("\nLoading Refractor...")
    from white_analysis.refractor import Refractor

    scorer = Refractor(onnx_path=onnx_path) if onnx_path else Refractor()

    _ctx = load_song_context(prod_path)
    concept_text = song_info.get("concept", "") or _ctx.get("concept", "")
    if not concept_text:
        concept_text = f"{song_info['color_name']} chromatic concept"
        print(f"  Warning: No concept text, using fallback: '{concept_text}'")
    concept_emb = scorer.prepare_concept(concept_text)
    print(f"  Concept encoded ({concept_emb.shape[0]}-dim)")
    aesthetic_hints = _ctx.get("aesthetic_hints") or {}
    _style_profile = _ctx.get("style_reference_profile") or {}

    # Load production plan for arc-aware tag adjustments
    from white_composition.production_plan import load_plan

    _prod_plan = load_plan(prod_path)
    _arc_by_label: dict[str, float] = {}
    if _prod_plan:
        for _ps in _prod_plan.sections:
            _arc_by_label[_ps.name.lower().replace("-", "_").replace(" ", "_")] = (
                _ps.arc
            )

    from white_core.music.narrative_constraints import (
        extract_constraints,
    )
    from white_core.music.narrative_constraints import (
        narrative_tag_adjustment as _narr_adj,
    )

    _narrative = None

    target = get_chromatic_target(song_info["color_name"])

    # --- 6. Disambiguate section labels ---
    label_occurrence: dict[str, int] = {}
    for section in sections:
        label = section["label"]
        label_occurrence[label] = label_occurrence.get(label, 0) + 1
        section["_occurrence"] = label_occurrence[label]

    duplicate_labels = {k for k, v in label_occurrence.items() if v > 1}
    for section in sections:
        label = section["label"]
        if label in duplicate_labels:
            section["_section_key"] = f"{label}_{section['_occurrence']}"
        else:
            section["_section_key"] = label

    # --- 7. Generate and score per section ---
    # Load album-level diversity registry (penalty for overused templates).
    _album_dir = find_album_dir(prod_path)
    _diversity_registry: dict[str, int] = (
        load_registry(_album_dir) if _album_dir else {}
    )
    if _diversity_registry:
        print(f"  Diversity registry: {len(_diversity_registry)} template(s) tracked")

    ranked_by_section: dict[str, list[dict]] = {}
    all_midi_outputs: list[tuple[str, bytes]] = []

    for section_idx, section in enumerate(sections):
        section_key = section["_section_key"]
        label = section["label"]
        label_display = section["label_display"]

        voicings = chord_data.get(label, [])
        if not voicings:
            voicings = chord_data.get(section_key, [])
        if not voicings:
            for k, v in chord_data.items():
                if k.lower().startswith(label):
                    voicings = v
                    break

        # Skip section when narrative declares lead_voice: none
        if _narrative:
            _sec_nc = extract_constraints(label, _narrative)
            if _sec_nc.get("skip_melody"):
                print(
                    f"\n--- Section: {label_display} — SKIPPED (narrative: lead_voice=none) ---"
                )
                continue
        else:
            _sec_nc = {}

        if not voicings:
            print(f"\n--- Section: {label_display} — SKIPPED (no chord voicings) ---")
            continue

        section_durations = hr_durations.get(label, None)
        if section_durations and len(section_durations) != len(voicings):
            print(
                f"  Warning: HR durations ({len(section_durations)}) != voicings ({len(voicings)}) for {label}, ignoring HR"
            )
            section_durations = None

        print(
            f"\n--- Section: {label_display} [{section_key}] ({len(voicings)} chords) ---"
        )

        target_energy = DEFAULT_ENERGY.get(label, "medium")

        # Determine dynamic curve for this section
        _dynamics_map: dict = song_info.get("raw_proposal", {}).get("dynamics", {})
        raw_curve = _dynamics_map.get(label) or _dynamics_map.get(section_key)
        section_curve = parse_curve(raw_curve) if raw_curve else infer_curve(label)

        # "instrumental" is the user-facing alias; template library uses "lead"
        template_use_case = "lead" if use_case == "instrumental" else use_case
        templates = select_templates(
            ALL_TEMPLATES, time_sig, target_energy, use_case=template_use_case
        )
        if not templates:
            print(
                f"\n  ⚠️  WARNING: No melody templates exist for {time_sig[0]}/{time_sig[1]} time. "
                f"Add templates to melody_patterns.py for this time signature. "
                f"Falling back to repeated root — results will be musically flat.\n"
            )
            templates = [make_fallback_pattern(time_sig)]

        print(
            f"  Templates: {len(templates)} candidates (energy target: {target_energy})"
        )

        # Evolutionary breeding (opt-in)
        if evolve and templates:
            from white_generation.patterns.pattern_evolution import (
                breed_melody_patterns,
            )

            chord_progression = (
                [{"root": v[0] if v else 60, "notes": v} for v in voicings]
                if voicings
                else [{"root": 60, "notes": [60]}]
            )
            print(
                f"  Breeding evolved candidates ({evolve_generations} generations, population {evolve_population})..."
            )
            evolved = breed_melody_patterns(
                concept_emb,
                chord_progression=chord_progression,
                seed_patterns=templates,
                generations=evolve_generations,
                population_size=evolve_population,
                top_n=top_k,
            )
            templates = templates + evolved
            print(f"  Templates after breeding: {len(templates)} candidates")

        # Extract chord tones for scoring
        chord_tones_pc = set()
        for voicing in voicings:
            for n in voicing:
                chord_tones_pc.add(n % 12)

        candidates = []
        for tmpl in templates:
            midi_bytes, resolved_notes = generate_melody_for_section(
                tmpl,
                voicings,
                singer,
                bpm=bpm,
                durations=section_durations,
                curve=section_curve,
            )

            # Theory scoring
            sing = singability_score(resolved_notes, singer, time_sig)
            ct = chord_tone_alignment(resolved_notes, chord_tones_pc, time_sig)
            cq = contour_quality(resolved_notes)
            theory = melody_theory_score(sing, ct, cq)

            theory_breakdown = {
                "singability": sing,
                "chord_tone_alignment": ct,
                "contour_quality": cq,
            }

            is_evolved = "evolved" in getattr(tmpl, "tags", [])
            candidates.append(
                {
                    "template": tmpl,
                    "midi_bytes": midi_bytes,
                    "resolved_notes": resolved_notes,
                    "theory": theory,
                    "theory_breakdown": theory_breakdown,
                    "pattern_name": tmpl.name,
                    "contour": tmpl.contour,
                    "energy": tmpl.energy,
                    "use_case": tmpl.use_case,
                    "is_evolved": is_evolved,
                }
            )

        # Preceding approved section's last note (for continuity penalty).
        # Scan backward to find the nearest prior section that has an approved MIDI —
        # skipping any gaps where the human hasn't approved yet.
        preceding_last_note: Optional[int] = None
        for prev_idx in range(section_idx - 1, -1, -1):
            prev_label = sections[prev_idx]["label"]
            approved_midi = prod_path / "melody" / "approved" / f"{prev_label}.mid"
            if approved_midi.exists():
                preceding_last_note = last_note_of_midi(approved_midi.read_bytes())
                if preceding_last_note is not None:
                    print(
                        f"  Continuity anchor: {prev_label} last note = {preceding_last_note} (max leap {continuity_semitones} st)"
                    )
                break

        # Score with Refractor
        print(f"  Scoring {len(candidates)} candidates...")
        scorer_candidates = [{"midi_bytes": c["midi_bytes"]} for c in candidates]
        scorer_results = scorer.score_batch(scorer_candidates, concept_emb=concept_emb)

        scorer_by_midi = {}
        for result in scorer_results:
            midi_key = id(result["candidate"]["midi_bytes"])
            scorer_by_midi[midi_key] = result

        # Composite scoring
        scored = []
        for i, cand in enumerate(candidates):
            midi_key = id(scorer_candidates[i]["midi_bytes"])
            scorer_result = scorer_by_midi.get(midi_key)
            if scorer_result is None:
                continue

            chromatic_match = compute_chromatic_match(scorer_result, target)
            comp, breakdown = melody_composite_score(
                cand["theory"],
                chromatic_match,
                scorer_result,
                cand["theory_breakdown"],
                theory_weight,
                chromatic_weight,
            )
            comp *= diversity_factor(cand["pattern_name"], _diversity_registry)
            # Derive first note from in-memory resolved_notes (avoids reparsing MIDI)
            _rn = cand.get("resolved_notes") or []
            _first_note = _rn[0][1] if _rn else None
            comp *= continuity_penalty(
                _first_note, preceding_last_note, continuity_semitones
            )
            _label_key = label.lower().replace("-", "_").replace(" ", "_")
            _tmpl_tags = getattr(cand["template"], "tags", [])
            tag_adj = aesthetic_tag_adjustment(
                _tmpl_tags, aesthetic_hints
            ) + style_profile_tag_adjustment(_style_profile, _tmpl_tags, "melody")
            if _label_key in _arc_by_label:
                tag_adj += arc_tag_adjustment(_arc_by_label[_label_key], _tmpl_tags)
            if _sec_nc:
                tag_adj += _narr_adj(_sec_nc, _tmpl_tags, "melody")
            comp = round(comp + tag_adj, 4)
            breakdown["tag_adjustment"] = tag_adj
            breakdown["composite"] = comp
            scored.append(
                {
                    "composite": comp,
                    "breakdown": breakdown,
                    "midi_bytes": cand["midi_bytes"],
                    "pattern_name": cand["pattern_name"],
                    "contour": cand["contour"],
                    "energy": cand["energy"],
                    "use_case": cand["use_case"],
                    "description": cand["template"].description,
                    "is_evolved": cand.get("is_evolved", False),
                }
            )

        scored.sort(key=lambda x: x["composite"], reverse=True)
        top = scored[:top_k]

        for rank, item in enumerate(top):
            item["rank"] = rank + 1
            prefix = "evolved_" if item.get("is_evolved") else ""
            item["id"] = f"{prefix}melody_{section_key}_{rank + 1:02d}"
            all_midi_outputs.append((f"{item['id']}.mid", item["midi_bytes"]))

        ranked_by_section[section_key] = top

        for item in top:
            theory_str = " ".join(
                f"{k}={v:.2f}" for k, v in item["breakdown"]["theory"].items()
            )
            print(
                f"  #{item['rank']} [{item['id']}] {item['pattern_name']:25s} "
                f"composite={item['breakdown']['composite']:.3f} "
                f"theory=[{theory_str}] "
                f"chromatic={item['breakdown']['chromatic']['match']:.3f}"
            )

    # --- 8. Write MIDI files ---
    melody_dir = prod_path / "melody"
    candidates_dir = melody_dir / "candidates"
    approved_dir = melody_dir / "approved"
    if candidates_dir.exists():
        for old_file in candidates_dir.glob("*.mid"):
            old_file.unlink()
    candidates_dir.mkdir(parents=True, exist_ok=True)
    approved_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nWriting {len(all_midi_outputs)} MIDI files to {candidates_dir}/")
    for filename, midi_bytes in all_midi_outputs:
        path = candidates_dir / filename
        path.write_bytes(midi_bytes)

    # --- 9. Write review YAML ---
    review = generate_melody_review_yaml(
        production_dir,
        sections,
        ranked_by_section,
        seed,
        {"theory": theory_weight, "chromatic": chromatic_weight},
        song_info,
        singer.name,
    )
    review_path = melody_dir / "review.yml"
    with open(review_path, "w") as f:
        yaml.dump(
            review,
            f,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
            width=float("inf"),
        )

    # --- 10. Summary ---
    total = sum(len(v) for v in ranked_by_section.values())
    print(f"\n{'=' * 60}")
    print("MELODY GENERATION COMPLETE")
    print(f"{'=' * 60}")
    print(f"Singer:     {singer.name} ({singer.voice_type})")
    print(f"Sections:   {len(sections)}")
    print(f"Candidates: {total}")
    print(f"Review:     {review_path}")
    print(f"\nNext: Edit {review_path} to label and approve candidates")
    print(
        f"Then: python -m app.generators.midi.production.promote_part --review {review_path}"
    )

    return ranked_by_section


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Melody generation pipeline — generate, score, and review melodies",
    )
    parser.add_argument(
        "--production-dir",
        required=True,
        help="Song production directory (must contain chords/approved/)",
    )
    parser.add_argument(
        "--sync-candidates",
        action="store_true",
        help=(
            "Scan candidates/ for .mid files not in review.yml and add stub entries. "
            "Does not regenerate or wipe anything. Use after dropping in manual MIDIs."
        ),
    )
    parser.add_argument(
        "--thread",
        default=None,
        help="shrink_wrapped thread directory (optional)",
    )
    parser.add_argument(
        "--song",
        default=None,
        help="Song proposal YAML filename (optional)",
    )
    parser.add_argument(
        "--singer",
        default=None,
        help=f"Singer name: {', '.join(SINGERS.keys())} (optional, inferred from key)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Top candidates per section (default: 5)",
    )
    parser.add_argument(
        "--theory-weight",
        type=float,
        default=0.3,
        help="Theory score weight (default: 0.3)",
    )
    parser.add_argument(
        "--chromatic-weight",
        type=float,
        default=0.7,
        help="Chromatic score weight (default: 0.7)",
    )
    parser.add_argument(
        "--onnx-path",
        default=None,
        help="Path to refractor.onnx (default: training/data/refractor.onnx)",
    )
    parser.add_argument(
        "--use-case",
        default="vocal",
        choices=["vocal", "instrumental", "lead"],
        help="Melody use case — filters template pool (default: vocal)",
    )
    parser.add_argument(
        "--evolve",
        action="store_true",
        help="Breed evolved melody pattern candidates via evolutionary algorithm",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=8,
        help="Number of evolutionary generations (default: 8, only used with --evolve)",
    )
    parser.add_argument(
        "--population",
        type=int,
        default=30,
        help="Evolutionary population size (default: 30, only used with --evolve)",
    )

    args = parser.parse_args()

    if args.sync_candidates:
        melody_dir = Path(args.production_dir) / "melody"
        sync_melody_candidates(melody_dir)
        return

    run_melody_pipeline(
        production_dir=args.production_dir,
        thread_dir=args.thread,
        song_filename=args.song,
        singer_name=args.singer,
        seed=args.seed,
        top_k=args.top_k,
        theory_weight=args.theory_weight,
        chromatic_weight=args.chromatic_weight,
        onnx_path=args.onnx_path,
        use_case=args.use_case,
        evolve=args.evolve,
        evolve_generations=args.generations,
        evolve_population=args.population,
    )


if __name__ == "__main__":
    main()
