#!/usr/bin/env python3
"""
Lyric generation pipeline for the Music Production Pipeline.

Generates N complete lyric drafts (all vocal sections) via Claude API, scores
each with Refractor in text-only mode, computes a syllable fitting score
(syllables vs. melody notes) per section, and writes melody/lyrics_review.yml
(append-only). Integrates with promote_part.py to copy an approved .txt to
melody/lyrics.txt.

Vocal sections are derived from arrangement.txt (track 4 = melody = vocal).
Song metadata is read from the song proposal YAML via chords/review.yml.
No production_plan.yml is required.

Pipeline position: chords → drums → bass → melody → arrangement export → LYRICS

Usage:
    python -m app.generators.midi.pipelines.lyric_pipeline \\
        --production-dir shrink_wrapped/.../production/yellow__... \\
        --num-candidates 3

    # Register manually placed .txt files
    python -m app.generators.midi.pipelines.lyric_pipeline \\
        --production-dir ... --sync-candidates
"""

import argparse
import math
import re
import sys
import mido
import yaml

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

from app.generators.artist_catalog import load_artist_context  # noqa: E402
from app.generators.midi.production.init_production import (
    load_initial_proposal,
    load_song_context,
)  # noqa: E402
from app.generators.midi.pipelines.chord_pipeline import (  # noqa: E402
    _to_python,
    compute_chromatic_match,
    get_chromatic_target,
)
from app.generators.midi.production.song_evaluator import _count_syllables  # noqa: E402

load_dotenv()

LYRICS_REVIEW_FILENAME = "lyrics_review.yml"

MELODY_CHANNEL = 4  # track 4 in arrangement.txt = melody = vocal


# ---------------------------------------------------------------------------
# Note counting + phrase extraction
# ---------------------------------------------------------------------------


def _count_notes(midi_path: Path) -> int:
    """Count note_on events with velocity > 0 across all tracks."""
    try:
        mid = mido.MidiFile(str(midi_path))
    except Exception:
        return 0
    count = 0
    for track in mid.tracks:
        for msg in track:
            if msg.type == "note_on" and msg.velocity > 0:
                count += 1
    return count


@dataclass
class Phrase:
    start_tick: int
    end_tick: int
    note_count: int


def extract_phrases(midi_path: Path, rest_threshold_beats: float = 0.5) -> list[Phrase]:
    """Group note-on events into phrases separated by rests.

    A new phrase begins when the gap between consecutive note-on events
    exceeds rest_threshold_beats (default 0.5 beats = half a beat).
    Single-note phrases are allowed.

    Returns a list of Phrase objects in order.
    """
    try:
        mid = mido.MidiFile(str(midi_path))
    except Exception:
        return []

    ticks_per_beat = mid.ticks_per_beat or 480
    threshold_ticks = int(rest_threshold_beats * ticks_per_beat)

    # Collect all note-on absolute ticks across all tracks
    note_ticks: list[int] = []
    for track in mid.tracks:
        abs_tick = 0
        for msg in track:
            abs_tick += msg.time
            if msg.type == "note_on" and msg.velocity > 0:
                note_ticks.append(abs_tick)

    if not note_ticks:
        return []

    note_ticks.sort()

    phrases: list[Phrase] = []
    phrase_start = note_ticks[0]
    phrase_notes = [note_ticks[0]]

    for tick in note_ticks[1:]:
        if tick - phrase_notes[-1] > threshold_ticks:
            phrases.append(
                Phrase(
                    start_tick=phrase_start,
                    end_tick=phrase_notes[-1],
                    note_count=len(phrase_notes),
                )
            )
            phrase_start = tick
            phrase_notes = [tick]
        else:
            phrase_notes.append(tick)

    phrases.append(
        Phrase(
            start_tick=phrase_start,
            end_tick=phrase_notes[-1],
            note_count=len(phrase_notes),
        )
    )

    return phrases


# ---------------------------------------------------------------------------
# Arrangement parser
# ---------------------------------------------------------------------------


def _parse_timecode_secs(tc: str) -> float:
    """Parse HH:MM:SS:FF.ff Logic timecode to seconds (30fps assumed)."""
    tc = tc.strip()
    parts = tc.split(":")
    if len(parts) != 4:
        return 0.0
    try:
        h, m, s = int(parts[0]), int(parts[1]), int(parts[2])
        frames = float(parts[3])
        return h * 3600.0 + m * 60.0 + s + frames / 30.0
    except (ValueError, IndexError):
        return 0.0


def parse_arrangement(arrangement_path: Path) -> list[dict]:
    """Parse arrangement.txt into a list of clip dicts.

    Each dict: {timecode_secs, clip_name, channel, duration_secs}
    Lines are tab-separated: timecode  clip_name  channel  duration
    """
    clips = []
    with open(arrangement_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split("\t") if p.strip()]
            if len(parts) < 4:
                continue
            try:
                timecode_secs = _parse_timecode_secs(parts[0])
                clip_name = parts[1]
                channel = int(parts[2])
                duration_secs = _parse_timecode_secs(parts[3])
                clips.append(
                    {
                        "timecode_secs": timecode_secs,
                        "clip_name": clip_name,
                        "channel": channel,
                        "duration_secs": duration_secs,
                    }
                )
            except (ValueError, IndexError):
                continue
    return clips


# ---------------------------------------------------------------------------
# Song proposal loader
# ---------------------------------------------------------------------------


def _find_and_load_proposal(production_dir: Path) -> dict:
    """Find and load the song proposal for a production directory.

    Reads thread + song_proposal from chords/review.yml to resolve the path.
    Returns a normalised metadata dict with: title, bpm, time_sig, key, color,
    concept, genres, mood, singer, sounds_like.
    Returns {} if the proposal cannot be found.
    """
    chord_review_path = production_dir / "chords" / "review.yml"
    if not chord_review_path.exists():
        return {}
    with open(chord_review_path) as f:
        chord_review = yaml.safe_load(f) or {}

    thread = chord_review.get("thread", "")
    song_proposal_file = chord_review.get("song_proposal", "")
    if not thread or not song_proposal_file:
        return {}

    thread_path = Path(thread)
    for candidate in [
        thread_path / "yml" / song_proposal_file,
        thread_path / song_proposal_file,
    ]:
        if candidate.exists():
            from app.generators.midi.production.production_plan import (
                load_song_proposal_unified,
            )

            unified = load_song_proposal_unified(candidate, thread_dir=thread_path)

            # Prefer sounds_like from song_context.yml (written by init_production)
            _ctx = load_song_context(production_dir)
            sounds_like = _ctx.get("sounds_like") or unified.get("sounds_like") or []

            # Singer: song_context > proposal > chord_review
            singer = (
                _ctx.get("singer")
                or unified.get("singer")
                or str(chord_review.get("singer", ""))
            )

            return {
                "title": unified["title"],
                "bpm": unified["bpm"],
                "time_sig": unified["time_sig"],
                "key": unified["key"],
                "color": unified["color"],
                "concept": unified["concept"],
                "genres": unified["genres"],
                "mood": unified["mood"],
                "singer": singer,
                "sounds_like": sounds_like,
            }

    return {}


# ---------------------------------------------------------------------------
# Vocal section reading from arrangement
# ---------------------------------------------------------------------------


def read_vocal_sections_from_arrangement(
    arrangement_path: Path,
    melody_dir: Path,
    bpm: int,
    time_sig_str: str,
) -> list[dict]:
    """Extract vocal sections from arrangement.txt.

    Track 4 (MELODY_CHANNEL) clips are vocal by definition — no vocals flag needed.
    Returns one entry per unique clip label, in first-seen order.

    Each entry: {approved_label, name, bars, play_count, total_notes, contour}
    """
    clips = parse_arrangement(arrangement_path)

    parts = str(time_sig_str).split("/")
    num, den = int(parts[0]), int(parts[1])
    beats_per_bar = num * (4.0 / den)
    secs_per_bar = beats_per_bar * (60.0 / bpm)

    # Load melody review for contour info
    melody_review_path = melody_dir / "review.yml"
    contour_by_label: dict[str, str] = {}
    if melody_review_path.exists():
        with open(melody_review_path) as f:
            melody_review = yaml.safe_load(f) or {}
        for cand in melody_review.get("candidates", []):
            label = cand.get("label")
            status = str(cand.get("status", "")).lower()
            if label and status in ("approved", "accepted"):
                contour_by_label[label] = cand.get("contour", "stepwise")

    # Collect track 4 clips in arrangement order — one entry per instance.
    # Duplicate labels get _2, _3 suffixes; the prompt uses these suffixed names
    # as [headers] so Claude writes one block per arrangement instance.
    melody_clips = [c for c in clips if c["channel"] == MELODY_CHANNEL]
    label_seen_count: dict[str, int] = {}

    approved_dir = melody_dir / "approved"
    sections = []
    for clip in melody_clips:
        label = clip["clip_name"]
        label_seen_count[label] = label_seen_count.get(label, 0) + 1
        n = label_seen_count[label]
        instance_key = label if n == 1 else f"{label}_{n}"

        bars = max(round(clip["duration_secs"] / secs_per_bar), 1)
        midi_path = approved_dir / f"{label}.mid"
        per_loop_notes = _count_notes(midi_path) if midi_path.exists() else 0

        sections.append(
            {
                "approved_label": label,  # base label → MIDI filename (strip _N suffix)
                "name": instance_key,  # unique instance key used as [header]
                "bars": bars,
                "play_count": 1,
                "total_notes": per_loop_notes,
                "contour": contour_by_label.get(label, "stepwise"),
            }
        )

    return sections


# ---------------------------------------------------------------------------
# Syllable fitting
# ---------------------------------------------------------------------------

_VERDICT_ORDER = ["spacious", "paste-ready", "tight but workable", "splits needed"]


def _fitting_verdict(ratio: float) -> str:
    if ratio < 0.75:
        return "spacious"
    elif ratio <= 1.10:
        return "paste-ready"
    elif ratio <= 1.30:
        return "tight but workable"
    else:
        return "splits needed"


def _verdict_rank(verdict: str) -> int:
    """Rank verdict severity; spacious == paste-ready (both = 0)."""
    v = verdict if verdict != "spacious" else "paste-ready"
    return _VERDICT_ORDER.index(v)


def _compute_fitting(
    candidate_text: str,
    vocal_sections: list[dict],
    melody_dir: Path,
) -> dict:
    """Compute per-phrase syllable fitting for each vocal section.

    When an approved MIDI exists, extracts phrase structure and scores each
    lyric line against its corresponding phrase's note count.  Falls back to
    section-level ratio when no MIDI or no phrases are detected.

    The overall verdict is driven by the worst-case phrase, not the mean.
    """
    parsed = _parse_sections(candidate_text)
    result: dict = {}
    worst_verdict = "paste-ready"

    for sec in vocal_sections:
        name = sec["name"]
        # Strip instance suffix (_2, _3, …) to find the base MIDI file
        base_label = re.sub(r"_\d+$", "", name)
        midi_path = melody_dir / "approved" / f"{base_label}.mid"
        phrases = extract_phrases(midi_path) if midi_path.exists() else []

        lyric_text = parsed.get(name, "")
        lyric_lines = [
            line.strip()
            for line in lyric_text.splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]

        if phrases:
            # Scale phrases to cover all plays of this loop
            phrases = phrases * sec.get("play_count", 1)
            phrase_data = []
            for i, phrase in enumerate(phrases):
                line_text = lyric_lines[i] if i < len(lyric_lines) else ""
                syl = _count_syllables(line_text) if line_text else 0
                notes = phrase.note_count
                ratio = round(syl / notes, 3) if notes > 0 else 1.0
                verdict = _fitting_verdict(ratio)
                phrase_data.append(
                    {
                        "notes": notes,
                        "syllables": syl,
                        "ratio": ratio,
                        "verdict": verdict,
                    }
                )

            worst_r = max(p["ratio"] for p in phrase_data)
            worst_v = _fitting_verdict(worst_r)
            mean_r = round(sum(p["ratio"] for p in phrase_data) / len(phrase_data), 3)

            result[name] = {
                "phrases": phrase_data,
                "worst_ratio": round(worst_r, 3),
                "worst_verdict": worst_v,
                "mean_ratio": mean_r,
                "overall": worst_v,
            }
        else:
            # Fallback: section-level ratio (no MIDI available)
            total_notes = sec["total_notes"]
            syllable_count = sum(_count_syllables(line) for line in lyric_lines)
            ratio = round(syllable_count / total_notes, 3) if total_notes > 0 else 1.0
            worst_v = _fitting_verdict(ratio)
            result[name] = {
                "syllables": syllable_count,
                "notes": total_notes,
                "ratio": ratio,
                "verdict": worst_v,
            }

        if _verdict_rank(worst_v) > _verdict_rank(worst_verdict):
            worst_verdict = worst_v

    result["overall"] = worst_verdict
    return result


# ---------------------------------------------------------------------------
# Keyword-based chromatic scoring (Bug 2 hybrid fallback)
# ---------------------------------------------------------------------------

_TEMPORAL_KEYWORDS: dict[str, list[str]] = {
    "past": [
        "used to",
        "back then",
        "ago",
        "yesterday",
        "once was",
        "always was",
        "i remember",
        "she remembered",
        "he remembered",
        "they remembered",
        "had been",
        "was there",
        "were there",
        "before you",
        "before she",
        "the old",
        "left behind",
    ],
    "present": [
        "right now",
        "in this moment",
        "still here",
        "still breathing",
        "still standing",
        "still watching",
        "happening now",
        "as we speak",
        "in this room",
        "in this place",
        "this very",
        "at this moment",
    ],
    "future": [
        "will be",
        "will walk",
        "will remember",
        "will find",
        "will come",
        "going to",
        "one day",
        "someday",
        "tomorrow",
        "soon you",
        "soon she",
        "when you will",
        "you will",
        "she will",
        "they will",
        "might become",
        "could become",
        "shall",
        "still to come",
    ],
}

_SPATIAL_KEYWORDS: dict[str, list[str]] = {
    "thing": [
        "object",
        "artifact",
        "machine",
        "device",
        "stone",
        "metal",
        "wood",
        "instrument",
        "tool",
        "structure",
        "substance",
        "material",
        "fragment",
        "piece of",
        "the thing",
        "the item",
    ],
    "place": [
        "city",
        "room",
        "street",
        "road",
        "field",
        "river",
        "mountain",
        "valley",
        "home",
        "door",
        "wall",
        "ground",
        "sky",
        "land",
        "world",
        "somewhere",
        "anywhere",
        "every where",
        "this place",
        "that place",
        "the space",
    ],
    "person": [
        "you",
        "your",
        "yours",
        "she",
        "her",
        "he",
        "his",
        "they",
        "their",
        "name",
        "face",
        "eyes",
        "hands",
        "voice",
        "body",
        "heart",
        "soul",
        "woman",
        "man",
        "someone",
        "whoever",
        "the one who",
    ],
}

_ONTOLOGICAL_KEYWORDS: dict[str, list[str]] = {
    "imagined": [
        "imagine",
        "imagined",
        "maybe",
        "perhaps",
        "possibly",
        "what if",
        "might be",
        "could be",
        "seems like",
        "appears to",
        "like a dream",
        "fabricated",
        "invented",
        "conjured",
        "not sure if",
        "possibly real",
        "fully fabricated",
        "possibly imagined",
    ],
    "forgotten": [
        "forgotten",
        "erased",
        "vanished",
        "gone now",
        "no longer here",
        "disappeared",
        "faded away",
        "buried",
        "lost forever",
        "never found",
        "unnamed",
        "unknown",
        "left no trace",
        "wiped away",
    ],
    "known": [
        "i know",
        "she knows",
        "we know",
        "it is real",
        "this is real",
        "certain",
        "without doubt",
        "undeniable",
        "proven",
        "obvious",
        "always been",
        "never changes",
        "confirmed",
        "recognized",
    ],
}


def _keyword_score(text: str) -> dict:
    """Keyword-based chromatic scoring for low-confidence Refractor fallback.

    Returns a result dict with temporal/spatial/ontological dicts keyed by
    mode name, matching the Refractor result structure.
    """
    text_lower = text.lower()

    def score_dim(
        keywords_by_mode: dict[str, list[str]], mode_names: list[str]
    ) -> dict:
        raw = {}
        for mode in mode_names:
            count = sum(text_lower.count(kw) for kw in keywords_by_mode.get(mode, []))
            raw[mode] = float(count) + 0.1  # floor avoids all-zero distributions
        total = sum(raw.values())
        return {m: raw[m] / total for m in mode_names}

    return {
        "temporal": score_dim(_TEMPORAL_KEYWORDS, ["past", "present", "future"]),
        "spatial": score_dim(_SPATIAL_KEYWORDS, ["thing", "place", "person"]),
        "ontological": score_dim(
            _ONTOLOGICAL_KEYWORDS, ["imagined", "forgotten", "known"]
        ),
        "confidence": 0.5,  # neutral — keyword scorer has no calibrated confidence
    }


def _blend_scores(
    refractor_result: dict, keyword_result: dict, confidence: float
) -> dict:
    """Blend Refractor and keyword scores when Refractor confidence is low.

    Weights:
      - confidence < 0.1  → 30% Refractor, 70% keyword
      - 0.1 ≤ confidence < 0.2 → 70% Refractor, 30% keyword
      - confidence ≥ 0.2 → 100% Refractor (caller should skip blending)
    """
    if confidence < 0.1:
        w_r, w_k = 0.3, 0.7
    else:
        w_r, w_k = 0.7, 0.3

    blended: dict = {}
    for dim in ("temporal", "spatial", "ontological"):
        r_dim = refractor_result.get(dim, {})
        k_dim = keyword_result.get(dim, {})
        modes = list(r_dim.keys()) or list(k_dim.keys())
        blended[dim] = {
            m: r_dim.get(m, 0.0) * w_r + k_dim.get(m, 0.0) * w_k for m in modes
        }
    # Raise effective confidence so compute_chromatic_match weights it fairly
    blended["confidence"] = min(confidence + 0.15, 0.5)
    return blended


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------


def collect_sub_lyrics(sub_proposal_dirs: list[Path]) -> list[dict]:
    """Collect approved (or all) lyric texts from each sub-proposal directory.

    For each dir, checks melody/candidates/lyrics_review.yml for approved entries;
    falls back to all melody/candidates/lyrics_*.txt files if no review exists.
    Returns list of {source_dir, color, lyrics_text} dicts.
    """
    results = []
    for sub_dir in sub_proposal_dirs:
        sub_dir = Path(sub_dir)
        candidates_dir = sub_dir / "melody" / "candidates"
        if not candidates_dir.exists():
            continue

        review_path = candidates_dir / "lyrics_review.yml"
        # Determine donor color from song_context or chord review
        color = ""
        ctx_path = sub_dir / "song_context.yml"
        if ctx_path.exists():
            with open(ctx_path) as f:
                color = (yaml.safe_load(f) or {}).get("color", "")
        if not color:
            cr_path = sub_dir / "chords" / "review.yml"
            if cr_path.exists():
                with open(cr_path) as f:
                    color = (yaml.safe_load(f) or {}).get("color", "")

        approved_files: list[Path] = []
        if review_path.exists():
            with open(review_path) as f:
                review = yaml.safe_load(f) or {}
            for entry in review.get("candidates", []):
                if entry.get("status") == "approved":
                    txt_path = candidates_dir / entry["file"]
                    if txt_path.exists():
                        approved_files.append(txt_path)

        if not approved_files:
            approved_files = sorted(candidates_dir.glob("lyrics_*.txt"))

        # Also check the promoted lyrics file (melody/lyrics.txt)
        promoted_lyrics = sub_dir / "melody" / "lyrics.txt"
        if promoted_lyrics.exists() and promoted_lyrics not in approved_files:
            approved_files = [promoted_lyrics] + list(approved_files)

        for txt_path in approved_files:
            text = txt_path.read_text(encoding="utf-8").strip()
            if text:
                results.append(
                    {"source_dir": str(sub_dir), "color": color, "lyrics_text": text}
                )

    return results


def _build_white_cutup_prompt(
    meta: dict,
    vocal_sections: list[dict],
    syllable_targets: dict,
    sub_lyrics: list[dict],
    artist_context: str = "",
) -> str:
    """Build the Claude prompt for White lyric cut-up generation.

    Includes sub-lyrics as explicit source material for a Burroughs/Gysin cut-up.
    Falls back to a standard synthesis prompt if sub_lyrics is empty.
    """
    lines = [
        f'You are writing lyrics for "{meta.get("title", "")}" — the White synthesis song.',
        "",
        "SONG METADATA:",
        "  Color: White (synthesis of all colors)",
        f"  BPM: {meta.get('bpm', '')}",
        f"  Time signature: {meta.get('time_sig', '')}",
        f"  Key: {meta.get('key', '')}",
        f"  Concept: {meta.get('concept', '')}",
        "",
    ]

    if sub_lyrics:
        lines += [
            "SOURCE LYRICS (cut-up material from the color sub-songs):",
            "Use these as raw material. Extract phrases, images, and lines.",
            "Recombine and transform them into a coherent new lyric that feels",
            "synthesised rather than collaged — the seams should disappear.",
            "Shared vocabulary, echoed images, and rhythmic callbacks to the source",
            "material are all welcome. Do NOT reproduce complete verses verbatim.",
            "",
        ]
        for src in sub_lyrics:
            color_label = src.get("color") or "unknown"
            lines.append(f"## {color_label}")
            lines.append(src["lyrics_text"])
            lines.append("")
    else:
        lines += [
            "This is a White synthesis song — a convergence of all chromatic themes.",
            "Write lyrics that draw together the threads of memory, place, and transformation.",
            "",
        ]

    if artist_context:
        lines.extend(["", artist_context, ""])

    lines += [
        "SECTIONS TO WRITE:",
        "(Headers are melody loop labels — each maps to one MIDI clip.)",
    ]

    import math as _math

    for sec in vocal_sections:
        name = sec["name"]
        lo, hi = syllable_targets.get(name, (0, 0))
        denom = max(sec["bars"] * sec["play_count"], 1)
        notes_per_bar = sec["total_notes"] / denom
        phrases: list = sec.get("phrases", [])

        lines.extend(
            [
                "",
                f"  [{name}]",
                f"    Bars per loop: {sec['bars']}  ×  {sec['play_count']} occurrence(s)",
                f"    Target syllables: {lo}–{hi}  (≈{notes_per_bar:.1f} notes/bar)",
            ]
        )
        if phrases:
            all_phrases = phrases * sec["play_count"]
            phrase_counts = [p.note_count for p in all_phrases]
            phrase_lo = [_math.floor(n * 0.8) for n in phrase_counts]
            phrase_hi = [_math.ceil(n * 1.15) for n in phrase_counts]
            ranges_str = ", ".join(f"{lo}–{hi}" for lo, hi in zip(phrase_lo, phrase_hi))
            play_note = (
                f" ({len(phrases)} per loop × {sec['play_count']} plays)"
                if sec["play_count"] > 1
                else ""
            )
            lines.extend(
                [
                    f"    Phrases: {len(all_phrases)} phrases{play_note} with {phrase_counts} notes",
                    f"    Syllable targets per phrase: [{ranges_str}]",
                    f"    Write exactly {len(all_phrases)} lines for this section.",
                ]
            )

    lines.extend(
        [
            "",
            "OUTPUT FORMAT:",
            "  Use [loop_label] headers exactly as listed above.",
            "  Write one block per section instance in arrangement order.",
            "  Output only the lyrics — no commentary, no explanations.",
            "  Lines starting with # are ignored.",
            "",
            "Now write the complete White synthesis lyrics:",
        ]
    )

    return "\n".join(lines)


def _build_prompt(
    meta: dict,
    vocal_sections: list[dict],
    syllable_targets: dict,
    artist_context: str = "",
) -> str:
    """Build the Claude prompt for lyric generation.

    meta keys used: title, color, bpm, time_sig, key, concept
    vocal_sections entries use 'name' as the [header] label (loop label).
    """
    color = meta.get("color", "")
    target = get_chromatic_target(color)
    temporal_modes = ["past", "present", "future"]
    spatial_modes = ["thing", "place", "person"]
    ontological_modes = ["imagined", "forgotten", "known"]

    def dominant_mode(modes, dist):
        idx = max(range(len(dist)), key=lambda i: dist[i])
        return modes[idx]

    dominant_temporal = dominant_mode(temporal_modes, target["temporal"])
    dominant_spatial = dominant_mode(spatial_modes, target["spatial"])
    dominant_ontological = dominant_mode(ontological_modes, target["ontological"])

    lines = [
        f'You are writing lyrics for a song titled "{meta.get("title", "")}".',
        "",
        "SONG METADATA:",
        f"  Color: {color}",
        f"  BPM: {meta.get('bpm', '')}",
        f"  Time signature: {meta.get('time_sig', '')}",
        f"  Key: {meta.get('key', '')}",
        f"  Concept: {meta.get('concept', '')}",
        "",
        "CHROMATIC TARGET (the emotional/conceptual space to express):",
        f"  Temporal mode: {dominant_temporal}  "
        "(past=memory/history, present=immediacy, future=anticipation)",
        f"  Spatial mode: {dominant_spatial}  "
        "(thing=object/artifact, place=location/environment, person=human/being)",
        f"  Ontological mode: {dominant_ontological}  "
        "(imagined=fictional/possible, forgotten=lost/erased, known=certain/present)",
        "",
        f"Write lyrics that express the {color} chromatic concept: "
        f"{dominant_temporal}, {dominant_spatial}, {dominant_ontological}.",
        "",
        "SECTIONS TO WRITE:",
        "(Headers are melody loop labels — each maps to one MIDI clip.)",
    ]

    for sec in vocal_sections:
        name = sec["name"]
        lo, hi = syllable_targets.get(name, (0, 0))
        denom = max(sec["bars"] * sec["play_count"], 1)
        notes_per_bar = sec["total_notes"] / denom
        phrases: list[Phrase] = sec.get("phrases", [])

        lines.extend(
            [
                "",
                f"  [{name}]",
                f"    Bars per loop: {sec['bars']}  ×  {sec['play_count']} occurrence(s)",
                f"    Melody contour: {sec['contour']}",
                f"    Target syllables: {lo}–{hi}  (≈{notes_per_bar:.1f} notes/bar)",
            ]
        )

        if phrases:
            # Scale phrase list to cover all plays of this loop
            all_phrases = phrases * sec["play_count"]
            phrase_counts = [p.note_count for p in all_phrases]
            phrase_lo = [math.floor(n * 0.8) for n in phrase_counts]
            phrase_hi = [math.ceil(n * 1.15) for n in phrase_counts]
            ranges_str = ", ".join(f"{lo}–{hi}" for lo, hi in zip(phrase_lo, phrase_hi))
            play_note = (
                f" ({len(phrases)} per loop × {sec['play_count']} plays)"
                if sec["play_count"] > 1
                else ""
            )
            lines.extend(
                [
                    f"    Phrases: {len(all_phrases)} phrases{play_note} with {phrase_counts} notes respectively",
                    f"    Syllable targets per phrase: [{ranges_str}]",
                    f"    IMPORTANT: Write exactly {len(all_phrases)} lines for this section,"
                    " one line per phrase.",
                    "    Each line should contain approximately the syllable count shown.",
                ]
            )

    if artist_context:
        lines.extend(["", artist_context])

    lines.extend(
        [
            "",
            "OUTPUT FORMAT:",
            "  Use [loop_label] headers exactly as listed above.",
            "  Write one block per section instance in arrangement order.",
            "  Output only the lyrics — no commentary, no explanations.",
            "  Lines starting with # are ignored (you may use them for stage directions).",
            "  When phrase counts are given, write exactly that many lines per section.",
            "",
            "Example:",
            "  [melody_verse_alternate]",
            "  First line of verse",
            "  Second line of verse",
            "",
            "  [melody_bridge]",
            "  First line of bridge",
            "  Second line of bridge",
            "",
            "Now write the complete lyrics:",
        ]
    )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# API call
# ---------------------------------------------------------------------------


def _call_api(client, prompt: str, model: str) -> str:
    response = client.messages.create(
        model=model,
        max_tokens=2048,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


# ---------------------------------------------------------------------------
# Section parsing
# ---------------------------------------------------------------------------


def _parse_sections(text: str) -> dict[str, str]:
    """Parse [section_name] headers from lyric text.

    Returns dict of section_name → lyric block (comment lines stripped).
    Section names are lowercased and spaces converted to underscores.
    """
    result: dict[str, str] = {}
    current_section: Optional[str] = None
    current_lines: list[str] = []

    for line in text.splitlines():
        stripped = line.strip()
        header_match = re.match(r"^\[([^\]]+)\]\s*$", stripped)
        if header_match:
            if current_section is not None:
                result[current_section] = "\n".join(current_lines).strip()
            current_section = header_match.group(1).strip().lower().replace(" ", "_")
            current_lines = []
        elif current_section is not None:
            if not stripped.startswith("#"):
                current_lines.append(line)

    if current_section is not None:
        result[current_section] = "\n".join(current_lines).strip()

    return result


# ---------------------------------------------------------------------------
# Review YAML load / init
# ---------------------------------------------------------------------------


def _load_or_init_review(melody_dir: Path, meta: dict, model: str, seed: int) -> dict:
    """Load existing lyrics_review.yml or create a fresh header dict."""
    review_path = melody_dir / LYRICS_REVIEW_FILENAME
    if review_path.exists():
        with open(review_path) as f:
            return yaml.safe_load(f) or {}

    return {
        "production_dir": str(melody_dir.parent),
        "pipeline": "lyric-generation",
        "bpm": meta.get("bpm"),
        "time_sig": meta.get("time_sig"),
        "color": meta.get("color"),
        "generated": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "model": model,
        "scoring_weights": {"chromatic": 1.0},
        "candidates": [],
    }


# ---------------------------------------------------------------------------
# Candidate ID generation
# ---------------------------------------------------------------------------


def _next_candidate_id(review: dict) -> str:
    max_n = 0
    for cand in review.get("candidates", []):
        cid = cand.get("id", "")
        m = re.match(r"lyrics_(\d+)$", cid)
        if m:
            n = int(m.group(1))
            if n > max_n:
                max_n = n
    return f"lyrics_{max_n + 1:02d}"


# ---------------------------------------------------------------------------
# Candidate sync
# ---------------------------------------------------------------------------


def sync_lyric_candidates(melody_dir: Path) -> int:
    """Scan melody/candidates/*.txt for untracked files and add stub entries."""
    review_path = melody_dir / LYRICS_REVIEW_FILENAME
    candidates_dir = melody_dir / "candidates"

    if not review_path.exists():
        print(f"ERROR: No lyrics_review.yml found at {review_path}")
        print("Run the lyric pipeline first to create a lyrics_review.yml base.")
        return 0

    with open(review_path) as f:
        review = yaml.safe_load(f) or {}

    existing_files = {
        Path(c["file"]).name for c in review.get("candidates", []) if c.get("file")
    }
    existing_ids = {c["id"] for c in review.get("candidates", []) if c.get("id")}

    if not candidates_dir.exists():
        print(f"No candidates/ directory at {candidates_dir}")
        return 0

    new_files = sorted(
        f for f in candidates_dir.glob("*.txt") if f.name not in existing_files
    )

    if not new_files:
        print("All candidate files are already tracked in lyrics_review.yml")
        return 0

    added = 0
    for txt_file in new_files:
        stub_id = txt_file.stem
        if stub_id in existing_ids:
            i = 2
            while f"{stub_id}_{i}" in existing_ids:
                i += 1
            stub_id = f"{stub_id}_{i}"

        stub = {
            "id": stub_id,
            "file": f"candidates/{txt_file.name}",
            "status": "pending",
            "notes": "",
        }
        review.setdefault("candidates", []).append(stub)
        existing_ids.add(stub_id)
        print(f"  + {txt_file.name}  →  id: {stub_id}")
        added += 1

    with open(review_path, "w") as f:
        yaml.dump(
            review, f, default_flow_style=False, sort_keys=False, allow_unicode=True
        )

    print(f"\nAdded {added} entries to lyrics_review.yml")
    print(f"Edit {review_path}")
    print("Set status: approved on the entry you want, then run promote_part")
    return added


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run_lyric_pipeline(
    production_dir: str,
    num_candidates: int = 3,
    model: str = "claude-sonnet-4-6",
    seed: int = 42,
    onnx_path: Optional[str] = None,
    skip_scoring: bool = False,
) -> dict:
    """Run the lyric generation pipeline end-to-end.

    Reads vocal sections from arrangement.txt (track 4 = vocal).
    Reads song metadata from the song proposal YAML.
    No production_plan.yml required.

    Returns:
        The lyrics_review.yml dict after writing.
    """
    prod_path = Path(production_dir)
    if not prod_path.exists():
        print(f"ERROR: Production directory not found: {prod_path}")
        sys.exit(1)

    melody_dir = prod_path / "melody"
    arrangement_path = prod_path / "arrangement.txt"

    print("=" * 60)
    print("LYRIC GENERATION PIPELINE")
    print("=" * 60)

    # --- 1. Check arrangement exists ---
    if not arrangement_path.exists():
        print(
            "ERROR: arrangement.txt not found — export from Logic before generating lyrics"
        )
        sys.exit(1)

    # --- 2. Load song metadata from proposal ---
    meta = _find_and_load_proposal(prod_path)
    if not meta:
        # Fall back to chord review for minimal metadata
        chord_review_path = prod_path / "chords" / "review.yml"
        if chord_review_path.exists():
            with open(chord_review_path) as f:
                cr = yaml.safe_load(f) or {}
            meta = {
                "title": "",
                "bpm": int(cr.get("bpm", 120)),
                "time_sig": str(cr.get("time_sig", "4/4")),
                "key": str(cr.get("key", "")),
                "color": str(cr.get("color", "")),
                "concept": "",
                "sounds_like": [],
                "genres": [],
                "mood": [],
                "singer": str(cr.get("singer", "")),
            }
        else:
            print(
                "ERROR: Could not load song metadata (no proposal or chord review found)"
            )
            sys.exit(1)

    # Prefer sounds_like from initial_proposal.yml (Claude-generated before pipeline ran)
    _initial = load_initial_proposal(prod_path)
    if _initial.get("sounds_like"):
        meta["sounds_like"] = _initial["sounds_like"]
    elif not meta.get("sounds_like"):
        meta["sounds_like"] = []

    print(f"Song:  {meta.get('title', '(untitled)')}")
    print(f"Color: {meta.get('color', '')}")
    print(
        f"BPM:   {meta.get('bpm')}  Time: {meta.get('time_sig')}  Key: {meta.get('key', '')}"
    )

    # --- 3. Read vocal sections from arrangement ---
    vocal_sections = read_vocal_sections_from_arrangement(
        arrangement_path, melody_dir, meta["bpm"], meta["time_sig"]
    )
    if not vocal_sections:
        print("ERROR: No melody clips found on track 4 in arrangement.txt")
        print(
            "Export the arrangement from Logic after placing melody clips on track 4."
        )
        sys.exit(1)

    # --- 3b. Extract MIDI phrase structure per section ---
    approved_dir = melody_dir / "approved"
    for sec in vocal_sections:
        midi_path = approved_dir / f"{sec['name']}.mid"
        sec["phrases"] = extract_phrases(midi_path) if midi_path.exists() else []

    print(f"\nVocal sections ({len(vocal_sections)}) from arrangement:")
    for sec in vocal_sections:
        phrase_info = f", {len(sec['phrases'])} phrases" if sec["phrases"] else ""
        print(
            f"  {sec['name']}: {sec['bars']}b × {sec['play_count']}"
            f" = {sec['total_notes']} notes{phrase_info}"
        )

    # --- 4. Syllable targets ---
    syllable_targets = {
        sec["name"]: (
            math.floor(sec["total_notes"] * 0.75),
            math.floor(sec["total_notes"] * 1.05),
        )
        for sec in vocal_sections
    }

    # --- 5. Build prompt ---
    artist_context = load_artist_context(meta.get("sounds_like") or [])
    is_white = str(meta.get("color", "")).strip().capitalize() == "White"
    if is_white:
        # White cut-up mode: collect sub-lyrics from sub_proposals in song_context,
        # falling back to bar_sources in chord review.yml
        ctx = load_song_context(prod_path)
        sub_dirs = [Path(p) for p in (ctx.get("sub_proposals") or [])]
        if not sub_dirs:
            chord_review_path = prod_path / "chords" / "review.yml"
            if chord_review_path.exists():
                with open(chord_review_path) as _f:
                    _cr = yaml.safe_load(_f) or {}
                seen = set()
                for candidate in _cr.get("candidates", []):
                    for bs in candidate.get("bar_sources", []):
                        sd = bs.get("source_dir")
                        if sd and sd not in seen:
                            seen.add(sd)
                            sub_dirs.append(Path(sd))
        sub_lyrics = collect_sub_lyrics(sub_dirs) if sub_dirs else []
        if sub_lyrics:
            print(
                f"\nWhite cut-up mode: collected lyrics from {len(sub_lyrics)} sub-song(s)"
            )
        else:
            print("\nWhite cut-up mode: no sub-lyrics found — using synthesis fallback")
        prompt = _build_white_cutup_prompt(
            meta, vocal_sections, syllable_targets, sub_lyrics, artist_context
        )
    else:
        prompt = _build_prompt(meta, vocal_sections, syllable_targets, artist_context)

    # --- 6. Generate candidates ---
    from anthropic import Anthropic

    client = Anthropic()
    print(f"\nGenerating {num_candidates} lyric candidate(s) via {model}...")
    texts = []
    for i in range(num_candidates):
        print(f"  Candidate {i + 1}/{num_candidates}...")
        text = _call_api(client, prompt, model)
        texts.append(text)

    # --- 7. Score with Refractor (text-only) ---
    scorer_results_map: dict[int, Optional[dict]] = {}
    target = get_chromatic_target(meta.get("color", ""))

    if not skip_scoring:
        print("\nLoading Refractor...")
        try:
            from training.refractor import Refractor

            scorer = Refractor(onnx_path=onnx_path) if onnx_path else Refractor()
            concept_text = (
                meta.get("concept") or f"{meta.get('color', '')} chromatic concept"
            )
            concept_emb = scorer.prepare_concept(concept_text)
            print(f"  Concept encoded ({concept_emb.shape[0]}-dim)")

            scorer_candidates = [{"lyric_text": t} for t in texts]
            scorer_results = scorer.score_batch(
                scorer_candidates, concept_emb=concept_emb
            )
            for result in scorer_results:
                idx = scorer_candidates.index(result["candidate"])
                scorer_results_map[idx] = result
        except Exception as e:
            print(f"  Warning: Refractor unavailable ({e}), skipping scoring")
    else:
        print("\nSkipping Refractor (--skip-scoring)")

    # --- 8. Compute fitting + chromatic match ---
    scored_entries = []
    for idx, text in enumerate(texts):
        result = scorer_results_map.get(idx)

        # Bug 2 fix: blend keyword scores when Refractor confidence is very low
        if result is not None:
            confidence = result.get("confidence", 1.0)
            if confidence < 0.2:
                keyword_result = _keyword_score(text)
                result = _blend_scores(result, keyword_result, confidence)

        chromatic_match = compute_chromatic_match(result, target) if result else 0.0
        fitting = _compute_fitting(text, vocal_sections, melody_dir)
        scored_entries.append(
            {
                "text": text,
                "original_idx": idx,
                "chromatic_result": result,
                "chromatic_match": chromatic_match,
                "fitting": fitting,
            }
        )

    scored_entries.sort(key=lambda e: e["chromatic_match"], reverse=True)

    # --- 9. Write candidate .txt files ---
    candidates_dir = melody_dir / "candidates"
    candidates_dir.mkdir(parents=True, exist_ok=True)

    review = _load_or_init_review(melody_dir, meta, model, seed)

    new_entries = []
    for rank, entry in enumerate(scored_entries):
        cid = _next_candidate_id(review)
        txt_path = candidates_dir / f"{cid}.txt"
        txt_path.write_text(entry["text"], encoding="utf-8")

        result = entry["chromatic_result"]
        if result is not None:
            chromatic_block = _to_python(
                {
                    "temporal": result["temporal"],
                    "spatial": result["spatial"],
                    "ontological": result["ontological"],
                    "confidence": round(result["confidence"], 4),
                    "match": round(entry["chromatic_match"], 4),
                }
            )
        else:
            chromatic_block = None
        fitting_block = _to_python(entry["fitting"])

        candidate_entry = {
            "id": cid,
            "file": f"candidates/{cid}.txt",
            "rank": rank + 1,
            "chromatic": chromatic_block,
            "fitting": fitting_block,
            "status": "pending",
            "notes": "",
        }
        review.setdefault("candidates", []).append(candidate_entry)
        new_entries.append(candidate_entry)

    # --- 10. Save review YAML ---
    review_path = melody_dir / LYRICS_REVIEW_FILENAME
    with open(review_path, "w") as f:
        yaml.dump(
            review, f, default_flow_style=False, sort_keys=False, allow_unicode=True
        )

    # --- 11. Summary ---
    print(f"\n{'=' * 60}")
    print("LYRIC GENERATION COMPLETE")
    print(f"{'=' * 60}")
    print(f"Candidates: {len(new_entries)}")
    print(f"Review:     {review_path}")
    print()
    print(f"{'Rank':<5} {'ID':<12} {'Match':<8} {'Overall Fit'}")
    print("-" * 40)
    for entry in new_entries:
        chromatic = entry.get("chromatic") or {}
        match = chromatic.get("match", None)
        match_str = f"{match:.3f}" if match is not None else "n/a  "
        overall = entry["fitting"].get("overall", "?")
        print(f"  #{entry['rank']:<3} {entry['id']:<12} {match_str}    {overall}")

    print(f"\nNext: Edit {review_path} to approve a candidate")
    print(
        f"Then: python -m app.generators.midi.production.promote_part --review {review_path}"
    )

    return review


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Lyric generation pipeline — generate, score, and review lyrics",
    )
    parser.add_argument(
        "--production-dir",
        required=True,
        help="Song production directory (must contain arrangement.txt)",
    )
    parser.add_argument(
        "--sync-candidates",
        action="store_true",
        help=(
            "Scan candidates/*.txt for files not in lyrics_review.yml and add stubs. "
            "Does not regenerate or wipe anything."
        ),
    )
    parser.add_argument(
        "--num-candidates",
        type=int,
        default=3,
        help="Number of lyric drafts to generate (default: 3)",
    )
    parser.add_argument(
        "--model",
        default="claude-sonnet-4-6",
        help="Claude model to use (default: claude-sonnet-4-6)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for review header (default: 42)",
    )
    parser.add_argument(
        "--onnx-path",
        default=None,
        help="Path to refractor.onnx (default: training/data/refractor.onnx)",
    )
    parser.add_argument(
        "--skip-scoring",
        action="store_true",
        help="Skip Refractor (useful when torch/DeBERTa unavailable locally)",
    )

    args = parser.parse_args()

    if args.sync_candidates:
        melody_dir = Path(args.production_dir) / "melody"
        sync_lyric_candidates(melody_dir)
        return

    run_lyric_pipeline(
        production_dir=args.production_dir,
        num_candidates=args.num_candidates,
        model=args.model,
        seed=args.seed,
        onnx_path=args.onnx_path,
        skip_scoring=args.skip_scoring,
    )


if __name__ == "__main__":
    main()
