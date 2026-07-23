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
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import mido
import pronouncing
import yaml
from dotenv import load_dotenv

from white_composition.init_production import (
    load_initial_proposal,
    load_song_context,
)  # noqa: E402
from white_composition.production_plan import (  # noqa: E402
    _infer_repeat_type,
    _normalize_repeat_type,
)
from white_core.enums.lyric_repeat_type import LyricRepeatType
from white_generation.artist_catalog import load_artist_context  # noqa: E402
from white_generation.lyric_negative_constraints import (  # noqa: E402
    format_for_prompt as format_negative_constraints_for_prompt,
)
from white_generation.lyric_negative_constraints import (  # noqa: E402
    generate_constraints as generate_lyric_negative_constraints,
)
from white_generation.lyric_negative_constraints import (  # noqa: E402
    load_constraints as load_lyric_negative_constraints,
)
from white_generation.lyric_negative_constraints import (  # noqa: E402
    write_constraints as write_lyric_negative_constraints,
)
from white_generation.pipelines.chord_pipeline import (  # noqa: E402
    _to_python,
    compute_chromatic_match,
    get_chromatic_target,
)


def _count_syllables(text: str) -> int:
    """Estimate syllable count — at least one syllable per word."""
    if not text:
        return 0
    total = 0
    for word in text.split():
        total += max(1, len(re.findall(r"[aeiouAEIOU]+", word)))
    return total


load_dotenv()

LYRICS_REVIEW_FILENAME = "lyrics_review.yml"

MELODY_CHANNEL = 4  # fallback when auto-detection finds nothing


def _detect_melody_channel(clips: list[dict], fallback: int = MELODY_CHANNEL) -> int:
    """Return the track number that carries vocal melody clips.

    Counts only non-instrumental melody_ clips (i.e. excludes anything ending
    in _inst).  This prevents melody_hook_inst on the lead channel from being
    chosen over the vocal channel when all named melody_ clips happen to be
    instrumental.  Falls back to MELODY_CHANNEL when no qualifying clips exist.
    """
    counts: dict[int, int] = {}
    for c in clips:
        name = c["clip_name"]
        if name.startswith("melody_") and not name.endswith("_inst"):
            counts[c["channel"]] = counts.get(c["channel"], 0) + 1
    return max(counts, key=lambda ch: counts[ch]) if counts else fallback


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
    """Group notes into phrases separated by actual rests.

    A new phrase begins when the gap between one note's end (note-off) and the
    next note's onset exceeds rest_threshold_beats (default 0.5 beats). Legato/
    sustained melodies — where one note's onset immediately follows the prior
    note's end with no rest — stay in a single phrase no matter how far apart
    their onsets are; only real silence between notes splits a phrase.

    Returns a list of Phrase objects in order.
    """
    try:
        mid = mido.MidiFile(str(midi_path))
    except Exception:
        return []

    ticks_per_beat = mid.ticks_per_beat or 480
    threshold_ticks = int(rest_threshold_beats * ticks_per_beat)

    # Pair each note-on with its matching note-off (per channel+pitch, FIFO —
    # handles retriggered/overlapping notes on the same pitch correctly) to get
    # real (onset, end) spans rather than just onset points.
    pending: dict[tuple[int, int], list[int]] = {}
    note_spans: list[tuple[int, int]] = []
    for track in mid.tracks:
        abs_tick = 0
        for msg in track:
            abs_tick += msg.time
            if msg.type == "note_on" and msg.velocity > 0:
                pending.setdefault((msg.channel, msg.note), []).append(abs_tick)
            elif msg.type == "note_off" or (
                msg.type == "note_on" and msg.velocity == 0
            ):
                key = (msg.channel, msg.note)
                if pending.get(key):
                    start = pending[key].pop(0)
                    note_spans.append((start, abs_tick))

    if not note_spans:
        return []

    note_spans.sort(key=lambda span: span[0])

    phrases: list[Phrase] = []
    phrase_start = note_spans[0][0]
    phrase_end = note_spans[0][1]
    phrase_note_count = 1

    for start, end in note_spans[1:]:
        if start - phrase_end > threshold_ticks:
            phrases.append(
                Phrase(
                    start_tick=phrase_start,
                    end_tick=phrase_end,
                    note_count=phrase_note_count,
                )
            )
            phrase_start = start
            phrase_note_count = 1
        else:
            phrase_note_count += 1
        phrase_end = max(phrase_end, end)

    phrases.append(
        Phrase(
            start_tick=phrase_start,
            end_tick=phrase_end,
            note_count=phrase_note_count,
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


def _parse_bar_beat_bars(tc: str) -> Optional[int]:
    """Parse a Logic bar/beat string like '8 0 0 0', returning bar count.

    Returns None if the string looks like a timecode (contains ':') or cannot
    be parsed.
    """
    tc = tc.strip()
    if ":" in tc:
        return None
    parts = tc.split()
    if not parts:
        return None
    try:
        return int(parts[0])
    except ValueError:
        return None


def parse_arrangement(arrangement_path: Path) -> list[dict]:
    """Parse arrangement.txt into a list of clip dicts.

    Handles two Logic export formats:
    - Timecode format: HH:MM:SS:FF  clip_name  channel  HH:MM:SS:FF
    - Bar/beat format: B B B B      clip_name  channel  B B B B

    Each dict: {timecode_secs, clip_name, channel, duration_secs, duration_bars}
    duration_bars is set when bar/beat format is detected; duration_secs may be 0.0.
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
                duration_bars = _parse_bar_beat_bars(parts[3])
                start_bars = _parse_bar_beat_bars(parts[0])
                clips.append(
                    {
                        "timecode_secs": timecode_secs,
                        "clip_name": clip_name,
                        "channel": channel,
                        "duration_secs": duration_secs,
                        "duration_bars": duration_bars,
                        "start_bars": start_bars,
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
    concept, genres, mood, singer, sounds_like, rhyme_scheme.
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
            from white_composition.production_plan import (
                load_song_proposal_unified,
            )

            unified = load_song_proposal_unified(candidate, thread_dir=thread_path)

            with open(candidate) as f:
                raw_proposal = yaml.safe_load(f) or {}
            rhyme_scheme = raw_proposal.get("rhyme_scheme") or {}

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
                "rhyme_scheme": rhyme_scheme,
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
    production_dir: Optional[Path] = None,
    melody_channel: int = MELODY_CHANNEL,
) -> list[dict]:
    """Extract vocal sections from arrangement.txt.

    The melody_channel identifies which track carries vocal clips (default: 4).
    Override with --melody-channel when using non-standard Logic track layouts.
    Returns one entry per clip instance in arrangement order.

    Each entry: {approved_label, name, bars, play_count, total_notes, contour,
                 lyric_repeat_type}

    lyric_repeat_type is loaded from production_plan.yml when production_dir is
    given; otherwise inferred from the label.  For 'exact' labels, instances
    beyond the first are tagged 'exact_repeat' so the prompt builder can skip them.
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

    # Load lyric_repeat_type overrides from production_plan.yml
    repeat_type_by_label: dict[str, str] = {}
    if production_dir is not None:
        plan_path = production_dir / "production_plan.yml"
        if plan_path.exists():
            with open(plan_path) as f:
                plan_data = yaml.safe_load(f) or {}
            for sec in plan_data.get("sections", []):
                lbl = sec.get("name", "")
                rt = _normalize_repeat_type(sec.get("lyric_repeat_type"))
                if lbl and rt != "fresh":
                    # Only store explicit overrides; 'fresh' is the default anyway
                    repeat_type_by_label[lbl] = rt

    # Collect melody-channel clips in arrangement order — one entry per instance.
    # Duplicate labels get _2, _3 suffixes; the prompt uses these suffixed names
    # as [headers] so Claude writes one block per arrangement instance.
    resolved_channel = _detect_melody_channel(clips, fallback=melody_channel)
    melody_clips = [c for c in clips if c["channel"] == resolved_channel]
    label_seen_count: dict[str, int] = {}
    # Track first-seen instance key for exact labels (for exact_repeat copying)
    exact_first_instance: dict[str, str] = {}

    approved_dir = melody_dir / "approved"
    sections = []
    for clip in melody_clips:
        label = clip["clip_name"]
        label_seen_count[label] = label_seen_count.get(label, 0) + 1
        n = label_seen_count[label]
        instance_key = label if n == 1 else f"{label}_{n}"

        if clip.get("duration_bars") is not None:
            bars = max(clip["duration_bars"], 1)
        else:
            bars = max(round(clip["duration_secs"] / secs_per_bar), 1)
        midi_path = approved_dir / f"{label}.mid"
        per_loop_notes = _count_notes(midi_path) if midi_path.exists() else 0

        # Determine repeat type: plan override > infer from label
        base_repeat_type = repeat_type_by_label.get(label) or _infer_repeat_type(label)

        if base_repeat_type == LyricRepeatType.EXACT and n == 1:
            exact_first_instance[label] = instance_key
            lyric_repeat_type = LyricRepeatType.EXACT
        elif base_repeat_type == LyricRepeatType.EXACT and n > 1:
            lyric_repeat_type = LyricRepeatType.EXACT_REPEAT
        else:
            lyric_repeat_type = base_repeat_type

        sections.append(
            {
                "approved_label": label,  # base label → MIDI filename (strip _N suffix)
                "name": instance_key,  # unique instance key used as [header]
                "bars": bars,
                "play_count": 1,
                "total_notes": per_loop_notes,
                "contour": contour_by_label.get(label, "stepwise"),
                "lyric_repeat_type": lyric_repeat_type,
                "exact_source": exact_first_instance.get(
                    label
                ),  # for exact_repeat copies
            }
        )

    return sections


# ---------------------------------------------------------------------------
# Syllable fitting
# ---------------------------------------------------------------------------

_VERDICT_ORDER = ["spacious", "paste-ready", "tight but workable", "splits needed"]

# Phrases at or below this note count get a widened syllable target range in the
# generation prompt — the strict 0.8x-1.15x multiplier collapses to a 1-2 syllable
# window for very short phrases, which pushed candidates toward the same small pool
# of monosyllabic words instead of treating melisma as a legitimate choice.
SHORT_PHRASE_NOTE_THRESHOLD = 3


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


def _phrase_syllable_range(note_count: int) -> tuple[int, int]:
    """Return (lo, hi) syllable target bounds for a phrase's note count.

    Phrases at or below SHORT_PHRASE_NOTE_THRESHOLD get the range widened on both
    ends so a word or short phrase sustained across the notes (melisma) is a legal,
    encouraged choice rather than the prompt implicitly pinning every short phrase
    to a near-single-syllable target.
    """
    lo = math.floor(note_count * 0.8)
    hi = math.ceil(note_count * 1.15)
    if note_count <= SHORT_PHRASE_NOTE_THRESHOLD:
        lo = max(0, lo - 1)
        hi = hi + 1
    return lo, hi


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
        repeat_type = _normalize_repeat_type(sec.get("lyric_repeat_type"))

        # exact_repeat instances copy fitting from their source instance
        if repeat_type == LyricRepeatType.EXACT_REPEAT:
            source_key = sec.get("exact_source") or re.sub(r"_\d+$", "", name)
            if source_key in result:
                result[name] = result[source_key]
            continue

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
# Rhyme scheme derivation
# ---------------------------------------------------------------------------

_NO_RHYME_SCHEME = "none"


def _default_rhyme_scheme(line_count: int) -> str:
    """Return the default scheme letters for a rhyme-eligible line count.

    2 lines -> AA (couplet). 4 lines -> XAXA (lines 2 and 4 rhyme; the "ballad
    meter" most real song lyrics use). Longer counts generalize the ballad
    meter: every even-numbered line (2, 4, 6, ...) shares one rhyme, odd-
    numbered lines are left free ("X") — this is what keeps real sections
    (commonly 6-14 lines once phrases are extracted correctly) from silently
    falling back to fully unrhymed. Fewer than 2 lines can't rhyme at all.
    """
    if line_count < 2:
        return _NO_RHYME_SCHEME
    if line_count == 2:
        return "AA"
    return "".join("A" if i % 2 == 1 else "X" for i in range(line_count))


def _rhyme_base_label(label: str) -> str:
    """Strip a trailing _<digit> suffix so parallel sections (verse_1, verse_2)
    resolve to the same structural base ("verse") for scheme reuse."""
    return re.sub(r"_\d+$", "", label)


def assign_rhyme_schemes(
    vocal_sections: list[dict],
    rhyme_scheme_overrides: Optional[dict] = None,
) -> dict[str, str]:
    """Assign a rhyme scheme string to each non-exact_repeat vocal section.

    Priority per section: explicit override (matched by instance name,
    approved_label, or structural base label) > count-based default. Sections
    sharing a structural base label (e.g. verse_1, verse_2) reuse whichever
    scheme was assigned to the first one encountered.
    """
    overrides = rhyme_scheme_overrides or {}
    schemes: dict[str, str] = {}
    scheme_by_base: dict[str, str] = {}

    for sec in vocal_sections:
        repeat_type = _normalize_repeat_type(sec.get("lyric_repeat_type"))
        if repeat_type == LyricRepeatType.EXACT_REPEAT:
            continue

        name = sec["name"]
        approved_label = sec.get("approved_label", name)
        base = _rhyme_base_label(approved_label)

        if base in scheme_by_base:
            schemes[name] = scheme_by_base[base]
            continue

        override = (
            overrides.get(name) or overrides.get(approved_label) or overrides.get(base)
        )
        if override:
            scheme = _NO_RHYME_SCHEME if str(override).lower() == "none" else override
        else:
            phrases = sec.get("phrases") or []
            line_count = len(phrases) * sec.get("play_count", 1)
            scheme = _default_rhyme_scheme(line_count)

        scheme_by_base[base] = scheme
        schemes[name] = scheme

    return schemes


def rhyme_scheme_line_pairs(scheme: str) -> list[tuple[int, int, str]]:
    """Return (line_i, line_j, letter) 1-indexed consecutive pairs that must rhyme.

    'X' letters are free (no constraint). A scheme of 'none' (or empty) has
    no pairs at all.
    """
    if not scheme or scheme.lower() == _NO_RHYME_SCHEME:
        return []

    groups: dict[str, list[int]] = {}
    for i, letter in enumerate(scheme, start=1):
        if letter.upper() == "X":
            continue
        groups.setdefault(letter.upper(), []).append(i)

    pairs: list[tuple[int, int, str]] = []
    for letter, lines in groups.items():
        for a, b in zip(lines, lines[1:]):
            pairs.append((a, b, letter))
    return pairs


def _rhyme_scheme_prompt_text(scheme: str) -> Optional[str]:
    """Human-readable rhyme instruction for a scheme string, or None if unrhymed."""
    pairs = rhyme_scheme_line_pairs(scheme)
    if not pairs:
        return None

    parts = [f"lines {a} and {b} rhyme" for a, b, _ in pairs]
    free = [i for i, letter in enumerate(scheme, start=1) if letter.upper() == "X"]
    text = f"Rhyme scheme {scheme.upper()}: " + "; ".join(parts)
    if free:
        noun = "line" if len(free) == 1 else "lines"
        text += f"; {noun} {', '.join(str(i) for i in free)} free (no rhyme required)"
    return text


# ---------------------------------------------------------------------------
# Rhyme verification (CMUdict + suffix-heuristic fallback)
# ---------------------------------------------------------------------------


def _line_final_word(line: str) -> str:
    """Return the last word-like token in a line, stripped of punctuation."""
    words = re.findall(r"[A-Za-z']+", line)
    return words[-1] if words else ""


def _cmudict_rhyme_key(word: str) -> Optional[str]:
    """Return the CMUdict rhyming part (stressed vowel onward) for a word, or
    None if the word isn't in the dictionary."""
    cleaned = re.sub(r"[^A-Za-z']", "", word).lower()
    if not cleaned:
        return None
    phones = pronouncing.phones_for_word(cleaned)
    if not phones:
        return None
    return pronouncing.rhyming_part(phones[0])


def _suffix_rhyme_heuristic(word_a: str, word_b: str) -> bool:
    """Same-family suffix heuristic for words absent from CMUdict — compares
    the last 2-3 letters. A crude but cheap same-family signal for invented
    or portmanteau vocabulary CMUdict has no entry for."""
    a = re.sub(r"[^A-Za-z']", "", word_a).lower()
    b = re.sub(r"[^A-Za-z']", "", word_b).lower()
    if not a or not b:
        return False
    for n in (3, 2):
        if len(a) >= n and len(b) >= n and a[-n:] == b[-n:]:
            return True
    return False


def check_rhyme_pair(word_a: str, word_b: str) -> dict:
    """Compare two line-final words for rhyme.

    Returns {"status": "match"|"fail"|"maybe", "method": "cmudict"|"suffix"|"unresolved"}.
    "match"/"fail" require both words to resolve via CMUdict, with method
    "cmudict". When either word is out-of-dictionary, the suffix heuristic is
    tried: a plausible match gives method "suffix", no match gives method
    "unresolved" — but the status is always "maybe" either way, since CMUdict
    simply can't judge this project's invented/portmanteau vocabulary, so it
    must never drive a revision request.
    """
    key_a = _cmudict_rhyme_key(word_a)
    key_b = _cmudict_rhyme_key(word_b)
    if key_a is not None and key_b is not None:
        return {"status": "match" if key_a == key_b else "fail", "method": "cmudict"}
    plausible = _suffix_rhyme_heuristic(word_a, word_b)
    return {"status": "maybe", "method": "suffix" if plausible else "unresolved"}


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


_HOOK_GUIDANCE_LINE = (
    "    HOOK GUIDANCE: Write a short, highly repeatable hook rather than"
    " narrative/descriptive verse content. Prefer a single strong central"
    " phrase (the song or section title works well). Repeating the same line"
    " or phrase within this section is fine. Favor simpler, more repetitive"
    " vocabulary than the surrounding verses."
)

_FICTION_FRAMING_LINE = (
    "This is a fictional concept-album songwriting exercise for an experimental"
    " music project. Any 'secret', 'code', 'encoding', 'protocol', or 'spy'"
    " language in the concept below describes a literary device (e.g. an"
    " acrostic, a thematic motif, a narrative conceit) that shapes the song's"
    " structure — it is not a request to produce real covert communication."
)


def _build_white_cutup_prompt(
    meta: dict,
    vocal_sections: list[dict],
    syllable_targets: dict,
    sub_lyrics: list[dict],
    artist_context: str = "",
    negative_constraints_block: str = "",
    rhyme_schemes: Optional[dict] = None,
) -> str:
    """Build the Claude prompt for White lyric cut-up generation.

    Includes sub-lyrics as explicit source material for a Burroughs/Gysin cut-up.
    Falls back to a standard synthesis prompt if sub_lyrics is empty.
    """
    lines = [
        f'You are writing lyrics for "{meta.get("title", "")}" — the White synthesis song.',
        "",
        _FICTION_FRAMING_LINE,
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

    if negative_constraints_block:
        lines.extend(["", negative_constraints_block, ""])

    lines += [
        "SECTIONS TO WRITE:",
        "(Headers are melody loop labels — each maps to one MIDI clip.)",
    ]

    variation_count_cutup: dict[str, int] = {}
    rhyme_schemes = rhyme_schemes or {}

    for sec in vocal_sections:
        repeat_type = _normalize_repeat_type(sec.get("lyric_repeat_type"))

        if repeat_type == LyricRepeatType.EXACT_REPEAT:
            continue

        name = sec["name"]
        base_label = sec.get("approved_label", name)
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

        rhyme_text = _rhyme_scheme_prompt_text(rhyme_schemes.get(name, ""))
        if rhyme_text:
            lines.append(f"    {rhyme_text}")

        if repeat_type == LyricRepeatType.EXACT:
            lines.append(
                "    # This section repeats verbatim — write it once, it will be reused"
            )
            lines.append(_HOOK_GUIDANCE_LINE)
        elif repeat_type == LyricRepeatType.VARIATION:
            variation_count_cutup[base_label] = (
                variation_count_cutup.get(base_label, 0) + 1
            )
            n = variation_count_cutup[base_label]
            if n > 1:
                lines.append(
                    f"    # Variation {n} of {base_label}: same meter and rhyme scheme"
                    f" as {base_label}, but new images and lines"
                )

        if phrases:
            all_phrases = phrases * sec["play_count"]
            phrase_counts = [p.note_count for p in all_phrases]
            phrase_ranges = [_phrase_syllable_range(n) for n in phrase_counts]
            ranges_str = ", ".join(f"{lo}–{hi}" for lo, hi in phrase_ranges)
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
            if any(n <= SHORT_PHRASE_NOTE_THRESHOLD for n in phrase_counts):
                lines.append(
                    "    Some phrases above have very few notes — for those, a single"
                    " word or short phrase sustained across the notes (melisma) is a"
                    " good choice; don't default to one syllable per note."
                )

    lines.extend(
        [
            "",
            "OUTPUT FORMAT:",
            "  Use [loop_label] headers exactly as listed above.",
            "  Write one block per unique section (exact sections appear once — the",
            "  arrangement handles repetition). Variation instances each get their own block.",
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
    negative_constraints_block: str = "",
    rhyme_schemes: Optional[dict] = None,
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
        _FICTION_FRAMING_LINE,
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

    # Track variation instance counts per base label for numbering
    variation_count: dict[str, int] = {}
    rhyme_schemes = rhyme_schemes or {}

    for sec in vocal_sections:
        repeat_type = _normalize_repeat_type(sec.get("lyric_repeat_type"))

        # exact_repeat instances are skipped — they reuse the first block
        if repeat_type == LyricRepeatType.EXACT_REPEAT:
            continue

        name = sec["name"]
        base_label = sec.get("approved_label", name)
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

        rhyme_text = _rhyme_scheme_prompt_text(rhyme_schemes.get(name, ""))
        if rhyme_text:
            lines.append(f"    {rhyme_text}")

        if repeat_type == LyricRepeatType.EXACT:
            lines.append(
                "    # This section repeats verbatim — write it once, it will be reused"
            )
            lines.append(_HOOK_GUIDANCE_LINE)
        elif repeat_type == LyricRepeatType.VARIATION:
            variation_count[base_label] = variation_count.get(base_label, 0) + 1
            n = variation_count[base_label]
            if n > 1:
                lines.append(
                    f"    # Variation {n} of {base_label}: same meter and rhyme scheme"
                    f" as {base_label}, but new images and lines"
                )

        if phrases:
            # Scale phrase list to cover all plays of this loop
            all_phrases = phrases * sec["play_count"]
            phrase_counts = [p.note_count for p in all_phrases]
            phrase_ranges = [_phrase_syllable_range(n) for n in phrase_counts]
            ranges_str = ", ".join(f"{lo}–{hi}" for lo, hi in phrase_ranges)
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
            if any(n <= SHORT_PHRASE_NOTE_THRESHOLD for n in phrase_counts):
                lines.append(
                    "    Some phrases above have very few notes — for those, a single"
                    " word or short phrase sustained across the notes (melisma) is a"
                    " good choice; don't default to one syllable per note."
                )

    if artist_context:
        lines.extend(["", artist_context])

    if negative_constraints_block:
        lines.extend(["", negative_constraints_block])

    lines.extend(
        [
            "",
            "OUTPUT FORMAT:",
            "  Use [loop_label] headers exactly as listed above.",
            "  Write one block per unique section (exact sections appear once — the",
            "  arrangement handles repetition). Variation instances each get their own block.",
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


def _call_messages(client, messages: list[dict], model: str) -> str:
    response = client.messages.create(model=model, max_tokens=2048, messages=messages)
    if not response.content:
        raise RuntimeError(
            f"Claude API returned no content blocks (stop_reason={response.stop_reason!r}). "
            "This usually means the model declined the request or the response was "
            "truncated before any output — check the prompt for content that may have "
            "triggered a refusal, then retry."
        )
    block = response.content[0]
    if block.type != "text":
        raise RuntimeError(
            f"Claude API returned a '{block.type}' content block "
            f"(stop_reason={response.stop_reason!r}) where lyric text was expected."
        )
    return block.text


def _call_api(client, prompt: str, model: str) -> str:
    return _call_messages(client, [{"role": "user", "content": prompt}], model)


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
# Verify / revise loop
# ---------------------------------------------------------------------------

MAX_REVISION_TURNS = 2


def _check_candidate(
    text: str,
    vocal_sections: list[dict],
    rhyme_schemes: dict[str, str],
) -> dict:
    """Check a lyric draft against per-phrase syllable targets and rhyme pairs.

    Returns {"syllable_issues": [...], "rhyme_issues": [...]}. Only "fail"
    rhyme checks are included — "match" and "maybe" are not issues.
    EXACT_REPEAT sections are skipped (they aren't separately generated).
    """
    parsed = _parse_sections(text)
    syllable_issues: list[dict] = []
    rhyme_issues: list[dict] = []

    for sec in vocal_sections:
        repeat_type = _normalize_repeat_type(sec.get("lyric_repeat_type"))
        if repeat_type == LyricRepeatType.EXACT_REPEAT:
            continue

        name = sec["name"]
        lyric_text = parsed.get(name, "")
        lines = [
            ln.strip()
            for ln in lyric_text.splitlines()
            if ln.strip() and not ln.strip().startswith("#")
        ]

        phrases: list[Phrase] = sec.get("phrases") or []
        all_phrases = phrases * sec.get("play_count", 1)
        for i, phrase in enumerate(all_phrases):
            lo, hi = _phrase_syllable_range(phrase.note_count)
            if i >= len(lines):
                # Missing line entirely — flag it so the revision turn asks
                # for it, rather than silently accepting an incomplete section.
                syllable_issues.append(
                    {
                        "section": name,
                        "line_idx": i + 1,
                        "text": "",
                        "syllables": 0,
                        "target": (lo, hi),
                    }
                )
                continue
            syl = _count_syllables(lines[i])
            if syl < lo or syl > hi:
                syllable_issues.append(
                    {
                        "section": name,
                        "line_idx": i + 1,
                        "text": lines[i],
                        "syllables": syl,
                        "target": (lo, hi),
                    }
                )

        scheme = rhyme_schemes.get(name, _NO_RHYME_SCHEME)
        for a, b, letter in rhyme_scheme_line_pairs(scheme):
            if a - 1 >= len(lines) or b - 1 >= len(lines):
                continue
            word_a = _line_final_word(lines[a - 1])
            word_b = _line_final_word(lines[b - 1])
            if not word_a or not word_b:
                continue
            check = check_rhyme_pair(word_a, word_b)
            if check["status"] == "fail":
                rhyme_issues.append(
                    {
                        "section": name,
                        "line_a": a,
                        "line_b": b,
                        "letter": letter,
                        "word_a": word_a,
                        "word_b": word_b,
                    }
                )

    return {"syllable_issues": syllable_issues, "rhyme_issues": rhyme_issues}


def _build_revision_message(issues: dict) -> str:
    """Build a follow-up turn listing only the failing lines and why."""
    lines = [
        "The draft above has a few lines that need fixing. Please revise only",
        "the lines listed below and return the FULL corrected lyrics text",
        "(all sections, with [headers] intact — not just the fixed lines).",
        "",
    ]
    for issue in issues["syllable_issues"]:
        lo, hi = issue["target"]
        if not issue["text"]:
            lines.append(
                f"- [{issue['section']}] line {issue['line_idx']}: MISSING — add a"
                f" line with roughly {lo}-{hi} syllables."
            )
        else:
            lines.append(
                f'- [{issue["section"]}] line {issue["line_idx"]}: "{issue["text"]}"'
                f" has {issue['syllables']} syllables, target is {lo}-{hi}."
            )
    for issue in issues["rhyme_issues"]:
        lines.append(
            f"- [{issue['section']}] lines {issue['line_a']} and {issue['line_b']}"
            f' (rhyme group {issue["letter"]}): "{issue["word_a"]}" and'
            f' "{issue["word_b"]}" do not rhyme.'
        )
    return "\n".join(lines)


def generate_lyric_candidate(
    client,
    prompt: str,
    model: str,
    vocal_sections: list[dict],
    rhyme_schemes: dict[str, str],
    max_revisions: int = MAX_REVISION_TURNS,
) -> tuple[str, dict]:
    """Generate a lyric draft, then verify and revise it up to max_revisions times.

    Checks syllable targets and rhyme-scheme pairs after each draft. When any
    check fails, sends a follow-up turn (same conversation) naming only the
    failing lines. Stops as soon as a draft passes, or after max_revisions
    follow-up turns — whichever comes first — and returns the best-effort
    draft either way.

    Returns (final_text, outcome) where outcome records the initial/final
    issue counts and how many revision turns were used.
    """
    messages: list[dict] = [{"role": "user", "content": prompt}]
    text = _call_messages(client, messages, model)
    messages.append({"role": "assistant", "content": text})

    issues = _check_candidate(text, vocal_sections, rhyme_schemes)
    outcome = {
        "turns_used": 0,
        "initial_syllable_issues": len(issues["syllable_issues"]),
        "initial_rhyme_issues": len(issues["rhyme_issues"]),
    }

    turns_used = 0
    while (
        issues["syllable_issues"] or issues["rhyme_issues"]
    ) and turns_used < max_revisions:
        messages.append({"role": "user", "content": _build_revision_message(issues)})
        text = _call_messages(client, messages, model)
        messages.append({"role": "assistant", "content": text})
        turns_used += 1
        issues = _check_candidate(text, vocal_sections, rhyme_schemes)

    outcome["turns_used"] = turns_used
    outcome["final_syllable_issues"] = len(issues["syllable_issues"])
    outcome["final_rhyme_issues"] = len(issues["rhyme_issues"])
    return text, outcome


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
        "seed.logicx": seed,
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


def _resolve_album_dir(production_dir: Path) -> Path:
    """Resolve the album (shrink_wrapped) root from a production directory.

    production_dir is <album_dir>/<thread_slug>/production/<production_slug>, so the
    album root is three levels up. Callers only ever check for a specific filename's
    existence under the result, so an unconventional layout just resolves to a
    directory with no matching file rather than raising.
    """
    return production_dir.parent.parent.parent


def run_lyric_pipeline(
    production_dir: str,
    num_candidates: int = 3,
    model: str = "claude-sonnet-4-6",
    seed: int = 42,
    onnx_path: Optional[str] = None,
    skip_scoring: bool = False,
    melody_channel: int = MELODY_CHANNEL,
    arrangement: Optional[str] = None,
    refresh_constraints: bool = False,
) -> dict:
    """Run the lyric generation pipeline end-to-end.

    Reads vocal sections from arrangement.txt (melody_channel = vocal, default 4).
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
    arrangement_path = (
        Path(arrangement) if arrangement else prod_path / "arrangement.txt"
    )

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
                "rhyme_scheme": {},
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
        arrangement_path,
        melody_dir,
        meta["bpm"],
        meta["time_sig"],
        production_dir=prod_path,
        melody_channel=melody_channel,
    )
    if not vocal_sections:
        print(
            f"ERROR: No melody clips found on track {melody_channel} in arrangement.txt"
        )
        print(
            f"Export the arrangement from Logic after placing melody clips on track {melody_channel}."
        )
        sys.exit(1)

    # --- 3b. Extract MIDI phrase structure per section ---
    approved_dir = melody_dir / "approved"
    for sec in vocal_sections:
        # Use approved_label (base label) — MIDI files are stored under the base
        # label even when the instance key has a _2/_3 suffix.
        midi_path = approved_dir / f"{sec['approved_label']}.mid"
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

    # --- 4a. Rhyme schemes ---
    rhyme_schemes = assign_rhyme_schemes(vocal_sections, meta.get("rhyme_scheme"))

    # --- 4b. Lyric negative constraints (album-wide word/imagery avoidance) ---
    album_dir = _resolve_album_dir(prod_path)
    if refresh_constraints:
        constraints = generate_lyric_negative_constraints(album_dir)
        write_lyric_negative_constraints(
            album_dir / "lyrics_negative_constraints.yml", constraints
        )
        print(
            f"Refreshed lyrics_negative_constraints.yml "
            f"({len(constraints['overused_words'])} overused word(s))"
        )
    else:
        constraints = load_lyric_negative_constraints(album_dir)
    negative_constraints_block = (
        format_negative_constraints_for_prompt(constraints) if constraints else ""
    )

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
            meta,
            vocal_sections,
            syllable_targets,
            sub_lyrics,
            artist_context,
            negative_constraints_block,
            rhyme_schemes,
        )
    else:
        prompt = _build_prompt(
            meta,
            vocal_sections,
            syllable_targets,
            artist_context,
            negative_constraints_block,
            rhyme_schemes,
        )

    # --- 6. Generate candidates (verify → revise, up to 2 follow-up turns) ---
    from anthropic import Anthropic

    client = Anthropic()
    print(f"\nGenerating {num_candidates} lyric candidate(s) via {model}...")
    texts = []
    verify_outcomes = []
    for i in range(num_candidates):
        print(f"  Candidate {i + 1}/{num_candidates}...")
        text, outcome = generate_lyric_candidate(
            client, prompt, model, vocal_sections, rhyme_schemes
        )
        if outcome["turns_used"]:
            print(
                f"    Revised {outcome['turns_used']} time(s) — "
                f"{outcome['initial_syllable_issues']}→{outcome['final_syllable_issues']} syllable issue(s), "
                f"{outcome['initial_rhyme_issues']}→{outcome['final_rhyme_issues']} rhyme issue(s)"
            )
        texts.append(text)
        verify_outcomes.append(outcome)

    # --- 7. Score with Refractor (text-only) ---
    scorer_results_map: dict[int, Optional[dict]] = {}
    target = get_chromatic_target(meta.get("color", ""))

    if not skip_scoring:
        print("\nLoading Refractor...")
        try:
            from white_analysis.refractor import Refractor

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
                "verify": verify_outcomes[idx],
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
            "verify": _to_python(entry["verify"]),
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
    parser.add_argument(
        "--melody-channel",
        type=int,
        default=MELODY_CHANNEL,
        help=f"Logic track number carrying melody/vocal clips (default: {MELODY_CHANNEL})",
    )
    parser.add_argument(
        "--arrangement",
        default=None,
        help="Path to arrangement.txt (default: <production-dir>/arrangement.txt)",
    )
    parser.add_argument(
        "--refresh-constraints",
        action="store_true",
        help=(
            "Regenerate lyrics_negative_constraints.yml from the album's promoted "
            "lyrics before building the prompt"
        ),
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
        melody_channel=args.melody_channel,
        arrangement=args.arrangement,
        refresh_constraints=args.refresh_constraints,
    )


if __name__ == "__main__":
    main()
