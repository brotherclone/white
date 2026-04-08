#!/usr/bin/env python3
"""
Promote approved part candidates from review.yml to the approved/ directory.

Reads a review.yml file, finds candidates with status=approved, and copies
their MIDI files to the approved/ directory with label-based filenames.

Usage:
    python -m app.generators.midi.production.promote_part \
        --review shrink_wrapped/.../production/.../chords/review.yml
"""

import argparse
import shutil
from pathlib import Path

import mido
import yaml

from app.util.diversity_tracker import (
    find_album_dir,
    load_registry,
    record_use,
    save_registry,
)

_QUARTET_VOICES = ["soprano", "alto", "tenor", "bass_voice"]
_QUARTET_CHANNELS = {"soprano": 0, "alto": 1, "tenor": 2, "bass_voice": 3}


def _split_quartet_midi(source: Path, approved_dir: Path, label_clean: str) -> None:
    """Split a multi-channel quartet MIDI into one file per voice.

    Writes <label>_soprano.mid, <label>_alto.mid, <label>_tenor.mid,
    <label>_bass_voice.mid alongside the combined file.  Tempo/meta
    events from track 0 are included in every split file.
    """
    try:
        mid = mido.MidiFile(filename=str(source))
    except Exception as e:
        print(f"  Warning: could not split quartet MIDI {source.name}: {e}")
        return

    for voice, channel in _QUARTET_CHANNELS.items():
        split = mido.MidiFile(ticks_per_beat=mid.ticks_per_beat)
        track = mido.MidiTrack()
        split.tracks.append(track)

        track.append(
            mido.MetaMessage("track_name", name=f"{label_clean}_{voice}", time=0)
        )

        # Replay all tracks, keeping only messages for this channel + meta
        # Preserve delta-time correctly by accumulating abs ticks then re-diffing
        events: list[tuple[int, object]] = []
        for src_track in mid.tracks:
            abs_tick = 0
            for msg in src_track:
                abs_tick += msg.time
                if msg.is_meta:
                    if msg.type in ("set_tempo", "time_signature", "key_signature"):
                        events.append((abs_tick, msg.copy(time=0)))
                elif hasattr(msg, "channel") and msg.channel == channel:
                    events.append((abs_tick, msg.copy(time=0)))

        events.sort(key=lambda x: x[0])

        prev_tick = 0
        for abs_tick, msg in events:
            delta = abs_tick - prev_tick
            track.append(msg.copy(time=delta))
            prev_tick = abs_tick

        track.append(mido.MetaMessage("end_of_track", time=0))

        dest = approved_dir / f"{label_clean}_{voice}.mid"
        split.save(str(dest))

    print(f"  → split into {', '.join(f'{v}.mid' for v in _QUARTET_VOICES)}")


def _rewrite_track_names(midi_path: Path, label: str) -> None:
    """Rewrite track_name MetaMessages in a promoted MIDI to match the label.

    Logic (and other DAWs) use the internal track_name MetaMessage as the
    region name, not the filename.  Without this fix a file promoted from
    ``melody_chorus_02.mid`` → ``melody_chorus.mid`` would still appear as
    "melody_chorus_02" (or "melody_chorus_2" after Logic strips leading zeros).
    """
    try:
        mid = mido.MidiFile(filename=str(midi_path))
        changed = False
        for track in mid.tracks:
            for i, msg in enumerate(track):
                if msg.type == "track_name" and msg.name and msg.name != label:
                    track[i] = mido.MetaMessage("track_name", name=label, time=msg.time)
                    changed = True
        if changed:
            mid.save(str(midi_path))
    except Exception as e:
        print(f"  Warning: could not rewrite track names in {midi_path.name}: {e}")


def promote_part(review_path: str, clean: bool = False):
    """Read review.yml and promote approved candidates."""
    review_file = Path(review_path)
    if not review_file.exists():
        print(f"ERROR: Review file not found: {review_file}")
        return

    with open(review_file) as f:
        review = yaml.safe_load(f)

    part_dir = review_file.parent
    candidates_dir = part_dir / "candidates"
    approved_dir = part_dir / "approved"
    approved_dir.mkdir(parents=True, exist_ok=True)

    if clean:
        removed = list(approved_dir.glob("*.mid"))
        for f in removed:
            f.unlink()
        if removed:
            print(f"Cleaned {len(removed)} file(s) from {approved_dir}/")
        lyrics_txt = part_dir / "lyrics.txt"
        if lyrics_txt.exists():
            lyrics_txt.unlink()
            print(f"Cleaned lyrics.txt from {part_dir}/")
        lyrics_draft = part_dir / "lyrics_draft.txt"
        if lyrics_draft.exists():
            lyrics_draft.unlink()
            print(f"Cleaned lyrics_draft.txt from {part_dir}/")

    candidates = review.get("candidates", [])
    approved = [
        c
        for c in candidates
        if str(c.get("status", "")).lower() in ("approved", "accepted")
        and not str(c.get("midi_file", "")).endswith("_scratch.mid")
    ]

    if not approved:
        print("No approved candidates found in review.yml")
        print(
            "Edit the review file and set status: approved on candidates you want to promote"
        )
        return

    # Split into MIDI candidates and lyric .txt candidates
    # Lyrics entries have a "file" key (pointing to .txt); MIDI entries have "midi_file"
    txt_approved = [c for c in approved if c.get("file") and not c.get("midi_file")]
    midi_approved = [c for c in approved if c.get("midi_file")]

    # --- Lyrics .txt promotion ---
    if len(txt_approved) > 1:
        ids = ", ".join(c.get("id", "?") for c in txt_approved)
        print("ERROR: Multiple approved lyric candidates found.")
        print("Only one lyrics.txt can be promoted at a time.")
        print(f"  Approved lyric IDs: {ids}")
        print("Reject all but one, then re-run promotion.")
        return

    if txt_approved:
        lyric_candidate = txt_approved[0]
        txt_filename = lyric_candidate.get("file", "")
        source_txt = candidates_dir / Path(txt_filename).name
        if not source_txt.exists():
            source_txt = part_dir / txt_filename
        if not source_txt.exists():
            print(f"  WARNING: Lyric file not found: {source_txt}")
        else:
            dest_txt = part_dir / "lyrics.txt"
            shutil.copy2(source_txt, dest_txt)
            draft_txt = part_dir / "lyrics_draft.txt"
            shutil.copy2(source_txt, draft_txt)
            print(
                f"  [{lyric_candidate.get('id')}] lyrics → lyrics.txt + lyrics_draft.txt"
            )

    # --- MIDI promotion ---
    if not midi_approved and not txt_approved:
        print("No approved candidates found in review.yml")
        print(
            "Edit the review file and set status: approved on candidates you want to promote"
        )
        return

    # Enforce one approved per label (for MIDI candidates)
    label_to_candidates: dict[str, list] = {}
    for candidate in midi_approved:
        label = (
            (candidate.get("label") or "unlabeled")
            .replace("-", "_")
            .replace(" ", "_")
            .lower()
        )
        label_to_candidates.setdefault(label, []).append(candidate)

    conflicts = {
        label: cands for label, cands in label_to_candidates.items() if len(cands) > 1
    }
    if conflicts:
        print("ERROR: Multiple approved candidates share the same label.")
        print("Resolve by rejecting all but one per label, then re-run promotion.\n")
        for label, cands in conflicts.items():
            ids = ", ".join(c.get("id", "?") for c in cands)
            print(f"  label '{label}': {ids}")
        return

    promoted = []
    for label_clean, cands in label_to_candidates.items():
        candidate = cands[0]
        label = candidate.get("label") or "unlabeled"
        midi_filename = candidate.get("midi_file", "")
        source = candidates_dir / Path(midi_filename).name

        if not source.exists():
            source = part_dir / midi_filename
        if not source.exists():
            print(f"  WARNING: MIDI file not found: {source}")
            continue

        dest_name = f"{label_clean}.mid"
        dest = approved_dir / dest_name
        shutil.copy2(source, dest)
        _rewrite_track_names(dest, label_clean)

        # Quartet phase: also split into per-voice files
        if part_dir.name == "quartet":
            _split_quartet_midi(dest, approved_dir, label_clean)
        promoted.append(
            {
                "id": candidate.get("id"),
                "label": label,
                "source": str(source.name),
                "dest": str(dest_name),
                "rank": candidate.get("rank"),
                "pattern_name": candidate.get("pattern_name"),
                **(
                    {"use_case": candidate["use_case"]}
                    if candidate.get("use_case")
                    else {}
                ),
            }
        )

    # Record promoted template names in the album diversity registry.
    if promoted:
        _album_dir = find_album_dir(review_file)
        if _album_dir:
            _registry = load_registry(_album_dir)
            for p in promoted:
                tmpl = p.get("pattern_name")
                if tmpl:
                    record_use(tmpl, _registry)
            save_registry(_album_dir, _registry)

    # Summary
    if promoted:
        print(f"\nPromoted {len(promoted)} MIDI candidates to {approved_dir}/")
        print("-" * 50)
        for p in promoted:
            print(f"  #{p['rank']} [{p['id']}] {p['label']:20s} → {p['dest']}")

    if txt_approved:
        print(f"\nLyrics promoted to {part_dir}/lyrics.txt")

    print(f"\nApproved directory: {approved_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Promote approved part candidates from review.yml"
    )
    parser.add_argument("--review", required=True, help="Path to review.yml")
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove all existing .mid files from approved/ before promoting",
    )
    args = parser.parse_args()
    promote_part(args.review, clean=args.clean)


if __name__ == "__main__":
    main()
