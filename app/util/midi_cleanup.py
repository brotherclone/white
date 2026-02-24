"""
midi_cleanup.py
───────────────
NOTE TO SELF (session: 2026-02-24, White Album / Green Agent first demo export)

The Problem:
    Logic Pro writes the full *project* length into the tempo map track (track 0)
    when exporting MIDI, regardless of where the last note actually falls. This
    manifests as a ghost tail of silence that accumulates catastrophically when
    loops are assembled — 107 bars × a few beats of dead time = 45s timing drift
    in the evaluator.

    Compounding issue: chord loops were generated assuming 4/4 (bars = ppq * 4)
    but "The Silence Where Abundance Used to Hum" is in 3/4. The bar-length math
    was wrong at source, then Logic's tempo track bloat made diagnosis harder.

    Symptom in evaluation YAML:
        timing drift 45.0s
        structural_integrity: 0.05
        name_mismatches: 9      ← evaluator can't reconcile section lengths
        airgigs_readiness: draft

    Root cause split:
        - 3/4 vs 4/4: fix at generation layer (pull time_sig from song_proposal)
        - Tempo track bloat: fix here, as post-export cleanup

The Fix (this file):
    Find the true last note event across all tracks, then truncate the tempo/meta
    track to match. Logic's note tracks are fine — only the headerless meta track
    overshoots.

Integration points:
    - Run as post-process after any Logic MIDI export before files land in approved/
    - Or wire into the evaluator as a pre-flight: flag files where
      tempo_track_length > last_note_tick and auto-trim before scoring
    - Could also live in the chord export pipeline right after mido.MidiFile.save()
"""

import mido
from pathlib import Path


def trim_midi_tempo_track(
    path_in: Path | str, path_out: Path | str | None = None
) -> mido.MidiFile:
    """
    Truncate Logic's bloated tempo/meta track to match the true last note event.

    Logic exports tempo maps that span the full project length rather than the
    content length. When assembling loops this ghost silence accumulates into
    serious timing drift. This trims track 0 (or any trackless meta track) to
    align with the last note_on/note_off tick across all note tracks.

    Args:
        path_in:  Source MIDI path.
        path_out: Destination path. If None, overwrites in place.

    Returns:
        The corrected MidiFile object.
    """
    mid = mido.MidiFile(path_in)
    path_out = path_out or path_in

    # find true last note event
    last_tick = 0
    for track in mid.tracks:
        abs_t = 0
        for msg in track:
            abs_t += msg.time
            if msg.type in ("note_on", "note_off"):
                last_tick = max(last_tick, abs_t)

    if last_tick == 0:
        raise ValueError(f"No note events found in {path_in}")

    new_tracks = []
    for track in mid.tracks:
        has_notes = any(m.type == "note_on" for m in track)
        if has_notes:
            new_tracks.append(track)
            continue

        # meta/tempo track — rebuild, stopping at last_tick
        new_msgs = []
        abs_t = 0
        for msg in track:
            new_abs = abs_t + msg.time
            if new_abs > last_tick:
                new_msgs.append(
                    mido.MetaMessage("end_of_track", time=last_tick - abs_t)
                )
                break
            new_msgs.append(msg)
            abs_t = new_abs

        new_track = mido.MidiTrack()
        new_track.extend(new_msgs)
        new_track.name = track.name
        new_tracks.append(new_track)

    mid.tracks = new_tracks
    mid.save(path_out)
    return mid


def batch_trim(approved_dir: Path, dry_run: bool = False) -> list[dict]:
    """
    Walk an approved/ directory tree and trim any MIDI with tempo track bloat.

    Flags files where tempo_track_length > last_note_tick (Logic export artifact)
    and optionally fixes them in place.

    Args:
        approved_dir: Root of approved/ phase dirs (approved/chords/, approved/drums/ etc.)
        dry_run:      If True, report problems without modifying files.

    Returns:
        List of dicts with path, original_ticks, last_note_tick, fixed.
    """
    report = []
    for midi_path in sorted(approved_dir.rglob("*.mid")):
        mid = mido.MidiFile(midi_path)
        ppq = mid.ticks_per_beat

        last_note_tick = 0
        meta_track_length = 0

        for track in mid.tracks:
            abs_t = 0
            for msg in track:
                abs_t += msg.time
                if msg.type in ("note_on", "note_off"):
                    last_note_tick = max(last_note_tick, abs_t)

            has_notes = any(m.type == "note_on" for m in track)
            if not has_notes:
                meta_track_length = max(meta_track_length, abs_t)

        bloated = meta_track_length > last_note_tick
        entry = {
            "path": midi_path,
            "meta_ticks": meta_track_length,
            "note_ticks": last_note_tick,
            "beats": round(last_note_tick / ppq, 2),
            "bloated": bloated,
            "fixed": False,
        }

        if bloated and not dry_run:
            trim_midi_tempo_track(midi_path, midi_path)
            entry["fixed"] = True

        report.append(entry)

    return report


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python midi_cleanup.py <approved_dir> [--dry-run]")
        sys.exit(1)

    approved = Path(sys.argv[1])
    dry = "--dry-run" in sys.argv

    results = batch_trim(approved, dry_run=dry)
    bloated = [r for r in results if r["bloated"]]

    print(
        f"Scanned {len(results)} files, {len(bloated)} bloated{' (dry run)' if dry else ' (fixed)'}:"
    )
    for r in bloated:
        status = "→ fixed" if r["fixed"] else "→ would fix"
        print(
            f"  {r['path'].relative_to(approved)}  "
            f"meta:{r['meta_ticks']} note:{r['note_ticks']} ({r['beats']} beats)  {status}"
        )
