#!/usr/bin/env python3
"""
State-aware pipeline orchestrator for the Music Production Pipeline.

Reads `song_context.yml` phase statuses and drives the pipeline forward,
eliminating the need to remember phase order, flags, or paths.

Usage:
    # Show where a song is in its lifecycle
    python -m app.generators.midi.production.pipeline_runner status \
        --production-dir <path>

    # Print the next command without running it
    python -m app.generators.midi.production.pipeline_runner next \
        --production-dir <path>

    # Run the next pending phase
    python -m app.generators.midi.production.pipeline_runner run \
        --production-dir <path>

    # Promote approved candidates for a phase
    python -m app.generators.midi.production.pipeline_runner promote \
        --production-dir <path> --phase chords [--yes]

    # Run a phase for all pending songs in a thread directory
    python -m app.generators.midi.production.pipeline_runner batch \
        --thread <thread-dir> --phase drums
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Optional

import yaml

from white_composition.init_production import load_song_context

# ---------------------------------------------------------------------------
# Phase definitions
# ---------------------------------------------------------------------------

PHASE_ORDER = [
    "init_production",
    "chords",
    "drums",
    "bass",
    "melody",
    "lyrics",
    "decisions",
]

# Review file relative to production_dir — None means no review gate
PHASE_REVIEW_FILES: dict[str, Optional[str]] = {
    "init_production": None,
    "chords": "chords/review.yml",
    "drums": "drums/review.yml",
    "bass": "bass/review.yml",
    "melody": "melody/review.yml",
    "lyrics": "melody/lyrics_review.yml",
    "decisions": None,
    # quartet is not in PHASE_ORDER (it's a parallel phase) but is a valid
    # promote target exposed via the candidate review UI
    "quartet": "quartet/review.yml",
}

# Phase icons for status display
_STATUS_ICONS = {
    "promoted": "✅",
    "generated": "🔄",
    "in_progress": "⚙️ ",
    "pending": "⏳",
}


def _build_phase_command(
    phase: str, production_dir: Path, ctx: dict, song_proposal: str = ""
) -> list[str]:
    """Build the subprocess argv for a generation phase."""
    prod = str(production_dir)
    # song_proposal arg takes precedence over whatever is stored in context
    proposal = song_proposal or ctx.get("song_proposal", "")
    singer = ctx.get("singer", "gabriel")

    base = [sys.executable, "-m"]
    if phase == "init_production":
        cmd = base + [
            "white_composition.init_production",
            "--production-dir",
            prod,
        ]
        if proposal:
            cmd += ["--song-proposal", proposal]
        return cmd
    if phase == "chords":
        cmd = base + [
            "white_generation.pipelines.chord_pipeline",
            "--production-dir",
            prod,
        ]
        if proposal:
            cmd += ["--song", proposal]
        return cmd
    if phase == "drums":
        return base + [
            "white_generation.pipelines.drum_pipeline",
            "--production-dir",
            prod,
        ]
    if phase == "bass":
        return base + [
            "white_generation.pipelines.bass_pipeline",
            "--production-dir",
            prod,
        ]
    if phase == "melody":
        return base + [
            "white_generation.pipelines.melody_pipeline",
            "--production-dir",
            prod,
            "--singer",
            singer,
        ]
    if phase == "lyrics":
        return base + [
            "white_generation.pipelines.lyric_pipeline",
            "--production-dir",
            prod,
        ]
    if phase == "decisions":
        return base + [
            "white_composition.production_decisions",
            "--production-dir",
            prod,
        ]
    if phase == "quartet":
        return base + [
            "white_generation.pipelines.quartet_pipeline",
            "--production-dir",
            prod,
            "--singer",
            singer,
        ]
    raise ValueError(f"Unknown phase: {phase}")


# ---------------------------------------------------------------------------
# Status helpers
# ---------------------------------------------------------------------------


def read_phase_statuses(production_dir: Path) -> dict[str, str]:
    """Return the phases dict from song_context.yml, defaulting missing phases to 'pending'."""
    ctx = load_song_context(production_dir)
    phases = dict(ctx.get("phases") or {})
    for phase in PHASE_ORDER:
        phases.setdefault(phase, "pending")
    return phases


def write_phase_status(production_dir: Path, phase: str, status: str) -> None:
    """Update a single phase status in song_context.yml."""
    ctx_path = production_dir / "song_context.yml"
    if not ctx_path.exists():
        return
    with open(ctx_path) as f:
        data = yaml.safe_load(f) or {}
    if "phases" not in data or data["phases"] is None:
        data["phases"] = {}
    data["phases"][phase] = status
    with open(ctx_path, "w") as f:
        yaml.dump(
            data, f, default_flow_style=False, sort_keys=False, allow_unicode=True
        )


def get_next_runnable_phase(statuses: dict[str, str]) -> Optional[str]:
    """Return the first pending phase whose predecessor is promoted (or no predecessor).

    Returns None if no phase can run.
    """
    for i, phase in enumerate(PHASE_ORDER):
        if statuses.get(phase, "pending") != "pending":
            continue
        if i == 0:
            return phase  # init_production has no predecessor
        predecessor = PHASE_ORDER[i - 1]
        if statuses.get(predecessor, "pending") == "promoted":
            return phase
    return None


def _status_icon(status: str) -> str:
    return _STATUS_ICONS.get(status, "❓")


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------


def cmd_status(production_dir: Path) -> None:
    """Print per-phase status for a production directory."""
    ctx = load_song_context(production_dir)
    title = ctx.get("title", "Unknown")
    color = ctx.get("color", "")
    header = f"{title}" + (f" ({color})" if color else "")

    print(f"\nSong: {header}")
    print(f"Dir:  {production_dir}\n")

    statuses = read_phase_statuses(production_dir)
    for phase in PHASE_ORDER:
        status = statuses.get(phase, "pending")
        icon = _status_icon(status)
        print(f"  {phase:<20s} {icon}  {status}")

    decisions_exists = (production_dir / "production_decisions.yml").exists()
    decisions_flag = "✅  exists" if decisions_exists else "—   not generated"
    print(f"\n  {'production_decisions.yml':<28s} {decisions_flag}")

    next_phase = get_next_runnable_phase(statuses)
    print()
    if next_phase is None:
        all_promoted = all(statuses.get(p) == "promoted" for p in PHASE_ORDER)
        if all_promoted:
            print("All phases promoted. Song complete.")
        else:
            print("No runnable phase — check for gaps or blocked dependencies.")
    else:
        cmd = _build_phase_command(next_phase, production_dir, ctx)
        print(f"Next: run {next_phase}")
        print(f"  Command: {' '.join(cmd)}")
    print()


def cmd_next(production_dir: Path) -> None:
    """Print the next runnable command without executing it."""
    ctx = load_song_context(production_dir)
    statuses = read_phase_statuses(production_dir)
    next_phase = get_next_runnable_phase(statuses)
    if next_phase is None:
        print("No runnable phase.")
        return
    cmd = _build_phase_command(next_phase, production_dir, ctx)
    print(" ".join(cmd))


def cmd_run(production_dir: Path, song_proposal: str = "") -> int:
    """Run the next pending phase, update status, then stop before promoting."""
    ctx = load_song_context(production_dir)
    statuses = read_phase_statuses(production_dir)
    next_phase = get_next_runnable_phase(statuses)

    if next_phase is None:
        print("No runnable phase. All phases are complete or blocked.")
        return 0

    print(f"\n→ Running phase: {next_phase}")
    write_phase_status(production_dir, next_phase, "in_progress")

    cmd = _build_phase_command(
        next_phase, production_dir, ctx, song_proposal=song_proposal
    )
    print(f"  Command: {' '.join(cmd)}\n")

    try:
        result = subprocess.run(cmd)
    except KeyboardInterrupt:
        write_phase_status(production_dir, next_phase, "pending")
        print(f"\n✗ Phase {next_phase} interrupted. Status reset to pending.")
        return 1

    if result.returncode != 0:
        write_phase_status(production_dir, next_phase, "pending")
        print(
            f"\n✗ Phase {next_phase} failed (exit {result.returncode}). Status reset to pending."
        )
        return result.returncode

    review_file = PHASE_REVIEW_FILES.get(next_phase)
    if review_file is None:
        # No review gate — auto-promote so downstream phases can proceed.
        write_phase_status(production_dir, next_phase, "promoted")
        print(f"\n✓ Phase {next_phase} complete. Auto-promoted (no review gate).")
    else:
        write_phase_status(production_dir, next_phase, "generated")
        print(f"\n✓ Phase {next_phase} complete. Status: generated.")
        review_path = production_dir / review_file
        print(f"\nReview candidates in: {review_path}")
        print("Then promote with:")
        print(
            f"  python -m app.generators.midi.production.pipeline_runner promote "
            f"--production-dir {production_dir} --phase {next_phase}"
        )
    return 0


def cmd_promote(production_dir: Path, phase: str, yes: bool = False) -> int:
    """Summarise review.yml and promote approved candidates."""
    review_rel = PHASE_REVIEW_FILES.get(phase)
    if review_rel is None:
        print(f"Phase '{phase}' has no review gate (e.g. init_production).")
        return 1

    review_path = production_dir / review_rel
    if not review_path.exists():
        print(f"Review file not found: {review_path}")
        print(
            f"Run generation phase first: pipeline run --production-dir {production_dir}"
        )
        return 1

    with open(review_path) as f:
        review = yaml.safe_load(f) or {}

    candidates = review.get("candidates", [])
    approved = [
        c
        for c in candidates
        if str(c.get("status", "")).lower() in ("approved", "accepted")
    ]

    print(f"\nPhase: {phase}")
    print(f"Review: {review_path}")
    print(f"  Total candidates: {len(candidates)}")
    print(f"  Approved: {len(approved)}")

    if not approved:
        print("\nNo approved candidates. Edit review.yml and set status: approved.")
        return 1

    if not yes:
        try:
            answer = input("\nPromote approved candidates? [y/N] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nAborted.")
            return 1
        if answer not in ("y", "yes"):
            print("Aborted.")
            return 1

    from white_composition.promote_part import promote_part

    promote_part(str(review_path))

    write_phase_status(production_dir, phase, "promoted")
    print(f"\n✓ Phase {phase} promoted. Status: promoted.")

    statuses = read_phase_statuses(production_dir)
    next_phase = get_next_runnable_phase(statuses)
    if next_phase:
        print(f"\nNext phase: {next_phase}")
        print(f"  pipeline run --production-dir {production_dir}")
    return 0


def cmd_batch(thread_dir: Path, phase: str) -> int:
    """Run a generation phase for all production dirs where that phase is pending."""
    if not thread_dir.exists():
        print(f"Thread directory not found: {thread_dir}")
        return 1

    # Find all production dirs (dirs containing song_context.yml)
    prod_dirs = [d.parent for d in thread_dir.rglob("song_context.yml")]
    prod_dirs.sort()

    pending = [
        d
        for d in prod_dirs
        if read_phase_statuses(d).get(phase, "pending") == "pending"
    ]

    if not pending:
        print(f"No production dirs with {phase}=pending in {thread_dir}")
        return 0

    print(f"\nBatch run: {phase} for {len(pending)} production dir(s)")
    failed = 0
    for d in pending:
        print(f"\n{'=' * 60}")
        ctx = load_song_context(d)
        print(f"  Song: {ctx.get('title', d.name)}")
        result_code = cmd_run(d)
        if result_code != 0:
            failed += 1

    print(f"\nBatch complete: {len(pending) - failed}/{len(pending)} succeeded.")
    return 0 if failed == 0 else 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pipeline_runner",
        description="State-aware pipeline orchestrator",
    )
    sub = p.add_subparsers(dest="subcommand", required=True)

    # status
    s = sub.add_parser("status", help="Show per-phase status")
    s.add_argument("--production-dir", required=True, type=Path)

    # next
    n = sub.add_parser("next", help="Print next runnable command")
    n.add_argument("--production-dir", required=True, type=Path)

    # run
    r = sub.add_parser("run", help="Run the next pending phase")
    r.add_argument("--production-dir", required=True, type=Path)
    r.add_argument(
        "--song-proposal",
        default="",
        help="Path to song proposal YAML (required for init_production phase)",
    )

    # promote
    pr = sub.add_parser("promote", help="Promote approved candidates")
    pr.add_argument("--production-dir", required=True, type=Path)
    pr.add_argument("--phase", required=True, choices=list(PHASE_REVIEW_FILES.keys()))
    pr.add_argument("--yes", "-y", action="store_true", help="Skip confirmation")

    # batch
    b = sub.add_parser("batch", help="Run a phase for all pending songs in a thread")
    b.add_argument("--thread", required=True, type=Path)
    b.add_argument("--phase", required=True, choices=list(PHASE_REVIEW_FILES.keys()))

    return p


def main(argv=None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.subcommand == "status":
        cmd_status(args.production_dir)
    elif args.subcommand == "next":
        cmd_next(args.production_dir)
    elif args.subcommand == "run":
        sys.exit(cmd_run(args.production_dir, song_proposal=args.song_proposal))
    elif args.subcommand == "promote":
        sys.exit(cmd_promote(args.production_dir, args.phase, yes=args.yes))
    elif args.subcommand == "batch":
        sys.exit(cmd_batch(args.thread, args.phase))


if __name__ == "__main__":
    main()
