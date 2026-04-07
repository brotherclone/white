#!/usr/bin/env python3
"""Song completion dashboard — phase matrix for all production runs.

Reads shrink_wrapped/<album>/ and prints a rich table showing which pipeline
phases are complete for every song.

Usage:
    python -m app.tools.song_dashboard
    python -m app.tools.song_dashboard --album-dir shrink_wrapped/the-breathing-machine-learns-to-sing
    python -m app.tools.song_dashboard --color black
    python -m app.tools.song_dashboard --phase melody
"""

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from rich import box
from rich.console import Console
from rich.table import Table

# ---------------------------------------------------------------------------
# Rainbow color ordering (for sort)
# ---------------------------------------------------------------------------

_COLOR_ORDER = [
    "red",
    "orange",
    "yellow",
    "green",
    "blue",
    "indigo",
    "violet",
    "white",
    "black",
]


def _color_rank(color: str) -> int:
    return _COLOR_ORDER.index(color.lower()) if color.lower() in _COLOR_ORDER else 99


# ---------------------------------------------------------------------------
# Phase status
# ---------------------------------------------------------------------------

PHASES = ["chords", "drums", "bass", "melody", "quartet"]

STATUS_APPROVED = "approved"
STATUS_PENDING = "pending"
STATUS_NO_CANDIDATES = "no_candidates"
STATUS_NOT_STARTED = "not_started"


def phase_status(production_dir: Path, phase: str) -> str:
    """Return completion status for a phase within a production directory."""
    phase_dir = production_dir / phase
    review_path = phase_dir / "review.yml"

    if not phase_dir.exists():
        return STATUS_NOT_STARTED
    if not review_path.exists():
        return STATUS_NO_CANDIDATES

    try:
        with open(review_path) as f:
            data = yaml.safe_load(f) or {}
    except Exception:
        return STATUS_NO_CANDIDATES

    candidates = data.get("candidates", [])
    if not candidates:
        return STATUS_NO_CANDIDATES

    approved_statuses = {"approved", "accepted"}
    if any(str(c.get("status", "")).lower() in approved_statuses for c in candidates):
        return STATUS_APPROVED

    return STATUS_PENDING


# ---------------------------------------------------------------------------
# SongStatus dataclass
# ---------------------------------------------------------------------------


@dataclass
class SongStatus:
    slug: str  # production directory name
    album_slug: str  # shrink_wrapped/<album> dirname
    color: str
    singer: str
    key: str
    bpm: str
    phase_statuses: dict = field(default_factory=dict)  # phase → status string
    total_approved_bars: int = 0
    plan_present: bool = False
    lyrics_present: bool = False


# ---------------------------------------------------------------------------
# Scanner
# ---------------------------------------------------------------------------


def _read_plan(production_dir: Path) -> dict:
    p = production_dir / "production_plan.yml"
    if not p.exists():
        return {}
    try:
        with open(p) as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def _read_chord_review(production_dir: Path) -> dict:
    p = production_dir / "chords" / "review.yml"
    if not p.exists():
        return {}
    try:
        with open(p) as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def _read_melody_review(production_dir: Path) -> dict:
    p = production_dir / "melody" / "review.yml"
    if not p.exists():
        return {}
    try:
        with open(p) as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def _approved_bars(plan: dict) -> int:
    total = 0
    for sec in plan.get("sections", []):
        bars = sec.get("bars", 0)
        play_count = sec.get("play_count", sec.get("repeat", 1))
        total += bars * play_count
    return total


def scan_production_dir(production_dir: Path, album_slug: str) -> SongStatus:
    """Build a SongStatus from a single production directory."""
    plan = _read_plan(production_dir)
    chord_review = _read_chord_review(production_dir)
    melody_review = _read_melody_review(production_dir)

    # Color: plan > chord review
    color = (
        plan.get("color")
        or chord_review.get("color")
        or chord_review.get("color_name")
        or "?"
    )

    # Singer: plan > chord review > melody review candidates
    singer = plan.get("singer") or chord_review.get("singer") or ""
    if not singer:
        for cand in melody_review.get("candidates", []):
            singer = cand.get("singer", "")
            if singer:
                break
    singer = singer or "—"

    # Key / BPM
    key = plan.get("key") or chord_review.get("key") or "?"
    bpm = str(plan.get("bpm") or chord_review.get("bpm") or "?")

    phases = {phase: phase_status(production_dir, phase) for phase in PHASES}
    total_bars = _approved_bars(plan) if plan else 0
    lyrics_present = (production_dir / "melody" / "lyrics.txt").exists()

    return SongStatus(
        slug=production_dir.name,
        album_slug=album_slug,
        color=color,
        singer=singer,
        key=key,
        bpm=bpm,
        phase_statuses=phases,
        total_approved_bars=total_bars,
        plan_present=(production_dir / "production_plan.yml").exists(),
        lyrics_present=lyrics_present,
    )


def scan_album(album_dir: Path) -> list[SongStatus]:
    """Scan a single album's production directories and return SongStatus list."""
    production_root = album_dir / "production"
    if not production_root.exists():
        return []

    results = []
    for prod_dir in sorted(production_root.iterdir()):
        if not prod_dir.is_dir():
            continue
        # Skip empty dirs
        if not any(prod_dir.iterdir()):
            continue
        status = scan_production_dir(prod_dir, album_dir.name)
        results.append(status)
    return results


def scan_all(root: Path) -> list[SongStatus]:
    """Walk all album dirs under root and collect SongStatus entries."""
    all_statuses = []
    for album_dir in sorted(root.iterdir()):
        if not album_dir.is_dir():
            continue
        if album_dir.name.startswith("."):
            continue
        all_statuses.extend(scan_album(album_dir))
    return all_statuses


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

_STATUS_SYMBOL = {
    STATUS_APPROVED: ("[green]✓[/green]", "✓"),
    STATUS_PENDING: ("[yellow]⚠[/yellow]", "⚠"),
    STATUS_NO_CANDIDATES: ("[red]✗[/red]", "✗"),
    STATUS_NOT_STARTED: ("[dim]—[/dim]", "—"),
}


def _cell(status: str) -> str:
    return _STATUS_SYMBOL.get(status, ("?", "?"))[0]


def build_table(statuses: list[SongStatus]) -> Table:
    table = Table(
        box=box.SIMPLE_HEAVY,
        show_header=True,
        header_style="bold white",
        title="[bold]Song Production Dashboard[/bold]",
        title_style="bold cyan",
        expand=True,
    )

    table.add_column("Production Run", style="white", no_wrap=True, min_width=28)
    table.add_column("Color", style="white", no_wrap=True, min_width=7)
    table.add_column("Singer", style="white", no_wrap=True, min_width=9)
    table.add_column("Key", style="white", no_wrap=True, min_width=9)
    table.add_column("BPM", style="white", no_wrap=True, justify="right", min_width=4)
    table.add_column("Chords", justify="center", no_wrap=True, min_width=6)
    table.add_column("Drums", justify="center", no_wrap=True, min_width=5)
    table.add_column("Bass", justify="center", no_wrap=True, min_width=4)
    table.add_column("Melody", justify="center", no_wrap=True, min_width=6)
    table.add_column("Qrtt", justify="center", no_wrap=True, min_width=4)
    table.add_column("Bars", justify="right", no_wrap=True, min_width=4)
    table.add_column("Plan", justify="center", no_wrap=True, min_width=4)
    table.add_column("Lyr", justify="center", no_wrap=True, min_width=3)

    # Sort by color rank then slug
    sorted_statuses = sorted(statuses, key=lambda s: (_color_rank(s.color), s.slug))

    prev_color = None
    for s in sorted_statuses:
        if prev_color is not None and s.color.lower() != prev_color:
            table.add_section()
        prev_color = s.color.lower()

        color_style = _color_style(s.color)
        # Show <color>__<version> — strip the long content slug in the middle
        parts = s.slug.split("__")
        if len(parts) >= 2:
            short_slug = f"{parts[0]}  {parts[-1].replace('_', ' ')}"
        else:
            short_slug = s.slug.replace("_", " ")

        bars_str = str(s.total_approved_bars) if s.total_approved_bars else "—"
        plan_str = "[green]✓[/green]" if s.plan_present else "[dim]—[/dim]"
        lyrics_str = "[green]✓[/green]" if s.lyrics_present else "[dim]—[/dim]"

        table.add_row(
            short_slug,
            f"[{color_style}]{s.color}[/{color_style}]",
            s.singer,
            s.key,
            s.bpm,
            _cell(s.phase_statuses.get("chords", STATUS_NOT_STARTED)),
            _cell(s.phase_statuses.get("drums", STATUS_NOT_STARTED)),
            _cell(s.phase_statuses.get("bass", STATUS_NOT_STARTED)),
            _cell(s.phase_statuses.get("melody", STATUS_NOT_STARTED)),
            _cell(s.phase_statuses.get("quartet", STATUS_NOT_STARTED)),
            bars_str,
            plan_str,
            lyrics_str,
        )

    return table


def _color_style(color: str) -> str:
    mapping = {
        "red": "red",
        "orange": "yellow",  # rich has no orange
        "yellow": "bright_yellow",
        "green": "green",
        "blue": "blue",
        "indigo": "blue",
        "violet": "magenta",
        "white": "white",
        "black": "bright_black",
    }
    return mapping.get(color.lower(), "white")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Song production phase dashboard")
    parser.add_argument(
        "--album-dir",
        type=Path,
        default=None,
        help="Scan a single album directory (default: scan all under shrink_wrapped/)",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("shrink_wrapped"),
        help="Root directory containing album folders (default: shrink_wrapped/)",
    )
    parser.add_argument(
        "--color",
        type=str,
        default=None,
        help="Filter to a specific color (e.g. black)",
    )
    parser.add_argument(
        "--phase",
        type=str,
        default=None,
        choices=PHASES,
        help="Show only songs where this phase is not yet approved",
    )
    args = parser.parse_args()

    if args.album_dir:
        statuses = scan_album(args.album_dir.resolve())
    else:
        root = args.root.resolve()
        if not root.exists():
            print(f"ERROR: {root} does not exist", file=sys.stderr)
            sys.exit(1)
        statuses = scan_all(root)

    if not statuses:
        print("No production directories found.")
        return

    # Filters
    if args.color:
        statuses = [s for s in statuses if s.color.lower() == args.color.lower()]
    if args.phase:
        statuses = [
            s for s in statuses if s.phase_statuses.get(args.phase) != STATUS_APPROVED
        ]

    if not statuses:
        print("No songs match the given filters.")
        return

    console = Console(width=160)
    table = build_table(statuses)
    console.print(table)

    # Summary line
    total = len(statuses)
    fully_done = sum(
        1
        for s in statuses
        if all(
            s.phase_statuses.get(p) == STATUS_APPROVED
            for p in ["chords", "drums", "bass", "melody"]
        )
    )
    console.print(
        f"[dim]{total} production run(s) — {fully_done} fully approved "
        f"(chords+drums+bass+melody)[/dim]"
    )


if __name__ == "__main__":
    main()
