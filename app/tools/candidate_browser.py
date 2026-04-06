#!/usr/bin/env python3
"""Candidate browser — interactive terminal UI for reviewing pipeline output.

Lists all candidates across phases for a production directory and lets the user
approve/reject them with keystrokes and audition MIDI via the macOS `open` command.

Usage:
    python -m app.tools.candidate_browser --production-dir shrink_wrapped/<album>/production/<slug>
    python -m app.tools.candidate_browser --production-dir ... --phase melody
    python -m app.tools.candidate_browser --production-dir ... --section intro

Keymap:
    w / s    move up / down
    a        approve selected candidate
    r        reject selected candidate
    p        play selected MIDI (macOS open)
    q        quit
"""

import argparse
import shutil
import subprocess
import sys
import termios
import tty
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich import box

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PHASES = ["chords", "drums", "bass", "melody", "quartet"]

_STATUS_COLOR = {
    "approved": "green",
    "accepted": "green",
    "rejected": "red",
    "pending": "yellow",
}

# Fixed overhead lines: detail panel (8) + help bar (3) + table chrome (4: box top/header/divider/box bottom)
_FIXED_OVERHEAD = 15


# ---------------------------------------------------------------------------
# Data layer
# ---------------------------------------------------------------------------


@dataclass
class CandidateEntry:
    phase: str
    section: str  # "" for phases without per-section candidates (chords/drums/bass)
    candidate_id: str
    midi_file: Path  # absolute
    review_yml: Path  # absolute
    status: str
    rank: int
    composite_score: float
    template: str
    scores: dict = field(default_factory=dict)


def _abs_midi(review_yml: Path, midi_relative: str) -> Path:
    return (review_yml.parent / midi_relative).resolve()


def _load_review(review_yml: Path) -> list[CandidateEntry]:
    try:
        with open(review_yml) as f:
            data = yaml.safe_load(f) or {}
    except Exception:
        return []

    phase = review_yml.parent.name
    entries = []
    for c in data.get("candidates", []):
        midi_rel = c.get("midi_file", "")
        midi_path = _abs_midi(review_yml, midi_rel) if midi_rel else Path()
        template = (
            c.get("pattern_name")
            or c.get("template")
            or c.get("progression", "")[:40]
            or c.get("id", "")
        )
        entries.append(
            CandidateEntry(
                phase=phase,
                section=c.get("section", ""),
                candidate_id=c.get("id", ""),
                midi_file=midi_path,
                review_yml=review_yml,
                status=str(c.get("status", "pending")).lower(),
                rank=int(c.get("rank", 99)),
                composite_score=float(c.get("scores", {}).get("composite", 0.0)),
                template=template,
                scores=c.get("scores", {}),
            )
        )
    return entries


def load_all_candidates(
    production_dir: Path,
    phase_filter: str | None = None,
    section_filter: str | None = None,
) -> list[CandidateEntry]:
    """Walk all phase review.yml files and return CandidateEntry list."""
    phases = [phase_filter] if phase_filter else PHASES
    entries: list[CandidateEntry] = []
    for phase in phases:
        review_yml = production_dir / phase / "review.yml"
        if not review_yml.exists():
            continue
        loaded = _load_review(review_yml)
        if section_filter:
            loaded = [e for e in loaded if e.section == section_filter]
        entries.extend(loaded)
    phase_idx = {p: i for i, p in enumerate(PHASES)}
    entries.sort(key=lambda e: (phase_idx.get(e.phase, 99), e.section, e.rank))
    return entries


def _update_review_yml(review_yml: Path, candidate_id: str, new_status: str) -> None:
    with open(review_yml) as f:
        data = yaml.safe_load(f) or {}
    for c in data.get("candidates", []):
        if c.get("id") == candidate_id:
            c["status"] = new_status
            break
    with open(review_yml, "w") as f:
        yaml.dump(data, f, allow_unicode=True, sort_keys=False)


def approve_candidate(entry: CandidateEntry) -> None:
    _update_review_yml(entry.review_yml, entry.candidate_id, "approved")
    entry.status = "approved"


def reject_candidate(entry: CandidateEntry) -> None:
    _update_review_yml(entry.review_yml, entry.candidate_id, "rejected")
    entry.status = "rejected"


def play_candidate(entry: CandidateEntry) -> None:
    if entry.midi_file.exists():
        subprocess.Popen(["open", str(entry.midi_file)])


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _score_bar(value: float, width: int = 8) -> str:
    filled = round(value * width)
    bar = "█" * filled + "░" * (width - filled)
    color = "green" if value >= 0.5 else "yellow" if value >= 0.3 else "red"
    return f"[{color}]{bar}[/{color}] {value:.3f}"


def _page_size() -> int:
    return max(3, shutil.get_terminal_size().lines - _FIXED_OVERHEAD)


def build_candidate_table(
    entries: list[CandidateEntry],
    selected_idx: int,
    offset: int,
) -> Table:
    page = _page_size()
    visible = entries[offset : offset + page]

    table = Table(
        box=box.SIMPLE_HEAVY,
        show_header=True,
        header_style="bold white",
        expand=True,
        pad_edge=False,
    )
    table.add_column(" ", justify="left", no_wrap=True, width=2)
    table.add_column("Phase", no_wrap=True, min_width=7)
    table.add_column("Section", no_wrap=True, min_width=10)
    table.add_column("ID", no_wrap=True, min_width=16)
    table.add_column("Template", no_wrap=True, min_width=20)
    table.add_column("Score", no_wrap=True, min_width=16)
    table.add_column("Status", no_wrap=True, min_width=8)

    for i, e in enumerate(visible):
        abs_idx = offset + i
        is_sel = abs_idx == selected_idx
        sel_marker = "[bold cyan]▶[/bold cyan]" if is_sel else " "
        status_color = _STATUS_COLOR.get(e.status, "white")
        status_str = f"[{status_color}]{e.status}[/{status_color}]"
        score_bar = _score_bar(e.composite_score)
        id_str = f"[bold]{e.candidate_id}[/bold]" if is_sel else e.candidate_id

        table.add_row(
            sel_marker,
            e.phase,
            e.section or "—",
            id_str,
            e.template[:28],
            score_bar,
            status_str,
        )

    return table


def build_detail_panel(entry: CandidateEntry | None) -> Panel:
    if entry is None:
        return Panel(
            "[dim]No candidate selected[/dim]", title="Score Breakdown", height=8
        )

    lines: list[str] = [
        f"[bold]{entry.candidate_id}[/bold]  "
        f"phase=[cyan]{entry.phase}[/cyan]  "
        f"section=[cyan]{entry.section or '—'}[/cyan]",
        f"Template: [yellow]{entry.template}[/yellow]",
        "",
    ]

    scores = entry.scores
    comp = scores.get("composite", 0.0)
    lines.append(f"Composite   {_score_bar(comp)}")

    theory_total = scores.get("theory_total")
    if theory_total is not None:
        lines.append(f"Theory      {_score_bar(float(theory_total))}")
    elif isinstance(scores.get("theory"), dict):
        theory_vals = list(scores["theory"].values())
        if theory_vals:
            mean_theory = sum(float(v) for v in theory_vals) / len(theory_vals)
            lines.append(f"Theory      {_score_bar(mean_theory)}")

    chroma = scores.get("chromatic", {})
    if isinstance(chroma, dict):
        match = chroma.get("match")
        if match is not None:
            lines.append(f"Chromatic   {_score_bar(float(match))}")

    return Panel(
        "\n".join(lines), title="Score Breakdown", border_style="dim", height=8
    )


def build_help_bar(total: int, selected_idx: int, status_msg: str) -> Panel:
    keys = "  [bold]w[/bold] up  [bold]s[/bold] down  [bold]a[/bold] approve  [bold]r[/bold] reject  [bold]p[/bold] play  [bold]q[/bold] quit"
    counter = f"  [dim]{selected_idx + 1}/{total}[/dim]"
    msg = f"  [bold yellow]{status_msg}[/bold yellow]" if status_msg else ""
    return Panel(keys + counter + msg, border_style="dim", height=3)


def build_screen(
    entries: list[CandidateEntry],
    selected_idx: int,
    offset: int,
    status_msg: str,
) -> Group:
    selected = entries[selected_idx] if entries else None
    title = Panel(
        f"[bold cyan]Candidate Browser[/bold cyan]  "
        f"[dim]{len(entries)} candidates — {sum(1 for e in entries if e.status in ('approved','accepted'))} approved[/dim]",
        height=3,
    )
    table_panel = Panel(
        build_candidate_table(entries, selected_idx, offset),
        border_style="blue",
    )
    return Group(
        title,
        table_panel,
        build_detail_panel(selected),
        build_help_bar(len(entries), selected_idx, status_msg),
    )


# ---------------------------------------------------------------------------
# Keyboard input
# ---------------------------------------------------------------------------


def _read_key() -> str:
    ch = sys.stdin.read(1)
    if ch == "\x1b":
        try:
            seq = sys.stdin.read(2)
        except Exception:
            return "esc"
        if seq == "[A":
            return "w"
        if seq == "[B":
            return "s"
        return "esc"
    return ch.lower()


# ---------------------------------------------------------------------------
# Main interactive loop
# ---------------------------------------------------------------------------


def run_browser(
    production_dir: Path,
    phase_filter: str | None = None,
    section_filter: str | None = None,
) -> None:
    entries = load_all_candidates(production_dir, phase_filter, section_filter)

    if not entries:
        print("No candidates found.")
        return

    selected_idx = 0
    offset = 0
    status_msg = ""
    console = Console()

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)

    try:
        tty.setraw(fd)
        with Live(
            build_screen(entries, selected_idx, offset, status_msg),
            console=console,
            refresh_per_second=20,
            screen=True,
        ) as live:
            while True:
                key = _read_key()
                status_msg = ""
                page = _page_size()

                if key in ("q", "esc", "\x03"):
                    break
                elif key == "w":
                    selected_idx = max(0, selected_idx - 1)
                    if selected_idx < offset:
                        offset = selected_idx
                elif key == "s":
                    selected_idx = min(len(entries) - 1, selected_idx + 1)
                    if selected_idx >= offset + page:
                        offset = selected_idx - page + 1
                elif key == "a":
                    entry = entries[selected_idx]
                    if entry.status in ("approved", "accepted"):
                        status_msg = (
                            f"No change for {entry.candidate_id}: status is {entry.status}"
                        )
                    else:
                        approve_candidate(entry)
                        status_msg = f"✓ Approved {entry.candidate_id}"
                        # Advance to next non-approved
                        for i in range(selected_idx + 1, len(entries)):
                            if entries[i].status not in ("approved", "accepted"):
                                selected_idx = i
                                if selected_idx >= offset + page:
                                    offset = selected_idx - page + 1
                                break
                elif key == "r":
                    entry = entries[selected_idx]
                    if entry.status in ("approved", "accepted", "rejected"):
                        status_msg = (
                            f"No change for {entry.candidate_id}: status is {entry.status}"
                        )
                    else:
                        reject_candidate(entry)
                        status_msg = f"✗ Rejected {entry.candidate_id}"
                elif key == "p":
                    entry = entries[selected_idx]
                    if entry.midi_file.exists():
                        play_candidate(entry)
                        status_msg = f"▶ {entry.midi_file.name}"
                    else:
                        status_msg = f"MIDI not found: {entry.midi_file.name}"

                live.update(build_screen(entries, selected_idx, offset, status_msg))

    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive candidate browser")
    parser.add_argument(
        "--production-dir",
        type=Path,
        required=True,
        help="Production directory (e.g. shrink_wrapped/<album>/production/<slug>)",
    )
    parser.add_argument(
        "--phase",
        type=str,
        default=None,
        choices=PHASES,
        help="Filter to a single phase",
    )
    parser.add_argument(
        "--section",
        type=str,
        default=None,
        help="Filter to a single section label",
    )
    args = parser.parse_args()

    production_dir = args.production_dir.resolve()
    if not production_dir.exists():
        print(f"ERROR: {production_dir} does not exist", file=sys.stderr)
        sys.exit(1)

    run_browser(production_dir, args.phase, args.section)


if __name__ == "__main__":
    main()
