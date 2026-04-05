#!/usr/bin/env python3
"""Candidate browser — interactive terminal UI for reviewing pipeline output.

Lists all candidates across phases for a production directory and lets the user
approve/reject them with keystrokes and audition MIDI via the macOS `open` command.

Usage:
    python -m app.tools.candidate_browser --production-dir shrink_wrapped/<album>/production/<slug>
    python -m app.tools.candidate_browser --production-dir ... --phase melody
    python -m app.tools.candidate_browser --production-dir ... --section intro

Keymap:
    ↑ / k    move up
    ↓ / j    move down
    a        approve selected candidate
    r        reject selected candidate
    p        play selected MIDI (macOS open)
    q / ESC  quit
"""

import argparse
import subprocess
import sys
import termios
import tty
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from rich.console import Console
from rich.layout import Layout
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


# ---------------------------------------------------------------------------
# Data layer
# ---------------------------------------------------------------------------


@dataclass
class CandidateEntry:
    phase: str
    section: str  # "" for phases without sections (chords/drums/bass)
    candidate_id: str
    midi_file: Path  # absolute
    review_yml: Path  # absolute
    status: str
    rank: int
    composite_score: float
    template: str
    scores: dict = field(default_factory=dict)


def _abs_midi(review_yml: Path, midi_relative: str) -> Path:
    """Resolve midi_file path (relative to review.yml's parent dir)."""
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
        # Template name: pattern_name > template > progression > id
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
    # Sort: phase order → section → rank
    phase_idx = {p: i for i, p in enumerate(PHASES)}
    entries.sort(key=lambda e: (phase_idx.get(e.phase, 99), e.section, e.rank))
    return entries


def _update_review_yml(review_yml: Path, candidate_id: str, new_status: str) -> None:
    """Write new_status to the matching candidate entry in review.yml."""
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


def _score_bar(value: float, width: int = 10) -> str:
    filled = round(value * width)
    bar = "█" * filled + "░" * (width - filled)
    color = "green" if value >= 0.5 else "yellow" if value >= 0.3 else "red"
    return f"[{color}]{bar}[/{color}] {value:.3f}"


def build_candidate_table(
    entries: list[CandidateEntry],
    selected_idx: int,
) -> Table:
    table = Table(
        box=box.SIMPLE_HEAVY,
        show_header=True,
        header_style="bold white",
        expand=True,
        highlight=False,
    )
    table.add_column("#", justify="right", no_wrap=True, min_width=3)
    table.add_column("Phase", no_wrap=True, min_width=7)
    table.add_column("Section", no_wrap=True, min_width=12)
    table.add_column("ID", no_wrap=True, min_width=18)
    table.add_column("Template", no_wrap=True, min_width=24)
    table.add_column("Score", no_wrap=True, min_width=18)
    table.add_column("Status", no_wrap=True, min_width=8)

    for i, e in enumerate(entries):
        is_selected = i == selected_idx
        row_style = "bold on dark_blue" if is_selected else ""
        prefix = "▶ " if is_selected else "  "

        status_color = _STATUS_COLOR.get(e.status, "white")
        status_str = f"[{status_color}]{e.status}[/{status_color}]"
        score_bar = _score_bar(e.composite_score)

        table.add_row(
            f"{i + 1}",
            f"{prefix}{e.phase}",
            e.section or "—",
            e.candidate_id,
            e.template[:32],
            score_bar,
            status_str,
            style=row_style,
        )

    return table


def build_detail_panel(entry: CandidateEntry | None) -> Panel:
    if entry is None:
        return Panel("[dim]No candidate selected[/dim]", title="Score Breakdown")

    lines: list[str] = []
    lines.append(
        f"[bold]{entry.candidate_id}[/bold]  phase=[cyan]{entry.phase}[/cyan]  section=[cyan]{entry.section or '—'}[/cyan]"
    )
    lines.append(f"Template: [yellow]{entry.template}[/yellow]")
    lines.append(f"MIDI: [dim]{entry.midi_file}[/dim]")
    lines.append("")

    scores = entry.scores
    comp = scores.get("composite", 0.0)
    lines.append(f"Composite  {_score_bar(comp)}")

    theory_total = scores.get("theory_total") or scores.get("theory", {}).get("total")
    if theory_total is not None:
        lines.append(f"Theory     {_score_bar(float(theory_total))}")
    elif isinstance(scores.get("theory"), dict):
        for k, v in scores["theory"].items():
            lines.append(f"  {k:<22} {_score_bar(float(v))}")

    chroma = scores.get("chromatic", {})
    if isinstance(chroma, dict):
        match = chroma.get("match")
        if match is not None:
            lines.append(f"Chromatic  {_score_bar(float(match))}")
        conf = chroma.get("confidence")
        if conf is not None:
            lines.append(f"  confidence  {float(conf):.4f}")

    return Panel("\n".join(lines), title="Score Breakdown", border_style="dim")


def build_help_bar() -> str:
    keys = [
        ("↑/k", "up"),
        ("↓/j", "down"),
        ("a", "approve"),
        ("r", "reject"),
        ("p", "play"),
        ("q", "quit"),
    ]
    parts = [f"[bold]{k}[/bold] {label}" for k, label in keys]
    return "  ".join(parts)


def build_layout(
    entries: list[CandidateEntry],
    selected_idx: int,
    status_msg: str = "",
) -> Layout:
    selected = entries[selected_idx] if entries else None

    layout = Layout()
    layout.split_column(
        Layout(name="table", ratio=3),
        Layout(name="detail", ratio=2),
        Layout(name="footer", size=3),
    )

    layout["table"].update(
        Panel(
            build_candidate_table(entries, selected_idx),
            title=f"[bold cyan]Candidate Browser[/bold cyan]  [dim]{len(entries)} candidates[/dim]",
        )
    )
    layout["detail"].update(build_detail_panel(selected))
    footer_text = build_help_bar()
    if status_msg:
        footer_text += f"  [bold yellow]{status_msg}[/bold yellow]"
    layout["footer"].update(Panel(footer_text, border_style="dim"))

    return layout


# ---------------------------------------------------------------------------
# Keyboard input (raw terminal, no extra deps)
# ---------------------------------------------------------------------------


def _read_key(fd: int) -> str:
    """Read one keypress from fd; returns a string key name."""
    ch = sys.stdin.read(1)
    if ch == "\x1b":
        # Possible escape sequence
        try:
            seq = sys.stdin.read(2)
        except Exception:
            return "esc"
        if seq == "[A":
            return "up"
        if seq == "[B":
            return "down"
        return "esc"
    return ch


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
    status_msg = ""
    console = Console(width=160)

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)

    try:
        tty.setraw(fd)
        with Live(
            build_layout(entries, selected_idx, status_msg),
            console=console,
            refresh_per_second=20,
            screen=True,
        ) as live:
            while True:
                key = _read_key(fd)
                status_msg = ""

                if key in ("q", "\x1b", "esc"):
                    break
                elif key in ("k", "up"):
                    selected_idx = max(0, selected_idx - 1)
                elif key in ("j", "down"):
                    selected_idx = min(len(entries) - 1, selected_idx + 1)
                elif key == "a":
                    entry = entries[selected_idx]
                    approve_candidate(entry)
                    status_msg = f"✓ Approved {entry.candidate_id}"
                    # Advance to next unapproved if possible
                    for i in range(selected_idx + 1, len(entries)):
                        if entries[i].status not in ("approved", "accepted"):
                            selected_idx = i
                            break
                elif key == "r":
                    entry = entries[selected_idx]
                    reject_candidate(entry)
                    status_msg = f"✗ Rejected {entry.candidate_id}"
                elif key == "p":
                    entry = entries[selected_idx]
                    if entry.midi_file.exists():
                        play_candidate(entry)
                        status_msg = f"▶ Playing {entry.midi_file.name}"
                    else:
                        status_msg = f"MIDI not found: {entry.midi_file.name}"

                live.update(build_layout(entries, selected_idx, status_msg))

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
