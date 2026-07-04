"""LP sequencing aesthetic advisor.

Read-only CLI Claude (or a human) can run to check the current `sides.yml`
arrangement against aesthetic goals — chromatic color balance and BPM/energy
spread — rather than duration alone (duration is handled by the sides UI/API
itself). Never modifies `sides.yml`; the human decides whether to act on
suggestions via the drag-and-drop sides page.

Usage:
    python -m white_composition.lp_sequence_advisor --album-dir shrink_wrapped
    python -m white_composition.lp_sequence_advisor --album-dir shrink_wrapped --output report.yml
    python -m white_composition.lp_sequence_advisor --dry-run
"""

import argparse
import os
from collections import Counter
from pathlib import Path

import yaml

from white_composition.lp_sides import SIDE_NAMES, load_sides, side_totals

# Flag a side's color distribution when one color makes up at least this
# fraction of its (>=2) placed songs.
COLOR_CLUSTER_THRESHOLD = 0.6


def _load_song_metadata(album_dir: Path, song_id: str) -> dict:
    """Read rainbow_color/bpm/mood/title for a song_id from its manifest_bootstrap.yml."""
    thread_slug, _, production_slug = song_id.partition("__")
    manifest_path = (
        Path(album_dir)
        / thread_slug
        / "production"
        / production_slug
        / "manifest_bootstrap.yml"
    )
    if not manifest_path.exists():
        return {}
    with open(manifest_path) as f:
        data = yaml.safe_load(f) or {}
    return {
        "title": data.get("title") or production_slug,
        "rainbow_color": data.get("rainbow_color"),
        "bpm": data.get("bpm"),
        "mood": data.get("mood") or [],
    }


def _generate_suggestions(
    side_reports: dict, song_meta: dict, sides: dict
) -> list[str]:
    """Plain-language suggestions for sides whose placed songs cluster on one color."""
    suggestions = []
    for side_name, report in side_reports.items():
        dist = report["color_distribution"]
        total = sum(dist.values())
        if total < 2:
            continue
        dominant_color, count = max(dist.items(), key=lambda kv: kv[1])
        fraction = count / total
        if fraction < COLOR_CLUSTER_THRESHOLD:
            continue

        candidate = None
        for other_side, other in sides.items():
            if other_side == side_name:
                continue
            for song in other.songs:
                color = song_meta.get(song.song_id, {}).get("rainbow_color")
                if color and color != dominant_color:
                    title = song_meta.get(song.song_id, {}).get("title", song.song_id)
                    candidate = (title, color, other_side)
                    break
            if candidate:
                break

        msg = (
            f"Side {side_name} is {fraction:.0%} {dominant_color} "
            f"({count}/{total} songs) — consider more color variety."
        )
        if candidate:
            title, color, other_side = candidate
            msg += f" '{title}' ({color}, currently on side {other_side}) could add contrast."
        suggestions.append(msg)
    return suggestions


def analyze_sides(album_dir: Path) -> dict:
    """Analyze the current sides.yml for color/BPM balance. Read-only."""
    album_dir = Path(album_dir)
    doc = load_sides(album_dir)
    totals = side_totals(doc)

    song_meta: dict[str, dict] = {}
    for side in doc.sides.values():
        for song in side.songs:
            song_meta[song.song_id] = _load_song_metadata(album_dir, song.song_id)

    side_reports: dict[str, dict] = {}
    for side_name in SIDE_NAMES:
        side = doc.sides[side_name]
        colors = [
            song_meta.get(s.song_id, {}).get("rainbow_color")
            for s in side.songs
            if song_meta.get(s.song_id, {}).get("rainbow_color")
        ]
        bpms = [
            song_meta.get(s.song_id, {}).get("bpm")
            for s in side.songs
            if song_meta.get(s.song_id, {}).get("bpm")
        ]
        side_reports[side_name] = {
            "song_count": len(side.songs),
            "total_seconds": totals[side_name]["total_seconds"],
            "over_limit": totals[side_name]["over_limit"],
            "color_distribution": dict(Counter(colors)),
            "bpm_range": {"min": min(bpms), "max": max(bpms)} if bpms else None,
        }

    suggestions = _generate_suggestions(side_reports, song_meta, doc.sides)

    return {
        "generated_from": str(album_dir),
        "side_limit_seconds": doc.side_limit_seconds,
        "sides": side_reports,
        "suggestions": suggestions,
    }


def format_report_text(report: dict) -> str:
    """Format an analysis report as human-readable text for stdout."""
    lines = []
    any_placed = any(s["song_count"] > 0 for s in report["sides"].values())
    if not any_placed:
        return "No songs are placed on any side yet — nothing to analyze."

    for side_name, side in report["sides"].items():
        if side["song_count"] == 0:
            lines.append(f"Side {side_name}: (empty)")
            continue
        colors = ", ".join(f"{c}×{n}" for c, n in side["color_distribution"].items())
        bpm = side["bpm_range"]
        bpm_str = f"{bpm['min']}-{bpm['max']} BPM" if bpm else "BPM unknown"
        limit_flag = " ⚠ OVER LIMIT" if side["over_limit"] else ""
        lines.append(
            f"Side {side_name}: {side['song_count']} song(s), "
            f"{side['total_seconds']:.0f}s{limit_flag} — colors: {colors or 'unknown'} — {bpm_str}"
        )

    if report["suggestions"]:
        lines.append("")
        lines.append("Suggestions:")
        for s in report["suggestions"]:
            lines.append(f"  - {s}")

    return "\n".join(lines)


def write_report(output_path: Path, report: dict) -> Path:
    with open(output_path, "w") as f:
        yaml.dump(
            report,
            f,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
            width=float("inf"),
        )
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Analyze LP-side sequencing for color/BPM balance (read-only)"
    )
    _sw_dir = os.getenv("SHRINKWRAP_OUTPUT_DIR", "shrink_wrapped")
    parser.add_argument(
        "--album-dir",
        type=Path,
        default=Path(_sw_dir),
        help="Album (shrink_wrapped) root directory (default: $SHRINKWRAP_OUTPUT_DIR)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Also write the report to this YAML path",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print the report without writing"
    )

    args = parser.parse_args()
    report = analyze_sides(args.album_dir)

    print(format_report_text(report))

    if args.dry_run:
        return
    if args.output:
        path = write_report(args.output, report)
        print(f"\nWrote report to {path}")


if __name__ == "__main__":
    main()
