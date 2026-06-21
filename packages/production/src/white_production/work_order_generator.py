from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import yaml

from white_core.concepts.rainbow_table_color import the_rainbow_table_colors
from white_core.enums.chain_artifact_type import ChainArtifactType
from white_core.enums.collaborator_platform import CollaboratorPlatform
from white_core.enums.collaborator_role import CollaboratorRole
from white_core.enums.work_order_status import WorkOrderStatus
from white_core.music.core.work_order import WorkOrder

log = logging.getLogger(__name__)

_INSTRUMENTAL_SECTIONS = {
    "intro",
    "outro",
    "instrumental",
    "solo",
    "interlude",
    "break",
}


def _load_song_context(production_dir: Path) -> dict:
    ctx_path = production_dir / "song_context.yml"
    if ctx_path.exists():
        return yaml.safe_load(ctx_path.read_text()) or {}
    # Legacy fallback
    proposal_path = production_dir / "initial_proposal.yml"
    if proposal_path.exists():
        return yaml.safe_load(proposal_path.read_text()) or {}
    return {}


def _approved_sections(production_dir: Path) -> list[str]:
    review_path = production_dir / "chords" / "review.yml"
    if not review_path.exists():
        return []
    review = yaml.safe_load(review_path.read_text()) or {}
    seen: set[str] = set()
    sections: list[str] = []
    for candidate in review.get("candidates", []):
        status = str(candidate.get("status", "")).lower()
        if status not in ("approved", "accepted"):
            continue
        label = candidate.get("label", "")
        if not label:
            continue
        key = label.lower().replace("-", "_").replace(" ", "_")
        if key in seen:
            continue
        seen.add(key)
        # Include bar count if available
        hr = candidate.get("hr_distribution")
        if hr and isinstance(hr, list):
            bar_count = sum(h.get("bars", 1) for h in hr if isinstance(h, dict))
        else:
            bar_count = len(candidate.get("chords", [])) or 4
        sections.append(f"{label} ({bar_count} bars)")
    return sections


def _color_description(color_name: str) -> str:
    for color in the_rainbow_table_colors.values():
        if color.color_name.lower() == color_name.lower():
            parts = []
            if color.temporal_mode:
                parts.append(f"temporal: {color.temporal_mode}")
            if color.objectional_mode:
                parts.append(f"spatial: {color.objectional_mode}")
            if color.ontological_mode:
                parts.append(
                    f"ontological: {', '.join(str(m) for m in color.ontological_mode)}"
                )
            if parts:
                return f"{color.color_name} ({'; '.join(parts)})"
            return color.color_name
    return color_name


def _has_character_sheet(production_dir: Path) -> bool:
    # Character sheets live in <thread>/md/character_sheet*.md
    thread_dir = production_dir.parent.parent
    return bool(list(thread_dir.glob("md/character_sheet*.md")))


def _has_approved_melody(production_dir: Path) -> bool:
    melody_dir = production_dir / "melody" / "approved"
    if not melody_dir.exists():
        return False
    return bool(list(melody_dir.glob("*.mid")))


def generate_work_order(
    production_dir: Path,
    collaborator_id: str,
    role: CollaboratorRole | str,
    platform: CollaboratorPlatform | str = CollaboratorPlatform.DIRECT,
) -> WorkOrder:
    """Return a pre-populated draft WorkOrder from song pipeline data.

    Does NOT write the file — caller persists via save_work_order.
    """
    production_dir = Path(production_dir)
    if isinstance(role, str):
        role = CollaboratorRole(role)
    if isinstance(platform, str):
        platform = CollaboratorPlatform(platform)

    ctx = _load_song_context(production_dir)
    song_slug = production_dir.name

    time_sig_raw = ctx.get("time_sig") or ctx.get("time_signature") or "4/4"
    if isinstance(time_sig_raw, dict):
        time_sig = (
            f"{time_sig_raw.get('numerator', 4)}/{time_sig_raw.get('denominator', 4)}"
        )
    else:
        time_sig = str(time_sig_raw)

    color_raw = ctx.get("color") or ctx.get("rainbow_color") or ""
    if isinstance(color_raw, dict):
        color_name = color_raw.get("color_name", "")
    else:
        color_name = str(color_raw)

    concept = str(ctx.get("concept") or "")
    color_desc = _color_description(color_name)
    if concept and color_desc:
        creative_direction = f"{concept}\n\nChromatic target: {color_desc}."
    elif concept:
        creative_direction = concept
    elif color_desc:
        creative_direction = f"Chromatic target: {color_desc}."
    else:
        creative_direction = ""

    sections = _approved_sections(production_dir)

    artifact_types: list[ChainArtifactType] = [
        ChainArtifactType.PROPOSAL,
        ChainArtifactType.CHROMATIC_BRIEF,
    ]
    if _has_character_sheet(production_dir):
        artifact_types.append(ChainArtifactType.CHARACTER_SHEET)
    if role == CollaboratorRole.VOCALIST and _has_approved_melody(production_dir):
        artifact_types.append(ChainArtifactType.MELODY_MIDI_STEM)

    now = datetime.now(timezone.utc)
    return WorkOrder(
        id=f"{collaborator_id}-{song_slug}",
        song_slug=song_slug,
        collaborator_id=collaborator_id,
        role=role,
        platform=platform,
        status=WorkOrderStatus.DRAFT,
        key=str(ctx.get("key", "")),
        bpm=int(ctx.get("bpm") or 120),
        time_signature=time_sig,
        sections=sections,
        creative_direction=creative_direction,
        artifact_types=artifact_types,
        created_at=now,
        updated_at=now,
    )
