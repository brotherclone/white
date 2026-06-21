from pathlib import Path

import yaml
from white_production.work_order_generator import generate_work_order

from white_core.enums.chain_artifact_type import ChainArtifactType
from white_core.enums.collaborator_platform import CollaboratorPlatform
from white_core.enums.collaborator_role import CollaboratorRole
from white_core.enums.work_order_status import WorkOrderStatus


def _write_song_context(prod_dir: Path, **overrides) -> None:
    defaults = {
        "title": "The Archivist's Rebellion",
        "bpm": 112,
        "time_sig": "4/4",
        "key": "D minor",
        "color": "Red",
        "concept": "A librarian discovers forbidden knowledge and must choose between silence and truth.",
    }
    defaults.update(overrides)
    (prod_dir / "song_context.yml").write_text(yaml.dump(defaults, width=float("inf")))


def _write_chord_review(prod_dir: Path, labels: list[str]) -> None:
    chord_dir = prod_dir / "chords"
    chord_dir.mkdir(parents=True, exist_ok=True)
    candidates = [
        {
            "label": label,
            "status": "approved",
            "chords": ["I", "IV", "V", "I"],
            "hr_distribution": [{"bars": 4}],
        }
        for label in labels
    ]
    (chord_dir / "review.yml").write_text(
        yaml.dump({"bpm": 112, "candidates": candidates}, width=float("inf"))
    )


def test_basic_generation(tmp_path):
    prod_dir = tmp_path / "the-archivists-rebellion"
    prod_dir.mkdir()
    _write_song_context(prod_dir)
    _write_chord_review(prod_dir, ["verse", "chorus"])

    wo = generate_work_order(prod_dir, "kate-koherence", CollaboratorRole.VOCALIST)
    assert wo.song_slug == "the-archivists-rebellion"
    assert wo.collaborator_id == "kate-koherence"
    assert wo.role == CollaboratorRole.VOCALIST
    assert wo.status == WorkOrderStatus.DRAFT
    assert wo.bpm == 112
    assert wo.key == "D minor"
    assert wo.time_signature == "4/4"


def test_sections_populated(tmp_path):
    prod_dir = tmp_path / "the-song"
    prod_dir.mkdir()
    _write_song_context(prod_dir)
    _write_chord_review(prod_dir, ["verse", "chorus", "bridge"])

    wo = generate_work_order(prod_dir, "kate-koherence", CollaboratorRole.VOCALIST)
    assert len(wo.sections) == 3
    assert any("verse" in s.lower() for s in wo.sections)
    assert any("chorus" in s.lower() for s in wo.sections)


def test_default_artifact_types(tmp_path):
    prod_dir = tmp_path / "the-song"
    prod_dir.mkdir()
    _write_song_context(prod_dir)

    wo = generate_work_order(prod_dir, "kate-koherence", CollaboratorRole.VOCALIST)
    assert ChainArtifactType.PROPOSAL in wo.artifact_types
    assert ChainArtifactType.CHROMATIC_BRIEF in wo.artifact_types


def test_character_sheet_included_when_present(tmp_path):
    prod_dir = tmp_path / "thread" / "production" / "the-song"
    prod_dir.mkdir(parents=True)
    md_dir = tmp_path / "thread" / "md"
    md_dir.mkdir(parents=True)
    (md_dir / "character_sheet.md").write_text("# Character Sheet")
    _write_song_context(prod_dir)

    wo = generate_work_order(prod_dir, "kate-koherence", CollaboratorRole.VOCALIST)
    assert ChainArtifactType.CHARACTER_SHEET in wo.artifact_types


def test_character_sheet_excluded_when_absent(tmp_path):
    prod_dir = tmp_path / "thread" / "production" / "the-song"
    prod_dir.mkdir(parents=True)
    _write_song_context(prod_dir)

    wo = generate_work_order(prod_dir, "kate-koherence", CollaboratorRole.VOCALIST)
    assert ChainArtifactType.CHARACTER_SHEET not in wo.artifact_types


def test_melody_midi_stem_for_vocalist_with_approved_midi(tmp_path):
    prod_dir = tmp_path / "thread" / "production" / "the-song"
    melody_dir = prod_dir / "melody" / "approved"
    melody_dir.mkdir(parents=True)
    (melody_dir / "melody_verse.mid").write_bytes(b"")
    _write_song_context(prod_dir)

    wo = generate_work_order(prod_dir, "kate-koherence", CollaboratorRole.VOCALIST)
    assert ChainArtifactType.MELODY_MIDI_STEM in wo.artifact_types


def test_melody_midi_stem_excluded_for_non_vocalist(tmp_path):
    prod_dir = tmp_path / "thread" / "production" / "the-song"
    melody_dir = prod_dir / "melody" / "approved"
    melody_dir.mkdir(parents=True)
    (melody_dir / "melody_verse.mid").write_bytes(b"")
    _write_song_context(prod_dir)

    wo = generate_work_order(prod_dir, "graham", CollaboratorRole.DRUMMER)
    assert ChainArtifactType.MELODY_MIDI_STEM not in wo.artifact_types


def test_melody_midi_stem_excluded_when_no_approved_midi(tmp_path):
    prod_dir = tmp_path / "thread" / "production" / "the-song"
    prod_dir.mkdir(parents=True)
    _write_song_context(prod_dir)

    wo = generate_work_order(prod_dir, "kate-koherence", CollaboratorRole.VOCALIST)
    assert ChainArtifactType.MELODY_MIDI_STEM not in wo.artifact_types


def test_creative_direction_includes_concept_and_color(tmp_path):
    prod_dir = tmp_path / "the-song"
    prod_dir.mkdir()
    _write_song_context(
        prod_dir, concept="Ancient grief crystallised into sound.", color="Blue"
    )

    wo = generate_work_order(prod_dir, "kate-koherence", CollaboratorRole.VOCALIST)
    assert "Ancient grief" in wo.creative_direction
    assert "Blue" in wo.creative_direction


def test_no_chord_review_yields_empty_sections(tmp_path):
    prod_dir = tmp_path / "the-song"
    prod_dir.mkdir()
    _write_song_context(prod_dir)

    wo = generate_work_order(prod_dir, "kate-koherence", CollaboratorRole.VOCALIST)
    assert wo.sections == []


def test_string_role_and_platform_accepted(tmp_path):
    prod_dir = tmp_path / "the-song"
    prod_dir.mkdir()
    _write_song_context(prod_dir)

    wo = generate_work_order(prod_dir, "kate-koherence", "vocalist", "airgigs")
    assert wo.role == CollaboratorRole.VOCALIST
    assert wo.platform == CollaboratorPlatform.AIRGIGS


def test_id_is_collaborator_plus_song_slug(tmp_path):
    prod_dir = tmp_path / "my-song"
    prod_dir.mkdir()
    _write_song_context(prod_dir)

    wo = generate_work_order(prod_dir, "kate-k", CollaboratorRole.VOCALIST)
    assert wo.id == "kate-k-my-song"


def test_missing_song_context_yields_defaults(tmp_path):
    prod_dir = tmp_path / "the-song"
    prod_dir.mkdir()
    # No song_context.yml

    wo = generate_work_order(prod_dir, "kate-koherence", CollaboratorRole.VOCALIST)
    assert wo.bpm == 120
    assert wo.key == ""
    assert wo.time_signature == "4/4"
