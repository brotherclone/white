import shutil
from pathlib import Path

import pytest
import yaml

from white_composition.logic_handoff import (
    MixStage,
    add_version,
    handoff,
    read_composition,
    write_stage,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def seed_logicx(tmp_path: Path) -> Path:
    """Create a minimal fake seed.logicx.logicx bundle."""
    seed = tmp_path / "seed.logicx.logicx"
    seed.mkdir()
    (seed / "projectData").write_text("fake")
    return seed


@pytest.fixture()
def production_dir(tmp_path: Path) -> Path:
    prod = tmp_path / "production" / "my_song_v1"
    prod.mkdir(parents=True)
    song_context = {
        "title": "My Test Song",
        "thread": "test-thread",
        "color": "orange",
        "concept": "A test concept.",
    }
    with open(prod / "song_context.yml", "w") as f:
        yaml.dump(song_context, f)
    for phase in ["chords", "drums", "bass", "melody"]:
        approved = prod / phase / "approved"
        approved.mkdir(parents=True)
        (approved / f"{phase}_loop.mid").write_bytes(b"\x4d\x54\x68\x64")
    (prod / "arrangement.txt").write_text("Intro 8 bars\nVerse 16 bars")
    return prod


@pytest.fixture()
def logic_output_dir(tmp_path: Path) -> Path:
    d = tmp_path / "Logic"
    d.mkdir()
    return d


@pytest.fixture(autouse=True)
def env_vars(
    logic_output_dir: Path, monkeypatch: pytest.MonkeyPatch, seed_logicx: Path
):
    monkeypatch.setenv("LOGIC_OUTPUT_DIR", str(logic_output_dir))
    import white_composition.logic_handoff as lh

    monkeypatch.setattr(lh, "SEED_PATH", seed_logicx)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_scaffold_creates_folder_and_logicx(
    production_dir: Path, logic_output_dir: Path
):
    song_dir = handoff(production_dir)

    assert song_dir.is_dir()
    logicx = song_dir / "My Test Song.logicx"
    assert logicx.is_dir(), "seed.logicx.logicx should be copied and renamed"
    assert (logicx / "projectData").exists()


def test_composition_yml_written_on_first_handoff(production_dir: Path):
    song_dir = handoff(production_dir)
    comp = read_composition(song_dir)

    assert comp is not None
    assert comp["song_title"] == "My Test Song"
    assert comp["current_stage"] == MixStage.STRUCTURE.value
    assert comp["current_version"] == 1
    assert len(comp["versions"]) == 1


def test_rehandoff_skips_seed_copy_preserves_composition(production_dir: Path):
    song_dir = handoff(production_dir)
    write_stage(song_dir, MixStage.LYRICS.value)

    handoff(production_dir)
    comp_after = read_composition(song_dir)

    assert (
        comp_after["current_stage"] == MixStage.LYRICS.value
    ), "composition.yml should not be overwritten"


def test_midi_copied_into_phase_subfolders(production_dir: Path):
    song_dir = handoff(production_dir)

    for phase in ["chords", "drums", "bass", "melody"]:
        phase_dir = song_dir / "MIDI" / phase
        assert phase_dir.is_dir()
        mid_files = list(phase_dir.glob("*.mid"))
        assert len(mid_files) == 1, f"Expected 1 MIDI file in {phase}"


def test_empty_approved_phase_creates_empty_folder(production_dir: Path):
    shutil.rmtree(production_dir / "drums" / "approved")
    song_dir = handoff(production_dir)
    assert (song_dir / "MIDI" / "drums").is_dir()
    assert list((song_dir / "MIDI" / "drums").glob("*.mid")) == []


def test_arrangement_txt_moved(production_dir: Path):
    song_dir = handoff(production_dir)
    assert (song_dir / "arrangement.txt").exists()
    assert not (production_dir / "arrangement.txt").exists()


def test_no_text_files_no_error(production_dir: Path):
    (production_dir / "arrangement.txt").unlink()
    song_dir = handoff(production_dir)
    assert song_dir.is_dir()


def test_missing_logic_output_dir_raises(
    production_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.delenv("LOGIC_OUTPUT_DIR", raising=False)
    with pytest.raises(EnvironmentError, match="LOGIC_OUTPUT_DIR"):
        handoff(production_dir)


def test_write_stage_updates_composition(production_dir: Path):
    song_dir = handoff(production_dir)
    write_stage(song_dir, MixStage.RECORDING.value)
    comp = read_composition(song_dir)
    assert comp["current_stage"] == MixStage.RECORDING.value


def test_write_stage_invalid_raises(production_dir: Path):
    song_dir = handoff(production_dir)
    with pytest.raises(ValueError, match="Invalid stage"):
        write_stage(song_dir, "not_a_stage")


def test_add_version_increments(production_dir: Path):
    song_dir = handoff(production_dir)
    write_stage(song_dir, MixStage.ROUGH_MIX.value)
    new_ver = add_version(song_dir)
    comp = read_composition(song_dir)

    assert new_ver == 2
    assert comp["current_version"] == 2
    assert comp["current_stage"] == MixStage.STRUCTURE.value
    assert len(comp["versions"]) == 2
