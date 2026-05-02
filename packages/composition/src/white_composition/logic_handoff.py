from __future__ import annotations

import os
import shutil
from datetime import date
from enum import Enum
from pathlib import Path

import yaml

COMPOSITION_FILENAME = "composition.yml"
SEED_PATH = (
    Path(__file__).parents[4] / "packages" / "composition" / "logic" / "seed.logicx"
)

MIDI_PHASES = ["chords", "drums", "bass", "melody"]
LYRICS_PATTERNS = ["lyrics*.txt", "lyrics*.lrc", "*.lrc"]


class MixStage(str, Enum):
    STRUCTURE = "structure"
    LYRICS = "lyrics"
    RECORDING = "recording"
    VOCAL_PLACEHOLDERS = "vocal_placeholders"
    AUGMENTATION = "augmentation"
    CLEANING = "cleaning"
    ROUGH_MIX = "rough_mix"
    MIX_CANDIDATE = "mix_candidate"
    FINAL_MIX = "final_mix"


_STAGE_ORDER = [s.value for s in MixStage]


def next_stage(current: str) -> str | None:
    try:
        idx = _STAGE_ORDER.index(current)
    except ValueError:
        return None
    return _STAGE_ORDER[idx + 1] if idx + 1 < len(_STAGE_ORDER) else None


def _logic_output_dir() -> Path:
    val = os.environ.get("LOGIC_OUTPUT_DIR", "")
    if not val:
        raise EnvironmentError("LOGIC_OUTPUT_DIR is not set — add it to .env")
    return Path(val)


def _song_dir(production_dir: Path) -> Path:
    from white_composition.init_production import load_song_context

    ctx = load_song_context(production_dir)
    thread_slug = ctx.get("thread") or production_dir.parent.parent.name
    title = ctx.get("title") or production_dir.name
    safe_title = title.replace("/", "-").replace(":", "-")
    return _logic_output_dir() / thread_slug / safe_title


def handoff(production_dir: Path) -> Path:
    production_dir = Path(production_dir)
    song_dir = _song_dir(production_dir)

    # 1. Scaffold Logic project folder
    if not song_dir.exists():
        song_dir.mkdir(parents=True)
        dest = song_dir / f"{song_dir.name}.logicx"
        if SEED_PATH.exists():
            shutil.copytree(SEED_PATH, dest)
        else:
            print(f"WARNING: seed.logicx not found at {SEED_PATH}")
    else:
        print(f"Logic folder already exists, skipping seed.logicx copy: {song_dir}")

    # 2. Copy approved MIDI into phase subfolders
    midi_root = song_dir / "MIDI"
    for phase in MIDI_PHASES:
        phase_midi_dir = midi_root / phase
        phase_midi_dir.mkdir(parents=True, exist_ok=True)
        approved = production_dir / phase / "approved"
        if approved.is_dir():
            for mid in approved.glob("*.mid"):
                shutil.copy2(mid, phase_midi_dir / mid.name)

    # 3. Copy lyrics into Logic song folder (prod → Logic)
    for pattern in LYRICS_PATTERNS:
        for src in production_dir.glob(pattern):
            shutil.copy2(src, song_dir / src.name)

    # 4. Sync arrangement.txt back from Logic folder → production dir so that
    # lyric gen and drift report can read it after the human has arranged in Logic.
    logic_arrangement = song_dir / "arrangement.txt"
    if logic_arrangement.exists():
        shutil.copy2(logic_arrangement, production_dir / "arrangement.txt")
        print(f"Synced arrangement.txt from Logic → {production_dir}")

    # 5. Create composition.yml if absent
    comp_path = song_dir / COMPOSITION_FILENAME
    if not comp_path.exists():
        from white_composition.init_production import load_song_context

        ctx = load_song_context(production_dir)
        composition = {
            "song_title": ctx.get("title", production_dir.name),
            "thread_slug": ctx.get("thread") or production_dir.parent.parent.name,
            "production_slug": production_dir.name,
            "logic_project_path": str(song_dir),
            "current_version": 1,
            "current_stage": MixStage.STRUCTURE.value,
            "versions": [
                {
                    "version": 1,
                    "created": date.today().isoformat(),
                    "stage": MixStage.STRUCTURE.value,
                    "notes": "",
                }
            ],
        }
        with open(comp_path, "w") as f:
            yaml.dump(
                composition,
                f,
                default_flow_style=False,
                sort_keys=False,
                allow_unicode=True,
                width=float("inf"),
            )

    return song_dir


def read_composition(logic_song_dir: Path) -> dict | None:
    path = Path(logic_song_dir) / COMPOSITION_FILENAME
    if not path.exists():
        return None
    with open(path) as f:
        return yaml.safe_load(f) or {}


def write_stage(logic_song_dir: Path, stage: str) -> None:
    if stage not in _STAGE_ORDER:
        raise ValueError(f"Invalid stage '{stage}'. Must be one of: {_STAGE_ORDER}")
    path = Path(logic_song_dir) / COMPOSITION_FILENAME
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    data["current_stage"] = stage
    with open(path, "w") as f:
        yaml.dump(
            data,
            f,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
            width=float("inf"),
        )


def add_version(logic_song_dir: Path) -> int:
    path = Path(logic_song_dir) / COMPOSITION_FILENAME
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    new_version = (data.get("current_version") or 1) + 1
    data["current_version"] = new_version
    data["current_stage"] = MixStage.STRUCTURE.value
    versions = data.get("versions") or []
    versions.append(
        {
            "version": new_version,
            "created": date.today().isoformat(),
            "stage": MixStage.STRUCTURE.value,
            "notes": "",
        }
    )
    data["versions"] = versions
    with open(path, "w") as f:
        yaml.dump(
            data,
            f,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
            width=float("inf"),
        )
    return new_version


def update_version_notes(logic_song_dir: Path, version: int, notes: str) -> None:
    path = Path(logic_song_dir) / COMPOSITION_FILENAME
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    versions = data.get("versions") or []
    for v in versions:
        if v.get("version") == version:
            v["notes"] = notes
            break
    else:
        raise ValueError(f"Version {version} not found in composition.yml")
    data["versions"] = versions
    with open(path, "w") as f:
        yaml.dump(
            data,
            f,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
            width=float("inf"),
        )


def resolve_song_dir(production_dir: Path) -> Path:
    """Return the Logic song dir for a production dir without running handoff."""
    return _song_dir(Path(production_dir))
