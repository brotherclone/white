#!/usr/bin/env python3
"""FastAPI backend for the web-based candidate browser.

Serves candidates as JSON, handles approve/reject writes, and streams MIDI files.
The Next.js frontend at localhost:3000 consumes this API.

Usage (single-song mode):
    python -m app.tools.candidate_server --production-dir shrink_wrapped/<album>/production/<slug>

Usage (album mode):
    python -m app.tools.candidate_server --shrink-wrapped-dir shrink_wrapped/
"""

import argparse
import contextlib
import io
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import urllib.parse
import wave
import webbrowser
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import uvicorn
import yaml
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field
from white_diary import ENTRIES_DIR

from white_api.candidate_browser import (
    CandidateEntry,
    approve_candidate,
    load_all_candidates,
    reject_candidate,
    set_label,
    set_use_case,
)
from white_api.routes.collaborators import make_collaborators_router
from white_api.routes.diary import make_diary_router
from white_api.routes.work_orders import make_work_orders_router

VALID_PHASES = {"chords", "drums", "bass", "melody", "lyrics", "quartet"}
EVOLVABLE_PHASES = {"drums", "bass", "melody"}

_EVOLVE_PIPELINE = {
    "drums": "white_generation.pipelines.drum_pipeline",
    "bass": "white_generation.pipelines.bass_pipeline",
    "melody": "white_generation.pipelines.melody_pipeline",
}

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Audio allowlist root — files served by /audio and /samples/export must live
# under this directory.  Override in tests by monkeypatching this variable.
# ---------------------------------------------------------------------------

_AUDIO_ROOT: Path = Path(__file__).parents[4].resolve()

# Parquet was built with an old absolute path for staged_raw_material.
# Remap to wherever it actually lives now.
_STAGED_RAW_MATERIAL_ROOT: Path = (
    _AUDIO_ROOT / "packages" / "extraction" / "staged_raw_material"
)
_STAGED_RAW_MATERIAL_ANCHOR = "staged_raw_material"


def _resolve_audio_path(raw: str | Path) -> Path:
    """Translate a parquet-baked path to the current filesystem location.

    Finds 'staged_raw_material' in the path parts and rewrites everything
    before it to the current location, avoiding symlink-expansion mismatches.
    """
    p = Path(raw)
    if p.exists():
        return p
    parts = p.parts
    try:
        idx = parts.index(_STAGED_RAW_MATERIAL_ANCHOR)
    except ValueError:
        return p
    rel = Path(*parts[idx + 1 :])
    candidate = _STAGED_RAW_MATERIAL_ROOT / rel
    return candidate if candidate.exists() else p


# ---------------------------------------------------------------------------
# Module-level state (mutated at startup and by /songs/activate)
# ---------------------------------------------------------------------------

_production_dir: Path | None = None
_shrink_wrapped_dir: Path | None = None
_active_song: dict | None = None

# Generate job state — one job at a time per server process
_generate_job: dict = {
    "status": "idle",
    "started_at": None,
    "finished_at": None,
    "error": None,
}

# Pipeline run job state — one phase run at a time
_run_job: dict = {
    "status": "idle",
    "phase": None,
    "started_at": None,
    "finished_at": None,
    "error": None,
}

# Logic handoff job state
_handoff_job: dict = {
    "status": "idle",
    "started_at": None,
    "finished_at": None,
    "error": None,
}

# Drift report job state
_drift_job: dict = {
    "status": "idle",
    "started_at": None,
    "finished_at": None,
    "error": None,
}


# ---------------------------------------------------------------------------
# Song scanning
# ---------------------------------------------------------------------------


def _coerce_key_string(key) -> str | None:
    """Normalise a key value that may be a string or a KeySignature dict."""
    if key is None:
        return None
    if isinstance(key, str):
        return key or None
    if isinstance(key, dict):
        note = key.get("note") or {}
        mode = key.get("mode") or {}
        pitch = note.get("pitch_name", "") if isinstance(note, dict) else ""
        mode_name = mode.get("name", "") if isinstance(mode, dict) else ""
        parts = [p for p in (pitch, mode_name) if p]
        return " ".join(parts) or None
    return str(key) or None


def _find_proposal(manifest_path: Path) -> Path | None:
    """Locate the song proposal yml for a production dir.

    Convention: <thread>/yml/<production_slug>.yml
    """
    prod_dir = manifest_path.parent
    production_slug = prod_dir.name
    thread_dir = prod_dir.parent.parent
    candidate = thread_dir / "yml" / f"{production_slug}.yml"
    return candidate if candidate.exists() else None


_MIX_STAGE_TO_SONG_STAGE: dict[str, str] = {
    "structure": "composition",
    "lyrics": "composition",
    "vocal_placeholders": "composition",
    "recording": "production",
    "augmentation": "production",
    "cleaning": "production",
    "rough_mix": "mixing",
    "mix_candidate": "mixing",
    "final_mix": "complete",
}


_VALID_SCHEMA_PREFIXES = {"", "1", "1.", "2"}


def _schema_is_valid(sv: str | None) -> bool:
    if sv is None:
        return True  # absent = legacy 1.x, tolerated
    sv = str(sv)
    return sv == "" or sv.startswith("1") or sv.startswith("2")


_LIFECYCLE_STATUSES = {"merged", "abandoned", "scrapped"}
_LP_CONSIDERATION_STATUSES = {"not_considered", "candidate", "placed"}


def _compute_stage(prod_dir: Path) -> str:
    bootstrap_path = prod_dir / "manifest_bootstrap.yml"
    if bootstrap_path.exists():
        try:
            with open(bootstrap_path) as f:
                mb = yaml.safe_load(f) or {}
            lc = mb.get("lifecycle_status")
            if lc in _LIFECYCLE_STATUSES:
                return lc
            sv = mb.get("schema_version")
            if sv is not None and not _schema_is_valid(sv):
                return "invalid"
            if not mb.get("title") and not mb.get("rainbow_color"):
                return "invalid"
        except Exception:
            return "invalid"

    if not (prod_dir / "song_context.yml").exists():
        return "ideation"
    try:
        from white_composition.logic_handoff import (
            COMPOSITION_FILENAME,
            resolve_song_dir,
        )

        logic_dir = resolve_song_dir(prod_dir)
        comp_path = logic_dir / COMPOSITION_FILENAME
        if comp_path.exists():
            with open(comp_path) as f:
                comp = yaml.safe_load(f) or {}
            mix_stage = comp.get("current_stage", "structure")
            return _MIX_STAGE_TO_SONG_STAGE.get(mix_stage, "composition")
    except Exception:
        pass
    return "generation"


def _has_mix_file(prod_dir: Path) -> bool:
    ctx_path = prod_dir / "song_context.yml"
    if not ctx_path.exists():
        return False
    try:
        with open(ctx_path) as f:
            ctx = yaml.safe_load(f) or {}
        mix = ctx.get("mix_file")
        return bool(mix and Path(mix).exists())
    except Exception:
        return False


def _read_concept(prod_dir: Path) -> str | None:
    ctx_path = prod_dir / "song_context.yml"
    if not ctx_path.exists():
        return None
    try:
        with open(ctx_path) as f:
            ctx = yaml.safe_load(f) or {}
        raw = ctx.get("concept")
        if not raw:
            return None
        text = str(raw).strip()
        return text or None
    except Exception:
        return None


def scan_songs(shrink_wrapped_dir: Path) -> list[dict]:
    """Walk shrink_wrapped_dir for manifest_bootstrap.yml files and return song entries."""
    songs = []
    for manifest_path in sorted(
        shrink_wrapped_dir.glob("*/production/*/manifest_bootstrap.yml")
    ):
        parts = manifest_path.parts
        thread_slug = parts[-4]
        production_slug = parts[-2]
        try:
            with open(manifest_path) as f:
                data = yaml.safe_load(f)
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        prod_dir = manifest_path.parent
        proposal_path = _find_proposal(manifest_path)
        sv = data.get("schema_version")
        schema_version = str(sv) if sv is not None else "1.x"
        songs.append(
            {
                "id": f"{thread_slug}__{production_slug}",
                "thread_slug": thread_slug,
                "production_slug": production_slug,
                "production_path": str(prod_dir),
                "title": data.get("title") or production_slug,
                "key": _coerce_key_string(data.get("key")),
                "bpm": data.get("bpm"),
                "time_sig": data.get("time_sig"),
                "rainbow_color": data.get("rainbow_color"),
                "singer": data.get("singer"),
                "schema_version": schema_version,
                "stub": bool(data.get("stub", False)),
                "initialized": (prod_dir / "song_context.yml").exists(),
                "has_mix": _has_mix_file(prod_dir),
                "stage": _compute_stage(prod_dir),
                "proposal_path": str(proposal_path) if proposal_path else None,
                "concept": _read_concept(prod_dir),
                # lifecycle fields — absent in older manifests, default to None/[]
                "lifecycle_status": data.get("lifecycle_status"),
                "merged_with": data.get("merged_with") or [],
                "uses_parts_from": data.get("uses_parts_from") or [],
                "lp_consideration": data.get("lp_consideration") or "not_considered",
            }
        )
    return songs


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app(
    production_dir: Path | None = None,
    *,
    shrink_wrapped_dir: Path | None = None,
    registry_dir: Path | None = None,
) -> FastAPI:
    global _production_dir, _shrink_wrapped_dir, _active_song
    _production_dir = production_dir
    _shrink_wrapped_dir = shrink_wrapped_dir
    _active_song = None

    app = FastAPI(title="Candidate Browser API")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
        allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
        allow_headers=["*"],
    )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _require_production_dir() -> Path:
        if _production_dir is None:
            raise HTTPException(
                status_code=503,
                detail="No song selected — POST /songs/activate first",
            )
        return _production_dir

    # ------------------------------------------------------------------
    # Collaborator, work-order, and diary routers
    # ------------------------------------------------------------------

    app.include_router(
        make_collaborators_router(
            lambda: _shrink_wrapped_dir, registry_dir=registry_dir
        )
    )
    app.include_router(
        make_work_orders_router(
            _require_production_dir,
            lambda: _shrink_wrapped_dir,
            registry_dir=registry_dir,
        )
    )
    app.include_router(make_diary_router(ENTRIES_DIR))

    def _all_candidates(
        phase: str | None = None, section: str | None = None
    ) -> list[CandidateEntry]:
        return load_all_candidates(
            _require_production_dir(), phase_filter=phase, section_filter=section
        )

    def _find(candidate_id: str, phase: str | None = None) -> CandidateEntry:
        for entry in _all_candidates(phase):
            if entry.candidate_id == candidate_id:
                return entry
        raise HTTPException(
            status_code=404, detail=f"Candidate '{candidate_id}' not found"
        )

    def _serialise(entry: CandidateEntry) -> dict:
        d = asdict(entry)
        d.pop("midi_file", None)
        d.pop("review_yml", None)
        d["id"] = entry.candidate_id
        d["midi_url"] = f"/midi/{entry.candidate_id}"
        return d

    # ------------------------------------------------------------------
    # Songs (album mode)
    # ------------------------------------------------------------------

    @app.get("/songs")
    def list_songs():
        if _shrink_wrapped_dir is None:
            raise HTTPException(
                status_code=503,
                detail="Server not in album mode — launch with --shrink-wrapped-dir",
            )
        return scan_songs(_shrink_wrapped_dir)

    class ActivateBody(BaseModel):
        id: str

    @app.post("/songs/activate")
    def activate_song(body: ActivateBody):
        global _production_dir, _active_song
        if _shrink_wrapped_dir is None:
            raise HTTPException(
                status_code=503,
                detail="Server not in album mode — launch with --shrink-wrapped-dir",
            )
        songs = scan_songs(_shrink_wrapped_dir)
        match = next((s for s in songs if s["id"] == body.id), None)
        if match is None:
            raise HTTPException(status_code=404, detail=f"Song '{body.id}' not found")
        _production_dir = Path(match["production_path"])
        _active_song = match
        return {"ok": True, "production_dir": str(_production_dir)}

    @app.get("/songs/active")
    def get_active_song():
        if _active_song and _production_dir:
            _active_song["stage"] = _compute_stage(_production_dir)
        return {"active": _active_song}

    def _resolve_song(song_id: str) -> tuple[dict, Path]:
        """Return (song_entry, prod_dir) for song_id, or raise 404."""
        if _shrink_wrapped_dir is None:
            raise HTTPException(status_code=503, detail="Server not in album mode")
        songs = scan_songs(_shrink_wrapped_dir)
        entry = next((s for s in songs if s["id"] == song_id), None)
        if entry is None:
            raise HTTPException(status_code=404, detail=f"Song '{song_id}' not found")
        return entry, Path(entry["production_path"])

    def _patch_manifest(prod_dir: Path, updates: dict) -> None:
        """Merge updates into manifest_bootstrap.yml atomically."""
        bootstrap_path = prod_dir / "manifest_bootstrap.yml"
        data: dict = {}
        if bootstrap_path.exists():
            with open(bootstrap_path) as f:
                loaded = yaml.safe_load(f)
            if isinstance(loaded, dict):
                data = loaded
        data.update(updates)
        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=bootstrap_path.parent, suffix=".yml.tmp"
        )
        try:
            with os.fdopen(tmp_fd, "w") as f:
                yaml.dump(
                    data, f, allow_unicode=True, sort_keys=False, width=float("inf")
                )
            os.replace(tmp_path, bootstrap_path)
        except Exception:
            with contextlib.suppress(OSError):
                os.unlink(tmp_path)
            raise

    def _set_lp_consideration(song_id: str, prod_dir: Path, status: str) -> None:
        """Patch lp_consideration and keep the active-song cache fresh if needed."""
        global _active_song
        _patch_manifest(prod_dir, {"lp_consideration": status})
        if (
            _shrink_wrapped_dir
            and _active_song
            and _active_song.get("production_path") == str(prod_dir)
        ):
            refreshed = next(
                (s for s in scan_songs(_shrink_wrapped_dir) if s["id"] == song_id),
                None,
            )
            if refreshed:
                _active_song = refreshed

    class LifecycleBody(BaseModel):
        status: str
        merged_with: list[str] = Field(default_factory=list)

    @app.post("/songs/{song_id}/lifecycle")
    def set_lifecycle(song_id: str, body: LifecycleBody):
        global _active_song
        if body.status not in _LIFECYCLE_STATUSES:
            raise HTTPException(
                status_code=422,
                detail=f"status must be one of: {sorted(_LIFECYCLE_STATUSES)}",
            )
        _entry, prod_dir = _resolve_song(song_id)
        if body.status == "merged":
            if not body.merged_with:
                raise HTTPException(
                    status_code=422,
                    detail="merged_with must contain at least one song ID",
                )
            # Resolve all partner IDs first so we 404 before writing anything.
            partner_dirs: list[tuple[str, Path]] = []
            for partner_id in body.merged_with:
                _pe, partner_dir = _resolve_song(partner_id)
                partner_dirs.append((partner_id, partner_dir))
            # Bilateral write — active song points to partners, each partner points back.
            _patch_manifest(
                prod_dir,
                {"lifecycle_status": "merged", "merged_with": body.merged_with},
            )
            for partner_id, partner_dir in partner_dirs:
                partner_merged_with = [song_id] + [
                    p for p in body.merged_with if p != partner_id
                ]
                _patch_manifest(
                    partner_dir,
                    {"lifecycle_status": "merged", "merged_with": partner_merged_with},
                )
        else:
            _patch_manifest(prod_dir, {"lifecycle_status": body.status})
        # Keep _active_song cache fresh if the patched song is currently active.
        if (
            _shrink_wrapped_dir
            and _active_song
            and _active_song.get("production_path") == str(prod_dir)
        ):
            refreshed = next(
                (s for s in scan_songs(_shrink_wrapped_dir) if s["id"] == song_id),
                None,
            )
            if refreshed:
                _active_song = refreshed
        return {"ok": True, "status": body.status}

    class LpConsiderationBody(BaseModel):
        status: str

    @app.post("/songs/{song_id}/lp-consideration")
    def set_lp_consideration(song_id: str, body: LpConsiderationBody):
        if body.status not in _LP_CONSIDERATION_STATUSES:
            raise HTTPException(
                status_code=422,
                detail=f"status must be one of: {sorted(_LP_CONSIDERATION_STATUSES)}",
            )
        _entry, prod_dir = _resolve_song(song_id)
        _set_lp_consideration(song_id, prod_dir, body.status)
        return {"ok": True, "status": body.status}

    @app.get("/songs/scrapped")
    def list_scrapped_songs():
        if _shrink_wrapped_dir is None:
            raise HTTPException(status_code=503, detail="Server not in album mode")
        return [
            s
            for s in scan_songs(_shrink_wrapped_dir)
            if s.get("lifecycle_status") == "scrapped"
        ]

    class UsesPartsFromBody(BaseModel):
        uses_parts_from: list[str]

    @app.patch("/songs/{song_id}/uses-parts-from")
    def set_uses_parts_from(song_id: str, body: UsesPartsFromBody):
        global _active_song
        _entry, prod_dir = _resolve_song(song_id)
        _patch_manifest(prod_dir, {"uses_parts_from": body.uses_parts_from})
        # Keep _active_song cache fresh if the patched song is currently active.
        if (
            _shrink_wrapped_dir
            and _active_song
            and _active_song.get("production_path") == str(prod_dir)
        ):
            refreshed = next(
                (s for s in scan_songs(_shrink_wrapped_dir) if s["id"] == song_id),
                None,
            )
            if refreshed:
                _active_song = refreshed
        return {"ok": True}

    # ------------------------------------------------------------------
    # Pipeline — init, run, status
    # ------------------------------------------------------------------

    @app.post("/pipeline/init")
    def pipeline_init():
        """Run init_production for the active song if not already initialized."""
        prod = _require_production_dir()
        if (prod / "song_context.yml").exists():
            from white_composition.pipeline_runner import (
                read_phase_statuses,
                write_phase_status,
            )

            statuses = read_phase_statuses(prod)
            if statuses.get("init_production") == "pending":
                write_phase_status(prod, "init_production", "promoted")
            return {"ok": True, "skipped": True}
        proposal_path = _active_song.get("proposal_path") if _active_song else None
        if not proposal_path:
            raise HTTPException(
                status_code=422,
                detail="No proposal yml found for this song — cannot initialize",
            )
        proposal = Path(proposal_path)
        if not proposal.exists():
            raise HTTPException(
                status_code=422,
                detail=f"Proposal file not found: {proposal_path}",
            )
        try:
            from white_composition.init_production import init_production
            from white_composition.pipeline_runner import write_phase_status

            init_production(prod, proposal)
            write_phase_status(prod, "init_production", "promoted")
        except (SystemExit, ValueError) as exc:
            raise HTTPException(
                status_code=500, detail=f"init_production failed: {exc}"
            ) from exc
        return {"ok": True, "skipped": False}

    @app.post("/pipeline/run")
    def pipeline_run():
        """Run the next pending pipeline phase for the active song."""
        global _run_job
        if _run_job["status"] == "running":
            raise HTTPException(status_code=409, detail="A phase is already running")
        prod = _require_production_dir()
        now = datetime.now(timezone.utc).isoformat()
        _run_job = {
            "status": "running",
            "phase": None,
            "started_at": now,
            "finished_at": None,
            "error": None,
        }

        def _run():
            global _run_job
            try:
                from white_composition.pipeline_runner import (
                    cmd_run,
                    get_next_runnable_phase,
                    read_phase_statuses,
                )

                statuses = read_phase_statuses(prod)
                next_phase = get_next_runnable_phase(statuses)
                _run_job["phase"] = next_phase
                if next_phase is None:
                    _run_job["status"] = "done"
                    return
                from white_composition.pipeline_runner import (
                    PHASE_REVIEW_FILES,
                    write_phase_status,
                )

                proposal = (_active_song or {}).get("proposal_path") or ""
                result = cmd_run(prod, song_proposal=proposal)
                if result != 0:
                    # cmd_run resets status to pending on non-zero exit.
                    # If candidates were generated anyway, recover the correct status.
                    review_file = PHASE_REVIEW_FILES.get(next_phase)
                    if review_file and (prod / review_file).exists():
                        write_phase_status(prod, next_phase, "generated")
                        _run_job["status"] = "done"
                    else:
                        _run_job["status"] = "error"
                        _run_job["error"] = (
                            f"Phase '{next_phase}' exited with code {result}"
                        )
                else:
                    _run_job["status"] = "done"
            except Exception as exc:
                _run_job["status"] = "error"
                _run_job["error"] = str(exc)
            finally:
                _run_job["finished_at"] = datetime.now(timezone.utc).isoformat()

        threading.Thread(target=_run, daemon=True).start()
        return {"status": "running", "started_at": now}

    @app.get("/pipeline/run/status")
    def pipeline_run_status():
        return _run_job

    @app.post("/pipeline/run/quartet")
    def pipeline_run_quartet():
        """Run the quartet (strings) generation phase for the active song."""
        global _run_job
        if _run_job["status"] == "running":
            raise HTTPException(status_code=409, detail="A phase is already running")
        prod = _require_production_dir()
        now = datetime.now(timezone.utc).isoformat()
        _run_job = {
            "status": "running",
            "phase": "quartet",
            "started_at": now,
            "finished_at": None,
            "error": None,
        }

        def _run():
            global _run_job
            try:
                from white_composition.pipeline_runner import (
                    PHASE_REVIEW_FILES,
                    write_phase_status,
                )

                write_phase_status(prod, "quartet", "in_progress")
                singer = (_active_song or {}).get("singer") or "gabriel"
                cmd = [
                    sys.executable,
                    "-m",
                    "white_generation.pipelines.quartet_pipeline",
                    "--production-dir",
                    str(prod),
                    "--singer",
                    singer,
                ]
                result = subprocess.run(cmd)
                review_file = PHASE_REVIEW_FILES.get("quartet")
                if result.returncode != 0:
                    if review_file and (prod / review_file).exists():
                        write_phase_status(prod, "quartet", "generated")
                        _run_job["status"] = "done"
                    else:
                        write_phase_status(prod, "quartet", "pending")
                        _run_job["status"] = "error"
                        _run_job["error"] = (
                            f"quartet exited with code {result.returncode}"
                        )
                else:
                    write_phase_status(prod, "quartet", "generated")
                    _run_job["status"] = "done"
            except Exception as exc:
                _run_job["status"] = "error"
                _run_job["error"] = str(exc)
            finally:
                _run_job["finished_at"] = datetime.now(timezone.utc).isoformat()

        threading.Thread(target=_run, daemon=True).start()
        return {"status": "running", "started_at": now}

    @app.get("/pipeline/status")
    def pipeline_status():
        """Return phase statuses for the active song."""
        prod = _require_production_dir()
        from white_composition.pipeline_runner import (
            PHASE_ORDER,
            get_next_runnable_phase,
            read_phase_statuses,
        )

        statuses = read_phase_statuses(prod)
        return {
            "initialized": (prod / "song_context.yml").exists(),
            "phases": statuses,
            "next_phase": get_next_runnable_phase(statuses),
            "phase_order": PHASE_ORDER,
        }

    # ------------------------------------------------------------------
    # Logic Handoff
    # ------------------------------------------------------------------

    @app.post("/handoff")
    def start_handoff():
        global _handoff_job
        if _handoff_job["status"] == "running":
            raise HTTPException(
                status_code=409, detail="A handoff job is already running"
            )
        prod = _require_production_dir()
        now = datetime.now(timezone.utc).isoformat()
        _handoff_job = {
            "status": "running",
            "started_at": now,
            "finished_at": None,
            "error": None,
        }

        def _run():
            global _handoff_job, _active_song
            try:
                from white_composition.logic_handoff import handoff

                handoff(prod)
                _handoff_job["status"] = "done"
                # Refresh _active_song so stage flips to "composition" immediately.
                if _shrink_wrapped_dir and _active_song:
                    song_id = _active_song.get("id")
                    refreshed = next(
                        (
                            s
                            for s in scan_songs(_shrink_wrapped_dir)
                            if s["id"] == song_id
                        ),
                        None,
                    )
                    if refreshed:
                        _active_song = refreshed
            except Exception as exc:
                _handoff_job["status"] = "error"
                _handoff_job["error"] = str(exc)
            finally:
                _handoff_job["finished_at"] = datetime.now(timezone.utc).isoformat()

        threading.Thread(target=_run, daemon=True).start()
        return {"status": "running", "started_at": now}

    @app.get("/handoff/status")
    def handoff_status():
        return _handoff_job

    # ------------------------------------------------------------------
    # Chromatic sample retrieval
    # ------------------------------------------------------------------

    _CLAP_PARQUET = (
        Path(__file__).parents[4]
        / "training"
        / "data"
        / "training_data_clap_embeddings.parquet"
    )
    _META_PARQUET = (
        Path(__file__).parents[4]
        / "training"
        / "data"
        / "training_data_with_embeddings.parquet"
    )

    _clap_df: object = None  # lazy-loaded DataFrame

    def _get_clap_df():
        nonlocal _clap_df
        if _clap_df is None:
            from white_composition.retrieve_samples import load_clap_index

            _clap_df = load_clap_index(str(_CLAP_PARQUET), str(_META_PARQUET))
        return _clap_df

    def _slice_wav_segment(
        wav_path: Path,
        start_sec: float | None,
        end_sec: float | None,
    ) -> bytes:
        """Extract [start_sec, end_sec] from a WAV and return as WAV bytes."""
        import math

        # pandas/pyarrow represents missing floats as NaN, not None;
        # use math.isnan directly so numpy.float64 values are handled correctly
        if start_sec is not None:
            try:
                if math.isnan(start_sec):
                    start_sec = None
            except (TypeError, ValueError):
                start_sec = None
        if end_sec is not None:
            try:
                if math.isnan(end_sec):
                    end_sec = None
            except (TypeError, ValueError):
                end_sec = None
        with wave.open(str(wav_path), "rb") as src:
            framerate = src.getframerate()
            total_frames = src.getnframes()
            n_channels = src.getnchannels()
            sampwidth = src.getsampwidth()
            start_frame = int(start_sec * framerate) if start_sec is not None else 0
            end_frame = (
                int(end_sec * framerate) if end_sec is not None else total_frames
            )
            start_frame = max(0, min(start_frame, total_frames))
            end_frame = max(start_frame, min(end_frame, total_frames))
            src.setpos(start_frame)
            frames = src.readframes(end_frame - start_frame)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as dst:
            dst.setnchannels(n_channels)
            dst.setsampwidth(sampwidth)
            dst.setframerate(framerate)
            dst.writeframes(frames)
        return buf.getvalue()

    def _check_audio_path(wav_path: Path) -> Path:
        resolved = wav_path.resolve()
        try:
            resolved.relative_to(_AUDIO_ROOT)
        except ValueError:
            raise HTTPException(
                status_code=403, detail="Audio file is outside allowed directory"
            )
        return resolved

    @app.get("/samples")
    def list_samples(top_n: int = 20):
        if _production_dir is None:
            raise HTTPException(
                status_code=503,
                detail="No song selected — POST /songs/activate first",
            )
        color = (_active_song or {}).get("rainbow_color") or None
        if not color:
            return []
        from white_composition.retrieve_samples import retrieve_by_color

        df = _get_clap_df()
        results = retrieve_by_color(df, color, top_n=top_n)
        for r in results:
            encoded = urllib.parse.quote(r["segment_id"], safe="")
            r["audio_url"] = f"/audio/{encoded}"
        return results

    @app.get("/audio/{segment_id:path}")
    def stream_audio(segment_id: str, request: Request):
        df = _get_clap_df()
        row = df[df["segment_id"] == segment_id]
        if row.empty:
            raise HTTPException(
                status_code=404, detail=f"Segment '{segment_id}' not found"
            )
        rec = row.iloc[0]
        wav_path = _check_audio_path(_resolve_audio_path(rec["source_audio_file"]))
        if not wav_path.exists():
            raise HTTPException(
                status_code=404, detail=f"WAV file not found: {wav_path}"
            )
        audio_bytes = _slice_wav_segment(
            wav_path,
            rec.get("start_seconds"),
            rec.get("end_seconds"),
        )
        total = len(audio_bytes)
        range_header = request.headers.get("range")
        if range_header:
            m = re.match(r"bytes=(\d*)-(\d*)", range_header)
            if m:
                start, end = 0, total - 1
                s, e = m.groups()
                if s:
                    start = int(s)
                if e:
                    end = min(int(e), total - 1)
                chunk = audio_bytes[start : end + 1]
                return Response(
                    content=chunk,
                    status_code=206,
                    media_type="audio/wav",
                    headers={
                        "Content-Range": f"bytes {start}-{end}/{total}",
                        "Content-Length": str(len(chunk)),
                        "Accept-Ranges": "bytes",
                    },
                )
        return Response(
            content=audio_bytes,
            media_type="audio/wav",
            headers={
                "Content-Length": str(total),
                "Accept-Ranges": "bytes",
            },
        )

    @app.post("/samples/{segment_id}/export")
    def export_sample(segment_id: str):
        logic_output_dir = os.environ.get("LOGIC_OUTPUT_DIR", "")
        if not logic_output_dir:
            raise HTTPException(
                status_code=503,
                detail="LOGIC_OUTPUT_DIR not set — add it to .env",
            )
        df = _get_clap_df()
        row = df[df["segment_id"] == segment_id]
        if row.empty:
            raise HTTPException(
                status_code=404, detail=f"Segment '{segment_id}' not found"
            )
        wav_path = _check_audio_path(
            _resolve_audio_path(row.iloc[0]["source_audio_file"])
        )
        if not wav_path.exists():
            raise HTTPException(
                status_code=404, detail=f"WAV file not found: {wav_path}"
            )
        if _active_song is None:
            raise HTTPException(
                status_code=503,
                detail="No song selected — POST /songs/activate first",
            )

        from white_composition.logic_handoff import resolve_song_dir

        try:
            song_dir = resolve_song_dir(Path(_active_song["production_path"]))
        except Exception as exc:
            raise HTTPException(
                status_code=500, detail=f"Could not resolve Logic project dir: {exc}"
            ) from exc

        logic_root = Path(logic_output_dir).resolve()
        dest_dir = (song_dir / "Samples").resolve()
        try:
            dest_dir.relative_to(logic_root)
        except ValueError:
            raise HTTPException(
                status_code=400, detail="Destination is outside LOGIC_OUTPUT_DIR"
            )
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / f"{segment_id}.wav"
        shutil.copy2(wav_path, dest)
        return {"ok": True, "dest": str(dest)}

    @app.post("/sync-arrangement")
    def sync_arrangement():
        """Pull arrangement.txt and MIDI files from Logic back into the production dir.

        Copies arrangement.txt from the Logic project folder, then copies every
        *.mid from Logic/MIDI/{phase}/ → production/{phase}/approved/ for all MIDI
        phases.  This allows chord/drum/bass/melody files created or edited in Logic
        to flow back into the pipeline.
        """
        prod = _require_production_dir()
        try:
            from white_composition.logic_handoff import (
                resolve_song_dir,
                sync_from_logic,
            )

            song_dir = resolve_song_dir(prod)
        except Exception as exc:
            raise HTTPException(
                status_code=500, detail=f"Could not resolve Logic project dir: {exc}"
            ) from exc

        logic_arrangement = song_dir / "arrangement.txt"
        if not logic_arrangement.exists():
            raise HTTPException(
                status_code=404,
                detail=f"No arrangement.txt found in Logic project at {logic_arrangement} — arrange the song in Logic first",
            )

        dest = prod / "arrangement.txt"
        shutil.copy2(logic_arrangement, dest)

        midi_synced = sync_from_logic(prod)

        return {
            "ok": True,
            "synced_from": str(logic_arrangement),
            "synced_to": str(dest),
            "midi_synced": midi_synced,
        }

    # ------------------------------------------------------------------
    # Plan Drift Report
    # ------------------------------------------------------------------

    class DriftReportBody(BaseModel):
        use_claude: bool = True

    @app.get("/drift-report")
    def get_drift_report():
        prod = _require_production_dir()
        from white_composition.drift_report import load_report

        report = load_report(prod)
        if report is None:
            raise HTTPException(
                status_code=404,
                detail="No plan_drift_report.yml found — POST /drift-report to generate",
            )
        return report.model_dump()

    @app.post("/drift-report")
    def start_drift_report(body: DriftReportBody = DriftReportBody()):
        global _drift_job
        if _drift_job["status"] == "running":
            raise HTTPException(
                status_code=409, detail="A drift report job is already running"
            )
        prod = _require_production_dir()
        arr_path = prod / "arrangement.txt"
        if not arr_path.exists():
            raise HTTPException(
                status_code=422,
                detail="arrangement.txt not found in production directory",
            )
        from white_composition.production_plan import PLAN_FILENAME, load_plan

        plan = load_plan(prod)
        if plan is None:
            raise HTTPException(
                status_code=422,
                detail=f"{PLAN_FILENAME} not found — generate a production plan first",
            )
        now = datetime.now(timezone.utc).isoformat()
        _drift_job = {
            "status": "running",
            "started_at": now,
            "finished_at": None,
            "error": None,
        }

        use_claude = body.use_claude

        def _run():
            global _drift_job
            try:
                from white_composition.drift_report import compare_plans, write_report

                report = compare_plans(plan, arr_path, use_claude=use_claude)
                write_report(prod, report)
                _drift_job["status"] = "done"
            except Exception as exc:
                _drift_job["status"] = "error"
                _drift_job["error"] = str(exc)
            finally:
                _drift_job["finished_at"] = datetime.now(timezone.utc).isoformat()

        threading.Thread(target=_run, daemon=True).start()
        return {"status": "running", "started_at": now}

    @app.get("/drift-report/status")
    def drift_report_status():
        return _drift_job

    @app.get("/composition")
    def get_composition():
        prod = _require_production_dir()
        from white_composition.logic_handoff import read_composition, resolve_song_dir

        song_dir = resolve_song_dir(prod)
        data = read_composition(song_dir)
        if data is None:
            return {"status": "not_initialized"}
        return data

    class BpmBody(BaseModel):
        bpm: int

    @app.post("/production/set-bpm")
    def set_bpm(body: BpmBody):
        """Change BPM for the active production.

        Updates song_context.yml, all phase review.yml files, retimes every
        approved MIDI, and regenerates assembled_melody.mid if present.
        """
        import mido as _mido
        import yaml as _yaml

        prod = _require_production_dir()
        new_bpm = body.bpm
        if new_bpm < 20 or new_bpm > 400:
            raise HTTPException(
                status_code=422, detail="BPM must be between 20 and 400"
            )

        new_tempo = _mido.bpm2tempo(new_bpm)
        updated_files: list[str] = []

        # 1. song_context.yml
        ctx_path = prod / "song_context.yml"
        if ctx_path.exists():
            with open(ctx_path) as f:
                ctx = _yaml.safe_load(f) or {}
            ctx["bpm"] = new_bpm
            with open(ctx_path, "w") as f:
                _yaml.dump(
                    ctx, f, allow_unicode=True, sort_keys=False, width=float("inf")
                )
            updated_files.append("song_context.yml")

        # 2. All phase review.yml files
        for review_path in prod.glob("*/review.yml"):
            with open(review_path) as f:
                rv = _yaml.safe_load(f) or {}
            if "bpm" in rv:
                rv["bpm"] = new_bpm
                tmp_fd, tmp_path = tempfile.mkstemp(
                    dir=review_path.parent, suffix=".yml.tmp"
                )
                try:
                    with os.fdopen(tmp_fd, "w") as f:
                        _yaml.dump(
                            rv,
                            f,
                            allow_unicode=True,
                            sort_keys=False,
                            width=float("inf"),
                        )
                    os.replace(tmp_path, review_path)
                except Exception:
                    with contextlib.suppress(OSError):
                        os.unlink(tmp_path)
                    raise
                updated_files.append(str(review_path.relative_to(prod)))

        # 3. Retime all approved MIDI files
        for mid_path in prod.glob("*/approved/*.mid"):
            try:
                mid = _mido.MidiFile(str(mid_path))
                changed = False
                for track in mid.tracks:
                    for msg in track:
                        if msg.type == "set_tempo":
                            msg.tempo = new_tempo
                            changed = True
                if changed:
                    mid.save(str(mid_path))
                    updated_files.append(str(mid_path.relative_to(prod)))
            except Exception:
                pass

        # 4. Regenerate assembled_melody.mid if it exists
        assembled = prod / "melody" / "assembled_melody.mid"
        arrangement = prod / "arrangement.txt"
        approved_dir = prod / "melody" / "approved"
        if assembled.exists() and arrangement.exists() and approved_dir.exists():
            try:
                from white_generation.pipelines.melody_auto_split import (
                    assemble_melody_midi,
                )

                ctx_path2 = prod / "song_context.yml"
                ts = "4/4"
                if ctx_path2.exists():
                    with open(ctx_path2) as f:
                        ts = str((_yaml.safe_load(f) or {}).get("time_sig", "4/4"))
                out = assemble_melody_midi(
                    arrangement_path=arrangement,
                    approved_dir=approved_dir,
                    bpm=new_bpm,
                    time_sig_str=ts,
                    output_path=assembled,
                )
                updated_files.append(str(out.relative_to(prod)))

                # Sync to Logic MIDI folder if handoff ran
                try:
                    from white_composition.logic_handoff import resolve_song_dir

                    logic_midi_dir = resolve_song_dir(prod) / "MIDI" / "melody"
                    if logic_midi_dir.is_dir():
                        shutil.copy2(out, logic_midi_dir / out.name)
                        updated_files.append(f"Logic/{out.name}")
                except Exception:
                    pass
            except Exception:
                pass  # assembled_melody regen is best-effort

        return {"ok": True, "bpm": new_bpm, "updated": updated_files}

    class StageBody(BaseModel):
        stage: str

    @app.patch("/composition/stage")
    def update_stage(body: StageBody):
        prod = _require_production_dir()
        from white_composition.logic_handoff import resolve_song_dir, write_stage

        song_dir = resolve_song_dir(prod)
        try:
            write_stage(song_dir, body.stage)
        except (ValueError, FileNotFoundError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {"ok": True, "stage": body.stage}

    @app.post("/composition/version")
    def new_version():
        prod = _require_production_dir()
        from white_composition.logic_handoff import add_version, resolve_song_dir

        song_dir = resolve_song_dir(prod)
        try:
            version = add_version(song_dir)
        except FileNotFoundError as exc:
            raise HTTPException(
                status_code=422, detail="composition.yml not found — run handoff first"
            ) from exc
        return {"ok": True, "version": version}

    class VersionNotesBody(BaseModel):
        version: int
        notes: str

    @app.patch("/composition/version/notes")
    def update_notes(body: VersionNotesBody):
        prod = _require_production_dir()
        from white_composition.logic_handoff import (
            resolve_song_dir,
            update_version_notes,
        )

        song_dir = resolve_song_dir(prod)
        try:
            update_version_notes(song_dir, body.version, body.notes)
        except (ValueError, FileNotFoundError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {"ok": True}

    # ------------------------------------------------------------------
    # Stage regression
    # ------------------------------------------------------------------

    class RegressBody(BaseModel):
        target_stage: str
        confirmed: bool = False
        diary_entry: str | None = None

    @app.post("/composition/regress")
    def regress_stage(body: RegressBody):
        prod = _require_production_dir()
        from white_composition.logic_handoff import (
            regression_info,
            resolve_song_dir,
            write_stage,
        )

        song_dir = resolve_song_dir(prod)
        try:
            from white_composition.logic_handoff import read_composition

            comp = read_composition(song_dir) or {}
            current = comp.get("current_stage", "structure")
            info = regression_info(song_dir, current, body.target_stage)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        if not body.confirmed:
            return info

        # Delete listed files
        for rel in info.get("files_to_delete", []):
            p = song_dir / rel
            if p.exists():
                try:
                    if p.is_dir():
                        import shutil as _shutil

                        _shutil.rmtree(p)
                    else:
                        p.unlink()
                except OSError as exc:
                    raise HTTPException(
                        status_code=500,
                        detail=f"Could not delete {rel}: {exc}",
                    ) from exc

        try:
            write_stage(song_dir, body.target_stage)
        except (ValueError, FileNotFoundError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        if body.diary_entry:
            try:
                from white_diary import write_entry

                write_entry(
                    song_slug=(_active_song or {}).get("production_slug", "unknown"),
                    author="system",
                    phase="regression",
                    title=f"Regressed to {body.target_stage}",
                    body=body.diary_entry,
                    tags=["regression"],
                    metadata={"from_stage": current, "to_stage": body.target_stage},
                )
            except Exception:
                pass

        return {"ok": True, "stage": body.target_stage}

    # ------------------------------------------------------------------
    # Mix file
    # ------------------------------------------------------------------

    @app.get("/production/mix/info")
    def mix_info():
        prod = _require_production_dir()
        from white_composition.init_production import load_song_context
        from white_composition.lp_sides import mix_duration_seconds

        ctx = load_song_context(prod)
        mix_file = ctx.get("mix_file") or None
        has_mix = bool(mix_file and Path(mix_file).exists())
        return {
            "has_mix": has_mix,
            "mix_file": mix_file,
            "duration_seconds": mix_duration_seconds(mix_file) if has_mix else None,
        }

    class MixFileBody(BaseModel):
        path: str

    def _write_mix_file(prod: Path, path: str) -> None:
        ctx_path = prod / "song_context.yml"
        if not ctx_path.exists():
            raise HTTPException(status_code=404, detail="song_context.yml not found")
        with open(ctx_path) as f:
            ctx = yaml.safe_load(f) or {}
        ctx["mix_file"] = path
        with open(ctx_path, "w") as f:
            yaml.dump(
                ctx,
                f,
                default_flow_style=False,
                sort_keys=False,
                allow_unicode=True,
                width=float("inf"),
            )

    @app.post("/production/mix/set")
    def set_mix_file(body: MixFileBody):
        prod = _require_production_dir()
        _write_mix_file(prod, body.path)
        return {"ok": True, "mix_file": body.path}

    @app.post("/songs/{song_id}/mix/set")
    def set_song_mix_file(song_id: str, body: MixFileBody):
        """Song-scoped mix write — resolves the production dir from song_id
        directly rather than the mutable `_production_dir` global, so a
        stale/desynced active-song pointer can't attach a mix to the wrong song.
        """
        _, prod_dir = _resolve_song(song_id)
        _write_mix_file(prod_dir, body.path)
        return {"ok": True, "mix_file": body.path}

    @app.get("/production/mix")
    def stream_mix():
        from fastapi.responses import FileResponse

        prod = _require_production_dir()
        from white_composition.init_production import load_song_context

        ctx = load_song_context(prod)
        mix_file = ctx.get("mix_file")
        if not mix_file:
            raise HTTPException(
                status_code=404, detail="No mix_file set in song_context.yml"
            )
        mix_path = Path(mix_file)
        if not mix_path.exists():
            raise HTTPException(
                status_code=404, detail=f"Mix file not found: {mix_path}"
            )
        suffix = mix_path.suffix.lower()
        media_type = {
            ".mp3": "audio/mpeg",
            ".wav": "audio/wav",
            ".aiff": "audio/aiff",
            ".aif": "audio/aiff",
            ".m4a": "audio/mp4",
        }.get(suffix, "audio/mpeg")
        return FileResponse(str(mix_path), media_type=media_type)

    # ------------------------------------------------------------------
    # LP-side sequencing
    # ------------------------------------------------------------------

    def _require_shrink_wrapped_dir() -> Path:
        if _shrink_wrapped_dir is None:
            raise HTTPException(
                status_code=503,
                detail="No album (shrink_wrapped) directory configured",
            )
        return _shrink_wrapped_dir

    def _resolve_song_for_sides(song_id: str) -> dict:
        album_dir = _require_shrink_wrapped_dir()
        entry = next((s for s in scan_songs(album_dir) if s["id"] == song_id), None)
        if entry is None:
            raise HTTPException(status_code=404, detail=f"Song '{song_id}' not found")
        return entry

    def _song_mix_duration(song_entry: dict) -> float | None:
        from white_composition.init_production import load_song_context
        from white_composition.lp_sides import mix_duration_seconds

        ctx = load_song_context(Path(song_entry["production_path"]))
        mix_file = ctx.get("mix_file")
        if not mix_file:
            return None
        return mix_duration_seconds(mix_file)

    @app.get("/sides")
    def list_sides():
        from white_composition.lp_sides import load_sides, side_totals

        album_dir = _require_shrink_wrapped_dir()
        doc = load_sides(album_dir)
        totals = side_totals(doc)
        return {
            "side_limit_seconds": doc.side_limit_seconds,
            "sides": {
                name: {
                    "songs": [s.model_dump() for s in side.songs],
                    **totals[name],
                }
                for name, side in doc.sides.items()
            },
        }

    class SideAssignBody(BaseModel):
        song_id: str
        position: int = 0

    @app.post("/sides/{side}/assign")
    def assign_to_side(side: str, body: SideAssignBody):
        from white_composition.lp_sides import (
            SIDE_NAMES,
            assign_song,
            load_sides,
            save_sides,
        )

        if side not in SIDE_NAMES:
            raise HTTPException(status_code=404, detail=f"Unknown side '{side}'")
        album_dir = _require_shrink_wrapped_dir()
        song_entry = _resolve_song_for_sides(body.song_id)
        if not song_entry["has_mix"]:
            raise HTTPException(
                status_code=400,
                detail=f"Song '{body.song_id}' has no mix file — cannot be sequenced",
            )
        duration = _song_mix_duration(song_entry)
        if duration is None:
            raise HTTPException(
                status_code=400,
                detail=f"Could not read mix duration for '{body.song_id}'",
            )
        doc = load_sides(album_dir)
        assign_song(doc, body.song_id, side, body.position, duration)
        save_sides(album_dir, doc)
        _set_lp_consideration(
            body.song_id, Path(song_entry["production_path"]), "placed"
        )
        return {
            "ok": True,
            "side": side,
            "songs": [s.model_dump() for s in doc.sides[side].songs],
        }

    class SideMoveBody(BaseModel):
        song_id: str
        to_side: str
        to_position: int = 0

    @app.post("/sides/{side}/move")
    def move_within_sides(side: str, body: SideMoveBody):
        from white_composition.lp_sides import (
            SIDE_NAMES,
            find_song_side,
            load_sides,
            move_song,
            save_sides,
        )

        if side not in SIDE_NAMES or body.to_side not in SIDE_NAMES:
            raise HTTPException(status_code=404, detail="Unknown side")
        album_dir = _require_shrink_wrapped_dir()
        doc = load_sides(album_dir)
        actual_side = find_song_side(doc, body.song_id)
        if actual_side != side:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"Song '{body.song_id}' is not on side '{side}' "
                    f"(currently on {actual_side!r})"
                ),
            )
        try:
            move_song(doc, body.song_id, body.to_side, body.to_position)
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
        save_sides(album_dir, doc)
        song_entry = _resolve_song_for_sides(body.song_id)
        _set_lp_consideration(
            body.song_id, Path(song_entry["production_path"]), "placed"
        )
        return {
            "ok": True,
            "side": body.to_side,
            "songs": [s.model_dump() for s in doc.sides[body.to_side].songs],
        }

    @app.delete("/sides/{side}/songs/{song_id}")
    def remove_from_side(side: str, song_id: str):
        from white_composition.lp_sides import (
            SIDE_NAMES,
            load_sides,
            remove_song,
            save_sides,
        )

        if side not in SIDE_NAMES:
            raise HTTPException(status_code=404, detail=f"Unknown side '{side}'")
        album_dir = _require_shrink_wrapped_dir()
        doc = load_sides(album_dir)
        if not any(s.song_id == song_id for s in doc.sides[side].songs):
            raise HTTPException(
                status_code=404, detail=f"Song '{song_id}' is not on side {side}"
            )
        remove_song(doc, song_id)
        save_sides(album_dir, doc)
        song_entry = _resolve_song_for_sides(song_id)
        next_status = "candidate" if song_entry["has_mix"] else "not_considered"
        _set_lp_consideration(song_id, Path(song_entry["production_path"]), next_status)
        return {"ok": True}

    @app.get("/songs/{song_id}/mix/info")
    def song_mix_info(song_id: str):
        from white_composition.init_production import load_song_context

        song_entry = _resolve_song_for_sides(song_id)
        ctx = load_song_context(Path(song_entry["production_path"]))
        mix_file = ctx.get("mix_file") or None
        has_mix = bool(mix_file and Path(mix_file).exists())
        return {
            "has_mix": has_mix,
            "mix_file": mix_file,
            "duration_seconds": _song_mix_duration(song_entry) if has_mix else None,
        }

    @app.get("/songs/{song_id}/mix")
    def stream_song_mix(song_id: str):
        from fastapi.responses import FileResponse

        from white_composition.init_production import load_song_context

        song_entry = _resolve_song_for_sides(song_id)
        ctx = load_song_context(Path(song_entry["production_path"]))
        mix_file = ctx.get("mix_file")
        if not mix_file:
            raise HTTPException(
                status_code=404, detail="No mix_file set in song_context.yml"
            )
        mix_path = Path(mix_file)
        if not mix_path.exists():
            raise HTTPException(
                status_code=404, detail=f"Mix file not found: {mix_path}"
            )
        suffix = mix_path.suffix.lower()
        media_type = {
            ".mp3": "audio/mpeg",
            ".wav": "audio/wav",
            ".aiff": "audio/aiff",
            ".aif": "audio/aiff",
            ".m4a": "audio/mp4",
        }.get(suffix, "audio/mpeg")
        return FileResponse(str(mix_path), media_type=media_type)

    # ------------------------------------------------------------------
    # Listening-playlist sync
    # ------------------------------------------------------------------

    @app.get("/playlists/config")
    def get_playlist_config():
        from white_composition.playlist_sync import load_playlist_config

        album_dir = _require_shrink_wrapped_dir()
        config = load_playlist_config(album_dir)
        return {"output_dir": config.output_dir}

    class PlaylistConfigBody(BaseModel):
        output_dir: str

    @app.post("/playlists/config")
    def set_playlist_config(body: PlaylistConfigBody):
        from white_composition.playlist_sync import PlaylistConfig, save_playlist_config

        album_dir = _require_shrink_wrapped_dir()
        save_playlist_config(album_dir, PlaylistConfig(output_dir=body.output_dir))
        return {"ok": True, "output_dir": body.output_dir}

    @app.get("/playlists/material")
    def get_material_produced():
        from white_composition.playlist_sync import (
            DEFAULT_MATERIAL_TARGET_SECONDS,
            load_playlist_config,
            material_produced_seconds,
        )

        album_dir = _require_shrink_wrapped_dir()
        config = load_playlist_config(album_dir)
        total_seconds, file_count = material_produced_seconds(config.output_dir)
        return {
            "total_seconds": total_seconds,
            "file_count": file_count,
            "target_seconds": DEFAULT_MATERIAL_TARGET_SECONDS,
        }

    @app.post("/playlists/sync")
    def sync_playlists_endpoint():
        from white_composition.lp_sides import load_sides
        from white_composition.playlist_sync import load_playlist_config, sync_playlists

        album_dir = _require_shrink_wrapped_dir()
        config = load_playlist_config(album_dir)
        sides_doc = load_sides(album_dir)
        songs = scan_songs(album_dir)
        try:
            counts = sync_playlists(songs, sides_doc, config.output_dir)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        return counts

    # ------------------------------------------------------------------
    # Lyrics review
    # ------------------------------------------------------------------

    _FITTING_VERDICT_ORDER = [
        "splits needed",
        "tight but workable",
        "paste-ready",
        "spacious",
    ]

    def _worst_fitting_verdict(fitting: dict) -> str | None:
        """Return the worst-case fitting verdict across all sections and phrases."""
        worst_idx = len(_FITTING_VERDICT_ORDER)
        for section_data in fitting.values():
            if not isinstance(section_data, dict):
                continue
            for phrase in section_data.get("phrases", []):
                v = phrase.get("verdict", "")
                try:
                    idx = _FITTING_VERDICT_ORDER.index(v)
                    worst_idx = min(worst_idx, idx)
                except ValueError:
                    pass
        return (
            _FITTING_VERDICT_ORDER[worst_idx]
            if worst_idx < len(_FITTING_VERDICT_ORDER)
            else None
        )

    @app.get("/lyrics")
    def get_lyrics():
        prod = _require_production_dir()
        review_path = prod / "melody" / "lyrics_review.yml"
        if not review_path.exists():
            raise HTTPException(
                status_code=404,
                detail="lyrics_review.yml not found — run lyric pipeline first",
            )
        with open(review_path) as f:
            review = yaml.safe_load(f) or {}
        promoted = (prod / "melody" / "lyrics.txt").exists()
        candidates_out = []
        for c in review.get("candidates", []):
            txt_path = prod / "melody" / c.get("file", "")
            text = txt_path.read_text() if txt_path.exists() else ""
            chromatic = c.get("chromatic") or {}
            fitting_verdict = _worst_fitting_verdict(c.get("fitting") or {})
            status = c.get("status", "pending")
            if promoted and status == "approved":
                status = "promoted"
            candidates_out.append(
                {
                    "id": c["id"],
                    "rank": c.get("rank", 0),
                    "status": status,
                    "text": text,
                    "match": chromatic.get("match"),
                    "fitting_verdict": fitting_verdict,
                }
            )
        return {
            "status": "promoted" if promoted else "pending",
            "candidates": candidates_out,
        }

    @app.post("/lyrics/{lyric_id}/approve")
    def approve_lyric(lyric_id: str):
        prod = _require_production_dir()
        review_path = prod / "melody" / "lyrics_review.yml"
        if not review_path.exists():
            raise HTTPException(status_code=404, detail="lyrics_review.yml not found")
        with open(review_path) as f:
            review = yaml.safe_load(f) or {}
        candidates = review.get("candidates", [])
        found = False
        for c in candidates:
            if c["id"] == lyric_id:
                c["status"] = "approved"
                found = True
            else:
                c["status"] = "pending"
        if not found:
            raise HTTPException(
                status_code=422, detail=f"Lyric candidate '{lyric_id}' not found"
            )
        review["candidates"] = candidates
        with open(review_path, "w") as f:
            yaml.dump(
                review,
                f,
                default_flow_style=False,
                sort_keys=False,
                allow_unicode=True,
                width=float("inf"),
            )
        return {"ok": True}

    @app.post("/lyrics/reset")
    def reset_lyrics():
        """Reset the lyric phase to pending and wipe all lyric artefacts.

        Use after the arrangement changes post-lyric-generation so the lyric
        pipeline can be re-run cleanly against the updated arrangement.txt.
        """
        prod = _require_production_dir()
        melody_dir = prod / "melody"
        from white_composition.pipeline_runner import write_phase_status

        removed = []
        for name in ("lyrics.txt", "lyrics_draft.txt", "lyrics_review.yml"):
            p = melody_dir / name
            if p.exists():
                p.unlink()
                removed.append(name)
        candidates_dir = melody_dir / "candidates"
        if candidates_dir.exists():
            for txt in candidates_dir.glob("lyrics_*.txt"):
                txt.unlink()
                removed.append(f"candidates/{txt.name}")

        write_phase_status(prod, "lyrics", "pending")
        return {"ok": True, "removed": removed}

    # ------------------------------------------------------------------
    # Candidates
    # ------------------------------------------------------------------

    @app.get("/candidates")
    def list_candidates(phase: str | None = None, section: str | None = None):
        entries = _all_candidates(phase, section)
        return [_serialise(e) for e in entries]

    @app.post("/candidates/{candidate_id}/approve")
    def approve(candidate_id: str):
        entry = _find(candidate_id)
        approve_candidate(entry)
        return {"ok": True, "id": candidate_id, "status": "approved"}

    @app.post("/candidates/{candidate_id}/reject")
    def reject(candidate_id: str):
        entry = _find(candidate_id)
        reject_candidate(entry)
        return {"ok": True, "id": candidate_id, "status": "rejected"}

    class LabelBody(BaseModel):
        label: str

    @app.patch("/candidates/{candidate_id}/label")
    def label(candidate_id: str, body: LabelBody):
        entry = _find(candidate_id)
        set_label(entry, body.label)
        return {"ok": True, "id": candidate_id, "label": body.label}

    class UseCaseBody(BaseModel):
        use_case: str

    @app.patch("/candidates/{candidate_id}/use_case")
    def use_case(candidate_id: str, body: UseCaseBody):
        entry = _find(candidate_id)
        set_use_case(entry, body.use_case)
        return {"ok": True, "id": candidate_id, "use_case": body.use_case}

    @app.get("/midi/{candidate_id}")
    def get_midi(candidate_id: str):
        entry = _find(candidate_id)
        if not entry.midi_file.exists():
            raise HTTPException(status_code=404, detail="MIDI file not found")
        return FileResponse(
            path=str(entry.midi_file),
            media_type="audio/midi",
            filename=entry.midi_file.name,
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/songs/{song_id}/midi/{candidate_id}")
    def get_song_midi(song_id: str, candidate_id: str):
        """Song-scoped MIDI lookup — bypasses the mutable `_production_dir`
        global so candidate ids that repeat across songs (e.g. chord_001..010)
        can't resolve against whichever song was last activated.
        """
        _, prod_dir = _resolve_song(song_id)
        entries = load_all_candidates(prod_dir)
        entry = next((e for e in entries if e.candidate_id == candidate_id), None)
        if entry is None:
            raise HTTPException(
                status_code=404, detail=f"Candidate '{candidate_id}' not found"
            )
        if not entry.midi_file.exists():
            raise HTTPException(status_code=404, detail="MIDI file not found")
        return FileResponse(
            path=str(entry.midi_file),
            media_type="audio/midi",
            filename=entry.midi_file.name,
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/production-dir")
    def get_production_dir():
        prod = _require_production_dir()
        return {"production_dir": str(prod)}

    # ------------------------------------------------------------------
    # Promote
    # ------------------------------------------------------------------

    class PromoteBody(BaseModel):
        phase: str

    @app.post("/promote")
    def promote(body: PromoteBody):
        prod = _require_production_dir()
        if body.phase not in VALID_PHASES:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid phase '{body.phase}'. Must be one of: {sorted(VALID_PHASES)}",
            )
        from white_composition.pipeline_runner import PHASE_REVIEW_FILES, cmd_promote

        review_file = PHASE_REVIEW_FILES.get(body.phase, f"{body.phase}/review.yml")
        review_path = prod / review_file
        if not review_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"No review file found for phase '{body.phase}'",
            )
        approved_dir = prod / body.phase / "approved"
        count_before = (
            len(list(approved_dir.glob("*.mid"))) if approved_dir.exists() else 0
        )

        captured = io.StringIO()
        with contextlib.redirect_stdout(captured):
            result = cmd_promote(prod, body.phase, yes=True)
        if result != 0:
            reason_lines = [
                line
                for line in captured.getvalue().splitlines()
                if "ERROR" in line or "No approved candidates" in line
            ]
            reason = (
                " ".join(reason_lines)
                or "ensure review.yml contains approved candidates"
            )
            raise HTTPException(
                status_code=409,
                detail=f"Promotion could not be completed for phase '{body.phase}' — {reason}",
            )
        count_after = (
            len(list(approved_dir.glob("*.mid"))) if approved_dir.exists() else 0
        )
        return {"ok": True, "promoted_count": max(0, count_after - count_before)}

    # ------------------------------------------------------------------
    # Register non-generated part
    # ------------------------------------------------------------------

    @app.post("/production/register-part")
    async def register_part_endpoint(
        midi_file: UploadFile = File(...),
        phase: str = Form(...),
        section: str = Form(...),
        label: str = Form(...),
    ):
        prod = _require_production_dir()
        if phase not in VALID_PHASES:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid phase '{phase}'. Must be one of: {sorted(VALID_PHASES)}",
            )
        content = await midi_file.read()
        with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        try:
            from white_composition.promote_part import register_part

            entry_dict = register_part(tmp_path, phase, section, label, prod)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        finally:
            os.unlink(tmp_path)

        review_yml = prod / phase / "review.yml"
        from white_api.candidate_browser import _load_review

        entries = _load_review(review_yml) if review_yml.exists() else []
        matched = next((e for e in entries if e.candidate_id == entry_dict["id"]), None)
        if matched is None:
            raise HTTPException(
                status_code=500,
                detail="Registered entry could not be reloaded from review.yml",
            )
        return _serialise(matched)

    # ------------------------------------------------------------------
    # Melody auto-split
    # ------------------------------------------------------------------

    class AutoSplitMelodyBody(BaseModel):
        phase_label: str
        min_split_beats: float = Field(default=1.0, gt=0)

    @app.post("/production/auto-split-melody")
    def auto_split_melody_endpoint(body: AutoSplitMelodyBody):
        prod = _require_production_dir()
        midi_path = prod / "melody" / "approved" / f"{body.phase_label}.mid"
        if not midi_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"No approved melody MIDI for label '{body.phase_label}' at {midi_path}",
            )
        lyrics_path = prod / "melody" / "lyrics.txt"
        if not lyrics_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"No lyrics.txt found at {lyrics_path}",
            )
        try:
            import mido as _mido

            src = _mido.MidiFile(str(midi_path))
            ticks_per_beat = src.ticks_per_beat or 480
            min_split_ticks = max(1, int(body.min_split_beats * ticks_per_beat))

            from white_generation.pipelines.melody_auto_split import auto_split_melody

            output_path, alignment = auto_split_melody(
                midi_path=midi_path,
                lyrics_path=lyrics_path,
                section=body.phase_label,
                min_split_ticks=min_split_ticks,
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        uncovered_count = sum(1 for a in alignment if a.get("uncovered"))
        return {
            "ok": True,
            "split_midi": str(output_path),
            "alignment": alignment,
            "uncovered_phrase_count": uncovered_count,
        }

    @app.post("/production/auto-split-melody/all")
    def auto_split_melody_all_endpoint():
        """Auto-split every lyric section instance — one split MIDI per instance."""
        prod = _require_production_dir()
        approved_dir = prod / "melody" / "approved"
        lyrics_path = prod / "melody" / "lyrics.txt"
        if not approved_dir.exists():
            raise HTTPException(
                status_code=404, detail="No approved melody directory found"
            )
        if not lyrics_path.exists():
            raise HTTPException(status_code=404, detail="No lyrics.txt found")

        try:
            from white_generation.pipelines.melody_auto_split import (
                auto_split_all_instances,
            )

            results = auto_split_all_instances(
                lyrics_path=lyrics_path,
                approved_dir=approved_dir,
                min_split_ticks=480,
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        # Sync split MIDIs into the Logic MIDI folder if handoff already ran
        logic_midi_dir: Path | None = None
        try:
            from white_composition.logic_handoff import resolve_song_dir

            song_dir = resolve_song_dir(prod)
            candidate = song_dir / "MIDI" / "melody"
            if candidate.is_dir():
                logic_midi_dir = candidate
        except Exception:
            pass

        if logic_midi_dir:
            for r in results:
                if not r.get("skipped"):
                    split_path = Path(r["split_midi"])
                    if split_path.exists():
                        shutil.copy2(split_path, logic_midi_dir / split_path.name)

        return {
            "ok": True,
            "results": results,
            "warnings": [r["warning"] for r in results if r.get("warning")],
            "logic_midi_dir": str(logic_midi_dir) if logic_midi_dir else None,
        }

    @app.post("/production/assemble-melody")
    def assemble_melody_endpoint():
        """Assemble all instance-split MIDIs into one full-length melody MIDI."""
        prod = _require_production_dir()
        arrangement_path = prod / "arrangement.txt"
        approved_dir = prod / "melody" / "approved"

        if not arrangement_path.exists():
            raise HTTPException(
                status_code=404,
                detail="No arrangement.txt — export from Logic first",
            )
        if not approved_dir.exists():
            raise HTTPException(
                status_code=404, detail="No approved melody directory found"
            )

        # Load BPM and time_sig from chord review
        chord_review_path = prod / "chords" / "review.yml"
        bpm = 120
        time_sig = "4/4"
        if chord_review_path.exists():
            import yaml as _yaml

            with open(chord_review_path) as f:
                cr = _yaml.safe_load(f) or {}
            bpm = int(cr.get("bpm", 120))
            time_sig = str(cr.get("time_sig", "4/4"))

        try:
            from white_generation.pipelines.melody_auto_split import (
                assemble_melody_midi,
                write_assembled_lyrics,
            )

            output_path = assemble_melody_midi(
                arrangement_path=arrangement_path,
                approved_dir=approved_dir,
                bpm=bpm,
                time_sig_str=time_sig,
            )

            assembled_lyrics_path: Path | None = None
            lyrics_path = prod / "melody" / "lyrics.txt"
            if lyrics_path.exists():
                assembled_lyrics_path = write_assembled_lyrics(
                    arrangement_path=arrangement_path,
                    lyrics_path=lyrics_path,
                )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        # Sync to Logic MIDI folder if handoff already ran
        logic_midi_dir: Path | None = None
        try:
            from white_composition.logic_handoff import resolve_song_dir

            song_dir = resolve_song_dir(prod)
            candidate = song_dir / "MIDI" / "melody"
            if candidate.is_dir():
                logic_midi_dir = candidate
        except Exception:
            pass

        if logic_midi_dir:
            shutil.copy2(output_path, logic_midi_dir / output_path.name)

        return {
            "ok": True,
            "assembled_midi": str(output_path),
            "assembled_lyrics": (
                str(assembled_lyrics_path) if assembled_lyrics_path else None
            ),
            "logic_midi_dir": str(logic_midi_dir) if logic_midi_dir else None,
        }

    # ------------------------------------------------------------------
    # Evolve
    # ------------------------------------------------------------------

    def _snapshot_decided(review_path: Path) -> tuple[list[dict], dict[str, bytes]]:
        """Return (decided_candidates, {midi_rel: bytes}) for approved/rejected entries."""
        if not review_path.exists():
            return [], {}
        with open(review_path) as f:
            data = yaml.safe_load(f) or {}
        decided = [
            c
            for c in (data.get("candidates") or [])
            if str(c.get("status", "pending")).lower() in ("approved", "rejected")
        ]
        midi_bytes: dict[str, bytes] = {}
        for c in decided:
            rel = c.get("midi_file", "")
            if rel:
                p = review_path.parent / rel
                if p.exists():
                    midi_bytes[rel] = p.read_bytes()
        return decided, midi_bytes

    def _merge_decided(
        review_path: Path, decided: list[dict], midi_bytes: dict[str, bytes]
    ) -> int:
        """Re-inject decided candidates into a freshly written review.yml.

        Restores any MIDI files that the pipeline wiped, then inserts the
        decided entries before the new pending ones so they appear at the top.
        Returns the number of re-injected entries.
        """
        if not decided:
            return 0

        # Restore MIDI files the pipeline may have deleted or overwritten
        for rel, data in midi_bytes.items():
            dest = review_path.parent / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
        # Load the review.yml the pipeline wrote (or start from scratch if it
        # failed before writing — preserve decided candidates either way).
        if review_path.exists():
            with open(review_path) as f:
                review = yaml.safe_load(f) or {}
        else:
            review = {}
        new_candidates = review.get("candidates") or []
        # Drop any new pending that share an id with a decided entry
        decided_ids = {c["id"] for c in decided}
        new_candidates = [c for c in new_candidates if c.get("id") not in decided_ids]
        review["candidates"] = decided + new_candidates
        # Atomic write so a mid-dump crash (or a concurrent writer) can't
        # leave review.yml empty or racing another writer's temp file.
        review_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_fd, tmp_path = tempfile.mkstemp(dir=review_path.parent, suffix=".yml.tmp")
        try:
            with os.fdopen(tmp_fd, "w") as f:
                yaml.dump(
                    review,
                    f,
                    default_flow_style=False,
                    sort_keys=False,
                    allow_unicode=True,
                    width=float("inf"),
                )
            os.replace(tmp_path, review_path)
        except Exception:
            with contextlib.suppress(OSError):
                os.unlink(tmp_path)
            raise
        return len(decided)

    class EvolveBody(BaseModel):
        phase: str

    @app.post("/evolve")
    def evolve(body: EvolveBody):
        prod = _require_production_dir()
        if body.phase not in EVOLVABLE_PHASES:
            raise HTTPException(
                status_code=400,
                detail=f"Phase '{body.phase}' does not support evolution. Must be one of: {sorted(EVOLVABLE_PHASES)}",
            )
        from white_composition.pipeline_runner import PHASE_REVIEW_FILES

        review_path = prod / (
            PHASE_REVIEW_FILES.get(body.phase) or f"{body.phase}/review.yml"
        )
        decided, saved_midi = _snapshot_decided(review_path)

        import secrets

        module = _EVOLVE_PIPELINE[body.phase]
        cmd = [
            sys.executable,
            "-m",
            module,
            "--production-dir",
            str(prod),
            "--evolve",
        ]
        # Unique 6-hex token injected as EVOLVE_RUN_ID so evolved candidate IDs
        # (e.g. evolved_a3f2b1_drum_verse_01) never collide with those from a
        # previous run — preventing _merge_decided from filtering out new evolved
        # candidates because their IDs matched older decided ones.
        evolve_env = {**os.environ, "EVOLVE_RUN_ID": secrets.token_hex(3)}
        result = subprocess.run(cmd, capture_output=True, text=True, env=evolve_env)
        if result.returncode != 0:
            raise HTTPException(
                status_code=500,
                detail=result.stderr.strip()
                or f"Evolution failed for phase '{body.phase}'",
            )
        restored = _merge_decided(review_path, decided, saved_midi)
        candidates_dir = prod / body.phase / "candidates"
        evolved_count = (
            len(list(candidates_dir.glob("evolved_*.mid")))
            if candidates_dir.exists()
            else 0
        )
        return {"ok": True, "evolved_count": evolved_count, "restored_count": restored}

    # ------------------------------------------------------------------
    # Generate (agent workflow + shrinkwrap)
    # ------------------------------------------------------------------

    @app.post("/generate")
    def start_generate():
        global _generate_job
        if _generate_job["status"] == "running":
            raise HTTPException(
                status_code=409, detail="A generate job is already running"
            )
        now = datetime.now(timezone.utc).isoformat()
        _generate_job = {
            "status": "running",
            "started_at": now,
            "finished_at": None,
            "error": None,
        }

        def _run():
            global _generate_job
            try:
                from white_composition.shrinkwrap_chain_artifacts import shrinkwrap
                from white_ideation.agents.workflow.concept_workflow import (
                    run_white_agent_workflow,
                )

                run_white_agent_workflow()
                if _shrink_wrapped_dir is not None:
                    artifacts_dir = Path("chain_artifacts")
                    shrinkwrap(artifacts_dir, _shrink_wrapped_dir, scaffold=True)
                _generate_job["status"] = "done"
            except Exception as exc:
                _generate_job["status"] = "error"
                _generate_job["error"] = str(exc)
            finally:
                _generate_job["finished_at"] = datetime.now(timezone.utc).isoformat()

        threading.Thread(target=_run, daemon=True).start()
        return {"status": "running", "started_at": now}

    @app.get("/generate/status")
    def generate_status():
        return _generate_job

    return app


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Candidate browser API server")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--production-dir", type=Path)
    mode.add_argument("--shrink-wrapped-dir", type=Path)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--no-open", action="store_true")
    args = parser.parse_args()

    if args.production_dir:
        production_dir = args.production_dir.resolve()
        if not production_dir.exists():
            print(f"ERROR: {production_dir} does not exist", file=sys.stderr)
            sys.exit(1)
        app = create_app(production_dir)
        label = f"Serving candidates from: {production_dir}"
        open_path = "/candidates"
    else:
        shrink_wrapped_dir = args.shrink_wrapped_dir.resolve()
        if not shrink_wrapped_dir.exists():
            print(f"ERROR: {shrink_wrapped_dir} does not exist", file=sys.stderr)
            sys.exit(1)
        app = create_app(shrink_wrapped_dir=shrink_wrapped_dir)
        label = f"Serving album from: {shrink_wrapped_dir}"
        open_path = "/"

    if not args.no_open:
        import threading

        def _open():
            import time

            time.sleep(1.0)
            webbrowser.open(f"http://localhost:{args.port}{open_path}")

        threading.Thread(target=_open, daemon=True).start()

    print(label)
    print(f"API: http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
