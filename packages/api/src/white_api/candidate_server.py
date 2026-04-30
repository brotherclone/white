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
import subprocess
import sys
import threading
import webbrowser
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import uvicorn
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from white_api.candidate_browser import (
    CandidateEntry,
    approve_candidate,
    load_all_candidates,
    reject_candidate,
    set_label,
    set_use_case,
)

VALID_PHASES = {"chords", "drums", "bass", "melody", "lyrics", "quartet"}
EVOLVABLE_PHASES = {"drums", "bass", "melody"}

_EVOLVE_PIPELINE = {
    "drums": "white_generation.pipelines.drum_pipeline",
    "bass": "white_generation.pipelines.bass_pipeline",
    "melody": "white_generation.pipelines.melody_pipeline",
}

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
        songs.append(
            {
                "id": f"{thread_slug}__{production_slug}",
                "thread_slug": thread_slug,
                "production_slug": production_slug,
                "production_path": str(prod_dir),
                "title": data.get("title") or production_slug,
                "key": _coerce_key_string(data.get("key")),
                "bpm": data.get("bpm"),
                "rainbow_color": data.get("rainbow_color"),
                "singer": data.get("singer"),
                "has_decisions": (prod_dir / "production_decisions.yml").exists(),
                "initialized": (prod_dir / "song_context.yml").exists(),
                "proposal_path": str(proposal_path) if proposal_path else None,
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
) -> FastAPI:
    global _production_dir, _shrink_wrapped_dir, _active_song
    _production_dir = production_dir
    _shrink_wrapped_dir = shrink_wrapped_dir
    _active_song = None

    app = FastAPI(title="Candidate Browser API")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
        allow_methods=["GET", "POST", "PATCH", "DELETE"],
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
        return {"active": _active_song}

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
        except SystemExit as exc:
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
            global _handoff_job
            try:
                from white_composition.logic_handoff import handoff

                handoff(prod)
                _handoff_job["status"] = "done"
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

    @app.get("/composition")
    def get_composition():
        prod = _require_production_dir()
        from white_composition.logic_handoff import read_composition, resolve_song_dir

        song_dir = resolve_song_dir(prod)
        data = read_composition(song_dir)
        if data is None:
            return {"status": "not_initialized"}
        return data

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
        review_path = prod / body.phase / "review.yml"
        if not review_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"No review.yml found for phase '{body.phase}'",
            )
        approved_dir = prod / body.phase / "approved"
        count_before = (
            len(list(approved_dir.glob("*.mid"))) if approved_dir.exists() else 0
        )

        from white_composition.pipeline_runner import cmd_promote

        result = cmd_promote(prod, body.phase, yes=True)
        if result != 0:
            raise HTTPException(
                status_code=409,
                detail=f"Promotion could not be completed for phase '{body.phase}' — ensure review.yml contains approved candidates",
            )
        count_after = (
            len(list(approved_dir.glob("*.mid"))) if approved_dir.exists() else 0
        )
        return {"ok": True, "promoted_count": max(0, count_after - count_before)}

    # ------------------------------------------------------------------
    # Evolve
    # ------------------------------------------------------------------

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
        module = _EVOLVE_PIPELINE[body.phase]
        cmd = [
            sys.executable,
            "-m",
            module,
            "--production-dir",
            str(prod),
            "--evolve",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise HTTPException(
                status_code=500,
                detail=result.stderr.strip()
                or f"Evolution failed for phase '{body.phase}'",
            )
        candidates_dir = prod / body.phase / "candidates"
        evolved_count = (
            len(list(candidates_dir.glob("evolved_*.mid")))
            if candidates_dir.exists()
            else 0
        )
        return {"ok": True, "evolved_count": evolved_count}

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
