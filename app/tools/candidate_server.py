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

from app.tools.candidate_browser import (
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


# ---------------------------------------------------------------------------
# Song scanning
# ---------------------------------------------------------------------------


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
        songs.append(
            {
                "id": f"{thread_slug}__{production_slug}",
                "thread_slug": thread_slug,
                "production_slug": production_slug,
                "production_path": str(manifest_path.parent),
                "title": data.get("title") or production_slug,
                "key": data.get("key"),
                "bpm": data.get("bpm"),
                "rainbow_color": data.get("rainbow_color"),
                "singer": data.get("singer"),
                "has_decisions": (
                    manifest_path.parent / "production_decisions.yml"
                ).exists(),
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

        from app.generators.midi.production.pipeline_runner import cmd_promote

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
                from app.agents.workflow.concept_workflow import (
                    run_white_agent_workflow,
                )
                from app.util.shrinkwrap_chain_artifacts import shrinkwrap

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
