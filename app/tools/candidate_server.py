#!/usr/bin/env python3
"""FastAPI backend for the web-based candidate browser.

Serves candidates as JSON, handles approve/reject writes, and streams MIDI files.
The Next.js frontend at localhost:3000 consumes this API.

Usage:
    python -m app.tools.candidate_server --production-dir shrink_wrapped/<album>/production/<slug>
    python -m app.tools.candidate_server --production-dir ... --port 8000 --no-open
"""

import argparse
import sys
import webbrowser
from dataclasses import asdict
from pathlib import Path

import uvicorn
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

# ---------------------------------------------------------------------------
# App factory (production_dir injected at startup)
# ---------------------------------------------------------------------------

_production_dir: Path | None = None


def create_app(production_dir: Path) -> FastAPI:
    global _production_dir
    _production_dir = production_dir

    app = FastAPI(title="Candidate Browser API")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _all_candidates(
        phase: str | None = None, section: str | None = None
    ) -> list[CandidateEntry]:
        return load_all_candidates(
            _production_dir, phase_filter=phase, section_filter=section
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
    # Routes
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
        return {"production_dir": str(_production_dir)}

    return app


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Candidate browser API server")
    parser.add_argument("--production-dir", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind to")
    parser.add_argument(
        "--no-open", action="store_true", help="Don't open browser on start"
    )
    args = parser.parse_args()

    production_dir = args.production_dir.resolve()
    if not production_dir.exists():
        print(f"ERROR: {production_dir} does not exist", file=sys.stderr)
        sys.exit(1)

    app = create_app(production_dir)

    if not args.no_open:
        import threading

        def _open():
            import time

            time.sleep(1.0)
            webbrowser.open(f"http://localhost:{args.port}")

        threading.Thread(target=_open, daemon=True).start()

    print(f"Serving candidates from: {production_dir}")
    print(f"API: http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
