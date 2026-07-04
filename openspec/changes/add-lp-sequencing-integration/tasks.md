## 1. Song status flag
- [x] 1.1 Add `lp_consideration` handling to the manifest patch helper (`_patch_manifest`
      in `candidate_server.py`), mirroring the existing `lifecycle_status` handling
- [x] 1.2 `POST /songs/{id}/lp-consideration` endpoint accepting
      `{status: "not_considered" | "candidate" | "placed"}`
- [x] 1.3 Hook `placed`/`candidate` auto-transitions into the `add-lp-side-sequencing`
      assign/move/remove endpoints (assign/move → `placed`, remove → `candidate` if a
      mix still exists, else `not_considered`)
- [x] 1.4 Add `lp_consideration` to `scan_songs()` output
- [x] 1.5 Filter pill (`lp: candidate` / `lp: placed`) and badge in the client song list

## 2. Navigation
- [x] 2.1 Add a "Sides" link — there's no single shared nav component in this client
      (each page has its own ad hoc header), so the link was added to the header of
      every page that already cross-links to others: home (`/`), board, and candidates

## 3. Claude sequencing advisor
- [x] 3.1 `lp_sequence_advisor.py`: load `sides.yml` + each placed song's
      `rainbow_color`/mood/BPM (from `manifest_bootstrap.yml`) and cached duration;
      compute per-side color clustering and BPM/energy spread
- [x] 3.2 CLI flags: `--album-dir` (default `$SHRINKWRAP_OUTPUT_DIR`), `--output report.yml`,
      `--dry-run` (print only, matching the `generate_negative_constraints.py` convention)
- [x] 3.3 Report includes plain-language suggestions (e.g. "Side B is all Blue/Indigo —
      consider moving a warmer-color song in") without modifying `sides.yml`
- [x] 3.4 Unit tests for the report generation given a fixture `sides.yml` + fixture
      manifests
