# Project Context

## Purpose

White is an AI-driven evolutionary music composition system. It generates chord, drum,
bass, and melody MIDI loops scored against a "chromatic" color-coded concept framework
(8 colors: Red, Orange, Yellow, Green, Blue, Indigo, Violet, White), hands them off to
Logic Pro for human arrangement and mixing, and trains fitness models on the resulting
labeled data.

## Tech Stack

- **Python** (UV workspace, 11 packages under `packages/`)
- **Next.js 14** (`packages/client/`) — composition board and candidate browser UI
- **Flask** (`packages/api/`) — generation API and composition board backend
- **ONNX** — runtime scoring via `Refractor` model
- **Modal** — GPU training (DeBERTa, CLAP, multimodal fusion)
- **ACE Studio 2** — vocal synthesis (SVS)
- **Logic Pro** — final arrangement and mixing

## Package Structure

| Package dir | Import namespace | Domain |
|---|---|---|
| `packages/core` | `white_core` | Pydantic structures, enums, shared types |
| `packages/training` | `white_training` | ML training, models, embeddings |
| `packages/extraction` | `white_extraction` | Segment extraction, audio/MIDI utils |
| `packages/analysis` | `white_analysis` | Refractor ONNX scorer, scoring helpers |
| `packages/generation` | `white_generation` | MIDI pattern templates + generation pipelines |
| `packages/composition` | `white_composition` | Production orchestration, Logic handoff, shrinkwrap |
| `packages/ideation` | `white_ideation` | White agent, concept ideation, reference data |
| `packages/api` | `white_api` | Flask candidate server + composition board API |
| `packages/diary` | `white_diary` | Greenfield — MDX song lifecycle |
| `packages/production` | `white_production` | Greenfield — Logic/audio world integration |
| `packages/client` | n/a (JS) | Next.js board and browser UI |

## Project Conventions

### Code Style

- Ruff (`I` rules / isort) enforced by pre-commit — `ruff check --fix` sorts imports
- Three import blocks per file: stdlib → third-party → first-party (`white_*`)
- No imports inside functions except for circular-import breaks (comment required)
- Pydantic models for structured data (API responses, pipeline outputs, YAML round-trips)
- `str, Enum` over raw strings for fixed-set values; enums live in `white_core/enums/`

### Architecture Patterns

- **Pipeline**: chord → drum → bass → melody → promote → production plan → Logic handoff
- **Scoring**: 30% theory + 70% chromatic (Refractor ONNX) for all generation phases
- **Review loop**: each phase writes `review.yml`; human labels → `promote_part` promotes approved MIDI
- **Composition board**: kanban tracking songs from `structure` → `final_mix` (9 mix stages)

### Testing Strategy

- Per-package tests under `packages/<name>/tests/`
- Root `tests/integration/` for cross-package scenarios
- Run: `pytest` from repo root (discovers all packages via root `pytest.ini`)

### Git Workflow

- `feature/* → develop → main`
- Always `--base develop` on PRs

## Domain Context

- **Chromatic framework**: 8 colors each encoding a temporal mode (past/present/future),
  ontological mode (thing/place/person), and spatial mode (known/imagined/forgotten)
- **Fitness models**: trained on 11,605 segments from 83 songs; the `Refractor` ONNX model
  scores MIDI candidates against a target color embedding
- **Lyric pipeline**: Claude drafts lyrics from chromatic synthesis docs; human refines in ACE Studio

## Important Constraints

- The composition pipeline MUST remain functional throughout any migration
- No compatibility shims or re-export stubs; cut over cleanly
- `training/data/` assets (`.pt`, `.onnx`, parquet) stay in place even as source moves
- MIDI channel 9 (0-indexed) is percussion; channel 0 is melody/bass

## External Dependencies

- **HuggingFace**: `earthlyframes/white-training-data` — labeled training segments (public, 15.3 GB)
- **Modal**: GPU compute for training and embedding extraction
- **ACE Studio 2.0** (local): vocal synthesis via MCP integration
- **Logic Pro** (local): final arrangement; controlled via `LOGIC_OUTPUT_DIR` env var
