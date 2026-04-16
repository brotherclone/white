## 1. Core function

- [ ] 1.1 Create `app/generators/midi/production/production_decisions.py` with
      `generate_decisions(production_dir: Path) -> dict` that reads:
      - `song_context.yml` — identity fields
      - `chords/review.yml`, `drums/review.yml`, `bass/review.yml`,
        `melody/review.yml` — per-phase candidate and approval counts, scores
      - `arrangement.txt` — section structure (parse via existing `parse_arrangement`)
      - `melody/mix_score.yml` — mix scores (if present)
      - `drift_report.yml` — vocal drift (if present)
- [ ] 1.2 Write `write_decisions_file(production_dir, decisions) -> Path` that saves
      `production_decisions.yml` to the production directory root

## 2. CLI

- [ ] 2.1 Add `--decisions` flag to the existing drift_report CLI, OR add a standalone
      `__main__` block to `production_decisions.py`:
      `python -m app.generators.midi.production.production_decisions --production-dir <path>`

## 3. Pipeline runner integration

- [ ] 3.1 In `pipeline_runner.py status` output, show whether `production_decisions.yml`
      exists (alongside the existing phase status rows)
- [ ] 3.2 Add `decisions` as a runnable step in the pipeline runner (after `score_mix`)

## 4. Session close slash command

- [ ] 4.1 Update `.claude/commands/white/session-close.md` to include generating
      `production_decisions.yml` as a step after scoring the mix

## 5. Tests

- [ ] 5.1 Unit test `generate_decisions` with a scaffolded production dir containing
      review.yml files — assert all identity fields present, phase decisions populated
- [ ] 5.2 Test graceful partial result when some phases are incomplete (no review.yml)
      and mix_score.yml / drift_report.yml are absent
