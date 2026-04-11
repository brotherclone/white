## 1. Core Data Model
- [x] 1.1 Define `PHASE_ORDER = ["init_production", "chords", "drums", "bass", "melody", "lyrics"]`
- [x] 1.2 Define `PHASE_REVIEW_FILES` mapping each phase to its review.yml path pattern
- [x] 1.3 Define `PHASE_COMMANDS` via `_build_phase_command(phase, production_dir, ctx)`

## 2. Status Reading
- [x] 2.1 `read_phase_statuses(production_dir) → dict[str, str]` — reads phases from song_context.yml
- [x] 2.2 `get_next_runnable_phase(statuses) → str | None` — first pending phase whose predecessor is promoted
- [x] 2.3 Tests: various status combos resolve correct next phase or None

## 3. `pipeline status` subcommand
- [x] 3.1 Print song title + dir header from song_context.yml
- [x] 3.2 Print per-phase status with ASCII icons (✅ promoted, 🔄 generated, ⏳ pending)
- [x] 3.3 Print "Next:" summary with the exact command to run next
- [x] 3.4 Tests: status output contains expected phase names and icons

## 4. `pipeline next` subcommand
- [x] 4.1 Print the next runnable command without executing it
- [x] 4.2 Tests: next output matches expected command for given status

## 5. `pipeline run` subcommand
- [x] 5.1 Find next runnable phase via `get_next_runnable_phase`
- [x] 5.2 Build the full command (subprocess) with `--production-dir` and phase-specific flags
- [x] 5.3 Run command via `subprocess.run`, streaming output
- [x] 5.4 On success: update phase status to `generated` in song_context.yml
- [x] 5.5 Print the promote command after generation
- [x] 5.6 Tests: run with mocked subprocess updates status correctly

## 6. `pipeline promote` subcommand
- [x] 6.1 Read review.yml for the phase, show summary (N candidates, N labelled)
- [x] 6.2 Ask for confirmation (or --yes flag to skip)
- [x] 6.3 Call promote_part logic; on success write `promoted` status to song_context.yml
- [x] 6.4 Tests: promote summary output + status update

## 7. `pipeline batch` subcommand
- [x] 7.1 Enumerate all production dirs in a thread directory
- [x] 7.2 Filter to dirs where the named phase is `pending`
- [x] 7.3 Run generation phase for each filtered dir
- [x] 7.4 Tests: batch correctly enumerates and filters dirs

## 8. promote_part.py — phase status sync
- [x] 8.1 After successful promotion: infer phase name from review.yml path, write `promoted` to song_context.yml
- [x] 8.2 Tests: promote_part updates song_context.yml when it exists

## 9. Tests
- [x] 9.1 Write `tests/generators/midi/production/test_pipeline_runner.py`
