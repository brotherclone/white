## 1. Data Collection
- [ ] 1.1 Write `scan_album(album_dir) → list[SongStatus]` — walks `shrink_wrapped/` for
       song directories, reads each song's `review.yml` files and `production_plan.yml`
- [ ] 1.2 Define `SongStatus` dataclass: slug, color, singer, key, phase_statuses dict,
       total_approved_bars, plan_present
- [ ] 1.3 Write `phase_status(review_yml_path) → str` — returns `approved`, `pending`,
       `no_candidates`, or `not_started`

## 2. Display
- [ ] 2.1 Build `rich.table` with columns: song, color, singer, chords, drums, bass,
       melody, quartet, bars, plan
- [ ] 2.2 Color-code status cells: green=approved, yellow=pending, red=not_started
- [ ] 2.3 Sort rows by color (rainbow order) then by song slug

## 3. CLI Entry Point
- [ ] 3.1 `python -m app.tools.song_dashboard --album-dir <path>`
- [ ] 3.2 `--color <name>` filter flag
- [ ] 3.3 `--phase <name>` flag: show only songs where that phase is not yet approved

## 4. Tests
- [ ] 4.1 Unit tests for `scan_album` and `phase_status` with fixture directories
