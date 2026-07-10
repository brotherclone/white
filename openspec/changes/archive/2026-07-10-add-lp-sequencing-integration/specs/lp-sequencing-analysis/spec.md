## ADDED Requirements

### Requirement: Sequencing Aesthetic Analysis CLI
The system SHALL provide `lp_sequence_advisor.py`, a read-only CLI script that Claude
(or a human) runs directly to analyze the current `sides.yml` arrangement against
aesthetic goals — chromatic color balance and mood/energy flow — rather than duration
alone.

#### Scenario: Report generated from current sides
- **WHEN** `lp_sequence_advisor.py --album-dir <shrink_wrapped_dir>` is run
- **THEN** it loads `sides.yml` and, for each placed song, reads `rainbow_color`, mood,
  and BPM from `manifest_bootstrap.yml`
- **AND** prints a per-side summary of color distribution and BPM/energy spread

#### Scenario: Suggestions included
- **WHEN** a side's placed songs are heavily concentrated in one or two chromatic colors
  or a narrow BPM band
- **THEN** the report includes a plain-language suggestion (e.g. naming the imbalance
  and a candidate song from another side that could help)

#### Scenario: Read-only guarantee
- **WHEN** `lp_sequence_advisor.py` is run in any mode
- **THEN** `sides.yml` and all `manifest_bootstrap.yml` files are read but never written

#### Scenario: Output modes
- **WHEN** `--output report.yml` is passed
- **THEN** the analysis is written to `report.yml` in addition to being printed
- **WHEN** `--dry-run` is passed instead
- **THEN** only stdout output is produced, matching the convention used by
  `generate_negative_constraints.py --dry-run`

#### Scenario: Empty or unplaced album
- **WHEN** `sides.yml` does not exist or has no placed songs
- **THEN** the tool prints a note that no analysis is possible yet and exits cleanly
  (exit code 0, no error)
