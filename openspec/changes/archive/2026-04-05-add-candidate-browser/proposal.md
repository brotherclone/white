# Change: Add candidate browser for reviewing and approving pipeline output

## Why
Approving candidates currently means opening `review.yml` in a text editor and changing
`label: candidate` to `label: approved` by hand. There is no way to audition MIDI
candidates in context without manually importing into Logic. A lightweight browser
dramatically reduces review friction.

## What Changes
- New CLI tool: `app/tools/candidate_browser.py` — terminal UI (using `rich`) that lists
  all pending candidates across phases for a song, shows scores, and lets the user
  approve/reject with keystrokes
- Candidates are displayed grouped by phase (chords → drums → bass → melody → quartet)
  then by section
- Selecting a candidate opens it in the system default MIDI player (via `open` on macOS)
- Approving writes the label change directly to `review.yml`
- No new dependencies beyond `rich` (already used in the project)

## Impact
- Affected specs: candidate-browser (new capability)
- Affected code: new `app/tools/candidate_browser.py`
- Not breaking — `review.yml` format unchanged
