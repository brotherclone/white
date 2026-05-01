# Change: Production Plan Drift Report

## Status: IDEA — not yet specced or tasked

This document captures the design thinking for the drift report feature so it
survives context resets. Implementation should begin with a proper spec session.

---

## Why This Matters

The current pipeline ends at Logic handoff. Claude proposes an arrangement
(`production_plan.yml` — bar counts, section order, energy arc), a human then
arranges and mixes in Logic, and we never know how far the final mix diverged
from the proposal. That gap is information:

- If Claude consistently proposes 8-bar choruses and the human always cuts them
  to 4, that's a calibration signal for future generation.
- If the section order gets shuffled every time, the production plan prompt
  needs rethinking.
- If the human follows the plan closely, the AI's compositional judgement is
  validated and can be weighted more heavily next time.

The drift report closes this feedback loop without requiring the human to do
anything beyond what they already do in Logic.

---

## What It Would Do

A CLI tool (and eventually an API endpoint) that:

1. Reads `production_plan.yml` — Claude's proposed arrangement
2. Reads a Logic-exported `arrangement.txt` from the Logic song folder (already
   exported for the lyric pipeline, so this is free)
3. Compares the two and writes `drift_report.yml` alongside `composition.yml`

The report captures:

```yaml
generated: '2026-05-01'
song_title: The Fibonacci Librarian's Infinite Index
proposed_sections: [intro, verse2, chorus, verse2_2, chorus_2, bridge, chorus_3, outro]
actual_sections:   [verse2, chorus, verse2_2, bridge, chorus_2, chorus_3]

drift:
  removed:  [intro, outro]           # in proposal, absent in arrangement.txt
  added:    []                        # in arrangement.txt, absent in proposal
  reordered: false                    # section order preserved (modulo removed)

bar_deltas:
  verse2:   { proposed: 8,  actual: 8,   delta: 0 }
  chorus:   { proposed: 8,  actual: 4,   delta: -4 }
  bridge:   { proposed: 4,  actual: 6,   delta: +2 }

energy_arc_correlation: 0.72          # Pearson r between proposed and actual energy

summary: "2 sections removed (intro, outro), chorus halved, bridge extended.
  Core arc followed. High confidence Claude's energy proposals are usable."
```

---

## Key Design Questions (resolve before speccing)

**1. What is the source of truth for "actual sections"?**
`arrangement.txt` only has clip names and channels. We need to map clip names
→ section names. The mapping could come from:
- Clip naming conventions enforced at the MIDI copy step (e.g. clips are named
  `chorus_01`, `verse2_02`)
- A separate `section_map.yml` the human fills in once per song
- Heuristic: match clip names fuzzy-against known section names

**2. Where does the report live?**
Option A: In the Logic song folder alongside `composition.yml` (lives on the
fast drive, not in the git repo).
Option B: In the production dir (lives in `shrink_wrapped/`, could be
committed).
Lean toward A — it's part of the composition record, not the generation record.

**3. How is it triggered?**
Option A: CLI only — `python -m white_composition.drift_report --production-dir`
Option B: Board button — "Generate Drift Report" appears after the song reaches
`rough_mix` stage, when the arrangement is stable enough to compare.
Option B is better UX but requires a board change and a new API endpoint.

**4. What feeds back into generation?**
The report is useful as a human-readable log right now. In a future phase,
`drift_report.yml` files across multiple songs could be aggregated to tune the
production plan prompt — e.g. "in 8/10 past songs, intros were removed; stop
proposing them." That aggregation is out of scope for the initial spec.

---

## Rough Implementation Shape

```
white_composition/
  drift_report.py
    compare_plans(production_plan, arrangement_txt) -> DriftReport
    write_report(logic_song_dir, report)
    load_report(logic_song_dir) -> DriftReport | None

  DriftReport (Pydantic)
    generated: str
    song_title: str
    proposed_sections: list[str]
    actual_sections: list[str]
    drift: DriftSummary
    bar_deltas: dict[str, BarDelta]
    energy_arc_correlation: float | None
    summary: str  # Claude-generated one-paragraph prose summary
```

The `summary` field calls Claude with the structured drift data and asks for a
one-paragraph English interpretation — same pattern as the production plan
generation step.

---

## Dependencies

- `production_plan.yml` must exist (currently generated but flagged as needing
  rework — see future work note in add-logic-handoff proposal)
- `arrangement.txt` must be present in the Logic song folder (already copied
  there by handoff)
- The section-name mapping question (above) must be resolved
