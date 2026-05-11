# Future Ideas

Captured here when there's no time to write a full spec. Each entry has enough
context to start a spec session without losing the thought.

---

## Multi-Song Composition Board

**What:** The board currently shows one song at a time (whichever is active).
A true kanban would show all handed-off songs as cards across the nine mix
stage columns — visible without switching active songs.

**Why:** With multiple songs in progress, you want to see the full picture at
a glance. "What's at rough_mix? What's been sitting at recording for two weeks?"

**Key design question:** `GET /composition` currently relies on the active song.
Need either:
- `GET /composition/<song_id>` — pass the logic_project_path from the songs index
- Or scan `LOGIC_OUTPUT_DIR` for all `composition.yml` files on page load

**Shape:** Board page fetches all songs, then fans out `fetchComposition` per
song (or a new `GET /compositions` endpoint that returns all). Each song gets
one card per stage column. Clicking a card activates that song and opens
a detail panel.

---

## Re-Handoff as Explicit MIDI Sync

**What:** A "Sync MIDI" action separate from full handoff. Only copies
approved MIDI files — no seed copy, no composition.yml creation.

**Why:** After evolving and re-promoting a phase (e.g. new melody candidates),
you want to push the updated MIDI to the Logic folder without re-running the
full handoff scaffolding.

**Shape:** `POST /handoff/sync` — runs only step 2 of `logic_handoff.handoff()`
(MIDI copy loop). Returns paths of files copied. Board button: "Sync MIDI",
appears whenever a song has been handed off (composition.yml exists).

---

## Production Plan Rework

**What:** `production_plan.yml` currently just lists sections in approval order
with `repeat: 1` — it's not a real compositional proposal, just a log.

**Why:** The drift report (see `plan-drift-report` change) needs a genuine
Claude-authored arrangement arc to diff against. Without that, the drift report
is just "what got cut" with no creative context.

**Shape:** `production_plan.py` calls Claude with:
- Full song proposal (concept, mood, color, singer, key, BPM)
- Approved loop inventory (sections, bar lengths, energy scores from phase reviews)

Claude proposes a real arrangement arc — intro/outro decisions, repeat strategy,
dynamic arc, rationale. Output adds `rationale` field and `proposed_by: claude`.

**Blocker for:** `plan-drift-report` (which diffs this plan against arrangement.txt)

---

## Drift Report

See `openspec/changes/plan-drift-report/proposal.md` for full design.

Short version: CLI tool that reads `production_plan.yml` + Logic-exported
`arrangement.txt` and writes `drift_report.yml` — section removals, bar deltas,
energy arc correlation, Claude-generated prose summary. Closes the feedback loop
between what Claude proposed and what the human actually built.

---

## Aggregated Drift Signal → Generation Tuning

**What:** After accumulating drift reports across N songs, aggregate the patterns
and use them to adjust the production plan prompt. "Claude proposes intros 80%
of the time; they get removed 90% of the time → stop proposing intros."

**Why:** Makes the system genuinely learn from human production decisions over time.

**Maturity:** Far future. Needs: drift reports on ≥5 songs, an aggregation
script, and a decision about where the tuning signal lives (prompt injection vs.
fine-tuning vs. a constraint file).

---

## Version Notes Auto-Summary

**What:** When the user adds a version on the board, optionally prompt Claude
for a one-line suggested note based on what changed (stage advancement, MIDI
sync, time elapsed since last version).

**Why:** Low friction — the user can accept, edit, or ignore. Makes the version
history actually readable six months later.

**Shape:** Small modal after "+ Version" click: "Suggested note: [rough strings
added, moved to augmentation]. Accept / Edit / Skip."

---

## Board Keyboard Shortcuts

**What:** Stage advancement and version creation via keyboard on the board.

**Rationale:** The candidate browser already has `a`/`r`/`p` shortcuts. The
board is keyboard-hostile right now for power users.

**Candidates:** `→` advance stage, `v` add version, `1`/`2`/`3` open lyric
candidates, `p` promote focused lyric, `Esc` close modal.
