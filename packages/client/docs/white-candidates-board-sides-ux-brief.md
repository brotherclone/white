# White — /candidates, /board, /sides: UX Kruft Brief

**Scope:** usability only, same terms as the first brief. Note: the Stage/Thread/Placement combobox + query-param fix from that brief is live on the home page and works well — this brief covers three more views, and flags where that same fix needs to propagate.

---

## /candidates — per-song generation view

1. **Breadcrumb mixes clickable and static segments with no visual distinction.** `← Songs / [Song Title] BPM / Sides` — "Songs" and "Sides" are links, the title and BPM are static text, but nothing (color, underline, weight) tells you which is which without hovering.
2. **Pipeline stepper is plain text, not a real stepper.** `→ chords · drums · bass · melody` conveys current step with a leading arrow and dimming, but has none of the clarity of the Board's lifecycle columns (see below) — worth reusing that visual language here: check / current / upcoming.
3. **9-column table (Phase/Section/ID/Template/Score/Status/Label/Use/Actions) untested at width** — I only saw the empty state ("No candidates found") for every song I opened. Worth a pass once populated to confirm it doesn't force the same undiscoverable horizontal scroll the Board has.
4. Keyboard-shortcut hint row (`a` approve / `f` reject / `p` play-stop) shows even with an empty table and nothing focused — cheap to dim/hide until there's a row to act on.

## /board — composition board

1. **Song picker is a single flat native `<select>` with ~97 options, unsorted by song, with duplicate labels.** Multiple options share an identical visible label — e.g. two separate entries both titled "Form B-Minor: The Stamp That Notarizes the Hand (Black)", three variants of "32.4°C (The Temperature at Which Everything Dies)" — distinguishable only by an invisible internal slug in the option value. This is worse than the old homepage chip row: you cannot reliably pick the right one by reading the label. Needs the same combobox treatment as Stage/Thread — grouped by song → thread/version, with the candidate id shown as dimmed secondary text so duplicates are actually distinguishable.
2. **Horizontal scroll on the lifecycle board is undiscoverable.** Nine columns exist — Structure, Lyrics, Vocal Placeholders, Recording, Augmentation, Cleaning, Rough Mix, Mix Candidate, Final Mix — only ~6 fit in view at a normal desktop width. The only hint is a clipped edge column and a barely-visible scrollbar sliver at the very bottom; no fade/gradient edge, no arrow. I didn't know the last three columns existed until I scrolled blind.
3. **The column status language is the best in the app — reuse it elsewhere.** Green check = complete, blue outline + dot = current stage, dim gray = upcoming. This is clearer than anything on /candidates or the home page and should be the pattern the pipeline stepper (above) borrows.
4. **Column checkmark can contradict the card inside it.** "Recording" shows a green ✓ at the column header, but the one card inside it is labeled "draft" — reads as contradictory unless you already know the checkmark tracks "phase reached" rather than "content finalized."
5. **Top audio player shows 0:00 / 0:00 next to a column that clearly has a draft take attached** (Graham Hopkins, draft, "view work order"). Unclear if the player is supposed to reflect the selected column's asset or a separate master — as-is it reads as broken.
6. Song title is shown twice in the header — once as a small gray label, once again as the selected value inside the picker immediately to its right. Redundant.

## /sides — LP side sequencing

1. **Same flat-list-no-search problem, one more place.** The "Available" pool on the left is a plain scroll of ~84 song rows, no search/filter, even though a chunk are permanently disabled with a trailing "no mix" label. This is the same problem the home page just fixed with the Stage/Thread combobox — worth applying here too, or at minimum, hide "no mix" entries behind a toggle instead of showing them inline-dimmed forever.
2. Per-side time budget ("5:42 / 20:00") is clear and legible — no note, this is good as-is.
3. No running total across all four sides. If the goal is sequencing a full LP, a top-line total runtime (used vs. remaining pool duration) would save doing the arithmetic by eye.
4. "Drop a mixed song here" empty-state copy on Side D is a good, specific instruction — pairs well with the disabled "no mix" state in the pool. Keep this pattern.

---

**Net:** the fix already shipped for Stage/Thread/Placement needs to propagate two more places — the /board song picker and the /sides pool list — same root cause (long flat list, no search, in the picker's case actively ambiguous due to duplicate labels). Going the other direction, the /board lifecycle columns have the clearest status-communication pattern in the app and should be what /candidates' pipeline stepper is upgraded to match.
