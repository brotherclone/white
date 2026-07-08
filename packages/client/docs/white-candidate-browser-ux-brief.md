# White — Candidate Browser: UX Refinements Brief

**Scope:** usability only. Look-and-feel (palette/type/texture toward the Grouper/BoC/MBV register) is tracked separately against the Earthly Frames style guide abstraction — not part of this pass.

**Target:** the song candidate browser served at `localhost:3000` (nav: Sides / Collaborators / Run Agent; routes `/sides`, `/collaborators`, `/agent`). Locate its source under `/Volumes/LucidNonsense/White/` — likely a `dashboard/`, `web/`, or top-level `app/` directory distinct from the Python pipeline. Confirm App Router vs Pages Router and check `package.json` for existing UI deps (Tailwind, shadcn/ui, cmdk) before adding new ones.

---

## 1. Replace stage-filter chips with a searchable combobox

Current: 12 mutually-exclusive pill buttons in one row — `all / ideation / generation / composition / production / mixing / complete / stub / merged / abandoned / scrapped / invalid`, each with a count, single-select (`all` shown active). This has outgrown chip UI.

Change to a combobox/command-palette control (cmdk-style if shadcn is already present, else a lightweight headless combobox — don't hand-roll a native `<select>` if avoidable, this will be used constantly and deserves type-ahead + arrow-key nav):

- Options list each stage with its live count, e.g. `production (51)`
- Single-select, type-ahead filter on the option list
- Keep `lp: candidate` / `lp: placed` **separate** from this control — it's a different dimension (placement, not pipeline stage), not just overflow. Leave those two as a small toggle group, but label the two groups distinctly ("Stage" / "Placement") so they no longer read as one continuous filter list.

## 2. Sync filter state to the URL

Use `nuqs` if it's a reasonable add (typed searchParams state, handles arrays natively — useful for placement below) — otherwise hand-roll with `useSearchParams` + `router.replace` from `next/navigation`.

- Params: `stage`, `placement`, `thread` (see #3)
- Omit the param entirely at default value (`stage` unset = "all") — keep URLs clean and diffable
- Use `router.replace(..., { scroll: false })`, not `push`, for filter changes — don't let every filter click eat a back-button history entry

## 3. Add a thread filter

The gray subtitle slug under each card title (e.g. `violet-fallback-defensive-violet-response`) appears to be the thread — cross-check against `chain_artifacts/<thread>/` per the pipeline's directory conventions. Confirm the card's slug field maps 1:1 to a `chain_artifacts` directory name before wiring the filter.

- High cardinality, grows over time → same combobox pattern as #1, not chips
- Nice-to-have: let this input fuzzy-match song titles too, so it doubles as the general search the page currently lacks entirely (right now there is no way to jump to a known song by name)

## 4. Restyle "Run Agent"

Currently identical ghost-pill styling to the two nav links beside it (Sides, Collaborators), despite being the one control that actually *does* something (triggers agent/generation work) rather than navigates.

- Solid/filled style in a warm accent, distinct from the neutral nav pills — not red/destructive (it's not a delete), but should read as "action" not "navigation"
- Separate it spatially from the Sides/Collaborators pair (extra margin or its own container) so it's not one fat-finger away from a harmless nav click
- Consider a confirm step if it can fire with no song/thread selected

---

## Notes for implementer

- Card buttons currently render with no accessible name (`button` elements, no text/aria-label) — flagged separately, not required for this brief but cheap to fix alongside #1 if touching this component anyway.
- Don't scope-creep into the visual pass (colors, type, card borders, grain/texture) — that's a separate, already-planned effort.
