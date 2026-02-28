## 1. Update vocal section reading

- [x] 1.1 Replaced `_read_vocal_sections()` with `read_vocal_sections_from_arrangement()`
  - One entry per unique arrangement clip label (e.g. `melody_verse_alternate`,
    `melody_verse_alternate_2`, `melody_bridge`)
  - `name` field = clip label (used as [header] in lyrics)
  - Contour looked up from `melody/review.yml` approved candidates
  - Note count from `melody/approved/<label>.mid`, defaults to 0

## 2. Update prompt builder

- [x] 2.1 `_build_prompt()` uses `sec["name"]` (the loop label) as the `[header]`
  — since `name` == `approved_label` == clip name from arrangement.txt

## 3. Update section parser and fitting

- [x] 3.1 `_parse_sections()` already parses any `[header]` — no change needed
- [x] 3.2 `_compute_fitting()` maps parsed block names to vocal section list by
  `name` key (which is now the loop label, not a generic section name)

## 4. Update review YAML schema

- [x] 4.1 `_load_or_init_review()` no longer stores `vocal_sections` in the header
  (sections are derived from arrangement.txt at runtime, not stored in review)

## 5. Tests

- [x] 5.1 `TestBuildPrompt::test_has_section_headers` — verifies `[verse]`, `[chorus]`
  headers appear (loop labels as headers)
- [x] 5.2 `TestLoadOrInitReview` — updated to use `make_meta()` dict,
  removed `vocal_sections` assertion
- [x] 5.3 `TestComputeFitting` — unchanged; uses `name` key for lookup (still correct)
- [x] 5.4 Integration tests — `_make_production_dir()` creates arrangement.txt with
  a `verse` clip on track 4; generated lyrics use `[verse]` as the loop label header
