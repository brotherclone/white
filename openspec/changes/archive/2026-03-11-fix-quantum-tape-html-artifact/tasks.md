## 1. Fix artifact_name derivation
- [x] 1.1 In `QuantumTapeLabelArtifact.__init__`, set `artifact_name` from `title` when not
      already set (same guard pattern as `AlternateTimelineArtifact`)

## 2. Add missing template fields to model
- [x] 2.1 Add `year_documented: Optional[str]` — the year the tape was archived (e.g. "2003")
- [x] 2.2 Add `original_date: Optional[str]` — A-side real-timeline label date
- [x] 2.3 Add `original_title: Optional[str]` — A-side real-timeline label text
- [x] 2.4 Add `tapeover_date: Optional[str]` — B-side alternate-timeline period date range
- [x] 2.5 Add `tapeover_title: Optional[str]` — B-side alternate-timeline title
- [x] 2.6 Add `subject_name: Optional[str]` — subject name (e.g. "Gabe Walsh")
- [x] 2.7 Add `age_during: Optional[str]` — subject age range during the period (e.g. "22-24")
- [x] 2.8 Add `location: Optional[str]` — geographic location during the period
- [x] 2.9 Add `catalog_number: Optional[str]` — tape catalog identifier

## 3. Update flatten()
- [x] 3.1 Include all nine new fields in `flatten()` return dict

## 4. Update blue_agent generate_tape_label node
- [x] 4.1 Derive and pass `year_documented` from `alternate.period.start_date.year`
- [x] 4.2 Derive and pass `original_date` from `alternate.period.start_date.year`
- [x] 4.3 Derive and pass `original_title` from `original_label_text`
- [x] 4.4 Derive and pass `tapeover_date` as formatted period range
- [x] 4.5 Derive and pass `tapeover_title` from `alternate.title`
- [x] 4.6 Derive and pass `subject_name = "Gabe Walsh"` (constant for blue)
- [x] 4.7 Derive and pass `age_during` from `alternate.period.age_range`
- [x] 4.8 Derive and pass `location` from `alternate.period.location`, fallback `"Unknown"`
- [x] 4.9 Derive and pass `catalog_number` as `f"QT-B-{year}-{thread_id[:6].upper()}"`

## 5. Update mock and tests
- [x] 5.1 Add all nine new fields to `tests/mocks/quantum_tape_label_mock.yml`
- [x] 5.2 Add test: `QuantumTapeLabelArtifact` with `title` set produces a `file_name`
      that does NOT contain `UNKNOWN`
- [x] 5.3 Add test: `save_file()` renders HTML where none of the nine template slots are
      empty
- [x] 5.4 Add test: `flatten()` includes all nine new fields
- [x] 5.5 Verify all existing tests still pass (16/16)
