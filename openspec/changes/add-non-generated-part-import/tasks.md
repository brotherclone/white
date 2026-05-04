## 1. Core Registration Function
- [ ] 1.1 Add `register_part(midi_path, phase, section, label, production_dir)` to `promote_part.py`
        — copies MIDI to `<phase>/approved/<label>.mid`, writes/updates review.yml entry with
        `generated: false`, `status: approved`, `scores: null`, `rank: null`
- [ ] 1.2 Validate MIDI is readable via mido before registering; raise ValueError on corrupt file
- [ ] 1.3 Prevent duplicate label registration (error if label already exists as approved)

## 2. API Endpoint
- [ ] 2.1 Add `POST /api/v1/production/register-part` to `candidate_server.py`
        — accepts `production_dir`, `phase`, `section`, `label`, and MIDI file upload
- [ ] 2.2 Return the registered `CandidateEntry` as JSON response

## 3. Candidate Browser Display
- [ ] 3.1 Add `generated: bool = True` field to `CandidateEntry` dataclass
- [ ] 3.2 Read `generated` flag from review.yml in `_load_review()`
- [ ] 3.3 Display a `[H]` marker (hand-written) instead of score columns for non-gen entries

## 4. Tests
- [ ] 4.1 Unit test `register_part()`: happy path, duplicate label error, corrupt MIDI error
- [ ] 4.2 Unit test `_load_review()` tolerates `generated: false` entries with null scores/rank
