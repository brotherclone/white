## 1. Core Registration Function
- [x] 1.1 Add `register_part(midi_path, phase, section, label, production_dir)` to `promote_part.py`
        — copies MIDI to `<phase>/approved/<label>.mid`, writes/updates review.yml entry with
        `generated: false`, `status: approved`, `scores: null`, `rank: null`
- [x] 1.2 Validate MIDI is readable via mido before registering; raise ValueError on corrupt file
- [x] 1.3 Prevent duplicate label registration (error if label already exists as approved)

## 2. API Endpoint
- [x] 2.1 Add `POST /production/register-part` to `candidate_server.py`
        — accepts `phase`, `section`, `label`, and MIDI file upload (uses active production dir)
- [x] 2.2 Return the registered `CandidateEntry` as JSON response

## 3. Candidate Browser Display
- [x] 3.1 Add `generated: bool = True` field to `CandidateEntry` dataclass
- [x] 3.2 Read `generated` flag from review.yml in `_load_review()`
- [x] 3.3 Display a `[H]` marker (hand-written) instead of score columns for non-gen entries

## 4. Tests
- [x] 4.1 Unit test `register_part()`: happy path, duplicate label error, corrupt MIDI error
- [x] 4.2 Unit test `_load_review()` tolerates `generated: false` entries with null scores/rank
