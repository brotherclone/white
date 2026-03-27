## 1. Last-Note Extraction
- [ ] 1.1 Write `last_note_of_midi(midi_bytes) → int | None` — returns MIDI pitch of last
       note-on event; returns None if no notes found
- [ ] 1.2 Write `first_note_of_candidate(candidate_dict) → int | None` — reads candidate
       MIDI bytes from file path in candidate dict

## 2. Continuity Penalty
- [ ] 2.1 Write `continuity_penalty(first_note, last_note, max_semitones=4) → float`
       — returns 0.85 if abs(first_note - last_note) > max_semitones, else 1.0
- [ ] 2.2 In `melody_pipeline.py` scoring loop: after computing composite score, multiply
       by `continuity_penalty(...)` if a preceding approved section exists

## 3. Configuration
- [ ] 3.1 Read `melodic_continuity_semitones` from `song_proposal.yml` if present
       (default 4); pass into scoring loop

## 4. Tests
- [ ] 4.1 Unit tests for `last_note_of_midi`, `first_note_of_candidate`, `continuity_penalty`
- [ ] 4.2 Integration test: two-section pipeline run; candidate whose first note is
       within range scores higher than one that leaps
