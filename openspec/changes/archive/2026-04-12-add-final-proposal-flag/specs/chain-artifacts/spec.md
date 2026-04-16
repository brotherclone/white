## MODIFIED Requirements

### Requirement: Constraint-Aware Proposal Generation
The system SHALL incorporate negative constraints when generating new song proposals to
increase output diversity. Each chromatic agent SHALL produce exactly one final
`SongProposalIteration` marked `is_final=True`, which is the proposal surfaced for
human review and production. Internal intermediate iterations (EVP updates, reaction
book revisions, game run counter-proposals) are retained in state and in the
`all_song_proposals` bundle but are NOT written as standalone files.

#### Scenario: Constraint loading at workflow start
- **WHEN** a new chain workflow starts and `shrink_wrapped/index.yml` exists
- **THEN** the constraints are loaded and made available to the White agent

#### Scenario: Constraint influence on proposals
- **WHEN** a new proposal is generated
- **THEN** the White agent MUST NOT produce a proposal matching that constraint

#### Scenario: Constraint influence logging
- **WHEN** a new proposal is generated
- **THEN** the system logs which constraints influenced the output

---

## ADDED Requirements

### Requirement: Final Proposal Flag
Each `SongProposalIteration` SHALL carry an `is_final` boolean field. Exactly one
iteration per chromatic agent run SHALL be marked `is_final=True` — the last resolved
proposal after all internal creative steps complete. Agents that produce a single
iteration treat it as implicitly final.

#### Scenario: Multi-iteration agent marks final
- **WHEN** Black, Red, or Yellow completes its internal workflow
- **THEN** exactly one `SongProposalIteration` in the run has `is_final=True`

#### Scenario: Single-iteration agent is implicitly final
- **WHEN** an agent produces exactly one proposal and none are marked `is_final`
- **THEN** `save_all_proposals` treats that single iteration as final

### Requirement: Selective Proposal File Output
`save_all_proposals` SHALL write standalone `song_proposal_<Color>_<id>.yml` files
only for iterations where `is_final=True`. All iterations SHALL continue to appear in
the `all_song_proposals_<thread>.yml` bundle for traceability.

#### Scenario: Only final proposals produce standalone files
- **WHEN** a thread has multiple iterations for a color (e.g. Black with EVP update)
- **THEN** only the `is_final=True` iteration is written as
  `song_proposal_Black_<id>.yml`; intermediate iterations appear only in
  `all_song_proposals_<thread>.yml`

#### Scenario: Full traceability preserved
- **WHEN** `all_song_proposals_<thread>.yml` is read
- **THEN** all iterations including non-final ones are present with their
  `is_final` flags

### Requirement: Opt-In HTML Artifact Generation
HTML artifact generation (character sheets, timeline pages, and other fiction rendering) SHALL be opt-in, controlled by a `--with-html` flag on `run_white_agent start`. When the flag is absent, agents that produce HTML SHALL skip that generation step entirely. HTML generation SHALL NOT run by default as it adds LLM and image generation cost with no current consumer. The capability is preserved for future UI integration (v2: candidate browser renders song-related fiction alongside MIDI review).

#### Scenario: HTML skipped by default
- **WHEN** `run_white_agent start` is invoked without `--with-html`
- **THEN** no HTML files are written to `chain_artifacts` and no related LLM or
  image generation calls are made

#### Scenario: HTML generated on request
- **WHEN** `run_white_agent start --with-html` is invoked
- **THEN** HTML artifacts (character sheets, timeline pages, etc.) are generated
  and written to `chain_artifacts/<thread>/html/` as before

#### Scenario: Shrinkwrap handles missing html directory
- **WHEN** a thread has no `html/` directory (because `--with-html` was not used)
- **THEN** shrinkwrap completes successfully and omits the `html/` directory from
  the output without error
