# violet-agent Specification

## Purpose
TBD - created by archiving change enhance-gabe-corpus-utilization. Update Purpose after archive.
## Requirements
### Requirement: RAG-Based Corpus Retrieval
The Violet Agent SHALL use embedding-based retrieval to select relevant corpus passages for each interview question.

#### Scenario: Question-aware retrieval
- **WHEN** simulated_interview generates a response to an interview question
- **THEN** the system SHALL embed the question text
- **AND** retrieve top-k (3-5) most semantically relevant corpus chunks
- **AND** inject retrieved chunks into the prompt context

#### Scenario: Embedding initialization
- **WHEN** VioletAgent is instantiated
- **THEN** the system SHALL load or generate corpus embeddings
- **AND** embeddings SHALL be cached to avoid regeneration on each init

#### Scenario: Graceful degradation
- **WHEN** embedding model is unavailable or corpus is malformed
- **THEN** the system SHALL fall back to truncated corpus (current behavior)
- **AND** log a warning about degraded retrieval

### Requirement: Psychological Depth in Voice Simulation
The Violet Agent SHALL incorporate psychological patterns from the corpus that balance confidence with vulnerability.

#### Scenario: Vulnerability surfacing
- **WHEN** generating simulated Gabe responses
- **THEN** the prompt SHALL include psychological grounding from corpus Part 3 (Jungian) and Part 4 (Personality)
- **AND** responses MAY include self-doubt, guilt acknowledgment, or creative wound references
- **AND** responses SHALL NOT be purely confident/cavalier

#### Scenario: Guilt-as-fuel pattern
- **WHEN** questions touch on creative motivation or artistic drive
- **THEN** relevant passages about guilt-as-creative-fuel SHALL be retrievable
- **AND** the "trauma transformed into methodology" pattern SHALL inform response generation

#### Scenario: INTP-with-feeling voice
- **WHEN** generating any simulated response
- **THEN** the voice SHALL reflect the documented pattern: intellectual precision + developed feeling function
- **AND** responses SHALL avoid pure cold analysis OR pure emotional expression

### Requirement: Interviewer Persona Briefing
The Violet Agent SHALL provide the interviewer persona with context about Gabe's documented patterns.

#### Scenario: Style-aware question generation
- **WHEN** generate_questions creates interview questions
- **THEN** the prompt SHALL include Gabe's documented aesthetic sensibilities (Part 5)
- **AND** the prompt SHALL include communication patterns (Part 6)
- **AND** generated questions MAY probe documented vulnerabilities or creative wounds

#### Scenario: Cultural context awareness
- **WHEN** briefing the interviewer persona
- **THEN** the system SHALL include Gabe's formative cultural touchstones (1975-1992)
- **AND** the interviewer SHALL have context about the "less friendly Daniel Johnston" aesthetic

### Requirement: Voice Baseline Card
The Violet Agent SHALL maintain a minimal voice card for consistent baseline guidance.

#### Scenario: Voice card presence
- **WHEN** building any prompt that simulates Gabe's voice
- **THEN** the system SHALL include a condensed voice card (~300 tokens)
- **AND** the voice card SHALL contain: bracket-switching pattern, key phrases, psychological tone notes

#### Scenario: Voice card content
- **WHEN** the voice card is constructed
- **THEN** it SHALL include the academic→profane pattern with examples
- **AND** it SHALL include self-deprecating acknowledgment of pretension
- **AND** it SHALL note the vulnerability beneath the intellectual confidence

