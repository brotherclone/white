# ML Model Integration

## ADDED Requirements

### Requirement: Model Loading and Inference
The system SHALL load trained PyTorch models and provide inference APIs.

#### Scenario: Load rebracketing classifier
- **WHEN** initializing model layer
- **THEN** multi-class rebracketing classifier is loaded from `/training/models/`

#### Scenario: Load chromatic style model
- **WHEN** initializing model layer
- **THEN** chromatic style model is loaded with weights

#### Scenario: GPU/CPU device management
- **WHEN** GPU is available
- **THEN** models are loaded to GPU for faster inference

#### Scenario: Model caching
- **WHEN** model is used multiple times
- **THEN** model stays loaded (not reloaded on each inference)

### Requirement: Model Versioning
The system SHALL track model versions and metadata.

#### Scenario: Store model version
- **WHEN** arrangement is generated
- **THEN** model version (e.g., "v1.2") is recorded in arrangement metadata

#### Scenario: Model comparison
- **WHEN** comparing model versions
- **THEN** same arrangement can be scored with different model versions

#### Scenario: Model rollback
- **WHEN** new model performs worse
- **THEN** previous version can be restored

### Requirement: Rebracketing Classifier Inference
The system SHALL use classifier to predict rebracketing types from text+MIDI.

#### Scenario: Extract markers
- **WHEN** lyrics contain "[(is not)]"
- **THEN** marker is extracted with text and position

#### Scenario: Extract MIDI segment
- **WHEN** marker is at bar 6, beat 3
- **THEN** MIDI from bars 4-8 is extracted for classification

#### Scenario: Predict rebracketing type
- **WHEN** classifier receives text="is not" and MIDI segment
- **THEN** prediction is returned (e.g., "ontological" with 0.92 confidence)

### Requirement: Rebracketing Validation
The system SHALL validate that markers are emphasized correctly.

#### Scenario: Compare prediction to ground truth
- **WHEN** ground truth is "ontological" and prediction is "ontological"
- **THEN** validation passes

#### Scenario: Detect mismatch
- **WHEN** ground truth is "ontological" but prediction is "spatial"
- **THEN** validation fails, feedback generated

#### Scenario: Confidence threshold
- **WHEN** prediction confidence < 0.8
- **THEN** validation fails (low confidence)

### Requirement: Rebracketing-Driven Regeneration
The system SHALL regenerate sections when validation fails.

#### Scenario: Increase emphasis
- **WHEN** marker validation fails
- **THEN** emphasis parameters are increased (1.5x multiplier)

#### Scenario: Regenerate region
- **WHEN** emphasis increased
- **THEN** marker region (±2 bars) is regenerated

#### Scenario: Re-validate
- **WHEN** region regenerated
- **THEN** classifier runs again on new version

#### Scenario: Limit iterations
- **WHEN** validation fails 3 times
- **THEN** stop regeneration, flag for human review

### Requirement: Chromatic Style Guidance
The system SHALL provide style guidance for each chromatic mode.

#### Scenario: Get Red mode guidance
- **WHEN** chromatic_mode="Red"
- **THEN** guidance includes: aggressive harmonies, heavy drums, intense melody

#### Scenario: Get Violet mode guidance
- **WHEN** chromatic_mode="Violet"
- **THEN** guidance includes: extended voicings, sparse texture, contemplative

#### Scenario: Map to parameters
- **WHEN** guidance received
- **THEN** generator parameters are set accordingly

### Requirement: Style Embedding Scoring
The system SHALL score generated music against style embeddings.

#### Scenario: Embed generated progression
- **WHEN** chord progression is generated
- **THEN** progression is embedded into style space

#### Scenario: Compare to target mode
- **WHEN** target is "Indigo" mode
- **THEN** generated embedding is compared to Indigo reference embedding

#### Scenario: Compute style score
- **WHEN** embeddings compared
- **THEN** similarity score (0-1) is returned

### Requirement: Temporal Energy Prediction
The system SHALL predict energy curve from concept text.

#### Scenario: Analyze concept for temporal cues
- **WHEN** concept is "gradual acceleration into chaos"
- **THEN** energy curve starts low, ends high

#### Scenario: Predict per-section energy
- **WHEN** song has 6 sections
- **THEN** 6 energy values are predicted (e.g., [0.2, 0.4, 0.6, 0.7, 0.9, 0.5])

#### Scenario: Confidence scores
- **WHEN** prediction is made
- **THEN** confidence for each section is provided

### Requirement: Energy-to-Parameters Mapping
The system SHALL map predicted energy to generator parameters.

#### Scenario: Low energy parameters
- **WHEN** energy < 0.4
- **THEN** sparse instrumentation, simple chords, low velocity

#### Scenario: High energy parameters
- **WHEN** energy > 0.7
- **THEN** full instrumentation, complex chords, high velocity

#### Scenario: Smooth transitions
- **WHEN** energy changes from 0.4 to 0.8
- **THEN** parameters transition gradually (not abrupt)

### Requirement: Energy Validation
The system SHALL validate generated energy matches predicted energy.

#### Scenario: Compute actual energy
- **WHEN** arrangement is generated
- **THEN** actual energy is computed from instrumentation, velocity, density

#### Scenario: Compare to prediction
- **WHEN** predicted energy is 0.8, actual is 0.75
- **THEN** close match, validation passes

#### Scenario: Detect mismatch
- **WHEN** predicted energy is 0.8, actual is 0.4
- **THEN** large mismatch, regeneration triggered

### Requirement: Multi-Modal Embedding
The system SHALL encode text, MIDI, and audio into shared embedding space.

#### Scenario: Encode concept text
- **WHEN** concept is "spatial fragmentation"
- **THEN** text embedding vector is generated

#### Scenario: Encode MIDI arrangement
- **WHEN** MIDI file is provided
- **THEN** MIDI embedding vector is generated

#### Scenario: Encode audio recording
- **WHEN** WAV file is provided
- **THEN** audio embedding vector is generated

#### Scenario: Compute similarity
- **WHEN** comparing text and MIDI embeddings
- **THEN** cosine similarity is computed

### Requirement: Embedding-to-Parameters Prediction
The system SHALL predict musical parameters from concept embeddings.

#### Scenario: Predict key and mode
- **WHEN** concept embedding is provided
- **THEN** key (e.g., "Bb Minor") and mode are predicted

#### Scenario: Predict tempo
- **WHEN** concept suggests slowness or speed
- **THEN** appropriate tempo (60-180 BPM) is predicted

#### Scenario: Predict chromatic mode
- **WHEN** concept has introspective themes
- **THEN** "Indigo" or "Violet" mode is predicted

### Requirement: Concept-Driven Generation
The system SHALL generate arrangements from concept text using ML predictions.

#### Scenario: Encode and predict
- **WHEN** concept is provided
- **THEN** text is encoded, parameters predicted

#### Scenario: Generate with predictions
- **WHEN** parameters predicted
- **THEN** arrangement pipeline generates with those parameters

#### Scenario: Validate result
- **WHEN** arrangement generated
- **THEN** generated embedding is compared to concept embedding

### Requirement: ML-Informed Scoring
The system SHALL combine theory-based and ML-based scores.

#### Scenario: Weighted combination
- **WHEN** theory_score=0.85, ml_score=0.78, weights=[0.6, 0.4]
- **THEN** combined_score = 0.6*0.85 + 0.4*0.78 = 0.822

#### Scenario: Confidence weighting
- **WHEN** ML prediction confidence is low (0.5)
- **THEN** ML score weight decreases, theory weight increases

#### Scenario: Ensemble voting
- **WHEN** multiple models score same arrangement
- **THEN** scores are aggregated (mean, median, or voting)

### Requirement: Synthetic Data Generation
The system SHALL generate new training data from generated arrangements.

#### Scenario: Generate for new concept
- **WHEN** concept not in Rainbow Table
- **THEN** arrangement is generated

#### Scenario: Export MIDI for performance
- **WHEN** arrangement generated
- **THEN** MIDI file is exported for musician

#### Scenario: Track generation metadata
- **WHEN** MIDI exported
- **THEN** concept, parameters, model version are stored

### Requirement: Recording to Training Data Conversion
The system SHALL convert musician recordings into training examples.

#### Scenario: Pair recording with concept
- **WHEN** musician records MIDI arrangement
- **THEN** audio is paired with original concept text

#### Scenario: Create training example
- **WHEN** pairing complete
- **THEN** (text, audio, MIDI, labels) example is created

#### Scenario: Add to dataset
- **WHEN** training example created
- **THEN** example is added to training dataset

### Requirement: Data Quality Control
The system SHALL validate quality before adding to training data.

#### Scenario: Agent validation
- **WHEN** arrangement generated
- **THEN** White Agent checks ontological correctness

#### Scenario: Quality threshold
- **WHEN** arrangement scores < 0.7
- **THEN** arrangement is rejected, not recorded

#### Scenario: Track rejections
- **WHEN** arrangement rejected
- **THEN** rejection reason is stored for analysis

### Requirement: Model Retraining Pipeline
The system SHALL retrain models when new data available.

#### Scenario: Detect new data
- **WHEN** 100+ new recordings added
- **THEN** retraining is triggered

#### Scenario: Merge datasets
- **WHEN** retraining
- **THEN** new data is merged with Rainbow Table baseline

#### Scenario: Validate new model
- **WHEN** retraining complete
- **THEN** new model is validated on held-out set

#### Scenario: Deploy if better
- **WHEN** new model outperforms old model
- **THEN** new model is deployed

### Requirement: Human Feedback Collection
The system SHALL collect musician feedback on generated MIDI.

#### Scenario: Playability feedback
- **WHEN** musician performs bass line
- **THEN** feedback: "easy", "medium", "hard", or "impossible"

#### Scenario: Musicality feedback
- **WHEN** musician evaluates arrangement
- **THEN** feedback: "good", "okay", or "poor"

#### Scenario: Element-specific feedback
- **WHEN** musician finds issue
- **THEN** feedback tagged to specific element (e.g., "bass bar 12 awkward")

### Requirement: Feedback Storage
The system SHALL store and retrieve musician feedback.

#### Scenario: Store feedback
- **WHEN** feedback received
- **THEN** stored with arrangement_id, element, rating, comment

#### Scenario: Aggregate feedback
- **WHEN** multiple musicians rate same element
- **THEN** ratings are aggregated (mean, mode)

#### Scenario: Detect consistent issues
- **WHEN** 3+ musicians flag same issue
- **THEN** issue is flagged for parameter adjustment

### Requirement: Feedback-Driven Refinement
The system SHALL use feedback to improve generation.

#### Scenario: Identify bad parameters
- **WHEN** parameter combination produces "impossible" playability
- **THEN** combination is flagged, weight reduced

#### Scenario: Boost good parameters
- **WHEN** parameter combination produces "good" ratings
- **THEN** combination weight is increased

#### Scenario: Feedback-weighted scoring
- **WHEN** generating new arrangement
- **THEN** scoring includes feedback history

### Requirement: Interpretability
The system SHALL explain ML decisions in human-readable form.

#### Scenario: Explain parameter prediction
- **WHEN** ML predicts key="Bb Minor"
- **THEN** explanation: "Concept 'dissolution' suggests minor key, 'recursive' suggests Bb tonal center"

#### Scenario: Explain style guidance
- **WHEN** ML suggests extended voicings for Violet mode
- **THEN** explanation: "Violet mode: introspective, complex → 9ths, 11ths, 13ths"

#### Scenario: Visualize embeddings
- **WHEN** user requests
- **THEN** t-SNE plot shows concept embedding relative to mode clusters

### Requirement: User Override
The system SHALL allow users to override ML suggestions.

#### Scenario: Override chromatic mode
- **WHEN** ML predicts "Indigo" but user wants "Red"
- **THEN** user override is accepted

#### Scenario: Override parameters
- **WHEN** ML predicts tempo=88 but user wants tempo=120
- **THEN** user override is used

#### Scenario: Force theory-only mode
- **WHEN** user disables ML
- **THEN** generation proceeds with theory only

### Requirement: Graceful Degradation
The system SHALL handle ML failures gracefully.

#### Scenario: Model inference error
- **WHEN** ML model throws exception
- **THEN** error logged, fall back to theory-only

#### Scenario: NaN outputs
- **WHEN** ML model outputs NaN or inf
- **THEN** outputs rejected, fall back to defaults

#### Scenario: Notify user
- **WHEN** ML unavailable
- **THEN** user is notified: "ML guidance unavailable, using theory only"

### Requirement: Chord Generator ML Integration
The system SHALL integrate ML guidance with chord generation.

#### Scenario: Accept chromatic mode
- **WHEN** generating chords
- **THEN** chromatic mode parameter is passed to style model

#### Scenario: Map to chord parameters
- **WHEN** style guidance received
- **THEN** chord voicing, complexity, variety weights are set

#### Scenario: ML-informed scoring
- **WHEN** scoring chord progressions
- **THEN** theory score + style embedding score combined

### Requirement: Melody Generator ML Integration
The system SHALL integrate ML guidance with melody generation.

#### Scenario: Concept embedding to melody
- **WHEN** concept encoded
- **THEN** melody tessitura, contour, rhythm predicted

#### Scenario: Validate marker emphasis
- **WHEN** melody generated
- **THEN** classifier validates rebracketing markers emphasized

### Requirement: Bass Generator ML Integration
The system SHALL integrate ML guidance with bass generation.

#### Scenario: Chromatic mode to bass style
- **WHEN** Red mode specified
- **THEN** aggressive bass patterns (root-fifth, driving)

#### Scenario: Temporal guidance to density
- **WHEN** high energy predicted
- **THEN** busier bass patterns (walking, syncopation)

### Requirement: Drum Generator ML Integration
The system SHALL integrate ML guidance with drum generation.

#### Scenario: Chromatic mode to groove
- **WHEN** Indigo mode specified
- **THEN** sparse jazz groove

#### Scenario: Temporal guidance to fills
- **WHEN** energy spike predicted
- **THEN** build-up fill before spike

### Requirement: Arrangement Pipeline ML Integration
The system SHALL integrate ML validation with full arrangement.

#### Scenario: Temporal energy curve
- **WHEN** generating arrangement
- **THEN** temporal model predicts energy for all sections

#### Scenario: Rebracketing validation
- **WHEN** arrangement complete
- **THEN** classifier validates all markers

#### Scenario: Regenerate failed sections
- **WHEN** validation fails
- **THEN** specific sections regenerated with adjusted parameters

### Requirement: Ensemble Model Scoring
The system SHALL use multiple models for robust predictions.

#### Scenario: Multiple model predictions
- **WHEN** arrangement generated
- **THEN** scored by v1.0, v1.1, v1.2 models

#### Scenario: Aggregate predictions
- **WHEN** predictions differ
- **THEN** majority vote or mean is used

#### Scenario: Flag low agreement
- **WHEN** models disagree strongly
- **THEN** arrangement flagged for human review

### Requirement: A/B Testing
The system SHALL compare ML-guided vs theory-only generation.

#### Scenario: Generate both versions
- **WHEN** A/B test requested
- **THEN** ML-guided and theory-only arrangements generated

#### Scenario: Export for comparison
- **WHEN** both generated
- **THEN** separate MIDI files exported

#### Scenario: Collect preference
- **WHEN** user listens to both
- **THEN** preference recorded (ML vs theory)

### Requirement: Model Performance Monitoring
The system SHALL monitor ML model performance over time.

#### Scenario: Track inference latency
- **WHEN** model inference runs
- **THEN** latency (ms) is logged

#### Scenario: Track prediction confidence
- **WHEN** predictions made
- **THEN** confidence trends are monitored

#### Scenario: Detect model drift
- **WHEN** confidence drops over time
- **THEN** drift detected, retraining triggered

### Requirement: Configurable ML Weights
The system SHALL allow configuring ML vs theory balance.

#### Scenario: Theory-heavy weighting
- **WHEN** weights={theory: 0.8, ml: 0.2}
- **THEN** theory scores dominate

#### Scenario: ML-heavy weighting
- **WHEN** weights={theory: 0.3, ml: 0.7}
- **THEN** ML scores dominate

#### Scenario: Balanced weighting
- **WHEN** weights={theory: 0.5, ml: 0.5}
- **THEN** equal influence

### Requirement: Feedback Loop Closure
The system SHALL close the human-in-the-loop from generation to recording to retraining.

#### Scenario: End-to-end flow
- **WHEN** concept provided
- **THEN** ML guides generation → MIDI exported → musician performs → recording added to training data → model retrained → better ML guidance

#### Scenario: Track loop iterations
- **WHEN** multiple iterations complete
- **THEN** model improvement tracked (v1.0 → v1.1 → v1.2)

#### Scenario: Convergence detection
- **WHEN** model improvements plateau
- **THEN** loop stabilizes, no further retraining needed
