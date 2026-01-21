# Change: Add ML Model Integration with Generators

## Why
The generators use music theory + brute-force search to create MIDI, while the training pipeline uses ML to understand rebracketing patterns and chromatic modes. Integration creates a feedback loop: trained models **guide** generation (not replace it), and generated music becomes **new training data**. This hybrid approach honors both musical craftsmanship and machine intelligence.

## What Changes
- Add `ModelGuidanceLayer` for integrating trained models with generators
- Implement rebracketing classifier integration (validate ontological correctness)
- Add chromatic style model integration (suggest musical character per mode)
- Implement temporal sequence model integration (guide progression pacing)
- Support for multi-modal embedding integration (concept → musical parameters)
- Add synthetic data generation pipeline (generators → training data)
- Implement human-in-the-loop feedback (musicians critique → model refinement)
- Integration with all generators (chord, melody, bass, drums, arrangement)

## Impact
- Affected specs: ml-integration (new capability)
- Affected code:
  - `app/generators/integration/` (new directory)
  - `app/generators/integration/model_guidance.py` - model inference layer
  - `app/generators/integration/rebracketing_validator.py` - classifier integration
  - `app/generators/integration/chromatic_guidance.py` - style model integration
  - `app/generators/integration/temporal_guidance.py` - sequence model integration
  - `app/generators/integration/synthetic_data.py` - generator → training data
  - Integration with ALL generators and training pipeline
- Dependencies: PyTorch, trained models from `/training/`, all generators
- Complexity: Very High - bidirectional integration, requires both systems complete

## Design Considerations

### Philosophy: ML as Guide, Not Generator

**What ML Does**:
- Rebracketing classifier: "Is this marker emphasized correctly?" (validation)
- Chromatic style model: "Red mode should feel aggressive - try these voicings" (suggestion)
- Temporal sequence model: "This progression should accelerate here" (pacing guidance)
- Multi-modal embeddings: "Concept X maps to harmonic character Y" (parameter mapping)

**What ML Does NOT Do**:
- Generate MIDI directly (that's the generators' job)
- Replace music theory (theory provides constraints)
- Replace human musicians (humans perform the MIDI)
- Make final creative decisions (composers and agents do that)

### Integration Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Arrangement Pipeline                  │
│  (Orchestrates chords → melody → bass → drums)          │
└────────────┬───────────────────────────────┬────────────┘
             │                               │
             ▼                               ▼
┌────────────────────────┐      ┌──────────────────────────┐
│   Music Theory Layer   │      │   ML Guidance Layer      │
│  - Chord progressions  │◄────►│  - Trained models        │
│  - Voice leading       │      │  - Inference             │
│  - Scoring functions   │      │  - Parameter suggestion  │
└────────────────────────┘      └──────────────────────────┘
             │                               │
             │                               │
             ▼                               ▼
┌────────────────────────────────────────────────────────┐
│              Brute-Force Search + Scoring              │
│  - Generate N candidates (theory-guided)               │
│  - Score with ML-informed criteria                     │
│  - Return top-K for human selection                    │
└────────────────────────────────────────────────────────┘
             │
             ▼
┌────────────────────────────────────────────────────────┐
│                   Agent Critique                       │
│  - White: Ontological correctness                      │
│  - Violet: Musical quality                             │
│  - Black: Creative alternatives                        │
└────────────────────────────────────────────────────────┘
             │
             ▼
┌────────────────────────────────────────────────────────┐
│            MIDI Export → Human Performance             │
└────────────────────────────────────────────────────────┘
             │
             ▼
┌────────────────────────────────────────────────────────┐
│         Recordings → New Training Data                 │
│  (Feedback loop: generated music trains models)        │
└────────────────────────────────────────────────────────┘
```

### Rebracketing Classifier Integration

**Use Case**: Validate that generated arrangement emphasizes markers correctly.

**Flow**:
1. Arrangement pipeline generates complete song
2. Extract rebracketing markers from lyrics
3. Run classifier on each marker region (multi-modal: text + generated MIDI)
4. Classifier predicts: Which rebracketing type is this?
5. Compare prediction to ground truth (from concept metadata)
6. If mismatch: marker not emphasized correctly → regenerate with stronger emphasis
7. If match: marker correctly emphasized → accept arrangement

**Example**:
```python
# After arrangement generation
markers = extract_markers(lyrics)  # "[(is not)]" → ontological rebracketing

for marker in markers:
    # Extract generated MIDI around marker (±2 bars)
    midi_segment = arrangement.extract_segment(
        start=marker.position - 2_bars,
        end=marker.position + 2_bars
    )

    # Run classifier
    prediction = rebracketing_classifier.predict(
        text=marker.text,
        audio=None,  # No audio yet, just MIDI
        midi=midi_segment
    )

    # Validate
    if prediction.type != marker.ground_truth_type:
        # Marker not emphasized correctly
        feedback = f"Marker '{marker.text}' should be {marker.ground_truth_type}, " \
                   f"but classifier thinks it's {prediction.type}. " \
                   f"Increase emphasis."

        # Regenerate with stronger emphasis
        arrangement.regenerate_marker_region(
            marker=marker,
            emphasis_multiplier=1.5  # More dramatic emphasis
        )
```

### Chromatic Style Model Integration

**Use Case**: Suggest musical character for each chromatic mode.

**Flow**:
1. User specifies chromatic mode (e.g., "Violet")
2. Style model provides guidance: "Violet = introspective, complex harmony, sparse texture"
3. Map to generator parameters:
   - Chord generator: Use extended voicings (9ths, 11ths, 13ths), avoid simple triads
   - Melody generator: Lower tessitura, minor keys, stepwise motion
   - Bass generator: Walking patterns, chromatic approaches
   - Drum generator: Sparse, jazz-influenced grooves
4. Generate with those parameters
5. Score generated output against style model embeddings
6. Repeat with variations, select best match

**Example**:
```python
# Get chromatic mode guidance
chromatic_mode = "Violet"
style_guidance = chromatic_style_model.get_guidance(chromatic_mode)

# style_guidance = {
#     "harmonic_complexity": 0.8,  # High complexity
#     "rhythmic_density": 0.3,     # Sparse
#     "melodic_contour": "stepwise",
#     "chord_extensions": True,
#     "suggested_voicings": ["maj9", "min11", "dom7#11"]
# }

# Map to chord generator parameters
chord_params = {
    "category": "extended",  # Not just triads
    "voice_leading_weight": 0.6,  # Emphasize smooth voice leading
    "variety_weight": 0.4,
    "preferred_qualities": style_guidance["suggested_voicings"]
}

# Generate chords with style-informed parameters
chords = chord_generator.generate_progression_brute_force(
    key_root="A",
    mode="Minor",
    length=8,
    num_candidates=1000,
    **chord_params
)

# Score against style model
for progression in chords:
    style_score = chromatic_style_model.score_progression(
        progression=progression,
        target_mode=chromatic_mode
    )
    # Higher score = better match to Violet aesthetic
```

### Temporal Sequence Model Integration

**Use Case**: Guide when progressions should accelerate, decelerate, or maintain tension.

**Flow**:
1. Temporal model analyzes concept text: "Where should intensity build?"
2. Model predicts energy curve over time
3. Map energy curve to generator parameters:
   - Low energy → simpler chords, slower harmonic rhythm
   - High energy → more complex chords, faster changes
   - Transitions → fills, crashes, modulations
4. Generate arrangement following energy curve
5. Validate generated energy matches predicted energy

**Example**:
```python
# Predict temporal energy curve from concept
concept_text = "The room is not a container - space unfolds recursively"
energy_curve = temporal_sequence_model.predict_energy(
    text=concept_text,
    num_sections=6  # Intro, V1, Ch, V2, Ch, Outro
)

# energy_curve = [0.2, 0.5, 0.8, 0.6, 0.9, 0.3]
#                Intro  V1   Ch   V2   Ch   Outro

# Map to sectional parameters
for section, energy in zip(song_structure, energy_curve):
    if energy < 0.4:  # Low energy
        section_params = {
            "chord_complexity": "simple",
            "bass_pattern": "root_notes",
            "drum_pattern": "minimal",
            "melody_tessitura": "low"
        }
    elif energy > 0.7:  # High energy
        section_params = {
            "chord_complexity": "extended",
            "bass_pattern": "walking",
            "drum_pattern": "full_kit",
            "melody_tessitura": "high"
        }

    section.generate_with_params(section_params)
```

### Multi-Modal Embedding Integration

**Use Case**: Map concept text to musical parameters via learned embeddings.

**Flow**:
1. Concept text → multi-modal encoder → embedding vector
2. Embedding vector → parameter predictor → musical parameters
3. Use parameters for generation
4. Trained on: Rainbow Table albums (text + audio + MIDI)

**Example**:
```python
# Encode concept
concept = "Identity fragments reassemble in reverse temporal order"
embedding = multimodal_encoder.encode_text(concept)

# Predict musical parameters from embedding
params = embedding_to_params_model.predict(embedding)

# params = {
#     "key": "D",
#     "mode": "Minor",
#     "chromatic_mode": "Indigo",  # Introspective, temporal themes
#     "tempo": 95,
#     "time_signature": "4/4",
#     "harmonic_rhythm": "slow",  # Chord changes every 2 bars
#     "melodic_character": "fragmented",  # Short phrases, rests
#     "rebracketing_emphasis": "high"  # Identity markers very prominent
# }

# Generate with predicted parameters
arrangement = arrangement_pipeline.generate(
    concept=concept,
    params=params
)
```

### Synthetic Data Generation Pipeline

**Use Case**: Generators create new training data for models.

**Flow**:
1. Generate arrangements for new concepts (not in Rainbow Table)
2. Export as MIDI
3. Human musicians perform and record
4. Recordings become new training examples
5. Retrain models with expanded dataset

**Bootstrapping Loop**:
```
Iteration 1:
  - Train on Rainbow Table (8 albums)
  - Models learn from ~14k segments

Iteration 2:
  - Models guide generators
  - Generate 100 new songs
  - Musicians perform → 100 new recordings
  - Retrain on Rainbow Table + 100 new songs

Iteration 3:
  - Improved models guide better generation
  - Generate 200 more songs
  - Retrain on Rainbow Table + 300 songs

...continue...
```

**Data Quality Control**:
- White Agent validates ontological correctness before recording
- Violet Agent validates musical quality before recording
- Human musicians provide qualitative feedback
- Only high-quality generated songs become training data

### Human-in-the-Loop Feedback

**Use Case**: Musicians and composers critique generated MIDI, feedback refines models.

**Flow**:
1. Generate arrangement
2. Export MIDI
3. Musician loads MIDI, performs, records
4. Musician provides feedback: "This bass line is awkward at bar 12"
5. Feedback tagged to specific musical element (bass, bar 12)
6. Model learns: "This parameter combination produced awkward bass"
7. Future generations avoid that combination

**Feedback Types**:
- **Playability**: "This is impossible to play on real bass"
- **Musicality**: "This melody doesn't fit the chords"
- **Ontological**: "This marker isn't emphasized enough"
- **Aesthetic**: "This doesn't feel like Violet mode"

### Integration Points

**Chord Generator + ML**:
- Chromatic style model suggests voicing styles
- Temporal model suggests harmonic rhythm
- Rebracketing classifier validates chord changes at markers

**Melody Generator + ML**:
- Multi-modal embeddings suggest melodic character
- Rebracketing classifier validates marker emphasis
- Chromatic style model suggests tessitura and contour

**Bass Generator + ML**:
- Chromatic style model suggests groove character
- Temporal model suggests when to increase/decrease density

**Drum Generator + ML**:
- Chromatic style model suggests drum style
- Temporal model suggests when to add fills/builds
- Rebracketing classifier validates rhythmic emphasis at markers

**Arrangement Pipeline + ML**:
- Temporal model provides overall energy curve
- Rebracketing classifier validates entire arrangement
- All models run in validation phase after generation

## Scoring Criteria

ML integration will be scored on:

1. **Guidance Accuracy** (0-1):
   - Chromatic mode guidance matches expected character: +1.0
   - Temporal energy predictions match musical reality: +0.9
   - Poor guidance (contradictory suggestions): -0.7

2. **Validation Accuracy** (0-1):
   - Rebracketing classifier correctly identifies emphasis: +1.0
   - Classifier false positives/negatives: -0.6

3. **Parameter Mapping Quality** (0-1):
   - Embeddings → parameters produce coherent music: +1.0
   - Parameters produce unplayable/unmusical results: -0.8

4. **Human-ML Agreement** (0-1):
   - Musicians agree with ML quality assessments: +1.0
   - ML thinks it's good, musicians say it's bad: -0.9

5. **Feedback Loop Effectiveness** (0-1):
   - Models improve after synthetic data retraining: +1.0
   - Models don't improve or degrade: -0.5

6. **Integration Transparency** (0-1):
   - Users understand what ML is doing: +1.0
   - Black-box decisions, no interpretability: -0.6

## Implementation Phases

**Phase 1: Model Inference Layer** (Foundation)
- Load trained PyTorch models
- Implement inference API
- Handle model versioning

**Phase 2: Rebracketing Validation** (Core)
- Integrate classifier with arrangement pipeline
- Validate marker emphasis after generation
- Trigger regeneration if validation fails

**Phase 3: Chromatic Style Guidance** (Enhancement)
- Map chromatic modes to musical parameters
- Score generated output against style embeddings
- Adjust generator weights based on mode

**Phase 4: Temporal Sequence Guidance** (Dynamics)
- Predict energy curve from concept
- Map energy to sectional parameters
- Validate generated energy matches prediction

**Phase 5: Multi-Modal Parameter Prediction** (Advanced)
- Encode concepts to embeddings
- Predict parameters from embeddings
- Generate with predicted parameters

**Phase 6: Synthetic Data Pipeline** (Feedback Loop)
- Export generated MIDI for performance
- Record new audio
- Create training examples from recordings
- Retrain models with expanded dataset

**Phase 7: Human Feedback Integration** (Quality Control)
- Collect musician feedback on generated MIDI
- Tag feedback to specific musical elements
- Use feedback to refine model guidance
- Close the human-in-the-loop

## Important Constraints

- **ML guides, doesn't replace**: Music theory and brute-force search remain primary generation mechanism
- **Human approval required**: No generated arrangement goes to recording without human review
- **Interpretability**: All ML decisions must be explainable (no black boxes)
- **Graceful degradation**: If ML models fail, generators still work (theory-only mode)
- **Versioning**: Track which model version generated which arrangement
- **Feedback privacy**: Musician feedback stored securely, not sold to Spotify

## Risk Mitigation

**Risk: ML gives bad guidance**
- Mitigation: Multiple scoring functions (ML + theory), human final approval

**Risk: Feedback loop creates training data collapse**
- Mitigation: Always keep Rainbow Table as canonical baseline

**Risk: Models overfit to generator outputs**
- Mitigation: Maintain separation between real recordings and synthetic data

**Risk: Black-box decisions alienate musicians**
- Mitigation: Full interpretability, musicians can override ML suggestions

## Success Metrics

1. **Validation Accuracy**: Rebracketing classifier agrees with human judgment >90%
2. **Style Coherence**: Generated arrangements match chromatic mode aesthetic (human eval)
3. **Temporal Prediction**: Energy curve matches listener perception (survey)
4. **Playability**: Musicians rate generated MIDI as "performable" >85%
5. **Improvement Over Time**: Each retrained model version scores better than previous

## Example Integration Flow

```
User Input:
  - Concept: "Memory dissolves into recursive fractals"
  - Chromatic Mode: "Indigo" (introspective, complex)

Step 1: Multi-Modal Encoding
  - Encode concept → embedding vector
  - Predict parameters: Key=Bb Minor, Tempo=88, Time=4/4

Step 2: Chromatic Style Guidance
  - Indigo mode → extended voicings, sparse drums, contemplative melody
  - Map to generator parameters

Step 3: Temporal Energy Prediction
  - Concept suggests slow build, no climax (recursive dissolution)
  - Energy curve: [0.3, 0.4, 0.5, 0.4, 0.3, 0.2]

Step 4: Theory-Guided Generation
  - Chord generator: Bb minor progressions with 9ths, 11ths
  - Melody generator: Stepwise, low tessitura, fragmented phrases
  - Bass generator: Walking bass, chromatic approaches
  - Drum generator: Sparse jazz groove, no crashes

Step 5: Brute-Force Search
  - Generate 500 candidates
  - Score with theory metrics + ML style matching
  - Return top 10

Step 6: Rebracketing Validation
  - Extract markers: "[(dissolves)]" "[(recursive)]"
  - Run classifier on each marker region
  - Validate emphasis (all markers correctly identified)

Step 7: Agent Critique
  - White: "Markers are subtle but present, matches 'dissolution' theme"
  - Violet: "Voice leading smooth, harmony appropriately complex"
  - Black: "What if drums dropped out entirely?"

Step 8: Export MIDI
  - Multi-track arrangement ready for performance
  - Metadata: Model version, parameters, scores

Step 9: Human Performance
  - Musician loads MIDI, performs with real instruments
  - Records audio

Step 10: Feedback Loop
  - Musician: "Bass at bar 14 was awkward, but overall very playable"
  - Tag feedback: {element: "bass", bar: 14, rating: "awkward"}
  - Retrain models with new recording + feedback
```

This integration closes the loop: concept → parameters → generation → validation → performance → recording → training data → improved models → better generation.
