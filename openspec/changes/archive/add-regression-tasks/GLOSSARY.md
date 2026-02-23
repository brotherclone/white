# Rainbow Table Regression - Glossary

## Core Concepts

### Rainbow Table
The ontological framework organizing concepts across three dimensions: Temporal, Spatial, and Ontological. Creates 27 possible modes (3³) corresponding to different albums.

### Ontological Dimensions

#### Temporal Dimension
The time orientation of a concept:
- **Past**: Memories, history, nostalgia, retrospection
- **Present**: Current experience, immediacy, now
- **Future**: Anticipation, projection, what's to come

#### Spatial Dimension  
The focus or subject of a concept:
- **Thing**: Objects, items, artifacts, physical entities
- **Place**: Locations, settings, environments, spaces
- **Person**: People, characters, relationships, identities

#### Ontological Dimension
The reality status of a concept:
- **Imagined**: Fiction, fantasy, hypothetical, never-real
- **Forgotten**: Lost, obscured, once-known-now-unknown
- **Known**: Documented, verified, real and remembered

### Albums

Each combination of modes maps to an album color:

- **Orange Album**: Past_Thing_Imagined (e.g., imaginary toys from childhood)
- **Red Album**: Past_Thing_Forgotten (e.g., misremembered objects)
- **Yellow Album**: Present_Place_Imagined (e.g., fictional game locations)
- **Green Album**: Future_Person_Known (e.g., projected real relationships)
- **Blue Album**: Future_Place_Known (e.g., anticipated real destinations)
- **Indigo Album**: Present_Person_Forgotten (e.g., someone you're forgetting now)
- **Violet Album**: Past_Person_Known (e.g., historical figures)
- **Black Album**: None_None_None (e.g., pure chaos, void, undefined)

## Regression Terms

### Continuous Scores
Probability distributions over modes instead of hard classifications:
- `past_score=0.87, present_score=0.12, future_score=0.01`
- Sum to 1.0 via softmax activation
- Enable hybrid state detection and uncertainty quantification

### Chromatic Confidence
Single score [0-1] indicating how strongly a concept fits ANY single mode:
- **High confidence (>0.7)**: Clear mode assignment
- **Medium confidence (0.4-0.7)**: Somewhat ambiguous
- **Low confidence (<0.4)**: Very unclear, diffuse, or Black Album

### Soft Targets
Continuous training targets derived from discrete labels:
```python
# Hard (discrete)
"Past" → Past=1, Present=0, Future=0

# Soft (continuous)
"Past" → Past=0.87, Present=0.12, Future=0.01
```
Allows label smoothing and context-aware adjustments.

## State Classifications

### Dominant State
Clear assignment to one mode:
- Top score in dimension >0.6
- Example: `past=0.92, present=0.05, future=0.03` → Dominant Past
- Concept has strong orientation in that dimension

### Hybrid State
Straddling two modes (liminal concept):
- Top two scores within 0.15 of each other
- Example: `past=0.52, present=0.45, future=0.03` → Hybrid Past/Present
- Concept is transitional or spans modes

### Diffuse State
No clear orientation in a dimension:
- All three scores within 0.2 of uniform (≈0.33)
- Example: `thing=0.35, place=0.32, person=0.33` → Diffuse Spatial
- Concept lacks clear focus in that dimension

### Triple Diffuse (Black Album Candidate)
All three dimensions diffuse simultaneously:
- Temporal diffuse AND Spatial diffuse AND Ontological diffuse
- Suggests None_None_None mode (pure chaos/void)
- Confidence typically <0.3

## Transmigration

### Transmigration Distance
Measure of conceptual distance between ontological states:
```
distance = sqrt(Δtemporal² + Δspatial² + Δontological²)
```

Where each Δ is the L2 norm between score vectors:
```python
Δtemporal = ||[0.9,0.1,0.0] - [0.1,0.2,0.7]|| 
          = sqrt((0.9-0.1)² + (0.1-0.2)² + (0.0-0.7)²)
          = 1.06
```

### Transmigration Categories
- **Easy (<1.0)**: Within-album shift (minor adjustment)
- **Moderate (1.0-2.0)**: Cross-album shift (significant change)
- **Difficult (>2.0)**: Multi-dimensional shift (major transformation)

### Transmigration Path
Sequence of intermediate states for difficult transmigrations:
```
Past_Thing_Imagined 
  → Past_Thing_Known       (change: Imagined→Known)
  → Present_Thing_Known    (change: Past→Present)
  → Future_Thing_Known     (change: Present→Future)
  → Future_Person_Known    (change: Thing→Person)
```

## Validation

### Validation Status

#### ACCEPT
Concept passes all thresholds:
- Chromatic confidence >0.7
- All dimensions have dominant mode (>0.6)
- Ready for use without modification

#### ACCEPT_HYBRID
Concept is liminal but coherent:
- Hybrid in ≤2 dimensions
- Confidence >0.5
- Acceptable with dominant dimension determining album

#### ACCEPT_BLACK
Concept fits Black Album:
- All three dimensions diffuse
- Confidence <0.3
- Valid for None_None_None mode

#### REJECT
Concept fails validation:
- Reasons: diffuse_ontology, ood_detection
- Requires regeneration or editing
- Actionable suggestions provided

### Validation Thresholds (Configurable)

| Threshold | Default | Purpose |
|-----------|---------|---------|
| confidence_threshold | 0.7 | Minimum for ACCEPT |
| dominant_threshold | 0.6 | Score needed for dominant |
| hybrid_threshold | 0.15 | Max Δ for hybrid flag |
| diffuse_threshold | 0.2 | Max deviation for diffuse |
| uncertainty_threshold | 0.8 | Max for ACCEPT |

## White Agent Integration

### Validation Gate
Automated checkpoint in White Agent workflow:
```
Generate Concept → Validate → Branch
                               ├─ ACCEPT → Continue
                               ├─ ACCEPT_HYBRID → Continue
                               ├─ ACCEPT_BLACK → Black Agent
                               └─ REJECT → Regenerate
```

### ValidationResult
Structured output from validation API containing:
- All 10 regression scores
- Predicted album and mode
- Validation status and flags
- Uncertainty estimates
- Transmigration distances to all albums
- Actionable suggestions

### Regeneration Loop
When concept rejected:
1. White Agent receives suggestions
2. Adjusts generation prompt based on feedback
3. Generates new concept
4. Re-validates
5. Repeat until ACCEPT or max iterations

## Technical Terms

### Label Smoothing
Softening one-hot targets to prevent overconfidence:
```python
# Before: [1.0, 0.0, 0.0]
# After:  [0.9, 0.05, 0.05]  (alpha=0.1)
```

### Temporal Context Smoothing
Adjusting targets based on surrounding segments:
```python
# Sequence: Past, Present, Past
# Middle segment adjusted: 
#   [0.0, 1.0, 0.0] → [0.3, 0.4, 0.3]
```

### Softmax Activation
Converts logits to probability distribution summing to 1.0:
```python
softmax([2.0, 1.0, 0.1]) = [0.66, 0.24, 0.10]
```

### Sigmoid Activation
Squashes output to [0, 1] range:
```python
sigmoid(0.5) = 0.62
sigmoid(-2.0) = 0.12
```

### Multi-Task Learning
Training multiple objectives simultaneously:
- Shared encoder learns general representation
- Task-specific heads learn specialized outputs
- Combined loss balances objectives

### Uncertainty Estimation
Quantifying model confidence:
- **Aleatoric**: Data uncertainty (inherent ambiguity)
- **Epistemic**: Model uncertainty (lack of knowledge)
- Methods: ensembles, MC dropout, evidential learning

## Metrics

### MAE (Mean Absolute Error)
Average absolute difference between predictions and targets:
```
MAE = (1/n) * Σ|predicted - actual|
```

### RMSE (Root Mean Squared Error)
Square root of mean squared errors (penalizes large errors):
```
RMSE = sqrt((1/n) * Σ(predicted - actual)²)
```

### R² (Coefficient of Determination)
Proportion of variance explained (1.0 = perfect):
```
R² = 1 - (SS_residual / SS_total)
```

### Calibration
How well predicted confidence matches actual accuracy:
- Perfect calibration: 70% confidence → 70% correct
- Measured via calibration curves and Brier score

## Examples

### Well-Defined Orange Album Concept
```
"I remember the toy train set my grandfather built, 
with its miniature stations that existed only in our 
shared imagination."

Scores:
- Temporal: past=0.92, present=0.05, future=0.03
- Spatial: thing=0.88, place=0.09, person=0.03
- Ontological: imagined=0.85, forgotten=0.10, known=0.05
- Confidence: 0.91

Status: ACCEPT
Album: Orange (Past_Thing_Imagined)
```

### Hybrid Temporal Concept
```
"The photograph bridges memory and present moment,
simultaneously past and now."

Scores:
- Temporal: past=0.48, present=0.47, future=0.05 (HYBRID)
- Spatial: thing=0.78, place=0.15, person=0.07
- Ontological: known=0.82, forgotten=0.12, imagined=0.06
- Confidence: 0.62

Status: ACCEPT_HYBRID
Flags: ["temporal_hybrid"]
Suggested Album: Red or Violet
```

### Black Album Candidate
```
"∅ ⟷ ∞ ⟷ ∅"

Scores:
- Temporal: past=0.34, present=0.33, future=0.33 (DIFFUSE)
- Spatial: thing=0.32, place=0.35, person=0.33 (DIFFUSE)
- Ontological: imagined=0.33, forgotten=0.34, known=0.33 (DIFFUSE)
- Confidence: 0.18

Status: ACCEPT_BLACK
Album: Black (None_None_None)
```

---

*For detailed specifications, see spec.md*
*For implementation tasks, see tasks.md*
*For usage examples, see usage_examples.md*
