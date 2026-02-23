# Regression Tasks - Rainbow Table Extensions

This OpenSpec change extends the base regression tasks specification with Rainbow Table-specific capabilities for ontological mode prediction, concept validation, and transmigration analysis.

## Overview

The regression system provides two complementary capabilities:

1. **Generic Regression** (Base Spec): Continuous metrics like intensity, fluidity, temporal complexity
2. **Rainbow Table Ontological Regression** (Extensions): Temporal/Spatial/Ontological mode distributions for the White Album framework

## Rainbow Table Ontological Framework

The Rainbow Table defines concept ontology through three dimensions:

- **Temporal**: Past, Present, Future
- **Spatial**: Thing, Place, Person  
- **Ontological**: Imagined, Forgotten, Known

Each dimension has 3 modes, creating 3³ = 27 possible combined states (e.g., Past_Thing_Imagined = Orange Album).

## Key Innovations

### 1. Continuous Mode Distributions
Instead of hard classification, predict probability distributions:
- `temporal_scores: [past, present, future]` with softmax
- `spatial_scores: [thing, place, person]` with softmax
- `ontological_scores: [imagined, forgotten, known]` with softmax
- `chromatic_confidence: [0-1]` with sigmoid

**Benefits:**
- Captures hybrid/liminal concepts spanning multiple modes
- Quantifies prediction uncertainty
- Enables smooth transmigration between modes

### 2. Hybrid State Detection
Automatically identifies concepts that straddle ontological boundaries:
- **Hybrid**: Top two scores within 0.15 (e.g., equally Past/Present)
- **Dominant**: Top score > 0.6 (clear mode assignment)
- **Diffuse**: All scores ≈0.33 (no clear mode, potential Black Album)

### 3. Transmigration Distance
Measures conceptual distance between ontological states:
```python
distance(Past_Thing_Imagined → Future_Person_Known) = 
  sqrt(temporal_distance² + spatial_distance² + ontological_distance²)
```

Used for:
- Feasibility assessment (is transformation possible?)
- Path planning (suggest intermediate states)
- Edit guidance (which dimension to change first)

### 4. White Agent Validation Gates
Automated accept/reject for generated concepts:
- **ACCEPT**: High confidence (>0.7), clear mode assignment
- **ACCEPT_HYBRID**: Liminal but coherent (2D hybrid max)
- **ACCEPT_BLACK**: All dimensions diffuse (None_None_None)
- **REJECT**: Low confidence or >2 diffuse dimensions

Provides actionable suggestions: "Increase past_score from 0.55 to 0.70"

### 5. Soft Target Generation
Converts discrete labels to continuous training targets:
```python
# One-hot baseline
"Past" → [1.0, 0.0, 0.0]

# Label smoothing (prevent overconfidence)
"Past" → [0.9, 0.05, 0.05]

# Temporal context (account for transitions)
prev=Past, curr=Present, next=Past → curr=[0.3, 0.4, 0.3]

# Black Album (uniform diffusion)
"None" → [0.33, 0.33, 0.33]
```

## Architecture

```
┌─────────────────┐
│ Shared Encoder  │ (BERT/DeBERTa)
└────────┬────────┘
         │
    ┌────┴─────┬─────────────┬──────────────┐
    │          │             │              │
┌───▼───┐  ┌──▼───┐     ┌───▼────┐    ┌───▼────┐
│Class. │  │Temp. │     │Spatial │    │Ontol.  │
│ Head  │  │Regr. │     │Regr.   │    │Regr.   │
│       │  │(3 sf)│     │(3 sf)  │    │(3 sf)  │
└───────┘  └──────┘     └────────┘    └────────┘
                                       ┌────────┐
                                       │Conf.   │
                                       │(1 sig) │
                                       └────────┘
```

- **Classification Head**: Discrete album prediction (baseline)
- **Temporal Regression**: [past, present, future] softmax
- **Spatial Regression**: [thing, place, person] softmax
- **Ontological Regression**: [imagined, forgotten, known] softmax
- **Confidence Regression**: chromatic_confidence sigmoid

Total: 10 continuous outputs + discrete classification

## Multi-Task Loss

```python
total_loss = (
    1.0 * classification_loss +
    0.8 * temporal_regression_loss +
    0.8 * spatial_regression_loss +
    0.8 * ontological_regression_loss +
    0.5 * confidence_regression_loss
)
```

Weights tuned to balance discrete/continuous learning.

## Training Data Pipeline

1. **Load Discrete Labels**: From Rainbow Table annotations
2. **Generate Soft Targets**: One-hot → label smoothing → context smoothing
3. **Validate Targets**: Check sums to 1.0, flag inconsistencies
4. **Multi-Task Batching**: Pair discrete labels with continuous targets
5. **Weighted Loss**: Reduce weight for uncertain annotations

## Validation Pipeline

```
┌─────────────┐
│ White Agent │ generates concept
└──────┬──────┘
       │
       ▼
┌──────────────────┐
│ Feature Extract  │ (tokenize, embed)
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│ Regression Model │ predicts 10 scores
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│ Hybrid Detection │ flags liminal states
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│ Validation Gates │ ACCEPT/REJECT + suggestions
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│ Return Result    │ to White Agent workflow
└──────────────────┘
```

## Integration Points

### LangGraph Workflow
```python
workflow.add_node("validate_concept", validate_concept_node)
workflow.add_conditional_edges(
    "validate_concept",
    lambda s: "proceed" if s.validation_passed else "regenerate"
)
```

### FastAPI Service
```bash
POST /validate
{
  "text": "A memory of building model trains..."
}

→ 200 OK
{
  "predicted_album": "Orange",
  "validation_status": "ACCEPT",
  "chromatic_confidence": 0.91,
  ...
}
```

### Caching Layer
- 1-hour TTL for validation results
- Key: SHA256(concept_text + model_version)
- Reduces latency for repeated validation

## Metrics to Track

### Training
- Per-dimension MAE, RMSE, R² (temporal, spatial, ontological)
- Confidence calibration (predicted vs actual uncertainty)
- Album classification accuracy from continuous scores
- Hybrid detection precision/recall

### Production
- Validation accept/reject rates
- Average confidence scores
- Hybrid concept frequency
- Album distribution of generated concepts
- Regeneration loop count
- Validation latency (p50, p95, p99)

## Success Criteria

1. Album prediction from continuous scores matches discrete classification **>95%**
2. Confidence correlates with actual accuracy (R² **>0.8**)
3. Hybrid detection aligns with human annotators
4. Transmigration distances align with human perception
5. Validation gates align with human judgment **>90%**
6. Single concept validation **<200ms**
7. Reduces low-quality White Agent concepts **>50%**

## Files

- `spec.md` - Requirements specification (base + Rainbow Table extensions)
- `tasks.md` - Implementation tasks (15 sections)
- `proposal.md` - Change rationale and impact
- `config_example.yml` - Configuration schema for Rainbow Table regression
- `usage_examples.md` - Code examples and integration patterns
- `README.md` - This file

## Next Steps

1. Implement sections 10-15 in tasks.md (White Agent integration)
2. Generate soft targets from existing Rainbow Table annotations
3. Train multi-task model with 10 regression outputs
4. Deploy validation API and integrate with White Agent workflow
5. Run A/B test comparing gated vs ungated concept generation
6. Tune validation thresholds based on human feedback

## References

- [Phase 4 White Album Extensions](../../../claude_working_area/phase4_white_album_extensions.md)
- [Rainbow Table Ontology](../../specs/rainbow-table/)
- [White Agent Workflow](../../../app/agents/white_agent.py)
