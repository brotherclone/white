# White Agent Integration - Usage Examples

## Basic Concept Validation

```python
from training.validation.concept_validator import ConceptValidator

# Initialize validator
validator = ConceptValidator(
    model_path="checkpoints/phase4_regression_best.pt",
    config_path="config_regression.yml"
)

# Validate White Agent concept
concept = """
A memory of building model trains in my father's basement, 
where the clicking of the tracks became a rhythm I still hear 
in my sleep. The trains ran on schedules that existed only 
in my imagination, delivering cargo to cities that never were.
"""

result = validator.validate_concept(concept)

# Check validation status
if result.validation_status == "ACCEPT":
    print(f"✅ Concept accepted for {result.predicted_album} Album")
    print(f"   Mode: {result.predicted_mode}")
    print(f"   Confidence: {result.chromatic_confidence:.2f}")
    
elif result.validation_status == "ACCEPT_HYBRID":
    print(f"⚠️  Hybrid concept detected")
    print(f"   Hybrid flags: {result.hybrid_flags}")
    print(f"   Suggested album: {result.predicted_album}")
    
elif result.validation_status == "ACCEPT_BLACK":
    print(f"🖤 Concept fits Black Album (None_None_None)")
    
else:  # REJECT
    print(f"❌ Concept rejected")
    print(f"   Reason: {result.rejection_reason}")
    print(f"   Suggestions:")
    for suggestion in result.suggestions:
        print(f"   - {suggestion}")
```

## Detailed Ontological Analysis

```python
# Detailed scores
print(f"\n📊 Ontological Scores:")
print(f"   Temporal: Past={result.temporal_scores['past']:.2f}, "
      f"Present={result.temporal_scores['present']:.2f}, "
      f"Future={result.temporal_scores['future']:.2f}")
print(f"   Spatial: Thing={result.spatial_scores['thing']:.2f}, "
      f"Place={result.spatial_scores['place']:.2f}, "
      f"Person={result.spatial_scores['person']:.2f}")
print(f"   Ontological: Imagined={result.ontological_scores['imagined']:.2f}, "
      f"Forgotten={result.ontological_scores['forgotten']:.2f}, "
      f"Known={result.ontological_scores['known']:.2f}")

# Transmigration analysis
print(f"\n🔄 Transmigration Distances:")
for album, distance in result.transmigration_distances.items():
    print(f"   {album}: {distance:.2f}")
```

## LangGraph Workflow Integration

```python
from langgraph.graph import StateGraph
from training.validation.concept_validator import ConceptValidator

# Add validation node to workflow
def validate_concept_node(state):
    """Validate White Agent concept before proceeding"""
    validator = ConceptValidator.get_instance()
    result = validator.validate_concept(state.white_concept)
    
    state.validation_result = result
    
    if result.validation_status in ["ACCEPT", "ACCEPT_HYBRID", "ACCEPT_BLACK"]:
        state.validation_passed = True
        state.target_album = result.predicted_album
    else:
        state.validation_passed = False
        state.regeneration_suggestions = result.suggestions
    
    return state

# Build graph with validation
workflow = StateGraph(WhiteAgentState)
workflow.add_node("white_agent", white_agent_node)
workflow.add_node("validate_concept", validate_concept_node)
workflow.add_node("black_agent", black_agent_node)
workflow.add_node("regenerate", regenerate_concept_node)

# Conditional edges based on validation
workflow.add_conditional_edges(
    "validate_concept",
    lambda s: "black_agent" if s.validation_passed else "regenerate"
)
```

## Batch Validation

```python
# Validate multiple concepts efficiently
concepts = [
    "A forgotten melody...",
    "The future shape of memory...",
    "Here, now, in this moment..."
]

results = validator.validate_batch(concepts)

for concept, result in zip(concepts, results):
    print(f"\nConcept: {concept[:50]}...")
    print(f"Album: {result.predicted_album}")
    print(f"Status: {result.validation_status}")
```

## FastAPI Endpoint

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class ConceptRequest(BaseModel):
    text: str

@app.post("/validate")
async def validate_concept_endpoint(request: ConceptRequest):
    """Real-time concept validation endpoint"""
    validator = ConceptValidator.get_instance()
    result = validator.validate_concept(request.text)
    
    return {
        "temporal_scores": result.temporal_scores,
        "spatial_scores": result.spatial_scores,
        "ontological_scores": result.ontological_scores,
        "chromatic_confidence": result.chromatic_confidence,
        "predicted_album": result.predicted_album,
        "predicted_mode": result.predicted_mode,
        "validation_status": result.validation_status,
        "hybrid_flags": result.hybrid_flags,
        "transmigration_distances": result.transmigration_distances,
        "suggestions": result.suggestions
    }
```

## Example Validation Result JSON

```json
{
  "temporal_scores": {
    "past": 0.87,
    "present": 0.12,
    "future": 0.01
  },
  "spatial_scores": {
    "thing": 0.92,
    "place": 0.05,
    "person": 0.03
  },
  "ontological_scores": {
    "imagined": 0.78,
    "forgotten": 0.15,
    "known": 0.07
  },
  "chromatic_confidence": 0.91,
  "predicted_album": "Orange",
  "predicted_mode": "Past_Thing_Imagined",
  "validation_status": "ACCEPT",
  "hybrid_flags": [],
  "uncertainty_estimates": {
    "temporal": 0.12,
    "spatial": 0.08,
    "ontological": 0.19
  },
  "transmigration_distances": {
    "Orange": 0.15,
    "Red": 0.45,
    "Yellow": 2.31,
    "Green": 2.56,
    "Blue": 2.18,
    "Indigo": 1.87,
    "Black": 1.92
  },
  "suggestions": [
    "Concept strongly fits Orange Album (Past_Thing_Imagined)",
    "High confidence in temporal (past) and spatial (thing) dimensions",
    "Clear imagined ontology appropriate for memory-based narrative"
  ],
  "rejection_reason": null
}
```

## Transmigration Planning

```python
from training.validation.transmigration import TransmigrationPlanner

planner = TransmigrationPlanner()

# Plan transmigration from Orange to Blue
source_mode = "Past_Thing_Imagined"
target_mode = "Future_Person_Known"

plan = planner.create_plan(source_mode, target_mode)

print(f"Transmigration Distance: {plan.total_distance:.2f}")
print(f"Difficulty: {plan.difficulty}")  # "easy", "moderate", "difficult"
print(f"\nDimension Priorities:")
for dim in plan.dimension_priorities:
    print(f"  {dim['dimension']}: {dim['required_change']:.2f}")

if plan.intermediate_states:
    print(f"\nSuggested Path:")
    print(f"  {source_mode}")
    for state in plan.intermediate_states:
        print(f"  → {state}")
    print(f"  → {target_mode}")
```

## Success Metrics

Phase 4 regression model is successful when:

1. **Accuracy**: Album prediction from continuous scores matches discrete classification >95%
2. **Calibration**: Confidence scores correlate with actual prediction accuracy (R² > 0.8)
3. **Hybrid Detection**: Correctly identifies liminal concepts that human annotators agree are hybrid
4. **Transmigration**: Distance metrics align with human perception of mode similarity
5. **Validation Gates**: Accept/reject decisions align with human judgment >90%
6. **Latency**: Single concept validation completes in <200ms
7. **Integration**: Successfully gates White Agent workflow, reducing low-quality concepts by >50%
