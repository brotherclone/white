# Phase 4 Architecture - Clean Separation

## The Two Worlds

```
┌─────────────────────────────────────────────────────────────┐
│  LOCAL WORKFLOW (Intel Mac)                                  │
│  /Volumes/LucidNonsense/White/                              │
│                                                               │
│  Dependencies: anthropic, langgraph, pydantic, pyyaml       │
│  NO torch, NO numpy, NO sklearn                             │
│                                                               │
│  ├── run_white_agent.py          ← Your main entry point    │
│  ├── app/                                                    │
│  │   ├── agents/                                            │
│  │   │   ├── white_agent.py      ← Generates concepts       │
│  │   │   ├── black_agent.py      ← EVP/sigils              │
│  │   │   ├── red_agent.py        ← Forbidden books         │
│  │   │   ├── orange_agent.py     ← Mythos stories          │
│  │   │   └── yellow_agent.py     ← Images                  │
│  │   └── models/                                            │
│  │       └── artifacts.py         ← Pydantic models only    │
│  └── chain_artifacts/             ← Generated concepts      │
│                                                               │
│  ✅ Works on Intel Mac                                       │
│  ✅ No ML dependencies                                       │
│  ✅ Fast, local execution                                    │
│  ✅ No blocking on external services                        │
└─────────────────────────────────────────────────────────────┘

                           │
                           │ Concepts saved to disk
                           ↓

┌─────────────────────────────────────────────────────────────┐
│  VALIDATION CHAIN (Separate, optional)                      │
│  /Volumes/LucidNonsense/White/training/                     │
│                                                               │
│  Dependencies: torch, transformers, numpy, sklearn          │
│  Requires: GPU (or slow CPU)                                │
│                                                               │
│  ├── validate_concepts.py         ← Standalone validator    │
│  ├── phase4_train_regression.py   ← Model training          │
│  ├── runpod_train_phase4.py      ← Cloud training          │
│  ├── output/                                                 │
│  │   └── phase4_best.pt           ← Trained model           │
│  └── validation_results/          ← Batch results           │
│                                                               │
│  ✅ Runs separately from workflow                            │
│  ✅ Batch processes concepts                                 │
│  ✅ Optional quality control                                 │
│  ✅ No impact on local workflow                             │
└─────────────────────────────────────────────────────────────┘
```

## Workflow Comparison

### Before (What I Mistakenly Suggested)

```python
# ❌ BAD: Putting torch in local workflow

# In app/agents/white_agent.py
import torch  # ← Breaks Intel Mac!
from concept_validator import ConceptValidator

def white_agent_node(state):
    concept = generate_concept()
    
    # Validation in critical path
    validator = ConceptValidator(model_path="phase4_best.pt")
    result = validator.validate_concept(concept)
    
    if result.validation_status == "REJECT":
        regenerate()  # Blocks workflow!
    
    return state

# Problems:
# - Requires torch locally
# - Blocks on ML inference
# - Can't run on Intel Mac
# - Slows down iteration
```

### After (Clean Separation)

```python
# ✅ GOOD: Keep workflow clean

# In app/agents/white_agent.py
# NO torch import!
# NO ML dependencies!

def white_agent_node(state):
    concept = generate_concept()
    state.white_concept = concept
    return state  # Just proceed!

# Later, separately in terminal:
# $ cd training/
# $ python validate_concepts.py --recent 10
# $ cat validation_results/validation_results.json
# $ # Review and decide what to do

# Benefits:
# - Local workflow works on Intel Mac
# - Fast, no ML overhead
# - Validation is optional
# - Can iterate quickly
```

## File Locations

```
/Volumes/LucidNonsense/White/
│
├── app/                          ← NO torch here
│   └── agents/
│       ├── white_agent.py        ← Pure Python
│       ├── black_agent.py        ← Pure Python
│       └── ...
│
├── training/                     ← torch OK here
│   ├── validate_concepts.py     ← NEW!
│   ├── phase4_train_regression.py
│   └── output/
│       └── phase4_best.pt
│
└── chain_artifacts/              ← Concepts saved here
    └── <thread-id>/
        └── white_concept.yml
```

## Usage Pattern

### Daily Development (No Validation)

```bash
# Just generate concepts
python run_white_agent.py

# Concepts flow through agents
# No validation, no torch, no ML
# Fast iteration
```

### Periodic Quality Check (With Validation)

```bash
# After generating 50 concepts:
cd training/
python validate_concepts.py --recent 50 --output-dir results/

# Review results
cat results/validation_results.json

# Optional: regenerate low-quality concepts
# Optional: analyze ontological patterns
# Optional: tune White Agent based on findings
```

## The Key Insight

**Validation doesn't need to be in the critical path.**

- Generate freely without ML overhead
- Validate in batches later
- Review and iterate
- No blocking, no slowdown

## Dependencies by Directory

```python
# /app/ requirements (local workflow)
anthropic
langgraph
pydantic
pyyaml
requests

# NO torch!
# NO transformers!
# NO sklearn!


# /training/ requirements (validation chain)
torch
transformers
numpy
pandas
scikit-learn
pyyaml

# These NEVER imported in /app/
```

## Summary

1. **Local workflow**: Clean, fast, Intel Mac compatible
2. **Validation chain**: Separate, optional, batch processing
3. **No mixing**: torch stays in /training, never in /app
4. **No blocking**: concepts generate freely
5. **Optional quality**: validate when you want insights

**This is the right architecture.** ✅
