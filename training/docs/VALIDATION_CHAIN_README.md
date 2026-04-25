# Phase 4 Concept Validation - Standalone Chain

**IMPORTANT:** This validation chain is **completely separate** from your main White Agent workflow.

## Architecture

```
LOCAL WORKFLOW (Intel Mac, no torch)
────────────────────────────────────
White Agent generates concepts → saved to /chain_artifacts/
    ↓
    (continues to Black Agent, Red Agent, etc.)
    ↓
    (NO validation, NO torch dependency)


VALIDATION CHAIN (runs separately in /training)
────────────────────────────────────
Later, optionally:
    cd training/
    python validate_concepts.py --recent 10
    ↓
    Reviews concepts after generation
    ↓
    Outputs validation report
```

## Key Points

✅ **Local workflow stays clean** - No torch, no ML, Intel Mac compatible
✅ **Validation is optional** - Run it when you want, or not at all
✅ **Batch processing** - Validate multiple concepts at once
✅ **No blocking** - Doesn't stop concept generation

## Usage

### 1. Generate Concepts (as normal)

```bash
# In your main White Album directory
python run_ideation.py

# Concepts saved to /chain_artifacts/<thread-id>/
```

### 2. Later: Validate Concepts (optional)

```bash
# Move to training directory
cd training/

# Validate recent concepts
python validate_concepts.py --recent 10

# Validate specific thread
python validate_concepts.py --thread-id f410b5f7-abc-def

# Validate from directory
python validate_concepts.py --concepts-dir /chain_artifacts/

# Validate single file
python validate_concepts.py --concept-file /path/to/concept.yml

# Save results
python validate_concepts.py --recent 10 --output-dir validation_results/
```

### 3. Review Results

```bash
# Check validation summary (printed to console)
# Or view saved JSON:
cat validation_results/validation_results.json
```

## Output Example

```
🔍 Validating 10 concepts...

[1/10] white_concept_001.yml
  ✅ ACCEPT
  Album: Orange
  Confidence: 0.91
  
[2/10] white_concept_002.yml
  ⚠️  ACCEPT_HYBRID
  Album: Yellow
  Confidence: 0.67
  Flags: temporal_hybrid, spatial_dominant_place

[3/10] white_concept_003.yml
  ❌ REJECT
  Album: Unknown
  Confidence: 0.32
  
════════════════════════════════════════════════════════════
VALIDATION SUMMARY
════════════════════════════════════════════════════════════
  ACCEPT: 6 (60.0%)
  ACCEPT_HYBRID: 3 (30.0%)
  REJECT: 1 (10.0%)

Total: 10 concepts validated
════════════════════════════════════════════════════════════
```

## When to Use Validation

**Don't use it if:**
- Your concepts are already high quality
- You're in rapid iteration mode
- You just want to keep generating

**Use it when:**
- You've generated 50+ concepts and want to review quality
- You want to identify patterns in what works
- You're curious about ontological distributions
- You want to filter concepts before manual review

## Requirements

Only needed in `/training` directory:

```bash
pip install torch numpy pyyaml
```

**NOT needed in main workflow!**

## Configuration

Adjust validation thresholds:

```bash
python validate_concepts.py --recent 10 \
  --confidence-threshold 0.7 \
  --dominant-threshold 0.6 \
  --hybrid-threshold 0.15 \
  --diffuse-threshold 0.2
```

## Model

By default uses: `output/phase4_best.pt`

After training Phase 4 on RunPod, download the model to:
```
/Volumes/LucidNonsense/White/training/output/phase4_best.pt
```

If model not found, will use random weights (for testing structure only).

## Workflow Integration (Optional)

If you want to incorporate validation results:

1. Run validation chain after batch generation
2. Review `validation_results.json`
3. Manually select ACCEPT/ACCEPT_HYBRID concepts
4. Feed those back into your workflow
5. Regenerate REJECT concepts if needed

**But this is all manual and optional!**

## What This Doesn't Do

❌ Block your local workflow
❌ Require changes to White Agent
❌ Need to run in real-time
❌ Require torch on Intel Mac

## What This Does Do

✅ Batch validate concepts after generation
✅ Provide ontological analysis
✅ Identify quality issues
✅ Generate validation reports
✅ Run completely separately

## Questions

**Q: Do I need to run this?**
A: No! It's optional quality control.

**Q: When should I run it?**
A: After generating 10-50 concepts, or whenever you want insights.

**Q: Will it break my workflow?**
A: No - it's completely separate.

**Q: What if I don't have the trained model?**
A: It runs with random weights (for testing). Train Phase 4 on RunPod to get real validation.

**Q: Can I run this on my Intel Mac?**
A: Yes, but it requires torch. Use `device='cpu'` and it'll be slow but functional.

---

**TL;DR:** Your local workflow stays clean. Validation is a separate optional tool in /training.
