# Change: Add Production Deployment and Chromatic Scoring API

## Why
The trained multimodal model serves as the fitness function for the Evolutionary Music Generator. Production deployment requires model export, optimization, and a scoring API that can evaluate batches of MIDI/audio candidates quickly enough to support multi-stage evolutionary composition (50+ candidates per stage, multiple stages per composition).

## What Changes
- Add model export to ONNX for optimized inference
- Create `ChromaticScoringAPI` (FastAPI) for fitness evaluation endpoints
- Add `ChromaticScorerTool` for LangGraph agent integration with the Evolutionary Music Generator
- Implement batch scoring optimization for evaluating candidate populations (50+ per call)
- Add model versioning and registry integration
- Implement monitoring and logging for production inference
- Add MIDI rendering pipeline for scoring MIDI-only candidates (render to audio for multimodal model)

## Impact
- Affected specs: production-deployment (new capability)
- Affected code:
  - `training/export/` - ONNX export and optimization
  - `training/api/` - ChromaticScoringAPI implementation
  - `training/tools/` - LangGraph tool integration
  - `app/generator/` - Evolutionary Music Generator consumes scoring API
- Dependencies: onnx, onnxruntime, fastapi, pydantic, fluidsynth (MIDI rendering)
