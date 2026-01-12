# Change: Add Production Deployment and Inference API

## Why
Trained models must be deployed for use by White Agent and other components. Production deployment requires model export, optimization, API endpoints, and integration with LangGraph agents.

## What Changes
- Add model export to ONNX for CPU inference optimization
- Create `RebracketingInferenceAPI` (Flask/FastAPI) for prediction endpoints
- Add `RebracketingAnalyzerTool` for LangGraph agent integration
- Implement streaming analysis for real-time segment processing
- Add model versioning and registry integration
- Implement batch inference optimization
- Add monitoring and logging for production inference

## Impact
- Affected specs: production-deployment (new capability)
- Affected code:
  - `training/export/` - ONNX export and optimization
  - `training/api/` - inference API implementation
  - `training/tools/` - LangGraph tool integration
  - Main project `app/tools/` - White Agent tool registration
- Dependencies: onnx, onnxruntime, fastapi, pydantic
