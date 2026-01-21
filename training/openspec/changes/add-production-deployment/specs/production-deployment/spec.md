# Production Deployment

## ADDED Requirements

### Requirement: ONNX Model Export
The system SHALL export trained PyTorch models to ONNX format for optimized CPU inference.

#### Scenario: PyTorch to ONNX conversion
- **WHEN** exporting a trained model
- **THEN** ONNX format is generated with correct input/output shapes

#### Scenario: Optimization passes
- **WHEN** ONNX export includes optimization
- **THEN** graph optimizations reduce inference latency

#### Scenario: Numerical equivalence verification
- **WHEN** ONNX export completes
- **THEN** outputs are verified to match PyTorch within tolerance

### Requirement: Inference API Endpoints
The system SHALL provide RESTful API endpoints for model inference.

#### Scenario: Single prediction endpoint
- **WHEN** POST /predict is called with segment data
- **THEN** rebracketing predictions are returned as JSON

#### Scenario: Batch prediction endpoint
- **WHEN** POST /predict_batch is called with multiple segments
- **THEN** predictions for all segments are returned efficiently

#### Scenario: Generation endpoint
- **WHEN** POST /generate is called with conditioning parameters
- **THEN** a generated segment is returned

#### Scenario: Health check endpoint
- **WHEN** GET /health is called
- **THEN** API status and model readiness are returned

### Requirement: Request Validation
The system SHALL validate API requests using Pydantic schemas.

#### Scenario: Schema validation
- **WHEN** a request is received
- **THEN** Pydantic validates structure and types

#### Scenario: Error responses
- **WHEN** invalid data is provided
- **THEN** 422 Unprocessable Entity with error details is returned

### Requirement: LangGraph Tool Integration
The system SHALL provide a tool for LangGraph agents to invoke rebracketing analysis.

#### Scenario: Tool registration
- **WHEN** White Agent initializes
- **THEN** RebracketingAnalyzerTool is available in the tool list

#### Scenario: Tool invocation
- **WHEN** an agent invokes the tool with concept text
- **THEN** rebracketing analysis predictions are returned

#### Scenario: Structured output
- **WHEN** the tool returns results
- **THEN** predictions include type, intensity, confidence

### Requirement: Streaming Analysis
The system SHALL analyze audio streams in real-time using sliding windows.

#### Scenario: Sliding window processing
- **WHEN** audio stream is received
- **THEN** windows are processed with configurable overlap

#### Scenario: Low-latency inference
- **WHEN** streaming analysis runs
- **THEN** predictions are returned with minimal delay

### Requirement: Model Versioning and Registry
The system SHALL manage multiple model versions and enable loading by version.

#### Scenario: Register model version
- **WHEN** a model is trained
- **THEN** it is registered with version tag and metadata

#### Scenario: Load by version
- **WHEN** API starts with specified model version
- **THEN** that version is loaded for inference

#### Scenario: A/B testing
- **WHEN** A/B testing is enabled
- **THEN** traffic is split between model versions

### Requirement: Inference Monitoring
The system SHALL monitor inference performance and prediction distributions.

#### Scenario: Latency logging
- **WHEN** predictions are made
- **THEN** inference latency is logged

#### Scenario: Prediction distribution monitoring
- **WHEN** predictions accumulate
- **THEN** distribution over rebracketing types is tracked

#### Scenario: Error tracking
- **WHEN** inference errors occur
- **THEN** errors are logged with context for debugging

### Requirement: Authentication and Rate Limiting
The system SHALL secure API endpoints with authentication and rate limiting.

#### Scenario: API key authentication
- **WHEN** requests are made
- **THEN** valid API key is required

#### Scenario: Rate limiting
- **WHEN** too many requests are made
- **THEN** 429 Too Many Requests is returned

### Requirement: Batch Inference Optimization
The system SHALL optimize inference for batched requests.

#### Scenario: Dynamic batching
- **WHEN** multiple requests arrive concurrently
- **THEN** they are batched for efficient GPU utilization

#### Scenario: Batch size configuration
- **WHEN** batch inference is configured
- **THEN** maximum batch size is enforced

### Requirement: Deployment Configuration
The system SHALL provide configuration for production deployment.

#### Scenario: API host and port
- **WHEN** config.deployment.api.host is "0.0.0.0" and port is 8000
- **THEN** API listens on those settings

#### Scenario: Model path
- **WHEN** config.deployment.model_path is specified
- **THEN** model is loaded from that path

#### Scenario: Worker processes
- **WHEN** config.deployment.workers is 4
- **THEN** API runs with 4 worker processes
