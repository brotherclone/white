"""
FastAPI validation endpoint for White Agent integration.

Provides real-time concept validation via HTTP API.
Supports single and batch validation with caching.
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
import time

# FastAPI imports (with fallback for environments without it)
try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware

    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

from .concept_validator import (
    ConceptValidator,
    ValidationResult,
)


# Pydantic models for API
class ConceptRequest(BaseModel):
    """Request model for concept validation."""

    text: str = Field(
        ..., min_length=1, max_length=10000, description="Concept text to validate"
    )


class BatchConceptRequest(BaseModel):
    """Request model for batch concept validation."""

    texts: List[str] = Field(
        ..., min_items=1, max_items=100, description="List of concept texts"
    )


class ScoresResponse(BaseModel):
    """Response model for dimension scores."""

    past: float
    present: float
    future: float


class SpatialScoresResponse(BaseModel):
    """Response model for spatial scores."""

    thing: float
    place: float
    person: float


class OntologicalScoresResponse(BaseModel):
    """Response model for ontological scores."""

    imagined: float
    forgotten: float
    known: float


class SuggestionResponse(BaseModel):
    """Response model for validation suggestion."""

    dimension: str
    current_value: float
    target_value: float
    message: str


class ValidationResponse(BaseModel):
    """Response model for validation result."""

    temporal_scores: Dict[str, float]
    spatial_scores: Dict[str, float]
    ontological_scores: Dict[str, float]
    chromatic_confidence: float
    predicted_album: str
    predicted_mode: str
    validation_status: str
    hybrid_flags: List[str]
    rejection_reason: Optional[str] = None
    transmigration_distances: Optional[Dict[str, float]] = None
    suggestions: List[SuggestionResponse] = []
    validation_time_ms: Optional[float] = None
    cache_hit: bool = False


class BatchValidationResponse(BaseModel):
    """Response model for batch validation."""

    results: List[ValidationResponse]
    total_time_ms: float
    count: int


class HealthResponse(BaseModel):
    """Response model for health check."""

    status: str
    model_loaded: bool
    cache_enabled: bool
    version: str


def create_app(
    model_path: Optional[str] = None,
    config_path: Optional[str] = None,
    enable_cache: bool = True,
    cache_ttl: int = 3600,
    cors_origins: List[str] = ["*"],
) -> "FastAPI":
    """
    Create FastAPI application for concept validation.

    Args:
        model_path: Path to trained model checkpoint
        config_path: Path to model configuration
        enable_cache: Enable validation result caching
        cache_ttl: Cache time-to-live in seconds
        cors_origins: Allowed CORS origins

    Returns:
        FastAPI application instance
    """
    if not HAS_FASTAPI:
        raise ImportError(
            "FastAPI not installed. Install with: pip install fastapi uvicorn"
        )

    app = FastAPI(
        title="Rainbow Table Concept Validator",
        description="Validates concepts for ontological coherence using the Rainbow Table framework",
        version="1.0.0",
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Initialize validator as app state
    @app.on_event("startup")
    async def startup():
        app.state.validator = ConceptValidator(
            model_path=model_path,
            config_path=config_path,
            enable_cache=enable_cache,
            cache_ttl=cache_ttl,
        )

    def _result_to_response(result: ValidationResult) -> ValidationResponse:
        """Convert ValidationResult to response model."""
        return ValidationResponse(
            temporal_scores=result.temporal_scores,
            spatial_scores=result.spatial_scores,
            ontological_scores=result.ontological_scores,
            chromatic_confidence=result.chromatic_confidence,
            predicted_album=result.predicted_album,
            predicted_mode=result.predicted_mode,
            validation_status=result.validation_status.value,
            hybrid_flags=result.hybrid_flags,
            rejection_reason=result.rejection_reason,
            transmigration_distances=result.transmigration_distances,
            suggestions=[
                SuggestionResponse(
                    dimension=s.dimension,
                    current_value=s.current_value,
                    target_value=s.target_value,
                    message=s.message,
                )
                for s in result.suggestions
            ],
            validation_time_ms=result.validation_time_ms,
            cache_hit=result.cache_hit,
        )

    @app.get("/health", response_model=HealthResponse, tags=["Health"])
    async def health_check():
        """Health check endpoint."""
        validator = app.state.validator
        return HealthResponse(
            status="healthy",
            model_loaded=validator._model is not None,
            cache_enabled=validator.cache is not None,
            version="1.0.0",
        )

    @app.post("/validate", response_model=ValidationResponse, tags=["Validation"])
    async def validate_concept(request: ConceptRequest):
        """
        Validate a single concept.

        Returns ontological scores, album prediction, and validation status.
        """
        try:
            result = app.state.validator.validate_concept(request.text)
            return _result_to_response(result)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post(
        "/validate/batch", response_model=BatchValidationResponse, tags=["Validation"]
    )
    async def validate_batch(request: BatchConceptRequest):
        """
        Validate multiple concepts in batch.

        More efficient than individual requests for multiple concepts.
        """
        start_time = time.time()

        try:
            results = app.state.validator.validate_batch(request.texts)
            responses = [_result_to_response(r) for r in results]

            return BatchValidationResponse(
                results=responses,
                total_time_ms=(time.time() - start_time) * 1000,
                count=len(results),
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/validate/quick", tags=["Validation"])
    async def validate_quick(request: ConceptRequest) -> Dict[str, Any]:
        """
        Quick validation returning only essential fields.

        Faster response for simple accept/reject decisions.
        """
        try:
            result = app.state.validator.validate_concept(request.text)
            return {
                "status": result.validation_status.value,
                "album": result.predicted_album,
                "confidence": result.chromatic_confidence,
                "accepted": result.is_accepted,
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.delete("/cache", tags=["Admin"])
    async def clear_cache():
        """Clear the validation cache."""
        if app.state.validator.cache:
            app.state.validator.cache.clear()
            return {"status": "cache_cleared"}
        return {"status": "cache_disabled"}

    @app.get("/thresholds", tags=["Config"])
    async def get_thresholds():
        """Get current validation thresholds."""
        validator = app.state.validator
        return {
            "confidence_threshold": validator.confidence_threshold,
            "dominant_threshold": validator.dominant_threshold,
            "hybrid_threshold": validator.hybrid_threshold,
            "diffuse_threshold": validator.diffuse_threshold,
            "uncertainty_threshold": validator.uncertainty_threshold,
        }

    return app


# Default app instance for uvicorn
def get_default_app():
    """Get default app instance for running with uvicorn."""
    import os

    return create_app(
        model_path=os.environ.get("MODEL_PATH"),
        config_path=os.environ.get("CONFIG_PATH"),
        enable_cache=os.environ.get("ENABLE_CACHE", "true").lower() == "true",
        cache_ttl=int(os.environ.get("CACHE_TTL", "3600")),
    )


# For running with: uvicorn validation.api:app
if HAS_FASTAPI:
    app = get_default_app()


if __name__ == "__main__":
    if not HAS_FASTAPI:
        print("FastAPI not installed. Install with: pip install fastapi uvicorn")
        exit(1)

    import uvicorn

    # Create app with mock model for testing
    test_app = create_app(model_path=None)

    print("Starting validation API server...")
    print("API docs available at: http://localhost:8000/docs")

    uvicorn.run(test_app, host="0.0.0.0", port=8000)
