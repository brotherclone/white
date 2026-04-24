"""
Validation module for Rainbow Table concept validation.

Provides:
- ConceptValidator: Validates concepts using regression model
- ValidationResult: Structured validation output
- Validation gates for White Agent integration
"""

from .concept_validator import (
    ConceptValidator,
    ValidationResult,
    ValidationStatus,
    ValidationSuggestion,
)

__all__ = [
    "ConceptValidator",
    "ValidationResult",
    "ValidationStatus",
    "ValidationSuggestion",
]
