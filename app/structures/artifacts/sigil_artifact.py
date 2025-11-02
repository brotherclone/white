# python
from typing import List, Optional, Any, Dict
import re

from pydantic import Field, model_validator

from app.structures.enums.sigil_state import SigilState
from app.structures.enums.sigil_type import SigilType
from app.structures.artifacts.base_chain_artifact import ChainArtifact
from app.structures.artifacts.text_chain_artifact_file import TextChainArtifactFile


class SigilArtifact(ChainArtifact):
    """Record of a created sigil for the Black Agent's paranoid tracking"""

    thread_id: Optional[str] = None
    wish: Optional[str] = None
    statement_of_intent: Optional[str] = None
    sigil_type: Optional[SigilType] = None
    glyph_description: Optional[str] = None
    glyph_components: List[str] = Field(default_factory=list)
    activation_state: Optional[SigilState] = None
    charging_instructions: Optional[str] = None
    chain_artifact_type: str = "sigil"
    artifact_report: Optional[TextChainArtifactFile] = None

    @model_validator(mode="before")
    def _normalize_artifact_report(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        # Keys that should be treated as artifact-report-like after normalization
        def _normalize_key(k: str) -> str:
            return re.sub(r'[^a-z0-9]', '', k.lower())

        def _is_report_key(k: str) -> bool:
            s = _normalize_key(k)
            # match keys like "artifact_report", "artifact-report-file", "artifactReport", "artifactfile", "report"
            return (
                ("artifact" in s and ("report" in s or "file" in s or "text" in s))
                or s == "artifact"
                or s == "report"
            )

        # Keys that indicate a dict is likely a BaseChainArtifactFile / TextChainArtifactFile
        _expected_file_keys = {"artifact_id", "base_path", "chain_artifact_file_type", "artifact_name", "file_name", "text_content"}

        def _is_valid_report_candidate(obj: Any) -> bool:
            # Only dict-like objects with at least one expected key are valid candidates
            if not isinstance(obj, dict):
                return False
            return any(k in obj for k in _expected_file_keys)

        def _find_in_structure(obj: Any) -> Optional[Any]:
            if isinstance(obj, dict):
                # First, check direct keys on this dict for report-like keys that map to dicts
                for k, v in obj.items():
                    try:
                        if k and _is_report_key(k) and v is not None and _is_valid_report_candidate(v):
                            return v
                    except Exception:
                        pass
                # Next, check if this dict itself looks like a report candidate
                if _is_valid_report_candidate(obj):
                    return obj
                # Recurse into children
                for k, v in obj.items():
                    found = _find_in_structure(v)
                    if found is not None:
                        return found
            elif isinstance(obj, list):
                for item in obj:
                    found = _find_in_structure(item)
                    if found is not None:
                        return found
            return None

        if not isinstance(values, dict):
            return values

        # If artifact_report already present and non-null, leave it.
        if values.get("artifact_report") is None:
            found = _find_in_structure(values)
            if found is not None:
                values["artifact_report"] = found

        return values

    model_config = {
        "extra": "ignore"
    }