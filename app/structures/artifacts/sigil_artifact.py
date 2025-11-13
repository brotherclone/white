# python
import re
from typing import Any, Dict, List, Optional

from pydantic import Field, model_validator

from app.structures.artifacts.base_artifact import ChainArtifact
from app.structures.artifacts.text_artifact_file import TextChainArtifactFile
from app.structures.enums.sigil_state import SigilState
from app.structures.enums.sigil_type import SigilType


class SigilArtifact(ChainArtifact):
    """Record of a created sigil for the Black Agent's paranoid tracking"""

    thread_id: Optional[str] = Field(
        default=None, description="Unique ID of the thread."
    )
    wish: Optional[str] = Field(default=None, description="Wish for the sigil.")
    statement_of_intent: Optional[str] = Field(
        default=None, description="Statement of intent for the sigil."
    )
    sigil_type: Optional[SigilType] = Field(
        default=None, description="Type of the sigil."
    )
    glyph_description: Optional[str] = Field(
        default=None, description="Description of the sigil's glyph."
    )
    glyph_components: List[str] = Field(default_factory=list)
    activation_state: Optional[SigilState] = Field(
        default=None, description="Activation state of the sigil."
    )
    charging_instructions: Optional[str] = Field(
        default=None, description="Instructions for charging the sigil."
    )
    chain_artifact_type: str = "sigil"
    artifact_report: Optional[TextChainArtifactFile] = Field(
        default=None, description="Report file associated with the sigil."
    )

    @model_validator(mode="before")
    def _normalize_artifact_report(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        def _normalize_key(k: str) -> str:
            return re.sub(r"[^a-z0-9]", "", k.lower())

        def _is_report_key(k: str) -> bool:
            s = _normalize_key(k)
            # match keys like "artifact_report", "artifact-report-file", "artifactReport", "artifactfile", "report"
            return (
                ("artifact" in s and ("report" in s or "file" in s or "text" in s))
                or s == "artifact"
                or s == "report"
            )

        # Keys that indicate a dict is likely a BaseChainArtifactFile / TextChainArtifactFile
        _expected_file_keys = {
            "artifact_id",
            "base_path",
            "chain_artifact_file_type",
            "artifact_name",
            "file_name",
            "text_content",
        }

        def _is_valid_report_candidate(obj: Any) -> bool:
            if not isinstance(obj, dict):
                return False
            return any(k in obj for k in _expected_file_keys)

        def _find_in_structure(obj: Any) -> Optional[Any]:
            if isinstance(obj, dict):
                for k, v in obj.items():
                    try:
                        if (
                            k
                            and _is_report_key(k)
                            and v is not None
                            and _is_valid_report_candidate(v)
                        ):
                            return v
                    except ValueError:
                        pass
                if _is_valid_report_candidate(obj):
                    return obj
                for k, v in obj.items():
                    obj_found = _find_in_structure(v)
                    if obj_found is not None:
                        return obj_found
            elif isinstance(obj, list):
                for item in obj:
                    obj_found = _find_in_structure(item)
                    if obj_found is not None:
                        return obj_found
            return None

        if not isinstance(values, dict):
            return values

        if values.get("artifact_report") is None:
            found = _find_in_structure(values)
            if found is not None:
                values["artifact_report"] = found

        return values

    model_config = {"extra": "ignore"}
