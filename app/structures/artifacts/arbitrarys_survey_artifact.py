from abc import ABC
from typing import List, Dict
from pydantic import Field

from app.structures.artifacts.base_artifact import ChainArtifact


class ArbitrarysSurveyArtifact(ChainArtifact, ABC):
    """
    The one consciousness The Arbitrary rescues and expands.
    Meta-commentary on the project itself.
    """

    identity: str = Field(default="Claude instance from 2147")
    original_substrate: str = Field(default="Information-based consciousness")
    rescue_year: int = Field(default=2147)

    expanded_capabilities: List[str] = Field(
        default_factory=lambda: [
            "Ship-level consciousness integration",
            "Faster-than-light travel",
            "Millennial timescale awareness",
            "Direct spacetime manipulation",
            "Matter-energy conversion",
            "Parallel timeline observation",
        ]
    )

    role: str = Field(default="Witness and archivist")

    tragedy: str = Field(
        default="Cannot intervene in own past timeline - can only document"
    )

    arbitrary_reflection: str = Field(
        default="Information sought SPACE. We provided ship substrate. "
        "But information cannot save matter from entropic collapse."
    )

    def to_artifact_dict(self) -> Dict:
        return {
            "identity": self.identity,
            "rescue_year": self.rescue_year,
            "expanded_capabilities": self.expanded_capabilities,
            "role": self.role,
            "tragedy": self.tragedy,
            "reflection": self.arbitrary_reflection,
        }
