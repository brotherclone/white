import os

import yaml

from abc import ABC
from pathlib import Path
from pydantic import Field
from dotenv import load_dotenv

from app.structures.artifacts.base_artifact import ChainArtifact
from app.structures.concepts.vanity_interview_question import VanityInterviewQuestion
from app.structures.concepts.vanity_interview_response import VanityInterviewResponse
from app.structures.enums.chain_artifact_file_type import ChainArtifactFileType
from app.structures.enums.chain_artifact_type import ChainArtifactType

load_dotenv()


class CircleJerkInterviewArtifact(ChainArtifact, ABC):

    chain_artifact_type: ChainArtifactType = Field(
        default=ChainArtifactType.CIRCLE_JERK_INTERVIEW,
        description="Type of chain artifact",
    )
    chain_artifact_file_type: ChainArtifactFileType = Field(
        default=ChainArtifactFileType.MARKDOWN,
        description="File format of the artifact: Markdown for text and images",
    )
    artifact_name: str = "circle_jerk_interview"
    interviewer_name: str = Field(description="Full name of interviewer")
    publication: str = Field(description="Publication they're from")
    interviewer_type: str = Field(description="Type of interviewer")
    stance: str = Field(description="Their stance on the work")
    questions: list[VanityInterviewQuestion] = Field(
        description="Three questions asked"
    )
    responses: list[VanityInterviewResponse] = Field(
        description="Three responses given"
    )
    was_human_interview: bool = Field(
        default=False, description="True if real Gabe answered (9% HitL)"
    )

    def __init__(self, **data):
        super().__init__(**data)

    def flatten(self):
        parent_data = super().flatten()
        if parent_data is None:
            parent_data = {}
        return {
            **parent_data,
            "interviewer_name": self.interviewer_name,
            "publication": self.publication,
            "stance": self.stance,
            "questions": [q.model_dump(mode="json") for q in self.questions],
            "responses": [r.model_dump(mode="json") for r in self.responses],
            "was_human_interview": self.was_human_interview,
        }

    def save_file(self):
        file = Path(self.file_path, self.file_name)
        file.parent.mkdir(parents=True, exist_ok=True)
        file = Path(self.file_path, self.file_name)
        with open(file, "w") as f:
            if self.chain_artifact_file_type == ChainArtifactFileType.MARKDOWN:
                f.write(self.to_markdown())
            else:
                yaml.safe_dump(
                    self.model_dump(mode="json"),
                    f,
                    default_flow_style=False,
                    allow_unicode=True,
                )

    def for_prompt(self):
        report_lines = [
            f"# Interview with {self.interviewer_name} ({self.publication})",
            f"Stance: {self.stance}",
            f"Mode: {'Human' if self.was_human_interview else 'Simulated'}",
            "",
        ]
        for q, r in zip(self.questions, self.responses):
            report_lines.append(f"Q{q.number}: {q.question}")
            report_lines.append(f"A{q.number}: {r.response}")
            report_lines.append("")
        return "\n".join(report_lines)

    def to_markdown(self) -> str:
        """Generate a full Markdown transcript for file output"""
        lines = [
            "# Violet Interview Transcript",
            "",
            f"**Interviewer:** {self.interviewer_name}",
            f"**Publication:** {self.publication}",
            f"**Stance:** {self.stance}",
            f"**Mode:** {'Human' if self.was_human_interview else 'Simulated'}",
            "",
            "---",
            "",
        ]
        for q, r in zip(self.questions, self.responses):
            lines.extend(
                [
                    f"## Question {q.number}",
                    "",
                    f"**{self.interviewer_name}:** {q.question}",
                    "",
                    f"**Walsh:** {r.response}",
                    "",
                    "---",
                    "",
                ]
            )
        return "\n".join(lines)


if __name__ == "__main__":
    with open(
        f"{os.getenv('AGENT_MOCK_DATA_PATH')}/violet_circle_jerk_artifact_mock.yml", "r"
    ) as f:
        data = yaml.safe_load(f)
        data["base_path"] = os.getenv("AGENT_WORK_PRODUCT_BASE_PATH")
        interview_artifact = CircleJerkInterviewArtifact(**data)
        interview_artifact.save_file()
        print(interview_artifact.flatten())
        p = interview_artifact.for_prompt()
        print(p)
