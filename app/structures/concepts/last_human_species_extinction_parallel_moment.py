import os
import yaml
from pydantic import Field, BaseModel
from dotenv import load_dotenv

load_dotenv()


class LastHumanSpeciesExtinctionParallelMoment(BaseModel):
    """A moment where species and human narratives intersect"""

    species_moment: str = Field(..., description="Ecological data point")
    human_moment: str = Field(..., description="Personal experience point")
    thematic_connection: str = Field(..., description="How they mirror each other")
    timestamp_relative: str = Field(
        ..., description="e.g., 'Three months before extinction'"
    )


if __name__ == "__main__":
    with open(
        os.path.join(
            os.getenv("AGENT_MOCK_DATA_PATH"),
            "last_human_parallel_moment_mock.yml",
        ),
        "r",
    ) as file:
        data = yaml.safe_load(file)
        moment = LastHumanSpeciesExtinctionParallelMoment(**data)
        print(moment)
