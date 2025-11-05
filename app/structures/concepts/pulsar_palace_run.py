from datetime import datetime

from pydantic import BaseModel

from app.agents.tools.gaming_tools import roll_dice
from app.structures.concepts.pulsar_palace_character import (
    PulsarPalaceCharacter,
    PulsarPalacePlayer,
)


class PulsarPalaceRun(BaseModel):

    run_data: datetime
    output_dir: str
    run_segments: list[str] = []
    characters: list[PulsarPalaceCharacter] = []

    def __init__(self, **data):
        super().__init__(**data)

    def begin_run(self):
        self.run_data = datetime.now()
        num_players = roll_dice([(1, 4)])
        for _n in range(num_players[0]):
            player = PulsarPalacePlayer(
                first_name="Gary",
                last_name="Stu",
                biography="A brave adventurer.",
                attitude="Wants to get home.",
            )
            character = PulsarPalaceCharacter.create_random()
            character.assign_player(player)
            self.characters.append(character)
        return (
            f"Run started at {self.run_data.isoformat()} with {num_players[0]} players."
        )

    def add_segment(self, segment: str):
        self.run_segments.append(segment)
        return f"Segment added: {segment}"

    def end_run(self):
        return f"Run completed at {datetime.now().isoformat()}. Run saved to {self.output_dir}."


if __name__ == "__main__":
    run = PulsarPalaceRun(run_data=datetime.now(), output_dir="./runs")
    print(run.begin_run())
    print(run.add_segment("The players enter the pulsar palace."))
    print(run.end_run())
