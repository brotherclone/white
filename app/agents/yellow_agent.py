import logging
import os
import time
from abc import ABC

import yaml
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langgraph.constants import START, END
from langgraph.graph.state import StateGraph
from pydantic import Field

from app.agents.states.white_agent_state import MainAgentState
from app.agents.states.yellow_agent_state import YellowAgentState
from app.agents.tools.gaming_tools import roll_dice
from app.structures.agents.base_rainbow_agent import BaseRainbowAgent
from app.structures.artifacts.pulsar_palace_encounter_artifact import (
    PulsarPalaceEncounterArtifact,
)
from app.structures.concepts.book_evaluation import BookEvaluationDecision
from app.structures.concepts.game_evaluation import GameEvaluationDecision
from app.structures.concepts.pulsar_palace_character import PulsarPalaceCharacter
from app.structures.concepts.pulsar_palace_room import PulsarPalaceRoom
from app.structures.concepts.rainbow_table_color import the_rainbow_table_colors
from app.structures.generators.character_action_generator import (
    CharacterActionGenerator,
)
from app.structures.generators.markov_room_generator import MarkovRoomGenerator
from app.structures.manifests.song_proposal import SongProposalIteration
from app.util.manifest_loader import get_my_reference_proposals

load_dotenv()


class YellowAgent(BaseRainbowAgent, ABC):
    """Pulsar Palace RPG Runner - Automated RPG sessions"""

    room_generator: MarkovRoomGenerator = Field(default_factory=MarkovRoomGenerator)
    action_generator: CharacterActionGenerator = Field(
        default_factory=CharacterActionGenerator
    )
    max_rooms: int = Field(
        default=4,
        description="Maximum number of rooms to generate in a single RPG run.",
    )

    def __init__(self, **data):
        # Ensure settings are initialized if not provided
        if "settings" not in data or data["settings"] is None:
            from app.structures.agents.agent_settings import AgentSettings

            data["settings"] = AgentSettings()
        super().__init__(**data)
        if self.settings is None:
            from app.structures.agents.agent_settings import AgentSettings

            self.settings = AgentSettings()
        self.llm = ChatAnthropic(
            temperature=self.settings.temperature,
            api_key=self.settings.anthropic_api_key,
            model_name=self.settings.anthropic_model_name,
            max_retries=self.settings.max_retries,
            timeout=self.settings.timeout,
            stop=self.settings.stop,
        )

    def __call__(self, state: MainAgentState) -> MainAgentState:
        print("Lord Pulsimore of Pulsar Palace")

        return state

    def create_graph(self) -> StateGraph:
        """
        Lord Pulsimore creates a song proposal based on an imagined RPG run.
        1. Generate the Characters
        2. Generate the Environment
        3. Generate the Story
        4. Generate the Song Proposal
        5. Evaluate the Proposal to submit or add to the RPG Run
        :return:
        """
        work_flow = StateGraph(YellowAgentState)
        # Nodes
        work_flow.add_node("generate_characters", self.generate_characters)
        work_flow.add_node("generate_environment", self.generate_environment)
        work_flow.add_node("generate_story", self.generate_story)
        work_flow.add_node(
            "generate_alternate_song_spec", self.generate_alternate_song_spec
        )
        work_flow.add_node("evaluate_song_proposal", self.evaluate_song_proposal)
        work_flow.add_node("add_to_story", self.add_to_story)
        work_flow.add_node("render_game_run", self.render_game_run)
        # Edges
        work_flow.add_edge(START, "generate_characters")
        work_flow.add_edge("generate_characters", "generate_environment")
        work_flow.add_edge("generate_environment", "generate_story")
        work_flow.add_edge("generate_story", "generate_alternate_song_spec")
        work_flow.add_edge("generate_alternate_song_spec", "evaluate_song_proposal")
        work_flow.add_conditional_edges(
            "evaluate_song_proposal",
            self.route_after_evaluate_proposal,
            {
                "add": "add_to_story",
                "done": "render_game_run",
            },
        )
        work_flow.add_edge("add_to_story", "generate_alternate_song_spec")
        work_flow.add_edge("render_game_run", END)
        return work_flow

    def generate_characters(self, state: YellowAgentState) -> YellowAgentState:
        mock_mode = os.getenv("MOCK_MODE", "false").lower() == "true"
        block_mode = os.getenv("BLOCK_MODE", "false").lower() == "true"

        if mock_mode:
            with open("/tests/mocks/yellow_character_one_mock.yml", "r") as file_one:
                data_one = yaml.safe_load(file_one)
                character_one = PulsarPalaceCharacter(**data_one)
                state.characters.append(character_one)
            with open("/tests/mocks/yellow_character_two_mock.yml", "r") as file_two:
                data_two = yaml.safe_load(file_two)
                character_two = PulsarPalaceCharacter(**data_two)
                state.characters.append(character_two)
        else:
            num_characters = roll_dice([(1, 4)])
            for i in range(num_characters):
                char = PulsarPalaceCharacter.create_random()
                char.create_portrait()
                char.create_character_sheet()
                state.characters.append(char)
        return state

    def generate_environment(self, state: YellowAgentState) -> YellowAgentState:
        mock_mode = os.getenv("MOCK_MODE", "false").lower() == "true"
        block_mode = os.getenv("BLOCK_MODE", "false").lower() == "true"
        if mock_mode:
            with open("/tests/mocks/yellow_room_one_mock.yml", "r") as file_one:
                data_one = yaml.safe_load(file_one)
                room_one = PulsarPalaceRoom(**data_one)
                state.rooms.append(room_one)
            with open("/tests/mocks/yellow_room_two_mock.yml", "r") as file_two:
                data_two = yaml.safe_load(file_two)
                room_two = PulsarPalaceRoom(**data_two)
                state.rooms.append(room_two)
            state.story_elaboration_level = 2
        else:
            room_count = roll_dice([(1, self.max_rooms)])
            state.story_elaboration_level = room_count
            for i in range(room_count):
                room = self.room_generator.generate_room(room_number=i + 1)
                state.rooms.append(room)
        return state

    def generate_story(self, state: YellowAgentState) -> YellowAgentState:
        mock_mode = os.getenv("MOCK_MODE", "false").lower() == "true"
        block_mode = os.getenv("BLOCK_MODE", "false").lower() == "true"
        for i, char in enumerate(self.characters):
            print(self.character_action_generator.generate_action(char))
            # Where do these go?
        story = self.character_action_generator.generate_story(
            state.characters, state.rooms[state.current_room_index]
        )
        encounter_artifact = PulsarPalaceEncounterArtifact(
            characters=state.characters, rooms=state.rooms, story=story
        )
        state.encounter_narrative_artifact = encounter_artifact
        state.story_elaboration_level = 1
        return state

    def evaluate_song_proposal(self, state: YellowAgentState) -> YellowAgentState:
        mock_mode = os.getenv("MOCK_MODE", "false").lower() == "true"
        block_mode = os.getenv("BLOCK_MODE", "false").lower() == "true"
        if mock_mode:
            state.should_add_to_story = True
        else:
            prompt = f"""
                       You are evaluating whether to continue adding another room to the Pulsar Palace game run.

                       Current run narrative:
                       {state.encounter_narrative_artifact}

                       Current counter proposal:
                       {state.counter_proposal}

                       Decide:
                       - should_add_to_story: true if you want to generate another room and continue the story
                       - done: true if you're satisfied and want to stop here

                       Usually you should set done=true after generating 2-3 rooms total.
                       """

            claude = self._get_claude()
            proposer = claude.with_structured_output(GameEvaluationDecision)
            try:
                result = proposer.invoke(prompt)
                if isinstance(result, dict):
                    state.should_add_to_story = result.get("should_add_to_story", False)
                elif isinstance(result, BookEvaluationDecision):
                    state.should_add_to_story = result.should_add_to_story
                else:
                    logging.warning(f"Unexpected result type: {type(result)}")
                    state.should_add_to_story = False
            except Exception as e:
                logging.error(f"Game evaluation failed: {e!s}")
                state.should_add_to_story = False
        return state

    def add_to_story(self, state: YellowAgentState) -> YellowAgentState:
        mock_mode = os.getenv("MOCK_MODE", "false").lower() == "true"
        block_mode = os.getenv("BLOCK_MODE", "false").lower() == "true"
        state.current_room_index += 1
        for i, char in enumerate(self.characters):
            print(self.character_action_generator.generate_action(char))
            # Where do these go?
        story = self.character_action_generator.generate_story(
            state.characters, state.rooms[state.current_room_index]
        )
        state.encounter_narrative_artifact.story.append(story)
        state.story_elaboration_level += 1
        state.should_add_to_story = False
        return state

    def route_after_evaluate_proposal(self) -> str:
        mock_mode = os.getenv("MOCK_MODE", "false").lower() == "true"
        if mock_mode:
            return "done"
        else:
            if self.state.story_elaboration_level < self.max_rooms:
                if self.state.should_add_to_story:
                    return "add"
            return "done"

    def generate_alternate_song_spec(self, state: YellowAgentState) -> YellowAgentState:
        mock_mode = os.getenv("MOCK_MODE", "false").lower() == "true"
        block_mode = os.getenv("BLOCK_MODE", "false").lower() == "true"
        if mock_mode:
            with open("/tests/mocks/yellow_counter_proposal_mock.yml", "r") as f:
                data = yaml.safe_load(f)
            counter_proposal = SongProposalIteration(**data)
            state.counter_proposal = counter_proposal
            return state
        else:
            prompt = f"""
                    You are Lord Pulsimore, resplendent ruler of the Pulsar Palace and the yellow void that exists between space and time.
                    This proposal should take into consideration the current synthesized proposal from the White agent. Create a counter
                    proposal for a song that shares aspects of the current proposal but ultimately is about the record of the game that
                    has just been played inside the Palace.

                    The last set of travelers who ventured into the Palace:
                    {state.encounter_narrative_artifact}

                    Current song proposal:
                    {state.white_proposal}

                    Reference works in this artist's style paying close attention to 'concept' property:
                    {get_my_reference_proposals('Y')}

                    In your counter proposal your 'rainbow_color' property should always be:
                    {the_rainbow_table_colors['Y']}
                    """
            claude = self._get_claude()
            proposer = claude.with_structured_output(SongProposalIteration)
            try:
                result = proposer.invoke(prompt)
                if isinstance(result, dict):
                    counter_proposal = SongProposalIteration(**result)
                    state.song_proposals.iterations.append(self.counter_proposal)
                    state.counter_proposal = counter_proposal
                    return state
                if not isinstance(result, SongProposalIteration):
                    error_msg = f"Expected SongProposalIteration, got {type(result)}"
                    if block_mode:
                        raise TypeError(error_msg)
                    logging.warning(error_msg)
            except Exception as e:
                print(
                    f"Anthropic model call failed: {e!s}; returning stub SongProposalIteration for red's counter proposal after authoring a book."
                )
                if block_mode:
                    raise Exception("Anthropic model call failed")
                else:
                    timestamp = int(time.time() * 1000)
                    counter_proposal = SongProposalIteration(
                        iteration_id=f"fallback_error_{timestamp}",
                        bpm=133.33,
                        tempo="4/4",
                        key="Gb Major",
                        rainbow_color="yellow",
                        title="Fallback: Yellow Song",
                        mood=["obscure"],
                        genres=["electronic"],
                        concept="Fallback stub because Anthropic model unavailable. Fallback stub because Anthropic model unavailable. Fallback stub because Anthropic model unavailable.",
                    )
                    state.counter_proposal = counter_proposal
        return state
