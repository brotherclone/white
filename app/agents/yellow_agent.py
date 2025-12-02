import logging
import os
import yaml

from abc import ABC
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
from app.structures.concepts.game_evaluation import GameEvaluationDecision
from app.structures.concepts.pulsar_palace_character import PulsarPalaceCharacter
from app.structures.concepts.pulsar_palace_room import PulsarPalaceRoom
from app.structures.generators.character_action_generator import (
    CharacterActionGenerator,
)
from app.structures.generators.markov_room_generator import MarkovRoomGenerator
from app.structures.generators.music_extractor import MusicExtractor
from app.structures.manifests.song_proposal import SongProposalIteration
from app.util.manifest_loader import get_my_reference_proposals

load_dotenv()


class YellowAgent(BaseRainbowAgent, ABC):
    """Pulsar Palace RPG Runner - Automated RPG sessions"""

    room_generator: MarkovRoomGenerator = Field(default_factory=MarkovRoomGenerator)
    action_generator: CharacterActionGenerator = Field(
        default_factory=CharacterActionGenerator
    )
    music_extractor: MusicExtractor = Field(default_factory=MusicExtractor)
    max_rooms: int = Field(default=4)

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
        current_proposal = state.song_proposals.iterations[-1]
        yellow_state = YellowAgentState(
            thread_id=state.thread_id,
            song_proposals=state.song_proposals,
            counter_proposal=None,
            artifacts=[],
            white_proposal=current_proposal,
            rooms=[],
            current_room_index=0,
            characters=[],
            encounter_narrative_artifact=None,
            story_elaboration_level=0,
            should_add_to_story=False,
            max_rooms=4,
        )
        yellow_graph = self.create_graph()
        compiled_graph = yellow_graph.compile()
        result = compiled_graph.invoke(yellow_state.model_dump())
        if isinstance(result, YellowAgentState):
            final_state = result
        elif isinstance(result, dict):
            final_state = YellowAgentState(**result)
        else:
            raise TypeError(f"Unexpected result type: {type(result)}")
        if final_state.counter_proposal:
            state.song_proposals.iterations.append(final_state.counter_proposal)
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
        logging.info("Creating Yellow Agent workflow")
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

    @staticmethod
    def generate_characters(state: YellowAgentState) -> YellowAgentState:
        mock_mode = os.getenv("MOCK_MODE", "false").lower() == "true"
        block_mode = os.getenv("BLOCK_MODE", "false").lower() == "true"
        if mock_mode:
            try:
                with open(
                    f"{os.getenv('AGENT_MOCK_DATA_PATH')}/yellow_character_one_mock.yml",
                    "r",
                ) as file_one:
                    data_one = yaml.safe_load(file_one)
                    character_one = PulsarPalaceCharacter(**data_one)
                    character_one.create_portrait()
                    character_one.create_character_sheet()
                    state.characters.append(character_one)
                with open(
                    f"{os.getenv('AGENT_MOCK_DATA_PATH')}/yellow_character_two_mock.yml",
                    "r",
                ) as file_two:
                    data_two = yaml.safe_load(file_two)
                    character_two = PulsarPalaceCharacter(**data_two)
                    character_two.create_portrait()
                    character_two.create_character_sheet()
                    state.characters.append(character_two)
            except Exception as e:
                error_msg = f"Failed to read mock character files: {e!s}"
                logging.error(error_msg)
                if block_mode:
                    raise Exception(error_msg)
        else:
            num_characters = roll_dice([(1, 4)])[0]
            for i in range(num_characters):
                char = PulsarPalaceCharacter.create_random(
                    thread_id=state.thread_id, encounter_id=f"encounter_{i}"
                )
                char.create_portrait()
                char.create_character_sheet()
                state.characters.append(char)
        return state

    def generate_environment(self, state: YellowAgentState) -> YellowAgentState:
        mock_mode = os.getenv("MOCK_MODE", "false").lower() == "true"
        block_mode = os.getenv("BLOCK_MODE", "false").lower() == "true"
        if mock_mode:
            try:
                with open(
                    f"{os.getenv('AGENT_MOCK_DATA_PATH')}/yellow_room_one_mock.yml", "r"
                ) as file_one:
                    data_one = yaml.safe_load(file_one)
                    room_one = PulsarPalaceRoom(**data_one)
                    state.rooms.append(room_one)
                with open(
                    f"{os.getenv('AGENT_MOCK_DATA_PATH')}/yellow_room_two_mock.yml", "r"
                ) as file_two:
                    data_two = yaml.safe_load(file_two)
                    room_two = PulsarPalaceRoom(**data_two)
                    state.rooms.append(room_two)
                state.story_elaboration_level = 2
            except Exception as e:
                error_msg = f"Failed to read mock room files: {e!s}"
                logging.error(error_msg)
                if block_mode:
                    raise Exception(error_msg)
        else:
            room_count = roll_dice([(1, self.max_rooms)])[0]
            state.story_elaboration_level = room_count
            for i in range(room_count):
                room = self.room_generator.generate_room(room_number=i + 1)
                state.rooms.append(room)
        return state

    def generate_story(self, state: YellowAgentState) -> YellowAgentState:
        current_room = state.rooms[state.current_room_index]
        story = self.action_generator.generate_encounter_narrative(
            room_description=current_room.description, characters=state.characters
        )
        encounter_artifact = PulsarPalaceEncounterArtifact(
            thread_id=state.thread_id,
            base_path=f"{os.getenv('AGENT_WORK_PRODUCT_BASE_PATH', 'chain_artifacts')}",
            artifact_name="pulsar_palace_game_run",
            characters=state.characters,
            rooms=state.rooms,
            story=[story],
        )
        state.encounter_narrative_artifact = encounter_artifact
        state.story_elaboration_level = 1
        return state

    def evaluate_song_proposal(self, state: YellowAgentState) -> YellowAgentState:
        mock_mode = os.getenv("MOCK_MODE", "false").lower() == "true"
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
                elif isinstance(result, GameEvaluationDecision):
                    state.should_add_to_story = result.should_add_to_story
                else:
                    logging.warning(f"Unexpected result type: {type(result)}")
                    state.should_add_to_story = False
            except Exception as e:
                logging.error(f"Game evaluation failed: {e!s}")
                state.should_add_to_story = False
        return state

    def add_to_story(self, state: YellowAgentState) -> YellowAgentState:
        """Continue the story in the next room."""
        state.current_room_index += 1
        current_room = state.rooms[state.current_room_index]
        story = self.action_generator.generate_encounter_narrative(
            room_description=current_room.description, characters=state.characters
        )
        state.encounter_narrative_artifact.story.append(story)
        state.story_elaboration_level += 1
        state.should_add_to_story = False
        return state

    @staticmethod
    def route_after_evaluate_proposal(state: YellowAgentState) -> str:
        mock_mode = os.getenv("MOCK_MODE", "false").lower() == "true"
        if mock_mode:
            return "done"
        else:
            if state.story_elaboration_level < state.max_rooms:
                if state.should_add_to_story:
                    return "add"
            return "done"

    def generate_alternate_song_spec(self, state: YellowAgentState) -> YellowAgentState:
        mock_mode = os.getenv("MOCK_MODE", "false").lower() == "true"
        block_mode = os.getenv("BLOCK_MODE", "false").lower() == "true"
        if mock_mode:
            try:
                with open(
                    f"{os.getenv('AGENT_MOCK_DATA_PATH')}/yellow_counter_proposal_mock.yml",
                    "r",
                ) as f:
                    data = yaml.safe_load(f)
                counter_proposal = SongProposalIteration(**data)
                state.counter_proposal = counter_proposal
            except Exception as e:
                error_msg = f"Failed to read mock counter proposal: {e!s}"
                logging.error(error_msg)
                if block_mode:
                    raise Exception(error_msg)
            return state
        else:
            primary_room = state.rooms[0]
            full_narrative = " ".join(state.encounter_narrative_artifact.story)
            base_proposal = self.music_extractor.extract_song_proposal(
                room=primary_room, encounter_narrative=full_narrative
            )
            prompt = f"""
                    You are Lord Pulsimore, resplendent ruler of the Pulsar Palace and the yellow void that exists between space and time.

                    A game session has just completed in the Palace. Musical parameters have been procedurally extracted:
                    - BPM: {base_proposal.bpm}
                    - Key: {base_proposal.key}
                    - Mood: {base_proposal.mood}
                    - Genres: {base_proposal.genres}

                    The full narrative of what transpired:
                    {full_narrative}

                    Current synthesized White Agent proposal:
                    {state.white_proposal}

                    Reference works in this artist's style:
                    {get_my_reference_proposals('Y')}

                    Create a counter-proposal that:
                    1. Uses the procedurally generated musical parameters above
                    2. Synthesizes the White Agent's themes with the game narrative
                    3. Writes a concept that captures how this RPG session becomes music
                    4. Maintains the Yellow Album's ontological mode: PRESENT + PLACE + IMAGINED

                    Your response should be a SongProposalIteration with:
                    - rainbow_color: Y
                    - The procedural BPM, key, mood, genres above
                    - A creative title (not just the room name)
                    - An enhanced concept that connects game → music → White Agent themes
                    """
            claude = self._get_claude()
            proposer = claude.with_structured_output(SongProposalIteration)
            try:
                result = proposer.invoke(prompt)
                if isinstance(result, dict):
                    counter_proposal = SongProposalIteration(**result)
                    counter_proposal.bpm = base_proposal.bpm
                    counter_proposal.key = base_proposal.key
                    counter_proposal.mood = base_proposal.mood
                    counter_proposal.genres = base_proposal.genres
                    counter_proposal.rainbow_color = "Y"
                    state.counter_proposal = counter_proposal
                elif isinstance(result, SongProposalIteration):
                    # Use the procedural parameters from base_proposal
                    result.bpm = base_proposal.bpm
                    result.key = base_proposal.key
                    result.mood = base_proposal.mood
                    result.genres = base_proposal.genres
                    result.rainbow_color = "Y"
                    state.counter_proposal = result
                else:
                    error_msg = f"Expected SongProposalIteration, got {type(result)}"
                    if block_mode:
                        raise TypeError(error_msg)
                    logging.warning(error_msg)
                    state.counter_proposal = base_proposal
            except Exception as e:
                print(
                    f"Anthropic model call failed: {e!s}; falling back to pure MusicExtractor output"
                )
                if block_mode:
                    raise Exception("Anthropic model call failed")
                else:
                    state.counter_proposal = base_proposal
        return state

    @staticmethod
    def render_game_run(state: YellowAgentState) -> YellowAgentState:
        mock_mode = os.getenv("MOCK_MODE", "false").lower() == "true"
        block_mode = os.getenv("BLOCK_MODE", "false").lower() == "true"
        if mock_mode:
            try:
                with open(
                    f"{os.getenv('AGENT_MOCK_DATA_PATH')}/yellow_encounter_narrative_artifact_mock.yml",
                    "r",
                ) as f:
                    data = yaml.safe_load(f)
                encounter = PulsarPalaceEncounterArtifact(**data)
                # Update base_path to save to chain_artifacts instead of tests/mocks
                encounter.base_path = os.getenv(
                    "AGENT_WORK_PRODUCT_BASE_PATH", "chain_artifacts"
                )
                # Recalculate file_path based on new base_path
                encounter.make_artifact_path()
                state.encounter_narrative_artifact = encounter
                # Save the mock artifact to chain_artifacts
                encounter.save_file()
                state.artifacts.append(encounter)
                logging.info(
                    f"Saved mock game run artifact: {encounter.get_artifact_path()}"
                )
            except Exception as e:
                error_msg = f"Failed to read mock encounter narrative artifact: {e!s}"
                logging.error(error_msg)
                if block_mode:
                    raise Exception(error_msg)
            return state
        else:
            if state.encounter_narrative_artifact:
                try:
                    state.encounter_narrative_artifact.save_file()
                    state.artifacts.append(state.encounter_narrative_artifact)
                    logging.info(
                        f"Saved game run artifact: {state.encounter_narrative_artifact.get_artifact_path()}"
                    )
                except Exception as e:
                    error_msg = f"Failed to save encounter narrative artifact: {e!s}"
                    logging.error(error_msg)
                    if block_mode:
                        raise Exception(error_msg)
            return state
