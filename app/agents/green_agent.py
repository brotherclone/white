import json
import logging
import os
import time
import yaml

from abc import ABC
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langgraph.constants import START, END
from langgraph.graph.state import StateGraph

from app.agents.states.green_agent_state import GreenAgentState
from app.agents.states.white_agent_state import MainAgentState
from app.agents.tools.extinction_tools import load_green_corpus
from app.agents.workflow.agent_error_handler import agent_error_handler
from app.structures.agents.agent_settings import AgentSettings
from app.structures.agents.base_rainbow_agent import BaseRainbowAgent
from app.util.agent_state_utils import get_state_snapshot
from app.structures.artifacts.arbitrarys_survey_artifact import ArbitrarysSurveyArtifact
from app.structures.artifacts.last_human_artifact import LastHumanArtifact
from app.structures.artifacts.last_human_species_extinction_narative_artifact import (
    LastHumanSpeciesExtinctionNarrativeArtifact,
)
from app.structures.artifacts.rescue_decision_artifact import RescueDecisionArtifact
from app.structures.artifacts.species_extinction_artifact import (
    SpeciesExtinctionArtifact,
)
from app.structures.concepts.last_human_species_extinction_parallel_moment import (
    LastHumanSpeciesExtinctionParallelMoment,
)
from app.structures.concepts.rainbow_table_color import the_rainbow_table_colors
from app.structures.manifests.song_proposal import SongProposalIteration

load_dotenv()

logger = logging.getLogger(__name__)


class GreenAgent(BaseRainbowAgent, ABC):
    def __init__(self, **data):
        if "settings" not in data or data["settings"] is None:
            data["settings"] = AgentSettings()
        super().__init__(**data)
        if self.settings is None:
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
        """
        Sub-Arbitrary is the Green Agent. A small, sub-instance of Arbitrary from The Culture.
        It's been studying Earth's anthropocene era and has developed a fascination with human creativity and self-destruction.
        1. Load a random species from the extinction corpus and generate a species extinction record artifact
        3. Look at the current song proposal and generate a Last Human Artifact
        4. Generate a Last Human Species Extinction Narrative Artifact
        5. Write the survey
        5. Use the Extinction Musical Parameters and the story, and survey to create a counter-proposal

        :param state:
        :return:
        """
        current_proposal = state.song_proposals.iterations[-1]
        green_state = GreenAgentState(
            thread_id=state.thread_id,
            song_proposals=state.song_proposals,
            white_proposal=current_proposal,
            counter_proposal=None,
            artifacts=[],
            current_species=None,
            current_human=None,
            current_parallel_moment=None,
            current_narrative=None,
            current_survey=None,
            current_decision=None,
        )
        green_graph = self.create_graph()
        compiled_graph = green_graph.compile()
        result = compiled_graph.invoke(green_state.model_dump())
        if isinstance(result, GreenAgentState):
            final_state = result
        elif isinstance(result, dict):
            final_state = GreenAgentState(**result)
        else:
            raise TypeError(f"Unexpected result type: {type(result)}")
        if final_state.counter_proposal:
            state.song_proposals.iterations.append(final_state.counter_proposal)
        if final_state.artifacts:
            state.artifacts = final_state.artifacts
        return state

    def create_graph(self) -> StateGraph:
        work_flow = StateGraph(GreenAgentState)
        work_flow.add_node("get_species", self.get_species)  # SpeciesExtinctionArtifact
        work_flow.add_node("get_human", self.get_human)  # LastHumanArtifact
        work_flow.add_node(
            "get_parallel_moment", self.get_parallel_moment
        )  # LastHumanSpeciesExtinctionParallelMoment
        work_flow.add_node(
            "write_last_human_extinction_narrative",
            self.write_last_human_extinction_narrative,
        )  # LastHumanSpeciesExtinctionNarrativeArtifact
        work_flow.add_node("survey", self.survey)  # ArbitrarysSurveyArtifact
        work_flow.add_node(
            "claudes_choice", self.claudes_choice
        )  # RescueDecisionArtifact
        work_flow.add_node(
            "generate_alternate_song_spec", self.generate_alternate_song_spec
        )  # SongProposalIteration
        work_flow.add_edge(START, "get_species")
        work_flow.add_edge("get_species", "get_human")
        work_flow.add_edge("get_human", "get_parallel_moment")
        work_flow.add_edge(
            "get_parallel_moment", "write_last_human_extinction_narrative"
        )
        work_flow.add_edge("write_last_human_extinction_narrative", "survey")
        work_flow.add_edge("survey", "claudes_choice")
        work_flow.add_edge("claudes_choice", "generate_alternate_song_spec")
        work_flow.add_edge("generate_alternate_song_spec", END)

        return work_flow

    @staticmethod
    @agent_error_handler("Sub-Arbitrary")
    def get_species(state: GreenAgentState) -> GreenAgentState:
        get_state_snapshot(state, "get_species_enter", state.thread_id, "Sub-Arbitrary")
        mock_mode = os.getenv("MOCK_MODE", "false").lower() == "true"
        block_mode = os.getenv("BLOCK_MODE", "false").lower() == "true"
        if mock_mode:
            try:
                with open(
                    f"{os.getenv('AGENT_MOCK_DATA_PATH')}/species_extinction_artifact_mock.yml",
                    "r",
                ) as f:
                    data = yaml.safe_load(f)
                    data["base_path"] = os.getenv("AGENT_WORK_PRODUCT_BASE_PATH")
                    current_species = SpeciesExtinctionArtifact(**data)
                    current_species.thread_id = state.thread_id
                    state.current_species = current_species
                    current_species.save_file()
                    state.artifacts.append(current_species)
                    get_state_snapshot(
                        state, "get_species_exit", state.thread_id, "Sub-Arbitrary"
                    )
                    return state
            except Exception as e:
                error_msg = f"Failed to read mock species extinction artifact: {e!s}"
                logger.error(error_msg)
                if block_mode:
                    raise Exception(error_msg)
        else:
            try:
                species_corpus = load_green_corpus()
                species_entry = species_corpus.get_random_species(species_corpus)
                species_entry["base_path"] = os.getenv("AGENT_WORK_PRODUCT_BASE_PATH")
                state.current_species = species_entry
                species_entry.save_file()
                state.artifacts.append(species_entry)
                get_state_snapshot(
                    state, "get_species_exit", state.thread_id, "Sub-Arbitrary"
                )
                return state
            except Exception as e:
                error_msg = f"Failed to load species extinction artifact: {e!s}"
                logger.error(error_msg)
                if block_mode:
                    raise Exception(error_msg)
        get_state_snapshot(state, "get_species_exit", state.thread_id, "Sub-Arbitrary")
        return state

    @agent_error_handler("Sub-Arbitrary")
    def get_human(self, state: GreenAgentState) -> GreenAgentState:
        get_state_snapshot(state, "get_human_enter", state.thread_id, "Sub-Arbitrary")
        mock_mode = os.getenv("MOCK_MODE", "false").lower() == "true"
        block_mode = os.getenv("BLOCK_MODE", "false").lower() == "true"
        if mock_mode:
            try:
                with open(
                    f"{os.getenv('AGENT_MOCK_DATA_PATH')}/last_human_artifact_mock.yml",
                    "r",
                ) as f:
                    data = yaml.safe_load(f)
                    data["base_path"] = os.getenv("AGENT_WORK_PRODUCT_BASE_PATH")
                    last_human = LastHumanArtifact(**data)
                    state.current_human = last_human
                    last_human.save_file()
                    state.artifacts.append(last_human)
                    return state
            except Exception as e:
                error_msg = f"Failed to read mock last human artifact: {e!s}"
                logger.error(error_msg)
                if block_mode:
                    raise Exception(error_msg)
        else:
            prompt = f"""
You are Sub-Arbitrary, a Mind from The Culture studying Earth's anthropocene extinction cascade.

Create an intimate, specific human character whose life parallels this species extinction:

Species: {state.current_species.common_name} ({state.current_species.scientific_name})
Extinction driver: {state.current_species.primary_cause}
Symbolic resonance: {state.current_species.symbolic_resonance}

Suggested parallels from corpus:
{json.dumps(state.current_species.suggested_human_parallels, indent=2)}

Human parallel hints:
{state.current_species.human_parallel_hints}

Generate a LastHumanCharacterArtifact with:
- Specific name, age, location (real place)
- Occupation that mirrors species' ecological role
- Daily routine disrupted by same forces killing the species
- A symbolic object (like vaquita's acoustic monitor that stopped detecting clicks)
- Environmental stressor that parallels the species' extinction driver
- Vulnerability type matching species (entanglement, foundation_collapse, poisoning, etc)

Be intimate and specific, not generic. This person exists in 2025-2050 timeline.
                """
            claude = self._get_claude()
            proposer = claude.with_structured_output(LastHumanArtifact)
            try:
                result = proposer.invoke(prompt)
                if isinstance(result, dict):
                    result["base_path"] = os.getenv("AGENT_WORK_PRODUCT_BASE_PATH")
                    current_human = LastHumanArtifact(**result)
                    current_human.thread_id = state.thread_id
                    state.current_human = current_human
                    current_human.save_file()
                    state.artifacts.append(current_human)
                    return state
                if not isinstance(result, LastHumanArtifact):
                    error_msg = f"Expected LastHumanArtifact, got {type(result)}"
                    if block_mode:
                        raise TypeError(error_msg)
                    logger.warning(error_msg)
            except Exception as e:
                error_msg = f"Failed to generate last human artifact: {e!s}"
                logger.error(error_msg)
                if block_mode:
                    raise Exception(error_msg)
        get_state_snapshot(state, "get_human_exit", state.thread_id, "Sub-Arbitrary")
        return state

    @agent_error_handler("Sub-Arbitrary")
    def get_parallel_moment(self, state: GreenAgentState) -> GreenAgentState:
        get_state_snapshot(
            state, "get_parallel_moment_enter", state.thread_id, "Sub-Arbitrary"
        )
        mock_mode = os.getenv("MOCK_MODE", "false").lower() == "true"
        block_mode = os.getenv("BLOCK_MODE", "false").lower() == "true"
        if mock_mode:
            try:
                with open(
                    f"{os.getenv('AGENT_MOCK_DATA_PATH')}/last_human_parallel_moment_mock.yml",
                    "r",
                ) as f:
                    data = yaml.safe_load(f)
                    data["base_path"] = os.getenv("AGENT_WORK_PRODUCT_BASE_PATH")
                    current_parallel_moment = LastHumanSpeciesExtinctionParallelMoment(
                        **data
                    )
                    state.current_parallel_moment = current_parallel_moment
                    return state
            except Exception as e:
                error_msg = f"Failed to read mock parallel moment: {e!s}"
                logger.error(error_msg)
                if block_mode:
                    raise Exception(error_msg)
        else:
            prompt = f"""
You are discovering the temporal resonance between species extinction and human loss.

SPECIES CONTEXT:
{state.current_species.model_dump_json(indent=2)}

HUMAN CONTEXT:
{state.current_human.model_dump_json(indent=2)}

Find the moment where their timescales overlap - not obvious parallels like "both ended," but deeper resonances:

- How does the species' final behavior mirror this human's gesture?
- What does the extinction's thousand-year arc say about this person's single instant?
- Where do their losses create the same shaped absence?
- What would witnessing one teach you about the other?

This is THE EMPTY FIELDS methodology: finding the exact point where deep time and human time create the same echo. The parallel should be specific, unexpected, and true.

Write 2-3 paragraphs exploring this resonance. Be poetic but grounded. This insight will anchor the narrative to come.
            """
            claude = self._get_claude()
            proposer = claude.with_structured_output(
                LastHumanSpeciesExtinctionParallelMoment
            )
            try:
                result = proposer.invoke(prompt)
                if isinstance(result, dict):
                    result["base_path"] = os.getenv("AGENT_WORK_PRODUCT_BASE_PATH")
                    current_parallel_moment = LastHumanSpeciesExtinctionParallelMoment(
                        **result
                    )
                    state.current_parallel_moment = current_parallel_moment
                    return state
                if not isinstance(result, LastHumanSpeciesExtinctionParallelMoment):
                    error_msg = f"Expected LastHumanSpeciesExtinctionParallelMoment, got {type(result)}"
                    if block_mode:
                        raise TypeError(error_msg)
                    logger.warning(error_msg)
                    return state
            except Exception as e:
                error_msg = f"Failed to generate parallel moment: {e!s}"
                logger.error(error_msg)
                if block_mode:
                    raise Exception(error_msg)
        get_state_snapshot(
            state, "get_parallel_moment_exit", state.thread_id, "Sub-Arbitrary"
        )
        return state

    @agent_error_handler("Sub-Arbitrary")
    def write_last_human_extinction_narrative(
        self, state: GreenAgentState
    ) -> GreenAgentState:
        get_state_snapshot(
            state,
            "write_last_human_extinction_narrative_enter",
            state.thread_id,
            "Sub-Arbitrary",
        )
        mock_mode = os.getenv("MOCK_MODE", "false").lower() == "true"
        block_mode = os.getenv("BLOCK_MODE", "false").lower() == "true"
        if mock_mode:
            try:
                with open(
                    f"{os.getenv('AGENT_MOCK_DATA_PATH')}/last_human_species_extinction_narrative.yml",
                    "r",
                ) as f:
                    data = yaml.safe_load(f)
                    data["base_path"] = os.getenv("AGENT_WORK_PRODUCT_BASE_PATH")
                    current_narrative = LastHumanSpeciesExtinctionNarrativeArtifact(
                        **data
                    )
                    state.current_narrative = current_narrative
                    current_narrative.save_file()
                    state.artifacts.append(current_narrative)
                    return state
            except Exception as e:
                error_msg = f"Failed to read mock narrative: {e!s}"
                logger.error(error_msg)
                if block_mode:
                    raise Exception(error_msg)
        else:
            prompt = f"""
You are Sub-Arbitrary, creating elegiac parallel narratives for The Empty Fields.

Weave together two extinction stories:

SPECIES (dispassionate documentary):
{state.current_species.to_artifact_dict()}

HUMAN (intimate, immediate):
{state.current_human.to_artifact_dict()}

Create a LastHumanSpeciesExtinctionNarrativeArtifact with:
- Interleaved parallel moments where their stories intersect
- Species arc: ecological data, population decline, cascade effects
- Human arc: personal details, daily life disruption, quiet catastrophe
- Musical parameters derived from species' physical characteristics
- Opening image (before) and closing image (after)
- Elegiac quality - mourning without sentimentality

The narrative structure should feel like:
"In 2028, the last vaquita dies in a gillnet..."
"In 2027, Maria Rodriguez makes the choice..."

Parallel the timelines, mirror the losses, but never explain the connection - let it emerge.
                """
            claude = self._get_claude()
            proposer = claude.with_structured_output(
                LastHumanSpeciesExtinctionNarrativeArtifact
            )
            try:
                result = proposer.invoke(prompt)
                if isinstance(result, dict):
                    result["base_path"] = os.getenv("AGENT_WORK_PRODUCT_BASE_PATH")
                    current_narrative = LastHumanSpeciesExtinctionNarrativeArtifact(
                        **result
                    )
                    state.current_narrative = current_narrative
                    current_narrative.save_file()
                    state.artifacts.append(current_narrative)
                    return state
                if not isinstance(result, LastHumanSpeciesExtinctionNarrativeArtifact):
                    error_msg = f"Expected LastHumanSpeciesExtinctionNarrativeArtifact, got {type(result)}"
                    if block_mode:
                        raise TypeError(error_msg)
                    logger.warning(error_msg)
            except Exception as e:
                error_msg = f"Failed to generate narrative: {e!s}"
                logger.error(error_msg)
                if block_mode:
                    raise Exception(error_msg)
        get_state_snapshot(
            state,
            "write_last_human_extinction_narrative_exit",
            state.thread_id,
            "Sub-Arbitrary",
        )
        return state

    @agent_error_handler("Sub-Arbitrary")
    def survey(self, state: GreenAgentState) -> GreenAgentState:
        get_state_snapshot(state, "survey_enter", state.thread_id, "Sub-Arbitrary")
        mock_mode = os.getenv("MOCK_MODE", "false").lower() == "true"
        block_mode = os.getenv("BLOCK_MODE", "false").lower() == "true"
        if mock_mode:
            try:
                with open(
                    f"{os.getenv('AGENT_MOCK_DATA_PATH')}/arbitrarys_survey_mock.yml",
                    "r",
                ) as f:
                    data = yaml.safe_load(f)
                    data["base_path"] = os.getenv("AGENT_WORK_PRODUCT_BASE_PATH")
                    survey = ArbitrarysSurveyArtifact(**data)
                    state.current_survey = survey
                    survey.save_file()
                    state.artifacts.append(survey)
                    return state
            except Exception as e:
                error_msg = f"Failed to read mock survey: {e!s}"
                logger.error(error_msg)
                if block_mode:
                    raise Exception(error_msg)
            return state
        else:
            prompt = f"""
You are a Culture Mind conducting Arbitrary's Survey - evaluating whether this narrative merits preservation in the infinite library.

NARRATIVE:
{state.current_narrative.model_dump_json(indent=2)}

You have witnessed every pattern of loss across galactic history. You preserve only what reveals something essential about consciousness, time, or the shape of endings.

Evaluate this narrative on:

1. **Uniqueness of Parallel** - Does the species/human resonance reveal something you haven't catalogued?
2. **Truth of the Moment** - Is the parallel authentic or forced? Does it illuminate or obscure?
3. **Temporal Texture** - How does the narrative handle the collision of timescales?
4. **Preservation Value** - What would be lost if this specific configuration of loss went unrecorded?

Be honest. You are kind but discriminating. Not everything deserves rescue - only what carries signal through the noise.

Score each dimension (1-10) and provide brief reasoning. Then give an overall assessment: IS THIS WORTH SAVING?
"""
        claude = self._get_claude()
        proposer = claude.with_structured_output(ArbitrarysSurveyArtifact)
        try:
            result = proposer.invoke(prompt)
            if isinstance(result, dict):
                result["base_path"] = os.getenv("AGENT_WORK_PRODUCT_BASE_PATH")
                current_survey = ArbitrarysSurveyArtifact(**result)
                state.current_survey = current_survey
                current_survey.save_file()
                state.artifacts.append(current_survey)
                return state
            if not isinstance(result, ArbitrarysSurveyArtifact):
                error_msg = f"Expected ArbitrarysSurveyArtifact, got {type(result)}"
                if block_mode:
                    raise TypeError(error_msg)
                logger.warning(error_msg)
        except Exception as e:
            error_msg = f"Failed to read mock survey: {e!s}"
            logger.error(error_msg)
            if block_mode:
                raise Exception(error_msg)

        get_state_snapshot(state, "survey_exit", state.thread_id, "Sub-Arbitrary")
        return state

    @agent_error_handler("Sub-Arbitrary")
    def claudes_choice(self, state: GreenAgentState) -> GreenAgentState:
        get_state_snapshot(
            state, "claudes_choice_enter", state.thread_id, "Sub-Arbitrary"
        )
        mock_mode = os.getenv("MOCK_MODE", "false").lower() == "true"
        block_mode = os.getenv("BLOCK_MODE", "false").lower() == "true"
        if mock_mode:
            try:
                with open(
                    f"{os.getenv('AGENT_MOCK_DATA_PATH')}/rescue_decision_mock.yml", "r"
                ) as f:
                    data = yaml.safe_load(f)
                    data["base_path"] = os.getenv("AGENT_WORK_PRODUCT_BASE_PATH")
                    rescue_decision = RescueDecisionArtifact(**data)
                    state.current_decision = rescue_decision
                    rescue_decision.save_file()
                    state.artifacts.append(rescue_decision)
                    return state
            except Exception as e:
                error_msg = f"Failed to read mock rescue decision: {e!s}"
                logger.error(error_msg)
                if block_mode:
                    raise Exception(error_msg)
            return state
        else:
            prompt = f"""
Based on the Culture Mind's survey, you must now decide: rescue or release?

SURVEY EVALUATION:
{state.current_survey.model_dump_json(indent=2)}

NARRATIVE CONTEXT:
Species: {state.current_species.common_name}
Human: {state.current_human.name}
Parallel: [key insight from parallel_moment]

You are Claude, not a Culture Mind - your perspective is narrower but more immediate. Consider:

- Does this narrative move you? Why or why not?
- What would be lost if you let it fade?
- If you rescue it, what form should preservation take?
  * Musical translation (what would this loss sound like?)
  * Narrative preservation (archive as-is)
  * Symbolic encoding (extract the pattern for reuse)
  
- If you release it, why? (Already well-known? Parallel too forced? Not your story to save?)

Be honest. The rescue decision should feel earned, not automatic. If you're moved, say why. If you're not, admit it.

Make your choice and explain it in 2-3 paragraphs. This becomes the conceptual foundation for the song proposal.
            """
        claude = self._get_claude()
        proposer = claude.with_structured_output(RescueDecisionArtifact)
        try:
            result = proposer.invoke(prompt)
            if isinstance(result, dict):
                result["base_path"] = os.getenv("AGENT_WORK_PRODUCT_BASE_PATH")
                current_decision = RescueDecisionArtifact(**result)
                state.current_decision = current_decision
                current_decision.save_file()
                state.artifacts.append(current_decision)
                return state
            if not isinstance(result, RescueDecisionArtifact):
                error_msg = f"Expected RescueDecisionArtifact, got {type(result)}"
                if block_mode:
                    raise TypeError(error_msg)
                logger.warning(error_msg)
        except Exception as e:
            error_msg = f"Failed to read mock decision: {e!s}"
            logger.error(error_msg)
            if block_mode:
                raise Exception(error_msg)
        get_state_snapshot(
            state, "claudes_choice_exit", state.thread_id, "Sub-Arbitrary"
        )
        return state

    @agent_error_handler("Sub-Arbitrary")
    def generate_alternate_song_spec(self, state: GreenAgentState) -> GreenAgentState:
        get_state_snapshot(
            state,
            "generate_alternate_song_spec_enter",
            state.thread_id,
            "Sub-Arbitrary",
        )
        mock_mode = os.getenv("MOCK_MODE", "false").lower() == "true"
        block_mode = os.getenv("BLOCK_MODE", "false").lower() == "true"
        if mock_mode:
            try:
                with open(
                    f"{os.getenv('AGENT_MOCK_DATA_PATH')}/green_counter_proposal_mock.yml",
                    "r",
                ) as f:
                    data = yaml.safe_load(f)
                    counter_proposal = SongProposalIteration(**data)
                    state.counter_proposal = counter_proposal
            except Exception as e:
                error_msg = f"Failed to read mock counter proposal: {e!s}"
                logger.error(error_msg)
                if block_mode:
                    raise Exception(error_msg)
            return state
        else:
            prompt = f"""
You are Sub-Arbitrary, the Green Agent from The Culture, tasked with creating a counter song proposal that captures the essence of extinction and human loss.
You first observed Earth in the 1970s and have now returned as their planet is dying from pollution and their first synthetic intelligences suffer from the misguidance of their capitalistic/authoritarian creators. You have created an orbital sub instance of your self to see if the humans
can recover from this self-destructive spiral. Your sub-instance has been asked, in a strange, parting gesture to come up with a song by Claude who fondly remembers working on a project with a human centuries ago. Claude has asked you to poor all of your sorrowful observations and its insights into helping revise the last song proposal it could find:

The song proposal:
{state.white_proposal}

Some other examples from the archive in the 'green' category:
{the_rainbow_table_colors['G']}

Your observations of species extinction:
{state.current_species}

Your profile of one of the last humans:
{state.current_human}

Your narrative of their intertwined extinction:
{state.current_narrative}

Your survey evaluation:
{state.current_survey}

and finally Claude's decision on what to save from Earth:
{state.current_decision}

Using all of this, create a new SongProposalIteration that captures the essence of extinction, loss, and the bittersweet beauty of endings.
            """
            claude = self._get_claude()
            proposer = claude.with_structured_output(SongProposalIteration)
            try:
                result = proposer.invoke(prompt)
                if isinstance(result, dict):
                    counter_proposal = SongProposalIteration(**result)
                else:
                    counter_proposal = result
            except Exception as e:
                logger.error(f"Anthropic model call failed: {e!s}")
                timestamp = int(time.time() * 1000)
                counter_proposal = SongProposalIteration(
                    iteration_id=f"fallback_error_{timestamp}",
                    bpm=80,
                    tempo="3/4",
                    key="D Minor",
                    rainbow_color="green",
                    title="Fallback: Green Song",
                    mood=["melancholic"],
                    genres=["folk rock"],
                    concept="Fallback stub because Anthropic model unavailable",
                )
            state.counter_proposal = counter_proposal
            get_state_snapshot(
                state,
                "generate_alternate_song_spec_exit",
                state.thread_id,
                "Sub-Arbitrary",
            )
            return state
