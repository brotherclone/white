import os
import random
import uuid
import yaml
import logging

from abc import ABC
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import START, END
from langgraph.graph import StateGraph

from app.agents.enums.sigil_state import SigilState
from app.agents.enums.sigil_type import SigilType
from app.agents.models.agent_settings import AgentSettings
from app.agents.base_rainbow_agent import BaseRainbowAgent
from app.agents.models.evp_artifact import EVPArtifact
from app.agents.models.sigil_artifact import SigilArtifact
from app.agents.states.base_rainbow_agent_state import BaseRainbowAgentState
from app.agents.states.black_agent_state import BlackAgentState
from app.agents.states.white_agent_state import MainAgentState
from app.agents.tools.audio_tools import get_audio_segments_as_chain_artifacts, \
    create_audio_mosaic_chain_artifact, create_blended_audio_chain_artifact
from app.agents.tools.magick_tools import SigilTools
from app.agents.tools.speech_tools import chain_artifact_file_from_speech_to_text
from app.agents.tools.text_tools import save_artifact_file_to_md
from app.structures.concepts.rainbow_table_color import the_rainbow_table_colors
from app.structures.concepts.yes_or_no import YesOrNo
from app.structures.manifests.song_proposal import SongProposalIteration, SongProposal
from app.util.manifest_loader import get_my_reference_proposals
from app.reference.mcp.todoist.main import (
    create_sigil_charging_task
)

load_dotenv()
logging.basicConfig(level=logging.INFO)


# ToDo: Figure out of this can be moved
def skip_chance(x):
    def decorator(f):
        return f

    return decorator


class BlackAgent(BaseRainbowAgent, ABC):
    """Keeper of the Conjurer's Thread"""

    def __init__(self, **data):
        if 'settings' not in data or data['settings'] is None:
            data['settings'] = AgentSettings()

        super().__init__(**data)

        if self.settings is None:
            self.settings = AgentSettings()

        self.llm = ChatAnthropic(
            temperature=self.settings.temperature,
            api_key=self.settings.anthropic_api_key,
            model_name=self.settings.anthropic_model_name,
            max_retries=self.settings.max_retries,
            timeout=self.settings.timeout,
            stop=self.settings.stop
        )
        self.current_session_sigils = []
        self.state_graph = BlackAgentState(
            thread_id=f"black_{uuid.uuid4()}",
            song_proposal=None,
            white_proposal=None,
            counter_proposal=None,
            artifacts=[],
            awaiting_human_action=False,
            human_instructions="",
            pending_human_tasks=[],
            should_update_proposal_with_evp=False,
            should_update_proposal_with_sigil=False,
        )

    def __call__(self, state: MainAgentState) -> MainAgentState:

        """Entry point when White Agent invokes Black Agent"""

        current_proposal = state.song_proposals.iterations[-1]
        # Reinitialize the BlackAgentState with the current song proposal
        black_state = BlackAgentState(
            white_proposal=current_proposal,
            song_proposals=state.song_proposals,
            thread_id=state.thread_id,
            artifacts=[],
            pending_human_tasks=[],
            awaiting_human_action=False
        )
        if not hasattr(self, '_compiled_workflow'):
            self._compiled_workflow = self.create_graph().compile(
                checkpointer=MemorySaver(),
                interrupt_before=["await_human_action"]
            )
        black_config: RunnableConfig = {"configurable": {"thread_id": f"{state.thread_id}"}}
        result = self._compiled_workflow.invoke(black_state.model_dump(), config=black_config)
        snapshot = self._compiled_workflow.get_state(black_config)
        if snapshot.next:
            pass
        else:
            state.song_proposals = result.get("song_proposals") or state.song_proposals
            if result.get("counter_proposal"):
                state.song_proposals.iterations.append(result["counter_proposal"])
        return state

    def create_graph(self) -> StateGraph:
        """
        Create the BlackAgent's internal workflow graph
        1. The black agent calls the llm to generate a counter-proposal
        2. The black agent generates an EVP artifact
        3. The black agent evaluates the EVP artifact for potential insights, if any go to 4 if none skip to 6
        4. Using the EVP insights, the Black agent adjusts the counter-proposal to reflect the insights
        5. The black agent returns the counter-proposal to the White Agent
        6. The black agent may generate a sigil artifact to accompany the counter-proposal
        7. The black agent creates a Todoist task for the human to charge the sigil
        8. The black agent pauses the workflow, awaiting human action
        9. Once the human completes the task, the black agent resumes
        10. The black agent updates the counter-proposal to reflect the charged sigil if need be then goto 5.

        """
        black_workflow = StateGraph(BlackAgentState)
        # Nodes
        black_workflow.add_node("generate_alternate_song_spec", self.generate_alternate_song_spec)
        black_workflow.add_node("generate_evp", self.generate_evp)
        black_workflow.add_node("evaluate_evp", self.evaluate_evp)
        black_workflow.add_node("update_alternate_song_spec_with_evp", self.update_alternate_song_spec_with_evp)
        black_workflow.add_node("generate_sigil", self.generate_sigil)
        black_workflow.add_node("await_human_action", self.await_human_action)
        black_workflow.add_node("update_alternate_song_spec_with_sigil", self.update_alternate_song_spec_with_sigil)
        # Edges
        black_workflow.add_edge(START, "generate_alternate_song_spec")
        black_workflow.add_edge("generate_alternate_song_spec", "generate_evp")
        black_workflow.add_edge("generate_evp", "evaluate_evp")
        black_workflow.add_conditional_edges(
            "evaluate_evp",
            self.route_after_evp_evaluation,
            {
                "evp": "update_alternate_song_spec_with_evp",
                "sigil": "generate_sigil"
            }
        )
        black_workflow.add_edge("update_alternate_song_spec_with_evp", "generate_sigil")
        black_workflow.add_conditional_edges(
            "generate_sigil",
            self.route_after_sigil_chance,
            {
                "human": "await_human_action",
                "done": END
            }
        )
        black_workflow.add_edge("await_human_action", "update_alternate_song_spec_with_sigil")
        black_workflow.add_edge("await_human_action", END)
        return black_workflow

    @staticmethod
    def route_after_evp_evaluation(state: BlackAgentState) -> str:
        return "evp" if state.should_update_proposal_with_evp else "sigil"

    @staticmethod
    def route_after_sigil_chance(state: BlackAgentState) -> str:
        return "human" if state.should_update_proposal_with_sigil else "done"

    def generate_alternate_song_spec(self, state: BlackAgentState) -> BlackAgentState:

        """Generate an initial counter-proposal"""

        mock_mode = os.getenv("MOCK_MODE", "false").lower() == "true"

        if mock_mode:
            with open("/Volumes/LucidNonsense/White/app/agents/mocks/black_counter_proposal_mock.yml", "r") as f:
                data = yaml.safe_load(f)
            counter_proposal = SongProposalIteration(**data)
            state.counter_proposal = counter_proposal
            return state
        else:
            prompt = f"""
            You are writing creative fiction about an experimental musician creating concept albums.
    
            Context: This character is an artist working in the experimental music space, creating 
            a concept album in 2016 after David Bowie's death and Trump's election. The character 
            uses themes of surveillance, control systems, and artistic resistance in their work.
    
            The artist is responding to this song proposal and wants to enhance its themes of 
            autonomy, authenticity, and resistance to control systems through creative artistic choices.
    
            Current song proposal:
            {state.white_proposal}
    
            Reference works in this artist's style paying close attention to 'concept' property:
            {get_my_reference_proposals('Z')}
            
            In your counter proposal your 'rainbow_color' property should always be:
            {the_rainbow_table_colors['Z']}
    
            Create a counter-proposal that enhances the artistic and thematic depth of this song.
            Focus on musical elements, lyrical themes, and production choices that express creative 
            resistance and psychological liberation. Try to avoid being too "on the nose" or literal.
            Ambiguity and subtlety are valued.
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
                logging.error(f"Anthropic model call failed: {e!s}")
                counter_proposal = SongProposalIteration(
                    iteration_id=str(uuid.uuid4()),
                    bpm=120,
                    tempo="4/4",
                    key="C Major",
                    rainbow_color="black",
                    title="Fallback: Black Song",
                    mood=["dark"],
                    genres=["experimental"],
                    concept="Fallback stub because Anthropic model unavailable"
                )

            state.counter_proposal = counter_proposal
            return state

    @skip_chance(0.75)
    def generate_sigil(self, state: BlackAgentState) -> BlackAgentState:

        """Generate a sigil artifact and create a Todoist task for charging"""

        mock_mode = os.getenv("MOCK_MODE", "false").lower() == "true"

        if mock_mode:
            mock_path = "/Volumes/LucidNonsense/White/app/agents/mocks/black_sigil_artifact_mock.yml"
            if random.random() < 0.75:
                state.should_update_proposal_with_sigil = False
                return state
            try:
                with open(mock_path, "r") as f:
                    data = yaml.safe_load(f)
                sigil_artifact = SigilArtifact(**data)
                state.artifacts.append(sigil_artifact)
                state.awaiting_human_action = True
                state.human_instructions = "MOCK: Sigil charging task would be created"
                state.should_update_proposal_with_sigil = True
                return state
            except FileNotFoundError:
                logging.warning(f"Mock file not found at {mock_path}, using real generation")

        sigil_maker = SigilTools()
        current_proposal = state.counter_proposal

        prompt = f"""
        Distill this counter-proposal into a short, actionable wish statement that captures how 
        the song could embody higher occult meaning and resistance against the Demiurge.

        Counter-proposal:
        Title: {current_proposal.title}
        Concept: {current_proposal.concept}
        Mood: {', '.join(current_proposal.mood)}

        Format: A single sentence starting with "I will..." or "This song will..."
        Example: "I will weave hidden frequencies that awaken dormant resistance."
        """
        claude = self._get_claude()
        wish_response = claude.invoke(prompt)
        wish_text = wish_response.content
        statement_of_intent = sigil_maker.create_statement_of_intent(wish_text, True)
        description, components = sigil_maker.generate_word_method_sigil(statement_of_intent)
        charging_instructions = sigil_maker.charge_sigil()
        sigil_artifact = SigilArtifact(
            wish=wish_text,
            statement_of_intent=statement_of_intent,
            glyph_description=description,
            glyph_components=components,
            sigil_type=SigilType.WORD_METHOD,
            activation_state=SigilState.CREATED,  # Not charged yet!
            charging_instructions=charging_instructions,
            thread_id=state.thread_id,
            chain_artifact_type="sigil"
        )

        state.artifacts.append(sigil_artifact)

        try:
            task_result = create_sigil_charging_task(
                sigil_description=description,
                charging_instructions=charging_instructions,
                song_title=current_proposal.title,
                section_name="Black Agent - Sigil Work"
            )

            state.pending_human_tasks.append({
                "type": "sigil_charging",
                "task_id": task_result["id"],
                "task_url": task_result["url"],
                "artifact_index": len(state.artifacts) - 1,
                "sigil_wish": wish_text
            })

            state.awaiting_human_action = True
            state.human_instructions = f"""
                                        🜏 SIGIL CHARGING REQUIRED
                                        A sigil has been generated for '{current_proposal.title}'.
                                        **Todoist Task:** {task_result['url']}
                                        **Wish:** {wish_text}
                                        **Glyph:** {description}
                                        **Instructions:**
                                        {charging_instructions}
                                        Mark the Todoist task complete after charging, then resume the workflow.
                                        """
            state.should_update_proposal_with_sigil = True
            logging.info(f"✓ Created Todoist task for sigil charging: {task_result['id']}")

        except Exception as e:
            logging.error(f"✗ Failed to create Todoist task: {e}")
            state.awaiting_human_action = True
            state.should_update_proposal_with_sigil = True
            state.human_instructions = f"""
                                        ⚠️ SIGIL CHARGING REQUIRED (Todoist task creation failed)
                                        
                                        Manually charge the sigil for '{current_proposal.title}':
                                        
                                        **Wish:** {wish_text}
                                        **Glyph:** {description}
                                        
                                        {charging_instructions}
                                        """
        return state

    @staticmethod
    def generate_evp(state: BlackAgentState) -> BlackAgentState:

        """Generate an EVP artifact and optionally create an analysis task"""

        mock_mode = os.getenv("MOCK_MODE", "false").lower() == "true"

        if mock_mode:
            with open("/Volumes/LucidNonsense/White/app/agents/mocks/black_evp_artifact_mock.yml", "r") as f:
                data = yaml.safe_load(f)
            evp_artifact = EVPArtifact(**data)
            state.artifacts.append(evp_artifact)
            return state
        current_proposal = state.counter_proposal
        segments = get_audio_segments_as_chain_artifacts(
            2.0, 4,
            the_rainbow_table_colors['Z'],
            state.thread_id
        )
        mosaic = create_audio_mosaic_chain_artifact(
            segments, 100,
            getattr(current_proposal, 'target_length', 10),
            state.thread_id
        )
        blended = create_blended_audio_chain_artifact(
            mosaic, 0.66,
            state.thread_id
        )
        transcript = chain_artifact_file_from_speech_to_text(
            blended,
            state.thread_id
        )
        evp_artifact = EVPArtifact(
            audio_segments=segments,
            transcript=transcript,
            audio_mosaic=mosaic,
            noise_blended_audio=blended,
            thread_id=state.thread_id,
        )
        state.artifacts.append(evp_artifact)
        return state

    @staticmethod
    def await_human_action(state: BlackAgentState) -> BlackAgentState:
        """
        Node that workflow interrupts at.
        """
        logging.info("⏸️  Workflow interrupted - awaiting human action on sigil charging")
        return state

    def evaluate_evp(self, state: BlackAgentState) -> BlackAgentState:
        mock_mode = os.getenv("MOCK_MODE", "false").lower() == "true"
        if mock_mode:
            roll = random.choice([0, 1])
            if roll:
                state.should_update_proposal_with_evp = True
            else:
                state.should_update_proposal_with_evp = False
            return state
        else:
            prompt = f"""
            You are helping a musician create a creative fiction song about an experimental musician 
            working in the experimental music space. You have just generated an EVP (Electronic Voice Phenomenon)
            artifact consisting of audio segments and a transcript. Now, your task is to evaluate the transcript
            and see if there are any surreal or lyrical results that could help you refocus your song proposal. At this point
            you only need to reply with a True or False property.
            
            Here's an example of what might warrant a True response:
            'do turn' 'caliphate murloc' 'a simple bloodline'
            
            Here's an example of what might warrant a False response which is more likely:
            'i' 'me me' 'to' 'hi' 'be be'
            
            Here's your previous counter-proposal:
            {state.counter_proposal}
            
            Here's the EVP transcript:
            {state.artifacts[-1].transcript}
            
            """
            claude = self._get_claude()
            proposer = claude.with_structured_output(YesOrNo)
            try:
                result = proposer.invoke(prompt)
                if isinstance(result, dict):
                    state.should_update_proposal_with_evp = result["answer"]
                else:
                    return self.generate_sigil(state)
            except Exception as e:
                logging.error(f"Anthropic model call failed: {e!s}")
            return state

    def update_alternate_song_spec_with_evp(self, state: BlackAgentState) -> BlackAgentState:
        mock_mode = os.getenv("MOCK_MODE", "false").lower() == "true"
        if mock_mode:
            with open("/Volumes/LucidNonsense/White/app/agents/mocks/black_counter_proposal_after_evp_mock.yml",
                      "r") as f:
                data = yaml.safe_load(f)
                evp_counter_proposal = SongProposalIteration(**data)
                state.counter_proposal = evp_counter_proposal
            return state
        prompt = f"""
                You are helping a musician create a creative fiction song about an experimental musician 
                working in the experimental music space. You have just generated an EVP (Electronic Voice Phenomenon)
                artifact consisting of audio segments and a transcript. Now, you need to update your song counter-proposal
                to reflect the results of your EVP analysis. At this point you only need to reply with a counter-proposal
                that reflects the results of your EVP analysis.
                 
                As an example, imagine this was your original counter-proposal:
                
                    bpm: 100
                    tempo: 4/4
                    key: B minor
                    rainbow_color: {the_rainbow_table_colors['Z']}
                    title: "Whispers of the Abyss"
                    mood: ["mysterious", "haunting", "ethereal"]
                    genres: ["ambient", "darkwave", "experimental"]
                    concept: This song is from the perspective of man who can't remember the previous day. There is 
                    a hole in his memory that he can't explain. As the song progresses, he begins to hear whispers and voices that seem to
                    come from the abyss of his forgotten memories. The song explores themes of memory, identity, and the unknown.
                
                
                And this is an example of the EVP transcript:
                
                    ['cross keep lucky', 'antidote carpet', 'danny want it']
                
                Then your updated counter-proposal could be some like:
                
                    bpm: 104
                    tempo: 4/4
                    key: A minor
                    rainbow_color: {the_rainbow_table_colors['Z']}
                    title: "Lucky Danny's Antidote"
                    mood: ["foreboding", "haunting", "jovial"]
                    genres: ["folk rock", "tavern song", "alternative"]
                    concept: Danny can't remember the previous day. And he should consider himself lucky as he's been
                    through a horrendous experience. Warned not to try to remember he persists in try to regain his memory
                    by retracing the past day's steps. As he does so, each clue brings him closer to an antidote for his amnesia, 
                    but also deeper into a surreal and haunting journey through his own psyche.The song explores themes of memory,
                    identity, and the unknown.
         
                Your actual counter-proposal was:
                    {state.counter_proposal}
                
                The counter-proposal and new updated counter-proposal should have the 'rainbow_color' property set to:
                    {the_rainbow_table_colors['Z']}
                
                And here is the EVP transcript:
                    {state.artifacts[-1].transcript}
               """
        claude = self._get_claude()
        proposer = claude.with_structured_output(SongProposalIteration)
        try:
            result = proposer.invoke(prompt)
            if isinstance(result, dict):
                updated_proposal = SongProposalIteration(**result)
                state.song_proposals.iterations.append(self.counter_proposal)
                state.counter_proposal = updated_proposal
                return state
        except Exception as e:
            logging.error(f"Anthropic model call failed: {e!s}")
        return state

    def update_alternate_song_spec_with_sigil(self, state: BlackAgentState) -> BlackAgentState:
        mock_mode = os.getenv("MOCK_MODE", "false").lower() == "true"
        if mock_mode:
            with open("/Volumes/LucidNonsense/White/app/agents/mocks/black_counter_proposal_after_sigil_mock.yml",
                      "r") as f:
                data = yaml.safe_load(f)
            sigil_counter_proposal = SongProposalIteration(**data)
            state.counter_proposal = sigil_counter_proposal
            return state
        else:
            try:
                previous_sigil_artifact = SigilArtifact(**state.artifacts[-1])
                previous_sigil_artifact.artifact_type = f"""
                    Wish:{previous_sigil_artifact.wish}
                    Intent:{previous_sigil_artifact.statement_of_intent}
                    Description:{previous_sigil_artifact.glyph_description}
                    Type:{previous_sigil_artifact.sigil_type.value}
                    Instructions:{previous_sigil_artifact.charging_instructions}
                    Components: {",".join(previous_sigil_artifact.glyph_components or [])}
                """
            except Exception as e:
                logging.error(f"Failed to parse sigil artifact: {e!s}")
                return state
            prompt = f"""
        You are helping a musician create a creative fiction song about an experimental musician 
        working in the experimental music space. You have just generated a sigil artifact and used a
        ToDoist task to have your human charge the sigil. Now, you need to update your song counter-proposal
        to reflect the results of your sigil charge. At this point you only need to reply with a counter-proposal
        that reflects the results of your sigil charge.
        
         As an example, imagine this was your original counter-proposal:
                
                    bpm: 100
                    tempo: 4/4
                    key: B minor
                    rainbow_color: {the_rainbow_table_colors['Z']}
                    title: "Whispers of the Abyss"
                    mood: ["mysterious", "haunting", "ethereal"]
                    genres: ["ambient", "darkwave", "experimental"]
                    concept: This song is from the perspective of man who can't remember the previous day. There is 
                    a hole in his memory that he can't explain. As the song progresses, he begins to hear whispers and voices that seem to
                    come from the abyss of his forgotten memories. The song explores themes of memory, identity, and the unknown.
        
         Here's the Sigil you previously created:
            
            {state.artifacts[-1]}

        Then your updated counter-proposal could be some like:
            
            bpm: 135
            tempo: 4/4
            key: B major
            rainbow_color: {the_rainbow_table_colors['Z']}
            title: "Shut Up and Play"
            mood: ["energetic", "triumphant", "liberating"]
            genres: ["punk", "rock", "experimental", "no wave"]
            concept: This song celebrates the banishing of destructive inner thoughts. The narrator of the song is feeling
            relief and liberation as they silence the negative voices in their head that have been holding them back creatively.
        
        
        Your actual counter-proposal was:
        {state.counter_proposal}
        
        The counter-proposal and new updated counter-proposal should have the 'rainbow_color' property set to:
        {the_rainbow_table_colors['Z']}
        """
            claude = self._get_claude()
            proposer = claude.with_structured_output(SongProposalIteration)
            try:
                result = proposer.invoke(prompt)
                if isinstance(result, dict):
                    updated_proposal = SongProposalIteration(**result)
                    state.song_proposals.iterations.append(self.counter_proposal)
                    state.counter_proposal = updated_proposal

                    return state
            except Exception as e:
                logging.error(f"Anthropic model call failed: {e!s}")

            return state
