"""
Violet Agent: The Interrogator
Dialectical pressure-testing through adversarial interviews
"""

import logging
import os
import random
import time
from abc import ABC
from pathlib import Path

import yaml
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

from app.agents.states.violet_agent_state import VioletAgentState
from app.agents.states.white_agent_state import MainAgentState
from app.structures.agents.base_rainbow_agent import BaseRainbowAgent
from app.structures.artifacts.circle_jerk_interview_artifact import (
    CircleJerkInterviewArtifact,
)
from app.structures.concepts.rainbow_table_color import the_rainbow_table_colors
from app.structures.concepts.vanity_persona import VanityPersona
from app.structures.concepts.vanity_interview_question import (
    VanityInterviewQuestion,
    VanityInterviewQuestionOutput,
)
from app.structures.concepts.vanity_interview_response import (
    VanityInterviewResponse,
)
from app.structures.manifests.song_proposal import SongProposalIteration
from app.util.manifest_loader import get_my_reference_proposals

load_dotenv()
logger = logging.getLogger(__name__)
console = Console()


class VioletAgent(BaseRainbowAgent, ABC):
    """
    Violet Agent: The Sultan of Solipsism

    Dialectical pressure-tester via adversarial interviews.
    Process: select persona → generate questions → roll for HitL →
             [human|simulated] interview → synthesize → revise proposal
    """

    def __init__(self, **data):
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
        corpus_path = Path(os.getenv("GABE_CORPUS_FILE", "./violet_assets/gabe_corpus"))
        self.gabe_corpus = self._load_corpus(corpus_path)
        self.hitl_probability = 0.09

    def __call__(self, state: MainAgentState) -> MainAgentState:
        """Main entry point - transform MainAgentState through Violet workflow"""
        violet_state = VioletAgentState(
            thread_id=state.thread_id,
            song_proposals=state.song_proposals,
            white_proposal=state.song_proposals.iterations[-1],
            counter_proposal=None,
            artifacts=[],
            interviewer_persona=None,
            interview_questions=None,
            interview_responses=None,
            circle_jerk_interview=None,
            needs_human_interview=False,
        )
        violet_graph = self.create_graph()
        compiled_graph = violet_graph.compile()
        result = compiled_graph.invoke(violet_state.model_dump())
        if isinstance(result, VioletAgentState):
            final_state = result
        elif isinstance(result, dict):
            final_state = VioletAgentState(**result)
        else:
            raise TypeError(f"Unexpected result type: {type(result)}")
        if final_state.counter_proposal:
            state.song_proposals.iterations.append(final_state.counter_proposal)
        if final_state.artifacts:
            state.artifacts.extend(final_state.artifacts)
        return state

    def create_graph(self) -> StateGraph:
        workflow = StateGraph(VioletAgentState)
        # Add nodes
        workflow.add_node("select_persona", self.select_persona)
        workflow.add_node("generate_questions", self.generate_questions)
        workflow.add_node("roll_for_hitl", self.roll_for_hitl)
        workflow.add_node("human_interview", self.human_interview)
        workflow.add_node("simulated_interview", self.simulated_interview)
        workflow.add_node("synthesize_interview", self.synthesize_interview)
        workflow.add_node(
            "generate_alternate_song_spec", self.generate_alternate_song_spec
        )
        # Add edges
        workflow.add_edge(START, "select_persona")
        workflow.add_edge("select_persona", "generate_questions")
        workflow.add_edge("generate_questions", "roll_for_hitl")
        # Conditional routing after roll
        workflow.add_conditional_edges(
            "roll_for_hitl",
            self.route_after_roll,
            {
                "human_interview": "human_interview",
                "simulated_interview": "simulated_interview",
            },
        )
        # Both paths converge
        workflow.add_edge("human_interview", "synthesize_interview")
        workflow.add_edge("simulated_interview", "synthesize_interview")
        workflow.add_edge("synthesize_interview", "generate_alternate_song_spec")
        workflow.add_edge("generate_alternate_song_spec", END)
        return workflow

    @staticmethod
    def _load_corpus(corpus_dir: Path) -> str:
        """Load all corpus files into a single string for RAG."""
        corpus_texts = []
        if corpus_dir.exists() and corpus_dir.is_dir():
            for file in corpus_dir.glob("*.md"):
                with open(file, "r") as f:
                    corpus_texts.append(f.read())
        elif corpus_dir.exists() and corpus_dir.is_file():
            # Single file
            with open(corpus_dir, "r") as f:
                corpus_texts.append(f.read())
        return "\n\n".join(corpus_texts)

    # =========================================================================
    # NODES
    # =========================================================================

    @staticmethod
    def select_persona(state: VioletAgentState) -> VioletAgentState:
        """Select a random interviewer persona (or reuse existing on rerun)"""
        logger.info("🎭 Selecting interviewer persona...")
        if state.interviewer_persona is not None:
            persona = state.interviewer_persona
            logger.info(
                f"   Reusing existing persona: {persona.first_name} {persona.last_name} "
                f"({persona.interviewer_type.value}) from {persona.publication}"
            )
            return state
        persona = VanityPersona()
        logger.info(
            f"   Selected: {persona.first_name} {persona.last_name} "
            f"({persona.interviewer_type.value}) from {persona.publication}"
        )
        logger.info(f"   Stance: {persona.stance}")
        state.interviewer_persona = persona
        return state

    def generate_questions(self, state: VioletAgentState) -> VioletAgentState:
        """Generate 3 targeted questions using LLM structured output"""
        logger.info("❓ Generating interview questions...")
        mock_mode = os.getenv("MOCK_MODE", "false").lower() == "true"
        block_mode = os.getenv("BLOCK_MODE", "false").lower() == "true"
        if mock_mode:
            try:
                mock_path = (
                    Path(os.getenv("AGENT_MOCK_DATA_PATH"))
                    / "violet_mock_questions.yml"
                )
                with open(mock_path, "r") as f:
                    data = yaml.safe_load(f)
                questions = [VanityInterviewQuestion(**q) for q in data["questions"]]
                state.interview_questions = questions
                logger.info(f"   Loaded {len(questions)} mock questions")
                return state
            except Exception as e:
                error_msg = f"Failed to load mock questions: {e}"
                logger.error(error_msg)
                if block_mode:
                    raise Exception(error_msg)
        persona = state.interviewer_persona
        proposal = state.white_proposal
        prompt = f"""

You are {persona.first_name} {persona.last_name}, a music critic from {persona.publication}.

Your interviewer type: {persona.interviewer_type.value}
Your stance: {persona.stance}
Your approach: {persona.approach}
Your goal: {persona.goal}

Your tactics:
{chr(10).join(f"- {t}" for t in persona.tactics)}

You're interviewing the artist about this song proposal:

CONCEPT: {proposal.concept}
KEY: {proposal.key}
BPM: {proposal.bpm}
MOOD: {proposal.mood}

Generate EXACTLY 3 sharp, provocative questions that embody your persona.
Each question should genuinely challenge the artist from your perspective.

Output as JSON with structure:
{{
  "questions": [
    {{"number": 1, "question": "..."}},
    {{"number": 2, "question": "..."}},
    {{"number": 3, "question": "..."}}
  ]
}}"""

        try:
            structured_llm = self.llm.with_structured_output(
                VanityInterviewQuestionOutput
            )
            result = structured_llm.invoke(prompt)
            state.interview_questions = result.questions
            logger.info(f"   Generated {len(result.questions)} questions")
            for q in result.questions:
                logger.info(f"   Q{q.number}: {q.question[:80]}...")
        except Exception as e:
            logger.error(f"Question generation failed: {e}")
            # Fallback to mock if available
            try:
                mock_path = (
                    Path(os.getenv("AGENT_MOCK_DATA_PATH"))
                    / "violet_mock_questions.yml"
                )
                with open(mock_path, "r") as f:
                    data = yaml.safe_load(f)
                questions = [VanityInterviewQuestion(**q) for q in data["questions"]]
                state.interview_questions = questions
                logger.info("   Using fallback mock questions")
            except ValueError as e:
                logger.warning(f"No mock questions available: {e}")
                # Last resort - generic questions
                state.interview_questions = [
                    VanityInterviewQuestion(
                        number=1, question="Can you explain your creative process?"
                    ),
                    VanityInterviewQuestion(
                        number=2, question="What inspired this work?"
                    ),
                    VanityInterviewQuestion(
                        number=3, question="What do you hope audiences take away?"
                    ),
                ]
                logger.warning("   Using generic fallback questions")
        return state

    def roll_for_hitl(self, state: VioletAgentState) -> VioletAgentState:
        """Roll dice to determine a human vs. simulated interview"""
        roll = random.random()
        needs_human = roll < self.hitl_probability
        logger.info(
            f"🎲 HitL Roll: {roll:.4f} - "
            f"{'🧑 HUMAN NEEDED' if needs_human else '🤖 SIMULATED'}"
        )
        state.needs_human_interview = needs_human
        return state

    @staticmethod
    def human_interview(state: VioletAgentState) -> VioletAgentState:
        """Pause for real Gabe to answer questions (HitL with rich UI)"""
        logger.info("👤 HUMAN INTERVIEW MODE")

        mock_mode = os.getenv("MOCK_MODE", "false").lower() == "true"
        block_mode = os.getenv("BLOCK_MODE", "false").lower() == "true"

        if mock_mode:
            try:
                mock_path = (
                    Path(os.getenv("AGENT_MOCK_DATA_PATH"))
                    / "violet_mock_responses.yml"
                )
                with open(mock_path, "r") as f:
                    data = yaml.safe_load(f)
                responses = [VanityInterviewResponse(**r) for r in data["responses"]]
                state.interview_responses = responses
                logger.info(f"   Loaded {len(responses)} mock responses")
                return state
            except Exception as e:
                error_msg = f"Failed to load mock responses: {e}"
                logger.error(error_msg)
                if block_mode:
                    raise Exception(error_msg)
                # Fall through to real human input
        persona = state.interviewer_persona
        questions = state.interview_questions
        # Display interviewer info
        console.print(
            Panel(
                f"[bold]{persona.first_name} {persona.last_name}[/bold]\n"
                f"[dim]{persona.publication}[/dim]\n\n"
                f"[yellow]Stance:[/yellow] {persona.stance}\n"
                f"[yellow]Goal:[/yellow] {persona.goal}",
                title="🎤 HUMAN INTERVIEW REQUIRED",
                border_style="magenta",
            )
        )
        console.print("\n")
        responses = []
        for q in questions:
            console.print(
                Panel(
                    f"[bold cyan]Question {q.number}:[/bold cyan]\n\n{q.question}",
                    border_style="cyan",
                )
            )
            response_text = Prompt.ask(
                f"\n[green]Your response to Q{q.number}[/green]",
                default="",
                show_default=False,
            )
            responses.append(
                VanityInterviewResponse(
                    question_number=q.number, response=response_text
                )
            )
            console.print("\n")
        state.interview_responses = responses
        logger.info(f"   Collected {len(responses)} human responses")
        return state

    def simulated_interview(self, state: VioletAgentState) -> VioletAgentState:
        """Simulate Gabe's responses using RAG corpus + LLM"""
        logger.info("🤖 SIMULATED INTERVIEW MODE")

        mock_mode = os.getenv("MOCK_MODE", "false").lower() == "true"
        block_mode = os.getenv("BLOCK_MODE", "false").lower() == "true"

        if mock_mode:
            try:
                mock_path = (
                    Path(os.getenv("AGENT_MOCK_DATA_PATH"))
                    / "violet_mock_responses.yml"
                )
                with open(mock_path, "r") as f:
                    data = yaml.safe_load(f)
                responses = [VanityInterviewResponse(**r) for r in data["responses"]]
                state.interview_responses = responses
                logger.info(f"   Loaded {len(responses)} mock responses")
                return state

            except Exception as e:
                error_msg = f"Failed to load mock responses: {e}"
                logger.error(error_msg)
                if block_mode:
                    raise Exception(error_msg)

        # Real simulation with RAG
        persona = state.interviewer_persona
        questions = state.interview_questions
        proposal = state.white_proposal
        responses = []

        for q in questions:
            logger.info(f"   Simulating response to Q{q.number}...")

            # RAG-enhanced prompt
            prompt = f"""You are simulating Gabe's response to an interview question.

GABE'S VOICE & PATTERNS (from past conversations):
{self.gabe_corpus[:3000]}

INTERVIEWER CONTEXT:
Name: {persona.first_name} {persona.last_name}
Publication: {persona.publication}
Stance: {persona.stance}

SONG PROPOSAL BEING DISCUSSED:
Title: {proposal.title}
Concept: {proposal.concept}

QUESTION:
{q.question}

Generate Gabe's response. Channel his voice:
- Technical precision when relevant
- References to rebracketing methodology if applicable
- Occasional dad jokes or self-aware humor
- Defensive but thoughtful when challenged
- Links technical choices to artistic vision
- May push back on false dichotomies

Keep response 2-4 sentences. Output as JSON:
{{"question_number": {q.number}, "response": "..."}}"""

            try:
                # Structured output
                structured_llm = self.llm.with_structured_output(
                    VanityInterviewResponse
                )
                response = structured_llm.invoke(prompt)
                response.question_number = q.number  # Ensure match
                responses.append(response)

                logger.info(f"      → {response.response[:60]}...")

            except Exception as e:
                logger.error(f"Response simulation failed for Q{q.number}: {e}")
                responses.append(
                    VanityInterviewResponse(
                        question_number=q.number,
                        response=f"[Simulated response unavailable for Q{q.number}]",
                    )
                )

        state.interview_responses = responses
        logger.info(f"   Generated {len(responses)} simulated responses")

        return state

    @staticmethod
    def synthesize_interview(state: VioletAgentState) -> VioletAgentState:
        """Create interview artifact and save transcript"""
        logger.info("📝 Synthesizing interview transcript...")
        persona = state.interviewer_persona
        # Create structured artifact
        artifact = CircleJerkInterviewArtifact(
            name=f"{persona.first_name}_{persona.last_name}_interview",
            interviewer_name=f"{persona.first_name} {persona.last_name}",
            publication=persona.publication,
            interviewer_type=persona.interviewer_type.value,
            stance=persona.stance,
            questions=state.interview_questions,
            responses=state.interview_responses,
            was_human_interview=state.needs_human_interview,
            create_dirs=True,
        )
        try:
            artifact.generate_file()
            logger.info(f"   Transcript saved: {artifact.get_file_path()}")
        except Exception as e:
            logger.error(f"Failed to save transcript: {e}")
        state.circle_jerk_interview = artifact
        state.artifacts.append(artifact)

        return state

    def generate_alternate_song_spec(self, state: VioletAgentState) -> VioletAgentState:
        """Generate a defensively revised proposal based on an interview"""
        logger.info("🛡️ Generating defensive counter-proposal...")
        mock_mode = os.getenv("MOCK_MODE", "false").lower() == "true"
        block_mode = os.getenv("BLOCK_MODE", "false").lower() == "true"
        if mock_mode:
            try:
                mock_path = (
                    Path(os.getenv("AGENT_MOCK_DATA_PATH"))
                    / "violet_counter_proposal_mock.yml"
                )
                with open(mock_path, "r") as f:
                    data = yaml.safe_load(f)
                counter_proposal = SongProposalIteration(**data)
                state.counter_proposal = counter_proposal
                logger.info(
                    f"   Loaded mock counter-proposal: {counter_proposal.title}"
                )
                return state
            except Exception as e:
                error_msg = f"Failed to load mock counter-proposal: {e}"
                logger.error(error_msg)
                if block_mode:
                    raise Exception(error_msg)

        # Real generation with defensive revision
        interview_artifact = state.circle_jerk_interview

        prompt = f"""You are revising a song proposal after a challenging interview.

ORIGINAL PROPOSAL:
{state.white_proposal.model_dump_json(indent=2)}

INTERVIEW CONTEXT:
{interview_artifact.artifact_report if interview_artifact else "No interview conducted"}

INTERVIEWER'S STANCE: {state.interviewer_persona.stance}

Now create a REVISED proposal that takes the interview with a "grain of salt":
- Acknowledge valid criticisms but defend your vision
- Might double-down on challenged elements
- Might lean HARDER into criticized aspects
- Might explain misunderstandings
- Can be defensive but should strengthen the proposal

This is dialectical synthesis - thesis/antithesis → synthesis.

Reference works in this artist's style (pay attention to 'concept' property):
{get_my_reference_proposals('V')}

CRITICAL: Your 'rainbow_color' property must be:
{the_rainbow_table_colors['V']}

Output a COMPLETE revised SongProposalIteration with ALL fields populated.
The revision should be INFORMED BY but not DEFEATED BY the criticism."""

        try:
            structured_llm = self.llm.with_structured_output(SongProposalIteration)
            counter_proposal = structured_llm.invoke(prompt)
            logger.info(f"   Generated counter-proposal: {counter_proposal.title}")

        except Exception as e:
            logger.error(f"Counter-proposal generation failed: {e}")

            # Fallback
            timestamp = int(time.time() * 1000)
            counter_proposal = SongProposalIteration(
                iteration_id=f"fallback_violet_{timestamp}",
                bpm=120,
                tempo="4/4",
                key="F Major",
                rainbow_color="violet",
                title="Fallback: Defensive Violet Response",
                mood=["defiant"],
                genres=["experimental"],
                concept="Fallback proposal - Violet counter-proposal generation unavailable",
            )

        state.counter_proposal = counter_proposal
        return state

    # =========================================================================
    # ROUTING
    # =========================================================================

    @staticmethod
    def route_after_roll(state: VioletAgentState) -> str:
        """Route to a human or simulated interview based on roll"""
        if state.get("needs_human_interview", False):
            return "human_interview"
        else:
            return "simulated_interview"
