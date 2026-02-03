"""
Violet Agent: The Interrogator
Dialectical pressure-testing through adversarial interviews
"""

import logging
import os
import time
from abc import ABC
from pathlib import Path

import yaml
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from app.agents.states.violet_agent_state import VioletAgentState
from app.agents.states.white_agent_state import MainAgentState
from app.agents.workflow.agent_error_handler import agent_error_handler
from app.structures.agents.base_rainbow_agent import BaseRainbowAgent
from app.util.agent_state_utils import get_state_snapshot
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


class VioletAgent(BaseRainbowAgent, ABC):
    """
    Violet Agent: The Sultan of Solipsism

    Dialectical pressure-tester via adversarial interviews.
    Process: select persona → generate questions → simulated interview →
             synthesize → revise proposal
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
            max_tokens=self.settings.max_tokens,
        )
        corpus_path = Path(
            os.getenv("GABE_CORPUS_FILE", "../reference/biographical/gabe_corpus.md")
        )
        self.gabe_corpus = self._load_corpus(corpus_path)

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
        workflow.add_node("simulated_interview", self.simulated_interview)
        workflow.add_node("synthesize_interview", self.synthesize_interview)
        workflow.add_node(
            "generate_alternate_song_spec", self.generate_alternate_song_spec
        )
        # Add edges - direct flow without HitL branching
        workflow.add_edge(START, "select_persona")
        workflow.add_edge("select_persona", "generate_questions")
        workflow.add_edge("generate_questions", "simulated_interview")
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
    @agent_error_handler("The Sultan of Solipsism")
    def select_persona(state: VioletAgentState) -> VioletAgentState:
        """Select a random interviewer persona (or reuse existing on rerun)"""
        get_state_snapshot(
            state, "select_persona_enter", state.thread_id, "The Sultan of Solipsism"
        )
        logger.info("🎭 Selecting interviewer persona...")
        if state.interviewer_persona is not None:
            persona = state.interviewer_persona
            logger.info(
                f"   Reusing existing persona: {persona.first_name} "
                f"{persona.last_name} ({persona.interviewer_type.value}) "
                f"from {persona.publication}"
            )
            get_state_snapshot(
                state, "select_persona_exit", state.thread_id, "The Sultan of Solipsism"
            )
            return state
        persona = VanityPersona()
        logger.info(
            f"   Selected: {persona.first_name} {persona.last_name} "
            f"({persona.interviewer_type.value}) from {persona.publication}"
        )
        logger.info(f"   Stance: {persona.stance}")
        state.interviewer_persona = persona
        get_state_snapshot(
            state, "select_persona_exit", state.thread_id, "The Sultan of Solipsism"
        )
        return state

    @agent_error_handler("The Sultan of Solipsism")
    def generate_questions(self, state: VioletAgentState) -> VioletAgentState:
        """Generate 3 targeted questions using LLM structured output"""
        get_state_snapshot(
            state,
            "generate_questions_enter",
            state.thread_id,
            "The Sultan of Solipsism",
        )
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
                get_state_snapshot(
                    state,
                    "generate_questions_exit",
                    state.thread_id,
                    "The Sultan of Solipsism",
                )
                return state
            except Exception as e:
                error_msg = f"Failed to load mock questions: {e}"
                logger.error(error_msg)
                if block_mode:
                    raise
        persona = state.interviewer_persona
        proposal = state.white_proposal
        prompt = f"""

You are {persona.first_name} {persona.last_name}, a music critic from
{persona.publication}.

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

Indie music journalists are enthusiasts, not music theorists. They know:
✅ Gear (if it's weird/vintage/meme-worthy): "Are you using a Mellotron?"
✅ Vibes/aesthetics: "This has real late-night-highway energy"  
✅ Recording process: "Did you track this in one take?"
✅ Influences: "I'm hearing some Eno in here?"
✅ Weird facts: "Is that a dentist drill sample?"

They do NOT know:
❌ Music theory: Keys, modes, chord progressions, harmonic analysis
❌ Specific BPM numbers (might say "fast" or "downtempo")
❌ Time signatures (unless it's obviously weird like 7/8, and even then they'd say "off-kilter")
❌ Technical terminology: "contrapuntal", "cadential", "modal interchange"

Generate questions from the perspective of an excited music nerd who:
- Reads Pitchfork and Tiny Mix Tapes
- Knows cultural references and gear memes
- Feels vibes more than analyzes structure
- Gets genuinely excited about weird sounds and recording stories
- Might be slightly stoned

Example authentic questions:
- "The synth on this is giving me real Stranger Things basement vibes - what are you running through?"
- "This feels like you recorded it in a bathroom at 3am, is that close?"
- "I'm getting strong [obscure band] energy here - was that intentional or am I projecting?"
- "That sound at 2:34 - is that a broken tape deck or are you just fucking with us?"

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

    @agent_error_handler("The Sultan of Solipsism")
    def simulated_interview(self, state: VioletAgentState) -> VioletAgentState:
        """Simulate Gabe's responses using RAG corpus + LLM"""
        get_state_snapshot(
            state,
            "simulated_interview_enter",
            state.thread_id,
            "The Sultan of Solipsism",
        )
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
                get_state_snapshot(
                    state,
                    "simulated_interview_exit",
                    state.thread_id,
                    "The Sultan of Solipsism",
                )
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


You are Gabe responding to interview questions.

YOUR SIGNATURE MOVE: Start high-register academic/theoretical, then IMMEDIATELY 
undercut with something profane/mundane/lowbrow. This is not optional - the 
bracket switching is the methodology.

THE PATTERN:
1. Dense theory/philosophy (10-30 words)
2. HARD CUT with "but also" / "though" / "like" / "which"
3. Profanity, 1980s TV, pop culture, food, mundane objects, or self-deprecation (10-20 words). 

GOOD EXAMPLES:
✅ "It's really about Heideggerian temporality meets information theory, 
    but also it fucking slaps and the bassline has Smurf's intro music vibes"

✅ "The composition explores ontological instability through temporal 
    displacement, like when you microwave pizza and the center's still 
    frozen but the edges are lava"

✅ "I'm working with constraint-based creativity as productive limitation—
    this, can't get my chocolate in my peanut-butter too easy now"

✅ "The missing beat becomes the space where freedom lives, which I'm 
    aware sounds like a fortune cookie had a baby with a grad seminar"

✅ "The rebracketing methodology reveals how information seeks embodiment 
    through temporal processes—I sound absolutely unhinged saying this 
    out loud, no andy caps"

BAD EXAMPLES (PURE METAPHORIPSUM):
❌ "The composition reveals how consciousness negotiates the boundary 
    between determination and emergence through systematic exploration 
    of constrained possibility spaces" [NO UNDERCUT = BORING]

❌ "I'm exploring the liminal space between human intentionality and 
    machine creativity" [TOO SERIOUS, NO WHIPLASH]

FREQUENCY: Every 2-3 sentences MUST have the pattern. You can't maintain 
epistemic dignity for more than 30 seconds.

ACCEPTABLE UNDERCUT CATEGORIES:
- Profanity: "but it also fucks with", "this is deranged", "absolute banger though"
- Pop culture: "giving divorced dad at Cinnabon energy", "Scooby Doo chase scene vibes"
- Food/mundane: "like gas station sushi", "Wendy's biggie frosty at 2am feeling"
- Internet slang: "no cap", "it goes hard", "this slaps", "unhinged behavior" BUT remember he's *Gen X* and would say Gen Y and Z slang with biter irony and distain.
- Self-awareness: "I sound insane", "pretentious as hell", "hear me out though"
- Physical/bodily: "makes your teeth hurt", "instant headache feeling"

The interviewer might not get the academic stuff, but you don't dumb it down—
you just immediately acknowledge how ridiculous it sounds by pivoting to 
something extremely lowbrow.

This isn't code-switching, it's bracket-demolishing. You're inhabiting both 
registers simultaneously and refusing to pick a lane.

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

        get_state_snapshot(
            state,
            "simulated_interview_exit",
            state.thread_id,
            "The Sultan of Solipsism",
        )
        return state

    @staticmethod
    @agent_error_handler("The Sultan of Solipsism")
    def synthesize_interview(state: VioletAgentState) -> VioletAgentState:
        """Create interview artifact and save transcript"""
        get_state_snapshot(
            state,
            "synthesize_interview_enter",
            state.thread_id,
            "The Sultan of Solipsism",
        )
        logger.info("📝 Synthesizing interview transcript...")
        persona = state.interviewer_persona
        # Create structured artifact
        artifact = CircleJerkInterviewArtifact(
            thread_id=state.thread_id,
            base_path=os.getenv("AGENT_WORK_PRODUCT_BASE_PATH", "chain_artifacts"),
            name=f"{persona.first_name}_{persona.last_name}_interview",
            interviewer_name=f"{persona.first_name} {persona.last_name}",
            publication=persona.publication,
            interviewer_type=persona.interviewer_type.value,
            stance=persona.stance,
            questions=state.interview_questions,
            responses=state.interview_responses,
            was_human_interview=False,  # Always simulated (HitL removed)
            create_dirs=True,
        )
        try:
            artifact.save_file()
            logger.info(f"   Transcript saved: {artifact.get_artifact_path()}")
        except Exception as e:
            logger.error(f"Failed to save transcript: {e}")
        state.circle_jerk_interview = artifact
        state.artifacts.append(artifact)

        get_state_snapshot(
            state,
            "synthesize_interview_exit",
            state.thread_id,
            "The Sultan of Solipsism",
        )
        return state

    @agent_error_handler("The Sultan of Solipsism")
    def generate_alternate_song_spec(self, state: VioletAgentState) -> VioletAgentState:
        """Generate a defensively revised proposal based on an interview"""
        get_state_snapshot(
            state,
            "generate_alternate_song_spec_enter",
            state.thread_id,
            "The Sultan of Solipsism",
        )
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
                get_state_snapshot(
                    state,
                    "generate_alternate_song_spec_exit",
                    state.thread_id,
                    "The Sultan of Solipsism",
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
{interview_artifact.for_prompt() if interview_artifact else "No interview conducted"}

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
                concept=(
                    "Fallback proposal - Violet counter-proposal "
                    "generation unavailable"
                ),
            )

        state.counter_proposal = counter_proposal
        get_state_snapshot(
            state,
            "generate_alternate_song_spec_exit",
            state.thread_id,
            "The Sultan of Solipsism",
        )
        return state
