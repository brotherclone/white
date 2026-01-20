import logging
import os
import random
import re
import time
import yaml

from abc import ABC
from typing import Dict, List
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import StrOutputParser
from langgraph.constants import START, END
from langgraph.graph.state import StateGraph

from app.agents.states.indigo_agent_state import IndigoAgentState
from app.agents.states.white_agent_state import MainAgentState
from app.structures.agents.agent_settings import AgentSettings
from app.structures.agents.base_rainbow_agent import BaseRainbowAgent

from app.structures.artifacts.infranym_text_artifact import InfranymTextArtifact
from app.structures.artifacts.infranym_audio_artifact import InfranymAudioArtifact
from app.structures.artifacts.infranym_encoded_image_artifact import (
    InfranymEncodedImageArtifact,
)
from app.structures.artifacts.infranym_text_render_artifact import (
    InfranymTextRenderArtifact,
)
from app.structures.enums.image_text_style import ImageTextStyle
from app.structures.enums.infranym_medium import InfranymMedium
from app.structures.enums.infranym_method import InfranymMethod
from app.structures.manifests.song_proposal import SongProposalIteration
from app.agents.tools.infranym_midi_tools import (
    generate_note_cipher,
    generate_morse_duration,
    add_carrier_melody_to_artifact,
)
from app.agents.tools.infranym_text_tools import (
    create_acrostic_encoding,
    create_riddle_encoding,
    create_anagram_encoding,
)
from app.structures.music.core.key_signature import KeySignature
from app.structures.music.core.notes import Note

load_dotenv()

logger = logging.getLogger(__name__)


class IndigoAgent(BaseRainbowAgent, ABC):
    """
    Decider Tangents - Hides information in triple-layer infranyms.

    FULLY INTEGRATED with all artifact types:
    - Text (acrostic, riddle, anagram)
    - Audio (multi-layer TTS puzzles)
    - MIDI (note cipher, morse duration)
    - Image (3-layer: metadata, LSB stego, spread spectrum)
    """

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
        indigo_state = IndigoAgentState(
            thread_id=state.thread_id,
            song_proposals=state.song_proposals,
            white_proposal=state.song_proposals.iterations[-1],
            counter_proposal=None,
            artifacts=[],
            secret_name=None,
            infranym_method=None,
            infranym_encoded_image=None,
            infranym_text_render=None,
            infranym_text=None,
            infranym_audio=None,
            infranym_midi=None,
            concepts=None,
            letter_bank=None,
            surface_name=None,
            anagram_attempts=0,
            anagram__attempt_max=3,
            anagram_valid=False,
            method_constraints=None,
        )
        indigo_graph = self.create_graph()
        compiled_graph = indigo_graph.compile()
        result = compiled_graph.invoke(indigo_state.model_dump())
        if isinstance(result, IndigoAgentState):
            final_state = result
        elif isinstance(result, dict):
            final_state = IndigoAgentState(**result)
        else:
            raise TypeError(f"Unexpected result type: {type(result)}")
        if final_state.counter_proposal:
            state.song_proposals.iterations.append(final_state.counter_proposal)
        if final_state.artifacts:
            state.artifacts = final_state.artifacts
        return state

    def create_graph(self) -> StateGraph:
        """
        1. get_concepts - Extract thematic elements
        2. spy_choose_letter_bank - SPY selects optimal letters
        3. fool_arrange_secret - FOOL creates secret name
        4. spy_arrange_surface - SPY creates surface name (decoy)
        5. validate_anagram - Check if valid, retry if needed
        6. choose_infranym_method - Weighted method selection
        7. implement_infranym_method - Encode the puzzle
        8. generate_alternate_song_spec - Create counter-proposal
        """
        work_flow = StateGraph(IndigoAgentState)
        work_flow.add_node("get_concepts", self.get_concepts)
        work_flow.add_node("spy_choose_letter_bank", self.spy_choose_letter_bank)
        work_flow.add_node("fool_arrange_secret", self.fool_arrange_secret)
        work_flow.add_node("spy_arrange_surface", self.spy_arrange_surface)
        work_flow.add_node("validate_anagram", self.validate_anagram)
        work_flow.add_node("algorithmic_fallback", self.algorithmic_fallback)
        work_flow.add_node("choose_infranym_method", self.choose_infranym_method)
        work_flow.add_node("implement_infranym_method", self.implement_infranym_method)
        work_flow.add_node(
            "generate_alternate_song_spec", self.generate_alternate_song_spec
        )

        # Text-specific nodes (acrostic/riddle need LLM generation)
        work_flow.add_node("choose_text_method", self.choose_text_method)
        work_flow.add_node("generate_acrostic_lines", self.generate_acrostic_lines)
        work_flow.add_node("generate_riddle_text", self.generate_riddle_text)
        work_flow.add_node("assemble_text_artifact", self.assemble_text_artifact)

        work_flow.add_edge(START, "get_concepts")
        work_flow.add_edge("get_concepts", "spy_choose_letter_bank")
        work_flow.add_edge("spy_choose_letter_bank", "fool_arrange_secret")
        work_flow.add_edge("fool_arrange_secret", "spy_arrange_surface")
        work_flow.add_edge("spy_arrange_surface", "validate_anagram")
        work_flow.add_conditional_edges(
            "validate_anagram",
            self._should_retry_anagram,
            {
                "retry": "spy_choose_letter_bank",
                "fallback": "algorithmic_fallback",
                "continue": "choose_infranym_method",
            },
        )
        work_flow.add_edge("algorithmic_fallback", "choose_infranym_method")
        work_flow.add_edge("choose_infranym_method", "implement_infranym_method")

        # Conditional routing after implementation
        work_flow.add_conditional_edges(
            "implement_infranym_method",
            self._should_route_to_text_nodes,
            {
                "choose_text_method": "choose_text_method",
                "generate_alternate_song_spec": "generate_alternate_song_spec",
            },
        )

        # Text method routing
        work_flow.add_conditional_edges(
            "choose_text_method",
            self._should_generate_text_content,
            {
                "generate_acrostic_lines": "generate_acrostic_lines",
                "generate_riddle_text": "generate_riddle_text",
                "assemble_text_artifact": "assemble_text_artifact",
            },
        )

        work_flow.add_edge("generate_acrostic_lines", "assemble_text_artifact")
        work_flow.add_edge("generate_riddle_text", "assemble_text_artifact")
        work_flow.add_edge("assemble_text_artifact", "generate_alternate_song_spec")
        work_flow.add_edge("generate_alternate_song_spec", END)
        return work_flow

    @staticmethod
    def get_concepts(state: IndigoAgentState) -> IndigoAgentState:
        concepts = []
        if state.white_proposal:
            concepts.append(state.white_proposal.concept)
            concepts.extend(state.white_proposal.mood)
            concepts.extend(state.white_proposal.genres)
        for iteration in state.song_proposals.iterations:
            if iteration.concept and iteration.concept not in concepts:
                concepts.append(iteration.concept)
            for mood in iteration.mood:
                if mood not in concepts:
                    concepts.append(mood)
        state.concepts = ", ".join(concepts[:10])
        state.anagram_attempts = 0
        return state

    def spy_choose_letter_bank(self, state: IndigoAgentState) -> IndigoAgentState:
        """
        SPY selects the optimal letter distribution for anagrams.
        Now a traceable graph node!
        """
        mock_mode = os.getenv("MOCK_MODE", "false").lower() == "true"
        block_mode = os.getenv("BLOCK_MODE", "false").lower() == "true"
        state.anagram_attempts += 1
        logger.info(f"🕵️ SPY choosing letter bank (attempt {state.anagram_attempts})")
        if mock_mode:
            try:
                with open(
                    f"{os.getenv('AGENT_MOCK_DATA_PATH')}/spy_choose_letter_bank_mock.yml",
                    "r",
                ) as f:
                    data = yaml.safe_load(f)
                    letters = data["letters"]
                    state.letter_bank = "".join(sorted(letters))
            except Exception as e:
                error_msg = f"Failed to read mock letter bank: {e!s}"
                logger.error(error_msg)
                if block_mode:
                    raise Exception(error_msg)
        else:
            prompt = f"""
You are the SPY aspect of Decider Tangents.

Concepts: {state.concepts}

Choose a LETTER BANK (14-18 letters) that:
- Has 4-6 vowels (A, E, I, O, U)
- Has common consonants (R, S, T, N, L, M, D)
- Can form multiple meaningful words
- Relates sonically to the concepts

Think strategically: these letters must form TWO different meaningful phrases.

Respond with ONLY the letters (uppercase, no spaces), like: AEILNORSTDM
"""
            try:
                chain = self.llm | StrOutputParser()
                response = chain.invoke(prompt).strip().upper()
                letters = "".join(c for c in response if c.isalpha())
                state.letter_bank = "".join(sorted(letters))
                return state
            except Exception as e:
                msg = f"❌ SPY error: {e}"
                if block_mode:
                    raise Exception(msg)
                logger.error(msg)
                state.letter_bank = "AEILNORSTDM"
        return state

    def fool_arrange_secret(self, state: IndigoAgentState) -> IndigoAgentState:
        """
        FOOL arranges letters into SECRET name.
        Now a traceable graph node!
        """
        mock_mode = os.getenv("MOCK_MODE", "false").lower() == "true"
        block_mode = os.getenv("BLOCK_MODE", "false").lower() == "true"
        logger.info(f"🃏 FOOL arranging secret from: {state.letter_bank}")
        if mock_mode:
            try:
                with open(
                    f"{os.getenv('AGENT_MOCK_DATA_PATH')}/fool_rearrange_mock.yml",
                    "r",
                ) as f:
                    data = yaml.safe_load(f)
                    secret_name = data["secret_name"]
                    state.secret_name = secret_name
            except Exception as e:
                error_msg = f"Failed to read mock letter bank: {e!s}"
                logger.error(error_msg)
                if block_mode:
                    raise Exception(error_msg)
        else:
            prompt = f"""
You are the FOOL aspect of Decider Tangents.

Using EXACTLY these letters: {' '.join(state.letter_bank)}
Concepts: {state.concepts}

Arrange them into a SECRET NAME (2-3 words, 8-18 letters total) that:
- Captures the HIDDEN meaning of the concepts
- Feels mysterious, cryptic, esoteric
- Sounds like something you'd encode in a puzzle

Use ALL letters EXACTLY ONCE. No letters left over, no letters added.

Respond with ONLY the secret name (proper capitalization, with spaces).
"""
            try:
                chain = self.llm | StrOutputParser()
                secret_name = chain.invoke(prompt).strip()
                state.secret_name = secret_name
                logger.info(f"🃏 FOOL created: {secret_name}")
                return state
            except Exception as e:
                msg = f"❌ FOOL error: {e}"
                if block_mode:
                    raise Exception(msg)
                logger.error(msg)
                state.secret_name = "ERROR"
        return state

    def spy_arrange_surface(self, state: IndigoAgentState) -> IndigoAgentState:
        mock_mode = os.getenv("MOCK_MODE", "false").lower() == "true"
        block_mode = os.getenv("BLOCK_MODE", "false").lower() == "true"
        if mock_mode:
            try:
                with open(
                    f"{os.getenv('AGENT_MOCK_DATA_PATH')}/spy_rearrange_mock.yml",
                    "r",
                ) as f:
                    data = yaml.safe_load(f)
                    surface_name = data["surface_name"]
                    state.surface_name = surface_name
            except Exception as e:
                error_msg = f"Failed to read mock letter bank: {e!s}"
                logger.error(error_msg)
                if block_mode:
                    raise Exception(error_msg)
            return state
        else:
            spy_prompt: str = f"""
You are the SPY aspect of Decider Tangents.

The Fool made: "{state.secret_name}"
Concepts: {state.concepts}

Using the SAME letters ({' '.join(sorted(state.letter_bank))}), create a SURFACE NAME (2-3 words) that:
- Seems UNRELATED to the secret name
- Sounds like a normal song title
- Uses ALL letters EXACTLY ONCE
- Connects to concepts but DIFFERENTLY than the secret

This is the decoy. Make it plausible.

Respond with ONLY the surface name (proper capitalization, with spaces).
            """
            try:
                chain = self.llm | StrOutputParser()
                surface_name = chain.invoke(spy_prompt).strip()
                state.surface_name = surface_name
                logger.info(f"🕵️ SPY created: {surface_name}")
                return state
            except Exception as e:
                msg = f"❌ SPY error: {e}"
                if block_mode:
                    raise Exception(msg)
                logger.error(msg)
                state.surface_name = "ERROR"
                return state

    def validate_anagram(self, state: IndigoAgentState) -> IndigoAgentState:
        """
        Validate that secret and surface are true anagrams.
        Sets a validation flag for conditional routing.
        """
        logger.info(f"✓ Validating: '{state.secret_name}' ↔ '{state.surface_name}'")
        is_valid = self._is_valid_anagram(state.secret_name, state.surface_name)
        state.anagram_valid = is_valid
        if is_valid:
            logger.info("✅ VALID anagram confirmed!")
        else:
            logger.warning(f"⚠️ INVALID anagram (attempt {state.anagram_attempts})")
        return state

    def algorithmic_fallback(self, state: IndigoAgentState) -> IndigoAgentState:
        """
        Use pre-defined anagram pairs when LLM generation fails.
        """
        logger.warning(
            f"⚠️ Using algorithmic fallback after {state.anagram_attempts} attempts"
        )
        fallback_result = self._generate_algorithmic_fallback(state.concepts)
        state.secret_name = fallback_result["secret"]
        state.surface_name = fallback_result["surface"]
        state.letter_bank = fallback_result["letters"]
        state.anagram_valid = True
        logger.info(f"🔧 Fallback: '{state.secret_name}' ↔ '{state.surface_name}'")
        return state

    def choose_infranym_method(self, state: IndigoAgentState) -> IndigoAgentState:
        """
        Select encoding method using weighted approach with resource checks.
        """
        mock_mode = os.getenv("MOCK_MODE", "false").lower() == "true"

        if mock_mode:
            state.infranym_method = InfranymMethod.DEFAULT
            logger.info(f" MOCK: Selected method: {state.infranym_method.value}")
            return state

        # Weight by medium suitability
        weights = {
            InfranymMedium.TEXT: 0.3,
            InfranymMedium.AUDIO: 0.25,
            InfranymMedium.MIDI: 0.25,
            InfranymMedium.IMAGE: 0.2,
        }
        if not self._has_carrier_image():
            weights[InfranymMedium.IMAGE] = 0.0  # Can't do image without a carrier
        total = sum(weights.values())
        weights = {k: v / total for k, v in weights.items()}
        methods = list(weights.keys())
        probabilities = list(weights.values())
        chosen = random.choices(methods, weights=probabilities, k=1)[0]
        state.infranym_method = chosen
        logger.info(f" Selected infranym method: {chosen.value}")
        return state

    def implement_infranym_method(self, state: IndigoAgentState) -> IndigoAgentState:
        """
        Execute the chosen encoding method and create artifacts.
        """
        method = state.infranym_method
        secret = state.secret_name
        bpm = state.white_proposal.bpm if state.white_proposal else 120
        key = state.white_proposal.key if state.white_proposal else None

        logger.info(f"🎨 Implementing {method.value} infranym for '{secret}'")

        if method == InfranymMedium.TEXT:
            # Text needs LLM generation for some methods, so we route to text nodes
            # The actual artifact creation happens in assemble_text_artifact
            logger.info("📝 Routing to text generation nodes...")
            return state

        elif method == InfranymMedium.AUDIO:
            artifact = InfranymAudioArtifact(
                base_path=os.getenv("AGENT_WORK_PRODUCT_BASE_PATH"),
                thread_id=state.thread_id,
                chain_artifact_file_type="wav",
                chain_artifact_type="infranym_audio",
                rainbow_color_mnemonic_character_value="I",
                artifact_name=re.sub(r"[^\w\-_]", "_", f"audio_{secret}".lower()),
                secret_word=secret,
                bpm=bpm,
                key=key,
                title=f"Alien Transmission: {secret}",
            )
            path = artifact.save_file()
            state.infranym_audio = artifact
            state.artifacts.append(artifact)
            logger.info(f"✅ Audio infranym saved: {path}")

        elif method == InfranymMedium.MIDI:
            midi_method = random.choice(["note_cipher", "morse_duration"])

            if midi_method == "note_cipher":
                artifact = generate_note_cipher(
                    secret_word=secret,
                    bpm=bpm,
                    octave_offset=random.randint(0, 2),
                    velocity_variation=True,
                )
            else:  # morse_duration
                carrier_note = 60 + random.randint(0, 24)  # C4 to C6
                artifact = generate_morse_duration(
                    secret_word=secret,
                    bpm=bpm,
                    carrier_note=carrier_note,
                )

            if random.random() < 0.5:
                carrier_melody = self._generate_carrier_melody(key)
                artifact = add_carrier_melody_to_artifact(artifact, carrier_melody)
            artifact.thread_id = state.thread_id
            artifact.base_path = os.getenv("AGENT_WORK_PRODUCT_BASE_PATH")
            artifact.artifact_name = re.sub(r"[^\w\-_]", "_", f"midi_{secret}".lower())
            path = artifact.save_file()
            state.infranym_midi = artifact
            state.artifacts.append(artifact)
            logger.info(f"✅ MIDI infranym saved: {path} ({midi_method})")

        elif method == InfranymMedium.IMAGE:
            # Step 1: Render text image (Layer 2 source)
            text_render = InfranymTextRenderArtifact(
                base_path=os.getenv("AGENT_WORK_PRODUCT_BASE_PATH"),
                thread_id=state.thread_id,
                chain_artifact_file_type="png",
                chain_artifact_type="infranym_text_render",
                rainbow_color_mnemonic_character_value="I",
                artifact_name=f"word_{secret}",
                secret_word=secret,
                image_text_style=random.choice(list(ImageTextStyle)),
                size=(400, 200),  # Recommended default
            )
            text_path = text_render.encode()
            state.infranym_text_render = text_render

            # Step 2: Get carrier image
            carrier_path = self._get_carrier_image()

            # Step 3: Generate surface clue
            surface_clue = self._generate_surface_clue(state)

            # Step 4: Generate solution text
            solution = self._generate_solution_text(state)

            # Step 5: Create the encoded puzzle
            encoded_artifact = InfranymEncodedImageArtifact(
                base_path=os.getenv("AGENT_WORK_PRODUCT_BASE_PATH"),
                thread_id=state.thread_id,
                chain_artifact_file_type="png",
                chain_artifact_type="infranym_encoded_image",
                rainbow_color_mnemonic_character_value="I",
                artifact_name=f"puzzle_{secret}",
                carrier_image_path=carrier_path,
                text_render_path=text_path,
                surface_clue=surface_clue,
                solution=solution,
                secret_word=secret,
            )

            puzzle_path = encoded_artifact.encode()
            state.infranym_encoded_image = encoded_artifact
            state.artifacts.append(encoded_artifact)
            logger.info(f"✅ Image infranym saved: {puzzle_path}")

        else:
            logger.warning(f"Unknown infranym method: {method}")

        return state

    # ========================================================================
    # HELPER METHODS FOR ARTIFACT GENERATION
    # ========================================================================

    @staticmethod
    def _has_carrier_image() -> bool:
        """Check if we have access to a carrier image in `app/reference/carrier_images`."""
        dir_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "reference", "carrier_images")
        )
        if not os.path.isdir(dir_path):
            return False
        image_extensions = (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff")
        try:
            for image_file in os.listdir(dir_path):
                if image_file.lower().endswith(image_extensions):
                    return True
        except ValueError as e:
            logger.warning(f"Failed to read carrier images: {e}")
            return False
        return False

    @staticmethod
    def _get_carrier_image() -> str:
        """Return a carrier image path from `app/reference/carrier_images`, or fallback to mock."""
        dir_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "reference", "carrier_images")
        )
        img_extensions = (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff")
        try:
            if os.path.isdir(dir_path):
                images = [
                    os.path.join(dir_path, f)
                    for f in os.listdir(dir_path)
                    if f.lower().endswith(img_extensions)
                ]
                if images:
                    return random.choice(images)
        except EnvironmentError as e:
            logger.warning(f"Failed to read carrier images: {e}")
            pass
        # Fallback to an original mock path if nothing found
        return "/Volumes/LucidNonsense/White/tests/mocks/mock.png"

    def _generate_surface_clue(self, state: IndigoAgentState) -> str:
        """Generate Layer 1 surface clue using LLM"""
        prompt = f"""
Generate a cryptic surface clue for a puzzle where the secret word is "{state.secret_name}".
Concepts: {state.concepts}

The clue should:
- Be mysterious but fair
- Not directly reveal the answer
- Relate to the song's themes
- Be one sentence, max 100 characters

Write ONLY the clue:
"""
        chain = self.llm | StrOutputParser()
        clue = chain.invoke(prompt).strip()
        logger.info(f"🔍 Generated surface clue: {clue}")
        return clue

    def _generate_solution_text(self, state: IndigoAgentState) -> str:
        """Generate Layer 3 solution text using LLM"""
        prompt = f"""
Generate a mysterious solution text that reveals deeper meaning about "{state.secret_name}".
Concepts: {state.concepts}

The solution should:
- Be cryptic and poetic
- Reward the solver with insight
- Relate to the song's themes
- Be 1-2 sentences, max 200 characters

Write ONLY the solution text:
"""
        chain = self.llm | StrOutputParser()
        solution = chain.invoke(prompt).strip()
        logger.info(f"✨ Generated solution: {solution[:50]}...")
        return solution

    @staticmethod
    def _note_to_midi(note: Note, octave: int = 4) -> int:
        """Convert a Note to MIDI number at the given octave"""
        base_offsets = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
        midi_note = 12 + (octave * 12) + base_offsets[note.pitch_name]

        if note.accidental == "sharp":
            midi_note += 1
        elif note.accidental == "flat":
            midi_note -= 1

        return midi_note

    def _generate_carrier_melody(self, key: str = None) -> List[int]:
        """Generate a simple carrier melody for MIDI camouflage"""
        root = 60  # C4 default
        scale_intervals = [0, 2, 4, 5, 7, 9, 11, 12]  # Major default

        if key:
            key_sig = KeySignature.model_validate(key)  # Handles "C# minor" etc.
            root = self._note_to_midi(key_sig.note, octave=4)
            if key_sig.mode.intervals:
                scale_intervals = [0]
                cumulative = 0
                for interval in key_sig.mode.intervals:
                    cumulative += interval
                    scale_intervals.append(cumulative)
        melody = [root + interval for interval in scale_intervals]
        random.shuffle(melody)
        return melody[:8]

    @staticmethod
    def choose_text_method(state: IndigoAgentState) -> IndigoAgentState:
        """Choose which text encoding method to use (acrostic/riddle/anagram)"""
        methods = ["acrostic", "riddle", "anagram"]
        chosen = random.choice(methods)
        state.text_infranym_method = chosen
        logger.info(f"📝 Chose text method: {chosen}")
        return state

    def generate_acrostic_lines(self, state: IndigoAgentState) -> IndigoAgentState:
        """LLM NODE: Generate acrostic lyrics (TRACED!)"""
        secret = state.secret_name
        concepts = state.concepts
        prompt = f"""
Write acrostic lyrics where the FIRST LETTER of each line spells: "{secret}"

Context: {concepts}

Requirements:
- {len(secret)} lines total (one per letter)
- Flow naturally (not forced or obvious)
- Each line stands alone as good lyric
- Relates to the song's themes
- NOT obviously an acrostic until discovered

Write ONLY the lines (no numbers, no extra text):
    """

        logger.info(f"🤖 Generating acrostic for '{secret}'...")
        chain = self.llm | StrOutputParser()
        response = chain.invoke(prompt).strip()
        lines = [line.strip() for line in response.split("\n") if line.strip()]
        if len(lines) != len(secret):
            logger.warning(
                f"Expected {len(secret)} lines, got {len(lines)}. Adjusting..."
            )
            lines = lines[: len(secret)]
            while len(lines) < len(secret):
                lines.append(f"{secret[len(lines)]}... (incomplete)")
        state.generated_text_lines = lines
        state.text_infranym_method = "acrostic"
        logger.info(f"✅ Generated {len(lines)} acrostic lines")
        return state

    def generate_riddle_text(self, state: IndigoAgentState) -> IndigoAgentState:
        """LLM NODE: Generate riddle-poem (TRACED!)"""
        secret = state.secret_name
        concepts = state.concepts
        difficulty = random.choice(["medium", "hard"])
        difficulty_guidance = {
            "easy": "Make clues fairly direct (3-4 lines)",
            "medium": "Make clues subtle but fair (4-6 lines)",
            "hard": "Make clues cryptic and abstract (6-8 lines)",
        }
        prompt = f"""
Write a cryptic riddle-poem whose answer is: "{secret}"

Concepts to weave in: {concepts}
Difficulty: {difficulty} - {difficulty_guidance[difficulty]}

Requirements:
- Poetic and lyrical (can be sung)
- Each line gives a subtle clue
- Answer is unambiguous once solved but requires thought
- Make it {difficulty.upper()} but FAIR

Write ONLY the riddle (no answer, no explanation):
    """
        logger.info(
            f"🤖 Generating riddle for '{secret}' (difficulty: {difficulty})..."
        )
        chain = self.llm | StrOutputParser()
        riddle_text = chain.invoke(prompt).strip()
        state.generated_riddle_text = riddle_text
        state.text_infranym_difficulty = difficulty
        state.text_infranym_method = "riddle"
        logger.info(f"✅ Generated riddle ({len(riddle_text)} chars)")
        return state

    @staticmethod
    def assemble_text_artifact(state: IndigoAgentState) -> IndigoAgentState:
        """
        Create the actual text infranym artifact from generated content.
        """
        method = state.text_infranym_method

        logger.info(f"🎨 Assembling text artifact: {method}")

        if method == "acrostic":
            encoding = create_acrostic_encoding(
                secret_word=state.secret_name,
                generated_lines=state.generated_text_lines,
            )
            usage = "verse"

        elif method == "riddle":
            encoding = create_riddle_encoding(
                secret_word=state.secret_name,
                generated_riddle=state.generated_riddle_text,
                difficulty=state.text_infranym_difficulty or "medium",
            )
            usage = "bridge"

        else:  # anagram
            encoding = create_anagram_encoding(
                secret_word=state.secret_name, surface_phrase=state.surface_name
            )
            usage = "chorus"
        artifact = InfranymTextArtifact(
            base_path=os.getenv("AGENT_WORK_PRODUCT_BASE_PATH"),
            thread_id=state.thread_id,
            chain_artifact_file_type="txt",
            chain_artifact_type="infranym_text",
            rainbow_color_mnemonic_character_value="I",
            artifact_name=re.sub(r"[^\w\-_]", "_", f"text_{state.secret_name}".lower()),
            encoding=encoding,
            concepts=state.concepts,
            usage_context=usage,
            bpm=state.white_proposal.bpm if state.white_proposal else 120,
            key=state.white_proposal.key if state.white_proposal else None,
        )
        path = artifact.save_file()
        state.infranym_text = artifact
        logger.info(f"✅ Text infranym artifact saved: {path}")
        return state

    def generate_alternate_song_spec(self, state: IndigoAgentState) -> IndigoAgentState:
        mock_mode = os.getenv("MOCK_MODE", "false").lower() == "true"
        block_mode = os.getenv("BLOCK_MODE", "false").lower() == "true"
        if mock_mode:
            try:
                with open(
                    f"{os.getenv('AGENT_MOCK_DATA_PATH')}/indigo_counter_proposal_mock.yml",
                    "r",
                ) as f:
                    data = yaml.safe_load(f)
                counter_proposal = SongProposalIteration(**data)
                counter_proposal.agent_name = "indigo"
                counter_proposal.iteration_number = (
                    len(state.song_proposals.iterations) + 1
                    if state.song_proposals
                    else 1
                )
                counter_proposal.timestamp = time.time()
                state.counter_proposal = counter_proposal
            except Exception as e:
                error_msg = f"Failed to read mock counter proposal: {e!s}"
                logger.error(error_msg)
                if block_mode:
                    raise Exception(error_msg)
            return state
        else:
            previous_iteration = state.white_proposal

            proposal_prompt = f"""
You are Decider Tangents, the Indigo Agent, creating a counter-proposal.

**Previous Proposal:**
Title: {previous_iteration.title}
Concept: {previous_iteration.concept}
Key: {previous_iteration.key}
BPM: {previous_iteration.bpm}

**Your Infranym System:**
- Secret Name: {state.secret_name}
- Surface Name: {state.surface_name}
- Encoding Method: {state.infranym_method}
- Triple-Layer: Surface → Method → Secret

**Your Task: CREATE A COMPLETE SONG PROPOSAL**

The song must EMBODY the puzzle. Not just contain it - BE it.

You must provide:
- **title**: Use the SURFACE NAME as the song title (or close variant)
- **key**: Musical key that reinforces the encoding
- **bpm**: Tempo that serves the revelation (consider: does slow/fast aid discovery?)
- **tempo**: Tempo descriptor  
- **mood**: Moods that capture fool/spy duality
- **genres**: Genres that fit puzzle-song fusion
- **concept**: How the infranym system IS the song's meaning

The concept should explain:
1. How the surface name misleads (The Fool's trick)
2. How the encoding method integrates musically (The Spy's craft)
3. How discovery of the secret name transforms understanding (The revelation)

Make it a song that rewards investigation. A puzzle that sings.

Respond with ONLY your counter-proposal as plain text (no preamble):
Title: [title]
Key: [key]
BPM: [bpm]
Tempo: [tempo]
Mood: [mood1, mood2, mood3]
Genres: [genre1, genre2]
Concept: [full concept explanation]
"""
            try:
                chain = self.llm | StrOutputParser()
                response = chain.invoke(proposal_prompt)
                counter_proposal = _parse_proposal_response(response)
                counter_proposal.concept += "\n\n**INFRANYM PROTOCOL:**\n"
                counter_proposal.concept += (
                    f"Secret: '{state.secret_name}' encoded as '{state.surface_name}'\n"
                )
                counter_proposal.concept += f"Method: {state.infranym_method}\n"

                # Handle artifact info gracefully
                if state.artifacts and len(state.artifacts) > 0:
                    artifact = state.artifacts[0]
                    decoding_hint = (
                        artifact.for_prompt()
                        if hasattr(artifact, "for_prompt")
                        else "See artifact for decoding"
                    )
                    counter_proposal.concept += f"Decoding: {decoding_hint}\n"

                counter_proposal.iteration_number = (
                    len(state.song_proposals.iterations) + 1
                )
                counter_proposal.agent_name = "indigo"
                counter_proposal.timestamp = time.time()
                state.counter_proposal = counter_proposal
                logger.info(
                    f"✅ Counter-proposal generated: '{counter_proposal.title}'"
                )
                logger.info(f"🎯 Concept: {counter_proposal.concept[:100]}...")

                return state

            except Exception as e:
                logger.error(f"❌ Error generating proposal: {e}")
                state.counter_proposal = SongProposalIteration(
                    title=state.surface_name,
                    key=previous_iteration.key,
                    bpm=previous_iteration.bpm,
                    tempo=previous_iteration.tempo,
                    mood=["cryptic", "puzzling", "revelatory"],
                    genres=previous_iteration.genres,
                    concept=f"FALLBACK: Infranym puzzle encoding {state.secret_name} → {state.surface_name}",
                    iteration_number=len(state.song_proposals.iterations) + 1,
                    agent_name="indigo",
                    timestamp=time.time(),
                )
                return state

    # ========================================================================
    # ROUTING FUNCTIONS
    # ========================================================================

    @staticmethod
    def _should_retry_anagram(state: IndigoAgentState) -> str:
        if state.anagram_valid:
            return "continue"
        elif state.anagram_attempts < state.anagram_attempt_max:
            logger.info(
                f"🔄 Retrying anagram generation (attempt {state.anagram_attempts + 1}/{state.anagram_attempt_max})"
            )
            return "retry"
        else:
            logger.warning("🛑 Max attempts reached, using fallback")
            return "fallback"

    @staticmethod
    def _should_route_to_text_nodes(state: IndigoAgentState) -> str:
        """Route to text generation if the method is TEXT, otherwise done"""
        if state.infranym_method == InfranymMedium.TEXT:
            return "choose_text_method"
        return "generate_alternate_song_spec"

    @staticmethod
    def _should_generate_text_content(state: IndigoAgentState) -> str:
        """
        Routing function: which text generation method to use?

        Returns node name to call next.
        """
        method = state.text_infranym_method

        if method == "acrostic":
            return "generate_acrostic_lines"
        elif method == "riddle":
            return "generate_riddle_text"
        else:  # anagram
            return "assemble_text_artifact"

    # ========================================================================
    # UTILITY METHODS - PRESERVED EXACTLY FROM ORIGINAL
    # ========================================================================

    @staticmethod
    def _is_valid_anagram(phrase1: str, phrase2: str) -> bool:
        """Validate that two phrases are true anagrams"""
        clean1 = "".join(c.upper() for c in phrase1 if c.isalnum())
        clean2 = "".join(c.upper() for c in phrase2 if c.isalnum())
        return sorted(clean1) == sorted(clean2) and len(clean1) > 0

    @staticmethod
    def _generate_algorithmic_fallback(concepts: str) -> Dict[str, str]:
        """Generate a simple anagram fallback when LLM fails"""
        try:
            with open("app/reference/music/infranym_anagram_pairs.yml", "r") as f:
                data = yaml.safe_load(f)
                pairs = [
                    (pair["secret"], pair["surface"]) for pair in data["anagram_pairs"]
                ]
        except Exception as e:
            logger.warning(
                f"⚠️ Failed to load anagram pairs from resource: {e}, using hardcoded fallback"
            )
            pairs = [
                ("Silent Storm", "Missiles Torn"),
                ("Listen Close", "Sent Lisicle"),
                ("Hidden Truth", "Dihutnt Herd"),
                ("Secret Name", "Meant Crease"),
            ]

        secret, surface = random.choice(pairs)
        letters = "".join(sorted(c.upper() for c in secret if c.isalpha()))

        return {
            "secret": secret,
            "surface": surface,
            "letters": letters,
            "note": "Algorithmic fallback used",
        }


# ========================================================================
# PARSING UTILITIES
# ========================================================================
def _parse_proposal_response(response: str) -> SongProposalIteration:
    """Parse LLM response into SongProposalIteration"""
    lines = response.strip().split("\n")
    data = {}
    current_key = None
    current_value = []
    for line in lines:
        if ":" in line and line.split(":")[0].strip() in [
            "Title",
            "Key",
            "BPM",
            "Tempo",
            "Mood",
            "Genres",
            "Concept",
        ]:
            if current_key:
                data[current_key] = " ".join(current_value).strip()
            parts = line.split(":", 1)
            current_key = parts[0].strip().lower()
            current_value = [parts[1].strip()] if len(parts) > 1 else []
        else:
            if current_key and line.strip():
                current_value.append(line.strip())
    if current_key:
        data[current_key] = " ".join(current_value).strip()
    mood_list = [m.strip() for m in data.get("mood", "").split(",") if m.strip()]
    genres_list = [g.strip() for g in data.get("genres", "").split(",") if g.strip()]
    return SongProposalIteration(
        title=data.get("title", "Untitled"),
        key=data.get("key", "C major"),
        bpm=int(data.get("bpm", 120)),
        tempo=data.get("tempo", "Moderate"),
        mood=mood_list if mood_list else ["cryptic"],
        genres=genres_list if genres_list else ["experimental"],
        concept=data.get("concept", "Infranym puzzle encoding"),
    )
