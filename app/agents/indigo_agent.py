import logging
import os
import random
import re
import time
import yaml

from abc import ABC
from typing import Dict, Any, List
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import StrOutputParser
from langgraph.constants import START, END
from langgraph.graph.state import StateGraph

from app.agents.states.indigo_agent_state import IndigoAgentState
from app.agents.states.white_agent_state import MainAgentState
from app.structures.agents.agent_settings import AgentSettings
from app.structures.agents.base_rainbow_agent import BaseRainbowAgent
from app.structures.enums.infranym_medium import InfranymMedium
from app.structures.manifests.song_proposal import SongProposalIteration

load_dotenv()


# ToDo: Something ignore or not working with mock mode
class IndigoAgent(BaseRainbowAgent, ABC):
    """Decider Tangents - Hides information"""

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
            infranym_image=None,
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
        6. choose_infranym_method - Hybrid method selection
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
        work_flow.add_edge("implement_infranym_method", "generate_alternate_song_spec")
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
        logging.info(f"🕵️ SPY choosing letter bank (attempt {state.anagram_attempts})")
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
                logging.error(error_msg)
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
                logging.error(msg)
                state.letter_bank = "AEILNORSTDM"
        return state

    def fool_arrange_secret(self, state: IndigoAgentState) -> IndigoAgentState:
        """
        FOOL arranges letters into SECRET name.
        Now a traceable graph node!
        """
        mock_mode = os.getenv("MOCK_MODE", "false").lower() == "true"
        block_mode = os.getenv("BLOCK_MODE", "false").lower() == "true"
        logging.info(f"🃏 FOOL arranging secret from: {state.letter_bank}")
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
                logging.error(error_msg)
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
                logging.info(f"🃏 FOOL created: {secret_name}")
                return state
            except Exception as e:
                msg = f"❌ FOOL error: {e}"
                if block_mode:
                    raise Exception(msg)
                logging.error(msg)
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
                logging.error(error_msg)
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
                logging.info(f"🕵️ SPY created: {surface_name}")
                return state
            except Exception as e:
                msg = f"❌ SPY error: {e}"
                if block_mode:
                    raise Exception(msg)
                logging.error(msg)
                state.surface_name = "ERROR"
                return state

    def validate_anagram(self, state: IndigoAgentState) -> IndigoAgentState:
        """
        Validate that secret and surface are true anagrams.
        Sets a validation flag for conditional routing.
        """
        logging.info(f"✓ Validating: '{state.secret_name}' ↔ '{state.surface_name}'")
        is_valid = self._is_valid_anagram(state.secret_name, state.surface_name)
        state.anagram_valid = is_valid
        if is_valid:
            logging.info("✅ VALID anagram confirmed!")
        else:
            logging.warning(f"⚠️ INVALID anagram (attempt {state.anagram_attempts})")
        return state

    def algorithmic_fallback(self, state: IndigoAgentState) -> IndigoAgentState:
        """
        Use pre-defined anagram pairs when LLM generation fails.
        """
        logging.warning(
            f"⚠️ Using algorithmic fallback after {state.anagram_attempts} attempts"
        )
        fallback_result = self._generate_algorithmic_fallback(state.concepts)
        state.secret_name = fallback_result["secret"]
        state.surface_name = fallback_result["surface"]
        state.letter_bank = fallback_result["letters"]
        state.anagram_valid = True
        logging.info(f"🔧 Fallback: '{state.secret_name}' ↔ '{state.surface_name}'")
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
                # Set tracking metadata for workflow routing
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
                logging.error(error_msg)
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
                counter_proposal = self._parse_proposal_response(response)
                # Add infranym metadata to concept
                counter_proposal.concept += "\n\n**INFRANYM PROTOCOL:**\n"
                counter_proposal.concept += (
                    f"Secret: '{state.secret_name}' encoded as '{state.surface_name}'\n"
                )
                counter_proposal.concept += f"Method: {state.infranym_method}\n"
                counter_proposal.concept += (
                    f"Decoding: {state.artifacts[0]['triple_layer']['decoding_hint']}"
                )
                # Add iteration metadata
                counter_proposal.iteration_number = (
                    len(state.song_proposals.iterations) + 1
                )
                counter_proposal.agent_name = "indigo"
                counter_proposal.timestamp = time.time()
                state.counter_proposal = counter_proposal
                logging.info(
                    f"✅ Counter-proposal generated: '{counter_proposal.title}'"
                )
                logging.info(f"🎯 Concept: {counter_proposal.concept[:100]}...")

                return state

            except Exception as e:
                logging.error(f"❌ Error generating proposal: {e}")
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

    @staticmethod
    def _should_retry_anagram(state: IndigoAgentState) -> str:
        if state.anagram_valid:
            return "continue"
        elif state.anagram_attempts < state.anagram_attempt_max:
            logging.info(
                f"🔄 Retrying anagram generation (attempt {state.anagram_attempts + 1}/{state.anagram_attempt_max})"
            )
            return "retry"
        else:
            logging.warning("🛑 Max attempts reached, using fallback")
            return "fallback"

    def choose_infranym_method(self, state: IndigoAgentState) -> IndigoAgentState:
        """
        Choose encoding method via hybrid approach.
        SPY analyzes constraints; FOOL chooses the game.
        """
        mock_mode = os.getenv("MOCK_MODE", "false").lower() == "true"
        block_mode = os.getenv("BLOCK_MODE", "false").lower() == "true"
        constraints = self._analyze_encoding_constraints(
            state.secret_name, state.surface_name
        )
        available_methods = []
        if constraints["midi_viable"]:
            available_methods.append(InfranymMedium.MIDI)
        if constraints["audio_viable"]:
            available_methods.append(InfranymMedium.AUDIO)
        if constraints["text_viable"]:
            available_methods.append(InfranymMedium.TEXT)
        if constraints["image_viable"]:
            available_methods.append(InfranymMedium.IMAGE)
        logging.info(f"📊 Encoding constraints: {constraints}")
        if mock_mode:
            try:
                with open(
                    f"{os.getenv('AGENT_MOCK_DATA_PATH')}/choose_infranym_method_mock.yml",
                    "r",
                ) as f:
                    data = yaml.safe_load(f)
                    chosen_method = data["chosen_method"]
                    # Convert string to enum if needed
                    if isinstance(chosen_method, str):
                        chosen_method = InfranymMedium(chosen_method.lower())
                    state.infranym_medium = chosen_method
                    state.method_constraints = constraints
                    return state
            except Exception as e:
                error_msg = f"Failed to read mock infranym method: {e!s}"
                logging.error(error_msg)
                if block_mode:
                    raise Exception(error_msg)
            return state
        else:
            choice_prompt = f"""
You are Decider Tangents, the Indigo Agent, choosing your encoding method.

**The Secret:** {state.secret_name}
**The Surface:** {state.surface_name}
**Concepts:** {state.concepts}

**Available Methods:**
{self._format_method_descriptions(available_methods, constraints)}

**Your Decision:**
Which method best serves the TRIPLE-LAYER revelation?
- Layer 1: Surface appears as "{state.surface_name}"
- Layer 2: Encoding technique itself becomes discovery
- Layer 3: Secret "{state.secret_name}" revealed

Consider:
- Which method makes the secret HARDEST but FAIREST to find?
- Which creates the most satisfying "aha!" moment?
- Which matches the song's conceptual essence?

Respond with ONLY the method name: MIDI, AUDIO, TEXT, or IMAGE
"""

        try:
            chain = self.llm | StrOutputParser()
            chosen_method_str = chain.invoke(choice_prompt).strip().upper()
            try:
                chosen_method = InfranymMedium(chosen_method_str.lower())
            except ValueError:
                chosen_method = None
            if chosen_method not in available_methods:
                logging.warning(
                    f"⚠️ Invalid choice '{chosen_method_str}', defaulting to TEXT"
                )
                chosen_method = InfranymMedium.TEXT  # TEXT is always viable
            state.infranym_medium = chosen_method
            state.method_constraints = constraints
            logging.info(f"🎯 Chosen method: {chosen_method.value}")
            return state

        except Exception as e:
            logging.error(f"❌ Error choosing method: {e}")
            state.infranym_medium = InfranymMedium.TEXT  # Safe fallback
            state.method_constraints = constraints
            return state

    def implement_infranym_method(self, state: IndigoAgentState) -> IndigoAgentState:
        """
        Implement the chosen infranym encoding.
        Creates triple-layer revelation system.
        """
        mock_mode = os.getenv("MOCK_MODE", "false").lower() == "true"
        logging.info(
            f"🩵 IndigoAgent: Implementing {state.infranym_medium.value} infranym"
        )

        try:
            # In mock mode, use simple fallback instead of LLM calls
            if mock_mode:
                result = {
                    "encoding_method": f"mock_{state.infranym_medium.value}",
                    "mock_data": f"Mock implementation for {state.infranym_medium.value}",
                    "decoding_hint": "This is mock data for testing",
                }
            # Dispatch to specific implementation
            elif state.infranym_medium == InfranymMedium.MIDI:
                result = self._implement_midi_infranym(state)
                state.infranym_midi = result
            elif state.infranym_medium == InfranymMedium.AUDIO:
                result = self._implement_audio_infranym(state)
                state.infranym_audio = result
            elif state.infranym_medium == InfranymMedium.TEXT:
                result = self._implement_text_infranym(state)
                state.infranym_text = result
            elif state.infranym_medium == InfranymMedium.IMAGE:
                result = self._implement_image_infranym(state)
                state.infranym_image = result
            else:
                raise ValueError(f"Unknown medium: {state.infranym_medium}")

            # Add triple-layer revelation metadata
            triple_layer = {
                "layer_1_surface": f"Song appears to be about: {state.surface_name}",
                "layer_2_method": f"Discover encoding method: {result['encoding_method']}",
                "layer_3_secret": f"Reveals hidden name: {state.secret_name}",
                "decoding_hint": result.get("decoding_hint", "Look deeper..."),
            }

            # Store in artifacts
            artifact = {
                "type": state.infranym_medium.value,
                "encoding": result,
                "triple_layer": triple_layer,
                "fool_spy_protocol": "The Fool hides; the Spy ensures fairness",
            }

            state.artifacts.append(artifact)

            logging.info(f"✅ {state.infranym_medium.value} infranym complete")
            logging.info(f"🎯 Triple layer: {triple_layer['layer_1_surface']}")

            return state

        except Exception as e:
            logging.error(f"❌ Error implementing infranym: {e}")
            # Create minimal fallback artifact
            state.artifacts.append(
                {
                    "type": "ERROR",
                    "error": str(e),
                    "fallback": "Text-based simple anagram",
                }
            )
            return state

    @staticmethod
    def _is_valid_anagram(phrase1: str, phrase2: str) -> bool:
        """Validate that two phrases are true anagrams"""
        clean1 = "".join(c.upper() for c in phrase1 if c.isalnum())
        clean2 = "".join(c.upper() for c in phrase2 if c.isalnum())
        return sorted(clean1) == sorted(clean2) and len(clean1) > 0

    @staticmethod
    def _generate_algorithmic_fallback(concepts: str) -> Dict[str, str]:
        """Generate simple anagram fallback when LLM fails"""
        # Load anagram pairs from resource file
        try:
            with open("app/reference/music/infranym_anagram_pairs.yml", "r") as f:
                data = yaml.safe_load(f)
                pairs = [
                    (pair["secret"], pair["surface"]) for pair in data["anagram_pairs"]
                ]
        except Exception as e:
            logging.warning(
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
    # HELPER METHODS: Method Selection & Constraints
    # ========================================================================

    @staticmethod
    def _analyze_encoding_constraints(secret: str, surface: str) -> Dict[str, Any]:
        """Algorithmic analysis of viable encoding methods"""
        letter_count = len(secret.replace(" ", ""))
        word_count = len(secret.split())
        has_numbers = any(c.isdigit() for c in secret)

        # MIDI: Limited by note range and practical melody length
        midi_viable = letter_count <= 24 and not has_numbers

        # AUDIO: Needs clear phonetic structure
        phoneme_count = len(re.findall(r"[aeiou]", secret.lower()))
        audio_viable = phoneme_count >= 4 and letter_count <= 30

        # TEXT: Always possible
        text_viable = True

        # IMAGE: Needs enough data for steganography
        image_viable = letter_count >= 8
        return {
            "midi_viable": midi_viable,
            "audio_viable": audio_viable,
            "text_viable": text_viable,
            "image_viable": image_viable,
            "letter_count": letter_count,
            "word_count": word_count,
            "phoneme_count": phoneme_count,
        }

    def _format_method_descriptions(
        self, available: List[InfranymMedium], constraints: Dict
    ) -> str:
        """Format available methods with descriptions"""
        descriptions = {
            InfranymMedium.MIDI: f"Musical encoding (note-to-letter cipher, morse code, tap code) - {constraints['letter_count']} letters",
            InfranymMedium.AUDIO: f"Sonic encoding (backmasking, spectrogram steganography, phoneme anagrams) - {constraints['phoneme_count']} phonemes",
            InfranymMedium.TEXT: f"Textual encoding (riddle-poems, acrostics, anagram puzzles) - {constraints['word_count']} words",
            InfranymMedium.IMAGE: f"Visual encoding (LSB steganography, anti-sigils, QR fragments) - {constraints['letter_count']} bytes",
        }

        return "\n".join(
            f"- {method.value.upper()}: {descriptions[method]}" for method in available
        )

    # ========================================================================
    # INFRANYM IMPLEMENTATIONS: The Four Encoding Methods
    # ========================================================================

    @staticmethod
    def _implement_midi_infranym(state: IndigoAgentState) -> Dict[str, Any]:
        """Encode secret name in MIDI"""
        secret = state.secret_name.replace(" ", "").upper()
        methods = ["note_cipher", "morse_duration", "tap_code_pitch"]
        method = random.choice(methods)
        if method == "note_cipher":
            # A-G direct mapping, then H=A, I=B, etc (modulo 7)
            notes = []
            for char in secret:
                if char.isalpha():
                    note_idx = (ord(char) - ord("A")) % 7
                    octave = 4 + ((ord(char) - ord("A")) // 7)
                    note_name = ["C", "D", "E", "F", "G", "A", "B"][note_idx]
                    notes.append(f"{note_name}{octave}")
            return {
                "encoding_method": "note_cipher",
                "note_sequence": notes,
                "midi_instructions": f"Play these notes in sequence: {' '.join(notes)}",
                "decoding_hint": "The melody spells a name (A-G cipher, ascending octaves)",
                "timestamp_suggestion": "Place at 1:11 (for discovery)",
            }

        elif method == "morse_duration":
            # Morse code via note durations
            # Load morse code from resource file
            try:
                with open("app/reference/encoding/morse_code.yml", "r") as f:
                    data = yaml.safe_load(f)
                    morse_map = data["morse_alphabet"]
            except Exception as e:
                logging.warning(
                    f"⚠️ Failed to load morse code from resource: {e}, using hardcoded fallback"
                )
                morse_map = {
                    "A": ".-",
                    "B": "-...",
                    "C": "-.-.",
                    "D": "-..",
                    "E": ".",
                    "F": "..-.",
                    "G": "--.",
                    "H": "....",
                    "I": "..",
                    "J": ".---",
                    "K": "-.-",
                    "L": ".-..",
                    "M": "--",
                    "N": "-.",
                    "O": "---",
                    "P": ".--.",
                    "Q": "--.-",
                    "R": ".-.",
                    "S": "...",
                    "T": "-",
                    "U": "..-",
                    "V": "...-",
                    "W": ".--",
                    "X": "-..-",
                    "Y": "-.--",
                    "Z": "--..",
                }

            morse_sequence = []
            for char in secret:
                if char in morse_map:
                    morse_sequence.append(morse_map[char])
            return {
                "encoding_method": "morse_duration",
                "morse_sequence": " / ".join(morse_sequence),
                "midi_instructions": "Dots = quarter notes, Dashes = half notes, on Middle C",
                "decoding_hint": "Rhythm is morse code (listen to note durations)",
                "timestamp_suggestion": "Place in intro or outro",
            }

        else:  # tap_code_pitch
            # Polybius square: row = octave, column = scale degree
            return {
                "encoding_method": "tap_code_pitch",
                "midi_instructions": f"Use tap code cipher (Polybius square) for: {secret}",
                "decoding_hint": "Octave = row, scale degree = column in 5x5 grid",
                "note": "Implement via: each letter → (row, col) → note pitch",
                "timestamp_suggestion": "Use throughout track as melodic theme",
            }

    def _implement_audio_infranym(self, state: IndigoAgentState) -> Dict[str, Any]:
        """Encode secret name in audio"""
        secret = state.secret_name
        surface = state.surface_name
        methods = ["backmask_whisper", "stenograph_spectrogram", "anagram_tts"]
        method = random.choice(methods)

        if method == "backmask_whisper":
            timestamp = f"{len(secret) // 10}:{(len(secret) % 10) * 6:02d}"
            return {
                "encoding_method": "backmask_whisper",
                "instructions": f"""
    1. Generate TTS of "{secret}" in whispered/breathy voice
    2. Reverse the audio (play backwards)
    3. Place at timestamp {timestamp}
    4. Mix at -18dB under main vocal track
    5. When listener reverses audio, secret is revealed
                    """,
                "decoding_hint": f"What do you hear at {timestamp} when played backwards?",
                "timestamp": timestamp,
            }

        elif method == "stenograph_spectrogram":
            return {
                "encoding_method": "stenograph_spectrogram",
                "instructions": f"""
    1. Generate spectrogram of chosen section
    2. Encode "{secret}" as text in 8-12kHz frequency band
    3. Use LSB steganography in spectrogram image  
    4. Reconstruct audio from modified spectrogram
    5. Secret visible only in spectral analysis
                    """,
                "decoding_hint": "View spectrogram between 8-12kHz. Look for text.",
                "tools_needed": "Audacity or similar spectral editor",
            }

        else:  # anagram_tts
            return {
                "encoding_method": "anagram_tts",
                "surface_phrase": surface,
                "secret_phrase": secret,
                "instructions": f"""
    1. Generate TTS saying: "{surface}"
    2. Add subtle stutter/glitch effect
    3. Ensure phonemes can be cut/rearranged to say: "{secret}"
    4. Place in verse or bridge
    5. Listeners can splice audio to discover anagram
                    """,
                "decoding_hint": "Cut and rearrange the syllables of this phrase",
            }

    def _implement_text_infranym(self, state: IndigoAgentState) -> Dict[str, Any]:
        """Encode secret name in lyrics/text"""
        secret = state.secret_name
        surface = state.surface_name
        concepts = state.concepts
        methods = ["riddle_poem", "acrostic_lyrics", "anagram_puzzle"]
        method = random.choice(methods)

        if method == "riddle_poem":
            riddle_prompt = f"""
Write a cryptic riddle-poem whose answer is: "{secret}"

Concepts to weave in: {concepts}

Requirements:
- 4-8 lines that can be sung
- Each line gives a subtle clue
- Poetic and lyrical (not obvious riddle format)
- Answer is unambiguous once solved but requires thought
- Can reference: {surface} as misdirection

Make it HARD but FAIR. Write the riddle NOW:
"""

            chain = self.llm | StrOutputParser()
            riddle = chain.invoke(riddle_prompt)
            return {
                "encoding_method": "riddle_poem",
                "riddle_text": riddle,
                "answer": secret,
                "decoding_hint": "The lyrics are a riddle. What's the answer?",
                "usage": "Can be sung as verse/chorus or spoken as interlude",
            }

        elif method == "acrostic_lyrics":
            acrostic_prompt = f"""
Write lyrics where the FIRST LETTER of each line spells: "{secret}"

Context: {concepts}

Requirements:
- Flow naturally (not forced or obvious)
- Each line stands alone as good lyric
- Relates to the song's themes
- NOT obviously an acrostic until discovered
- 2-4 lines per verse/chorus as needed

Write the lyrics NOW:
"""

            chain = self.llm | StrOutputParser()
            lyrics = chain.invoke(acrostic_prompt)
            return {
                "encoding_method": "acrostic_lyrics",
                "lyrics": lyrics,
                "secret": secret,
                "decoding_hint": "Read down the first letter of each line",
                "usage": "Use as main lyrics for verse/chorus",
            }

        else:  # anagram_puzzle
            return {
                "encoding_method": "anagram_puzzle",
                "surface_phrase": surface,
                "secret_phrase": secret,
                "instructions": f"""
    1. Include the phrase "{surface}" prominently in lyrics
    2. Emphasize it through: repetition, isolation, vocal effect
    3. Provide subtle visual/sonic hint that letters rearrange
    4. Perhaps: unusual spacing, repeated letters, echoed syllables
    5. Liner notes could have "{surface}" in special font
                    """,
                "decoding_hint": f"Rearrange the letters of '{surface}'",
                "usage": "Works as hook, refrain, or title phrase",
            }

    def _implement_image_infranym(self, state: IndigoAgentState) -> Dict[str, Any]:
        """Encode secret name in album art/visuals"""
        secret = state.secret_name
        concepts = state.concepts
        methods = ["lsb_steganography", "anti_sigil", "qr_fragments"]
        method = random.choice(methods)

        if method == "lsb_steganography":
            return {
                "encoding_method": "lsb_steganography",
                "secret_text": secret,
                "instructions": """
    1. Generate album art image (high-res PNG)
    2. Encode secret text in LSB of RGB channels
    3. Use steghide, stegpy, or PIL library
    4. Visual appearance unchanged
    5. Extractable with standard steg tools
                    """,
                "decoding_hint": "Extract hidden data from album art using steganography tool",
                "tools_needed": "steghide, stegpy, or similar",
            }

        elif method == "anti_sigil":
            anti_sigil_prompt = f"""Design an ANTI-SIGIL (opposite of Black Agent's chaos sigils).

    Secret to conceal: "{secret}"
    Concepts: {concepts}

    Where sigils CHANNEL intention through simplification,
    Anti-sigils CONCEAL intention through complexity.

    Create a design that:
    - Hides "{secret}" in ornate, overlapping geometric layers
    - Requires "defocusing" to perceive (magic eye effect)
    - Uses sacred geometry to obscure rather than reveal
    - Contains geometric hint: secret's letter count = number of primary shapes

    Describe the anti-sigil's structure and decoding method:"""

            chain = self.llm | StrOutputParser()
            design = chain.invoke(anti_sigil_prompt)

            return {
                "encoding_method": "anti_sigil",
                "design_description": design,
                "secret": secret,
                "decoding_hint": "Defocus your eyes. Count the primary geometric shapes.",
                "artistic_note": "Inverse of Black Agent sigil work - conceals rather than channels",
            }

        else:  # qr_fragments
            return {
                "encoding_method": "qr_fragments",
                "secret_text": secret,
                "instructions": f"""
    1. Generate QR code encoding: "{secret}"
    2. Fragment QR into 4-9 pieces
    3. Distribute fragments across album art:
       - Hidden in textures
       - Woven into patterns
       - Camouflaged in shadows/highlights
    4. Fragments must be found and reassembled
    5. Reassembled QR scans to reveal secret
                    """,
                "decoding_hint": "Find the QR fragments hidden in the artwork. Reassemble them.",
                "difficulty": "High - requires careful observation",
            }


# ========================================================================
# HELPER METHODS: Parsing & Utilities
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
