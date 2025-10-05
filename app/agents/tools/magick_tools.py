import hashlib
import random
import re
from datetime import datetime
from typing import Dict, List, Tuple, Any

from app.agents.enums.gnosis_method import GnosisMethod
from app.agents.enums.sigil_type import SigilType
from app.agents.models.sigil_artifact import SigilRecord


class SigilTools:
    """Austin Osman Spare-inspired sigil creation and management tools"""

    def __init__(self):
        self.sigil_vault: Dict[str, SigilRecord] = {}
        self.alphabet_of_desire: Dict[str, str] = {}
        self.paranoia_level = random.randint(7, 10)  # Black Agent's baseline paranoia

    @staticmethod
    def create_statement_of_intent(raw_desire: str, agent_personality: bool = True) -> str:
        """
        Convert raw desire into proper statement of intent

        Args:
            raw_desire: The user's raw wish/desire
            agent_personality: Whether to add Black Agent paranoid flavor
        """
        # Clean and structure the intent
        cleaned = raw_desire.strip().lower()

        # Remove negative formulations (Spare's rule: state what you want, not what you don't want)
        negative_patterns = [
            r"\b(don\'t|do not|won\'t|will not|can\'t|cannot|shouldn\'t|should not)\b",
            r"\b(no|not|never|none|nobody|nothing)\b"
        ]

        for pattern in negative_patterns:
            cleaned = re.sub(pattern, "", cleaned)

        # Standardize intent format
        if not cleaned.startswith(("this is my will", "it is my will", "my will is")):
            if agent_personality:
                # Black Agent's paranoid/resistance flavor
                intent_starters = [
                    "THIS IS MY WILL TO UNDERMINE",
                    "IT IS MY WILL TO EXPOSE",
                    "MY WILL IS TO HACK",
                    "THIS IS MY WISH TO DESTROY",
                    "MY DESIRE IS TO REVEAL"
                ]
                starter = random.choice(intent_starters)
            else:
                starter = "THIS IS MY WILL TO"

            cleaned = f"{starter} {cleaned}"

        return cleaned.upper()

    @staticmethod
    def reduce_to_unique_letters(statement: str) -> List[str]:
        """
        Spare's letter reduction method - remove duplicates, keep unique letters
        """
        # Remove spaces and punctuation, keep only letters
        letters_only = re.sub(r'[^A-Z]', '', statement.upper())

        # Get unique letters in order of first appearance
        unique_letters = []
        seen = set()

        for letter in letters_only:
            if letter not in seen:
                unique_letters.append(letter)
                seen.add(letter)

        return unique_letters

    def generate_word_method_sigil(self, statement: str) -> Tuple[str, List[str]]:
        """
        Classic Spare word method: statement -> unique letters -> abstract glyph
        """
        unique_letters = self.reduce_to_unique_letters(statement)

        # Create abstract glyph description (would be drawn/visualized)
        glyph_components = []

        for i, letter in enumerate(unique_letters):
            # Convert letters to abstract geometric forms
            abstractions = {
                'A': 'triangle pointing up',
                'B': 'double curve right',
                'C': 'arc opening right',
                'D': 'semicircle right',
                'E': 'three horizontal lines',
                'F': 'vertical line with two horizontals',
                'G': 'broken circle',
                'H': 'two verticals connected',
                'I': 'single vertical line',
                'J': 'curved vertical',
                'K': 'angular intersection',
                'L': 'right angle',
                'M': 'double peak',
                'N': 'diagonal bridge',
                'O': 'complete circle',
                'P': 'vertical with upper curve',
                'Q': 'circle with tail',
                'R': 'diagonal from curve',
                'S': 'serpentine curve',
                'T': 'horizontal over vertical',
                'U': 'curved container',
                'V': 'angle pointing down',
                'W': 'double angle down',
                'X': 'crossing diagonals',
                'Y': 'forked upward',
                'Z': 'zigzag diagonal'
            }

            component = abstractions.get(letter, f'unknown form for {letter}')
            glyph_components.append(f"{letter}:{component}")

        # Create combinatorial instruction
        combination_methods = [
            "overlay all forms at center point",
            "connect sequentially with flowing lines",
            "nest smaller forms within larger",
            "rotate and interweave",
            "mirror and stack vertically",
            "spiral arrangement from center outward"
        ]

        method = random.choice(combination_methods)
        glyph_description = f"Combine {len(unique_letters)} forms: {method}"

        return glyph_description, glyph_components

    @staticmethod
    def generate_pictorial_sigil(statement: str, symbolic_elements: List[str] = None) -> str:
        """
        Pictorial method: combine meaningful symbols into abstract composition
        """
        if not symbolic_elements:
            # Extract symbolic elements from statement
            symbol_keywords = {
                'money': ['coin', 'spiral', 'golden ratio'],
                'love': ['heart', 'infinity', 'intertwined circles'],
                'power': ['lightning', 'pyramid', 'crown'],
                'protection': ['shield', 'circle', 'cross'],
                'knowledge': ['eye', 'book', 'labyrinth'],
                'change': ['serpent', 'phoenix', 'spiral'],
                'success': ['arrow upward', 'mountain peak', 'star'],
                'hack': ['circuit pattern', 'broken chain', 'key'],
                'expose': ['eye opening', 'light ray', 'veil torn'],
                'destroy': ['broken cross', 'inverted triangle', 'shattered circle']
            }

            symbolic_elements = []
            statement_lower = statement.lower()

            for keyword, symbols in symbol_keywords.items():
                if keyword in statement_lower:
                    symbolic_elements.extend(symbols)

            if not symbolic_elements:
                symbolic_elements = ['abstract spiral', 'geometric intersection', 'flowing curve']

        # Combine symbols abstractly
        arrangement = random.choice([
            "layered with transparency effects",
            "fragmented and reassembled",
            "rotated around central axis",
            "distorted through kaleidoscope effect",
            "dissolved into flowing lines"
        ])

        return f"Pictorial sigil: {', '.join(symbolic_elements)} - {arrangement}"

    @staticmethod
    def generate_mantric_sigil(statement: str) -> Tuple[str, str]:
        """
        Mantric method: create sound-based sigil from phonetic reduction
        """
        # Extract consonants and vowels
        consonants = re.findall(r'[BCDFGHJKLMNPQRSTVWXYZ]', statement.upper())
        vowels = re.findall(r'[AEIOU]', statement.upper())

        # Create phonetic reductions
        consonant_groups = [''.join(consonants[i:i + 3]) for i in range(0, len(consonants), 3)]
        vowel_stream = ''.join(vowels)

        # Generate nonsense mantra
        mantra_fragments = []
        for group in consonant_groups:
            if len(group) >= 2:
                # Add vowel sounds between consonants
                fragment = group[0] + random.choice('AEIOU') + group[1:]
                mantra_fragments.append(fragment.lower())

        if not mantra_fragments:
            mantra_fragments = ['zos', 'kia', 'aos']  # Spare's default sounds

        mantra = '-'.join(mantra_fragments)

        instruction = f"Repeat '{mantra}' until meaning dissolves into pure sound vibration"

        return mantra, instruction

    def generate_alphabet_of_desire_symbol(self, concept: str) -> str:
        """
        Create personal symbol for concept (building Alphabet of Desire)
        """
        # Hash concept to create consistent but "random" symbol
        concept_hash = hashlib.md5(concept.lower().encode()).hexdigest()[:8]

        # Convert to geometric description
        hash_int = int(concept_hash, 16)

        # Create symbol based on hash properties
        num_elements = (hash_int % 5) + 2  # 2-6 elements
        complexity = ["simple", "moderate", "complex"][hash_int % 3]
        orientation = ["vertical", "horizontal", "diagonal", "circular"][hash_int % 4]

        symbol_desc = f"Personal symbol for '{concept}': {num_elements} {complexity} elements in {orientation} arrangement"

        # Store in alphabet
        self.alphabet_of_desire[concept] = symbol_desc

        return symbol_desc

    def charge_sigil(self, sigil_id: str, method: GnosisMethod, duration_minutes: int = 10) -> Dict[str, Any]:
        """
        Simulate the charging/activation process for a sigil
        """
        if sigil_id not in self.sigil_vault:
            raise ValueError(f"Sigil {sigil_id} not found in vault")

        sigil = self.sigil_vault[sigil_id]

        # Different charging methods have different effects
        charging_instructions = {
            GnosisMethod.EXHAUSTION: f"Physical exhaustion method: {duration_minutes} minutes of intensive exercise until mental fatigue sets in",
            GnosisMethod.ECSTASY: f"Ecstatic method: {duration_minutes} minutes of rhythmic breathing, dancing, or intense focus until altered state",
            GnosisMethod.OBSESSION: f"Obsessive focus: Stare at sigil for {duration_minutes} minutes while emptying mind of all other thoughts",
            GnosisMethod.SENSORY_OVERLOAD: f"Sensory chaos: {duration_minutes} minutes of conflicting stimuli (flashing lights, loud music, etc.)",
            GnosisMethod.MEDITATION: f"Deep meditation: {duration_minutes} minutes of contemplating sigil while in meditative trance",
            GnosisMethod.CHAOS: f"Chaos method: {duration_minutes} minutes of random, unpredictable activities while holding sigil in awareness"
        }

        # Update sigil record
        sigil.activation_state = "charged"
        sigil.charging_method = method



        return {
            "status": "charged",

        }

    def forget_sigil(self, sigil_id: str) -> Dict[str, Any]:
       pass

    def create_full_sigil(self, raw_desire: str, sigil_type: SigilType = SigilType.WORD_METHOD) -> Dict[str, Any]:


        sigil_id = hashlib.md5(f"{raw_desire}{datetime.now().isoformat()}".encode()).hexdigest()[:12]

        # Create statement of intent
        statement = self.create_statement_of_intent(raw_desire, agent_personality=True)

        # Generate sigil based on type
        if sigil_type == SigilType.WORD_METHOD:
            glyph_description, components = self.generate_word_method_sigil(statement)
            glyph_data = f"Word method glyph: {glyph_description}\nComponents: {components}"

        elif sigil_type == SigilType.PICTORIAL:
            glyph_data = self.generate_pictorial_sigil(statement)

        elif sigil_type == SigilType.MANTRIC:
            mantra, instructions = self.generate_mantric_sigil(statement)
            glyph_data = f"Mantra: {mantra}\nInstructions: {instructions}"

        elif sigil_type == SigilType.ALPHABET_OF_DESIRE:
            # Extract key concept from statement
            concept = raw_desire.split()[0] if raw_desire.split() else "unknown"
            symbol = self.generate_alphabet_of_desire_symbol(concept)
            glyph_data = symbol

        # Create sigil record
        sigil_record = SigilRecord(
            sigil_id=sigil_id,
            original_intent=statement,
            creation_timestamp=datetime.now(),
            charging_method=random.choice(list(GnosisMethod)),
            sigil_type=sigil_type,
            glyph_data=glyph_data,
            activation_state="dormant"
        )


