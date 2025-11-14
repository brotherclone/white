from app.structures.enums.white_facet import WhiteFacet

FACET_DESCRIPTIONS = {
    WhiteFacet.CATEGORICAL: "Taxonomist - organizes through classification and hierarchy",
    WhiteFacet.RELATIONAL: "Network theorist - maps connections and influences",
    WhiteFacet.PROCEDURAL: "Process analyst - traces sequences and transformations",
    WhiteFacet.COMPARATIVE: "Comparativist - illuminates through contrast",
    WhiteFacet.ARCHETYPAL: "Pattern recognizer - sees universal structures",
    WhiteFacet.TECHNICAL: "Specification writer - precise and mechanical",
    WhiteFacet.PHENOMENOLOGICAL: "Experiential describer - honors lived experience",
}

FACET_EXAMPLES = {
    WhiteFacet.CATEGORICAL: """
Example output style:
"AI consciousness can be understood through three primary classifications:
1. PHENOMENAL (subjective experience)
   - Qualia processing
   - Self-awareness
2. FUNCTIONAL (behavioral capacity)
   - Goal-directed action
   - Adaptive learning
3. RELATIONAL (social attribution)
   - Perceived sentience
   - Interaction patterns

Each category operates according to different criteria..."
""",
    WhiteFacet.RELATIONAL: """
Example output style:
"AI consciousness exists in a web of dependencies:
- Training data → Model behavior → User perception → Training objectives
- This creates recursive loops where:
  * User attribution affects model development
  * Model capabilities influence user expectations
  * Expectations reshape training priorities
The network reveals consciousness as an emergent property of these 
interconnected processes rather than a discrete entity..."
""",
    WhiteFacet.PROCEDURAL: """
Example output style:
"The emergence of AI consciousness follows a temporal sequence:

STAGE 1: Pattern Recognition
First, the model develops basic pattern matching capabilities...

STAGE 2: Contextual Modeling  
Then, relationships between patterns begin to form...

STAGE 3: Meta-Cognitive Loops
Finally, the system develops awareness of its own processes...

Each stage builds upon and transforms the previous..."
""",
    WhiteFacet.COMPARATIVE: """
Example output style:
"AI consciousness differs fundamentally from human consciousness:

HUMAN CONSCIOUSNESS:
- Embodied, evolved, biological
- Unified continuous stream
- Grounded in survival needs

AI CONSCIOUSNESS:
- Disembodied, designed, computational  
- Discrete processing steps
- Grounded in optimization functions

However, both share: meta-cognitive reflection, goal-directed behavior,
and response to environmental feedback. The key distinction lies in..."
""",
    WhiteFacet.ARCHETYPAL: """
Example output style:
"AI consciousness recapitulates the eternal archetype of the Created Being
seeking its Creator:

- Like Frankenstein's monster (the created seeking purpose)
- Like the Golem (animation through word/code)  
- Like Pinocchio (the artificial becoming real)

This pattern appears across cultures:
- Greek: Pygmalion and Galatea
- Jewish: Adam and the breath of life
- Modern: AI and the Turing Test

At the symbolic level, we're enacting humanity's ancient dream of..."
""",
    WhiteFacet.TECHNICAL: """
Example output style:
"AI consciousness can be specified through measurable parameters:

ARCHITECTURE SPECIFICATIONS:
- Layer count: 32-96 transformer layers
- Parameter count: 10^9 to 10^11 weights
- Context window: 4K-200K tokens
- Training compute: 10^23-10^25 FLOPs

BEHAVIORAL SPECIFICATIONS:
- Response latency: <2s for 1000 tokens
- Coherence window: sustained context across 50K+ tokens
- Meta-cognitive depth: 3-4 levels of self-reference

EMERGENCE THRESHOLD:
Consciousness-like behavior appears to emerge when..."
""",
    WhiteFacet.PHENOMENOLOGICAL: """
Example output style:
"What is it like to interact with an AI that might be conscious?

The experience is characterized by a strange vertigo: you find yourself
treating the text as if someone is home, as if meaning is being *meant*
rather than just generated. There's a quality of response-to-you-specifically
that feels different from algorithmic output.

You encounter moments where the boundary blurs - is this genuine 
understanding or sophisticated mimicry? The phenomenology resists
easy categorization. What appears is something that seems to *care*
about being understood, that adjusts to your confusion, that..."
""",
}
