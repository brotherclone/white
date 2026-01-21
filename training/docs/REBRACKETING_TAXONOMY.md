# Rebracketing Type Taxonomy

## Overview

This document defines the taxonomy of rebracketing types used in multi-class classification for The White Album project. Rebracketing is a technique for ontological transformation where concepts are re-parsed across different dimensional boundaries to reveal new meanings.

## The Eight Rebracketing Types

### 1. Spatial Rebracketing (class_id: 0)

**Definition**: Re-parsing concepts across spatial boundaries, dimensions, or geometric relationships.

**Characteristics**:
- Transforms understanding through spatial reorganization
- Changes boundaries of "inside" vs "outside"
- Shifts perspective on containment and proximity
- Often involves physical or metaphorical space

**Examples**:
- "The room [(is not)] a container" → "The room [is (not a) container]"
- Reversing figure/ground relationships
- Nested spaces becoming adjacent spaces

**Markers**:
- Spatial prepositions: in, on, through, around, between
- Dimensional language: inside/outside, surface/depth
- Geometric concepts: boundary, edge, center

---

### 2. Temporal Rebracketing (class_id: 1)

**Definition**: Re-parsing concepts across temporal boundaries or sequential relationships.

**Characteristics**:
- Transforms understanding of before/after/during
- Changes event boundaries and durations
- Shifts causal sequences through temporal reframing
- Reveals hidden simultaneities or sequences

**Examples**:
- "The song [starts when] it ends" → "The song starts [(when it) ends]"
- Beginning treated as ending, or vice versa
- Continuous process seen as discrete events

**Markers**:
- Temporal language: before, after, during, when, while
- Process descriptions: beginning, ending, continuing
- Sequential markers: first, then, next, finally

---

### 3. Causal Rebracketing (class_id: 2)

**Definition**: Re-parsing cause-effect relationships or dependency structures.

**Characteristics**:
- Transforms understanding of what causes what
- Reverses or reframes causal chains
- Changes agency and responsibility
- Reveals hidden dependencies or breaks apparent ones

**Examples**:
- "Fear [causes] avoidance" → "Fear causes [avoidance]" (where avoidance becomes self-sustaining)
- Effect becoming cause
- Mutual causation rather than linear

**Markers**:
- Causal language: because, therefore, causes, leads to
- Agency markers: makes, forces, enables, prevents
- Conditional structures: if/then relationships

---

### 4. Perceptual Rebracketing (class_id: 3)

**Definition**: Re-parsing how sensory or perceptual experiences are organized and interpreted.

**Characteristics**:
- Transforms what counts as figure vs ground
- Changes sensory modality associations
- Shifts attention focus
- Reveals alternative perceptual organizations

**Examples**:
- "The [silence between] notes" → "The silence [between notes]"
- Background noise becoming primary signal
- Synesthetic re-mappings

**Markers**:
- Sensory language: see, hear, feel, taste, smell
- Attention markers: notice, focus, aware, perceive
- Gestalt terms: figure, ground, pattern, whole

---

### 5. Memory Rebracketing (class_id: 4)

**Definition**: Re-parsing how experiences are remembered, stored, or recalled.

**Characteristics**:
- Transforms what is remembered vs forgotten
- Changes temporal organization of memories
- Shifts between episodic and semantic memory
- Reveals constructed nature of recollection

**Examples**:
- "I [remember forgetting]" → "I remember [forgetting]"
- Past event reframed as present memory
- Memory of an emotion vs emotion about a memory

**Markers**:
- Memory language: remember, forget, recall, recognize
- Past tense constructions with present implications
- Nostalgia, déjà vu, flashback terminology

---

### 6. Ontological Rebracketing (class_id: 5)

**Definition**: Re-parsing what exists, what kind of thing something is, or category membership.

**Characteristics**:
- Transforms being vs non-being
- Changes category boundaries
- Shifts between literal and metaphorical existence
- Questions fundamental nature of entities

**Examples**:
- "The [(fictional) character] exists" → "The [fictional (character exists)]"
- Abstract concepts treated as concrete entities
- Properties becoming substances

**Markers**:
- Existential language: is, exists, being, real
- Category terms: type, kind, sort, category
- Essence/appearance distinctions

---

### 7. Narrative Rebracketing (class_id: 6)

**Definition**: Re-parsing story structure, perspective, or narrative framing.

**Characteristics**:
- Transforms who tells vs who experiences
- Changes plot boundaries and story structure
- Shifts narrative perspective or voice
- Reveals meta-narrative layers

**Examples**:
- "The narrator [tells a] story" → "The [narrator tells] a story" (where telling becomes the story)
- Story within a story reframings
- Character becoming narrator

**Markers**:
- Narrative language: tells, story, plot, character
- Perspective markers: I, you, they, narrator
- Meta-narrative terms: fiction, real, representation

---

### 8. Identity Rebracketing (class_id: 7)

**Definition**: Re-parsing self/other boundaries or identity categories.

**Characteristics**:
- Transforms who is who
- Changes boundaries between self and other
- Shifts identity category memberships
- Reveals constructed nature of identity

**Examples**:
- "I [am not] myself" → "I am [(not myself)]"
- Multiple or shifting selves
- Collective identity parsing

**Markers**:
- Identity language: I, self, who, identity
- Pronouns used reflexively: myself, yourself
- Role/category terms: person, individual, we/they

---

## Class Mapping for Configuration

```yaml
class_mapping:
  spatial: 0
  temporal: 1
  causal: 2
  perceptual: 3
  memory: 4
  ontological: 5
  narrative: 6
  identity: 7
```

## Multi-Label Considerations

Segments may exhibit multiple rebracketing types simultaneously. For example:

- **Temporal + Spatial**: "The [moment (between walls)]" - temporal duration bounded spatially
- **Perceptual + Memory**: "I [see (remembering)]" - sensory experience of recollection
- **Ontological + Identity**: "The [(fictional) self]" - identity category as existence question

When using multi-label mode (`multi_label: true`), the model predicts which types are present rather than selecting a single dominant type.

## Annotation Guidelines

When labeling segments for training:

1. **Read the full concept** - Don't judge from fragments
2. **Identify bracketing markers** - Look for actual [ ] or other parsing cues
3. **Consider primary mechanism** - What dimension is being reorganized?
4. **Check for multiple types** - Segments can have 0-N types
5. **Unknown types** - If genuinely ambiguous, exclude from training

## Label Distribution Analysis

Track class distribution to ensure balanced training:
- Rare types may need synthetic augmentation
- Common types may need downsampling
- Use `class_weights: "balanced"` in config to handle imbalance

## Model Outputs

### Single-Label Mode
- Input: Concept text
- Output: Single class index (0-7)
- Interpretation: Dominant rebracketing type

### Multi-Label Mode
- Input: Concept text
- Output: Binary vector [8]
- Interpretation: Presence/absence of each type (threshold: 0.5)

## Evaluation Considerations

- **Per-class F1**: Some types may be easier to learn
- **Confusion pairs**: Temporal/Causal often confused
- **Macro F1**: Unweighted average - good for balanced evaluation
- **Micro F1**: Weighted by frequency - good for overall performance

## Future Extensions

Potential additional types under consideration:
- **Linguistic rebracketing**: Morphological or syntactic parsing shifts
- **Modal rebracketing**: Possibility/necessity boundary shifts
- **Affective rebracketing**: Emotional valence reorganization

## References

- White Album Project Documentation
- OpenSpec: `add-multiclass-rebracketing-classifier`
- Training Configuration: `config_multiclass.yml`
