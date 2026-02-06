# Add Evolutionary Music Generator

## Problem

White Concept Agent generates concepts (text), but there's no system to:
1. Compose music that matches the chromatic mode
2. Explore variations systematically
3. Score candidates for chromatic consistency
4. Prune low-scoring candidates
5. Build arrangements incrementally (chords → drums → bass → melody)

## Solution

Build an evolutionary music composition system that uses the ML model as a fitness function.

## Architecture
```
┌─────────────────────────────────────────────────────────────┐
│ White Concept Agent                                         │
│ Output: "Spatial rebracketing in GREEN (Place) mode"        │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Evolutionary Music Generator                                │
│                                                             │
│ Stage 1: Chord Progressions                                │
│   - Generate 50 variations (rules, constraints, random)    │
│   - Score each with ML model                               │
│   - Keep top 3                                             │
│                                                             │
│ Stage 2: Drums                                             │
│   - For each top chord progression:                        │
│     - Generate 50 drum patterns                            │
│     - Score with ML model                                  │
│     - Keep top 3 per progression (9 total)                 │
│                                                             │
│ Stage 3: Bass                                              │
│   - For each top chord+drums:                              │
│     - Generate 50 bass lines                               │
│     - Score with ML model                                  │
│     - Keep top 3 (27 total candidates)                     │
│                                                             │
│ Stage 4: Melody/Harmony/FX...                              │
│   - Repeat pruning strategy                                │
│                                                             │
│ Stage N: Final Selection                                   │
│   - Top 3-5 candidates → human evaluation                  │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ DAW Export (Logic Pro, Ableton, etc.)                      │
└─────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Generation Engines
```python
class ChordProgressionGenerator:
    """Generate chord progressions matching chromatic mode."""
    
    def generate(self, concept, target_mode, n=50):
        """
        Args:
            concept: Text concept from White Agent
            target_mode: (temporal, spatial, ontological) tuple
            n: Number of variations to generate
        
        Returns:
            List of MIDI chord progressions
        """
        variations = []
        
        for i in range(n):
            # Strategy mix:
            if i < n/3:
                # Rule-based (music theory constraints)
                chords = self.theory_based(target_mode)
            elif i < 2*n/3:
                # Constrained random (weighted by mode)
                chords = self.weighted_random(target_mode)
            else:
                # Mutate existing (if retraining loop)
                chords = self.mutate(variations[i % len(variations)])
            
            variations.append(chords)
        
        return variations

class DrumPatternGenerator:
    """Generate drum patterns for a given chord progression."""
    
    def generate(self, chords, target_mode, n=50):
        # Similar strategy: theory, random, mutation
        pass

class BassLineGenerator:
    """Generate bass lines following chord roots + rhythmic variation."""
    
    def generate(self, chords, drums, target_mode, n=50):
        # Follow chord roots, add passing tones, rhythmic variety
        pass
```

### 2. Chromatic Scorer (ML Model Wrapper)
```python
class ChromaticScorer:
    """Score music candidates for chromatic consistency."""
    
    def __init__(self, model_path):
        self.model = load_multimodal_model(model_path)  # From Phase 3
    
    def score_batch(self, candidates, target_mode):
        """
        Score multiple candidates at once (batch inference).
        
        Args:
            candidates: List of MIDI files or AudioSegment objects
            target_mode: Target (temporal, spatial, ontological) distribution
        
        Returns:
            List of scores with confidence metrics
        """
        scores = []
        
        for candidate in candidates:
            # Render MIDI to audio (or use pre-rendered)
            audio = self.render_audio(candidate.midi)
            
            # Extract features
            audio_emb = self.model.audio_encoder(audio)
            midi_emb = self.model.midi_encoder(candidate.midi)
            lyric_emb = self.model.lyric_encoder(candidate.lyrics) if candidate.lyrics else None
            
            # Predict ontological distributions
            pred_temporal, pred_spatial, pred_ontological = self.model(
                audio_emb, midi_emb, lyric_emb
            )
            
            # Compute distance from target
            distance = kl_divergence(pred_temporal, target_mode[0]) + \
                       kl_divergence(pred_spatial, target_mode[1]) + \
                       kl_divergence(pred_ontological, target_mode[2])
            
            score = 1.0 / (1.0 + distance)  # Higher is better
            
            scores.append({
                'candidate_id': candidate.id,
                'score': score,
                'predicted_mode': (pred_temporal, pred_spatial, pred_ontological),
                'distance': distance,
                'confidence': min(pred_temporal.max(), pred_spatial.max(), pred_ontological.max())
            })
        
        return scores
```

### 3. Evolutionary Orchestrator
```python
class EvolutionaryComposer:
    """Orchestrate multi-stage evolutionary composition."""
    
    def __init__(self, concept, target_mode, scorer):
        self.concept = concept
        self.target_mode = target_mode
        self.scorer = scorer
        
        self.chord_gen = ChordProgressionGenerator()
        self.drum_gen = DrumPatternGenerator()
        self.bass_gen = BassLineGenerator()
    
    def compose(self, n_variants=50, top_k=3):
        """
        Generate arrangement through multi-stage evolution.
        
        Args:
            n_variants: Number of variations per stage
            top_k: Number of top candidates to keep per stage
        
        Returns:
            List of final candidates (MIDI + metadata)
        """
        # Stage 1: Chords
        chord_candidates = self.chord_gen.generate(
            self.concept, self.target_mode, n=n_variants
        )
        chord_scores = self.scorer.score_batch(chord_candidates, self.target_mode)
        top_chords = self.prune(chord_candidates, chord_scores, k=top_k)
        
        # Stage 2: Drums
        drum_candidates = []
        for chord in top_chords:
            drums = self.drum_gen.generate(chord, self.target_mode, n=n_variants)
            drum_candidates.extend(drums)
        
        drum_scores = self.scorer.score_batch(drum_candidates, self.target_mode)
        top_drums = self.prune(drum_candidates, drum_scores, k=top_k)
        
        # Stage 3: Bass
        bass_candidates = []
        for drums in top_drums:
            basses = self.bass_gen.generate(drums.chords, drums, self.target_mode, n=n_variants)
            bass_candidates.extend(basses)
        
        bass_scores = self.scorer.score_batch(bass_candidates, self.target_mode)
        top_bass = self.prune(bass_candidates, bass_scores, k=top_k)
        
        # Stage N: Continue for melody, harmony, FX...
        
        return top_bass  # Final candidates
    
    def prune(self, candidates, scores, k=3):
        """Keep top-k candidates by score."""
        sorted_candidates = sorted(
            zip(candidates, scores),
            key=lambda x: x[1]['score'],
            reverse=True
        )
        return [c for c, s in sorted_candidates[:k]]
```

## Integration with Existing Systems

### Use Existing Chord Generator
```python
# app/generator/chord_progression.py already exists!
# Wrap it in the evolutionary framework:

class ChordProgressionGenerator:
    def __init__(self):
        # Reuse existing ChordProgressionGenerator
        from app.generator.chord_progression import ChordProgressionGenerator as ExistingGen
        self.generator = ExistingGen()
    
    def generate(self, concept, target_mode, n=50):
        variations = []
        
        for i in range(n):
            # Use different strategies from existing generator
            if i < n/3:
                chords = self.generator.generate_from_mode(target_mode)
            elif i < 2*n/3:
                chords = self.generator.generate_random()
            else:
                # Mutate previous variants
                parent = variations[i % len(variations)]
                chords = self.mutate_progression(parent)
            
            variations.append(chords)
        
        return variations
```

## Evaluation

### Fitness Function Quality
- **Correlation with human judgment**: Do high-scoring candidates sound more "GREEN" to humans?
- **Diversity**: Are top-k candidates sufficiently different from each other?
- **Mode discrimination**: Can the scorer distinguish GREEN from RED?

### Generation Quality
- **Musicality**: Are generated progressions/patterns musically coherent?
- **Variety**: Does the system explore diverse solutions?
- **Convergence**: Does pruning eliminate bad candidates effectively?

## User Workflow
```bash
# 1. White Agent generates concept
python -m app.white_agent.generate_concept --mode GREEN

# Output: "Spatial rebracketing through coastal erosion"

# 2. Evolutionary composer generates music
python -m app.generator.evolutionary_compose \
    --concept "Spatial rebracketing through coastal erosion" \
    --mode GREEN \
    --n-variants 50 \
    --top-k 3 \
    --stages chords,drums,bass,melody

# Output: 3 MIDI files with scores:
# - candidate_001.mid (score: 0.94)
# - candidate_002.mid (score: 0.91)
# - candidate_003.mid (score: 0.89)

# 3. Human evaluation
# Listen to candidates, choose favorite
# Export to Logic Pro for final production
```

## Timeline

- **Week 1**: Wrap existing chord generator in evolutionary framework
- **Week 2**: Implement drum + bass generators
- **Week 3**: Integrate Phase 3 multimodal scorer
- **Week 4**: Testing, refinement, human evaluation

## Dependencies

- Phase 3 (Multimodal Fusion) must be complete
- Phase 4 (Regression) already complete
- Existing `app/generator/chord_progression.py`
- MIDI rendering (FluidSynth, TiMidity, or send to Logic Pro)

## Future Extensions

- **Interactive evolution**: Human marks favorites, system generates more variations
- **Multi-objective optimization**: Balance chromatic consistency + musicality + novelty
- **Style transfer**: "Make this RED progression sound like GREEN"
- **Lyrics generation**: Generate vocals matching the instrumental arrangement