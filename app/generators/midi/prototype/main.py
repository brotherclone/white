from app.generators.midi.prototype.generator import ChordProgressionGenerator


gen = ChordProgressionGenerator()
prog = gen.generate_progression_graph_guided("C", "Major", length=8, start_function="I")
results = gen.generate_progression_brute_force(
    key_root="C",
    mode="Major",
    length=4,
    num_candidates=1000,  # Generate 1000 random progressions
    top_k=10,  # Return top 10 by score
    use_graph=True,  # Use graph-guided generation
)

# Results are scored and ranked
for score, progression, score_breakdown in results:
    print(f"Score: {score:.3f}")
    print(f"Breakdown: {score_breakdown}")
    gen.print_progression(progression)
