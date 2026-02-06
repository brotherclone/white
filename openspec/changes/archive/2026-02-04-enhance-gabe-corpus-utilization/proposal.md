# Change: Enhance Gabe Corpus Utilization in Violet Agent Interviews

## Why
The `gabe_corpus.md` contains rich personality data (7 parts covering voice, humor, psychological patterns, aesthetic sensibilities) but only ~3000 characters are currently used in the `simulated_interview` prompt. The resulting interview responses lack:

1. **Psychological depth** — The current faux voice is too confident/cavalier. The corpus documents guilt-as-creative-fuel, creative rejection trauma, the wound-as-insight pattern. Gabe's voice oscillates between intellectual confidence and self-doubt, not pure bravado.

2. **Bracket-switching nuance** — The academic→profane pattern is prompted but without corpus examples to ground it.

3. **Interviewer awareness** — The interviewer persona has no context about who they're interviewing (Gabe's documented patterns, aesthetic sensibilities, vulnerabilities).

## What Changes
- **RAG-based Retrieval**: Embed corpus chunks and retrieve semantically relevant passages based on interview question content
- **Psychological Grounding**: Surface Part 3 (Jungian) and Part 4 (Personality) content — the guilt complex, creative rejection wounds, INTP-with-feeling patterns
- **Interviewer Briefing**: Feed `generate_questions` with Gabe's documented style so interviewers can probe meaningfully
- **Voice Calibration**: Include phrase bank and humor patterns from Parts 5-6

## Selected Approach: B (RAG with Embeddings)

Embed corpus chunks and retrieve semantically relevant passages based on the interview question being asked.

**Why this approach:**
- Dynamic retrieval means psychological content surfaces when questions touch on vulnerability/doubt
- Scales naturally as corpus grows
- Question-aware context — a question about "inspiration" retrieves different corpus chunks than one about "process"
- Aligns with existing project patterns (RAG is already conceptually present in the codebase)

**Implementation notes:**
- Use sentence-transformers or similar for embeddings
- Chunk corpus by markdown headers (## Part N sections, then ### subsections)
- Store embeddings at init, retrieve top-k chunks per question
- Combine with a minimal "voice card" for consistent baseline

## Impact
- Affected code: `app/agents/violet_agent.py` (simulated_interview, generate_questions, _load_corpus)
- New files: Corpus embedder utility, possibly cached embeddings file
- Affected files: `app/reference/biographical/gabe_corpus.md` (may need consistent header structure)

## Psychological Tone Shift

### Current (too cavalier):
> "It's really about Heideggerian temporality meets information theory, but also it fucking slaps"

### Target (with vulnerability):
> "It's really about Heideggerian temporality meets information theory — which, I'm aware this sounds like I'm trying too hard, but also it fucking slaps and honestly I don't know if that's enough"

The difference: self-awareness that includes doubt, not just ironic detachment. The corpus documents:
- "I can't reconcile the guilt of again ruining [my father's] artistic aspirations"
- "I feel very isolated and prefer my own imaginary world"
- "kicked out of gifted and talented program for spacing out"
- The repeated pattern of creative rejection → internalized doubt → transformation into methodology

This isn't pure confidence. It's earned confidence sitting on top of documented wounds.

## Open Questions (Resolved)
1. ~~Which approach?~~ → **Approach B (RAG)**
2. ~~Should corpus be restructured?~~ → Light restructuring for consistent headers if needed
3. ~~What patterns are missing?~~ → **Psychological depth, vulnerability, self-doubt**
4. ~~Should generate_questions use corpus?~~ → **Yes, brief the interviewer**
