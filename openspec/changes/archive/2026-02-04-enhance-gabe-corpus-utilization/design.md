# Design: Gabe Corpus RAG Implementation

## Context
The Violet Agent simulates Gabe's voice in interview responses. Currently it truncates the corpus to 3000 chars, missing psychological depth and nuance. We need question-aware retrieval that surfaces relevant personality patterns.

## Goals
- Retrieve semantically relevant corpus passages per interview question
- Surface psychological depth (vulnerability, doubt) not just confident bravado
- Brief interviewers on Gabe's documented patterns
- Stay within reasonable token budgets (~4000 tokens corpus content max)

## Non-Goals
- Full vector database infrastructure (overkill for single-document retrieval)
- Real-time corpus updates (corpus changes infrequently)
- Multi-user embedding isolation (single-user system)

## Decisions

### Embedding Model: sentence-transformers/all-MiniLM-L6-v2
**Why:** Lightweight, fast, good semantic similarity for short passages. Already common in Python ML stacks. No API dependency.

**Alternatives considered:**
- OpenAI embeddings — Better quality but adds API dependency, cost, latency
- Anthropic embeddings — Not yet available
- Larger sentence-transformers — Diminishing returns for this corpus size

### Chunking Strategy: Markdown Header-Based
**Why:** Corpus is already structured with `## Part N` and `### Subsection` headers. Natural semantic boundaries.

```
Chunk examples:
- "## Part 3: Jungian Psychoanalysis" → full section as one chunk
- "### Shadow Integration" → subsection chunk
- "### Creative Rejection Trauma" → subsection chunk
```

**Chunk size target:** 200-500 tokens per chunk. Split large sections at `###` level.

### Embedding Storage: JSON file alongside corpus
**Why:** Simple, no database dependency. Corpus is ~10KB, embeddings add ~100KB. Regenerate on corpus file change (check mtime).

```
app/reference/biographical/
├── gabe_corpus.md
└── gabe_corpus_embeddings.json  # Generated, gitignored
```

### Retrieval: Top-5 chunks per question
**Why:** Balance between context richness and token budget. 5 chunks × ~400 tokens = ~2000 tokens, leaving room for voice card and prompt structure.

### Voice Card: Static 300-token baseline
**Why:** Ensures consistent voice even if retrieval returns unusual chunks. Always present in prompts.

Content outline:
```markdown
VOICE BASELINE:
- Pattern: Academic/theoretical (10-30 words) → hard cut → profane/mundane (10-20 words)
- Frequency: Every 2-3 sentences
- Tone: Intellectual confidence WITH self-doubt. Not pure bravado.
- Key tells: Self-aware about pretension, guilt-as-fuel, wound-as-insight
- Phrases: "rebracketing methodology", "one thing per file", "let reality inform architecture"
- Humor: Dad jokes + immediate self-flagellation, absurdist deadpan, tech frustration comedy
```

## Architecture

```
VioletAgent.__init__
    └── _load_corpus()
            ├── Load gabe_corpus.md
            ├── Check embeddings cache (mtime comparison)
            │   ├── Cache valid → Load embeddings
            │   └── Cache stale → Regenerate embeddings
            └── Store: self.corpus_chunks, self.corpus_embeddings

simulated_interview(question)
    ├── Embed question text
    ├── Retrieve top-5 similar chunks
    ├── Build prompt:
    │   ├── Voice card (static 300 tokens)
    │   ├── Retrieved chunks (~2000 tokens)
    │   └── Question + proposal context
    └── Generate response

generate_questions()
    ├── Retrieve chunks relevant to "aesthetic", "style", "vulnerabilities"
    ├── Brief interviewer on Gabe's documented patterns
    └── Generate questions that can probe meaningfully
```

## Risks & Mitigations

| Risk                                | Mitigation                                                  |
|-------------------------------------|-------------------------------------------------------------|
| Embedding model adds dependency     | sentence-transformers is well-maintained, can pin version   |
| Retrieval returns irrelevant chunks | Voice card provides baseline; can tune similarity threshold |
| Token budget exceeded               | Hard cap at 4000 tokens corpus content; truncate if needed  |
| Slow init from embedding generation | Cache embeddings; only regenerate on corpus change          |

## Migration Plan
1. Add sentence-transformers to requirements
2. Implement corpus chunker + embedder as utility
3. Update VioletAgent._load_corpus() to use new infrastructure
4. Modify prompts to use retrieval + voice card
5. Test with sample interviews
6. Monitor token usage and response quality

## Open Questions
- Should embeddings be generated at build time vs runtime? (Leaning runtime with cache)
- Exact similarity threshold for "relevant" chunks? (Start with cosine > 0.3, tune from there)
