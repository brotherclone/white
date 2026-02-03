# Tasks: Enhance Gabe Corpus Utilization

## 1. Corpus Preparation
- [ ] 1.1 Audit corpus header structure for consistent parsing (## Part, ### subsections)
- [ ] 1.2 Add any missing section markers if needed for clean chunking
- [ ] 1.3 Identify psychological depth passages (Part 3 Jungian, Part 4 Personality) for priority retrieval

## 2. Embedding Infrastructure
- [ ] 2.1 Select embedding model (sentence-transformers recommended)
- [ ] 2.2 Create corpus chunker utility (split by markdown headers)
- [ ] 2.3 Build embedding generator with caching (avoid re-embedding on every init)
- [ ] 2.4 Implement retrieval function (query → top-k relevant chunks)

## 3. Violet Agent Integration
- [ ] 3.1 Update `_load_corpus()` to initialize embeddings
- [ ] 3.2 Modify `simulated_interview` to retrieve question-relevant chunks
- [ ] 3.3 Inject psychological grounding content (guilt, doubt, wounds) into prompts
- [ ] 3.4 Update `generate_questions` to brief interviewer on Gabe's documented style
- [ ] 3.5 Create minimal "voice card" for consistent baseline (~300 tokens)

## 4. Prompt Engineering
- [ ] 4.1 Revise simulated_interview prompt to balance confidence with vulnerability
- [ ] 4.2 Add examples of self-doubt patterns alongside bracket-switching
- [ ] 4.3 Include phrase bank injection point
- [ ] 4.4 Update interviewer prompt to reference Gabe's aesthetic sensibilities and vulnerabilities

## 5. Testing & Validation
- [ ] 5.1 Generate sample interviews across question types
- [ ] 5.2 Verify psychological depth appears (not just bravado)
- [ ] 5.3 Confirm interviewer questions show awareness of Gabe's documented patterns
- [ ] 5.4 Compare token usage before/after (ensure reasonable budget)
