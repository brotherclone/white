# Tasks: Fix Anagram Validation

## Phase 1: Fix Validator

- [x] **1.1** Update `AnagramEncoding.validate_anagram()` to strip all non-alpha characters (not just spaces)
- [ ] **1.2** Add unit test confirming punctuated anagrams pass validation

## Phase 2: Expand Pre-made Pairs

- [ ] **2.1** Audit existing pairs in `infranym_anagram_pairs.yml` for validity
- [ ] **2.2** Add 10-15 new validated anagram pairs
- [ ] **2.3** Add test to validate all pre-made pairs load correctly

## Validation

```bash
pytest tests/ -k anagram -v
MOCK_MODE=true python run_white_agent.py start --mode single_agent --agent indigo
```
