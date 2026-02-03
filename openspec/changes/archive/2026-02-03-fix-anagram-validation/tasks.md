# Tasks: Fix Anagram Validation

## Phase 1: Fix Validator

- [x] **1.1** Update `AnagramEncoding.validate_anagram()` to strip all non-alpha characters (not just spaces)
- [x] **1.2** Add unit test confirming punctuated anagrams pass validation

## Phase 2: Expand Pre-made Pairs

- [x] **2.1** Audit existing pairs in `anagram_pairs.py` for validity (all 29 original pairs valid)
- [x] **2.2** Add 10-15 new validated anagram pairs (added 21 new pairs, total now 50)
- [x] **2.3** Add test to validate all pre-made pairs load correctly

## Validation

```bash
pytest tests/reference/test_anagram_pairs.py -v  # 109 tests pass
```
