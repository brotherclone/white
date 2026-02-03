# Change: Fix Anagram Validation and Expand Pre-made Pairs

## Why

Anagram validation fails with punctuation in surface phrases. The validator strips spaces but not apostrophes or other punctuation, causing valid anagrams like "I'm A Dot In Place" (anagram of "A Decimal Point") to be rejected.

**Error:**
```
ValueError: 'I'm A Dot In Place' is not a valid anagram of 'A Decimal Point'
```

**Root cause:** `AnagramEncoding.validate_anagram()` uses `.replace(" ", "")` but doesn't strip punctuation.

## What Changes

### 1. Fix Validator
Update `app/agents/tools/encodings/anagram_encodings.py` to strip all non-alphabetic characters:
```python
# Before
secret = info.data.get("secret_word", "").upper().replace(" ", "")
surface = v.upper().replace(" ", "")

# After
import re
secret = re.sub(r'[^A-Z]', '', info.data.get("secret_word", "").upper())
surface = re.sub(r'[^A-Z]', '', v.upper())
```

### 2. Expand Pre-made Pairs
Add more validated anagram pairs to `app/reference/music/infranym_anagram_pairs.yml` for richer fallback options. Current count: 12 pairs.

## Impact

- Affected code: `anagram_encodings.py`, `infranym_anagram_pairs.yml`
- Fixes runtime errors during text artifact assembly
- Backward compatible - existing valid pairs still work
