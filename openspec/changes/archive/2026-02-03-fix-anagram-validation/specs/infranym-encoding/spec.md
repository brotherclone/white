## ADDED Requirements

### Requirement: Anagram Validation
The AnagramEncoding validator SHALL compare only alphabetic characters when validating anagram pairs, ignoring spaces, punctuation, and case.

#### Scenario: Punctuated anagram validation
- **WHEN** validating an anagram pair containing punctuation (apostrophes, commas, periods)
- **THEN** the validator SHALL strip all non-alphabetic characters before comparison
- **AND** the validation SHALL pass if the remaining letters are anagrams

#### Scenario: Case-insensitive validation
- **WHEN** validating an anagram pair with mixed case
- **THEN** the validator SHALL normalize to uppercase before comparison
- **AND** the original casing SHALL be preserved in the stored value
