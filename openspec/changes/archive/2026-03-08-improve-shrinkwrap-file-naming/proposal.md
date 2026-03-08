# Change: Clean file names during shrink-wrap

## Why
Raw chain artifact filenames embed UUIDs, color character codes, and agent/thread name prefixes that make the output directory hard to read and browse. The reference run (`shrink_wrapped/all-frequencies-return-to-source-the-consciousness-archaeology-complete/`) demonstrates the desired clean state, but that cleaning was done by hand. This change bakes the same logic into the shrink-wrap process so every run produces readable output automatically.

## What Changes
- `copy_thread_files()` applies filename cleaning rules before writing each file to the output directory
- A new `clean_filename()` helper strips UUID prefixes, color-char codes, agent/thread name prefixes, and `song_proposal_<Color...>_` prefixes
- Collision handling: when two files in the same subdirectory would produce the same clean name, a `_2`, `_3`, … suffix is appended before the extension
- After copying, any `file_name` field found in YAML frontmatter or Markdown YAML front-matter blocks is rewritten to the clean filename
- Directory names no longer carry a `white-` prefix (the current `generate_directory_name()` already produces `{color}-{title}`; this change confirms and spec-documents that behaviour)

## Filename Cleaning Rules

| Input pattern | Output |
|---|---|
| `<uuid>_<char>_<name>.<ext>` | `<name>.<ext>` |
| `white_agent_<thread-uuid>_<TYPE>.<ext>` | `<type_lowercase>.<ext>` |
| `all_song_proposals_<thread-uuid>.<ext>` | `all_song_proposals.<ext>` |
| `song_proposal_<Color (0xHEX)>_<name>.<ext>` | `<name>.<ext>` |
| `song_proposal_<char>_<name>.<ext>` | `<name>.<ext>` |
| No match (already clean) | unchanged |

## Impact
- Affected specs: `chain-artifacts`
- Affected code: `app/util/shrinkwrap_chain_artifacts.py`
- Existing shrink_wrapped directories are **not** retroactively renamed by this change — they reflect prior manual or automated runs
- `file_name` field updates are best-effort: if parsing fails the file is still copied with the clean filename
