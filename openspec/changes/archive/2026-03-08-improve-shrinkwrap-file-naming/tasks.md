## 1. Core filename cleaning
- [x] 1.1 Add `clean_filename(raw_name: str) -> str` to `shrinkwrap_chain_artifacts.py` implementing the five cleaning rules
- [x] 1.2 Add `resolve_collision()`: when the clean name already exists in the destination subdir, append `_2`, `_3`, … before the extension
- [x] 1.3 Update `copy_thread_files()` to call `clean_filename()` and `resolve_collision()` for every file before writing

## 2. In-file reference updates
- [x] 2.1 After copying, rewrite any `file_name:` line in YAML files to the clean filename
- [x] 2.2 After copying, rewrite any `file_name:` line in Markdown files (including YAML-body `.md` files) to the clean filename

## 3. Directory naming
- [x] 3.1 Confirmed `generate_directory_name()` produces `{color}-{slugified-title}` with no `white-` prefix — no code change needed

## 4. Tests
- [x] 4.1 Unit test `clean_filename()` for each of the five input patterns and the no-match pass-through
- [x] 4.2 Unit test collision handling produces `name_2.ext` on first collision and `name_3.ext` on second
- [x] 4.3 Integration test: shrinkwrap a mock thread directory and assert all output files carry clean names and updated `file_name` fields
