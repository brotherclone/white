## 1. Core filename cleaning
- [ ] 1.1 Add `clean_filename(raw_name: str, thread_id: str) -> str` to `shrinkwrap_chain_artifacts.py` implementing the five cleaning rules
- [ ] 1.2 Add collision handling: when the clean name already exists in the destination subdir, append `_2`, `_3`, … before the extension
- [ ] 1.3 Update `copy_thread_files()` to call `clean_filename()` for every file before writing

## 2. In-file reference updates
- [ ] 2.1 After copying, scan each copied YAML file for a top-level `file_name:` key and rewrite its value to the clean filename
- [ ] 2.2 After copying, scan each copied Markdown file for YAML front-matter (between `---` fences) containing `file_name:` and rewrite it

## 3. Directory naming
- [ ] 3.1 Confirm `generate_directory_name()` produces `{color}-{slugified-title}` with no `white-` prefix (no code change expected — just verify and document)

## 4. Tests
- [ ] 4.1 Unit test `clean_filename()` for each of the five input patterns and the no-match pass-through
- [ ] 4.2 Unit test collision handling produces `name_2.ext` on first collision and `name_3.ext` on second
- [ ] 4.3 Integration test: shrinkwrap a mock thread directory and assert all output files carry clean names and updated `file_name` fields
