## ADDED Requirements

### Requirement: Single virtual environment
The project SHALL maintain exactly one virtual environment (`.venv`) for all development,
testing, and pipeline execution. A second environment (`.venv312`) SHALL NOT be required
for any task. This is achieved by pinning `transformers>=4.47,<5` in `pyproject.toml`.

#### Scenario: Full test suite runs under .venv
- **WHEN** `python -m pytest tests/` is run under `.venv`
- **THEN** all tests pass without any subprocess re-dispatch to an alternate interpreter

#### Scenario: Refractor imports cleanly under .venv
- **WHEN** `from training.refractor import Refractor` is executed under `.venv`
- **THEN** no ImportError and no venv-mismatch warning is emitted

#### Scenario: Pipeline commands use .venv
- **WHEN** any pipeline step from SONG_GENERATION_PROCESS.md is run
- **THEN** the command SHALL use `.venv/bin/python` not `.venv312/bin/python`

### Requirement: transformers version pinned below 5.x
`pyproject.toml` SHALL pin `transformers>=4.47,<5` to prevent uv from resolving to a
breaking release and to document the constraint explicitly.

#### Scenario: uv resolves transformers to 4.x
- **WHEN** `uv sync` is run
- **THEN** `uv.lock` SHALL record a transformers version in the `4.x` series

### Requirement: Python 3.12 minimum declared
`pyproject.toml` SHALL declare `requires-python = ">=3.12"` to match the `.python-version`
pin and make the project's actual Python requirement explicit.

#### Scenario: requires-python matches .python-version
- **GIVEN** `.python-version` contains `3.12`
- **THEN** `pyproject.toml` `requires-python` SHALL be `">=3.12"`

### Requirement: venv312 test infrastructure removed
The codebase SHALL contain no venv312 pytest marker, no conftest subprocess re-runner,
and no test decorators referencing venv312. The marker SHALL NOT appear in pytest.ini
or tests/conftest.py.

#### Scenario: No venv312 marker in test suite
- **WHEN** `grep -r "venv312" tests/` is run
- **THEN** no matches SHALL be returned
