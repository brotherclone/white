## Context

The `.venv312` workaround exists because `uv` resolves `transformers` to 5.x when
unpinned, and transformers 5.x introduced a breaking change to `torch.load()` pickle
behaviour. The Refractor scorer needs to load a DeBERTa tokenizer/model via transformers
and a piano roll CNN via torch. Both work under transformers 4.x. The ONNX inference
path (`onnxruntime`) is unaffected by this version split.

## Goals / Non-Goals

- **Goals**: one venv, one `python` command, clean test suite, no subprocess re-runners
- **Non-Goals**: upgrading to transformers 5.x, changing torch version, touching training
  scripts (Modal runs those in its own image)

## Decisions

### Pin transformers to `>=4.47,<5`

4.47 is the minimum that supports numpy 2.x without deprecation warnings. The upper
bound `<5` blocks uv from resolving to the breaking release. When transformers 5.x
stabilises its `torch.load()` behaviour, this bound can be removed.

### Add torch and sentencepiece to pyproject.toml main dependencies

`sentencepiece` is required by the DeBERTa v2 tokenizer (the fast tokenizer falls back
to a broken tiktoken path when sentencepiece is absent). `torch>=2.2.0,<2.3.0` is
required because `prepare_concept()` loads DeBERTa as a PyTorch model and all 4.x+ chord
pipelines call it. On Intel Mac only 2.2.x wheels are available; a `tolist()` bridge in
`_encode_text()` and `_encode_audio()` avoids the torch 2.2.x / numpy 2.x binary
incompatibility (`tensor.numpy()` errors but `np.array(tensor.tolist())` works).

### Rebuild strategy: delete and recreate

Rather than mutating `.venv` in place, the cleanest path is:
1. `rm -rf .venv`
2. `uv sync` — resolves fresh with the new transformers pin
3. Verify Refractor can import and score

This avoids stale package state from the previous incompatible resolution.

### Verification gate before deleting .venv312

Before deleting `.venv312`, verify the consolidated `.venv` passes:
- `python -m pytest tests/` — full suite
- `python -c "from training.refractor import Refractor; r = Refractor(); print('ok')`
- One live `score()` call against a known MIDI file (smoke test)

Only delete `.venv312` after this gate passes.

## Risks / Trade-offs

- **transformers 4.x pin blocks future upgrade** → mitigation: pin is documented and
  removal is a one-line change when 5.x fixes its compatibility
- **numpy 2.x compat in transformers 4.47+** → if edge cases appear, bump floor to 4.51
- **`uv sync` may introduce other version changes** → review lock diff before committing

## Open Questions

- Does `sentence-transformers>=3.0.0` (optional training dep) also need a transformers
  bound? Check after `uv sync` resolves.
