## ADDED Requirements

### Requirement: MultimodalFusionModel — 5th Modality

The `MultimodalFusionModel` SHALL accept a 5th input modality (`sounds_like_emb`)
following the same null-embedding + modality-dropout pattern as existing modalities.

#### Scenario: Model accepts sounds_like modality

- **WHEN** `MultimodalFusionModel` is instantiated
- **THEN** it SHALL contain a `null_sounds_like` learned parameter (768-dim)
- **AND** the fusion MLP's first linear layer SHALL have input dimension 3328
  (audio 512 + midi 512 + concept 768 + lyric 768 + sounds_like 768)
- **AND** all other layer dimensions SHALL remain unchanged

#### Scenario: forward() with sounds_like present

- **WHEN** `model.forward(..., sounds_like_emb, has_sounds_like)` is called
- **AND** `has_sounds_like` is `True` for a batch item
- **THEN** the provided `sounds_like_emb` is concatenated into the fusion input
- **AND** output shapes remain: temporal[3], spatial[3], ontological[3], confidence[1]

#### Scenario: forward() with sounds_like absent

- **WHEN** `has_sounds_like` is `False` for a batch item
- **THEN** `null_sounds_like` is substituted for that item's embedding
- **AND** the model produces valid outputs identical in shape to the present case

#### Scenario: Modality dropout during training

- **WHEN** `model.training` is True and `modality_dropout > 0`
- **THEN** a random fraction of `has_sounds_like=True` items have their
  `sounds_like_emb` replaced with `null_sounds_like` at the same dropout rate
  as other modalities (default 0.15)

---

### Requirement: Modal Training Script — 5th Modality

`training/modal_midi_fusion.py` SHALL load the sounds-like parquet from the
Modal volume and pass it to the model as the 5th modality.

#### Scenario: Training with sounds_like available

- **WHEN** `modal run training/modal_midi_fusion.py` is executed
- **AND** `sounds_like_embeddings.parquet` is present in the Modal volume
- **THEN** the script loads the parquet and joins on `segment_id`
- **AND** passes `sounds_like_emb` (float32[768]) and `has_sounds_like` (bool) per sample

#### Scenario: Training without sounds_like parquet

- **WHEN** `sounds_like_embeddings.parquet` is absent from the Modal volume
- **THEN** the script prints a warning and trains with all
  `has_sounds_like: False` (null path)
- **AND** training does not abort

#### Scenario: Phase 5 fine-tune from Phase 3 weights

- **WHEN** `--finetune-from refractor.pt` is passed
- **THEN** the script loads Phase 3 weights for all layers that exist in both models
- **AND** the `null_sounds_like` parameter and the resized first fusion layer are
  freshly initialised
- **AND** training proceeds at the specified LR (recommended 1e-5 for fine-tuning)

---

### Requirement: ONNX Export — 5th Modality

`training/export_onnx.py` SHALL include the `sounds_like_emb` and `has_sounds_like`
tensors as named inputs in the ONNX graph.

#### Scenario: ONNX export includes sounds_like

- **WHEN** `training/export_onnx.py` is run against the Phase 5 model
- **THEN** the exported ONNX graph SHALL have input names including
  `sounds_like_emb` and `has_sounds_like`
- **AND** dynamic axes SHALL mark the batch dimension for `sounds_like_emb`

---

### Requirement: Refractor — Optional Sounds-Like Injection

`training/refractor.py` SHALL optionally accept sounds-like context for
inference, while remaining backward-compatible when called without it.

#### Scenario: Score with sounds_like texts

- **WHEN** `scorer.score(midi_bytes, sounds_like_texts=["description A", "description B"])`
  is called
- **THEN** each description string is embedded via DeBERTa
- **AND** mean-pooled into a 768-dim vector
- **AND** passed as `sounds_like_emb` with `has_sounds_like=True`

#### Scenario: Score without sounds_like (backward compatibility)

- **WHEN** `scorer.score(midi_bytes)` is called without sounds_like arguments
- **THEN** `null_sounds_like` is used (has_sounds_like=False)
- **AND** output format is identical to the current implementation
- **AND** no error is raised

#### Scenario: Null path when all descriptions are empty

- **WHEN** `sounds_like_texts` is provided but all strings are empty or None
- **THEN** `has_sounds_like=False` is used
- **AND** a warning is printed
