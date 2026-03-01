## 0. Prerequisite

- [ ] 0.1 `add-artist-style-catalog` must be applied and catalog populated with at
       least a few reviewed entries before training can show a benefit; however all
       code in this change tolerates an empty catalog (null path)

## 1. Embedding Build Script — build_sounds_like_embeddings.py

- [ ] 1.1 Create `training/build_sounds_like_embeddings.py`
- [ ] 1.2 Implement `parse_sounds_like(raw: str) → list[str]` — strips
       `, discogs_id: \d+` fragments; returns clean artist name list
- [ ] 1.3 Implement `load_catalog(catalog_path) → dict[str, dict]` — reads
       `app/data/artist_catalog.yml`; returns `{artist_name: entry}` preferring
       `reviewed` entries over `draft`
- [ ] 1.4 Implement `embed_descriptions(artist_names, catalog, deberta_tokenizer,
       deberta_model) → tuple[np.ndarray, int, int]` — embed found descriptions,
       mean-pool, return (vector_768, found_count, total_count)
- [ ] 1.5 Implement `build_sounds_like_parquet(training_parquet_path, catalog_path,
       output_path)` — main pipeline: per-song groupby, embed, broadcast to segments,
       write parquet
- [ ] 1.6 Print coverage summary on completion
- [ ] 1.7 CLI: `--training-parquet`, `--catalog` (default `app/data/artist_catalog.yml`),
       `--output` (default `training/data/sounds_like_embeddings.parquet`)

## 2. Model Update — multimodal_fusion.py

- [ ] 2.1 Add `null_sounds_like = nn.Parameter(torch.randn(768) * 0.02)` to
       `MultimodalFusionModel.__init__`
- [ ] 2.2 Change fusion MLP first layer from `nn.Linear(2560, 1024)` to
       `nn.Linear(3328, 1024)`
- [ ] 2.3 Add `sounds_like_emb`, `has_sounds_like` parameters to `forward()`
- [ ] 2.4 Apply modality mask + dropout for sounds_like (same pattern as lyric/audio)
- [ ] 2.5 Add `sounds_like_emb` to the `torch.cat(...)` call
- [ ] 2.6 Update docstring: input dim 2560→3328, list 5th modality

## 3. Modal Training Script — modal_midi_fusion.py

- [ ] 3.1 Add loading of `sounds_like_embeddings.parquet` from Modal volume (with
       graceful fallback if absent)
- [ ] 3.2 Join sounds_like parquet to main dataset on `segment_id`
- [ ] 3.3 Pass `sounds_like_emb` + `has_sounds_like` from dataset batch to model
- [ ] 3.4 Add `--finetune-from <pt_path>` flag: load matching layer weights,
       re-init `null_sounds_like` + first fusion layer
- [ ] 3.5 Update `cpu_image` to include DeBERTa dependencies if pre-embedding sounds_like
       on CPU before upload (TBD — may keep embeddings pre-computed locally)

## 4. ONNX Export — export_onnx.py

- [ ] 4.1 Add `sounds_like_emb = torch.randn(1, 768)` and
       `has_sounds_like = torch.ones(1, dtype=torch.bool)` to dummy inputs
- [ ] 4.2 Add `sounds_like_emb` with `{0: "batch"}` dynamic axis to ONNX export call
- [ ] 4.3 Verify exported ONNX runs via `onnxruntime` with and without sounds_like

## 5. Refractor — optional sounds_like injection

- [ ] 5.1 Add `sounds_like_texts: list[str] | None = None` and
       `sounds_like_emb: np.ndarray | None = None` parameters to `score()`
- [ ] 5.2 When `sounds_like_texts` provided: embed via DeBERTa, mean-pool,
       set `has_sounds_like=True`
- [ ] 5.3 When `sounds_like_emb` provided directly: use as-is, `has_sounds_like=True`
- [ ] 5.4 Otherwise: zero vector, `has_sounds_like=False`
- [ ] 5.5 Pass through ONNX session as new named inputs

## 6. Tests

- [ ] 6.1 `test_parse_sounds_like_strips_discogs_id`
- [ ] 6.2 `test_parse_sounds_like_multiple_artists`
- [ ] 6.3 `test_embed_descriptions_mean_pools`
- [ ] 6.4 `test_embed_descriptions_no_catalog_match` — returns zeros + has_sounds_like=False
- [ ] 6.5 `test_embed_descriptions_partial_match`
- [ ] 6.6 `test_build_parquet_row_count_matches_training_data`
- [ ] 6.7 `test_fusion_model_forward_5th_modality` — shape check with has_sounds_like=True
- [ ] 6.8 `test_fusion_model_forward_null_path` — has_sounds_like=False uses null param
- [ ] 6.9 `test_fusion_model_input_dim_is_3328`
- [ ] 6.10 `test_scorer_sounds_like_texts_path` — mock DeBERTa, verify tensor shape
- [ ] 6.11 `test_scorer_backward_compat_no_sounds_like` — existing call signature unchanged

## 7. Retrain + Validate

- [ ] 7.1 Upload `sounds_like_embeddings.parquet` to Modal volume
- [ ] 7.2 Run `modal run training/modal_midi_fusion.py --skip-preprocess
         --finetune-from refractor.pt --epochs 30 --lr 1e-5` (Phase 5)
- [ ] 7.3 Export ONNX: `modal run training/export_onnx.py`
- [ ] 7.4 Copy `refractor.pt` + `refractor.onnx` to `training/data/`
- [ ] 7.5 Run `python -m pytest tests/training/` to verify scorer still passes
- [ ] 7.6 Compare Phase 5 vs Phase 3 accuracy metrics; update MEMORY.md
