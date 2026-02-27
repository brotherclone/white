# Design: Sounds-Like Training Feature

## Architecture Overview

```
                 DeBERTa (frozen, pre-computed)
                        ↓
  artist_catalog.yml ──► build_sounds_like_embeddings.py
                        │   • parse sounds_like string per song
                        │   • look up descriptions in catalog
                        │   • embed each description (768-dim)
                        │   • mean-pool across artists per song
                        │   • broadcast to all segments of that song
                        ↓
       sounds_like_embeddings.parquet
       (segment_id, sounds_like_emb[768], has_sounds_like)
                        ↓
     modal_midi_fusion.py ──► MultimodalFusionModel
                        │
    [audio 512 + midi 512 + concept 768 + lyric 768 + sounds_like 768]
                        = 3328-dim fusion input
                        ↓
              temporal[3] / spatial[3] / ontological[3] / confidence[1]
```

## Embedding Strategy

### Why mean-pool across artists?

A song's `sounds_like` typically lists 2–5 artists. Each has a 768-dim description
embedding. Simple mean-pooling is:
- Order-invariant (matching the unordered nature of style references)
- Well-established for set-based embeddings
- Prevents over-indexing on a single artist when several are listed

The mean is taken over the subset of artists that have `status: reviewed` entries
first; if none are reviewed, `draft` entries are used as fallback (matching the
pipeline injection policy in `add-artist-style-catalog`).

### Why 768-dim (not compressed)?

- Matches `lyric_emb` and `concept_emb` dimensions — the fusion MLP already
  handles 768-dim inputs; no extra projection layer needed
- DeBERTa's natural output dimension — no loss from compression
- The fusion MLP's first layer (3328→1024) already performs the projection

### Null embedding for missing artists

When a song has no catalog entries for any of its `sounds_like` artists:
- `has_sounds_like = False`
- The learned `null_sounds_like` parameter is used (same as `null_lyric`)
- This is the expected state during early training before catalog is populated

## Model Architecture Change

```python
# Before (Phase 3)
fusion input = [audio(512) + midi(512) + concept(768) + lyric(768)] = 2560

# After (Phase 5)
fusion input = [audio(512) + midi(512) + concept(768) + lyric(768) + sounds_like(768)] = 3328
```

The `MultimodalFusionModel.__init__` adds:
```python
self.null_sounds_like = nn.Parameter(torch.randn(768) * 0.02)
```
And `self.fusion` first layer becomes `nn.Linear(3328, 1024)` (was 2560).

The `forward()` signature gains two new parameters:
```python
sounds_like_emb,   # [batch, 768]
has_sounds_like,   # [batch] bool
```

## Modality Dropout

Same dropout rate (0.15) applied to sounds_like during training. This is important
because at inference time, the catalog will be partially filled — the model must
generalise to the null-embedding case.

The existing training dataset has `has_sounds_like: True` for only the fraction of
songs where the catalog covers all (or some) `sounds_like` artists. We expect
roughly 60–80% coverage after a first pass of catalog generation + review, meaning
20–40% of segments naturally train on the null path even without forced dropout.

## Parquet Schema

`training/data/sounds_like_embeddings.parquet`:

| column | dtype | notes |
|--------|-------|-------|
| `segment_id` | str | join key with DeBERTa parquet |
| `song_slug` | str | for debugging / groupby |
| `sounds_like_raw` | str | original comma-separated string |
| `artists_found` | int | count of artists with catalog entries |
| `artists_total` | int | count of artists in sounds_like string |
| `has_sounds_like` | bool | True when artists_found > 0 |
| `sounds_like_emb` | list[float32] (768) | mean-pooled embedding |

## ChromaticScorer Integration

`chromatic_scorer.py` already works without sounds_like (null path). Adding
optional injection allows the lyric pipeline and artist catalog CLI to score
descriptions with style context:

```python
# Option A: pass pre-computed embedding
scorer.score(midi_bytes, sounds_like_emb=emb_768)

# Option B: pass artist description texts (scorer embeds them)
scorer.score(midi_bytes, sounds_like_texts=["Description of Artist A", ...])

# Option C: no sounds_like (null path, current behaviour)
scorer.score(midi_bytes)
```

For the generative pipelines (lyric, chord), Option C is fine for now — the catalog
injection benefit is at training time, not inference. Option B is useful when
re-scoring with `--rescore-lyrics` and the catalog is available.

## Training Strategy

- Phase 5 is a **continuation from Phase 3 weights** (not from scratch):
  - Load `fusion_model.pt` as initialisation
  - The `null_sounds_like` parameter is freshly initialised (the rest of the
    weights are preserved)
  - The fusion MLP first layer is re-initialised (input dim changed: 2560→3328)
    — this is unavoidable; the other fusion layers retain weights
  - Fine-tune all parameters at reduced LR (1e-5 rather than 1e-4) for 30 epochs
- Alternative: train from scratch (50 epochs, full LR) — simpler but slower
- Recommendation: fine-tune from Phase 3 to reduce GPU cost and leverage the
  already-learned chromatic feature space

## ONNX Export

New dummy inputs:
```python
sounds_like_emb = torch.randn(1, 768)
has_sounds_like = torch.ones(1, dtype=torch.bool)
```
Dynamic axes: `sounds_like_emb` with `{0: "batch"}`.

## Risks

| Risk | Mitigation |
|------|-----------|
| Catalog sparsity at training time | null-embedding path handles it; modality dropout ensures robustness |
| Sounds-like descriptions encode biographical bias | Prompt in artist_catalog.py explicitly excludes biography; descriptions are aesthetic-only |
| 5th modality increases overfitting | Dataset is 11,605 segments; adding 768→768 routing is modest; existing dropout unchanged |
| ONNX model size grows | First fusion layer weight grows by 768×1024 params ≈ 786K extra — from ~4.3M to ~5.1M |
