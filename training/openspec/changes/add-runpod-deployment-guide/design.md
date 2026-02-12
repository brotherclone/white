## Context

Training runs on RunPod GPU instances with data stored on RunPod network volumes. The current workflow has several undocumented pain points around region selection, file transfers, and script execution order.

## Goals / Non-Goals

**Goals**:
- Prevent wasted GPU time from avoidable setup mistakes
- Provide a repeatable playbook for the local → RunPod → local cycle
- Document all known gotchas in one place

**Non-Goals**:
- Automating the full deployment (that's infrastructure-as-code, out of scope)
- CI/CD pipeline for training (overkill for current scale)

## Decisions

### D1: Region Recommendation

**Decision**: Recommend **US-KS-2** or **US-CA-2** over US-MD-1 for new network volumes.

**Why**:
- Both support the **S3-compatible API** for direct file uploads (no need to spin up a CPU pod as a gateway)
- Both are larger hubs with better GPU diversity and inventory
- US-MD-1 has **no S3 API** access, meaning uploads require an active pod ($)

**If already committed to US-MD-1**: Use a CPU pod ($0.09/hr) as an upload gateway. Keep it running only long enough to transfer data, then terminate.

**Region constraints**:
- Network volumes are **permanently region-locked** — cannot attach a volume from one region to a pod in another
- Moving data between regions requires spinning up pods in both regions and using `runpodctl` or `rsync`
- GPU availability is dynamic — check the RunPod console at pod creation time

### D2: File Transfer Strategy

**Decision**: Archive small files, upload parquets directly.

**Why**:
- RunPod network volumes have a **64KB minimum allocation per file** — thousands of small audio segments waste storage
- Parquet files are already efficient (columnar, compressed)
- For initial upload: use S3 API if available, otherwise CPU pod + `runpodctl send`

**Upload checklist** (what to transfer):
```
Must upload:
  training/data/training_segments_metadata.parquet  (~small, metadata only)
  training/data/training_segments_media.parquet      (~large, has audio/MIDI binary)
  training/data/base_manifest_db.parquet             (~small, manifest join table)
  training/models/                                   (model code)
  training/notebooks/                                (Jupyter notebooks)
  app/ (extraction code if re-running on RunPod)

Do NOT upload:
  staged_raw_material/  (huge, raw WAV files — only needed if re-extracting)
  chain_artifacts/      (not relevant to training)
  .git/                 (large, not needed on RunPod)
```

### D3: Execution Order

**Decision**: Enforce a strict 4-step sequence with verification between each step.

```
Step 1: python -m app.extractors.manifest_extractor.build_base_manifest_db
  → Verify: base_manifest_db.parquet has 1327 rows, all 8 colors present

Step 2: python -m app.extractors.segment_extractor.build_training_segments_db
  → Verify: training_segments_metadata.parquet has >10K rows, Green segments present

Step 3: DeBERTa embedding pass
  → Verify: text embeddings column populated for vocal segments

Step 4: Run training notebooks in order (Phase 1 → 2 → 4 → 3)
  → Verify: W&B metrics, checkpoint files
```

**Why steps 1-2 must run before upload (if possible)**: Re-running extraction is the longest step (~6 hours). If extraction must happen on RunPod (because raw audio lives there), ensure steps 1-2 run first.

## Risks / Trade-offs

| Risk | Impact | Mitigation |
|------|--------|------------|
| Region has no GPUs available | Stranded network volume, can't train | Pick high-traffic region; check availability before creating volume |
| Network volume runs out of space | Training fails mid-run | Monitor usage; archive small files; provision 2x expected size |
| RunPod account runs out of funds | Volume deleted, data lost | Keep local backup of all training data and results |
| S3 API not available in chosen region | Must use CPU pod for uploads (extra cost) | Choose US-KS-2 or US-CA-2 which have S3 API |
