# Implementation Tasks

## 1. Deployment Guide Document
- [ ] 1.1 Write region selection guidance (GPU availability varies by region, network volumes are region-locked)
- [ ] 1.2 Document the US-MD-1 limitations (no S3 API, narrower GPU selection)
- [ ] 1.3 Recommend alternative regions (US-KS-2, US-CA-2 have S3 API for direct uploads)
- [ ] 1.4 Document the network volume + GPU pod lifecycle pattern
- [ ] 1.5 Document file upload strategies (CPU pod gateway vs S3 API vs runpodctl)
- [ ] 1.6 Document the 64KB minimum allocation gotcha for small files
- [ ] 1.7 Add cost comparison table (network volume vs container disk vs stopped pod)

## 2. Pre-Upload Checklist
- [ ] 2.1 Verify `base_manifest_db.parquet` is current (includes all 8 album prefixes 01-08)
- [ ] 2.2 Verify extraction pipeline code changes are committed (structure fallback, MIDI fixes)
- [ ] 2.3 Verify parquet files exist and have expected row counts
- [ ] 2.4 List files to upload with sizes (avoid uploading unnecessary files)
- [ ] 2.5 Document which notebooks need re-running and in what order

## 3. RunPod Execution Order
- [ ] 3.1 Document the correct script execution order:
  1. `build_base_manifest_db.py` (rebuild manifest DB with all albums)
  2. `build_training_segments_db.py` (extract segments with structure fallback)
  3. DeBERTa embedding pass (re-embed new extraction)
  4. Training notebooks in order
- [ ] 3.2 Document environment variable setup on RunPod
- [ ] 3.3 Document how to verify each step succeeded before proceeding
- [ ] 3.4 Document downloading results back to local

## 4. Pre-Flight Validation Script (Optional)
- [ ] 4.1 Write script that checks local state before upload (manifest DB freshness, parquet existence, git status)
- [ ] 4.2 Print summary of what will be uploaded and estimated sizes
