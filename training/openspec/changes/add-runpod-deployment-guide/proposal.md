# Change: Add RunPod Deployment Guide

## Why

The training pipeline runs on RunPod GPU instances, but the network-drive-to-GPU workflow has undocumented gotchas that cause wasted time and money: region lock-in limits GPU selection, large file re-uploads take hours, and the manifest DB / extraction scripts must run in a specific order. A concrete playbook prevents repeating these mistakes.

## What Changes

- New deployment guide covering the network volume + GPU pod pattern
- Pre-flight checklist script that validates local state before uploading
- Step-by-step instructions for the full cycle: local prep → upload → GPU run → download results

## Impact

- No code changes to existing training pipeline
- New documentation and optional pre-flight script
- Affected workflows: any RunPod training run
