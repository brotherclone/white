# Project Context

## Purpose
Train multi-modal models on The Rainbow Table albums by The Earthly Frames for use in creation of the final White album.

## Tech Stack
- Python
- PyTorch
- TensorFlow
- Hugging Face Transformers
- Weights & Biases
- Runpod

## Project Conventions

### Code Style
Python PEP 8 standards with type hints

### Architecture Patterns
This module is residing in a much larget project, so installing anything outside of the /training directory is not recommended.

### Testing Strategy
Unit tests are written using pytest.

### Git Workflow
Uses gitflow but at the main project level.

## Domain Context
See /Volumes/LucidNonsense/White/claude_working_area/white_album_project_diary.md and /Volumes/LucidNonsense/White/white_album_project_diary.md

## Important Constraints
- Python version determined by Runpod VM

## External Dependencies
- Runpod for cloud GPU training
- Weights & Biases for experiment tracking
- Hugging Face for model hosting
- Data in parquet format
