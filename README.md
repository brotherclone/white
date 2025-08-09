# Unnamed White Album, the final LP in The Rainbow Table by The Earthly Frames

This repository contains the Rainbow Song Processing System, a tool designed to process and analyze music files for machine learning applications.
A system for processing music files (audio, MIDI, and lyrics) to create training samples for machine learning models.

## Overview

The Rainbow Song Processing System extracts features from music files and prepares them for use in machine learning models. It processes raw material files (audio, MIDI, lyrics) according to song metadata specified in YAML files, segments them based on song structure, and generates training samples.

## Features

- Extract audio segments from WAV files
- Process MIDI data for note-level analysis
- Extract lyrics from LRC files synchronized with audio
- Segment songs based on structure (verse, chorus, etc.)
- Generate training samples with comprehensive metadata
- Vector-based search through training samples
- Agent-based system for specialized processing tasks

## Installation

1. Clone this repository
2. Install the dependencies:

```bash
pip install -r requirements.txt
```

3. Prepare your raw materials in the `staged_raw_material` directory

## Usage

### Basic Usage

Process all songs in the default directory:

```bash
python main.py
```

### Advanced Options

```bash
python main.py --dir /path/to/materials --keep-working-dir --verbose
```

Options:
- `--dir`: Specify an alternative raw materials directory
- `--keep-working-dir`: Do not clear the working directory between songs
- `--verbose`: Enable verbose logging

### Directory Structure

Your raw materials should be organized as follows:

```
staged_raw_material/
  song_01/
    song.yml
    song.wav
    vocals.wav
    drums.mid
    song.lrc
  song_02/
    ...
```

### YAML Configuration

Each song should have a YAML file with metadata:

```yaml
title: Song Title
artist: Artist Name
album_title: Album Title
release_date: YYYY-MM-DD
bpm: 120
key: C major
structure:
  - section_name: Intro
    start_time: 00:00
    end_time: 00:15
    section_description: Instrumental introduction
  - section_name: Verse 1
    start_time: 00:15
    end_time: 00:45
    section_description: First verse with vocals
# ... more sections
```

## Agents

The system includes several specialized agents:

- **BaseRainbowAgent**: Foundation class for all agents
- **Andy**: Audio analysis agent
- **Dorothy**: Data organization agent
- **Martin**: MIDI processing agent
- **Nancarrow**: Note analysis agent
- **Subutai**: Song structure agent

## Development

### Running Tests

```bash
python -m unittest discover tests
```

### Migration Commands

```bash
python -m app.cli.migration_commands migrate up
```

## License

[Specify your license here]

## Contributing

[Contribution guidelines]
