#!/usr/bin/env python3
"""
Rainbow Song Training Sample Generator

This script processes raw music materials in the staged_raw_material directory,
extracts audio, MIDI, and lyrics data, and creates training samples for machine learning.

Usage:
    python main.py [--dir DIRECTORY] [--keep-working-dir]

Options:
    --dir DIRECTORY          Specify an alternative raw materials directory
    --keep-working-dir       Do not clear the working directory between songs
    --verbose                Enable verbose logging
"""

import os
import sys
import shutil
import logging
import argparse
import time
from typing import Optional
from pathlib import Path
from tqdm import tqdm

import app.objects.rainbow_song_meta
import app.objects.rainbow_song

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("rainbow_processor.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def clear_working_directory() -> None:
    """
    Clear and recreate the working directory for temporary files.

    Returns:
        None
    """
    working_dir = os.path.join(os.path.dirname(__file__), "working")
    if os.path.exists(working_dir):
        try:
            shutil.rmtree(working_dir)
            logger.info(f"Removed contents of {working_dir}")
        except (OSError, shutil.Error) as e:
            logger.error(f"Failed to clear working directory: {e}")
            return

    try:
        os.makedirs(working_dir, exist_ok=True)
        logger.info(f"Created fresh working directory at {working_dir}")
    except OSError as e:
        logger.error(f"Failed to create working directory: {e}")


def process_yaml_file(yaml_file_path: str, base_path: str, track_materials_path: str) -> bool:
    """
    Process a single YAML file to create training samples.

    Args:
        yaml_file_path: Path to the YAML file
        base_path: Base path for the song materials
        track_materials_path: Path to the track materials

    Returns:
        bool: True if processing was successful, False otherwise
    """
    try:
        file_name = os.path.basename(yaml_file_path)
        logger.info(f"Processing YAML file: {yaml_file_path}")

        # Create metadata object
        meta = app.objects.rainbow_song_meta.RainbowSongMeta(
            yaml_file_name=file_name,
            base_path=base_path,
            track_materials_path=track_materials_path
        )

        # Create song object and generate training samples
        song = app.objects.rainbow_song.RainbowSong(meta_data=meta)
        song.create_training_samples()

        logger.info(f"Successfully processed {yaml_file_path}")
        return True

    except Exception as e:
        logger.error(f"Error processing {yaml_file_path}: {e}", exc_info=True)
        return False


def find_yaml_files(directory: str) -> list:
    """
    Recursively find all YAML files in the given directory and its subdirectories.

    Args:
        directory: Directory path to search

    Returns:
        list: List of (yaml_path, base_path, subdir) tuples
    """
    yaml_files = []

    if not os.path.isdir(directory):
        logger.error(f"Directory not found: {directory}")
        return yaml_files

    for subdir in os.listdir(directory):
        subdir_path = os.path.join(directory, subdir)

        if not os.path.isdir(subdir_path):
            continue

        logger.debug(f"Scanning directory: {subdir_path}")
        for file in os.listdir(subdir_path):
            if file.endswith((".yml", ".yaml")):
                yaml_path = os.path.join(subdir_path, file)
                base_path = os.path.dirname(subdir_path)
                yaml_files.append((yaml_path, base_path, subdir))

    return yaml_files


def main() -> int:
    """
    Main function that processes raw materials and creates training samples.

    Returns:
        int: Exit code (0 for success, non-zero for errors)
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Process raw music materials and create training samples.")
    parser.add_argument("--dir", type=str, help="Specify an alternative raw materials directory")
    parser.add_argument("--keep-working-dir", action="store_true",
                        help="Do not clear the working directory between songs")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()

    # Configure logging level
    if args.verbose:
        logger.setLevel(logging.DEBUG)
        logger.debug("Verbose logging enabled")

    # Determine the raw materials path
    raw_materials_path = args.dir if args.dir else os.path.join(os.path.dirname(__file__), "staged_raw_material")

    # Validate raw materials path
    if not os.path.isdir(raw_materials_path):
        logger.error(f"The directory {raw_materials_path} does not exist. Please ensure the raw materials are staged correctly.")
        return 1

    # Find all YAML files to process
    logger.info(f"Scanning for YAML files in {raw_materials_path}")
    yaml_files = find_yaml_files(raw_materials_path)

    if not yaml_files:
        logger.warning(f"No YAML files found in {raw_materials_path}")
        return 0

    logger.info(f"Found {len(yaml_files)} YAML files to process")

    # Process each YAML file
    successful = 0
    failed = 0

    # Create initial working directory
    clear_working_directory()

    for i, (yaml_path, base_path, subdir) in enumerate(tqdm(yaml_files, desc="Processing songs")):
        logger.info(f"Processing file {i+1}/{len(yaml_files)}: {os.path.basename(yaml_path)}")

        # Process the file
        if process_yaml_file(yaml_path, base_path, subdir):
            successful += 1
        else:
            failed += 1

        # Clear working directory between files unless --keep-working-dir is specified
        if not args.keep_working_dir:
            clear_working_directory()

    # Report results
    logger.info(f"Processing complete. Successfully processed {successful} files, {failed} failures.")

    if failed > 0:
        logger.warning(f"Some files failed to process. Check the log for details.")
        return 1

    return 0


if __name__ == "__main__":
    start_time = time.time()
    exit_code = main()
    elapsed_time = time.time() - start_time
    logger.info(f"Total processing time: {elapsed_time:.2f} seconds")
    sys.exit(exit_code)
