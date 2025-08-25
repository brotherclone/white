import yaml
import re
import sys
import os
import glob
from typing import Dict, List, Tuple, Any


def validate_timestamp_format(yaml_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validates that all timestamps in the structure section follow the [MM:SS.mmm] format.

    Args:
        yaml_data: Parsed YAML content

    Returns:
        Tuple of (is_valid, list_of_error_messages)
    """
    time_stamp_errors = []
    timestamp_pattern = re.compile(r"^\[\d{2}:\d{2}\.\d{3}\]$")

    if not isinstance(yaml_data, dict) or "structure" not in yaml_data:
        return True, time_stamp_errors

    if not isinstance(yaml_data["structure"], list):
        time_stamp_errors.append("Structure section is not a list")
        return False, time_stamp_errors

    for i, section in enumerate(yaml_data["structure"]):
        if not isinstance(section, dict):
            time_stamp_errors.append(f"Structure section {i + 1} is not a dictionary")
            continue

        for key in ["start_time", "end_time"]:
            if key in section:
                timestamp = section[key]
                if not timestamp_pattern.match(timestamp):
                    time_stamp_errors.append(f"Invalid timestamp format in section {i + 1} ({key}): {timestamp}")

    return len(time_stamp_errors) == 0, time_stamp_errors


def validate_lyrics_has_lrc(yaml_data: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Validates that if lyrics is true, there's a corresponding lrc_file.

    Args:
        yaml_data: Parsed YAML content

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isinstance(yaml_data, dict):
        return False, "Invalid YAML structure: not a dictionary"

    # Check if lyrics field exists and is true
    if yaml_data.get("lyrics"):
        # Check if lrc_file exists and is not empty
        if "lrc_file" not in yaml_data or not yaml_data["lrc_file"]:
            return False, "Lyrics are marked as true, but no lrc_file is specified"

    return True, ""


def validate_directory(directory_path: str) -> Tuple[bool, List[str]]:
    """
    Validates all YAML files in a directory.

    Args:
        directory_path: Path to directory containing YAML files

    Returns:
        Tuple of (all_valid, list_of_error_messages)
    """
    all_errors = []
    all_valid = True

    # Find all YAML files in the directory
    yaml_files = glob.glob(os.path.join(directory_path, "**/*.yml"), recursive=True)

    for yaml_file in yaml_files:
        is_valid, errors = validate_yaml_file(yaml_file)
        if not is_valid:
            all_valid = False
            all_errors.extend(errors)

    return all_valid, all_errors


def extract_file_references(yaml_data: Dict[str, Any]) -> Dict[str, List[str]]:
    """
    Extract all file references from the YAML data.

    Args:
        yaml_data: Parsed YAML content

    Returns:
        Dictionary with file types as keys and lists of file paths as values
    """
    if not isinstance(yaml_data, dict):
        return {}

    file_refs = {
        "lrc": [],
        "wav": [],
        "mid": []
    }

    # Check for LRC file
    if "lrc_file" in yaml_data and yaml_data["lrc_file"]:
        file_refs["lrc"].append(yaml_data["lrc_file"])

    # Check for main audio file
    if "main_audio_file" in yaml_data and yaml_data["main_audio_file"]:
        file_refs["wav"].append(yaml_data["main_audio_file"])

    # Check for audio tracks
    if "audio_tracks" in yaml_data and isinstance(yaml_data["audio_tracks"], list):
        for track in yaml_data["audio_tracks"]:
            if not isinstance(track, dict):
                continue

            if "audio_file" in track and track["audio_file"]:
                file_refs["wav"].append(track["audio_file"])

            if "midi_file" in track and track["midi_file"]:
                file_refs["mid"].append(track["midi_file"])

            if "midi_group_file" in track and track["midi_group_file"]:
                file_refs["mid"].append(track["midi_group_file"])

    return file_refs


def validate_file_existence(yaml_data: Dict[str, Any], yaml_dir: str) -> Tuple[bool, List[str]]:
    """
    Validates that all referenced files exist on disk.

    Args:
        yaml_data: Parsed YAML content
        yaml_dir: Directory containing the YAML file

    Returns:
        Tuple of (is_valid, list_of_error_messages)
    """
    file_reference_errors = []

    # Get all file references
    file_refs = extract_file_references(yaml_data)

    # Check each file type
    for file_type, files in file_refs.items():
        for file_path in files:
            full_path = os.path.join(yaml_dir, file_path)
            if not os.path.exists(full_path):
                file_reference_errors.append(f"Referenced file does not exist: {file_path}")

    return len(file_reference_errors) == 0, file_reference_errors


def timestamp_to_ms(timestamp: str) -> int:
    """
    Convert a timestamp string in format [MM:SS.mmm] to milliseconds.

    Args:
        timestamp: Time in format [MM:SS.mmm]

    Returns:
        Time in milliseconds
    """
    # Remove brackets
    timestamp = timestamp.strip('[]')

    # Split by :
    minutes_str, seconds_str = timestamp.split(':')

    # Convert to numbers
    minutes = int(minutes_str)
    seconds = float(seconds_str)

    # Convert to milliseconds
    return int((minutes * 60 + seconds) * 1000)


def validate_required_properties(yaml_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validates that all required properties are present in the YAML.

    Args:
        yaml_data: Parsed YAML content

    Returns:
        Tuple of (is_valid, list_of_error_messages)
    """
    errors = []

    # Define required top-level properties
    required_properties = [
        "bpm", "manifest_id", "tempo", "key", "rainbow_color", "title",
        "release_date", "album_sequence", "main_audio_file", "TRT",
        "vocals", "lyrics", "structure", "mood", "sounds_like",
        "genres", "concept", "audio_tracks"
    ]

    # Check for required properties
    for prop in required_properties:
        if prop not in yaml_data:
            errors.append(f"Missing required property: {prop}")

    # Validate structure if present
    if "structure" in yaml_data and isinstance(yaml_data["structure"], list):
        for i, section in enumerate(yaml_data["structure"]):
            if not isinstance(section, dict):
                errors.append(f"Structure section {i + 1} is not a dictionary")
                continue

            # Check required section properties
            section_props = ["section_name", "start_time", "end_time", "description"]
            for prop in section_props:
                if prop not in section:
                    errors.append(f"Structure section {i + 1} missing required property: {prop}")

    # Validate audio_tracks if present
    if "audio_tracks" in yaml_data and isinstance(yaml_data["audio_tracks"], list):
        for i, track in enumerate(yaml_data["audio_tracks"]):
            if not isinstance(track, dict):
                errors.append(f"Audio track {i + 1} is not a dictionary")
                continue

            # Check required track properties
            if "id" not in track:
                errors.append(f"Audio track {i + 1} missing required property: id")
            if "description" not in track:
                errors.append(f"Audio track {i + 1} missing required property: description")
            if "audio_file" not in track:
                errors.append(f"Audio track {i + 1} missing required property: audio_file")

    return len(errors) == 0, errors


def validate_structure_timestamps(yaml_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validates that the timestamps in the structure section don't overlap.

    Args:
        yaml_data: Parsed YAML content

    Returns:
        Tuple of (is_valid, list_of_error_messages)
    """
    timestamp_structure_errors = []

    if not isinstance(yaml_data, dict) or "structure" not in yaml_data:
        return True, timestamp_structure_errors

    if not isinstance(yaml_data["structure"], list):
        timestamp_structure_errors.append("Structure section is not a list")
        return False, timestamp_structure_errors

    # Sort sections by start time
    try:
        sections = sorted(yaml_data["structure"], key=lambda x: timestamp_to_ms(x["start_time"]))
    except (KeyError, ValueError) as e:
        timestamp_structure_errors.append(f"Error parsing structure timestamps: {str(e)}")
        return False, timestamp_structure_errors

    # Check for overlaps
    for i in range(1, len(sections)):
        prev_section = sections[i - 1]
        curr_section = sections[i]

        prev_end_ms = timestamp_to_ms(prev_section["end_time"])
        curr_start_ms = timestamp_to_ms(curr_section["start_time"])

        if curr_start_ms < prev_end_ms:
            timestamp_structure_errors.append(
                f"Overlapping sections: '{prev_section['section_name']}' ({prev_section['end_time']}) overlaps with '{curr_section['section_name']}' ({curr_section['start_time']})")

    return len(timestamp_structure_errors) == 0, timestamp_structure_errors


def validate_yaml_file(file_path: str) -> Tuple[bool, List[str]]:
    """
    Validates a single YAML file.

    Args:
        file_path: Path to the YAML file

    Returns:
        Tuple of (is_valid, list_of_error_messages)
    """
    yaml_errors = []

    try:
        with open(file_path, 'r') as file:
            yaml_data = yaml.safe_load(file)

        # Run validation: required properties
        is_valid, req_errors = validate_required_properties(yaml_data)
        if not is_valid:
            for err in req_errors:
                yaml_errors.append(f"{os.path.basename(file_path)}: {err}")

        # Run validation: lyrics has lrc
        is_valid, error = validate_lyrics_has_lrc(yaml_data)
        if not is_valid:
            yaml_errors.append(f"{os.path.basename(file_path)}: {error}")

        # Run validation: file existence
        yaml_dir = os.path.dirname(file_path)
        is_valid, file_errors = validate_file_existence(yaml_data, yaml_dir)
        if not is_valid:
            for err in file_errors:
                yaml_errors.append(f"{os.path.basename(file_path)}: {err}")

        # Run validation: structure timestamps
        is_valid, structure_errors = validate_structure_timestamps(yaml_data)
        if not is_valid:
            for err in structure_errors:
                yaml_errors.append(f"{os.path.basename(file_path)}: {err}")

        is_valid, timestamp_errors = validate_timestamp_format(yaml_data)
        if not is_valid:
            for err in timestamp_errors:
                yaml_errors.append(f"{os.path.basename(file_path)}: {err}")

        return len(yaml_errors) == 0, yaml_errors

    except Exception as e:
        return False, [f"{os.path.basename(file_path)}: Error parsing YAML: {str(e)}"]


if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("Usage: python manifest_validator.py <directory_path>")
        sys.exit(1)
    directory_path = sys.argv[1]
    all_valid, errors = validate_directory(directory_path)
    if all_valid:
        print("All YAML files are valid.")
    else:
        print("Some YAML files have errors:")
        for error in errors:
            print(f" - {error}")
        sys.exit(1)
