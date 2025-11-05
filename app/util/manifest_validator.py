import glob
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import discogs_client
import yaml
from dotenv import load_dotenv


def get_discogs_cache_path() -> Path:
    """Get path to Discogs validation cache"""
    cache_dir = Path.home() / ".earthly_frames" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / "discogs_validation_cache.json"


def load_discogs_cache() -> Dict[str, str]:
    """Load cached Discogs ID validations"""
    cache_path = get_discogs_cache_path()
    if cache_path.exists():
        with open(cache_path, "r") as f:
            return json.load(f)
    return {}


def save_discogs_cache(cache: Dict[str, str]) -> None:
    """Save Discogs ID validation cache"""
    cache_path = get_discogs_cache_path()
    with open(cache_path, "w") as f:
        json.dump(cache, f, indent=2)


def validate_discogs_ids(
    yaml_data: Dict[str, Any], rate_limit_delay: float = 1.0, use_cache: bool = True
) -> Tuple[bool, List[str]]:
    """
    Validates that the Discogs IDs match the artist names with caching support.

    Args:
        yaml_data: Parsed YAML content
        rate_limit_delay: Seconds to wait between API requests (default: 1.0)
        use_cache: Whether to use cached results (default: True)
    """
    discog_errors = []

    if not isinstance(yaml_data, dict) or "sounds_like" not in yaml_data:
        return True, discog_errors

    if not isinstance(yaml_data["sounds_like"], list):
        discog_errors.append("sounds_like section is not a list")
        return False, discog_errors

    cache = load_discogs_cache() if use_cache else {}
    cache_updated = False

    try:
        load_dotenv()
        user_agent = "earthly_frames_discogs/1.0"
        discogs = discogs_client.Client(
            user_agent, user_token=os.environ.get("USER_ACCESS_TOKEN")
        )

        total_artists = len(yaml_data["sounds_like"])

        for i, artist in enumerate(yaml_data["sounds_like"]):
            if not isinstance(artist, dict):
                discog_errors.append(f"Artist entry {i + 1} is not a dictionary")
                continue

            if "name" not in artist or "discogs_id" not in artist:
                discog_errors.append(
                    f"Artist entry {i + 1} missing required 'name' or 'discogs_id' property"
                )
                continue

            artist_name = artist["name"]
            discogs_id = str(artist["discogs_id"])
            cache_key = f"{discogs_id}"

            if cache_key in cache:
                api_artist_name = cache[cache_key]
                print(
                    f"  Checking artist {i + 1}/{total_artists}: {artist_name} (cached)...",
                    end=" ",
                )

                if api_artist_name.lower() != artist_name.lower():
                    discog_errors.append(
                        f"Discogs ID {discogs_id} corresponds to '{api_artist_name}', not '{artist_name}'"
                    )
                    print("❌")
                else:
                    print("✅")
                continue
            try:
                print(
                    f"  Checking artist {i + 1}/{total_artists}: {artist_name} (ID: {discogs_id})...",
                    end=" ",
                )

                discogs_artist = discogs.artist(discogs_id)
                api_artist_name = discogs_artist.name

                cache[cache_key] = api_artist_name
                cache_updated = True

                if api_artist_name.lower() != artist_name.lower():
                    discog_errors.append(
                        f"Discogs ID {discogs_id} corresponds to '{api_artist_name}', not '{artist_name}'"
                    )
                    print("❌")
                else:
                    print("✅")

            except discogs_client.exceptions.HTTPError as e:
                if e.status_code == 429:
                    print(
                        f"⏳ (rate limited, waiting {rate_limit_delay * 3}s)...",
                        end=" ",
                    )
                    time.sleep(rate_limit_delay * 3)
                    try:
                        discogs_artist = discogs.artist(discogs_id)
                        api_artist_name = discogs_artist.name
                        cache[cache_key] = api_artist_name
                        cache_updated = True

                        if api_artist_name.lower() != artist_name.lower():
                            discog_errors.append(
                                f"Discogs ID {discogs_id} corresponds to '{api_artist_name}', not '{artist_name}'"
                            )
                            print("❌")
                        else:
                            print("✅")
                    except Exception as retry_error:
                        discog_errors.append(
                            f"Error checking Discogs ID {discogs_id} (retry failed): {str(retry_error)}"
                        )
                        print("❌")
                else:
                    discog_errors.append(
                        f"Error checking Discogs ID {discogs_id}: {str(e)}"
                    )
                    print("❌")
            except Exception as e:
                discog_errors.append(
                    f"Error checking Discogs ID {discogs_id}: {str(e)}"
                )
                print("❌")

            if i < total_artists - 1:
                time.sleep(rate_limit_delay)

    except Exception as e:
        discog_errors.append(f"Error initializing Discogs client: {str(e)}")

    if cache_updated and use_cache:
        save_discogs_cache(cache)

    return len(discog_errors) == 0, discog_errors


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
                    time_stamp_errors.append(
                        f"Invalid timestamp format in section {i + 1} ({key}): {timestamp}"
                    )

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

    if yaml_data.get("lyrics"):
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
    dir_all_valid = True

    yml_files = glob.glob(os.path.join(directory_path, "**/*.yml"), recursive=True)

    for yml_file in yml_files:
        is_valid, errors = validate_yaml_file(yml_file)
        if not is_valid:
            dir_all_valid = False
            all_errors.extend(errors)

    return dir_all_valid, all_errors


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

    file_refs = {"lrc": [], "wav": [], "mid": []}

    if "lrc_file" in yaml_data and yaml_data["lrc_file"]:
        file_refs["lrc"].append(yaml_data["lrc_file"])

    if "main_audio_file" in yaml_data and yaml_data["main_audio_file"]:
        file_refs["wav"].append(yaml_data["main_audio_file"])

    if "audio_tracks" in yaml_data and isinstance(yaml_data["audio_tracks"], list):
        for audio_track in yaml_data["audio_tracks"]:
            if not isinstance(audio_track, dict):
                continue

            if "audio_file" in audio_track and audio_track["audio_file"]:
                file_refs["wav"].append(audio_track["audio_file"])

            if "midi_file" in audio_track and audio_track["midi_file"]:
                file_refs["mid"].append(audio_track["midi_file"])

            if "midi_group_file" in audio_track and audio_track["midi_group_file"]:
                file_refs["mid"].append(audio_track["midi_group_file"])

    return file_refs


def validate_file_existence(
    yaml_data: Dict[str, Any], yaml_dir: str
) -> Tuple[bool, List[str]]:
    """
    Validates that all referenced files exist on disk.
    Returns errors with 'not found' in the message for consistency.
    """
    file_reference_errors = []

    if "lrc_file" in yaml_data and yaml_data["lrc_file"]:
        lrc_path = os.path.join(yaml_dir, yaml_data["lrc_file"])
        if not os.path.exists(lrc_path):
            file_reference_errors.append(f"LRC file not found: {yaml_data['lrc_file']}")

    if "main_audio_file" in yaml_data and yaml_data["main_audio_file"]:
        main_audio_path = os.path.join(yaml_dir, yaml_data["main_audio_file"])
        if not os.path.exists(main_audio_path):
            file_reference_errors.append(
                f"Main audio file not found: {yaml_data['main_audio_file']}"
            )

    if "audio_tracks" in yaml_data and isinstance(yaml_data["audio_tracks"], list):
        for i, a_track in enumerate(yaml_data["audio_tracks"]):
            if not isinstance(a_track, dict):
                continue

            track_id = a_track.get("id", i + 1)
            track_desc = a_track.get("description", "Unknown")
            track_player = a_track.get("player", "GABE")  # Default to GABE

            if "audio_file" in a_track and a_track["audio_file"]:
                audio_path = os.path.join(yaml_dir, a_track["audio_file"])
                if not os.path.exists(audio_path):
                    file_reference_errors.append(
                        f"Audio track {track_id} ({track_desc} - {track_player}): WAV file not found: {a_track['audio_file']}"
                    )

            if "midi_file" in a_track and a_track["midi_file"]:
                midi_path = os.path.join(yaml_dir, a_track["midi_file"])
                if not os.path.exists(midi_path):
                    file_reference_errors.append(
                        f"Audio track {track_id} ({track_desc} - {track_player}): MIDI file not found: {a_track['midi_file']}"
                    )

            if "midi_group_file" in a_track and a_track["midi_group_file"]:
                midi_group_path = os.path.join(yaml_dir, a_track["midi_group_file"])
                if not os.path.exists(midi_group_path):
                    file_reference_errors.append(
                        f"Audio track {track_id} ({track_desc} - {track_player}): MIDI group file not found: {a_track['midi_group_file']}"
                    )

    return len(file_reference_errors) == 0, file_reference_errors


def validate_manifest_completeness(yaml_file_path: str) -> Dict[str, Any]:
    """
    Get a summary of what's complete and what's missing in a manifest.
    Returns all keys expected by tests.
    """

    with open(yaml_file_path, "r") as f:
        yaml_data = yaml.safe_load(f)
    yaml_dir = os.path.dirname(yaml_file_path)
    m_id = yaml_data.get("manifest_id", "unknown")

    result = {
        "manifest_id": m_id,
        "has_lrc": False,
        "has_main_audio": False,
        "has_all_audio": True,
        "has_midi": False,
        "has_lyrics": yaml_data.get("lyrics", False),
        "total_tracks": 0,
        "complete_tracks": 0,
        "incomplete_tracks": [],
        "missing_audio": [],
        "missing_midi": [],
        "missing_files": [],
        "completion_percentage": 0.0,
    }

    # LRC
    if "lrc_file" in yaml_data and yaml_data["lrc_file"]:
        lrc_path = os.path.join(yaml_dir, yaml_data["lrc_file"])
        result["has_lrc"] = os.path.exists(lrc_path)
        if not result["has_lrc"]:
            result["missing_files"].append(f"LRC: {yaml_data['lrc_file']}")

    # Main audio
    if "main_audio_file" in yaml_data and yaml_data["main_audio_file"]:
        main_path = os.path.join(yaml_dir, yaml_data["main_audio_file"])
        result["has_main_audio"] = os.path.exists(main_path)
        if not result["has_main_audio"]:
            result["missing_files"].append(
                f"Main audio: {yaml_data['main_audio_file']}"
            )

    # Audio tracks
    if "audio_tracks" in yaml_data and isinstance(yaml_data["audio_tracks"], list):
        result["total_tracks"] = len(yaml_data["audio_tracks"])
        for audio_track in yaml_data["audio_tracks"]:
            audio_file = audio_track.get("audio_file")
            if audio_file:
                audio_path = os.path.join(yaml_dir, audio_file)
                if not os.path.exists(audio_path):
                    result["incomplete_tracks"].append(audio_file)
                    result["missing_audio"].append(audio_file)
        result["complete_tracks"] = result["total_tracks"] - len(
            result["incomplete_tracks"]
        )
        result["has_all_audio"] = len(result["incomplete_tracks"]) == 0

    # MIDI
    midi_file = yaml_data.get("midi_file")
    if midi_file:
        midi_path = os.path.join(yaml_dir, midi_file)
        result["has_midi"] = os.path.exists(midi_path)
        if not result["has_midi"]:
            result["missing_midi"].append(midi_file)

    # Also check for midi_file in audio_tracks
    if "audio_tracks" in yaml_data and isinstance(yaml_data["audio_tracks"], list):
        for audio_track in yaml_data["audio_tracks"]:
            midi_file = audio_track.get("midi_file")
            if midi_file:
                midi_path = os.path.join(yaml_dir, midi_file)
                if not os.path.exists(midi_path):
                    result["missing_midi"].append(midi_file)

    # Completion percentage calculation
    total_items = 1  # main audio
    completed_items = 1 if result["has_main_audio"] else 0
    if "audio_tracks" in yaml_data and isinstance(yaml_data["audio_tracks"], list):
        total_items += len(yaml_data["audio_tracks"])
        completed_items += result["complete_tracks"]
    if yaml_data.get("midi_file") or any(
        at.get("midi_file") for at in yaml_data.get("audio_tracks", [])
    ):
        total_items += len(result["missing_midi"]) + (
            1 if yaml_data.get("midi_file") else 0
        )
        completed_items += len(result["missing_midi"]) == 0
    if yaml_data.get("lyrics"):
        total_items += 1
        completed_items += 1 if result["has_lrc"] else 0
    result["completion_percentage"] = (
        round(100.0 * completed_items / total_items, 2) if total_items > 0 else 0.0
    )
    return result


def timestamp_to_ms(timestamp: str) -> int:
    """
    Convert a timestamp string in format [MM:SS.mmm] to milliseconds.

    Args:
        timestamp: Time in format [MM:SS.mmm]

    Returns:
        Time in milliseconds
    """
    # Remove brackets
    timestamp = timestamp.strip("[]")

    # Split by :
    minutes_str, seconds_str = timestamp.split(":")

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
    property_errors = []

    # Define required top-level properties
    required_properties = [
        "bpm",
        "manifest_id",
        "tempo",
        "key",
        "rainbow_color",
        "title",
        "release_date",
        "album_sequence",
        "main_audio_file",
        "TRT",
        "vocals",
        "lyrics",
        "structure",
        "mood",
        "sounds_like",
        "genres",
        "concept",
        "audio_tracks",
    ]

    for prop in required_properties:
        if prop not in yaml_data:
            property_errors.append(f"Missing required property: {prop}")

    if "structure" in yaml_data and isinstance(yaml_data["structure"], list):
        for i, section in enumerate(yaml_data["structure"]):
            if not isinstance(section, dict):
                property_errors.append(f"Structure section {i + 1} is not a dictionary")
                continue

            section_props = ["section_name", "start_time", "end_time", "description"]
            for prop in section_props:
                if prop not in section:
                    property_errors.append(
                        f"Structure section {i + 1} missing required property: {prop}"
                    )

    if "audio_tracks" in yaml_data and isinstance(yaml_data["audio_tracks"], list):
        for i, audio_track in enumerate(yaml_data["audio_tracks"]):
            if not isinstance(audio_track, dict):
                property_errors.append(f"Audio track {i + 1} is not a dictionary")
                continue
            if "id" not in audio_track:
                property_errors.append(
                    f"Audio track {i + 1} missing required property: id"
                )
            if "description" not in audio_track:
                property_errors.append(
                    f"Audio track {i + 1} missing required property: description"
                )
            file_types = ["audio_file", "midi_file", "midi_group_file"]
            if not any(file_type in audio_track for file_type in file_types):
                property_errors.append(
                    f"Audio track {i + 1} missing at least one file type (audio_file, midi_file, or midi_group_file)"
                )

    return len(property_errors) == 0, property_errors


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

    try:
        sections = sorted(
            yaml_data["structure"], key=lambda x: timestamp_to_ms(x["start_time"])
        )
    except (KeyError, ValueError) as e:
        timestamp_structure_errors.append(
            f"Error parsing structure timestamps: {str(e)}"
        )
        return False, timestamp_structure_errors

    # Check for overlaps
    for i in range(1, len(sections)):
        prev_section = sections[i - 1]
        curr_section = sections[i]

        prev_end_ms = timestamp_to_ms(prev_section["end_time"])
        curr_start_ms = timestamp_to_ms(curr_section["start_time"])

        if curr_start_ms < prev_end_ms:
            timestamp_structure_errors.append(
                f"Overlapping sections: '{prev_section['section_name']}' ({prev_section['end_time']}) overlaps with '{curr_section['section_name']}' ({curr_section['start_time']})"
            )

    return len(timestamp_structure_errors) == 0, timestamp_structure_errors


def check_no_tk_fields(data, path=""):
    tk_errors = []
    if isinstance(data, dict):
        for k, v in data.items():
            tk_errors.extend(check_no_tk_fields(v, f"{path}.{k}" if path else k))
    elif isinstance(data, list):
        for idx, item in enumerate(data):
            tk_errors.extend(check_no_tk_fields(item, f"{path}[{idx}]"))
    elif data == "TK":
        tk_errors.append(f"Manifest field at '{path}' is set to 'TK' (to come)")
    return tk_errors


def validate_field_values(yaml_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Perform stricter validation on common manifest field values.

    Checks include:
    - bpm is a positive integer
    - tempo is either an int or a string like '4/4'
    - rainbow_color is a known rainbow table key or color name
    - main_audio_file and audio_tracks references are not placeholder 'TK'
    - mood and genres are lists of strings
    - sounds_like is a list of dicts with name and discogs_id
    - top-level required fields are non-empty and not 'TK'
    """
    errors: List[str] = []

    if not isinstance(yaml_data, dict):
        return False, ["Invalid YAML structure: expected a dict"]

    # bpm
    bpm = yaml_data.get("bpm")
    if bpm is None:
        errors.append("bpm is missing")
    else:
        try:
            if isinstance(bpm, str) and bpm.isdigit():
                bpm_val = int(bpm)
            else:
                bpm_val = int(bpm)
            if bpm_val <= 0:
                errors.append(f"bpm must be a positive integer: {bpm}")
        except Exception:
            errors.append(f"bpm must be an integer: {bpm}")

    # tempo: allow formats like '4/4' or integer beats per bar
    tempo = yaml_data.get("tempo")
    if tempo is None:
        errors.append("tempo is missing")
    else:
        if isinstance(tempo, str):
            if not re.match(r"^\d+(?:/\d+)?$", tempo.strip()):
                errors.append(
                    f"tempo must be numeric or in 'N/D' form (e.g. '4/4'): {tempo}"
                )
        elif isinstance(tempo, (int, float)):
            pass
        else:
            errors.append(f"tempo has invalid type: {type(tempo).__name__}")

    # rainbow_color: try to validate against the rainbow table keys or names
    rc = yaml_data.get("rainbow_color")
    if rc is None:
        errors.append("rainbow_color is missing")
    else:
        try:
            from app.structures.concepts.rainbow_table_color import \
                the_rainbow_table_colors

            allowed_keys = set(the_rainbow_table_colors.keys())
            allowed_names = {v.color_name for v in the_rainbow_table_colors.values()}
            if isinstance(rc, str):
                rc_clean = rc.strip()
                if (len(rc_clean) == 1 and rc_clean in allowed_keys) or (
                    rc_clean in allowed_names
                ):
                    pass
                else:
                    errors.append(
                        f"rainbow_color '{rc}' is not a recognized key or color name"
                    )
            else:
                errors.append(f"rainbow_color has invalid type: {type(rc).__name__}")
        except Exception:
            # If rainbow table isn't importable, skip strict check but ensure it's not TK
            if rc == "TK":
                errors.append("rainbow_color must be set (found 'TK')")

    # main_audio_file shouldn't be 'TK' or empty
    main_audio = yaml_data.get("main_audio_file")
    if not main_audio:
        errors.append("main_audio_file is missing or empty")
    elif isinstance(main_audio, str) and main_audio.strip() == "TK":
        errors.append("main_audio_file is placeholder 'TK'")

    # mood and genres must be lists of strings
    for key in ("mood", "genres"):
        val = yaml_data.get(key)
        if val is None:
            errors.append(f"{key} is missing")
        elif not isinstance(val, list):
            errors.append(f"{key} must be a list of strings")
        else:
            if not all(
                isinstance(x, str) and x.strip() != "" and x != "TK" for x in val
            ):
                errors.append(f"{key} contains invalid or placeholder entries")

    # sounds_like list entries
    sl = yaml_data.get("sounds_like")
    if sl is None:
        errors.append("sounds_like is missing")
    elif not isinstance(sl, list):
        errors.append("sounds_like must be a list")
    else:
        for i, artist in enumerate(sl):
            if not isinstance(artist, dict):
                errors.append(f"sounds_like[{i}] is not a dict")
                continue
            name = artist.get("name")
            discogs_id = artist.get("discogs_id")
            if not name or name == "TK":
                errors.append(f"sounds_like[{i}].name missing or placeholder")
            if not discogs_id or discogs_id == "TK":
                errors.append(f"sounds_like[{i}].discogs_id missing or placeholder")

    # audio_tracks validation
    at = yaml_data.get("audio_tracks")
    if at is None:
        errors.append("audio_tracks is missing")
    elif not isinstance(at, list):
        errors.append("audio_tracks must be a list")
    else:
        for i, track in enumerate(at):
            if not isinstance(track, dict):
                errors.append(f"audio_tracks[{i}] is not a dict")
                continue
            # id
            if "id" not in track:
                errors.append(f"audio_tracks[{i}] missing id")
            # ensure at least one file reference and not TK
            files_present = False
            for fkey in ("audio_file", "midi_file", "midi_group_file"):
                if fkey in track and track[fkey] and track[fkey] != "TK":
                    files_present = True
            if not files_present:
                errors.append(
                    f"audio_tracks[{i}] has no valid file reference (audio_file/midi_file/midi_group_file)"
                )

    # Check for pervasive 'TK' placeholders anywhere
    tk_errors = check_no_tk_fields(yaml_data)
    for e in tk_errors:
        errors.append(e)

    return len(errors) == 0, errors


def validate_yaml_file(file_path: str) -> Tuple[bool, List[str]]:
    """Validates a single YAML file.

    Args:
        file_path: Path to the YAML file

    Returns:
        Tuple of (is_valid, list_of_error_messages)
    """
    errors = []

    try:
        with open(file_path, "r") as file:
            yaml_data = yaml.safe_load(file)

        # Run validation: required properties
        is_valid, req_errors = validate_required_properties(yaml_data)
        if not is_valid:
            for err in req_errors:
                errors.append(f"{os.path.basename(file_path)}: {err}")

        # Run validation: stricter field values
        is_valid, field_errors = validate_field_values(yaml_data)
        if not is_valid:
            for err in field_errors:
                errors.append(f"{os.path.basename(file_path)}: {err}")

        # Run validation: lyrics has lrc
        is_valid, lrc_error = validate_lyrics_has_lrc(yaml_data)
        if not is_valid:
            errors.append(f"{os.path.basename(file_path)}: {lrc_error}")

        # Run validation: file existence
        yaml_dir = os.path.dirname(file_path)
        is_valid, file_errors = validate_file_existence(yaml_data, yaml_dir)
        if not is_valid:
            for err in file_errors:
                errors.append(f"{os.path.basename(file_path)}: {err}")

        # Run validation: structure timestamps
        is_valid, structure_errors = validate_structure_timestamps(yaml_data)
        if not is_valid:
            for err in structure_errors:
                errors.append(f"{os.path.basename(file_path)}: {err}")

        # Run validation: timestamp format
        is_valid, timestamp_errors = validate_timestamp_format(yaml_data)
        if not is_valid:
            for err in timestamp_errors:
                errors.append(f"{os.path.basename(file_path)}: {err}")

        # Run validation: Discogs IDs
        is_valid, discogs_errors = validate_discogs_ids(yaml_data)
        if not is_valid:
            for err in discogs_errors:
                errors.append(f"{os.path.basename(file_path)}: {err}")

        tk_errors = check_no_tk_fields(yaml_data)
        if tk_errors:
            for err in tk_errors:
                errors.append(f"{os.path.basename(file_path)}: {err}")

        return len(errors) == 0, errors

    except Exception as e:
        return False, [f"{os.path.basename(file_path)}: Error parsing YAML: {str(e)}"]


if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("Usage: python manifest_validator.py <directory_path>")
        sys.exit(1)

    directory_path = sys.argv[1]

    # Run validation
    all_valid, errors = validate_directory(directory_path)

    if all_valid:
        print("✅ All YAML files are valid.")
    else:
        print("❌ Some YAML files have errors:")
        for error in errors:
            print(f"  - {error}")

    # Show completeness summary for each manifest
    print("\nMANIFEST COMPLETENESS SUMMARY:")
    yaml_files = glob.glob(os.path.join(directory_path, "**/*.yml"), recursive=True)

    for yaml_file in sorted(yaml_files):
        try:
            completeness = validate_manifest_completeness(yaml_file)
            manifest_id = completeness["manifest_id"]
            percentage = completeness["completion_percentage"]

            status = "✅" if percentage >= 100 else "⚠️"
            print(
                f"\n{status} {manifest_id}: {percentage:.0f}% complete "
                f"({completeness['complete_tracks']}/{completeness['total_tracks']} tracks)"
            )

            if completeness["incomplete_tracks"]:
                print("  Missing files in tracks:")
                for track in completeness["incomplete_tracks"]:
                    print(
                        f"    - Track {track['id']} ({track['description']} - {track['player']}):"
                    )
                    for missing in track["missing"]:
                        print(f"        • {missing}")
        except Exception as e:
            print(
                f"  ⚠️ Could not check completeness for {os.path.basename(yaml_file)}: {e}"
            )

    if not all_valid:
        sys.exit(1)
