import random
import re
import base64

from app.objects.rainbow_color import RainbowColor


def safe_filename(title: str) -> str:
    return re.sub(r'[\\/*?:"<>|\' ]+', '_', title).strip('_')

def just_lyrics(lyric_events_in_range) -> str:
    pass

def make_lrc_fragment(album, song, artist, lyric_events_in_range) -> str:
    lrc = f"[al:{album}]\n[ti:{song}]\n[ar:{artist}]\n"
    pass

def to_str_dict(d: dict) -> dict:
    return {
        k: v if v is None or isinstance(v, (bool, bytes)) else str(v)
        for k, v in d.items()
    }

def bytes_to_base64_str(b):
    if b is None:
        return None
    return base64.b64encode(b).decode('utf-8')

def get_random_musical_key() -> str:
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    modes = ['major', 'minor']
    rare_modes = ['dorian', 'phrygian', 'lydian', 'mixolydian', 'locrian']
    note = random.choice(notes)
    if random.random() < 0.08:
        mode = random.choice(rare_modes)
    else:
        mode = random.choice(modes)
    return f"{note} {mode}"

def uuid_representer(dumper, data):
    return dumper.represent_scalar('tag:yaml.org,2002:str', str(data))

def enum_representer(dumper, data):
    return dumper.represent_scalar('tag:yaml.org,2002:str', str(data.name))

def convert_to_rainbow_color(color_value):
    """Convert a string value to RainbowColor enum"""
    if isinstance(color_value, str):
        try:
            return getattr(RainbowColor, color_value)
        except (AttributeError, KeyError):
            return RainbowColor.Z
    return color_value

def quote_yaml_values(yaml_str):
    def replacer(match):
        key, value = match.group(1), match.group(2)
        if ':' in value and not (value.startswith('"') or value.startswith("'")):
            value = f'"{value.strip()}"'
        return f"{key}: {value}"
    return re.sub(r'^(\s*\w+):\s*(.+)$', replacer, yaml_str, flags=re.MULTILINE)