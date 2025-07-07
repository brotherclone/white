import re
import os
import base64
import string

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
