import re
import base64

def safe_filename(title: str) -> str:
    return re.sub(r'[\\/*?:"<>|\' ]+', '_', title).strip('_')

def just_lyrics(lyric_content: dict) -> str:
    return ''.join([line['content'] for line in lyric_content])

def make_lrc_fragment(album:str, song:str, artist: str, lyric_content: dict) -> str:
    lrc = f"[ti: {song}]\n"
    lrc += f"[ar: {artist}]\n"
    lrc += f"[al: {album}]\n"
    for lyric_content in lyric_content:
        lrc += f"{lyric_content['time_stamp']}\n"
        lrc += f"{lyric_content['content']}\n"
    return lrc

def to_str_dict(d: dict) -> dict:
    return {
        k: v if v is None or isinstance(v, (bool, bytes)) else str(v)
        for k, v in d.items()
    }

def bytes_to_base64_str(b):
    if b is None:
        return None
    return base64.b64encode(b).decode('utf-8')