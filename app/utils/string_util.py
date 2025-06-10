import re

def safe_filename(title: str) -> str:
    return re.sub(r'[\\/*?:"<>|\' ]+', '_', title).strip('_')