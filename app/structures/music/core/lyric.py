from pydantic import BaseModel

class LyricPair(BaseModel):
    """
    Represents a pair of lyrics with their corresponding timestamps.
    """
    lyric: str
    timestamp: str # [00:23.196]

    def __init__(self, **data):
        super().__init__(**data)
        if not self.lyric or not self.timestamp:
            raise ValueError("Both lyric and timestamp must be provided.")


class LyricFile(BaseModel):
    """
    Represents a lyric file containing multiple lyric pairs.
    """
    title: str
    artist: str
    album: str
    lyrics: list[LyricPair]

    def __init__(self, **data):
        super().__init__(**data)
  