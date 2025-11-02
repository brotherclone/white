import re
from pydantic import BaseModel, model_validator
from typing import Any

class Duration(BaseModel):

    minutes: int
    seconds: float

    @model_validator(mode='before')
    @classmethod
    def validate_duration(cls, data: Any) -> Any:
        """Validate and parse duration strings in [MM:SS.mmm] format"""
        if isinstance(data, str):
            # Strip brackets if present
            clean_str = data.strip('[]')
            pattern = r'^(\d+):(\d+\.\d+)$'
            match = re.match(pattern, clean_str)

            if match:
                return {
                    'minutes': int(match.group(1)),
                    'seconds': float(match.group(2))
                }
        return data

    def __str__(self):
        return f"[{self.minutes:02d}:{self.seconds:06.3f}]"