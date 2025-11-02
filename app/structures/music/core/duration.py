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

    @classmethod
    def validate(cls, value: Any) -> Any:
        """Compatibility wrapper around model_validate used by older tests.

        If validation succeeds, returns a Duration instance. If validation fails
        (e.g. the input is an arbitrary string that doesn't match the pattern),
        return the original value to preserve previous behavior expected by tests.
        """
        try:
            # Use model_validate to apply our validators
            return cls.model_validate(value)
        except Exception:
            # On validation failure, return original input (per test expectations)
            return value
