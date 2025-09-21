from pydantic import BaseModel

class Duration(BaseModel):

    minutes: int
    seconds: float

    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        if isinstance(v, str):
            # Strip brackets if present
            clean_str = v.strip('[]')
            pattern = r'^(\d+):(\d+\.\d+)$'
            match = re.match(pattern, clean_str)

            if match:
                minutes = int(match.group(1))
                seconds = float(match.group(2))
                return cls(minutes=minutes, seconds=seconds)
        return v

    def __str__(self):
        return f"[{self.minutes:02d}:{self.seconds:06.3f}]"