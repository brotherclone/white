from typing import Any

from pydantic import BaseModel, field_validator

from app.structures.music.core.duration import Duration


class ManifestSongStructure(BaseModel):
    section_name: str
    start_time: str | Duration
    end_time: str | Duration
    description: str | None = None

    @field_validator("start_time", "end_time", mode="before")
    def _ensure_duration(cls, v):
        # Convert dicts produced by YAML parsing into Duration instances
        if isinstance(v, dict):
            return Duration(**v)
        return v

    def __getitem__(self, key: str) -> Any:
        if hasattr(self, key):
            return getattr(self, key)
        data = self.model_dump() if hasattr(self, "model_dump") else self.__dict__
        if isinstance(data, dict) and key in data:
            return data[key]
        raise KeyError(key)

    def __contains__(self, key: str) -> bool:
        if hasattr(self, key):
            return True
        data = self.model_dump() if hasattr(self, "model_dump") else self.__dict__
        return isinstance(data, dict) and key in data

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default
