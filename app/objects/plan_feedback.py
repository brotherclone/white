import uuid
from typing import Any

from pydantic import BaseModel


class RainbowPlanFeedback(BaseModel):
    plan_id: uuid.UUID | str | None = None
    field_name: str | None = None
    rating: float | None = None
    comment: str | None = None
    suggested_replacement_value: Any | None = None
