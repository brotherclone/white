import uuid

from pydantic import BaseModel

from app.objects.plan_feedback import RainbowPlanFeedback


class RainbowImplementationNotes(BaseModel):

    notes: str | None = None
    feedback: RainbowPlanFeedback | None = None

    def __init__(self, **data):
        super().__init__(**data)
        self.feedback = RainbowPlanFeedback(
            plan_id=None,
            field_name="implementation_notes",
            rating=None,
            comment=None,
            suggested_replacement_value=None
        )
