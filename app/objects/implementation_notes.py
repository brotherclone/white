import uuid

from pydantic import BaseModel

from app.objects.plan_feedback import RainbowPlanFeedback


class RainbowImplementationNotes(BaseModel):

    plain_id: uuid.UUID | None = None
    notes: RainbowPlanFeedback | None = None

    def __init__(self, **data):
        super().__init__(**data)
        self.notes = RainbowPlanFeedback(
            plan_id=self.plain_id,
            field_name="implementation_notes",
            rating=None,
            comment=None,
            suggested_replacement_value=None
        )
