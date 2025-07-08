import uuid

from pydantic import BaseModel

from app.enums.plan_state import PlanState
from app.enums.rainbow_color import RainbowColor
from app.objects.implementation_notes import RainbowImplementationNotes
from app.objects.plan_feedback import RainbowPlanFeedback
from app.objects.sounds_like import RainbowSoundsLike


class RainbowSongPlan(BaseModel):
    batch_id: uuid.UUID | None = None
    plan_id: uuid.UUID
    plan_state: PlanState = PlanState.incomplete
    associated_resource: str | None = None
    key: str | None = None
    bpm: int | None = None
    tempo: str | None = None
    moods: list[str] | None = None
    moods_feedback: RainbowPlanFeedback | None = None
    sounds_like: RainbowSoundsLike | None = None
    sounds_like_feedback: RainbowPlanFeedback | None = None
    rainbow_color: RainbowColor
    rainbow_color_feedback: RainbowPlanFeedback | None = None
    plan: str | None
    plan_feedback: RainbowPlanFeedback | None = None
    genres: list[str] | None = None
    genres_feedback: RainbowPlanFeedback | None = None
    implementation_notes: RainbowImplementationNotes | None = None


    def __init__(self):
        super().__init__()
        self.batch_id = uuid.uuid4()
        self.plan_id = uuid.uuid4()
        self.moods_feedback = RainbowPlanFeedback(
            plan_id=self.plan_id,
            field_name="moods",
            rating=None,
            comment=None,
            suggested_replacement_value=None
        )
        self.sounds_like_feedback = RainbowPlanFeedback(
            plan_id=self.plan_id,
            field_name="sounds_like",
            rating=None,
            comment=None,
            suggested_replacement_value=None
        )
        self.rainbow_color_feedback = RainbowPlanFeedback(
            plan_id=self.plan_id,
            field_name="rainbow_color",
            rating=None,
            comment=None,
            suggested_replacement_value=None
        )
        self.plan_feedback = RainbowPlanFeedback(
                plan_id=self.plan_id,
                field_name="plan",
                rating=None,
                comment=None,
                suggested_replacement_value=None
            )
        self.genres_feedback = RainbowPlanFeedback(
                plan_id=self.plan_id,
                field_name="genres",
                rating=None,
                comment=None,
                suggested_replacement_value=None
            )
        self.implementation_notes = RainbowImplementationNotes(
            plain_id=self.plan_id,
            notes=RainbowPlanFeedback(
                plan_id=self.plan_id,
                field_name="implementation_notes",
                rating=None,
                comment=None,
                suggested_replacement_value=None
            )
        )




    def save_file(self):
        pass

