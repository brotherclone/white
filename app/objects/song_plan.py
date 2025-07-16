import uuid
import yaml

from enum import Enum
from pydantic import BaseModel

from app.enums.plan_state import PlanState
from app.objects.rainbow_color import RainbowColor
from app.objects.plan_feedback import RainbowPlanFeedback
from app.objects.rainbow_song_meta import RainbowSongStructureModel
from app.objects.sounds_like import RainbowSoundsLike
from app.utils.string_util import uuid_representer, enum_representer


class RainbowSongPlan(BaseModel):
    batch_id: uuid.UUID | str | None = None
    plan_id: uuid.UUID | str | None = None
    plan_state: PlanState = PlanState.incomplete
    associated_resource: str | None = None
    key: str | None = None
    bpm: int | None = None
    tempo: str | None = None
    moods: list[str] | None = None
    moods_feedback: RainbowPlanFeedback | None = None
    sounds_like: list[RainbowSoundsLike] | None = None
    sounds_like_feedback: RainbowPlanFeedback | None = None
    rainbow_color: RainbowColor | None = None
    rainbow_color_feedback: RainbowPlanFeedback | None = None
    plan: str | None = None
    plan_feedback: RainbowPlanFeedback | None = None
    genres: list[str] | None = None
    genres_feedback: RainbowPlanFeedback | None = None
    structure: list[RainbowSongStructureModel] | None = None
    structure_feedback: RainbowPlanFeedback | None = None
    implementation_notes:  RainbowPlanFeedback | None = None


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
        self.structure_feedback = RainbowPlanFeedback(
                plan_id=self.plan_id,
                field_name="structure",
                rating=None,
                comment=None,
                suggested_replacement_value=None
        )
        self.implementation_notes = RainbowPlanFeedback(
                plan_id=self.plan_id,
                field_name="implementation_notes",
                rating=None,
                comment=None,
                suggested_replacement_value=None
            )

    def to_yaml(self):
        yaml_dumper = yaml.SafeDumper
        yaml_dumper.add_representer(uuid.UUID, uuid_representer)
        yaml_dumper.add_multi_representer(Enum, enum_representer)
        return yaml.dump(self.dict(), default_flow_style=False, allow_unicode=True, Dumper=yaml_dumper)
