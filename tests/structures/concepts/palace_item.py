from enum import Enum

from pydantic import BaseModel

from app.structures.concepts.palace_item_effect import PulsarPalaceEffect

class PalaceItemUsageType(Enum):

    ON_ENTER_ROOM = "on_enter_room"
    ON_INSPECT = "on_inspect"
    ON_ATTACK_WITH = "on_attack_with"
    ON_DEFEND_WITH = "on_defend_with"
    ON_DESTROY = "on_destroy"
    ON_PICK_UP = "on_pick_up"
    ON_DROP = "on_drop"

class PalaceItemEffect(BaseModel):
    usage_type: PalaceItemUsageType
    effect: PulsarPalaceEffect

    def __init__(self, **data):
        super().__init__(**data)

class PalaceItem(BaseModel):

    id: int
    name: str
    description: str
    effects: list[PalaceItemEffect] = []

    def __init__(self, **data):
        super().__init__(**data)