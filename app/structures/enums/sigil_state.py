from enum import Enum

class SigilState(str, Enum):
    CREATED = "created"
    AWAITING_CHARGE = "awaiting charge"
    CHARGING = "charging"
    CHARGED = "charged"
    BURIED = "buried"
    UNKNOWN = "unknown"