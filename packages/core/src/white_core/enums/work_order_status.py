from enum import Enum


class WorkOrderStatus(str, Enum):
    DRAFT = "draft"
    SENT = "sent"
    IN_PROGRESS = "in_progress"
    DELIVERED = "delivered"
    ACCEPTED = "accepted"
    REVISION_REQUESTED = "revision_requested"
