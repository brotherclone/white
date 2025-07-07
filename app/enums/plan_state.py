from enum import Enum


class PlanState(Enum):
    incomplete = "incomplete"
    ready_to_review = "ready_to_review"
    accepted = "accepted"
    rejected = "rejected"