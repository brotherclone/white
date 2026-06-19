from enum import Enum


class BudgetStatus(str, Enum):
    PENDING = "pending"
    AGREED = "agreed"
    INVOICED = "invoiced"
    PAID = "paid"
