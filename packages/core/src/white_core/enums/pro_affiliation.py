from enum import Enum


class PROAffiliation(str, Enum):
    ASCAP = "ascap"
    BMI = "bmi"
    SESAC = "sesac"
    SOCAN = "socan"
    PRS = "prs"
    OTHER = "other"
    NONE = "none"
