from __future__ import annotations

from datetime import date

from pydantic import BaseModel

from white_core.enums.collaborator_platform import CollaboratorPlatform
from white_core.enums.collaborator_role import CollaboratorRole
from white_core.enums.pro_affiliation import PROAffiliation


class AvailabilityWindow(BaseModel):
    unavailable_from: date
    unavailable_until: date
    reason: str | None = None


class PlatformProfile(BaseModel):
    platform: CollaboratorPlatform
    url: str
    username: str | None = None


class Collaborator(BaseModel):
    id: str
    name: str
    roles: list[CollaboratorRole]
    email: str | None = None
    photo_url: str | None = None
    platforms: list[PlatformProfile] = []
    website: str | None = None
    socials: dict[str, str] = {}
    pro_affiliation: PROAffiliation = PROAffiliation.NONE
    pro_number: str | None = None
    availability_windows: list[AvailabilityWindow] = []
    notes: str = ""
