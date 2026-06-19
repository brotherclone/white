from datetime import date

from white_core.enums.collaborator_platform import CollaboratorPlatform
from white_core.enums.collaborator_role import CollaboratorRole
from white_core.enums.pro_affiliation import PROAffiliation
from white_core.music.core.collaborator import (
    AvailabilityWindow,
    Collaborator,
    PlatformProfile,
)


def _make_collaborator(**kwargs) -> Collaborator:
    defaults = {
        "id": "kate_koherence",
        "name": "Kate Koherence",
        "roles": [CollaboratorRole.VOCALIST],
    }
    defaults.update(kwargs)
    return Collaborator(**defaults)


def test_minimal_collaborator():
    c = _make_collaborator()
    assert c.id == "kate_koherence"
    assert c.pro_affiliation == PROAffiliation.NONE
    assert c.platforms == []
    assert c.availability_windows == []
    assert c.socials == {}


def test_platform_profile():
    p = PlatformProfile(
        platform=CollaboratorPlatform.AIRGIGS, url="https://airgigs.com/kate"
    )
    assert p.platform == CollaboratorPlatform.AIRGIGS
    assert p.username is None


def test_platform_profile_with_username():
    p = PlatformProfile(
        platform=CollaboratorPlatform.SOUNDBETTER,
        url="https://soundbetter.com/kate",
        username="kate_sings",
    )
    assert p.username == "kate_sings"


def test_availability_window_basic():
    w = AvailabilityWindow(
        unavailable_from=date(2026, 7, 1),
        unavailable_until=date(2026, 7, 15),
        reason="US tour",
    )
    assert w.reason == "US tour"


def test_availability_window_no_reason():
    w = AvailabilityWindow(
        unavailable_from=date(2026, 8, 1),
        unavailable_until=date(2026, 8, 10),
    )
    assert w.reason is None


def test_collaborator_with_platforms_and_windows():
    c = _make_collaborator(
        email="kate@example.com",
        platforms=[
            PlatformProfile(
                platform=CollaboratorPlatform.DIRECT, url="https://katekoherence.com"
            )
        ],
        pro_affiliation=PROAffiliation.ASCAP,
        pro_number="12345678",
        availability_windows=[
            AvailabilityWindow(
                unavailable_from=date(2026, 7, 1),
                unavailable_until=date(2026, 7, 15),
                reason="US tour",
            )
        ],
        socials={"instagram": "https://instagram.com/kate"},
    )
    assert c.email == "kate@example.com"
    assert len(c.platforms) == 1
    assert c.pro_affiliation == PROAffiliation.ASCAP
    assert c.pro_number == "12345678"
    assert len(c.availability_windows) == 1
    assert c.socials["instagram"] == "https://instagram.com/kate"


def test_collaborator_multi_role():
    c = _make_collaborator(roles=[CollaboratorRole.VOCALIST, CollaboratorRole.MIXING])
    assert len(c.roles) == 2


def test_collaborator_serialises_to_dict():
    c = _make_collaborator(
        platforms=[
            PlatformProfile(
                platform=CollaboratorPlatform.AIRGIGS, url="https://airgigs.com/k"
            )
        ]
    )
    d = c.model_dump()
    assert d["platforms"][0]["platform"] == "airgigs"
    assert d["roles"][0] == "vocalist"
    assert d["pro_affiliation"] == "none"


def test_collaborator_roundtrip_json():
    c = _make_collaborator(
        availability_windows=[
            AvailabilityWindow(
                unavailable_from=date(2026, 9, 1),
                unavailable_until=date(2026, 9, 30),
            )
        ]
    )
    reloaded = Collaborator.model_validate_json(c.model_dump_json())
    assert reloaded.id == c.id
    assert reloaded.availability_windows[0].unavailable_from == date(2026, 9, 1)
