import pytest
from white_production.collaborator_registry import (
    delete_collaborator,
    list_collaborators,
    load_collaborator,
    save_collaborator,
)

from white_core.enums.collaborator_role import CollaboratorRole
from white_core.music.core.collaborator import Collaborator


def _make(collaborator_id: str = "test-singer", **kwargs) -> Collaborator:
    defaults = {
        "id": collaborator_id,
        "name": "Test Singer",
        "roles": [CollaboratorRole.VOCALIST],
    }
    defaults.update(kwargs)
    return Collaborator(**defaults)


def test_save_and_load(tmp_path):
    c = _make()
    save_collaborator(c, registry_dir=tmp_path)
    loaded = load_collaborator("test-singer", registry_dir=tmp_path)
    assert loaded.id == c.id
    assert loaded.name == c.name
    assert loaded.roles == c.roles


def test_save_creates_yml_file(tmp_path):
    c = _make()
    save_collaborator(c, registry_dir=tmp_path)
    assert (tmp_path / "test-singer.yml").exists()


def test_load_unknown_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_collaborator("nobody", registry_dir=tmp_path)


def test_list_empty_dir(tmp_path):
    assert list_collaborators(registry_dir=tmp_path) == []


def test_list_missing_dir(tmp_path):
    assert list_collaborators(registry_dir=tmp_path / "does_not_exist") == []


def test_list_returns_all(tmp_path):
    for i in range(3):
        save_collaborator(
            _make(f"collab-{i}", name=f"Person {i}"), registry_dir=tmp_path
        )
    result = list_collaborators(registry_dir=tmp_path)
    assert len(result) == 3
    ids = {c.id for c in result}
    assert ids == {"collab-0", "collab-1", "collab-2"}


def test_save_overwrites(tmp_path):
    c = _make(notes="original")
    save_collaborator(c, registry_dir=tmp_path)
    c2 = _make(notes="updated")
    save_collaborator(c2, registry_dir=tmp_path)
    loaded = load_collaborator("test-singer", registry_dir=tmp_path)
    assert loaded.notes == "updated"


def test_delete_removes_file(tmp_path):
    c = _make()
    save_collaborator(c, registry_dir=tmp_path)
    delete_collaborator("test-singer", registry_dir=tmp_path)
    assert not (tmp_path / "test-singer.yml").exists()


def test_delete_unknown_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        delete_collaborator("nobody", registry_dir=tmp_path)


def test_delete_with_active_work_orders_raises(tmp_path):
    c = _make()
    save_collaborator(c, registry_dir=tmp_path)
    with pytest.raises(ValueError, match="active work orders"):
        delete_collaborator(
            "test-singer", registry_dir=tmp_path, active_song_slugs=["my-song"]
        )


def test_roundtrip_preserves_all_fields(tmp_path):
    from datetime import date

    from white_core.enums.collaborator_platform import CollaboratorPlatform
    from white_core.enums.pro_affiliation import PROAffiliation
    from white_core.music.core.collaborator import AvailabilityWindow, PlatformProfile

    c = Collaborator(
        id="full-collab",
        name="Full Collab",
        roles=[CollaboratorRole.MIXING],
        email="full@example.com",
        pro_affiliation=PROAffiliation.BMI,
        pro_number="99887766",
        platforms=[
            PlatformProfile(
                platform=CollaboratorPlatform.DIRECT, url="https://full.example.com"
            )
        ],
        socials={"instagram": "https://instagram.com/full"},
        availability_windows=[
            AvailabilityWindow(
                unavailable_from=date(2026, 10, 1),
                unavailable_until=date(2026, 10, 15),
                reason="holiday",
            )
        ],
        notes="prefers dry files",
    )
    save_collaborator(c, registry_dir=tmp_path)
    loaded = load_collaborator("full-collab", registry_dir=tmp_path)
    assert loaded == c
