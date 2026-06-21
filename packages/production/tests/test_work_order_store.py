from datetime import date

import pytest
from white_production.work_order_store import (
    list_work_orders,
    load_work_order,
    save_work_order,
)

from white_core.enums.collaborator_role import CollaboratorRole
from white_core.enums.work_order_status import WorkOrderStatus
from white_core.music.core.work_order import WorkOrder


def _make_wo(collaborator_id: str = "kate-koherence", **kwargs) -> WorkOrder:
    defaults = {
        "id": f"{collaborator_id}-the-song",
        "song_slug": "the-song",
        "collaborator_id": collaborator_id,
        "role": CollaboratorRole.VOCALIST,
    }
    defaults.update(kwargs)
    return WorkOrder(**defaults)


def test_save_creates_work_orders_dir(tmp_path):
    prod_dir = tmp_path / "production" / "the-song"
    wo = _make_wo()
    save_work_order(prod_dir, wo)
    assert (prod_dir / "work_orders" / "kate-koherence.yml").exists()


def test_save_and_load_roundtrip(tmp_path):
    prod_dir = tmp_path / "the-song"
    wo = _make_wo(key="D minor", bpm=112)
    save_work_order(prod_dir, wo)
    loaded = load_work_order(prod_dir, "kate-koherence")
    assert loaded.song_slug == "the-song"
    assert loaded.key == "D minor"
    assert loaded.bpm == 112


def test_load_unknown_raises(tmp_path):
    prod_dir = tmp_path / "the-song"
    prod_dir.mkdir(parents=True)
    with pytest.raises(FileNotFoundError):
        load_work_order(prod_dir, "nobody")


def test_list_empty(tmp_path):
    prod_dir = tmp_path / "the-song"
    prod_dir.mkdir(parents=True)
    assert list_work_orders(prod_dir) == []


def test_list_missing_dir(tmp_path):
    assert list_work_orders(tmp_path / "nonexistent") == []


def test_list_returns_all(tmp_path):
    prod_dir = tmp_path / "the-song"
    for cid in ["collab-a", "collab-b", "collab-c"]:
        wo = _make_wo(collaborator_id=cid)
        save_work_order(prod_dir, wo)
    result = list_work_orders(prod_dir)
    assert len(result) == 3
    cids = {wo.collaborator_id for wo in result}
    assert cids == {"collab-a", "collab-b", "collab-c"}


def test_save_updates_updated_at(tmp_path):
    prod_dir = tmp_path / "the-song"
    wo = _make_wo()
    original_updated = wo.updated_at
    save_work_order(prod_dir, wo)
    # updated_at should be set to now (≥ original)
    assert wo.updated_at >= original_updated


def test_overwrite_preserves_data(tmp_path):
    prod_dir = tmp_path / "the-song"
    wo = _make_wo(creative_direction="first draft")
    save_work_order(prod_dir, wo)
    wo.creative_direction = "revised"
    wo.status = WorkOrderStatus.SENT
    save_work_order(prod_dir, wo)
    loaded = load_work_order(prod_dir, "kate-koherence")
    assert loaded.creative_direction == "revised"
    assert loaded.status == WorkOrderStatus.SENT


def test_roundtrip_with_deadline_and_budget(tmp_path):
    prod_dir = tmp_path / "the-song"
    from white_core.enums.budget_status import BudgetStatus

    wo = _make_wo(
        deadline=date(2026, 9, 1),
        budget_agreed=300.0,
        budget_status=BudgetStatus.AGREED,
    )
    save_work_order(prod_dir, wo)
    loaded = load_work_order(prod_dir, "kate-koherence")
    assert loaded.deadline == date(2026, 9, 1)
    assert loaded.budget_agreed == 300.0
    assert loaded.budget_status == BudgetStatus.AGREED
