from datetime import date

import pytest
from white_production.production_calendar import (
    build_deadline_event_payload,
    build_followup_event_payload,
    create_deadline_event,
    create_followup_event,
    delete_event,
    update_work_order_calendar,
)

from white_core.enums.collaborator_role import CollaboratorRole
from white_core.music.core.work_order import WorkOrder


def _make_wo(**kwargs) -> WorkOrder:
    defaults = {
        "id": "kate-the-song",
        "song_slug": "the-song",
        "collaborator_id": "kate-koherence",
        "role": CollaboratorRole.VOCALIST,
    }
    defaults.update(kwargs)
    return WorkOrder(**defaults)


# ---------------------------------------------------------------------------
# Payload builders
# ---------------------------------------------------------------------------


def test_followup_payload_structure():
    wo = _make_wo(
        follow_up_date=date(2026, 7, 15),
        follow_up_reason="Back from tour",
    )
    payload = build_followup_event_payload(wo, "The Archivist's Rebellion")
    assert "Follow up" in payload["summary"]
    assert "kate-koherence" in payload["summary"]
    assert "Back from tour" in payload["description"]
    assert "start" in payload and "end" in payload


def test_followup_payload_requires_follow_up_date():
    wo = _make_wo()
    with pytest.raises(ValueError, match="follow_up_date"):
        build_followup_event_payload(wo, "The Song")


def test_deadline_payload_structure():
    wo = _make_wo(deadline=date(2026, 8, 1))
    payload = build_deadline_event_payload(wo, "The Song")
    assert "Deadline" in payload["summary"]
    assert "kate-koherence" in payload["summary"]
    assert "vocalist" in payload["description"]


def test_deadline_payload_requires_deadline():
    wo = _make_wo()
    with pytest.raises(ValueError, match="deadline"):
        build_deadline_event_payload(wo, "The Song")


# ---------------------------------------------------------------------------
# create_followup_event
# ---------------------------------------------------------------------------


def test_create_followup_no_fn_returns_none():
    wo = _make_wo(follow_up_date=date(2026, 7, 15))
    assert create_followup_event(wo, "The Song") is None


def test_create_followup_no_date_returns_none():
    wo = _make_wo()
    assert (
        create_followup_event(wo, "The Song", _create_event_fn=lambda p: "id") is None
    )


def test_create_followup_calls_fn_and_returns_id():
    wo = _make_wo(follow_up_date=date(2026, 7, 15))
    event_id = create_followup_event(
        wo, "The Song", _create_event_fn=lambda p: "gcal_abc"
    )
    assert event_id == "gcal_abc"


def test_create_followup_degrades_on_fn_exception():
    wo = _make_wo(follow_up_date=date(2026, 7, 15))

    def boom(payload):
        raise RuntimeError("network error")

    result = create_followup_event(wo, "The Song", _create_event_fn=boom)
    assert result is None


# ---------------------------------------------------------------------------
# create_deadline_event
# ---------------------------------------------------------------------------


def test_create_deadline_no_fn_returns_none():
    wo = _make_wo(deadline=date(2026, 8, 1))
    assert create_deadline_event(wo, "The Song") is None


def test_create_deadline_calls_fn():
    wo = _make_wo(deadline=date(2026, 8, 1))
    event_id = create_deadline_event(
        wo, "The Song", _create_event_fn=lambda p: "gcal_xyz"
    )
    assert event_id == "gcal_xyz"


def test_create_deadline_degrades_on_fn_exception():
    wo = _make_wo(deadline=date(2026, 8, 1))

    def boom(payload):
        raise RuntimeError("quota exceeded")

    assert create_deadline_event(wo, "The Song", _create_event_fn=boom) is None


# ---------------------------------------------------------------------------
# delete_event
# ---------------------------------------------------------------------------


def test_delete_event_no_fn_noop():
    delete_event("gcal_abc")  # should not raise


def test_delete_event_calls_fn():
    deleted = []
    delete_event("gcal_abc", _delete_event_fn=lambda eid: deleted.append(eid))
    assert deleted == ["gcal_abc"]


def test_delete_event_empty_id_noop():
    delete_event(
        "", _delete_event_fn=lambda eid: (_ for _ in ()).throw(AssertionError())
    )


def test_delete_event_degrades_on_fn_exception():
    def boom(eid):
        raise RuntimeError("not found")

    delete_event("gcal_abc", _delete_event_fn=boom)  # should not raise


# ---------------------------------------------------------------------------
# update_work_order_calendar
# ---------------------------------------------------------------------------


def test_update_creates_followup_event():
    wo = _make_wo(follow_up_date=date(2026, 7, 15))
    wo = update_work_order_calendar(
        wo, "The Song", "followup", _create_event_fn=lambda p: "new_id"
    )
    assert wo.calendar_event_id == "new_id"


def test_update_replaces_existing_event():
    wo = _make_wo(
        follow_up_date=date(2026, 7, 15),
        calendar_event_id="old_event",
    )
    deleted = []
    wo = update_work_order_calendar(
        wo,
        "The Song",
        "followup",
        _create_event_fn=lambda p: "new_event",
        _delete_event_fn=lambda eid: deleted.append(eid),
    )
    assert "old_event" in deleted
    assert wo.calendar_event_id == "new_event"


def test_update_deadline_event():
    wo = _make_wo(deadline=date(2026, 8, 1))
    wo = update_work_order_calendar(
        wo, "The Song", "deadline", _create_event_fn=lambda p: "deadline_id"
    )
    assert wo.calendar_event_id == "deadline_id"


def test_update_calendar_integration_unavailable_sets_none():
    wo = _make_wo(follow_up_date=date(2026, 7, 15))
    wo = update_work_order_calendar(wo, "The Song", "followup")
    assert wo.calendar_event_id is None
