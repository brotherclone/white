from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Literal

from white_core.music.core.work_order import WorkOrder

log = logging.getLogger(__name__)

EventType = Literal["followup", "deadline"]


def _date_to_iso(d: date) -> str:
    """ISO-8601 date string (YYYY-MM-DD) for GCal all-day events."""
    return d.isoformat()


def build_followup_event_payload(work_order: WorkOrder, song_title: str) -> dict:
    """Return a GCal-compatible event dict for a follow-up reminder."""
    if not work_order.follow_up_date:
        raise ValueError(
            "work_order.follow_up_date is required to build a follow-up event"
        )
    reason = work_order.follow_up_reason or "Follow up"
    date_str = _date_to_iso(work_order.follow_up_date)

    end_str = _date_to_iso(work_order.follow_up_date + timedelta(days=1))
    return {
        "summary": f"Follow up: {work_order.collaborator_id} — {song_title}",
        "description": f"{reason}\nWork order: {work_order.id}",
        "start": {"date": date_str},
        "end": {"date": end_str},
        "reminders": {"useDefault": True},
    }


def build_deadline_event_payload(work_order: WorkOrder, song_title: str) -> dict:
    """Return a GCal-compatible event dict for a delivery deadline."""
    if not work_order.deadline:
        raise ValueError("work_order.deadline is required to build a deadline event")
    date_str = _date_to_iso(work_order.deadline)

    end_str = _date_to_iso(work_order.deadline + timedelta(days=1))
    return {
        "summary": f"Deadline: {work_order.collaborator_id} — {song_title}",
        "description": (
            f"Delivery deadline for {work_order.role.value} on '{song_title}'.\n"
            f"Work order: {work_order.id}"
        ),
        "start": {"date": date_str},
        "end": {"date": end_str},
        "reminders": {"useDefault": True},
    }


def create_followup_event(
    work_order: WorkOrder,
    song_title: str,
    *,
    _create_event_fn=None,
) -> str | None:
    """Create a GCal follow-up reminder. Returns the event_id or None on failure.

    _create_event_fn: injectable; should accept (payload: dict) -> str (event_id).
    When None, degrades gracefully (calendar integration not available).
    """
    if not work_order.follow_up_date:
        log.warning("create_followup_event called with no follow_up_date — skipping")
        return None
    if _create_event_fn is None:
        log.warning(
            "Calendar integration unavailable (no _create_event_fn provided) — skipping"
        )
        return None
    try:
        payload = build_followup_event_payload(work_order, song_title)
        return _create_event_fn(payload)
    except Exception as exc:
        log.warning("GCal follow-up event creation failed: %s", exc)
        return None


def create_deadline_event(
    work_order: WorkOrder,
    song_title: str,
    *,
    _create_event_fn=None,
) -> str | None:
    """Create a GCal deadline event. Returns the event_id or None on failure."""
    if not work_order.deadline:
        log.warning("create_deadline_event called with no deadline — skipping")
        return None
    if _create_event_fn is None:
        log.warning(
            "Calendar integration unavailable (no _create_event_fn provided) — skipping"
        )
        return None
    try:
        payload = build_deadline_event_payload(work_order, song_title)
        return _create_event_fn(payload)
    except Exception as exc:
        log.warning("GCal deadline event creation failed: %s", exc)
        return None


def delete_event(
    event_id: str,
    *,
    _delete_event_fn=None,
) -> None:
    """Delete a GCal event by ID. No-ops gracefully on failure."""
    if not event_id:
        return
    if _delete_event_fn is None:
        log.warning(
            "Calendar integration unavailable — cannot delete event %s", event_id
        )
        return
    try:
        _delete_event_fn(event_id)
    except Exception as exc:
        log.warning("GCal event deletion failed for %s: %s", event_id, exc)


def update_work_order_calendar(
    work_order: WorkOrder,
    song_title: str,
    event_type: EventType,
    *,
    _create_event_fn=None,
    _delete_event_fn=None,
) -> WorkOrder:
    """Replace any existing calendar event on the work order with a new one.

    Always replaces — deletes the old event first if calendar_event_id is set.
    Returns the (mutated) work order with calendar_event_id updated.
    """
    if work_order.calendar_event_id:
        delete_event(work_order.calendar_event_id, _delete_event_fn=_delete_event_fn)
        work_order.calendar_event_id = None

    if event_type == "followup":
        event_id = create_followup_event(
            work_order, song_title, _create_event_fn=_create_event_fn
        )
    else:
        event_id = create_deadline_event(
            work_order, song_title, _create_event_fn=_create_event_fn
        )

    work_order.calendar_event_id = event_id
    return work_order
