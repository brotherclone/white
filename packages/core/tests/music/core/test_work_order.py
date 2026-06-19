from datetime import date

from white_core.enums.budget_status import BudgetStatus
from white_core.enums.chain_artifact_type import ChainArtifactType
from white_core.enums.collaborator_platform import CollaboratorPlatform
from white_core.enums.collaborator_role import CollaboratorRole
from white_core.enums.work_order_status import WorkOrderStatus
from white_core.music.core.work_order import RoyaltySplit, WorkOrder


def _make_work_order(**kwargs) -> WorkOrder:
    defaults = {
        "id": "wo_kate_archivist",
        "song_slug": "the_archivists_rebellion",
        "collaborator_id": "kate_koherence",
        "role": CollaboratorRole.VOCALIST,
    }
    defaults.update(kwargs)
    return WorkOrder(**defaults)


def test_minimal_work_order():
    wo = _make_work_order()
    assert wo.status == WorkOrderStatus.DRAFT
    assert wo.platform == CollaboratorPlatform.DIRECT
    assert wo.budget_status == BudgetStatus.PENDING
    assert wo.artifact_types == []
    assert wo.sections == []
    assert wo.royalty_split is None
    assert wo.calendar_event_id is None


def test_work_order_with_musical_context():
    wo = _make_work_order(
        key="D minor",
        bpm=112,
        time_signature="4/4",
        sections=["verse", "chorus", "bridge"],
        creative_direction="Melancholic, sparse, haunting vocal — think Kate Bush meets Grouper.",
        part_notes="Verse: one-take intimacy. Chorus: layer 3 takes.",
    )
    assert wo.bpm == 112
    assert len(wo.sections) == 3
    assert "Kate Bush" in wo.creative_direction


def test_work_order_artifacts():
    wo = _make_work_order(
        artifact_types=[
            ChainArtifactType.CHROMATIC_BRIEF,
            ChainArtifactType.MELODY_MIDI_STEM,
        ]
    )
    assert ChainArtifactType.CHROMATIC_BRIEF in wo.artifact_types
    assert len(wo.artifact_types) == 2


def test_work_order_budget_fields():
    wo = _make_work_order(
        budget_agreed=250.0,
        budget_paid=0.0,
        budget_currency="USD",
        budget_status=BudgetStatus.AGREED,
    )
    assert wo.budget_agreed == 250.0
    assert wo.budget_status == BudgetStatus.AGREED


def test_work_order_scheduling():
    wo = _make_work_order(
        deadline=date(2026, 8, 1),
        follow_up_date=date(2026, 7, 1),
        follow_up_reason="Kate back from tour",
    )
    assert wo.deadline == date(2026, 8, 1)
    assert wo.follow_up_reason == "Kate back from tour"


def test_royalty_split_basic():
    rs = RoyaltySplit(
        collaborator_id="kate_koherence",
        song_slug="the_archivists_rebellion",
        mechanical_pct=10.0,
        performance_pct=15.0,
    )
    assert rs.mechanical_pct == 10.0
    assert rs.sync_pct == 0.0


def test_work_order_with_royalty_split():
    rs = RoyaltySplit(
        collaborator_id="kate_koherence",
        song_slug="the_archivists_rebellion",
        performance_pct=10.0,
    )
    wo = _make_work_order(royalty_split=rs)
    assert wo.royalty_split is not None
    assert wo.royalty_split.performance_pct == 10.0


def test_work_order_calendar_event():
    wo = _make_work_order(calendar_event_id="gcal_event_abc123")
    assert wo.calendar_event_id == "gcal_event_abc123"


def test_work_order_status_transition():
    wo = _make_work_order()
    wo.status = WorkOrderStatus.SENT
    assert wo.status == WorkOrderStatus.SENT


def test_work_order_serialises_enums_as_strings():
    wo = _make_work_order(
        artifact_types=[ChainArtifactType.CHROMATIC_BRIEF],
        status=WorkOrderStatus.IN_PROGRESS,
    )
    d = wo.model_dump()
    assert d["status"] == "in_progress"
    assert d["role"] == "vocalist"
    assert d["artifact_types"][0] == "chromatic_brief"


def test_work_order_roundtrip_json():
    wo = _make_work_order(
        key="F# minor",
        bpm=90,
        deadline=date(2026, 9, 15),
        artifact_types=[ChainArtifactType.PRODUCTION_PLAN_ARTIFACT],
    )
    reloaded = WorkOrder.model_validate_json(wo.model_dump_json())
    assert reloaded.id == wo.id
    assert reloaded.bpm == 90
    assert reloaded.deadline == date(2026, 9, 15)
    assert reloaded.artifact_types[0] == ChainArtifactType.PRODUCTION_PLAN_ARTIFACT
