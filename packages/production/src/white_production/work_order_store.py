from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import yaml

from white_core.music.core.work_order import WorkOrder

_WORK_ORDERS_DIR = "work_orders"


def _wo_path(production_dir: Path, collaborator_id: str) -> Path:
    return production_dir / _WORK_ORDERS_DIR / f"{collaborator_id}.yml"


def load_work_order(production_dir: Path, collaborator_id: str) -> WorkOrder:
    path = _wo_path(production_dir, collaborator_id)
    if not path.exists():
        raise FileNotFoundError(
            f"No work order found for '{collaborator_id}' in {production_dir}"
        )
    raw = yaml.safe_load(path.read_text()) or {}
    return WorkOrder.model_validate(raw)


def save_work_order(production_dir: Path, work_order: WorkOrder) -> None:
    work_order.updated_at = datetime.now(timezone.utc)
    path = _wo_path(production_dir, work_order.collaborator_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = work_order.model_dump(mode="json")
    path.write_text(
        yaml.dump(payload, allow_unicode=True, sort_keys=False, width=float("inf"))
    )


def list_work_orders(production_dir: Path) -> list[WorkOrder]:
    wo_dir = production_dir / _WORK_ORDERS_DIR
    if not wo_dir.exists():
        return []
    result = []
    for yml in sorted(wo_dir.glob("*.yml")):
        raw = yaml.safe_load(yml.read_text()) or {}
        result.append(WorkOrder.model_validate(raw))
    return result
