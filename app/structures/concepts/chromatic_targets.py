"""
Canonical CHROMATIC_TARGETS — derived at import time from the_rainbow_table_colors.

Single source of truth for per-color probability vectors used by all scoring,
drift-reporting, and CDM inference code in the pipeline.

Vector ordering:
  temporal    → [past, present, future]
  spatial     → [thing, place, person]
  ontological → [imagined, forgotten, known]
"""

from app.structures.concepts.rainbow_table_color import (
    RainbowColorObjectionalMode,
    RainbowColorOntologicalMode,
    RainbowColorTemporalMode,
    the_rainbow_table_colors,
)

TEMPORAL_MODES: tuple[str, ...] = ("past", "present", "future")
SPATIAL_MODES: tuple[str, ...] = ("thing", "place", "person")
ONTOLOGICAL_MODES: tuple[str, ...] = ("imagined", "forgotten", "known")

_UNIFORM = [1 / 3, 1 / 3, 1 / 3]

_TEMPORAL_MAP = {
    RainbowColorTemporalMode.PAST: [0.8, 0.1, 0.1],
    RainbowColorTemporalMode.PRESENT: [0.1, 0.8, 0.1],
    RainbowColorTemporalMode.FUTURE: [0.1, 0.1, 0.8],
}

_SPATIAL_MAP = {
    RainbowColorObjectionalMode.THING: [0.8, 0.1, 0.1],
    RainbowColorObjectionalMode.PLACE: [0.1, 0.8, 0.1],
    RainbowColorObjectionalMode.PERSON: [0.1, 0.1, 0.8],
}

_ONTOLOGICAL_IDX = {
    RainbowColorOntologicalMode.IMAGINED: 0,
    RainbowColorOntologicalMode.FORGOTTEN: 1,
    RainbowColorOntologicalMode.KNOWN: 2,
}


def _ontological_vector(modes) -> list[float]:
    if not modes:
        return list(_UNIFORM)
    if len(modes) == 1:
        vec = [0.1, 0.1, 0.1]
        vec[_ONTOLOGICAL_IDX[modes[0]]] = 0.8
        return vec
    if len(modes) == 2:
        vec = [0.1, 0.1, 0.1]
        for m in modes:
            vec[_ONTOLOGICAL_IDX[m]] = 0.4
        return vec
    return list(_UNIFORM)


def _build_chromatic_targets() -> dict[str, dict[str, list[float]]]:
    targets: dict[str, dict[str, list[float]]] = {}
    for color in the_rainbow_table_colors.values():
        targets[color.color_name] = {
            "temporal": list(_TEMPORAL_MAP.get(color.temporal_mode, _UNIFORM)),
            "spatial": list(_SPATIAL_MAP.get(color.objectional_mode, _UNIFORM)),
            "ontological": _ontological_vector(color.ontological_mode),
        }
    return targets


CHROMATIC_TARGETS: dict[str, dict[str, list[float]]] = _build_chromatic_targets()
