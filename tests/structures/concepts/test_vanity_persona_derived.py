import pytest

from app.structures.enums.vanity_interviewer_type import VanityInterviewerType
from app.structures.concepts.vanity_persona import VanityPersona


@pytest.fixture
def make_vanity_persona():
    def _make(**overrides):
        defaults = dict(
            first_name="Alice",
            last_name="Smith",
            publication="MocknRoll",
            interviewer_type=VanityInterviewerType.HOSTILE_SKEPTICAL,
        )
        defaults.update(overrides)
        return VanityPersona(**defaults)

    return _make


@pytest.mark.parametrize(
    "itype, expected_stance",
    [
        (VanityInterviewerType.HOSTILE_SKEPTICAL, "This is pretentious nonsense"),
        (VanityInterviewerType.VANITY_PRESSING_FAN, "Why'd you abandon accessibility?"),
        (VanityInterviewerType.EXPERIMENTAL_PURIST, "You sold out"),
        (VanityInterviewerType.EARNEST_BUT_WRONG, "I really want to understand!"),
    ],
)
def test_interviewer_type_derived_fields(make_vanity_persona, itype, expected_stance):
    p = make_vanity_persona(interviewer_type=itype, publication="FixedPub")

    assert p.interviewer_type == itype
    assert p.stance == expected_stance
    assert isinstance(p.tactics, list) and len(p.tactics) >= 5
    assert isinstance(p.example_questions, list) and len(p.example_questions) >= 5
    assert p.publication == "FixedPub"


def test_random_choice_sets_interviewer_type(monkeypatch):
    # Force random.choice to return EXPERIMENTAL_PURIST when selecting an enum
    def fake_choice(seq):
        # If seq contains enum members, return an enum; otherwise return a deterministic string
        first = seq[0] if seq else None
        if hasattr(first, "value"):
            return VanityInterviewerType.EXPERIMENTAL_PURIST
        return "ChosenPub"

    monkeypatch.setattr("random.choice", fake_choice)

    # Construct without interviewer_type or publication
    p = VanityPersona()
    assert p.interviewer_type == VanityInterviewerType.EXPERIMENTAL_PURIST
    assert p.stance == "You sold out"


def test_publication_selection_by_type(monkeypatch, make_vanity_persona):
    # Force random.choice for publication selection to return a known string
    monkeypatch.setattr("random.choice", lambda seq: "ChosenPublication")

    p = VanityPersona(
        interviewer_type=VanityInterviewerType.EARNEST_BUT_WRONG, publication=None
    )
    assert p.publication == "ChosenPublication"

    p2 = VanityPersona(
        interviewer_type=VanityInterviewerType.VANITY_PRESSING_FAN, publication=None
    )
    assert p2.publication == "ChosenPublication"

    p3 = VanityPersona(
        interviewer_type=VanityInterviewerType.EXPERIMENTAL_PURIST, publication=None
    )
    assert p3.publication == "ChosenPublication"

    p4 = VanityPersona(
        interviewer_type=VanityInterviewerType.HOSTILE_SKEPTICAL, publication=None
    )
    assert p4.publication == "ChosenPublication"
