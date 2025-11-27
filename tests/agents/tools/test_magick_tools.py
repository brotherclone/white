import random

from app.agents.tools.magick_tools import SigilTools
from app.structures.enums.gnosis_method import GnosisMethod


def test_paranoia_level_in_range():
    s = SigilTools()
    assert 7 <= s.paranoia_level <= 10


def test_create_statement_of_intent_removes_negative_and_adds_starter(monkeypatch):
    # Make starter deterministic
    monkeypatch.setattr(random, "choice", lambda seq: "IT IS MY WILL TO EXPOSE")
    raw = "I don't want to be poor or have no money"
    stmt = SigilTools.create_statement_of_intent(raw, agent_personality=True)
    assert "DON'T" not in stmt
    assert "NO" not in stmt
    assert stmt.startswith("IT IS MY WILL TO EXPOSE")
    assert stmt == stmt.upper()


def test_create_statement_of_intent_no_personality_uses_default():
    raw = "gain wealth"
    stmt = SigilTools.create_statement_of_intent(raw, agent_personality=False)
    assert stmt.startswith("THIS IS MY WILL TO")
    assert stmt == stmt.upper()


def test_reduce_to_unique_letters_ignores_non_letters_and_preserves_order():
    out = SigilTools.reduce_to_unique_letters("Hello, hello!")
    assert out == ["H", "E", "L", "O"]


def test_generate_word_method_sigil_produces_expected_components(monkeypatch):
    # Choose a deterministic combination method
    monkeypatch.setattr(
        random, "choice", lambda seq: "connect sequentially with flowing lines"
    )
    statement = "A b a z"
    glyph_desc, comps = SigilTools().generate_word_method_sigil(statement)
    assert "Combine 3 forms" in glyph_desc
    # Check mapping for a few letters
    assert any(c.startswith("A:triangle") for c in comps)
    assert any(c.startswith("B:double curve") for c in comps)
    assert any(c.startswith("Z:zigzag") for c in comps)


def test_generate_pictorial_sigil_with_explicit_symbols(monkeypatch):
    monkeypatch.setattr(
        random, "choice", lambda seq: "layered with transparency effects"
    )
    s = SigilTools.generate_pictorial_sigil(
        "ignored", symbolic_elements=["heart", "star"]
    )
    assert "heart" in s
    assert "star" in s
    assert "layered with transparency effects" in s


def test_generate_pictorial_sigil_from_keywords(monkeypatch):
    # Ensure arrangement is deterministic
    monkeypatch.setattr(random, "choice", lambda seq: "rotated around central axis")
    s = SigilTools.generate_pictorial_sigil("I want money and power")
    # keyword 'money' should surface 'coin' symbol
    assert "coin" in s or "golden ratio" in s
    assert "rotated around central axis" in s


def test_generate_mantric_sigil_builds_mantra_and_instruction(monkeypatch):
    # Force vowel insertion to be 'E' for deterministic fragment
    monkeypatch.setattr(random, "choice", lambda seq: "E")
    mantra, instr = SigilTools.generate_mantric_sigil("BCDFG")
    # mantra should be non-empty and appear in instruction
    assert isinstance(mantra, str) and len(mantra) > 0
    assert mantra in instr


def test_generate_mantric_sigil_defaults_when_no_consonants():
    mantra, instr = SigilTools.generate_mantric_sigil("aeiou")
    # Should fall back to default fragments if no consonants found
    assert mantra in ("zos", "kia", "aos")
    assert mantra in instr


def test_generate_alphabet_of_desire_symbol_stores_description_consistently():
    s = SigilTools()
    desc = s.generate_alphabet_of_desire_symbol("Money")
    assert "Personal symbol for 'Money':" in desc
    assert s.alphabet_of_desire.get("Money") == desc
    # repeated call yields same stored string
    desc2 = s.generate_alphabet_of_desire_symbol("Money")
    assert desc2 == s.alphabet_of_desire["Money"]


def test_charge_sigil_returns_method_specific_instruction(monkeypatch):
    # Deterministic duration and chosen method
    monkeypatch.setattr(random, "randint", lambda a, b: 10)
    monkeypatch.setattr(random, "choice", lambda seq: GnosisMethod.MEDITATION)
    out = SigilTools.charge_sigil()
    assert "Deep meditation" in out or "meditation" in out.lower()
    assert "10" in out
