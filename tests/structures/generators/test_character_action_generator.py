"""Tests for CharacterActionGenerator including character state mutations"""

from app.structures.generators.character_action_generator import (
    CharacterActionGenerator,
)


def test_generator_initialization():
    """Test that CharacterActionGenerator can be instantiated"""
    generator = CharacterActionGenerator()
    assert generator is not None


def test_generate_outcome_with_different_character_counts():
    """Test that generate_outcome returns strings for different character counts"""
    generator = CharacterActionGenerator()

    # Low chaos (1 character)
    outcome_low = generator._generate_outcome(1)
    assert isinstance(outcome_low, str)
    assert len(outcome_low) > 0

    # Medium chaos (2-3 characters)
    outcome_medium = generator._generate_outcome(2)
    assert isinstance(outcome_medium, str)
    assert len(outcome_medium) > 0

    # High chaos (4+ characters)
    outcome_high = generator._generate_outcome(4)
    assert isinstance(outcome_high, str)
    assert len(outcome_high) > 0


def test_apply_encounter_mutations_returns_copy():
    """Test that _apply_encounter_mutations returns a different object"""
    generator = CharacterActionGenerator()

    # Create a minimal mock character for testing
    class MockCharacter:
        def __init__(self):
            self.name = "Test"
            self.disposition = type("obj", (object,), {"disposition": "Angry"})()
            self.on_max = 10
            self.on_current = 5
            self.off_max = 10
            self.off_current = 5
            self.portrait = None

    character = MockCharacter()

    mutated = generator._apply_encounter_mutations(character)

    # Should be a different object
    assert mutated is not character
    # Stats should potentially have changed
    assert hasattr(mutated, "on_current")
    assert hasattr(mutated, "off_current")


def test_apply_encounter_mutations_respects_boundaries():
    """Test that mutations don't go below 0"""
    generator = CharacterActionGenerator()

    class MockCharacter:
        def __init__(self):
            self.name = "Test"
            self.disposition = type("obj", (object,), {"disposition": "Sick"})()
            self.on_max = 10
            self.on_current = 0  # Already at minimum
            self.off_max = 10
            self.off_current = 0
            self.portrait = None

    character = MockCharacter()
    mutated = generator._apply_encounter_mutations(character)

    # Should not go below 0
    assert mutated.on_current >= 0
    assert mutated.off_current >= 0


def test_apply_encounter_mutations_respects_max_boundaries():
    """Test that mutations don't exceed max + 10"""
    generator = CharacterActionGenerator()

    class MockCharacter:
        def __init__(self):
            self.name = "Test"
            self.disposition = type("obj", (object,), {"disposition": "Curious"})()
            self.on_max = 10
            self.on_current = 5
            self.off_max = 8
            self.off_current = 4
            self.portrait = None

    character = MockCharacter()
    mutated = generator._apply_encounter_mutations(character)

    # Should not exceed max + 10
    assert mutated.on_current <= mutated.on_max + 10
    assert mutated.off_current <= mutated.off_max + 10


def test_apply_encounter_mutations_angry_disposition():
    """Test mutation effects for Angry disposition"""
    generator = CharacterActionGenerator()

    class MockCharacter:
        def __init__(self):
            self.name = "Angry Character"
            self.disposition = type("obj", (object,), {"disposition": "Angry"})()
            self.on_max = 10
            self.on_current = 8
            self.off_max = 10
            self.off_current = 2
            self.portrait = None

    character = MockCharacter()
    mutated = generator._apply_encounter_mutations(character)

    # Angry should affect stats (may lose ON, gain OFF)
    assert isinstance(mutated.on_current, int)
    assert isinstance(mutated.off_current, int)
    assert 0 <= mutated.on_current <= mutated.on_max + 10
    assert 0 <= mutated.off_current <= mutated.off_max + 10


def test_apply_encounter_mutations_curious_disposition():
    """Test mutation effects for Curious disposition"""
    generator = CharacterActionGenerator()

    class MockCharacter:
        def __init__(self):
            self.name = "Curious Character"
            self.disposition = type("obj", (object,), {"disposition": "Curious"})()
            self.on_max = 10
            self.on_current = 5
            self.off_max = 10
            self.off_current = 5
            self.portrait = None

    character = MockCharacter()
    mutated = generator._apply_encounter_mutations(character)

    # Curious should affect stats (may gain ON, lose OFF)
    assert 0 <= mutated.on_current <= mutated.on_max + 10
    assert 0 <= mutated.off_current <= mutated.off_max + 10


def test_apply_encounter_mutations_unknown_disposition():
    """Test that unknown dispositions get default (0,0) change"""
    generator = CharacterActionGenerator()

    class MockCharacter:
        def __init__(self):
            self.name = "Unknown Character"
            self.disposition = type(
                "obj", (object,), {"disposition": "UnknownDisposition"}
            )()
            self.on_max = 10
            self.on_current = 5
            self.off_max = 10
            self.off_current = 5
            self.portrait = None

    character = MockCharacter()
    mutated = generator._apply_encounter_mutations(character)

    # Unknown disposition gets (0,0) + variance, should still be in valid range
    assert 0 <= mutated.on_current <= mutated.on_max + 10
    assert 0 <= mutated.off_current <= mutated.off_max + 10
