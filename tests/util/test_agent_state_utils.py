"""Tests for agent state utilities."""

from app.util.agent_state_utils import safe_add


def test_safe_add_both_none():
    """Test safe_add with both values None."""
    result = safe_add(None, None)
    assert result is None


def test_safe_add_first_none():
    """Test safe_add with first value None."""
    result = safe_add(None, [1, 2, 3])
    assert result == [1, 2, 3]


def test_safe_add_second_none():
    """Test safe_add with second value None."""
    result = safe_add([1, 2, 3], None)
    assert result == [1, 2, 3]


def test_safe_add_both_lists():
    """Test safe_add with both values as lists - uses replacement semantics."""
    result = safe_add([1, 2], [3, 4])
    # New value replaces old value (not concatenation)
    assert result == [3, 4]


def test_safe_add_empty_lists():
    """Test safe_add with empty lists."""
    result = safe_add([], [])
    assert result == []


def test_safe_add_one_empty_list():
    """Test safe_add with one empty list - replacement semantics."""
    # Empty new list replaces old list
    result = safe_add([1, 2], [])
    assert result == []

    # New list replaces empty old list
    result = safe_add([], [3, 4])
    assert result == [3, 4]


def test_safe_add_strings():
    """Test safe_add with strings - uses replacement semantics."""
    result = safe_add("hello", " world")
    # New string replaces old string (not concatenation)
    assert result == " world"


def test_safe_add_mixed_types():
    """Test safe_add with mixed types - uses replacement semantics."""
    result = safe_add([1, 2], [3.0, 4.0])
    # New list replaces old list
    assert result == [3.0, 4.0]
