from white_core.concepts.white_facet_system import (
    WhiteFacetSystem,
    get_facet_statistics,
)
from white_core.enums.white_facet import WhiteFacet


def test_white_facet_system():
    """Test basic WhiteFacetSystem functionality"""
    # Test random facet selection
    facet = WhiteFacetSystem.select_random_facet()
    assert isinstance(facet, WhiteFacet)
    assert facet in list(WhiteFacet)


def test_select_weighted_facet():
    """Test weighted facet selection"""
    facet = WhiteFacetSystem.select_weighted_facet()
    assert isinstance(facet, WhiteFacet)
    assert facet in list(WhiteFacet)


def test_get_facet_prompt():
    """Test getting facet-specific prompt"""
    facet = WhiteFacet.CATEGORICAL
    prompt = WhiteFacetSystem.get_facet_prompt(facet)
    assert isinstance(prompt, str)
    assert len(prompt) > 0


def test_build_white_initial_prompt_default():
    """Test building initial prompt with defaults"""
    prompt, selected_facet = WhiteFacetSystem.build_white_initial_prompt()
    assert isinstance(prompt, str)
    assert isinstance(selected_facet, WhiteFacet)
    assert "USER REQUEST" in prompt
    assert "YOUR TASK" in prompt


def test_build_white_initial_prompt_with_input():
    """Test building initial prompt with custom user input"""
    user_input = "Create a song about dreams"
    prompt, facet = WhiteFacetSystem.build_white_initial_prompt(user_input=user_input)
    assert user_input in prompt
    assert isinstance(facet, WhiteFacet)


def test_build_white_initial_prompt_specific_facet():
    """Test building prompt with specific facet"""
    specific_facet = WhiteFacet.RELATIONAL
    prompt, facet = WhiteFacetSystem.build_white_initial_prompt(a_facet=specific_facet)
    assert facet == specific_facet
    assert specific_facet.value.upper() in prompt


def test_build_white_initial_prompt_no_weights():
    """Test building prompt with uniform random selection"""
    prompt, facet = WhiteFacetSystem.build_white_initial_prompt(use_weights=False)
    assert isinstance(facet, WhiteFacet)
    assert isinstance(prompt, str)


def test_facet_prompt_includes_facet_mode():
    """Test that generated prompt includes facet mode information"""
    facet = WhiteFacet.TECHNICAL
    prompt, returned_facet = WhiteFacetSystem.build_white_initial_prompt(a_facet=facet)
    assert returned_facet == facet
    assert facet.value.upper() in prompt


# ---------------------------------------------------------------------------
# log_facet_selection
# ---------------------------------------------------------------------------


def test_log_facet_selection_returns_dict():
    facet = WhiteFacet.CATEGORICAL
    result = WhiteFacetSystem.log_facet_selection(facet)
    assert isinstance(result, dict)


def test_log_facet_selection_has_expected_keys():
    result = WhiteFacetSystem.log_facet_selection(WhiteFacet.RELATIONAL)
    assert "facet" in result
    assert "description" in result
    assert "example_style" in result


def test_log_facet_selection_facet_value():
    facet = WhiteFacet.PHENOMENOLOGICAL
    result = WhiteFacetSystem.log_facet_selection(facet)
    assert result["facet"] == facet.value


def test_log_facet_selection_all_facets():
    for facet in WhiteFacet:
        result = WhiteFacetSystem.log_facet_selection(facet)
        assert result["facet"] == facet.value
        assert isinstance(result["description"], str)
        assert isinstance(result["example_style"], str)


# ---------------------------------------------------------------------------
# get_facet_statistics
# ---------------------------------------------------------------------------


def test_get_facet_statistics_keys():
    stats = get_facet_statistics()
    assert "total_facets" in stats
    assert "facets" in stats
    assert "has_system_prompts" in stats
    assert "has_descriptions" in stats
    assert "has_examples" in stats


def test_get_facet_statistics_total_count():
    stats = get_facet_statistics()
    assert stats["total_facets"] == len(list(WhiteFacet))


def test_get_facet_statistics_all_data_present():
    stats = get_facet_statistics()
    assert stats["has_system_prompts"] is True
    assert stats["has_descriptions"] is True
    assert stats["has_examples"] is True


def test_get_facet_statistics_facets_list():
    stats = get_facet_statistics()
    assert set(stats["facets"]) == {f.value for f in WhiteFacet}


# ---------------------------------------------------------------------------
# get_facet_prompt — all facets
# ---------------------------------------------------------------------------


def test_get_facet_prompt_all_facets():
    for facet in WhiteFacet:
        prompt = WhiteFacetSystem.get_facet_prompt(facet)
        assert isinstance(prompt, str)
        assert len(prompt) > 0


# ---------------------------------------------------------------------------
# select_weighted_facet — distribution sanity
# ---------------------------------------------------------------------------


def test_select_weighted_facet_returns_valid_facet():
    for _ in range(20):
        facet = WhiteFacetSystem.select_weighted_facet()
        assert isinstance(facet, WhiteFacet)


def test_select_random_facet_covers_all_over_many_trials():
    import random

    random.seed(0)
    seen = set()
    for _ in range(200):
        seen.add(WhiteFacetSystem.select_random_facet())
    assert seen == set(WhiteFacet)
