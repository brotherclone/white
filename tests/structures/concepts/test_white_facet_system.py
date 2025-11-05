from app.structures.concepts.white_facet_system import WhiteFacetSystem
from app.structures.enums.white_facet import WhiteFacet


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
