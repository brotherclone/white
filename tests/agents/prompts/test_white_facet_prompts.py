from app.agents.prompts.white_facet_prompts import FACET_SYSTEM_PROMPTS
from app.structures.enums.white_facet import WhiteFacet


def test_all_facets_present():
    missing = [f for f in WhiteFacet if f not in FACET_SYSTEM_PROMPTS]
    assert missing == [], f"Missing prompts for facets: {missing}"


def test_prompts_are_strings_and_non_empty():
    for facet, prompt in FACET_SYSTEM_PROMPTS.items():
        assert isinstance(prompt, str), f"Prompt for {facet} is not a str"
        assert prompt.strip(), f"Prompt for {facet} is empty or whitespace"


def test_prompts_start_with_expected_phrase():
    for facet in WhiteFacet:
        prompt = FACET_SYSTEM_PROMPTS[facet]
        assert prompt.lstrip().startswith(
            f"You are operating in {facet.name} mode"
        ), f"Prompt for {facet} does not start with expected phrase"
