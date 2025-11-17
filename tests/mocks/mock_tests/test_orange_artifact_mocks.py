from pathlib import Path

import yaml

from app.structures.artifacts.newspaper_artifact import NewspaperArtifact


def test_newspaper_mocks():
    path = (
        Path(__file__).resolve().parents[3]
        / "tests"
        / "mocks"
        / "orange_base_story_mock.yml"
    )
    with path.open("r") as f:
        data = yaml.safe_load(f)
    article = NewspaperArtifact(**data)
    assert isinstance(article, NewspaperArtifact)
    assert (
        article.headline == "Local Band Vanishes After Practice Session in Vernon Woods"
    )
    assert article.date == "1987-10-15"
    assert article.source == "Sussex County Independent"
    assert article.location == "Vernon Township, NJ"
    assert article.tags == ["rock_bands", "unexplained", "youth_crime"]


def test_newspaper_article_page():
    path = (
        Path(__file__).resolve().parents[3]
        / "tests"
        / "mocks"
        / "orange_base_story_mock.yml"
    )
    with path.open("r") as f:
        data = yaml.safe_load(f)
    article = NewspaperArtifact(**data)
    assert article.page.chain_artifact_file_type.value == "md"
    assert article.page.rainbow_color.mnemonic_character_value == "O"
    assert article.page.rainbow_color.temporal_mode.value == "Past"
    assert article.page.rainbow_color.objectional_mode.value == "Thing"
    assert article.page.rainbow_color.ontological_mode[0].value == "Known"
    assert article.page.artifact_id == "019"
    assert article.page.artifact_name == "article"
    assert article.page.file_name == "019_orange_article.md"
