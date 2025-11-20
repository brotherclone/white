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
    assert article.chain_artifact_file_type.value == "md"
    assert article.rainbow_color_mnemonic_character_value == "0"
    assert article.artifact_name == "story"
    assert article.chain_artifact_type == "newspaper_article"
