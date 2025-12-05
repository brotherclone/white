from app.structures.artifacts.arbitrarys_survey_artifact import ArbitrarysSurveyArtifact
from app.structures.artifacts.base_artifact import ChainArtifact


def test_inheritance():
    assert issubclass(ArbitrarysSurveyArtifact, ChainArtifact)


# ToDo: Add additional tests for ArbitrarysSurveyArtifact fields and validation logic
