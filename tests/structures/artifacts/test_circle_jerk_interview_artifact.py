from app.structures.artifacts.circle_jerk_interview_artifact import (
    CircleJerkInterviewArtifact,
)
from app.structures.concepts.vanity_interview_question import VanityInterviewQuestion
from app.structures.concepts.vanity_interview_response import VanityInterviewResponse


def make_artifact():
    """Create a CircleJerkInterviewArtifact with explicit interviewer properties.

    We're intentionally repeating the VanityPersona-derived values here (stance, approach, etc.)
    to keep the test self-contained and avoid relying on shared fixtures.
    """
    questions = [
        VanityInterviewQuestion(number=1, question="So what's this about?"),
        VanityInterviewQuestion(number=2, question="Is this pretentious?"),
        VanityInterviewQuestion(number=3, question="Who's your influence?"),
    ]

    responses = [
        VanityInterviewResponse(question_number=1, response="It's about the sound."),
        VanityInterviewResponse(question_number=2, response="No, it's vital."),
        VanityInterviewResponse(question_number=3, response="Too many to name."),
    ]

    return CircleJerkInterviewArtifact(
        thread_id="test",
        interviewer_name="Alice Smith",
        publication="MocknRoll",
        interviewer_type="hostile_skeptical",
        stance="This is pretentious nonsense",
        questions=questions,
        responses=responses,
        was_human_interview=False,
    )


def test_init():
    """Test basic CircleJerkInterviewArtifact instantiation"""
    artifact = make_artifact()
    assert artifact.thread_id == "test"
    assert artifact.interviewer_name == "Alice Smith"
    assert artifact.publication == "MocknRoll"
    assert artifact.stance == "This is pretentious nonsense"
    assert isinstance(artifact.questions, list) and len(artifact.questions) == 3
    assert isinstance(artifact.responses, list) and len(artifact.responses) == 3


def test_flatten():
    artifact = make_artifact()
    flat = artifact.flatten()
    # Basic keys should be present
    assert flat["interviewer_name"] == "Alice Smith"
    assert flat["publication"] == "MocknRoll"
    assert flat["stance"] == "This is pretentious nonsense"
    assert isinstance(flat["questions"], list)
    assert isinstance(flat["responses"], list)
    # Questions/responses should have the expected shapes
    q0 = flat["questions"][0]
    r0 = flat["responses"][0]
    assert q0["number"] == 1
    assert q0["question"] == "So what's this about?"
    assert r0["response"] == "It's about the sound."


def test_to_markdown_and_for_prompt():
    artifact = make_artifact()
    md = artifact.to_markdown()
    # Contains interviewer and publication
    assert "**Interviewer:** Alice Smith" in md
    assert "**Publication:** MocknRoll" in md
    # Contains at least one question and response block
    assert "## Question 1" in md
    assert "**Alice Smith:** So what's this about?" in md
    assert "**Walsh:** It's about the sound." in md

    prompt = artifact.for_prompt()
    assert "# Interview with Alice Smith (MocknRoll)" in prompt
    assert "Stance: This is pretentious nonsense" in prompt
    assert "Q1: So what's this about?" in prompt
    assert "A1: It's about the sound." in prompt
