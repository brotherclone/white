"""Tests for vanity interview question structures."""

import pytest
from pydantic import ValidationError

from app.structures.concepts.vanity_interview_question import (
    VanityInterviewQuestion,
    VanityInterviewQuestionOutput,
)


class TestVanityInterviewQuestion:
    """Test VanityInterviewQuestion model."""

    def test_create_valid_question(self):
        """Test creating a valid interview question."""
        question = VanityInterviewQuestion(
            number=1, question="What inspired you to create this album?"
        )

        assert question.number == 1
        assert question.question == "What inspired you to create this album?"

    def test_question_numbers_1_to_3(self):
        """Test questions with numbers 1-3."""
        for num in [1, 2, 3]:
            question = VanityInterviewQuestion(number=num, question=f"Question {num}")
            assert question.number == num

    def test_question_with_various_text(self):
        """Test questions with various text content."""
        questions = [
            "How would you describe your creative process?",
            "What influences shaped this work?",
            "Can you elaborate on the themes explored?",
        ]

        for i, q_text in enumerate(questions, 1):
            question = VanityInterviewQuestion(number=i, question=q_text)
            assert question.question == q_text

    def test_question_with_long_text(self):
        """Test question with long text."""
        long_question = "In your opinion, considering the broader context of experimental music and the evolution of sound design over the past several decades, how would you characterize the unique sonic palette and compositional approach demonstrated in this particular work?"

        question = VanityInterviewQuestion(number=1, question=long_question)

        assert question.question == long_question

    def test_question_with_empty_string(self):
        """Test question with empty string is allowed."""
        question = VanityInterviewQuestion(number=1, question="")
        assert question.question == ""

    def test_question_with_special_characters(self):
        """Test question with special characters."""
        special_question = 'What\'s your view on "experimental" music? (And why?)'

        question = VanityInterviewQuestion(number=2, question=special_question)

        assert question.question == special_question

    def test_question_missing_number(self):
        """Test that question requires number field."""
        with pytest.raises(ValidationError) as exc_info:
            VanityInterviewQuestion(question="Test question")

        assert "number" in str(exc_info.value)

    def test_question_missing_question(self):
        """Test that question requires question field."""
        with pytest.raises(ValidationError) as exc_info:
            VanityInterviewQuestion(number=1)

        assert "question" in str(exc_info.value)

    def test_question_with_non_integer_number(self):
        """Test that number must be an integer."""
        with pytest.raises(ValidationError):
            VanityInterviewQuestion(number="one", question="Test")

    def test_question_serialization(self):
        """Test question serialization to dict."""
        question = VanityInterviewQuestion(
            number=2, question="How do you approach composition?"
        )

        data = question.model_dump()

        assert data["number"] == 2
        assert data["question"] == "How do you approach composition?"

    def test_question_from_dict(self):
        """Test creating question from dictionary."""
        data = {"number": 3, "question": "What's next for your artistic journey?"}

        question = VanityInterviewQuestion(**data)

        assert question.number == 3
        assert question.question == "What's next for your artistic journey?"


class TestVanityInterviewQuestionOutput:
    """Test VanityInterviewQuestionOutput model."""

    def test_create_valid_output(self):
        """Test creating valid question output."""
        questions = [
            VanityInterviewQuestion(number=1, question="Question 1"),
            VanityInterviewQuestion(number=2, question="Question 2"),
            VanityInterviewQuestion(number=3, question="Question 3"),
        ]

        output = VanityInterviewQuestionOutput(questions=questions)

        assert len(output.questions) == 3
        assert output.questions[0].number == 1
        assert output.questions[2].question == "Question 3"

    def test_output_with_three_questions(self):
        """Test output typically has three questions."""
        output = VanityInterviewQuestionOutput(
            questions=[
                VanityInterviewQuestion(number=1, question="Q1"),
                VanityInterviewQuestion(number=2, question="Q2"),
                VanityInterviewQuestion(number=3, question="Q3"),
            ]
        )

        assert len(output.questions) == 3

    def test_output_with_zero_questions(self):
        """Test output can be created with zero questions."""
        output = VanityInterviewQuestionOutput(questions=[])
        assert len(output.questions) == 0

    def test_output_with_one_question(self):
        """Test output with single question."""
        output = VanityInterviewQuestionOutput(
            questions=[VanityInterviewQuestion(number=1, question="Solo question")]
        )

        assert len(output.questions) == 1

    def test_output_with_more_than_three_questions(self):
        """Test output can have more than three questions."""
        questions = [
            VanityInterviewQuestion(number=i, question=f"Question {i}")
            for i in range(1, 6)
        ]

        output = VanityInterviewQuestionOutput(questions=questions)

        assert len(output.questions) == 5

    def test_output_missing_questions_field(self):
        """Test that output requires questions field."""
        with pytest.raises(ValidationError) as exc_info:
            VanityInterviewQuestionOutput()

        assert "questions" in str(exc_info.value)

    def test_output_with_invalid_question_type(self):
        """Test that questions must be VanityInterviewQuestion objects."""
        with pytest.raises(ValidationError):
            VanityInterviewQuestionOutput(questions=["not", "valid", "questions"])

    def test_output_serialization(self):
        """Test output serialization to dict."""
        output = VanityInterviewQuestionOutput(
            questions=[
                VanityInterviewQuestion(number=1, question="First"),
                VanityInterviewQuestion(number=2, question="Second"),
            ]
        )

        data = output.model_dump()

        assert "questions" in data
        assert len(data["questions"]) == 2
        assert data["questions"][0]["number"] == 1
        assert data["questions"][1]["question"] == "Second"

    def test_output_from_dict(self):
        """Test creating output from dictionary."""
        data = {
            "questions": [
                {"number": 1, "question": "Q1"},
                {"number": 2, "question": "Q2"},
                {"number": 3, "question": "Q3"},
            ]
        }

        output = VanityInterviewQuestionOutput(**data)

        assert len(output.questions) == 3
        assert output.questions[0].number == 1
        assert output.questions[2].question == "Q3"

    def test_output_questions_ordering(self):
        """Test that questions maintain their order."""
        questions = [
            VanityInterviewQuestion(number=3, question="Third"),
            VanityInterviewQuestion(number=1, question="First"),
            VanityInterviewQuestion(number=2, question="Second"),
        ]

        output = VanityInterviewQuestionOutput(questions=questions)

        # Order should be preserved as given
        assert output.questions[0].number == 3
        assert output.questions[1].number == 1
        assert output.questions[2].number == 2

    def test_output_with_duplicate_numbers(self):
        """Test output allows duplicate question numbers."""
        questions = [
            VanityInterviewQuestion(number=1, question="First Q1"),
            VanityInterviewQuestion(number=1, question="Second Q1"),
        ]

        output = VanityInterviewQuestionOutput(questions=questions)

        assert len(output.questions) == 2
        assert output.questions[0].number == output.questions[1].number == 1

    def test_output_iteration(self):
        """Test iterating over questions in output."""
        output = VanityInterviewQuestionOutput(
            questions=[
                VanityInterviewQuestion(number=i, question=f"Q{i}") for i in range(1, 4)
            ]
        )

        question_numbers = [q.number for q in output.questions]
        assert question_numbers == [1, 2, 3]

    def test_output_access_by_index(self):
        """Test accessing questions by index."""
        output = VanityInterviewQuestionOutput(
            questions=[
                VanityInterviewQuestion(number=1, question="First"),
                VanityInterviewQuestion(number=2, question="Second"),
                VanityInterviewQuestion(number=3, question="Third"),
            ]
        )

        assert output.questions[0].question == "First"
        assert output.questions[1].question == "Second"
        assert output.questions[2].question == "Third"
