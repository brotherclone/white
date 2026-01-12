"""Tests for vanity interview response structures."""

import pytest
from pydantic import ValidationError

from app.structures.concepts.vanity_interview_response import (
    VanityInterviewResponse,
    VanityInterviewResponseOutput,
)


class TestVanityInterviewResponse:
    """Test VanityInterviewResponse model."""

    def test_create_valid_response(self):
        """Test creating a valid interview response."""
        response = VanityInterviewResponse(
            question_number=1,
            response="The album was inspired by late-night studio sessions and urban soundscapes.",
        )

        assert response.question_number == 1
        assert "inspired by" in response.response

    def test_response_to_questions_1_to_3(self):
        """Test responses to questions 1-3."""
        for num in [1, 2, 3]:
            response = VanityInterviewResponse(
                question_number=num, response=f"Response to question {num}"
            )
            assert response.question_number == num

    def test_response_with_various_text(self):
        """Test responses with various text content."""
        responses = [
            "My creative process is organic and intuitive.",
            "The influences range from musique concrète to ambient techno.",
            "The themes explore memory, technology, and human connection.",
        ]

        for i, r_text in enumerate(responses, 1):
            response = VanityInterviewResponse(question_number=i, response=r_text)
            assert response.response == r_text

    def test_response_with_long_text(self):
        """Test response with long text."""
        long_response = """I think the creative process is something that evolves over time.
        It's not just about sitting down and forcing inspiration, but rather creating the
        conditions where ideas can emerge naturally. For this work specifically, I spent
        months gathering field recordings, experimenting with synthesis techniques, and
        allowing the material to guide the direction rather than imposing a predetermined
        structure. The result is something that feels both intentional and discovered."""

        response = VanityInterviewResponse(question_number=1, response=long_response)

        assert response.response == long_response

    def test_response_with_empty_string(self):
        """Test response with empty string is allowed."""
        response = VanityInterviewResponse(question_number=1, response="")
        assert response.response == ""

    def test_response_with_special_characters(self):
        """Test response with special characters."""
        special_response = "I'd say it's about 50% synthesis, 30% field recordings, and 20% \"happy accidents\" (as Bob Ross would say)."

        response = VanityInterviewResponse(question_number=2, response=special_response)

        assert response.response == special_response

    def test_response_with_unicode(self):
        """Test response with unicode characters."""
        unicode_response = "The piece draws from João Gilberto's bossa nova and Ryūichi Sakamoto's minimalism."

        response = VanityInterviewResponse(question_number=1, response=unicode_response)

        assert response.response == unicode_response

    def test_response_missing_question_number(self):
        """Test that response requires question_number field."""
        with pytest.raises(ValidationError) as exc_info:
            VanityInterviewResponse(response="Test response")

        assert "question_number" in str(exc_info.value)

    def test_response_missing_response(self):
        """Test that response requires response field."""
        with pytest.raises(ValidationError) as exc_info:
            VanityInterviewResponse(question_number=1)

        assert "response" in str(exc_info.value)

    def test_response_with_non_integer_question_number(self):
        """Test that question_number must be an integer."""
        with pytest.raises(ValidationError):
            VanityInterviewResponse(question_number="one", response="Test")

    def test_response_serialization(self):
        """Test response serialization to dict."""
        response = VanityInterviewResponse(
            question_number=2, response="The composition draws from various traditions."
        )

        data = response.model_dump()

        assert data["question_number"] == 2
        assert data["response"] == "The composition draws from various traditions."

    def test_response_from_dict(self):
        """Test creating response from dictionary."""
        data = {
            "question_number": 3,
            "response": "The future involves exploring new collaborative possibilities.",
        }

        response = VanityInterviewResponse(**data)

        assert response.question_number == 3
        assert (
            response.response
            == "The future involves exploring new collaborative possibilities."
        )

    def test_response_with_negative_question_number(self):
        """Test response with negative question number."""
        response = VanityInterviewResponse(question_number=-1, response="Test")
        assert response.question_number == -1

    def test_response_with_large_question_number(self):
        """Test response with large question number."""
        response = VanityInterviewResponse(question_number=100, response="Test")
        assert response.question_number == 100


class TestVanityInterviewResponseOutput:
    """Test VanityInterviewResponseOutput model."""

    def test_create_valid_output(self):
        """Test creating valid response output."""
        responses = [
            VanityInterviewResponse(question_number=1, response="Response 1"),
            VanityInterviewResponse(question_number=2, response="Response 2"),
            VanityInterviewResponse(question_number=3, response="Response 3"),
        ]

        output = VanityInterviewResponseOutput(responses=responses)

        assert len(output.responses) == 3
        assert output.responses[0].question_number == 1
        assert output.responses[2].response == "Response 3"

    def test_output_with_three_responses(self):
        """Test output typically has three responses."""
        output = VanityInterviewResponseOutput(
            responses=[
                VanityInterviewResponse(question_number=1, response="R1"),
                VanityInterviewResponse(question_number=2, response="R2"),
                VanityInterviewResponse(question_number=3, response="R3"),
            ]
        )

        assert len(output.responses) == 3

    def test_output_with_zero_responses(self):
        """Test output can be created with zero responses."""
        output = VanityInterviewResponseOutput(responses=[])
        assert len(output.responses) == 0

    def test_output_with_one_response(self):
        """Test output with single response."""
        output = VanityInterviewResponseOutput(
            responses=[
                VanityInterviewResponse(question_number=1, response="Solo response")
            ]
        )

        assert len(output.responses) == 1

    def test_output_with_more_than_three_responses(self):
        """Test output can have more than three responses."""
        responses = [
            VanityInterviewResponse(question_number=i, response=f"Response {i}")
            for i in range(1, 6)
        ]

        output = VanityInterviewResponseOutput(responses=responses)

        assert len(output.responses) == 5

    def test_output_missing_responses_field(self):
        """Test that output requires responses field."""
        with pytest.raises(ValidationError) as exc_info:
            VanityInterviewResponseOutput()

        assert "responses" in str(exc_info.value)

    def test_output_with_invalid_response_type(self):
        """Test that responses must be VanityInterviewResponse objects."""
        with pytest.raises(ValidationError):
            VanityInterviewResponseOutput(responses=["not", "valid", "responses"])

    def test_output_serialization(self):
        """Test output serialization to dict."""
        output = VanityInterviewResponseOutput(
            responses=[
                VanityInterviewResponse(question_number=1, response="First"),
                VanityInterviewResponse(question_number=2, response="Second"),
            ]
        )

        data = output.model_dump()

        assert "responses" in data
        assert len(data["responses"]) == 2
        assert data["responses"][0]["question_number"] == 1
        assert data["responses"][1]["response"] == "Second"

    def test_output_from_dict(self):
        """Test creating output from dictionary."""
        data = {
            "responses": [
                {"question_number": 1, "response": "R1"},
                {"question_number": 2, "response": "R2"},
                {"question_number": 3, "response": "R3"},
            ]
        }

        output = VanityInterviewResponseOutput(**data)

        assert len(output.responses) == 3
        assert output.responses[0].question_number == 1
        assert output.responses[2].response == "R3"

    def test_output_responses_ordering(self):
        """Test that responses maintain their order."""
        responses = [
            VanityInterviewResponse(question_number=3, response="Third"),
            VanityInterviewResponse(question_number=1, response="First"),
            VanityInterviewResponse(question_number=2, response="Second"),
        ]

        output = VanityInterviewResponseOutput(responses=responses)

        # Order should be preserved as given
        assert output.responses[0].question_number == 3
        assert output.responses[1].question_number == 1
        assert output.responses[2].question_number == 2

    def test_output_with_duplicate_question_numbers(self):
        """Test output allows duplicate question numbers."""
        responses = [
            VanityInterviewResponse(question_number=1, response="First R1"),
            VanityInterviewResponse(question_number=1, response="Second R1"),
        ]

        output = VanityInterviewResponseOutput(responses=responses)

        assert len(output.responses) == 2
        assert (
            output.responses[0].question_number
            == output.responses[1].question_number
            == 1
        )

    def test_output_iteration(self):
        """Test iterating over responses in output."""
        output = VanityInterviewResponseOutput(
            responses=[
                VanityInterviewResponse(question_number=i, response=f"R{i}")
                for i in range(1, 4)
            ]
        )

        question_numbers = [r.question_number for r in output.responses]
        assert question_numbers == [1, 2, 3]

    def test_output_access_by_index(self):
        """Test accessing responses by index."""
        output = VanityInterviewResponseOutput(
            responses=[
                VanityInterviewResponse(question_number=1, response="First"),
                VanityInterviewResponse(question_number=2, response="Second"),
                VanityInterviewResponse(question_number=3, response="Third"),
            ]
        )

        assert output.responses[0].response == "First"
        assert output.responses[1].response == "Second"
        assert output.responses[2].response == "Third"

    def test_output_with_mixed_question_numbers(self):
        """Test output with non-sequential question numbers."""
        output = VanityInterviewResponseOutput(
            responses=[
                VanityInterviewResponse(question_number=5, response="Fifth"),
                VanityInterviewResponse(question_number=2, response="Second"),
                VanityInterviewResponse(question_number=8, response="Eighth"),
            ]
        )

        assert len(output.responses) == 3
        assert output.responses[0].question_number == 5
        assert output.responses[1].question_number == 2
        assert output.responses[2].question_number == 8
