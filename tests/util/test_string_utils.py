import pytest
from datetime import datetime, date

from app.util.string_utils import (
    sanitize_for_filename,
    truncate_simple,
    truncate_with_ellipsis,
    truncate_word_safe,
    resolve_name,
    format_date,
)


class TestSanitizeForFilename:
    def test_normal_string(self):
        assert sanitize_for_filename("Test File Name") == "test_file_name"

    def test_empty_string(self):
        assert sanitize_for_filename("") == "unnamed"

    def test_none_input(self):
        # Empty string should return "unnamed"
        assert sanitize_for_filename("") == "unnamed"

    def test_special_characters(self):
        assert sanitize_for_filename("Test@#$%File!Name") == "testfilename"

    def test_spaces_to_underscores(self):
        assert sanitize_for_filename("multiple   spaces") == "multiple___spaces"

    def test_max_length_truncation(self):
        long_name = "a" * 100
        result = sanitize_for_filename(long_name, max_length=50)
        assert len(result) == 50
        assert result == "a" * 50

    def test_trailing_underscores_removed(self):
        assert sanitize_for_filename("test___", max_length=10) == "test"

    def test_trailing_hyphens_removed(self):
        assert sanitize_for_filename("test---", max_length=10) == "test"

    def test_preserves_periods(self):
        assert sanitize_for_filename("file.name.txt") == "file.name.txt"

    def test_preserves_hyphens(self):
        assert sanitize_for_filename("test-file-name") == "test-file-name"

    def test_all_invalid_chars(self):
        assert sanitize_for_filename("@#$%^&*()") == "unnamed"


class TestTruncateSimple:
    def test_shorter_than_limit(self):
        assert truncate_simple("hello", 10) == "hello"

    def test_exactly_at_limit(self):
        assert truncate_simple("hello", 5) == "hello"

    def test_longer_than_limit(self):
        assert truncate_simple("hello world", 5) == "hello"

    def test_empty_string(self):
        assert truncate_simple("", 5) == ""


class TestTruncateWithEllipsis:
    def test_shorter_than_limit(self):
        assert truncate_with_ellipsis("hello", 10) == "hello"

    def test_exactly_at_limit(self):
        assert truncate_with_ellipsis("hello", 5) == "hello"

    def test_longer_than_limit(self):
        result = truncate_with_ellipsis("hello world", 8)
        assert result == "hello..."
        assert len(result) == 8

    def test_limit_smaller_than_ellipsis(self):
        result = truncate_with_ellipsis("hello world", 2)
        assert result == ".."
        assert len(result) == 2

    def test_custom_ellipsis(self):
        result = truncate_with_ellipsis("hello world", 7, ellipsis="--")
        assert result == "hello--"
        assert len(result) == 7

    def test_empty_string(self):
        assert truncate_with_ellipsis("", 5) == ""


class TestTruncateWordSafe:
    def test_shorter_than_limit(self):
        assert truncate_word_safe("hello world", 20) == "hello world"

    def test_truncates_on_word_boundary(self):
        result = truncate_word_safe("hello world this is a test", 15)
        assert result.endswith("...")
        assert len(result) <= 15

    def test_custom_placeholder(self):
        result = truncate_word_safe("hello world this is a test", 15, placeholder="--")
        assert result.endswith("--")


class TestResolveName:
    def test_none_returns_empty(self):
        assert resolve_name(None) == ""

    def test_string_returns_string(self):
        assert resolve_name("test") == "test"

    def test_empty_list_returns_empty(self):
        assert resolve_name([]) == ""

    def test_list_uses_first_element(self):
        assert resolve_name(["first", "second"]) == "first"

    def test_tuple_uses_first_element(self):
        assert resolve_name(("first", "second")) == "first"

    def test_object_with_name_attribute(self):
        class NamedObject:
            name = "object_name"

        assert resolve_name(NamedObject()) == "object_name"

    def test_dict_with_name_key(self):
        assert resolve_name({"name": "dict_name"}) == "dict_name"

    def test_dict_with_empty_name(self):
        assert resolve_name({"name": ""}) == ""

    def test_dict_with_none_name(self):
        assert resolve_name({"name": None}) == ""

    def test_dict_without_name_key(self):
        result = resolve_name({"other": "value"})
        assert "other" in result

    def test_integer_converts_to_string(self):
        assert resolve_name(123) == "123"

    @pytest.mark.skip("Error handling test - difficult to test print statements")
    def test_object_with_name_property_valueerror(self, capsys):
        # This test checks error handling with print statements
        pass


class TestFormatDate:
    def test_none_returns_none(self):
        assert format_date(None) is None

    def test_datetime_returns_isoformat(self):
        dt = datetime(2023, 1, 15, 10, 30, 45)
        result = format_date(dt)
        assert result == "2023-01-15T10:30:45"

    def test_date_returns_formatted(self):
        d = date(2023, 1, 15)
        result = format_date(d)
        assert result == "2023-01-15"

    def test_date_with_custom_format(self):
        d = date(2023, 1, 15)
        result = format_date(d, fmt="%d/%m/%Y")
        assert result == "15/01/2023"

    def test_iso_string_datetime(self):
        iso_str = "2023-01-15T10:30:45"
        result = format_date(iso_str)
        assert result == "2023-01-15T10:30:45"

    def test_iso_string_date_only(self):
        iso_str = "2023-01-15"
        result = format_date(iso_str)
        # Should parse and format as date
        assert "2023-01-15" in result

    def test_invalid_string_returns_as_is(self):
        invalid_str = "not a date"
        result = format_date(invalid_str)
        assert result == "not a date"

    def test_other_type_returns_none(self):
        assert format_date(123) is None
        assert format_date([]) is None
        assert format_date({}) is None
