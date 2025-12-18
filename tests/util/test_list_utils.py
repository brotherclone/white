import pytest

from app.util.list_utils import (
    pick_by_fraction,
    pick_by_fraction_centered,
    interpolate_numeric_list,
)


class TestPickByFraction:
    def test_pick_first_item_with_zero_fraction(self):
        items = ["a", "b", "c", "d"]
        assert pick_by_fraction(items, 0.0) == "a"

    def test_pick_last_item_with_one_fraction(self):
        items = ["a", "b", "c", "d"]
        assert pick_by_fraction(items, 1.0) == "d"

    def test_pick_middle_item(self):
        items = ["a", "b", "c", "d"]
        # 0.5 * 4 = 2.0, min(int(2.0), 3) = 2
        assert pick_by_fraction(items, 0.5) == "c"

    def test_pick_item_with_quarter_fraction(self):
        items = ["a", "b", "c", "d"]
        # 0.25 * 4 = 1.0, min(int(1.0), 3) = 1
        assert pick_by_fraction(items, 0.25) == "b"

    def test_pick_item_with_three_quarter_fraction(self):
        items = ["a", "b", "c", "d"]
        # 0.75 * 4 = 3.0, min(int(3.0), 3) = 3
        assert pick_by_fraction(items, 0.75) == "d"

    def test_clamps_fraction_above_one(self):
        items = ["a", "b", "c"]
        assert pick_by_fraction(items, 1.5) == "c"

    def test_clamps_fraction_below_zero(self):
        items = ["a", "b", "c"]
        assert pick_by_fraction(items, -0.5) == "a"

    def test_single_item_list_always_returns_item(self):
        items = ["only"]
        assert pick_by_fraction(items, 0.0) == "only"
        assert pick_by_fraction(items, 0.5) == "only"
        assert pick_by_fraction(items, 1.0) == "only"

    def test_two_item_list(self):
        items = ["first", "second"]
        assert pick_by_fraction(items, 0.0) == "first"
        assert pick_by_fraction(items, 0.4) == "first"
        assert pick_by_fraction(items, 0.5) == "second"
        assert pick_by_fraction(items, 1.0) == "second"

    def test_empty_list_raises_value_error(self):
        with pytest.raises(ValueError, match="items must be non-empty"):
            pick_by_fraction([], 0.5)

    def test_works_with_integers(self):
        items = [10, 20, 30, 40]
        assert pick_by_fraction(items, 0.0) == 10
        assert pick_by_fraction(items, 1.0) == 40


class TestPickByFractionCentered:
    def test_pick_first_item_with_zero_fraction(self):
        items = ["a", "b", "c", "d"]
        # 0.0 * (4-1) = 0.0, round(0.0) = 0
        assert pick_by_fraction_centered(items, 0.0) == "a"

    def test_pick_last_item_with_one_fraction(self):
        items = ["a", "b", "c", "d"]
        # 1.0 * (4-1) = 3.0, round(3.0) = 3
        assert pick_by_fraction_centered(items, 1.0) == "d"

    def test_pick_middle_item(self):
        items = ["a", "b", "c", "d"]
        # 0.5 * (4-1) = 1.5, round(1.5) = 2
        assert pick_by_fraction_centered(items, 0.5) == "c"

    def test_pick_item_with_quarter_fraction(self):
        items = ["a", "b", "c", "d"]
        # 0.25 * (4-1) = 0.75, round(0.75) = 1
        assert pick_by_fraction_centered(items, 0.25) == "b"

    def test_pick_item_with_three_quarter_fraction(self):
        items = ["a", "b", "c", "d"]
        # 0.75 * (4-1) = 2.25, round(2.25) = 2
        assert pick_by_fraction_centered(items, 0.75) == "c"

    def test_clamps_fraction_above_one(self):
        items = ["a", "b", "c"]
        assert pick_by_fraction_centered(items, 1.5) == "c"

    def test_clamps_fraction_below_zero(self):
        items = ["a", "b", "c"]
        assert pick_by_fraction_centered(items, -0.5) == "a"

    def test_single_item_list_always_returns_item(self):
        items = ["only"]
        assert pick_by_fraction_centered(items, 0.0) == "only"
        assert pick_by_fraction_centered(items, 0.5) == "only"
        assert pick_by_fraction_centered(items, 1.0) == "only"

    def test_two_item_list(self):
        items = ["first", "second"]
        # 0.0 * 1 = 0, round(0) = 0
        assert pick_by_fraction_centered(items, 0.0) == "first"
        # 0.4 * 1 = 0.4, round(0.4) = 0
        assert pick_by_fraction_centered(items, 0.4) == "first"
        # 0.6 * 1 = 0.6, round(0.6) = 1
        assert pick_by_fraction_centered(items, 0.6) == "second"
        # 1.0 * 1 = 1.0, round(1.0) = 1
        assert pick_by_fraction_centered(items, 1.0) == "second"

    def test_empty_list_raises_value_error(self):
        with pytest.raises(ValueError, match="items must be non-empty"):
            pick_by_fraction_centered([], 0.5)

    def test_works_with_integers(self):
        items = [10, 20, 30, 40]
        assert pick_by_fraction_centered(items, 0.0) == 10
        assert pick_by_fraction_centered(items, 1.0) == 40


class TestInterpolateNumericList:
    def test_interpolate_first_value_with_zero_fraction(self):
        values = [0.0, 10.0, 20.0, 30.0]
        assert interpolate_numeric_list(values, 0.0) == 0.0

    def test_interpolate_last_value_with_one_fraction(self):
        values = [0.0, 10.0, 20.0, 30.0]
        assert interpolate_numeric_list(values, 1.0) == 30.0

    def test_interpolate_middle_value(self):
        values = [0.0, 10.0, 20.0, 30.0]
        # 0.5 * 3 = 1.5, floor = 1, t = 0.5
        # 10.0 * 0.5 + 20.0 * 0.5 = 15.0
        result = interpolate_numeric_list(values, 0.5)
        assert result == pytest.approx(15.0)

    def test_interpolate_quarter_value(self):
        values = [0.0, 10.0, 20.0, 30.0]
        # 0.25 * 3 = 0.75, floor = 0, t = 0.75
        # 0.0 * 0.25 + 10.0 * 0.75 = 7.5
        result = interpolate_numeric_list(values, 0.25)
        assert result == pytest.approx(7.5)

    def test_interpolate_three_quarter_value(self):
        values = [0.0, 10.0, 20.0, 30.0]
        # 0.75 * 3 = 2.25, floor = 2, t = 0.25
        # 20.0 * 0.75 + 30.0 * 0.25 = 22.5
        result = interpolate_numeric_list(values, 0.75)
        assert result == pytest.approx(22.5)

    def test_clamps_fraction_above_one(self):
        values = [0.0, 10.0, 20.0]
        assert interpolate_numeric_list(values, 1.5) == 20.0

    def test_clamps_fraction_below_zero(self):
        values = [0.0, 10.0, 20.0]
        assert interpolate_numeric_list(values, -0.5) == 0.0

    def test_single_value_always_returns_value(self):
        values = [42.0]
        assert interpolate_numeric_list(values, 0.0) == 42.0
        assert interpolate_numeric_list(values, 0.5) == 42.0
        assert interpolate_numeric_list(values, 1.0) == 42.0

    def test_two_value_list(self):
        values = [0.0, 100.0]
        assert interpolate_numeric_list(values, 0.0) == 0.0
        assert interpolate_numeric_list(values, 0.25) == pytest.approx(25.0)
        assert interpolate_numeric_list(values, 0.5) == pytest.approx(50.0)
        assert interpolate_numeric_list(values, 0.75) == pytest.approx(75.0)
        assert interpolate_numeric_list(values, 1.0) == 100.0

    def test_empty_list_raises_value_error(self):
        with pytest.raises(ValueError, match="values must be non-empty"):
            interpolate_numeric_list([], 0.5)

    def test_works_with_integers(self):
        values = [0, 10, 20, 30]
        result = interpolate_numeric_list(values, 0.5)
        assert result == pytest.approx(15.0)

    def test_works_with_negative_values(self):
        values = [-10.0, 0.0, 10.0]
        result = interpolate_numeric_list(values, 0.5)
        assert result == pytest.approx(0.0)

    def test_interpolation_beyond_last_index_returns_last_value(self):
        values = [0.0, 10.0, 20.0]
        # When fraction places us beyond last index
        result = interpolate_numeric_list(values, 0.99999)
        assert result == pytest.approx(20.0, abs=0.01)
