from app.agents.tools.gaming_tools import roll_dice

def test_rolls_single_die_correctly():
    total, details = roll_dice([(1, 6)])
    assert total >= 1 and total <= 6
    assert len(details) == 1
    assert details[0][0] == 1
    assert details[0][1] == 6
    assert len(details[0][2]) == 1

def test_rolls_multiple_dice_correctly():
    total, details = roll_dice([(2, 6)])
    assert total >= 2 and total <= 12
    assert len(details) == 1
    assert details[0][0] == 2
    assert details[0][1] == 6
    assert len(details[0][2]) == 2

def test_handles_multiple_dice_types():
    total, details = roll_dice([(1, 20), (2, 4)])
    assert total >= 3 and total <= 28
    assert len(details) == 2
    assert details[0][0] == 1
    assert details[0][1] == 20
    assert len(details[0][2]) == 1
    assert details[1][0] == 2
    assert details[1][1] == 4
    assert len(details[1][2]) == 2

def test_handles_empty_input():
    total, details = roll_dice([])