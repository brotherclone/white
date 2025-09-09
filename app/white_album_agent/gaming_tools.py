import random


def roll_dice(rolls):
    """
    Rolls dice as specified by a list of (number, sides) tuples.
    Example: [(1, 20), (1, 4)] rolls 1d20 + 1d4.
    Returns (total, details)
    """
    results = []
    total = 0
    for num, sides in rolls:
        dice = [random.randint(1, sides) for _ in range(num)]
        results.append((num, sides, dice))
        total += sum(dice)
    return total, results