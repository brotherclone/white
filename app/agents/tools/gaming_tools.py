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

def no_repeat_roll_dice(roll_one, roll_two):
    while True:
        first_roll = roll_dice(roll_one)[0]
        second_roll = roll_dice(roll_two)[0]
        if first_roll != second_roll:
            return first_roll, second_roll


def infranym_seeder():
    """
    Generates a random infranym seeder configuration.
    :return:
    """
    parts = roll_dice([(1, 6)])[0]
    tracks = roll_dice([(1, 4)])[0] + 1
    count = roll_dice([(1, 8)])[0]
    return {
        "parts": parts,
        "tracks": tracks,
        "count": count,
        "instructions": f"Create {count} seeds, each with {parts} parts, and {tracks} tracks."
    }



