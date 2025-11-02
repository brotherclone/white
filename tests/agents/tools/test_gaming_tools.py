import app.agents.tools.gaming_tools as gt


def test_roll_dice_monkeypatched(monkeypatch):
    def fake_randint(a, b):
        return (a + b) // 2
    monkeypatch.setattr('random.randint', fake_randint)
    total, details = gt.roll_dice([(1, 6), (2, 4)])
    assert total == 7
    assert isinstance(details, list)
    assert details[0][2][0] == 3


def test_no_repeat_roll_dice(monkeypatch):
    seq = [ (10, []), (10, []), (5, []), (6, []) ]
    it = iter(seq)
    def fake_roll_dice(arg):
        return next(it)
    monkeypatch.setattr(gt, 'roll_dice', fake_roll_dice)
    first, second = gt.no_repeat_roll_dice((1,6), (1,6))
    assert first == 5 and second == 6


def test_infranym_seeder_monkeypatched(monkeypatch):
    seq = iter([3, 2, 4])
    def fake_roll(rolls):
        return (next(seq), [])
    monkeypatch.setattr(gt, 'roll_dice', fake_roll)
    res = gt.infranym_seeder()
    assert isinstance(res, dict)
    assert 'parts' in res and 'tracks' in res and 'count' in res
    assert 'instructions' in res

