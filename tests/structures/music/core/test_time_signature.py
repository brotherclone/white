import pytest
from app.structures.music.core.time_signature import TimeSignature

def test_time_signature_valid():
    ts = TimeSignature(numerator=4, denominator=4)
    assert ts.numerator == 4
    assert ts.denominator == 4
    assert str(ts) == '4/4'

def test_time_signature_invalid_denominator():
    with pytest.raises(ValueError):
        TimeSignature(numerator=3, denominator=5)

def test_time_signature_equality():
    ts1 = TimeSignature(numerator=3, denominator=4)
    ts2 = TimeSignature(numerator=3, denominator=4)
    ts3 = TimeSignature(numerator=4, denominator=4)
    assert ts1 == ts2
    assert ts1 != ts3

