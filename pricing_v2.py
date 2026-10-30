"""Tick rounding. Pure functions; run `python pricing_v2.py` for the self-tests."""
import math


# Floats can't represent 0.07 exactly, so 0.07 / 0.01 = 7.000000000000001 and
# a naive ceil() would give 8 ticks. We snap to the nearest whole tick first
# when we're within a hair of it.

def _ticks(price, tick):
    n = price / tick
    return round(n) if abs(n - round(n)) < 1e-9 else n


def round_down(price, tick):
    """Largest multiple of tick <= price (use for bids)."""
    return round(math.floor(_ticks(price, tick)) * tick, 10)


def round_up(price, tick):
    """Smallest multiple of tick >= price (use for asks)."""
    return round(math.ceil(_ticks(price, tick)) * tick, 10)


def _test():
    assert round_down(80401.4, 1) == 80401 and round_up(80401.4, 1) == 80402
    assert round_down(80401.0, 1) == 80401 and round_up(80401.0, 1) == 80401
    assert round_up(0.07, 0.01) == 0.07 and round_down(0.07, 0.01) == 0.07
    assert round_down(0.123, 0.01) == 0.12 and round_up(0.123, 0.01) == 0.13
    print("pricing_v2.py: all tests passed")


if __name__ == "__main__":
    _test()
