"""Pricing helpers: Black-Scholes, ticks, put-call parity.

Pure functions only - no network, no clock, no exchange - so they are easy to
test. Run `python pricing.py` to run the self-tests at the bottom.

Conventions: zero interest rates, t in years, vol as a decimal (0.45 = 45%).
"""
import math

SECONDS_PER_YEAR = 365 * 86400


def norm_cdf(x):
    """Standard normal cumulative distribution, via math.erf (no scipy needed)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def years_left(seconds_left):
    return max(seconds_left, 0.0) / SECONDS_PER_YEAR


def move_std(spot, vol, t):
    """One standard deviation of the spot move between now and expiry, in USD."""
    return spot * vol * math.sqrt(max(t, 0.0))


# --- Black-Scholes -----------------------------------------------------------

def _d1(spot, strike, t, vol):
    sd = vol * math.sqrt(t)
    return (math.log(spot / strike) + 0.5 * sd * sd) / sd


def _expired(t, vol):
    # At (or numerically at) expiry the formula divides by ~0: use payoffs instead.
    return t <= 0 or vol <= 0 or vol * math.sqrt(t) < 1e-9


def call_price(spot, strike, t, vol):
    if _expired(t, vol):
        return max(spot - strike, 0.0)
    d1 = _d1(spot, strike, t, vol)
    d2 = d1 - vol * math.sqrt(t)
    return spot * norm_cdf(d1) - strike * norm_cdf(d2)


def put_price(spot, strike, t, vol):
    # Put-call parity with zero rates: C - P = S - K. Deriving the put this way
    # means our two theos can never disagree with each other.
    return call_price(spot, strike, t, vol) - (spot - strike)


def call_delta(spot, strike, t, vol):
    if _expired(t, vol):
        return 1.0 if spot > strike else (0.5 if spot == strike else 0.0)
    return norm_cdf(_d1(spot, strike, t, vol))


def put_delta(spot, strike, t, vol):
    return call_delta(spot, strike, t, vol) - 1.0


def theo(kind, spot, strike, t, vol):
    """Fair value of one lot. The future is worth spot (zero rates)."""
    if kind == "call":
        return call_price(spot, strike, t, vol)
    if kind == "put":
        return put_price(spot, strike, t, vol)
    if kind == "future":
        return spot
    raise ValueError(f"unknown kind {kind!r}")


def delta(kind, spot, strike, t, vol):
    """Change in value per $1 move in spot, per lot. Future = 1."""
    if kind == "call":
        return call_delta(spot, strike, t, vol)
    if kind == "put":
        return put_delta(spot, strike, t, vol)
    if kind == "future":
        return 1.0
    raise ValueError(f"unknown kind {kind!r}")


# --- normal (Bachelier) model ---------------------------------------------------
# For a market on a single unknown number (height of a building, number of
# bananas): your estimate is `fwd` with uncertainty `sd` (one standard
# deviation, in the same units). It doesn't random-walk like a price, so
# Black-Scholes doesn't fit; a normal distribution around your estimate does.

def norm_pdf(x):
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def normal_call(fwd, strike, sd):
    if sd <= 1e-12:
        return max(fwd - strike, 0.0)
    d = (fwd - strike) / sd
    return (fwd - strike) * norm_cdf(d) + sd * norm_pdf(d)


def normal_put(fwd, strike, sd):
    return normal_call(fwd, strike, sd) - (fwd - strike)     # parity, as for BS


def normal_call_delta(fwd, strike, sd):
    if sd <= 1e-12:
        return 1.0 if fwd > strike else (0.5 if fwd == strike else 0.0)
    return norm_cdf((fwd - strike) / sd)


# --- ticks -------------------------------------------------------------------
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


# --- put-call parity ---------------------------------------------------------

def parity_arbs(strike, call, put, fut, min_edge=0.0):
    """Look for prices that break C - P = F - K by more than the spreads.

    call, put, fut are (best_bid, best_ask) tuples; None means no order there.
    Uses the prices you would actually trade at, so the spread is already
    paid for. Returns a list of (what_to_do, locked_in_profit_per_lot).
    """
    cb, ca = call
    pb, pa = put
    fb, fa = fut
    out = []
    # Long call + short put = long future at K. Do that, sell the real future.
    if None not in (ca, pb, fb):
        edge = (fb - strike) - (ca - pb)
        if edge > min_edge:
            out.append(("buy call, sell put, sell future", edge))
    # Short call + long put = short future at K. Do that, buy the real future.
    if None not in (cb, pa, fa):
        edge = (cb - pa) - (fa - strike)
        if edge > min_edge:
            out.append(("sell call, buy put, buy future", edge))
    return out


# --- self-tests --------------------------------------------------------------

def _close(a, b, tol=1e-6):
    return abs(a - b) <= tol


def _test():
    # Textbook value: S=K=100, 1 year, 20% vol, zero rates -> 7.9656
    assert _close(call_price(100, 100, 1.0, 0.20), 7.9656, 1e-4)

    # Parity holds for our theos, ITM and OTM.
    for s in (79000, 80400, 81500):
        t = years_left(3600)
        c, p = call_price(s, 80400, t, 0.45), put_price(s, 80400, t, 0.45)
        assert _close(c - p, s - 80400, 1e-6), (s, c, p)
        assert c >= 0 and p >= 0

    # Deltas: call between 0 and 1, put = call - 1, ATM call ~0.5.
    d = call_delta(80400, 80400, years_left(3600), 0.45)
    assert 0.5 < d < 0.51, d
    assert _close(put_delta(80400, 80400, years_left(3600), 0.45), d - 1)

    # At expiry: intrinsic value and 0/1 delta (option becomes a binary bet).
    assert call_price(80500, 80400, 0, 0.45) == 100
    assert put_price(80500, 80400, 0, 0.45) == 0
    assert call_delta(80500, 80400, 0, 0.45) == 1.0
    assert put_delta(80300, 80400, 0, 0.45) == -1.0

    # One-sd move for 1 hour at 45% vol on $80k is about $385.
    assert 380 < move_std(80000, 0.45, years_left(3600)) < 390

    # Ticks.
    assert round_down(80401.4, 1) == 80401 and round_up(80401.4, 1) == 80402
    assert round_down(80401.0, 1) == 80401 and round_up(80401.0, 1) == 80401
    assert round_up(0.07, 0.01) == 0.07 and round_down(0.07, 0.01) == 0.07
    assert round_down(0.123, 0.01) == 0.12 and round_up(0.123, 0.01) == 0.13

    # Parity arb: synthetic long costs 300 - 200 = 100, future bid - K = 150.
    arbs = parity_arbs(80400, call=(290, 300), put=(200, 210), fut=(80550, 80560))
    assert len(arbs) == 1 and arbs[0][0].startswith("buy call") and _close(arbs[0][1], 50)
    # Fair prices: no arb.
    assert parity_arbs(80400, (290, 300), (200, 210), (80480, 80500)) == []
    # Same rich future, but a side each direction needs is missing: no arb, no crash.
    assert parity_arbs(80400, (290, None), (None, 210), (80550, None)) == []

    # Normal model: ATM call = sd / sqrt(2 pi); parity; deltas; zero sd = intrinsic.
    assert _close(normal_call(310, 310, 20), 20 / math.sqrt(2 * math.pi))
    assert _close(normal_call(320, 300, 15) - normal_put(320, 300, 15), 20)
    assert _close(normal_call_delta(310, 310, 20), 0.5)
    assert normal_call(320, 300, 0) == 20 and normal_put(280, 300, 0) == 20

    print("pricing.py: all tests passed")


if __name__ == "__main__":
    _test()
