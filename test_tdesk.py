"""Offline tests for the tournament desk (no exchange): python test_tdesk.py"""
import os
import tempfile
import time

import numpy as np

import tpricer

tpricer.STATE_FILE = os.path.join(tempfile.mkdtemp(), "tdesk_state.json")   # never touch the real state
import tdesk_server as ts                                                    # noqa: E402
from mockx import to_px                                                      # noqa: E402

CTL = {**ts.CONTROLS_DEFAULT, "nTight": 0, "wideSince": 0.0}    # long since the last result: fully narrowed
NOW = 1e9


def lvl(price, mine=False, owner="x"):
    return {"price": to_px(price), "qty": 1, "orders": [{"mine": mine, "qty": 1, "owner": None if mine else owner}]}


def inst(iid, name, bids=(), asks=(), pos=0):
    return {"id": iid, "name": name, "tickSize": to_px(1), "bids": [lvl(*b) if isinstance(b, tuple) else lvl(b) for b in bids],
            "asks": [lvl(*a) if isinstance(a, tuple) else lvl(a) for a in asks], "position": pos}


def setup16(insts_override=None):
    st = tpricer.default_state()
    teams = st["teams"]
    insts = [inst(f"i{k}", t) for k, t in enumerate(teams)]
    for k, v in (insts_override or {}).items():
        insts[k] = v
    return st, teams, {t: [0, 100] for t in teams}, {t: f"i{k}" for k, t in enumerate(teams)}, ts.book_tops({"instruments": insts})


def plan(fair, tops, ctl=None, half=2.0, positions=None, feas=None, blocked=None, now=NOW):
    st, teams, feas0, by_team, _ = setup16()
    return ts.plan_quotes(st, feas or feas0, np.full(len(teams), fair) if np.isscalar(fair) else fair, np.full(len(teams), half),
                          tops, by_team, positions or {}, ctl or CTL, blocked, now)


def test_plan():
    st, teams, feas, by_team, empty = setup16()
    # narrowed, empty book: our own market, model width 2 x half = 4 around PA 20; one lot a side
    p, r = plan(20.0, empty)
    assert p["i0"] == [("bid", 18.0, 1), ("ask", 22.0, 1)] and r[0]["mode"] == "own", p["i0"]
    # right after a result: START WIDTH (20) wide, then narrowing in a straight line
    p, _ = plan(50.0, empty, {**CTL, "wideSince": NOW})
    assert p["i0"] == [("bid", 40.0, 1), ("ask", 60.0, 1)], p["i0"]
    half_way = NOW - CTL["narrowSecs"] / 2
    p, _ = plan(50.0, empty, {**CTL, "wideSince": half_way})
    assert p["i0"] == [("bid", 44.0, 1), ("ask", 56.0, 1)], p["i0"]                    # 12 wide = 4 + (20-4)/2
    # PA inside someone's 45 / 55 market while we're still wide: copy it
    _, _, _, _, t1 = setup16({0: inst("i0", teams[0], bids=[45], asks=[55])})
    p, r = plan(50.0, t1, {**CTL, "wideSince": NOW})
    assert p["i0"] == [("bid", 45.0, 1), ("ask", 55.0, 1)] and r[0]["mode"] == "copy", p["i0"]
    # ... once our width is tighter than theirs we sit inside, but never tighter than the MM spread
    p, _ = plan(50.0, t1, {**CTL, "rebateWidth": 6.0})
    assert p["i0"] == [("bid", 47.0, 1), ("ask", 53.0, 1)], p["i0"]
    # PA outside their market (they offer 45, PA 50): our own market, never crossing them; take their offer
    _, _, _, _, t2 = setup16({0: inst("i0", teams[0], bids=[40], asks=[45])})
    p, r = plan(50.0, t2)
    assert r[0]["mode"] == "own" and r[0]["take"] == ("bid", 45.0, 1) and max(x[1] for x in p["i0"] if x[0] == "bid") == 44.0, (p["i0"], r[0])
    # known payout -> 0.2 wide around it; anyone paying more / asking less than it can pay is taken
    feas2 = {**feas, teams[2]: [5, 5], teams[3]: [75, 100]}
    _, _, _, _, t3 = setup16({2: inst("i2", teams[2], bids=[7]), 3: inst("i3", teams[3], asks=[70])})
    p, r = plan(80.0, t3, feas=feas2)
    assert r[2]["take"] == ("ask", 7.0, 1) and r[3]["take"] == ("bid", 70.0, 1), (r[2], r[3])
    # pair check: teams 0 and 1 are the two finalists (always 160 between them). Team 1 is offered at 90
    # and bid 85, so our offer on team 0 must be >= 70 and our bid <= 75, whatever our fair says
    N = len(teams)
    pmin, pmax = np.zeros((N, N)), np.full((N, N), 200.0)
    pmin[0, 1] = pmin[1, 0] = pmax[0, 1] = pmax[1, 0] = 160.0
    _, _, _, _, tf = setup16({1: inst("i1", teams[1], bids=[85], asks=[90])})
    p, r = ts.plan_quotes(st, feas, np.full(N, 72.0), np.full(N, 2.0), tf, by_team, {}, CTL, None, NOW, (pmin, pmax))
    assert p["i0"] == [("bid", 70.0, 1), ("ask", 74.0, 1)], p["i0"]                 # 70 / 74 is fine as it is
    p, r = ts.plan_quotes(st, feas, np.full(N, 64.0), np.full(N, 2.0), tf, by_team, {}, CTL, None, NOW, (pmin, pmax))
    assert dict((s, px) for s, px, _ in p["i0"])["ask"] >= 70.0 and "pair check" in r[0]["note"], (p["i0"], r[0]["note"])
    p, r = ts.plan_quotes(st, feas, np.full(N, 80.0), np.full(N, 2.0), tf, by_team, {}, CTL, None, NOW, (pmin, pmax))
    assert dict((s, px) for s, px, _ in p["i0"])["bid"] <= 75.0, p["i0"]
    # fat-finger band: 30% of fair or 3 points, whichever is wider
    fc = {**CTL, "fatPct": 0.3, "fatMin": 3.0}
    assert ts.fat_finger_band(30.0, fc) == 9.0 and ts.fat_finger_band(2.5, fc) == 3.0 and ts.fat_finger_band(0.0, fc) == 3.0
    # partly known (3 wins: 20 - 100), wide right after a result: the bid never goes below 20
    p, _ = plan(24.0, empty, {**CTL, "wideSince": NOW}, feas={**feas, teams[0]: [20, 100]})
    assert p["i0"][0] == ("bid", 20.0, 1) and p["i0"][1][1] == 34.0, p["i0"]
    assert ts.known_quotes(5, 0.01) == (4.9, 5.1) and ts.known_quotes(0, 0.01) == (0.0, 0.2)
    assert ts.known_quotes(100, 0.01) == (99.8, 100.0) and ts.known_quotes(5, 1) == (4.0, 6.0)
    # traded on our bid: no bid until re-enabled, offer still there
    p, r = plan(20.0, empty, blocked={"i0": {"bid"}})
    assert p["i0"] == [("ask", 22.0, 1)] and "traded on bid" in r[0]["note"], p["i0"]
    # pulled team / closed contract: nothing
    p, r = plan(20.0, empty, {**CTL, "off": [teams[1]]})
    assert p["i1"] == [] and r[1]["why"] == "pulled" and p["i2"]
    # max position (10 by default; per market if set): long 10 -> no bid; Nova allowed 12 -> bid again
    p, _ = plan(20.0, empty, positions={"i0": 10})
    assert [o[0] for o in p["i0"]] == ["ask"], p["i0"]
    p, _ = plan(20.0, empty, {**CTL, "maxPosAdj": {teams[0]: 12}}, positions={"i0": 10})
    assert [o[0] for o in p["i0"]] == ["bid", "ask"], p["i0"]
    # MM requirement: the N smallest-model-spread markets narrow to the MM spread (2.9 -> 3 on tick 1)
    N = len(teams)
    hm = np.arange(N, dtype=float) + 2.0
    p, r = ts.plan_quotes(st, feas, np.full(N, 50.0), hm, empty, by_team, {}, {**CTL, "nTight": 14}, None, NOW)
    assert [x["tight"] for x in r] == [True] * 14 + [False] * (N - 14) and ts.n_tight(empty, {**CTL, "nTight": 40}) == N
    assert p["i0"][1][1] - p["i0"][0][1] <= 4 and p["i16"][1][1] - p["i16"][0][1] == 36, (p["i0"], p["i16"])
    # 310 self-check: everyone else offers 10 (160 for the other 16) -> our offer must be >= 150
    _, _, _, _, cheap = setup16({k: inst(f"i{k}", t, asks=[10]) for k, t in enumerate(teams)})
    p, _ = plan(100.0, cheap)
    assert not any(s == "ask" and px < 150 for s, px, q in p["i0"]), p["i0"]
    # other market makers' mids: n counts the distinct owners quoting both sides
    view = {"instruments": [inst("i0", teams[0], bids=[(40, False, "a")], asks=[(44, False, "a")])]}
    view["instruments"][0]["bids"][0]["orders"].append({"mine": False, "qty": 1, "owner": "b"})
    view["instruments"][0]["asks"][0]["orders"].append({"mine": False, "qty": 1, "owner": "b"})
    mid, n = ts.market_mids(st, ts.book_tops(view), {teams[0]: "i0"}, view)
    assert mid[0] == 42 and n[0] == 2 and np.isnan(mid[1])


def test_mapping_and_alerts():
    teams = [f"Team{i}" for i in range(1, 17)]
    names = [f"TEAM {i} payout" for i in range(1, 17)]
    m, probs = ts.auto_map(names, teams)
    assert not probs and m["TEAM 1 payout"] == "Team1" and m["TEAM 10 payout"] == "Team10", (m, probs)
    assert not ts.check_mapping(m, teams, names)
    bad = {**m, "TEAM 2 payout": "Team1"}
    assert ts.check_mapping(bad, teams, names)
    part = {"Australia": "Team1", "belgium": "Team2", "Russia": "Team3"}      # the 3-contract test market
    assert not ts.check_mapping(part, teams, list(part))
    _, probs = ts.auto_map(names + ["Mystery"], teams)
    assert any("Mystery" in p for p in probs)


def final_model():
    """The two finalists (teams 0 and 1) play for 100 / 75; everyone else's payout is known."""
    teams = [f"T{i}" for i in range(16)]
    st = {"teams": teams}
    rng = np.random.default_rng(0)
    w = rng.random(20000) < 0.5
    pay = np.zeros((20000, 16))
    pay[:, 0], pay[:, 1] = np.where(w, 100, 75), np.where(w, 75, 100)
    pay[:, 2:] = [50, 25] + [10] * 4 + [5] * 4 + [0] * 4
    feas = {t: [75, 100] if k < 2 else [pay[0, k]] * 2 for k, t in enumerate(teams)}
    sim = {"theo": pay.mean(axis=0), "cov": np.cov(pay, rowvar=False)}
    return st, {"sim": sim, "feasible": feas}, pay


def lock_model(M, pay):
    n = pay.shape[1]
    return {**M, "lockSum": np.array([[np.ptp(pay[:, i] + pay[:, j]) < 1e-9 for j in range(n)] for i in range(n)]),
            "lockDiff": np.array([[np.ptp(pay[:, i] - pay[:, j]) < 1e-9 for j in range(n)] for i in range(n)]),
            "sumVal": pay[0][:, None] + pay[0][None, :], "diffVal": pay[0][:, None] - pay[0][None, :]}


def hedge_rows(teams, live, pos, avg):
    return [{"team": t, "iid": f"i{k}", "settled": k not in live, "pos": pos if k == 0 else 0,
             "avg": avg if k == 0 else None, "oBid": None, "oAsk": None} for k, t in enumerate(teams)]


def test_fairs_and_hedges():
    st, M, pay = final_model()
    ctl = {**CTL, "deflPct": 0.5, "autoDefl": True}
    # long 1 of finalist 0, bought when our edge was 1 at 50% (made 9 at 11 style): D = 0.5, its PA drops
    # by 0.5, the other finalist's rises by the same - their total stays 175
    F = ts.fairs(st, M, {**ctl, "deflAmt": {"T0": 0.5}})
    base, theo, pa = F["base"], F["theo"], F["pa"]
    assert abs(pa[0] - theo[0] + 0.5) < 1e-6 and abs(pa[0] + pa[1] - 175) < 1e-6, pa[:2]
    assert np.allclose(pa[2:], base[2:])                                   # known payouts: no position skew
    # flipping the switch or the % doesn't move a PA we already have
    for other in ({"autoDefl": False}, {"deflPct": 0.9}):
        assert np.allclose(ts.fairs(st, M, {**ctl, "deflAmt": {"T0": 0.5}, **other})["pa"], pa)
    # a theo edit on finalist 0 pulls finalist 1 the other way; the PA edit doesn't touch theo
    theo2 = ts.fairs(st, M, {**ctl, "theoAdj": {"T0": 2.0}})["theo"]
    assert abs(theo2[0] - base[0] - 2) < 1e-6 and abs(theo2[1] - base[1] + 2) < 1e-6
    F3 = ts.fairs(st, M, {**ctl, "paAdj": {"T0": 1.0}})
    assert np.allclose(F3["theo"], base) and abs(F3["pa"][0] - base[0] - 1) < 1e-6 and abs(F3["pa"][1] - base[1] + 1) < 1e-6
    # market mid 90 on finalist 0, default 50/50: fair = (90 + ours) / 2; known teams ignore the market
    mid, n = np.full(16, np.nan), np.zeros(16)
    mid[0], n[0], mid[5], n[5] = 90.0, 2, 30.0, 1
    F4 = ts.fairs(st, M, ctl, (mid, n))
    assert abs(F4["theo"][0] - (90 + base[0]) / 2) < 1e-9 and F4["our"][0] == base[0] and F4["theo"][5] == base[5]
    # our share w: fair = (1 - w) x market mid + w x ours; 0 -> the market mid, 1 -> ours
    for w in (0.0, 0.3, 1.0):
        assert abs(ts.fairs(st, M, {**ctl, "ourShare": w}, (mid, n))["theo"][0] - ((1 - w) * 90 + w * base[0])) < 1e-9
    # the PA offset kept through a result is added on top
    F5 = ts.fairs(st, M, {**ctl, "paKeep": {"T0": -1.5}})
    assert abs(F5["pa"][0] - F5["theo"][0] + 1.5) < 1e-9
    # deflection book at 100%: make 9 at 11 (edge 1), get lifted -> D = -1; two more lots at edge 2 -> -5;
    # buy 2 back -> scaled to -5/3; flat -> 0; through zero starts at today's edge
    saved = ts.save_controls
    ts.save_controls = lambda ch: ch
    try:
        teams16 = [f"T{i}" for i in range(16)]
        w = np.full(16, 2.0)
        book = {"deflPos": {}, "deflAmt": {}, "deflPct": 1.0, "autoDefl": True}
        def step(pos, width, **kw):
            book.update(kw)
            w[0] = width
            q = np.zeros(16)
            q[0] = pos
            book["deflAmt"] = ts.update_defl_book({"teams": teams16}, book, q, w)
            book["deflPos"] = {"T0": float(pos)} if pos else {}
            return book["deflAmt"].get("T0", 0.0)
        assert step(-1, 2.0) == -1.0 and step(-3, 4.0) == -5.0 and abs(step(-1, 9.0) + 5 / 3) < 1e-12
        assert step(0, 9.0) == 0.0 and step(2, 6.0) == 6.0 and step(-1, 4.0) == -2.0
        # switched off: trades no longer move D (we deflect by hand); back on: only new trades count
        assert step(-4, 4.0, autoDefl=False) == -2.0 and step(0, 4.0) == -2.0
        assert step(-1, 4.0, autoDefl=True, deflPct=0.5) == -2.0 - 0.5 * 2.0
    finally:
        ts.save_controls = saved
    # the user's example: bought finalist 0 at 80, fair 87.5; buy the other finalist at 90 (gives up 2.5,
    # locks 175 - 170 = +5) beats selling ours at 81 (gives up 6.5, locks +1)
    LM = lock_model(M, pay)
    rows = hedge_rows(st["teams"], {0, 1}, 1, 80.0)
    rows[0].update(oBid=81.0, oAsk=83.0)
    rows[1].update(oBid=88.0, oAsk=90.0)
    fair = M["sim"]["theo"].copy()
    fair[:2] = 87.5
    h = ts.hedges(0, rows, M["sim"]["cov"], fair, LM)
    assert [(x["team"], x["side"], x["price"], round(x["edge"], 6), round(x["lockPnl"], 6)) for x in h] == \
           [("T1", "bid", 90.0, -2.5, 5.0), ("T0", "ask", 81.0, -6.5, 1.0)], h
    # semi-finals 0 v 1 and 2 v 3, we're short 0: short 1 (its opponent) is the natural hedge, but if its
    # bid gives up a lot of edge, shorting 2 or 3 comes first; every candidate is a sale
    rng = np.random.default_rng(1)
    N = 40000
    w1, w2, fw, tw = (rng.random(N) < 0.5 for _ in range(4))
    pay4 = np.zeros((N, 16))
    a, b = np.where(w1, 0, 1), np.where(w1, 1, 0)                    # semi 1 winner / loser
    c, d = np.where(w2, 2, 3), np.where(w2, 3, 2)
    r = np.arange(N)
    pay4[r, np.where(fw, a, c)], pay4[r, np.where(fw, c, a)] = 100, 75
    pay4[r, np.where(tw, b, d)], pay4[r, np.where(tw, d, b)] = 50, 25
    M4 = lock_model({"sim": {"cov": np.cov(pay4, rowvar=False), "theo": pay4.mean(axis=0)}}, pay4)
    th = np.where(np.arange(16) < 4, 62.5, 0.0)                     # 62.5 each
    rows = hedge_rows([f"S{i}" for i in range(16)], {0, 1, 2, 3}, -1, 60.0)
    rows[0].update(oAsk=64.5)                                        # buying back gives up 2.0
    rows[1].update(oBid=55.0)                                        # gives up 7.5
    rows[2].update(oBid=61.0)                                        # gives up 1.5
    rows[3].update(oBid=60.0)                                        # gives up 2.5
    h = ts.hedges(0, rows, M4["sim"]["cov"], th, M4)
    assert [(x["team"], x["side"]) for x in h] == [("S2", "ask"), ("S0", "bid"), ("S3", "ask")], h
    assert all(x["lockPnl"] is None for x in h if x["team"] != "S0") and -0.2 < h[0]["corr"] < -0.05, h
    rows[1].update(oBid=62.0)                                        # now the opponent is cheap to sell: first
    assert ts.hedges(0, rows, M4["sim"]["cov"], th, M4)[0]["team"] == "S1"
    # average price of the shares still open, and what the position is worth in the market
    assert ts.avg_cost([(2, 10.0), (2, 12.0)]) == (4, 11.0)
    assert ts.avg_cost([(2, 10.0), (-1, 15.0)]) == (1, 10.0)
    assert ts.avg_cost([(2, 10.0), (-3, 15.0)]) == (-1, 15.0)
    assert ts.avg_cost([(2, 10.0), (-2, 15.0)]) == (0, None)
    assert ts.market_value(3, 10.0, 12.0, 2.9) == 10.0 and ts.market_value(-3, 10.0, None, 2.9) is None
    assert ts.market_value(0, 10.0, 12.0, 2.9) == 11.0 and ts.market_value(0, 10.0, None, 2.0) == 11.0
    assert ts.market_value(0, None, 12.0, 2.0) == 11.0 and ts.market_value(0, None, None, 2.0) is None


def test_requote():
    live = [{"side": "bid", "price": 10.0, "qty": 4}, {"side": "ask", "price": 12.9, "qty": 4}]
    assert ts.close_enough([("bid", 10.1, 4), ("ask", 13.0, 4)], live, 0.2)
    assert not ts.close_enough([("bid", 10.5, 4), ("ask", 13.0, 4)], live, 0.2)
    assert not ts.close_enough([("bid", 10.0, 5), ("ask", 12.9, 4)], live, 0.2)
    assert not ts.close_enough([("bid", 10.0, 4)], live, 0.2)


def test_reconcile_reaches_every_market():
    """15 commands/s, 16 contracts x 6 orders: every contract must end up quoted (none starved)."""
    class FM:
        def __init__(self, book):
            self.book, self.left, self.n = book, 15, 0
        def budget(self):
            return self.left
        def order(self, iid, s, p, q, on_ack=None):
            self.left -= 1
            self.n += 1
            self.book.my_orders[f"o{self.n}"] = {"iid": iid, "side": s, "price": p, "qty": q}
        def cancel(self, oid):
            self.left -= 1
            self.book.my_orders.pop(oid, None)

    class B:
        my_orders = {}
        pos = {}
    book = B()
    fm = FM(book)
    g = ts.DeskGateway(False)
    plan = {f"i{k}": [("bid", 10.0 - j, 1) for j in range(3)] + [("ask", 12.0 + j, 1) for j in range(3)] for k in range(16)}
    g.plan["m"], g.tol["m"] = plan, 0.2
    c = {"book": book, "fm": fm}
    t = [1000.0]
    real_time = time.time
    ts.time.time = lambda: t[0]
    try:
        for _ in range(12):                                   # 12 one-second windows
            fm.left = 15
            g.reconcile("m", c)
            t[0] += 1.0
        quoted = {o["iid"] for o in book.my_orders.values()}
        assert quoted == set(plan), sorted(set(plan) - quoted)
        # every price moves by 1: all get re-quoted over the next windows, none left behind
        g.plan["m"] = {iid: [(s, p + 1, q) for s, p, q in o] for iid, o in plan.items()}
        for _ in range(30):
            fm.left = 15
            g.reconcile("m", c)
            t[0] += 1.0
        assert all(ts.close_enough(g.plan["m"][iid], [o for o in book.my_orders.values() if o["iid"] == iid], 0)
                   for iid in plan)
        # hard limit on the LIVE position: long 24 with limit 25 -> only 1 lot of bids may rest, offers all
        book.my_orders.clear()
        book.pos = {"i0": 24}
        g.hard["m"], g.rest_pos["m"] = 25, {"i0": 20}            # snapshot is behind: the worse (24) counts
        g.plan["m"] = {"i0": plan["i0"]}
        g.last.clear()
        fm.left = 15
        g.reconcile("m", c)
        sides = [o["side"] for o in book.my_orders.values()]
        assert sides.count("bid") == 1 and sides.count("ask") == 3, sides
        # a manual order on the same contract is left alone by the quoter
        book.my_orders["oM"] = {"iid": "i0", "side": "ask", "price": 99.0, "qty": 2}
        g.manual["m"] = {"oM"}
        g.last.clear()
        fm.left = 15
        g.reconcile("m", c)
        assert "oM" in book.my_orders
        assert g.manual_quote("m", "i0", 10.0, 12.0, 1)["ok"] is False     # paper gateway refuses
    finally:
        ts.time.time = real_time


def test_auto_deflection_count():
    """AUTO DEFLECTION on, count 3: keep quoting through 2 fills, the 3rd switches the market off;
    off: the side traded on goes off at once; at the position limit: off at once."""
    class Book:
        def __init__(self):
            self.fills, self.pos, self.my_orders = [], {}, {}
        def name(self, iid):
            return iid
    class FM:
        def __init__(self):
            self.cancelled = []
        def cancel(self, oid):
            self.cancelled.append(oid)
    def fill(c, qty, iid="i0"):
        c["book"].pos[iid] = c["book"].pos.get(iid, 0) + qty
        c["book"].fills.append({"iid": iid, "qty": qty, "aggressor": False, "oid": "x"})
    g = ts.DeskGateway(False)
    c = {"book": Book(), "fm": FM()}
    g.defl_cfg["m"] = (True, 3, {"i0": 10})
    for k in range(2):
        fill(c, -1)
        g.on_fills("m", c)
        assert not g.blocked.get("m", {}).get("i0") and g.hits["m"]["i0"] == k + 1
    c["book"].my_orders = {"o1": {"iid": "i0", "side": "bid"}, "o2": {"iid": "i0", "side": "ask"}}
    fill(c, -1)
    g.on_fills("m", c)
    assert g.blocked["m"]["i0"] == {"bid", "ask"} and sorted(c["fm"].cancelled) == ["o1", "o2"]
    g.unblock("m", "i0")                                                  # start quoting: count starts again
    assert not g.hits["m"].get("i0") and not g.blocked["m"].get("i0")
    g.defl_cfg["m"] = (True, 3, {"i0": 4})                                 # short 3 already: one more = the limit
    fill(c, -1)
    g.on_fills("m", c)
    assert g.blocked["m"]["i0"] == {"bid", "ask"}
    g2, c2 = ts.DeskGateway(False), {"book": Book(), "fm": FM()}
    g2.defl_cfg["m"] = (False, 3, {"i0": 10})                              # auto deflection off: just that side
    fill(c2, 1)
    g2.on_fills("m", c2)
    assert g2.blocked["m"]["i0"] == {"bid"}
    # clicking our own top-of-book offer: cancels it and keeps the quoter off that side
    g3, c3 = ts.DeskGateway(True), {"book": Book(), "fm": FM()}
    g3.conns["m"] = c3
    c3["book"].my_orders = {"o7": {"iid": "i0", "side": "ask", "price": 33.0}, "o8": {"iid": "i0", "side": "bid", "price": 30.0}}
    assert g3.cancel_level("m", "i0", "ask", 33.0)["ok"] and c3["fm"].cancelled == ["o7"] and g3.blocked["m"]["i0"] == {"ask"}
    assert not g3.cancel_level("m", "i0", "ask", 34.0)["ok"]                 # nothing of ours there
    assert ts.DeskGateway(False).cancel_level("m", "i0", "ask", 33.0)["ok"] is False   # paper: refused


def test_order_owners():
    """Who is behind each price level, from the event feed: placed, partly filled, cancelled, ours = "us"."""
    h = ts.History()
    evs = [{"seq": 1, "type": "ORDER_PLACED", "orderId": "o1", "instrumentId": "i0", "owner": "igor", "side": "bid", "price": to_px(30), "qty": 2},
           {"seq": 2, "type": "ORDER_PLACED", "orderId": "o2", "instrumentId": "i0", "owner": "samz", "side": "bid", "price": to_px(30), "qty": 1},
           {"seq": 3, "type": "ORDER_PLACED", "orderId": "o3", "instrumentId": "i0", "owner": "me", "side": "ask", "price": to_px(33), "qty": 1},
           {"seq": 4, "type": "TRADE", "instrumentId": "i0", "price": to_px(30), "qty": 1, "buyer": "igor", "seller": "x",
            "aggressorSide": "ask", "restingOrderId": "o1", "aggressingOrderId": "o9"},
           {"seq": 5, "type": "ORDER_CANCELLED", "orderId": "o2"}]
    saved = ts.ps.events
    ts.ps.events = lambda mid, seq: [e for e in evs if e["seq"] > seq]
    try:
        h.update("m", {"seq": 5, "you": "me"}, {})
    finally:
        ts.ps.events = saved
    own = h.owners("i0")
    assert dict(own[("bid", 30.0)]) == {"igor": 1} and dict(own[("ask", 33.0)]) == {"us": 1}, dict(own)


def test_end_to_end():
    st = tpricer.default_state()
    st["sims"] = 20000
    tpricer.save_state(st)
    teams = st["teams"]
    insts = [inst(f"i{k}", f"{t} contract", bids=[1], asks=[60], pos=(4 if k == 0 else 0)) for k, t in enumerate(teams)]
    view = {"instruments": insts, "status": "open", "name": "fake",
            "myTrades": [{"seq": 1, "at": 5, "instrumentId": "i0", "signedQty": 4, "price": to_px(30), "aggressor": False}]}
    ts.ps.snapshot = lambda mid, max_age=0.3: view
    ts.gateway = ts.DeskGateway(False)
    ts.save_controls({"market": "fake", "nTight": 0})
    m, probs = ts.auto_map([i["name"] for i in insts], teams)
    assert not probs
    ts.save_controls({"mapping": m})
    ts.save_controls({"mappingOk": True})
    ts.tick()
    s1 = ts.state_json()
    rows = s1["rows"]
    d = [r["pa"] - r["theo"] for r in rows]
    c1 = ts.controls()                                                     # long 4: each lot deflects 50% of its edge
    assert c1["deflPos"] == {teams[0]: 4.0} and abs(d[0] + c1["deflAmt"][teams[0]]) < 1e-6, (d[0], c1["deflAmt"])
    assert abs(c1["deflAmt"][teams[0]] - 0.5 * 4 * rows[0]["width"] / 2) <= 0.5 * 4 * 1.0, (c1["deflAmt"], rows[0]["width"])  # quotes rounded to the tick (1)
    assert all(x > -1e-9 for x in d[1:]) and abs(sum(d)) < 1e-6, d          # others up, total unchanged
    r0 = rows[0]
    assert r0["pos"] == 4 and r0["avg"] == 30 and r0["mktValue"] == 1 and r0["pnl"] == 4 * (1 - 30) and r0["hitAt"] == 5
    assert s1["totalPnl"] == r0["pnl"] and s1["stage"] == "qualifier" and s1["fixtures"]["qual"]["a"] == "Apex"
    hs = r0["hedges"]                                                     # long Nova: buy its rivals, or sell it at 1
    assert all(h["side"] == "bid" and h["corr"] < 0 for h in hs if h["team"] != teams[0]) and hs == sorted(hs, key=lambda h: -h["edge"])
    close = next((h for h in hs if h["team"] == teams[0]), None)
    assert close is None or (close["side"] == "ask" and close["lockPnl"] == 1 - 30), hs
    assert s1["api"]["read"] and not s1["api"]["orders"] and r0["maxPos"] == 10 and r0["tier"] == 1
    # fat-finger guard on typed prices: inside fair +/- max(30%, 3) is fine, outside is refused
    assert ts.fat_finger("i0", r0["fatLo"] + 0.01, r0["fatHi"] - 0.01) is None
    assert "fat-finger" in ts.fat_finger("i0", None, r0["fatHi"] + 1) and "fat-finger" in ts.fat_finger("i0", 0.5 * r0["fatLo"], None)
    assert abs(r0["fatHi"] - r0["theo"] - max(0.3 * r0["theo"], 3.0)) < 1e-9
    # seed nudged 4 -> 3.6: stronger, re-priced
    st0 = tpricer.load_state()
    tpricer.save_state({**st0, "ratings": {teams[3]: 2.4}})             # Pulse: shared #1 seed 3.0 -> 2.4, a buff
    sb = ts.state_json()
    assert rows[3]["seed"] == 3.0 and abs(rows[3]["base"] - rows[4]["base"]) < 1.5          # same rank, near-same price
    assert sb["rows"][3]["seedVal"] == 2.4 and sb["rows"][3]["seedEdited"] and sb["rows"][3]["base"] > rows[3]["base"]
    tpricer.save_state(st0)
    # one other market maker everywhere (1 / 60, mid 30.5): fair = (1 x 30.5 + 1 x ours) / 2
    assert r0["nMM"] == 1 and r0["mktMid"] == 30.5 and abs(r0["theo"] - (30.5 + r0["our"]) / 2) < 1e-9
    # the qualifier: Glaive wins -> Apex is out (payout 0) and minimised straight away
    ts.apply_results({"Q": {"a": "Apex", "b": "Glaive", "winner": "Glaive", "ga": 3, "gb": 1}})
    sq = ts.state_json()
    rq = {r["team"]: r for r in sq["rows"]}
    assert sq["stage"] == "swiss" and rq["Apex"]["range"] == [0, 0] and rq["Apex"]["theo"] == 0 and "Apex" in sq["controls"]["hidden"]
    assert sq["fixtures"]["rounds"][0]["groups"]["0-0"][-1] == "Glaive"
    # a theo edit pins our fair for that team; ours still add up to 310
    ts.save_controls({"theoAdj": {teams[1]: 1.0}, "paAdj": {teams[2]: -2.0}})
    s2 = ts.state_json()
    assert abs(s2["rows"][1]["our"] - s2["rows"][1]["base"] - 1) < 1e-6 and abs(sum(r["our"] for r in s2["rows"]) - 310) < 1e-6
    gap = {r["team"]: r["pa"] - r["theo"] for r in s2["rows"]}
    # a Swiss result: re-priced, our edits dropped, every team's PA keeps its gap to theo, markets start wide again
    ts.apply_results({"S1:Nova-Strata": {"a": "Nova", "b": "Strata", "winner": "Strata", "ga": 0, "gb": 1}})
    s3 = ts.state_json()
    assert s3["rows"][0]["base"] < s2["rows"][0]["base"] and s3["narrowLeft"] > 0
    assert not s3["controls"]["theoAdj"] and not s3["controls"]["paAdj"] and s3["rows"][1]["our"] == s3["rows"][1]["base"]
    assert all(abs((r["pa"] - r["theo"]) - gap[r["team"]]) < 1e-6 for r in s3["rows"]), [(r["pa"] - r["theo"], gap[r["team"]]) for r in s3["rows"]]
    r3 = {r["team"]: r for r in s3["rows"]}
    assert r3["Strata"]["rec"] == {"w": 1, "l": 0, "status": "swiss"} and r3["Nova"]["rec"]["l"] == 1
    # renaming carries results, edits, pulls, minimised markets, layout, mapping, tiers and regions
    ts.save_controls({"edgeAdj": {"Nova": 3.0}, "off": ["Strata"], "layout": {"order": ["Strata", "Nova"]}})
    st = tpricer.load_state()
    ren = {"Nova": "Nova GG", "Strata": "Strata X"}
    new = [ren.get(t, t) for t in st["teams"]]
    ts.setup({"teams": new, "groups": {k: [ren.get(t, t) for t in v] for k, v in st["groups"].items()}})
    st2, c2 = tpricer.load_state(), ts.controls()
    assert st2["results"]["S1:Nova GG-Strata X"] == {"a": "Nova GG", "b": "Strata X", "winner": "Strata X", "ga": 0, "gb": 1}, st2["results"]
    assert c2["edgeAdj"] == {"Nova GG": 3.0} and c2["off"] == ["Strata X"] and c2["hidden"] == ["Apex"]
    assert c2["layout"]["order"] == ["Strata X", "Nova GG"] and c2["mapping"]["Nova contract"] == "Nova GG"
    assert st2["tiers"]["Nova GG"] == 1 and st2["regions"]["Strata X"] == "Sa"
    try:                                                                   # moving teams between pots: refused
        g = dict(st2["groups"])
        k1, k2 = list(g)[:2]
        ts.setup({"teams": new, "groups": {**g, k1: g[k2], k2: g[k1]}})
        raise AssertionError("pot move with results should be refused")
    except ValueError:
        pass    # the trader's edge on a market overrides the width schedule (and the MM-spread floor)
    _, _, _, _, empty = setup16()
    p, r = plan(50.0, empty, {**CTL, "wideSince": NOW, "edgeAdj": {"Nova": 1.0}})
    assert p["i0"] == [("bid", 49.0, 1), ("ask", 51.0, 1)] and p["i1"][1][1] - p["i1"][0][1] == 20, (p["i0"], p["i1"])
    return s1


if __name__ == "__main__":
    t0 = time.time()
    test_plan()
    test_mapping_and_alerts()
    test_fairs_and_hedges()
    test_requote()
    test_reconcile_reaches_every_market()
    test_auto_deflection_count()
    test_order_owners()
    s = test_end_to_end()
    print(f"test_tdesk.py: all tests passed ({time.time() - t0:.1f} s); PA - theo, long 4 Team1:",
          " ".join(f"{r['team']} {r['pa'] - r['theo']:+.2f}" for r in s["rows"][:8]))