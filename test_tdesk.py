"""Offline tests for the tournament desk (no exchange): python test_tdesk.py"""
import os
import tempfile
import time

import numpy as np

import tpricer

tpricer.STATE_FILE = os.path.join(tempfile.mkdtemp(), "tdesk_state.json")   # never touch the real state
import tdesk_server as ts                                                    # noqa: E402
from mockx import to_px                                                      # noqa: E402

CTL = {**ts.CONTROLS_DEFAULT, "rebate": False, "twoSided": False}    # plain quoting; rebate tested on its own


def lvl(price, mine=False):
    return {"price": to_px(price), "qty": 1, "orders": [{"mine": mine, "qty": 1}]}


def inst(iid, name, bids=(), asks=(), pos=0):
    return {"id": iid, "name": name, "tickSize": to_px(1), "bids": [lvl(*b) if isinstance(b, tuple) else lvl(b) for b in bids],
            "asks": [lvl(*a) if isinstance(a, tuple) else lvl(a) for a in asks], "position": pos, "legs": []}


def test_ladder_example():
    # the user's example: fair 10, making 9 at 11 for 4 -> 12 for 8 behind
    ctl = {**CTL, "topSize": 4, "depth": 1, "step": 1, "exp": 1.0}
    asks = ts.ladder(ctl, 10.0, 11.0, 1, 100, 1, [], "ask", 99)
    bids = ts.ladder(ctl, 10.0, 9.0, 1, 99, 1, [], "bid", 99)
    assert asks == [("ask", 11.0, 4), ("ask", 12.0, 8)], asks
    assert bids == [("bid", 9.0, 4), ("bid", 8.0, 8)], bids
    sq = ts.ladder({**ctl, "exp": 0.5, "depth": 3}, 10.0, 11.0, 1, 100, 1, [], "ask", 99)
    assert [q for _, _, q in sq] == [4, 6, 7, 8], sq                   # 4 x sqrt(1,2,3,4)
    # someone resting at 14 (more edge than our top): we sit there too, size 4 x 4 = 16
    j = ts.ladder({**ctl, "depth": 0}, 10.0, 11.0, 1, 100, 1, [14.0, 30.0], "ask", 99)
    assert j == [("ask", 11.0, 4), ("ask", 14.0, 16)], j
    # position room caps the total
    r = ts.ladder(ctl, 10.0, 11.0, 1, 100, 1, [], "ask", 6)
    assert r == [("ask", 11.0, 4), ("ask", 12.0, 2)], r


def test_plan():
    st = tpricer.default_state()
    teams = st["teams"]
    feas = {t: [0, 100] for t in teams}
    fair = np.full(16, 19.375)
    half = np.full(16, 2.0)
    insts = [inst(f"i{k}", t, bids=[15], asks=[25]) for k, t in enumerate(teams)]
    insts[0] = inst("i0", teams[0], bids=[16, (14, True)], asks=[19], pos=0)
    view = {"instruments": insts}
    tops = ts.book_tops(view)
    assert tops["i0"]["oBids"] == [16.0]                             # our own level is not "someone else"
    by_team = {t: f"i{k}" for k, t in enumerate(teams)}
    plan, rows = ts.plan_quotes(st, feas, fair, half, tops, by_team, {}, {**CTL, "depth": 0})
    # team 0: fair 19.375, half 2 -> 17 / 22, and someone offers 19: our ask can't stay at 22 above them
    # without stepping in -> 19 is less than fair+minEdge so we stay at 22; our bid 17 doesn't cross 19
    # (plus we join behind their 16 bid: edge 3.375 vs 2.375 at the top -> 4 x 1.42 = 6)
    assert plan["i0"] == [("bid", 17.0, 4), ("bid", 16.0, 6), ("ask", 22.0, 4)], plan["i0"]
    # someone offers BELOW our bid: our bid is pulled under their offer (never cross)
    t2 = ts.book_tops({"instruments": [inst("i0", teams[0], asks=[16])] + insts[1:]})
    p2, _ = ts.plan_quotes(st, feas, fair, half, t2, by_team, {}, {**CTL, "depth": 0})
    assert max(p for s, p, q in p2["i0"] if s == "bid") == 15.0, p2["i0"]
    # wide market 15 / 25: we sit at 17 / 22 (inside, our own spread) and join their 15 / 25 behind
    assert plan["i1"] == [("bid", 17.0, 4), ("bid", 15.0, 7), ("ask", 22.0, 4), ("ask", 25.0, 9)], plan["i1"]
    # step in: fair 20, half 4 -> 16 / 24, market 16 / 23 -> one tick inside: 17 / 22 (3 and 2 edge >= min 1)
    t3 = ts.book_tops({"instruments": [inst("i0", teams[0], bids=[16], asks=[23])] + insts[1:]})
    p3, _ = ts.plan_quotes(st, feas, np.full(16, 20.0), np.full(16, 4.0), t3, by_team, {}, {**CTL, "depth": 0})
    assert p3["i0"] == [("bid", 17.0, 4), ("bid", 16.0, 5), ("ask", 22.0, 4), ("ask", 23.0, 6)], p3["i0"]
    # minEdge 3.5: stepping in would leave too little edge -> stay at 16 / 24
    p4, _ = ts.plan_quotes(st, feas, np.full(16, 20.0), np.full(16, 4.0), t3, by_team, {}, {**CTL, "depth": 0, "minEdge": 3.5})
    assert p4["i0"] == [("bid", 16.0, 4), ("ask", 24.0, 4)], p4["i0"]
    # fixed payout -> no quotes
    feas2 = {**feas, teams[2]: [5, 5]}
    plan2, rows2 = ts.plan_quotes(st, feas2, fair, half, tops, by_team, {}, CTL)
    assert plan2["i2"] == [("bid", 4.0, 4), ("ask", 6.0, 4)] and "fixed" in rows2[2]["note"]   # risk-free
    # one team pulled -> no orders there, the rest still quoted
    p6, r6 = ts.plan_quotes(st, feas, fair, half, tops, by_team, {}, {**CTL, "off": [teams[1]]})
    assert p6["i1"] == [] and r6[1]["why"] == "pulled" and p6["i2"]
    # budget: bid 17 risks 17 a lot (payout can be 0); 50 -> 2 lots at 17 (34), 1 at 16 (16), nothing deeper
    p7, _ = ts.plan_quotes(st, feas, fair, half, tops, by_team, {}, {**CTL, "budget": 50.0, "depth": 2})
    assert [o for o in p7["i1"] if o[0] == "bid"] == [("bid", 17.0, 2), ("bid", 16.0, 1)], p7["i1"]
    # long 10 Team2 at fair 19.375 uses 193.75 of a 150 budget -> nothing more on either side
    p8, r8 = ts.plan_quotes(st, feas, fair, half, tops, by_team, {"i1": 10}, {**CTL, "budget": 150.0})
    assert p8["i1"] == [] and r8[1]["note"].startswith("budget used up"), (p8["i1"], r8[1])
    test_rebate(st, teams, feas, by_team, tops)
    t5 = ts.book_tops({"instruments": [{**insts[0], "closed": True}] + insts[1:]})       # closed contract: no quotes
    p5, r5 = ts.plan_quotes(st, feas, fair, half, t5, by_team, {}, CTL)
    assert p5["i0"] == [] and "closed" in r5[0]["why"]
    # 310 self-check: everyone else offers 10 (150 for the other 15) -> our offer must be >= 160
    cheap = [inst(f"i{k}", t, asks=[10]) for k, t in enumerate(teams)]
    plan3, _ = ts.plan_quotes(st, feas, np.full(16, 100.0), half, ts.book_tops({"instruments": cheap}), by_team, {}, CTL)
    assert not any(s == "ask" and p < 160 for s, p, q in plan3["i0"]), plan3["i0"]
    # ... and everyone else bids 15 (225) -> our bid must be <= 85
    rich = [inst(f"i{k}", t, bids=[15]) for k, t in enumerate(teams)]
    plan4, _ = ts.plan_quotes(st, feas, np.full(16, 95.0), half, ts.book_tops({"instruments": rich}), by_team, {}, CTL)
    assert max(p for s, p, q in plan4["i0"] if s == "bid") <= 85, plan4["i0"]


def test_rebate(st, teams, feas, by_team, tops):
    rc = {**CTL, "rebate": True, "rebateMargin": 1, "depth": 0}
    mh = np.arange(16, dtype=float) + 2.0                     # team k has model half k + 2
    empty = ts.book_tops({"instruments": [inst(f"i{k}", t) for k, t in enumerate(teams)]})
    assert ts.n_tight(empty, rc) == 14                         # 80% of 16 = 13, + 1 spare
    assert ts.tight_teams(st, mh, by_team, empty, feas, rc) == set(teams[:14])
    p, r = ts.plan_quotes(st, feas, np.full(16, 50.0), mh, empty, by_team, {}, rc)
    assert [x["tight"] for x in r] == [True] * 14 + [False] * 2
    b, a = p["i0"][0][1], p["i0"][1][1]
    assert a - b <= 2.9 + 1e-9 and b < 50 < a, p["i0"]
    assert p["i15"][1][1] - p["i15"][0][1] == 34                # not a rebate team: normal width
    # someone bids 52 on team 0 (above our tight offer): the tight pair would have to bid above fair
    # -> team 0 can't be tight; the next-smallest spread (team 14) takes its place
    busy = ts.book_tops({"instruments": [inst("i0", teams[0], bids=[52])] + [inst(f"i{k}", t) for k, t in enumerate(teams)][1:]})
    p2, r2 = ts.plan_quotes(st, feas, np.full(16, 50.0), mh, busy, by_team, {}, rc)
    assert not r2[0]["tight"] and r2[14]["tight"] and sum(x["tight"] for x in r2) == 14
    # (tick 1 here) our tight pair is 49 / 51; someone bids 51 -> slide up to 50 / 52, still <= 2.9 wide
    near = ts.book_tops({"instruments": [inst("i0", teams[0], bids=[51])] + [inst(f"i{k}", t) for k, t in enumerate(teams)][1:]})
    p3, r3 = ts.plan_quotes(st, feas, np.full(16, 50.0), mh, near, by_team, {}, rc)
    assert r3[0]["tight"] and [o[1] for o in p3["i0"] if o[1] in (50.0, 52.0)] == [50.0, 52.0], p3["i0"]
    # payout already fixed at 5: a tick either side, counts as tight
    p4, r4 = ts.plan_quotes(st, {**feas, teams[3]: [5, 5]}, np.full(16, 50.0), mh, empty, by_team, {}, rc)
    assert p4["i3"] == [("bid", 4.0, 4), ("ask", 6.0, 4)] and r4[3]["tight"], p4["i3"]
    # knocked out (payout 0): still two-sided, 0 bid / 1 tick offer, counts for the rebate
    p6, r6 = ts.plan_quotes(st, {**feas, teams[3]: [0, 0]}, np.full(16, 50.0), mh, empty, by_team, {}, rc)
    assert p6["i3"] == [("bid", 0.0, 4), ("ask", 1.0, 4)] and r6[3]["tight"], p6["i3"]
    # two-sided: long at max position still shows a 1-lot bid, until max + hard extra
    tc = {**rc, "twoSided": True, "maxPos": 5, "hardExtra": 2}
    p5, r5 = ts.plan_quotes(st, feas, np.full(16, 50.0), mh, empty, by_team, {"i0": 5, "i1": 7}, tc)
    assert [o for o in p5["i0"] if o[0] == "bid"] == [("bid", r5[0]["bid"], 1)] and "1 lot" in r5[0]["note"], p5["i0"]
    assert not [o for o in p5["i1"] if o[0] == "bid"] and "hard limit" in r5[1]["note"]


def test_take_crossing():
    st = tpricer.default_state()
    teams = st["teams"]
    feas = {t: [0, 100] for t in teams}
    by_team = {t: f"i{k}" for k, t in enumerate(teams)}
    tc = {**CTL, "take": True, "takeEdge": 1.0, "takeMax": 5}
    insts = [inst(f"i{k}", t) for k, t in enumerate(teams)]
    insts[0] = inst("i0", teams[0], bids=[25])                   # fair 20, half 2 -> we'd offer 22: 25 crosses
    insts[1] = inst("i1", teams[1], asks=[16])                   # we'd bid 18: 16 crosses
    insts[2] = inst("i2", teams[2], bids=[21])                   # 21 < 22: no take
    _, r = ts.plan_quotes(st, feas, np.full(16, 20.0), np.full(16, 2.0), ts.book_tops({"instruments": insts}), by_team, {}, tc)
    assert r[0]["take"] == ("ask", 25.0, 1) and r[1]["take"] == ("bid", 16.0, 1) and not r[2].get("take"), [x.get("take") for x in r[:3]]
    _, r = ts.plan_quotes(st, feas, np.full(16, 20.0), np.full(16, 2.0), ts.book_tops({"instruments": insts}), by_team,
                          {"i0": -20}, tc)                       # already short max: don't sell more
    assert not r[0].get("take")
    _, r = ts.plan_quotes(st, feas, np.full(16, 20.0), np.full(16, 2.0), ts.book_tops({"instruments": insts}), by_team, {},
                          {**tc, "take": False})
    assert not any(x.get("take") for x in r)


def test_mapping_and_alerts():
    teams = [f"Team{i}" for i in range(1, 17)]
    names = [f"TEAM {i} payout" for i in range(1, 17)]
    m, probs = ts.auto_map(names, teams)
    assert not probs and m["TEAM 1 payout"] == "Team1" and m["TEAM 10 payout"] == "Team10", (m, probs)
    assert not ts.check_mapping(m, teams, names)
    bad = {**m, "TEAM 2 payout": "Team1"}
    assert ts.check_mapping(bad, teams, names)
    part = {"Australia": "Team1", "belgium": "Team2", "Russia": "Team3"}      # the 3-contract test market
    assert not ts.check_mapping(part, teams, list(part)) and len(ts.unmapped_teams(part, teams, list(part))) == 13
    _, probs = ts.auto_map(names + ["Mystery"], teams)
    assert any("Mystery" in p for p in probs)
    # the exchange's real test market: tick 0.01, depth step still 1.00 apart (price units)
    lad = ts.ladder({**CTL, "topSize": 4, "depth": 1}, 10.0, 11.0, 0.01, 100, 0.01, [], "ask", 99)
    assert lad == [("ask", 11.0, 4), ("ask", 12.0, 8)], lad


def test_blend_and_requote():
    st = tpricer.default_state()
    teams = st["teams"]
    feas = {t: [0, 100] for t in teams}
    by_team = {t: f"i{k}" for k, t in enumerate(teams)}
    theo = np.full(16, 310 / 16)
    insts = [inst(f"i{k}", t) for k, t in enumerate(teams)]
    insts[0] = inst("i0", teams[0], bids=[28], asks=[32])           # market says 30 for team 0
    insts[1] = inst("i1", teams[1], bids=[(5, True)], asks=[9])      # only OUR bid: no market mid
    b, mids = ts.blend(st, theo, ts.book_tops({"instruments": insts}), by_team, feas, 0.7)
    assert mids[0] == 30 and np.isnan(mids[1]) and abs(b.sum() - 310) < 1e-9
    raw0 = 0.7 * 19.375 + 0.3 * 30
    assert abs(b[0] - raw0 * 310 / (raw0 + 15 * 19.375)) < 1e-9 and b[0] > b[2]
    # manual deflection: Team1 -2 moves its fair by exactly -2; with the Jacobian the others pick it up
    M = {"key": "k-test", "feasible": {}}
    ctl = {**CTL, "deflection": 0.25, "manualDefl": {teams[0]: -2.0}}
    f, prop = ts.market_fairs(st, M, ctl, np.zeros(16), np.full(16, 2.0), theo)
    assert not prop and abs(f[0] - (theo[0] - 2)) < 1e-9 and np.allclose(f[1:], theo[1:])
    J = np.full((16, 16), -10 / 15)
    np.fill_diagonal(J, 10.0)
    saved = dict(ts.MODEL)
    ts.MODEL.update(key="k-test", J=J)
    try:
        f, prop = ts.market_fairs(st, M, ctl, np.zeros(16), np.full(16, 2.0), theo)
    finally:
        ts.MODEL.clear()
        ts.MODEL.update(saved)
    assert prop and abs(f[0] - (theo[0] - 2)) < 1e-9 and np.allclose(f[1:] - theo[1:], 2 / 15) and abs(f.sum() - 310) < 1e-9
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


def test_end_to_end():
    st = tpricer.default_state()
    st["sims"] = 20000
    tpricer.save_state(st)
    teams = st["teams"]
    insts = [inst(f"i{k}", f"{t} contract", bids=[1], asks=[60], pos=(4 if k == 0 else 0)) for k, t in enumerate(teams)]
    view = {"instruments": insts, "myTrades": [{"instrumentId": "i0", "signedQty": 4, "price": to_px(30)}],
            "status": "open", "name": "fake"}
    ts.ps.snapshot = lambda mid, max_age=0.3: view
    ts.gateway = ts.DeskGateway(False)
    ts.save_controls({"market": "fake", "ourWeight": 1.0, "rebate": False})   # pure model; blend/rebate tested separately
    m, probs = ts.auto_map([i["name"] for i in insts], teams)
    assert not probs
    ts.save_controls({"mapping": m})
    ts.save_controls({"mappingOk": True})
    job = ts.tick()
    s0 = ts.state_json()
    assert not s0["propagated"]
    r0 = s0["rows"][0]
    assert abs(r0["fair"] - (r0["theo"] - 0.25 * 4 * r0["half"])) < 1e-6      # long 4 at 25%: one half-spread
    job.join(60)
    ts.tick(job)
    s1 = ts.state_json()
    assert s1["propagated"]
    rows = s1["rows"]
    d = [r["fair"] - r["theo"] for r in rows]
    assert abs(d[0] + rows[0]["half"]) < 0.6, d[0]                         # hits its target (MC tolerance)
    assert all(x > -0.3 for x in d[1:]) and abs(sum(d)) < 0.5, d           # others up (± sim noise), total unchanged
    group_a = [k for k, t in enumerate(teams) if t in st["groups"]["A"] and k != 0]
    assert max(d[k] for k in group_a) > np.mean(d[1:]), d                  # own group (opponents) move most
    assert s1["rows"][0]["pnl"] is not None and len(s1["fixtures"]) == 24 + 12
    ko = {f["id"]: f for f in s1["fixtures"] if f.get("ko")}
    assert (ko["QFA"]["srcA"], ko["QFA"]["srcB"]) == ("1A", "winner C2") and ko["T"]["srcA"] == "loser SF1"
    assert s1["api"]["read"] and not s1["api"]["orders"]
    i0 = s1["instruments"]["rows"][0]
    assert i0["pos"] == 4 and abs(i0["avg"] - 30) < 1e-9 and i0["team"] == teams[0]
    assert i0["mtmSrc"] == "mid" and i0["pnlMtm"] == 4 * (30.5 - 30) and s1["rows"][0]["pnl"] == i0["pnlMtm"]
    assert s1["totalPnl"] == i0["pnlMtm"]
    # a group result re-prices and keeps the total at 310
    g = st["groups"]["A"]
    st = tpricer.load_state()
    st["results"][f"A:{g[0]}-{g[1]}"] = {"a": g[0], "b": g[1], "ga": 0, "gb": 3}
    tpricer.save_state(st)
    s2 = ts.state_json()
    assert abs(s2["sumTheo"] - 310) < 1e-6 and s2["rows"][0]["theo"] < s1["rows"][0]["theo"]
    return s1


if __name__ == "__main__":
    t0 = time.time()
    test_ladder_example()
    test_plan()
    test_mapping_and_alerts()
    test_blend_and_requote()
    test_take_crossing()
    test_reconcile_reaches_every_market()
    s = test_end_to_end()
    print(f"test_tdesk.py: all tests passed ({time.time() - t0:.1f} s); deflection long 4 Team1:",
          " ".join(f"{r['team']} {r['fair'] - r['theo']:+.2f}" for r in s["rows"][:8]))
