"""Tournament desk (brief 1): pricer + quoter + dashboard for the 16-team tournament.

    python tdesk_server.py            # paper: everything works, the quotes are only shown
    python tdesk_server.py --trade    # live: can send orders (asks you to type LIVE)
    then open http://localhost:8003

Quoting, per mapped contract (numbers from tpricer.py and the controls on the page):
    half-spread = k * sigma_next + c        (sigma_next: how much the theo moves on the next result)
    deflection: our position moves our fair by DEFLECTION x half-spread per share
        (25%: long 4 -> fair drops one whole half-spread: new offer = old fair, new fair = old bid).
        It propagates: the move is turned into a strength (goal-rate) change and the whole
        tournament is re-simulated, so everyone else moves too - most of all likely opponents.
    top of book: bid = fair - half, ask = fair + half, stepped one tick inside anyone else
        if that still leaves MIN EDGE; size TOP SIZE.
    depth: DEPTH more levels, STEP apart (price units), size = TOP SIZE x (edge / top edge)^EXP
        (EXP 1: fair 10, 11 for 4 -> 12 for 8; EXP 0.5 = square-root scaling), and we also
        sit behind anyone resting further out (within MAX DEPTH of our top), sized the same way.
Rebate (on by default): the exchange pays for being within REBATE WIDTH on COVERAGE (80%) of the
    open contracts, so that many teams - the ones with the smallest model spread - are quoted no
    wider than REBATE WIDTH; the rest keep the normal k * sigma_next + c width.
Arbs are left to the separate arb bot.
    never outside the payouts still possible for that team, never past MAX POS, never
    crossing someone else's price, and never letting our quote plus everyone else's best
    prices lock in a profit against us (the 310 self-check).
Click "Result incoming" before a result is revealed (pulls everything); enter the result,
then switch quoting back on.
"""
import difflib
import json
import math
import os
import re
import sys
import threading
import time
import traceback
from collections import defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

import numpy as np

import deskgate as ps                  # REST snapshot cache, trading gateway
import pricing
import tpricer
from mockx import ApiError, from_px

PORT = int(os.environ.get("TDESK_PORT", 8003))     # set TDESK_PORT to run a second copy alongside
HTML_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tdesk.html")
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")   # core.js, sidebar.js, tab_*.js
TOKEN_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tdesk_token.txt")


def _token():
    """Same page token across restarts, so an open page keeps working (403 'forbidden' otherwise)."""
    try:
        with open(TOKEN_FILE, encoding="utf-8") as f:
            t = f.read().strip()
        if len(t) == 32:
            return t
    except OSError:
        pass
    t = ps.secrets.token_hex(16)
    with open(TOKEN_FILE, "w", encoding="utf-8") as f:
        f.write(t)
    return t


TOKEN = _token()
ALLOWED_HOSTS = {f"localhost:{PORT}", f"127.0.0.1:{PORT}"}
PLAN_MAX_AGE = 5.0                       # the gateway pulls everything if the plan is older than this
CONTROLS_DEFAULT = {"deflection": 0.25, "topSize": 4, "depth": 2, "step": 1.0, "exp": 1.0, "maxDepth": 5.0,
                    "maxOrder": 20, "k": 1.0, "c": 1.0, "minEdge": 1.0, "maxPos": 20, "improve": True,
                    "quoting": False, "clickQty": 1, "market": "", "mapping": {}, "mappingOk": False,
                    "budget": 2000.0, "off": [], "rebate": True, "rebateWidth": 2.9, "coverage": 0.8,
                    "ourWeight": 0.7, "requoteMin": 0.2, "newsPause": 5.0,
                    "rebateMargin": 1, "twoSided": True, "hardExtra": 5,
                    "take": True, "takeEdge": 1.0, "takeMax": 5, "manualDefl": {}, "manualStep": 0.5}

LOCK = threading.Lock()
MODEL = {"key": None}


# --- saved state: tpricer's tdesk_state.json, plus our "controls" ---------------------------------
def controls():
    return {**CONTROLS_DEFAULT, **tpricer.load_state().get("controls", {})}


def save_controls(changes):
    st = tpricer.load_state()
    c = {**CONTROLS_DEFAULT, **st.get("controls", {})}
    for k, v in changes.items():
        if k not in CONTROLS_DEFAULT:
            continue
        d = CONTROLS_DEFAULT[k]
        if k == "manualDefl":
            c[k] = {str(a): float(b) for a, b in (v or {}).items() if isinstance(b, (int, float)) and b and abs(b) <= 100}
        elif isinstance(d, dict):
            c[k] = {str(a): str(b) for a, b in (v or {}).items() if b}
        elif isinstance(d, list):
            c[k] = sorted({str(x) for x in (v or [])})
        elif isinstance(d, bool):
            c[k] = bool(v)
        elif isinstance(d, str):
            c[k] = str(v).strip()
        elif isinstance(v, (int, float)) and v >= 0:
            c[k] = type(d)(v)
    if "mapping" in changes and "mappingOk" not in changes:
        c["mappingOk"] = False                      # an edited mapping has to be confirmed again
    if "market" in changes and changes["market"] != st.get("controls", {}).get("market"):
        c["mappingOk"], c["quoting"] = False, False
    st["controls"] = c
    with open(tpricer.STATE_FILE + ".tmp", "w", encoding="utf-8") as f:
        json.dump(st, f, indent=1)
    os.replace(tpricer.STATE_FILE + ".tmp", tpricer.STATE_FILE)
    return c


def model():
    """The simulation for the current setup + results (re-run only when they change)."""
    st = tpricer.load_state()
    key = json.dumps({k: st[k] for k in ("teams", "groups", "pBeat", "goals", "kappa", "nudges",
                                          "confidence", "results", "sims", "ratings")}, sort_keys=True)
    with LOCK:
        if MODEL["key"] != key:
            t0 = time.time()
            post = tpricer.posterior(st)
            sim = tpricer.simulate(st, post=post)
            MODEL.clear()
            MODEL.update(key=key, sim=sim, post=post, readouts=tpricer.readouts(st, sim),
                         feasible=tpricer.feasible(st), slots=slots(st, sim), J=None,
                         ms=round((time.time() - t0) * 1000))
        return st, dict(MODEL)


def slots(st, sim):
    """Who plays in each knockout match, as far as the results entered so far say."""
    res, teams = st.get("results", {}), st["teams"]
    place = {}
    for g in tpricer.GROUPS:
        if sum(1 for m in res if m.startswith(g + ":") and res[m].get("ga") is not None) == 6:
            for t in st["groups"][g]:
                place[f"{int(np.argmax(sim['finish'][teams.index(t)])) + 1}{g}"] = t
    out = {}
    for mid, x, y in tpricer.CROSS + tpricer.QF + tpricer.SF:
        out[mid] = [place.get(x), place.get(y)]
        r = res.get(mid)
        if r and r.get("winner"):
            place[mid] = r["winner"]
            place[mid + "L"] = r["b"] if r["winner"] == r["a"] else r["a"]
    out["F"] = [place.get("SF1"), place.get("SF2")]
    out["T"] = [place.get("SF1L"), place.get("SF2L")]
    return out


# --- contract mapping ----------------------------------------------------------------------------
def _norm(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def auto_map(names, teams):
    """contract name -> team: exact, then 'contains the team name' (longest wins), then fuzzy."""
    tn = {t: _norm(t) for t in teams}
    out, problems = {}, []
    for n in names:
        k = _norm(n)
        hit = next((t for t, v in tn.items() if v == k), None)
        if not hit:
            inside = sorted((t for t, v in tn.items() if v and v in k), key=lambda t: -len(tn[t]))
            if inside and (len(inside) == 1 or len(tn[inside[0]]) > len(tn[inside[1]])):
                hit = inside[0]
        if not hit:
            close = difflib.get_close_matches(k, list(tn.values()), n=1, cutoff=0.75)
            hit = next((t for t, v in tn.items() if close and v == close[0]), None)
        if hit:
            out[n] = hit
        else:
            problems.append(f"{n}: no match")
    vals = list(out.values())
    problems += [f"{t}: matched by more than one contract" for t in sorted(set(vals)) if vals.count(t) > 1]
    return out, problems


def check_mapping(mapping, teams, names):
    """Problems that block quoting. Teams without a contract are allowed (the test market has only
    3): they're simply not quoted - see unmapped_teams()."""
    probs = [f"contract '{n}' not mapped" for n in names if n not in mapping]
    vals = [mapping[n] for n in names if n in mapping]
    probs += [f"{t} mapped twice" for t in sorted(set(vals)) if vals.count(t) > 1]
    probs += [f"'{v}' is not a team" for v in vals if v not in teams]
    return probs


def unmapped_teams(mapping, teams, names):
    vals = {mapping[n] for n in names if n in mapping}
    return [t for t in teams if t not in vals]


# --- the quote plan (pure: no exchange calls, so it can be tested) --------------------------------
def book_tops(view):
    """Per contract: best bid/ask, and everyone else's levels (levels holding only our orders left out)."""
    tops = {}
    for inst in view["instruments"]:
        def others(side):
            return [from_px(l["price"]) for l in side if not all(o.get("mine") for o in l["orders"])]
        def other_qty(side):                    # everyone else's size at their best level
            for l in side:
                q = sum(o["qty"] for o in l["orders"] if not o.get("mine"))
                if q:
                    return q
            return 0
        tops[inst["id"]] = {"oBidQty": other_qty(inst["bids"]), "oAskQty": other_qty(inst["asks"]),
                            "bid": from_px(inst["bids"][0]["price"]) if inst["bids"] else None,
                            "ask": from_px(inst["asks"][0]["price"]) if inst["asks"] else None,
                            "oBids": others(inst["bids"]), "oAsks": others(inst["asks"]),
                            "closed": bool(inst.get("closed") or inst.get("settlementValue") is not None),
                            "tick": inst["tickSize"] / 10_000}
    return tops


def ladder(ctl, f, top, lo, hi, tick, joins, side, room, worst=None, risk_left=float("inf")):
    """Our orders on one side: top level + depth + joins behind others, size scaled by edge.
    worst: the payout that hurts this side most (bids: lowest possible, offers: highest); each lot
    risks |price - worst|, and the side stops once that would use up risk_left (the budget)."""
    sgn = -1 if side == "bid" else 1                        # further from fair = down for bids
    edge0 = max(abs(top - f), tick)
    step = max(ctl["step"], tick)                           # step and join range are in price units
    prices = [round(round((top + sgn * j * step) / tick) * tick, 9) for j in range(ctl["depth"] + 1)]
    prices += [p for p in joins if 0 < sgn * (p - top) <= ctl["maxDepth"] + 1e-9]
    out = []
    for p in sorted({round(x, 9) for x in prices}, key=lambda x: sgn * x):
        if not lo - 1e-9 <= p <= hi + 1e-9 or room <= 0:
            continue
        q = int(round(ctl["topSize"] * (abs(p - f) / edge0) ** ctl["exp"]))
        q = max(1, min(q, ctl["maxOrder"], room))
        per = abs(p - worst) if worst is not None else 0.0
        if per > 0:
            q = min(q, int(risk_left // per))
        if q < 1:
            break
        out.append((side, p, q))
        room -= q
        risk_left -= q * per
    return out


def risk_used(st, feas, fair, by_team, positions):
    """What our positions can still lose from fair: long -> down to the lowest payout, short -> up to the highest."""
    total = 0.0
    for k, t in enumerate(st["teams"]):
        q = positions.get(by_team.get(t), 0) if by_team.get(t) else 0
        lo, hi = feas.get(t, [0, 100])
        total += q * (fair[k] - lo) if q > 0 else -q * (hi - fair[k])
    return float(total)


def n_tight(tops, ctl):
    """How many contracts to quote within the rebate width: COVERAGE of the open ones, + MARGIN spare."""
    if not ctl["rebate"]:
        return 0
    n_open = sum(1 for x in tops.values() if not x.get("closed"))
    return min(n_open, math.ceil(ctl["coverage"] * n_open - 1e-9) + int(ctl["rebateMargin"]))


def tight_teams(st, half, by_team, tops, feas, ctl):
    """First guess at the rebate markets (used for the deflection half-spread): the quotable
    teams with the smallest model spread. plan_quotes makes the final pick."""
    cands = [k for k, t in enumerate(st["teams"]) if by_team.get(t) and not tops[by_team[t]].get("closed")
             and t not in ctl["off"] and feas.get(t, [0, 100])[0] != feas.get(t, [0, 100])[1]]
    cands.sort(key=lambda k: half[k])
    return {st["teams"][k] for k in cands[:n_tight(tops, ctl)]}


def quoted_half(st, half, tight, ctl):
    """The half-spread we actually quote (deflection is a fraction of THIS)."""
    return np.array([min(h, ctl["rebateWidth"] / 2) if t in tight else h for t, h in zip(st["teams"], half)])


def plan_quotes(st, feas, fair, half, tops, by_team, positions, ctl):
    """Every order we want resting: {contract id: [(side, price, qty)]}, plus a row per team.

    Rebate (ctl rebate): n_tight() contracts are quoted no wider than REBATE WIDTH. A contract whose
    payout is already fixed is quoted a tick either side of it (risk-free, and it counts). The rest
    are picked by smallest model spread among those where a tight pair is possible without crossing
    anyone or bidding above / offering below our fair. TWO-SIDED: a side blocked by max position
    or the budget still shows 1 lot (until HARD EXTRA lots past max position), so every market has
    a bid and an offer."""
    plan, rows = {}, []
    left = ctl["budget"] - risk_used(st, feas, fair, by_team, positions)   # per side, per team
    o_bid = {iid: t["oBids"][0] for iid, t in tops.items() if t["oBids"]}
    o_ask = {iid: t["oAsks"][0] for iid, t in tops.items() if t["oAsks"]}
    w = ctl["rebateWidth"]
    info = {}
    for k, t in enumerate(st["teams"]):                          # pass 1: a normal and a tight pair each
        iid = by_team.get(t)
        lo, hi = feas.get(t, [0, 100])
        f, h = float(fair[k]), float(half[k])
        row = {"team": t, "iid": iid, "fair": f, "half": h, "orders": [], "why": "",
               "pos": positions.get(iid, 0) if iid else 0}
        rows.append(row)
        if not iid:
            row["why"] = "no contract"
            continue
        if tops[iid].get("closed") or t in ctl["off"]:
            row["why"] = "pulled" if t in ctl["off"] else "contract closed / settled"
            plan[iid] = []
            continue
        tick = tops[iid]["tick"]
        if lo == hi:                                             # payout known: a tick either side is free money;
            info[k] = {"fixed": True, "tick": tick, "bid": round(max(lo - tick, 0.0), 9),   # out (payout 0): 0 / 1 tick,
                       "ask": round(lo + tick, 9)}                                     # still two-sided for the rebate
            continue
        h = max(h, tick)
        ob, oa = o_bid.get(iid), o_ask.get(iid)
        others = [x for x in by_team.values() if x != iid]
        # 310 self-check: nobody may sell the whole field for more than 310 by hitting our bid and
        # everyone else's bids, or buy it for less than 310 by lifting our offer and everyone else's.
        cap_bid = (pricing.round_down(tpricer.TOTAL - sum(o_bid[x] for x in others), tick)
                   if len(others) == 15 and all(x in o_bid for x in others) else None)
        cap_ask = (pricing.round_up(tpricer.TOTAL - sum(o_ask[x] for x in others), tick)
                   if len(others) == 15 and all(x in o_ask for x in others) else None)
        bid, ask = pricing.round_down(f - h, tick), pricing.round_up(f + h, tick)
        if ctl["improve"]:                                       # top of book, if it still pays MIN EDGE
            if ob is not None and ob >= bid and ob + tick <= f - ctl["minEdge"] + 1e-9:
                bid = round(ob + tick, 9)
            if oa is not None and oa <= ask and oa - tick >= f + ctl["minEdge"] - 1e-9:
                ask = round(oa - tick, 9)
        if oa is not None:                                       # passive only: never cross anyone
            bid = min(bid, round(oa - tick, 9))
        if ob is not None:
            ask = max(ask, round(ob + tick, 9))
        if cap_bid is not None:
            bid = min(bid, cap_bid)
        if cap_ask is not None:
            ask = max(ask, cap_ask)
        # tight pair: centred on fair, slid (keeping the width) to stay behind anyone else's price
        tb = pricing.round_up(f - w / 2, tick)
        ta = pricing.round_down(tb + w, tick)
        if ob is not None and ob >= ta:
            ta = round(ob + tick, 9)
            tb = pricing.round_up(ta - w, tick)
        if oa is not None and oa <= tb:
            tb = round(oa - tick, 9)
            ta = pricing.round_down(tb + w, tick)
        if cap_bid is not None:
            tb = min(tb, cap_bid)
        if cap_ask is not None:
            ta = max(ta, cap_ask)
        ok = ((ob is None or ob < ta) and (oa is None or oa > tb) and ta - tb <= w + 1e-9
              and tb <= f + 1e-9 and ta >= f - 1e-9 and tb >= tick and ta <= hi)
        info[k] = {"fixed": False, "tick": tick, "bid": bid, "ask": ask, "tb": tb, "ta": ta, "tightOk": ok,
                   "b0": pricing.round_down(f - h, tick), "a0": pricing.round_up(f + h, tick),       # before clamps:
                   "tb0": pricing.round_up(f - w / 2, tick), "ta0": pricing.round_down(pricing.round_up(f - w / 2, tick) + w, tick)}

    need = n_tight(tops, ctl)
    tight = {k for k, x in info.items() if x["fixed"] and x["bid"] is not None}
    for k in sorted((k for k, x in info.items() if not x["fixed"] and x["tightOk"]), key=lambda k: half[k]):
        if len(tight) >= need:
            break
        tight.add(k)

    for k, x in info.items():                                    # pass 2: the orders
        t, row = st["teams"][k], rows[k]
        iid, tick, pos = by_team[t], x["tick"], row["pos"]
        lo, hi = feas.get(t, [0, 100])
        f = row["fair"]
        tb_ = tops[iid]
        ob, oa = o_bid.get(iid), o_ask.get(iid)
        if x["fixed"]:
            orders = ([("bid", x["bid"], ctl["topSize"])] if x["bid"] is not None else []) + [("ask", x["ask"], ctl["topSize"])]
            row.update(orders=orders, bid=x["bid"], ask=x["ask"], tight=k in tight, note=f"payout fixed at {lo}: risk-free quotes")
            plan[iid] = orders
            if ctl["take"] and ob is not None and ob > lo + 1e-9:           # someone pays more than it can pay out
                row["take"] = ("ask", ob, min(tb_["oBidQty"], ctl["takeMax"]))
            elif ctl["take"] and oa is not None and oa < lo - 1e-9:
                row["take"] = ("bid", oa, min(tb_["oAskQty"], ctl["takeMax"]))
            continue
        bid, ask = (x["tb"], x["ta"]) if k in tight else (x["bid"], x["ask"])
        if k in tight:
            row["half"] = (ask - bid) / 2
        orders, notes = [], []
        if bid < f + 1e-9:       # bids: above 0, below the most they can still be paid
            orders += ladder(ctl, f, bid, tick, hi - tick, tick, tops[iid]["oBids"], "bid", ctl["maxPos"] - pos, lo, left)
        if ask > f - 1e-9:       # offers: above the least they're sure to be paid
            orders += ladder(ctl, f, ask, lo + tick, hi, tick, tops[iid]["oAsks"], "ask", ctl["maxPos"] + pos, hi, left)
        for side, px, room in (("bid", bid, ctl["maxPos"] + ctl["hardExtra"] - pos),
                               ("ask", ask, ctl["maxPos"] + ctl["hardExtra"] + pos)):
            if any(o[0] == side for o in orders):
                continue
            if ctl["twoSided"] and room > 0 and tick <= px <= hi:
                orders.append((side, px, 1))                     # keep the market two-sided: 1 lot
                notes.append(f"{side}: 1 lot only (limit)")
            else:
                notes.append(f"no {side}" + (" (hard limit)" if ctl["twoSided"] else ""))
        if pos >= ctl["maxPos"]:
            notes.insert(0, f"long {pos} >= max position")
        elif pos <= -ctl["maxPos"]:
            notes.insert(0, f"short {-pos} >= max position")
        if left <= 0:
            notes.insert(0, "budget used up")
        # someone else's bid at/above where we'd offer (or offer at/below where we'd bid): trade on it now
        if ctl["take"]:
            thr_b, thr_a = (x["tb0"], x["ta0"]) if k in tight else (x["b0"], x["a0"])
            thr_b, thr_a = min(thr_b, f - ctl["takeEdge"]), max(thr_a, f + ctl["takeEdge"])
            if ob is not None and ob >= thr_a - 1e-9:
                q = min(tb_["oBidQty"], ctl["takeMax"], ctl["maxPos"] + pos,
                        int(left // max(hi - ob, 1e-9)) if left > 0 else 0)
                if q >= 1:
                    row["take"] = ("ask", ob, int(q))
            elif oa is not None and oa <= thr_b + 1e-9:
                q = min(tb_["oAskQty"], ctl["takeMax"], ctl["maxPos"] - pos,
                        int(left // max(oa - lo, 1e-9)) if left > 0 else 0)
                if q >= 1:
                    row["take"] = ("bid", oa, int(q))
        plan[iid] = orders
        row.update(orders=orders, bid=bid, ask=ask, tight=k in tight, note="; ".join(notes))
    return plan, rows


def close_enough(want, live, tol):
    """Same orders per side (count and sizes) and every price within tol: leave them resting."""
    for side in ("bid", "ask"):
        w = sorted((p, q) for s, p, q in want if s == side)
        l = sorted((o["price"], o["qty"]) for o in live if o["side"] == side)
        if len(w) != len(l) or any(qa != qb or abs(pa - pb) > tol + 1e-9 for (pa, qa), (pb, qb) in zip(w, l)):
            return False
    return True


def blend(st, theo, tops, by_team, feas, w):
    """Our theo blended with the market's: w x ours + (1 - w) x market mid (everyone else's best bid
    and offer, kept inside the payouts still possible), only where both sides are quoted; then
    scaled so the 16 still add up to 310. Returns (blended, mids)."""
    mids = np.full(16, np.nan)
    for k, t in enumerate(st["teams"]):
        tb = tops.get(by_team.get(t)) if by_team.get(t) else None
        if tb and tb["oBids"] and tb["oAsks"] and not tb.get("closed"):
            lo, hi = feas.get(t, [0, 100])
            mids[k] = min(max((tb["oBids"][0] + tb["oAsks"][0]) / 2, lo), hi)
    out = np.where(np.isnan(mids), theo, w * theo + (1 - w) * np.nan_to_num(mids))
    if out.sum() > 0:
        out = out * tpricer.TOTAL / out.sum()
    return out, mids


# --- gateway: make the resting orders match the plan, paced under the rate limit -----------------
class DeskGateway(ps.Gateway):
    def __init__(self, enabled):
        super().__init__(enabled)
        self.plan = {}            # market id -> {contract id: [(side, price, qty)]}
        self.plan_at = {}
        self.quoting = {}         # market id -> bool
        self.last = {}            # (market id, contract id) -> when we last changed it
        self.pulled = {}          # market id -> contract ids pulled one by one
        self.tol = {}             # market id -> re-quote only if a price is off by more than this
        self.news_pause = 5.0     # seconds out after a market event (tick change, news ...)
        self.takes = {}           # market id -> {contract id: (side, price, qty)} crossing our prices
        self.took = {}            # (market, contract, side, price) -> when we last took it
        self.hard = {}            # market id -> hard position limit per contract (max pos + hard extra)
        self.rest_pos = {}        # market id -> positions from the last exchange snapshot
        self.manual = {}          # market id -> order ids typed on the page (the quoter leaves them alone)
        self.rejected = {}        # (market, contract, side, price) -> when the exchange last rejected it

    def pump(self):
        super().pump()                                      # news / market events pull everything ...
        for mid, c in list(self.conns.items()):             # ... but we only sit out NEWS_PAUSE here (not 60 s):
            c["paused_until"] = min(c["paused_until"], time.time() + self.news_pause)   # results have their own button
        for mid, c in list(self.conns.items()):
            if not (self.enabled and self.quoting.get(mid)):
                continue
            if time.time() - self.plan_at.get(mid, 0) > PLAN_MAX_AGE:
                print(f"{time.strftime('%H:%M:%S')}  {mid}: quote plan is stale - pulling everything")
                self.stop(mid)
                continue
            if time.time() >= c["paused_until"]:
                try:
                    self.take_step(mid, c)
                    self.reconcile(mid, c)
                except Exception as e:
                    print("quote error:", e)

    def reconcile(self, mid, c):
        """Make our resting orders match the plan, within the rate budget. Contracts with nothing
        resting go first, then the longest-untouched, so every market gets quoted; a contract
        whose orders are all within TOL of the plan is left alone (no churn)."""
        book, fm, now = c["book"], c["fm"], time.time()
        tol = self.tol.get(mid, 0.0)
        live_by = defaultdict(dict)
        manual = self.manual.get(mid, set())
        for oid, o in book.my_orders.items():
            if oid not in manual:                           # manual orders are the trader's, not ours to move
                live_by[o["iid"]][oid] = o
        items = sorted(self.plan.get(mid, {}).items(),
                       key=lambda it: (bool(live_by.get(it[0])), self.last.get((mid, it[0]), 0)))
        for iid, want in items:
            if now - self.last.get((mid, iid), 0) < 0.5:
                continue
            live = live_by.get(iid, {})
            if close_enough(want, live.values(), tol):
                continue
            if fm.budget() < 2:
                return                                      # paced: the rest next time round
            self.last[(mid, iid)] = now
            live_keys = {(o["side"], round(o["price"], 6), o["qty"]): oid for oid, o in live.items()}
            want_set = {(s, round(p, 6), q) for s, p, q in want}
            for key, oid in live_keys.items():              # cancels first ...
                if key not in want_set:
                    if fm.budget() < 1:
                        return
                    fm.cancel(oid)
            # hard position check on the LIVE position (fills arrive here instantly; the plan's
            # position can be a second old): position + everything resting on that side <= limit
            exposure = {s: sum(q for (s2, px, q) in live_keys if s2 == s and (s2, px, q) in want_set) for s in ("bid", "ask")}
            for s, p, q in want:                            # ... then top of book outwards
                if (s, round(p, 6), q) not in live_keys:
                    if self.long_short(mid, book, iid, s) + exposure[s] + q > self.hard.get(mid, 10 ** 9):
                        continue
                    rkey = (mid, iid, s, round(p, 6))
                    if now - self.rejected.get(rkey, 0) < 30:       # the exchange said no: don't spam it
                        continue
                    if fm.budget() < 1:
                        return

                    def on_ack(ack, rkey=rkey):
                        if not ack.get("ok"):
                            self.rejected[rkey] = time.time()
                    fm.order(iid, s, p, q, on_ack=on_ack)
                    exposure[s] += q

    def stop(self, mid):
        """Quoting off and every order of ours in the market pulled."""
        self.quoting[mid] = False
        self.plan[mid] = {}
        if self.enabled:
            self.conn(mid)["fm"].cancel_all()
            print(f"{time.strftime('%H:%M:%S')}  {mid}: quoting OFF, pulled all my orders")
        return {"ok": True}

    def pull_one(self, mid, iid, off):
        """Pull (or re-allow) one contract's quotes, straight away."""
        p = self.pulled.setdefault(mid, set())
        if off:
            p.add(iid)
            self.plan.setdefault(mid, {})[iid] = []
            if self.enabled:
                self.conn(mid)["fm"].post({"type": "CANCEL_ALL_ON_INSTRUMENT", "instrumentId": iid}, force=True)
                print(f"{time.strftime('%H:%M:%S')}  {mid}: pulled quotes on {self.conn(mid)['book'].name(iid)}")
        else:
            p.discard(iid)
        return {"ok": True}

    def resume(self, mid):
        """End a news pause now (news pulls everything and waits NEWS_PAUSE_SECS)."""
        if mid in self.conns:
            self.conns[mid]["paused_until"] = 0.0
        return {"ok": True}

    def manual_quote(self, mid, iid, bid, ask, qty):
        """A resting bid and/or offer typed on the page. The quoter leaves these alone (it only
        manages its own orders); KILL, Result incoming and a contract's pull button still cancel them."""
        if not self.enabled:
            return {"ok": False, "reason": "paper mode: restart with --trade to send orders"}
        c = self.conn(mid)
        book, fm = c["book"], c["fm"]
        if bid is None and ask is None:
            return {"ok": False, "reason": "enter a bid and/or an offer"}
        if bid is not None and ask is not None and bid >= ask:
            return {"ok": False, "reason": "bid must be below the offer"}
        for side, px in (("bid", bid), ("ask", ask)):
            if px is not None:
                bad = self.check(book, iid, side, px, qty, False)
                if bad:
                    return bad
        res = {}
        if fm.quote(iid, bid=bid, ask=ask, qty=qty, on_ack=lambda a: res.update(ack=a), ) is None:
            return {"ok": False, "reason": "over our rate budget - try again in a second"}
        self.wait(c, lambda: "ack" in res)
        ack = res.get("ack")
        if ack is None:
            return {"ok": False, "reason": "no reply within 2s - check your orders before sending again"}
        if not ack.get("ok"):
            return {"ok": False, "reason": f"rejected: {ack.get('reason')} {ack.get('detail') or ''}".strip()}
        oids = [f"o{s}" for s in ack.get("seqs", [])]
        self.manual.setdefault(mid, set()).update(oids)
        self.wait(c, lambda: False, secs=0.05)
        filled = sum(abs(f["qty"]) for f in book.fills if f["oid"] in oids)
        print(f"{time.strftime('%H:%M:%S')}  {mid}: MANUAL QUOTE {book.name(iid)} {bid} / {ask} x{qty} -> filled {filled}")
        return {"ok": True, "filled": filled}

    def orders(self, mid):
        """Our resting orders (from the live book), marked manual or quoter."""
        if not self.enabled or mid not in self.conns:
            return []
        b, man = self.conns[mid]["book"], self.manual.get(mid, set())
        return [{"oid": oid, "iid": o["iid"], "name": b.name(o["iid"]), "side": o["side"], "price": o["price"],
                 "qty": o["qty"], "manual": oid in man} for oid, o in b.my_orders.items()]

    def long_short(self, mid, book, iid, side):
        """How long (side bid) / short (side ask) we already are, taking the worse of the live
        position from our fills and the exchange's last snapshot (a conductor reset, fills elsewhere)."""
        a, b = book.pos.get(iid, 0), self.rest_pos.get(mid, {}).get(iid, 0)
        return max(a, b) if side == "bid" else -min(a, b)

    def take_step(self, mid, c):
        """Trade on anyone crossing our prices: fill-now-or-cancel at their price, once per 2 s per level."""
        fm, now = c["fm"], time.time()
        for iid, (side, price, qty) in self.takes.get(mid, {}).items():
            key = (mid, iid, side, round(price, 6))
            if now - self.took.get(key, 0) < 2.0 or iid in self.pulled.get(mid, set()) or fm.budget() < 2:
                continue
            qty = min(qty, self.hard.get(mid, 10 ** 9) - self.long_short(mid, c["book"], iid, side))
            if qty < 1:
                continue
            self.took[key] = now

            def done(ack, mid=mid):
                if ack.get("ok") and ack.get("seqs"):
                    self.cancel_now.append((mid, f"o{ack['seqs'][0]}"))      # whatever didn't fill
            fm.order(iid, side, price, qty, on_ack=done)
            print(f"{time.strftime('%H:%M:%S')}  {mid}: TAKE {'BUY' if side == 'bid' else 'SELL'} {qty} "
                  f"{c['book'].name(iid)} @ {price:g} (crossed our price)")

    def set_plan(self, mid, plan, on, tol=0.0, news_pause=5.0, takes=None, hard=10 ** 9, positions=None):
        self.tol[mid], self.news_pause, self.hard[mid] = tol, news_pause, hard
        self.rest_pos[mid] = positions or {}
        self.takes[mid] = takes or {}
        pulled = self.pulled.get(mid, set())
        plan = {iid: ([] if iid in pulled else o) for iid, o in plan.items()}
        self.plan[mid], self.plan_at[mid] = plan, time.time()
        if not on:
            return self.stop(mid) if self.quoting.get(mid) else {"ok": True}
        if not self.quoting.get(mid):
            self.conn(mid)                                  # connect now (snapshot, keepalive)
            print(f"{time.strftime('%H:%M:%S')}  {mid}: quoting ON")
        self.quoting[mid] = True
        return {"ok": True}


gateway = None
STATUS = {"rows": [], "tops": {}, "error": None, "view": None, "at": 0}


def market_fairs(st, M, ctl, q, half, base):
    """Deflected fairs. Own move = DEFLECTION x position x half-spread + any manual deflection typed on
    the page (per team, in price); propagated to every team through
    the strength Jacobian (worked out once per setup in the background; own-team-only until then).
    Kept inside the payouts each team can still get. Returns (fair, propagated?)."""
    theo = base
    move = -ctl["deflection"] * q * half                    # from our position ...
    move = move + np.array([ctl["manualDefl"].get(t, 0.0) for t in st["teams"]])   # ... plus the trader's own
    if not np.any(move):
        return theo, True
    with LOCK:
        J = MODEL.get("J") if MODEL.get("key") == M["key"] else None
    fair = theo + (tpricer.deflect_linear(J, {k: float(m) for k, m in enumerate(move)}) if J is not None else move)
    lo = np.array([M["feasible"].get(t, [0, 100])[0] for t in st["teams"]], dtype=float)
    hi = np.array([M["feasible"].get(t, [0, 100])[1] for t in st["teams"]], dtype=float)
    return np.clip(fair, lo, hi), J is not None


def propagate(st, M):
    """Background job, once per setup/result: how every theo moves when one team's strength moves."""
    try:
        _, J = tpricer.jacobian(st, n=50000, h=0.2, post=M["post"])
        with LOCK:
            if MODEL.get("key") == M["key"]:
                MODEL["J"] = J
    except Exception:
        traceback.print_exc()


def tick(job=None):
    """Read the book, work out the plan, hand it to the gateway. Returns the propagation job."""
    ctl = controls()
    mid = ctl["market"]
    if not mid:
        return job
    st, M = model()
    view = ps.snapshot(mid, max_age=0.5)
    tops = book_tops(view)
    names = {i["id"]: i["name"] for i in view["instruments"]}
    by_team = {ctl["mapping"][n]: iid for iid, n in names.items() if n in ctl["mapping"]}
    positions = {i["id"]: i["position"] for i in view["instruments"]}
    q = np.array([positions.get(by_team.get(t), 0) for t in st["teams"]], dtype=float)
    model_half = ctl["k"] * M["sim"]["sigmaNext"] + ctl["c"]
    tight = tight_teams(st, model_half, by_team, tops, M["feasible"], ctl)
    half = quoted_half(st, model_half, tight, ctl)
    blended, mids = blend(st, M["sim"]["theo"], tops, by_team, M["feasible"], ctl["ourWeight"])
    fair, done = market_fairs(st, M, ctl, q, half, blended)
    if MODEL.get("J") is None and (job is None or not job.is_alive()):
        job = threading.Thread(target=propagate, args=(st, M), daemon=True)
        job.start()
    lim = view.get("positionLimits") or {}
    if lim.get("enabled"):                              # the exchange's own position limit wins
        ctl = {**ctl, "maxPos": min(ctl["maxPos"], int(lim.get("defaultLimit", ctl["maxPos"])))}
    plan, rows = plan_quotes(st, M["feasible"], fair, model_half, tops, by_team, positions, ctl)
    for k, r in enumerate(rows):
        r["mid"] = None if np.isnan(mids[k]) else float(mids[k])
        r["blend"] = float(blended[k])
    ok = not check_mapping(ctl["mapping"], st["teams"], list(names.values()))   # Quoting ON is the confirmation
    if gateway.enabled:
        takes = {r["iid"]: r["take"] for r in rows if r.get("take") and ok}
        gateway.call(gateway.set_plan, mid, plan if ok else {}, bool(ctl["quoting"] and ok), ctl["requoteMin"],
                     ctl["newsPause"], takes, ctl["maxPos"] + (ctl["hardExtra"] if ctl["twoSided"] else 0), positions)
    STATUS.update(rows=rows, tops=tops, view=view, byTeam=by_team, propagated=done, error=None, at=time.time(),
                  risk=risk_used(st, M["feasible"], fair, by_team, positions))
    return job


def brain():
    """Every second."""
    job = None
    while True:
        t0 = time.time()
        try:
            job = tick(job)
        except ApiError as e:
            STATUS["error"] = str(e)
        except Exception as e:
            traceback.print_exc()
            STATUS["error"] = f"{type(e).__name__}: {e}"
        time.sleep(max(0.2, 1.0 - (time.time() - t0)))


def fixtures(st, slot):
    res = st.get("results", {})
    out = []
    for rnd_i, rnd in enumerate(tpricer.ROUNDS):
        for g in tpricer.GROUPS:
            for p, q in rnd:
                a, b = st["groups"][g][p], st["groups"][g][q]
                mid = f"{g}:{a}-{b}"
                out.append({"id": mid, "stage": f"Group {g} · R{rnd_i + 1}", "a": a, "b": b, "res": res.get(mid)})
    names = {"C": "Crossover", "Q": "Quarter-final", "S": "Semi-final", "F": "Final", "T": "3rd place"}
    src = {m: ("winner " + x if x[0] in "CQ" else x, "winner " + y if y[0] in "CQ" else y)
           for m, x, y in tpricer.CROSS + tpricer.QF + tpricer.SF}
    src.update(F=("winner SF1", "winner SF2"), T=("loser SF1", "loser SF2"))
    for mid in tpricer.KO_ORDER:
        a, b = slot.get(mid, [None, None])
        out.append({"id": mid, "stage": f"{names[mid[0]]} {mid}", "a": a, "b": b, "res": res.get(mid), "ko": True,
                    "srcA": src[mid][0], "srcB": src[mid][1]})
    return out


def instruments(view, ctl, fairs):
    """Every instrument in the market straight from the API (mapped or not): book, our orders, position, P&L."""
    names = {i["id"]: i["name"] for i in view["instruments"]}
    cash = defaultdict(float)
    for t in view.get("myTrades", []):
        cash[t["instrumentId"]] -= t["signedQty"] * from_px(t["price"])
    out = []
    for inst in view["instruments"]:
        iid, pos = inst["id"], inst["position"]

        def side(levels):
            return [{"price": from_px(l["price"]), "qty": l["qty"],
                     "mine": sum(o["qty"] for o in l["orders"] if o.get("mine"))} for l in levels]
        bids, asks = side(inst["bids"]), side(inst["asks"])
        settle = from_px(inst.get("settlementValue"))
        mid = (bids[0]["price"] + asks[0]["price"]) / 2 if bids and asks else None
        fair = fairs.get(iid)
        last = from_px(inst.get("lastTrade"))
        mark = lambda m: None if m is None else cash[iid] + pos * m
        # mark to market: settlement, else the market mid, else the last trade, else our fair
        mtm_px, mtm_src = next(((p, s) for p, s in ((settle, "settled"), (mid, "mid"), (last, "last"), (fair, "fair"))
                                if p is not None), (None, ""))
        out.append({"iid": iid, "name": inst["name"], "team": ctl["mapping"].get(inst["name"]),
                    "mtmPx": mtm_px, "mtmSrc": mtm_src, "pnlMtm": mark(mtm_px) if pos or cash[iid] else 0.0,
                    "bids": bids[:3], "asks": asks[:3], "last": from_px(inst.get("lastTrade")),
                    "myBids": [(l["price"], l["mine"]) for l in bids if l["mine"]],
                    "myAsks": [(l["price"], l["mine"]) for l in asks if l["mine"]],
                    "pos": pos, "avg": -cash[iid] / pos if pos else None, "cash": cash[iid],
                    "fair": fair, "mid": mid, "settle": settle, "closed": inst.get("closed"),
                    "pnlFair": mark(settle if settle is not None else fair),
                    "pnlMid": mark(settle if settle is not None else mid)})
    trades = [{"at": t.get("at"), "name": names.get(t["instrumentId"], t["instrumentId"]), "qty": t["signedQty"],
               "price": from_px(t["price"])} for t in view.get("myTrades", [])][::-1][:20]
    reb = (view.get("rebate") or {})
    return {"rows": out, "trades": trades, "units": view.get("units", ""), "myPnl": view.get("myPnl"),
            "exchangeMtm": from_px(view.get("myMtm")),
            "limits": view.get("positionLimits"), "rebate": {"enabled": reb.get("enabled"), "width": from_px(reb.get("width")),
                                                            "coverage": reb.get("coveragePct"), "mine": reb.get("mine")}}


def state_json():
    st, M = model()
    ctl = controls()
    sim = M["sim"]
    view = STATUS.get("view") if ctl["market"] else None
    names = [i["name"] for i in view["instruments"]] if view else []
    srows = {r["team"]: r for r in STATUS.get("rows") or []} if view else {}
    tops = STATUS.get("tops") or {}
    by_team = (STATUS.get("byTeam") or {}) if view else {}
    half = ctl["k"] * sim["sigmaNext"] + ctl["c"]
    insts = instruments(view, ctl, {r["iid"]: r["fair"] for r in srows.values() if r.get("iid") and r.get("fair") is not None}) if view else None
    rows = []
    for k, t in enumerate(st["teams"]):
        r = srows.get(t, {})
        iid = by_team.get(t)
        tb = tops.get(iid, {})
        pos = r.get("pos", 0)
        im = next((x for x in (insts or {}).get("rows", []) if x["iid"] == iid), None) if iid else None
        rows.append({"team": t, "seed": k + 1, "group": next(g for g in tpricer.GROUPS if t in st["groups"][g]),
                     "nudge": st["nudges"].get(t, 0), "S": float(sim["mean"][k]), "sdS": float(sim["sdS"][k]),
                     "finish": [float(x) for x in sim["finish"][k]], "pQF": float(sim["pQF"][k]),
                     "pSF": float(sim["pSF"][k]), "pF": float(sim["pF"][k]), "pWin": float(sim["pWin"][k]),
                     "theo": float(sim["theo"][k]), "sd": float(sim["sd"][k]), "sigmaNext": float(sim["sigmaNext"][k]),
                     "half": r.get("half", float(half[k])), "tight": bool(r.get("tight")),
                     "mid": r.get("mid"), "blend": r.get("blend"), "note": r.get("note", ""), "take": r.get("take"),
                     "fair": r.get("fair"), "range": M["feasible"].get(t),
                     "pos": pos, "iid": iid, "contract": next((n for n, x in ctl["mapping"].items() if x == t), None),
                     "orders": r.get("orders", []), "why": r.get("why", ""), "ourBid": r.get("bid"), "ourAsk": r.get("ask"),
                     "bid": tb.get("bid"), "ask": tb.get("ask"), "tick": tb.get("tick"),
                     "pnl": im["pnlMtm"] if im else None})
    auto, auto_probs = auto_map(names, st["teams"]) if names else ({}, [])
    cover = None
    if insts:                                            # what's actually resting right now vs the rebate rule
        op = [r for r in insts["rows"] if not r["closed"] and r["settle"] is None]
        two = [r for r in op if r["myBids"] and r["myAsks"]]
        within = [r for r in two if min(p for p, _ in r["myAsks"]) - max(p for p, _ in r["myBids"]) <= ctl["rebateWidth"] + 1e-9]
        cover = {"open": len(op), "twoSided": len(two), "within": len(within),
                 "need": math.ceil(ctl["coverage"] * len(op) - 1e-9), "planned": sum(1 for r in srows.values() if r.get("tight"))}
    my_orders = gateway.call(gateway.orders, ctl["market"], timeout=2) if gateway and gateway.enabled and ctl["market"] else []
    return {"instruments": insts, "cover": cover, "myOrders": my_orders if isinstance(my_orders, list) else [],"state": {k: st[k] for k in ("teams", "groupMode", "groups", "pBeat", "goals", "kappa", "nudges",
                                          "confidence", "results", "sims", "ratings")},
            "controls": ctl, "rows": rows, "readouts": M["readouts"], "n": sim["n"], "simMs": M["ms"],
            "sumTheo": float(sim["theo"].sum()), "presets": tpricer.PRESETS,
            "api": {"read": bool(view) and not STATUS.get("error") and time.time() - STATUS["at"] < 5,
                    "orders": bool(gateway and gateway.enabled and ctl["market"] in gateway.conns)},
            "contracts": names, "autoMap": auto, "autoProblems": auto_probs,
            "mappingProblems": check_mapping(ctl["mapping"], st["teams"], names) if names else ["no market loaded"],
            "unmappedTeams": unmapped_teams(ctl["mapping"], st["teams"], names) if names else [],
            "propagated": STATUS.get("propagated", True), "error": STATUS.get("error") if ctl["market"] else None,
            "trading": bool(gateway and gateway.enabled),
            "quotingLive": bool(gateway and gateway.quoting.get(ctl["market"])),
            "pausedFor": max(0.0, gateway.conns[ctl["market"]]["paused_until"] - time.time())
                         if gateway and ctl["market"] in gateway.conns else 0.0,
            "risk": STATUS.get("risk", 0.0) if view else 0.0,
            "market": view["name"] if view else None, "status": view.get("status") if view else None,
            "age": round(time.time() - STATUS["at"], 1) if STATUS["at"] else None,
            "fixtures": fixtures(st, M["slots"]),
            "totalPnl": sum(r["pnlMtm"] or 0 for r in insts["rows"]) if insts else 0}


class Handler(BaseHTTPRequestHandler):
    def send_json(self, obj, code=200):
        body = json.dumps(obj, default=float).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        url = urlparse(self.path)
        if url.path == "/api/state":
            try:
                return self.send_json(state_json())
            except Exception as e:
                traceback.print_exc()
                return self.send_json({"error": f"{type(e).__name__}: {e}"}, 500)
        m = re.fullmatch(r"/static/([a-z0-9_]+\.(js|css))", url.path)
        if m:                                   # the page's parts: core, sidebar and one file per tab
            path = os.path.join(STATIC_DIR, m.group(1))
            if not os.path.isfile(path):
                return self.send_error(404)
            with open(path, "rb") as f:
                body = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "text/javascript; charset=utf-8" if m.group(2) == "js" else "text/css; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            return self.wfile.write(body)
        if url.path in ("/", "/index.html"):
            with open(HTML_FILE, encoding="utf-8") as f:
                body = f.read().replace("__TRADE_TOKEN__", TOKEN).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            return self.wfile.write(body)
        self.send_error(404)

    def do_POST(self):
        if self.headers.get("Host") not in ALLOWED_HOSTS or self.headers.get("X-Trade-Token") != TOKEN:
            return self.send_json({"ok": False, "reason": "forbidden"}, 403)
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        except ValueError:
            return self.send_json({"ok": False, "reason": "bad request"}, 400)
        path = urlparse(self.path).path
        mid = controls()["market"]
        try:
            if path == "/api/setup":                       # teams, groups, strength
                st = tpricer.load_state()
                if "teams" in body and st.get("results") and body["teams"] != st["teams"]:
                    raise ValueError("results are entered: clear them before changing the teams")
                for k in ("teams", "groupMode", "groups", "pBeat", "goals", "kappa", "nudges", "confidence", "sims", "ratings"):
                    if k in body:
                        st[k] = body[k]
                tpricer.save_state(st)
                model()
                return self.send_json({"ok": True})
            if path == "/api/result":                      # enter / clear a match result
                st = tpricer.load_state()
                r = body.get("result")
                if r is None:
                    st["results"].pop(body["id"], None)
                else:
                    st["results"][body["id"]] = r
                tpricer.save_state(st)
                model()
                return self.send_json({"ok": True})
            if path == "/api/pull":                        # one team's quotes off / back on
                team, off = str(body["team"]), bool(body.get("off", True))
                ctl = controls()
                save_controls({"off": (set(ctl["off"]) | {team}) if off else (set(ctl["off"]) - {team})})
                iid = (STATUS.get("byTeam") or {}).get(team)
                if gateway.enabled and mid and iid:
                    return self.send_json(gateway.call(gateway.pull_one, mid, iid, off))
                return self.send_json({"ok": True})
            if path == "/api/resume":
                if gateway.enabled and mid:
                    return self.send_json(gateway.call(gateway.resume, mid))
                return self.send_json({"ok": True})
            if path == "/api/controls":
                return self.send_json({"ok": True, "controls": save_controls(body.get("set") or {})})
            if path in ("/api/incoming", "/api/kill"):     # pull everything NOW
                save_controls({"quoting": False})
                if gateway.enabled and mid:
                    return self.send_json(gateway.call(gateway.stop, mid))
                return self.send_json({"ok": True})
            if path == "/api/quote":                       # manual resting bid and/or offer
                num = lambda v: None if v in (None, "") else float(v)
                qty = body.get("qty")
                return self.send_json(gateway.call(gateway.manual_quote, mid, body.get("iid"), num(body.get("bid")),
                                                   num(body.get("ask")), qty if isinstance(qty, int) else -1))
            if path == "/api/cancel":                      # one order
                return self.send_json(gateway.call(gateway.cancel, mid, str(body.get("oid"))))
            if path == "/api/order":
                qty = body.get("qty")
                return self.send_json(gateway.call(gateway.place, mid, body.get("iid"), body.get("side"),
                                                   qty if isinstance(qty, int) else -1, body.get("price"), "ioc", False))
        except (ValueError, KeyError, TypeError) as e:
            return self.send_json({"ok": False, "reason": str(e)})
        self.send_error(404)

    def log_message(self, *args):
        pass


def main():
    global gateway
    trading = "--trade" in sys.argv
    if trading:
        print("*** TRADING ENABLED: the page at localhost:8003 can send REAL orders. ***")
        if input("Type LIVE to continue: ").strip().lstrip("﻿") != "LIVE":
            raise SystemExit("not confirmed - exiting")
    ps.gateway = gateway = DeskGateway(trading)
    gateway.start()
    save_controls({"quoting": False})                      # never starts quoting by itself
    model()
    threading.Thread(target=brain, daemon=True).start()
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"tournament desk on http://localhost:{PORT}  ({'LIVE' if trading else 'paper'}; Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("stopped - the exchange will pull any API orders this server placed (~5 s)")


if __name__ == "__main__":
    main()
