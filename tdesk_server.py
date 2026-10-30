"""Tournament desk (brief 1): pricer + quoter + dashboard for the 16-team tournament.

    python tdesk_server.py            # paper: everything works, the quotes are only shown
    python tdesk_server.py --trade    # live: can send orders (asks you to type LIVE)
    then open http://localhost:8003

Fair values, per team (numbers from tpricer.py and the controls on the page):
    our   = model theo + the trader's edits. An edit pins that team; every other team moves by it
            covariance with the pinned ones (most correlated move most), so totals that are fixed
            (all 16 = 310, the two finalists = 175 ...) never move. Edits are dropped on each result.
    theo  = (1 - w) x the other market makers' mid + w x our, w = OUR SHARE (our fair alone where nobody quotes), n = market makers quoting both sides.
    PA    = theo - gamma x (Sigma q) + the trader's PA edits (propagated the same way) + the PA - theo
            gap kept through the last result. Sigma q: our position times the payout covariance, so
            being long England lowers England AND raises whoever England's payout trades off against.
            gamma is set so a share of a typical market moves its own PA by DEFL.
    known payout (lo == hi from the results): theo = PA = that value, quoted 0.2 wide around it.
Quoting, per mapped contract: one 1-lot bid and offer (plan_quotes has the rules). Starts START WIDTH
    wide after every result and narrows to the MM spread over NARROW SECS; copies the other market
    makers' top of book while our PA is inside it. Free money (a known payout mispriced, a price
    outside the range a team can still pay) is always taken.
    Traded against: that side is pulled and stays off until "quotes ON" is clicked for that market;
    the position moves PA so the other side gets more competitive.
"""
import difflib
import json
import math
import os
import re
import secrets
import socket
import sys
import threading
import time
import traceback
from collections import defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np

import deskgate as ps                  # REST snapshot cache, trading gateway
import pricing
import tpricer
from mockx import ApiError, from_px

PORT = int(os.environ.get("TDESK_PORT", 8003))     # set TDESK_PORT to run a second copy alongside
HTML_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tdesk.html")
TOKEN_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tdesk_token.txt")
KNOWN_WIDTH = 0.2                        # width of our market around a payout the results have fixed


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
SHARE = "--share" in sys.argv            # other laptops on the network may use this desk (with the link)
PLAN_MAX_AGE = 5.0                       # the gateway pulls everything if the plan is older than this
QUOTE_SIZE = 1                           # one bid and one offer of 1 lot per market
CONTROLS_DEFAULT = {"ourShare": 0.5, "fatPct": 0.3, "fatMin": 3.0,"deflPct": 0.5, "autoDefl": True, "deflCount": 3, "maxPosAdj": {}, "k": 1.0, "c": 1.0, "maxPos": 10,"startWidth": 20.0, "narrowSecs": 600.0,
                    "wideSince": 0.0, "quoting": False, "clickQty": 1, "market": "", "mapping": {}, "mappingOk": False,
                    "budget": 2000.0, "off": [], "rebateWidth": 2.9, "nTight": 14,
                    "requoteMin": 0.0, "newsPause": 5.0, "hardExtra": 5,
                    "take": True, "takeEdge": 1.0, "takeMax": 5, "theoAdj": {}, "paAdj": {}, "paKeep": {},
                    "edgeAdj": {}, "hidden": [], "layout": {}, "deflPos": {}, "deflAmt": {}}

LOCK = threading.Lock()
STATE_LOCK = threading.RLock()           # one writer at a time to tdesk_state.json (several people on one desk)
MODEL = {"key": None}


# --- saved state: tpricer's tdesk_state.json, plus our "controls" ---------------------------------
def controls():
    saved = tpricer.load_state().get("controls", {})
    return {k: saved.get(k, d) for k, d in CONTROLS_DEFAULT.items()}


def save_controls(changes):
    with STATE_LOCK:
        return _save_controls(changes)


def _save_controls(changes):
    st = tpricer.load_state()
    c = {**CONTROLS_DEFAULT, **{k: v for k, v in st.get("controls", {}).items() if k in CONTROLS_DEFAULT}}
    for k, v in changes.items():
        if k not in CONTROLS_DEFAULT:
            continue
        d = CONTROLS_DEFAULT[k]
        if k in ("theoAdj", "paAdj", "paKeep", "edgeAdj"):
            c[k] = {str(a): float(b) for a, b in (v or {}).items() if isinstance(b, (int, float)) and abs(b) <= 100}
        elif k in ("deflPos", "deflAmt"):
            c[k] = {str(a): float(b) for a, b in (v or {}).items() if isinstance(b, (int, float))}
        elif k == "maxPosAdj":
            c[k] = {str(a): int(b) for a, b in (v or {}).items() if isinstance(b, (int, float)) and 0 <= b <= 1000}
        elif k == "layout":
            c[k] = v if isinstance(v, dict) else {}
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
    tpricer._write(st)
    return c


def model():
    """The simulation for the current setup + results (re-run only when they change)."""
    st = tpricer.load_state()
    key = json.dumps({k: st.get(k) for k in ("teams", "tiers", "regions", "ranks", "pBeat", "bo", "confidence",
                                              "results", "sims", "ratings")}, sort_keys=True)
    with LOCK:
        if MODEL["key"] != key:
            t0 = time.time()
            post = tpricer.posterior(st)
            sim = tpricer.simulate(st, post=post)
            pay = sim.pop("pay")
            # pairs whose combined payout can't move, (long X + long Y) or (long X + short Y), and what it is
            n = pay.shape[1]
            lock_sum, lock_diff = np.zeros((n, n), bool), np.zeros((n, n), bool)
            for i in range(n):
                lock_sum[i] = np.ptp(pay[:, [i]] + pay, axis=0) < 1e-9
                lock_diff[i] = np.ptp(pay[:, [i]] - pay, axis=0) < 1e-9
            MODEL.clear()
            MODEL.update(key=key, sim=sim, post=post, readouts=tpricer.readouts(st, sim),
                         feasible=tpricer.feasible(st), lockSum=lock_sum, lockDiff=lock_diff,
                         sumVal=pay[0][:, None] + pay[0][None, :], diffVal=pay[0][:, None] - pay[0][None, :],
                         pairMin=np.array([(pay[:, [i]] + pay).min(axis=0) for i in range(n)]),    # what each pair of
                         pairMax=np.array([(pay[:, [i]] + pay).max(axis=0) for i in range(n)]),    # teams can pay together
                         ms=round((time.time() - t0) * 1000))
        return st, dict(MODEL)


# --- fair values ---------------------------------------------------------------------------------
def market_mids(st, tops, by_team, view):
    """Per team: everyone else's top-of-book mid (nan if not two-sided) and n = how many other
    market makers are quoting both sides there (1 if the exchange doesn't say who owns orders)."""
    mid, n = np.full(len(st["teams"]), np.nan), np.zeros(len(st["teams"]))
    insts = {i["id"]: i for i in (view or {}).get("instruments", [])}
    for k, t in enumerate(st["teams"]):
        tb, inst = tops.get(by_team.get(t)), insts.get(by_team.get(t))
        if not tb or not tb["oBids"] or not tb["oAsks"] or tb.get("closed"):
            continue
        mid[k] = (tb["oBids"][0] + tb["oAsks"][0]) / 2
        def owners(side):
            return {o.get("owner") for l in inst[side] for o in l["orders"] if not o.get("mine") and o.get("owner")}
        both = owners("bids") & owners("asks")
        n[k] = len(both) or 1
    return mid, n


def update_defl_book(st, ctl, q, width):
    """Deflection book, per team: our position and D = how far the shares we hold have moved our PA:
    each share adds DEFL PCT x the edge (half our width) at the moment we got it. Adding shares adds
    to D; reducing scales D down in proportion (flat -> 0); going through zero starts again at today's
    edge. The switch and the % only change what FUTURE trades add: with AUTO DEFLECTION off, trades
    leave D alone (we deflect by hand) - nothing already in D moves when either is changed.
    Saved when it changes. Returns the deflAmt dict."""
    pos, amt, changed = dict(ctl["deflPos"]), dict(ctl["deflAmt"]), False
    for k, t in enumerate(st["teams"]):
        new, old, d = float(q[k]), pos.get(t, 0.0), amt.get(t, 0.0)
        if new == old:
            continue
        pos[t], changed = new, True
        if not ctl["autoDefl"]:
            continue
        e = ctl["deflPct"] * float(width[k]) / 2 if not np.isnan(width[k]) else (d / old if old else 0.0)
        if old == 0 or (old > 0) == (new > 0) and abs(new) > abs(old):
            d += e * (new - old)
        elif new == 0 or (old > 0) == (new > 0):
            d *= new / old
        else:
            d = e * new
        amt[t] = d
    if changed:
        save_controls({"deflPos": {t: p for t, p in pos.items() if p}, "deflAmt": {t: a for t, a in amt.items() if a}})
    return amt


def fairs(st, M, ctl, mkt=None):
    """Per team (see the top of the file):
    base  model theo;  our = base + our theo edits (propagated by covariance)
    theo  = (1 - w) x the other market makers' mid + w x our, w = OUR SHARE (our fair alone where nobody quotes)
    pa0   = theo + deflection + the PA offset kept through the last result;  pa = pa0 + PA edits.
    deflection: team j's own PA moves by D[j] against our position (D: update_defl_book - make 9 at
    11, get lifted, 50% -> PA 10.5), and every other team by its covariance with j:
    - Sigma[:, j] x D[j] / Sigma[j, j]. Each column of Sigma adds up to 0 over any set whose total
    payout is fixed, so those totals never move."""
    sim, feas = M["sim"], M["feasible"]
    lo = np.array([feas.get(t, [0, 100])[0] for t in st["teams"]], dtype=float)
    hi = np.array([feas.get(t, [0, 100])[1] for t in st["teams"]], dtype=float)
    known = lo == hi
    cov = sim["cov"].copy()
    cov[known, :] = 0
    cov[:, known] = 0
    idx = {t: k for k, t in enumerate(st["teams"])}
    clip = lambda x: np.where(known, lo, np.clip(x, lo, hi))
    base = clip(sim["theo"])
    our = clip(base + tpricer.propagate(cov, {idx[t]: d for t, d in ctl["theoAdj"].items() if t in idx}))
    mid, n = mkt if mkt is not None else (np.full(len(base), np.nan), np.zeros(len(base)))
    w = min(max(float(ctl["ourShare"]), 0.0), 1.0)                # weight on our fair; the rest on the market mid
    has = ~np.isnan(mid) & ~known
    theo = np.where(has, (1 - w) * np.clip(np.nan_to_num(mid), lo, hi) + w * our, our)
    var = np.diag(cov)
    D = np.array([ctl["deflAmt"].get(t, 0.0) for t in st["teams"]])
    defl = -(cov @ np.where(var > 1e-9, D / np.where(var > 1e-9, var, 1.0), 0.0))
    keep = np.array([ctl["paKeep"].get(t, 0.0) for t in st["teams"]])
    pa0 = theo + defl + keep
    pa = pa0 + tpricer.propagate(cov, {idx[t]: d for t, d in ctl["paAdj"].items() if t in idx})
    return {"base": base, "our": our, "theo": clip(theo), "pa0": clip(pa0), "pa": clip(pa), "n": n, "mid": mid}


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
    """Problems that block quoting. Teams without a contract are allowed (just not quoted)."""
    probs = [f"contract '{n}' not mapped" for n in names if n not in mapping]
    vals = [mapping[n] for n in names if n in mapping]
    probs += [f"{t} mapped twice" for t in sorted(set(vals)) if vals.count(t) > 1]
    probs += [f"'{v}' is not a team" for v in vals if v not in teams]
    return probs


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


def risk_used(st, feas, fair, by_team, positions):
    """What our positions can still lose from fair: long -> down to the lowest payout, short -> up to the highest."""
    total = 0.0
    for k, t in enumerate(st["teams"]):
        q = positions.get(by_team.get(t), 0) if by_team.get(t) else 0
        lo, hi = feas.get(t, [0, 100])
        total += q * (fair[k] - lo) if q > 0 else -q * (hi - fair[k])
    return float(total)


def n_tight(tops, ctl):
    """How many contracts to quote within the MM spread: N TIGHT, at most the open ones."""
    n_open = sum(1 for x in tops.values() if not x.get("closed"))
    return max(0, min(n_open, int(ctl["nTight"])))


def known_quotes(v, tick):
    """A payout fixed at v: bid / offer KNOWN_WIDTH apart around it, kept inside 0..100."""
    bid, ask = pricing.round_down(v - KNOWN_WIDTH / 2, tick), pricing.round_up(v + KNOWN_WIDTH / 2, tick)
    if bid < 0:
        bid, ask = 0.0, pricing.round_up(KNOWN_WIDTH, tick)
    if ask > 100:
        bid, ask = pricing.round_down(100 - KNOWN_WIDTH, tick), 100.0
    return round(bid, 9), round(ask, 9)


def width_now(ctl, target, now):
    """Our market width: START WIDTH right after a result, narrowing in a straight line to target
    over NARROW SECS."""
    frac = max(0.0, 1 - (now - ctl["wideSince"]) / max(ctl["narrowSecs"], 1e-9))
    return target + max(ctl["startWidth"] - target, 0.0) * frac


def market_widths(st, feas, half, tops, by_team, ctl, now):
    """(our width per team - nan where we don't quote or the payout is known -, known teams, tight teams).
    The trader's edge for a market wins; otherwise width_now() towards the MM SPREAD for the N TIGHT
    smallest-model-spread markets, 2 x (k * sigma_next + c) (at least the MM spread) for the rest."""
    minw = ctl["rebateWidth"]
    live = [k for k, t in enumerate(st["teams"]) if by_team.get(t) and not tops[by_team[t]].get("closed") and t not in ctl["off"]]
    known = {k for k in live if feas.get(st["teams"][k], [0, 100])[0] == feas.get(st["teams"][k], [0, 100])[1]}
    rest = sorted((k for k in live if k not in known), key=lambda k: half[k])
    tight = known | set(rest[:max(0, n_tight(tops, ctl) - len(known))])
    width = np.full(len(st["teams"]), np.nan)
    for k in rest:
        t, tick = st["teams"][k], tops[by_team[st["teams"][k]]]["tick"]
        edge = ctl["edgeAdj"].get(t)
        width[k] = (max(2 * edge, 2 * tick) if edge is not None
                    else width_now(ctl, minw if k in tight else max(minw, 2 * float(half[k])), now))
    return width, known, tight


def max_pos(ctl, team):
    """This team's position limit: its own (Position management) or the global one."""
    return int(ctl["maxPosAdj"].get(team, ctl["maxPos"]))


def pair_limits(k, st, by_team, tops, pairs, iid):
    """(highest bid, lowest offer) we can show on team k so nobody can lock in a profit against us by
    trading it together with one other team at that team's best price (ours included): our bid +
    their best bid <= the most the two can pay out together, our offer + their best offer >= the least.
    pairs = (pairMin, pairMax) from the simulation. Also returns which team binds, for the notes."""
    cap, floor, why = 1e9, -1e9, []
    if pairs is None:
        return cap, floor, why
    pmin, pmax = pairs
    for j, t in enumerate(st["teams"]):
        jid = by_team.get(t)
        if j == k or not jid or jid == iid or tops[jid].get("closed"):
            continue
        b, a = tops[jid].get("bid"), tops[jid].get("ask")
        if b is not None and pmax[k, j] - b < cap:
            cap = pmax[k, j] - b
            why.append(f"bid capped by {t}")
        if a is not None and pmin[k, j] - a > floor:
            floor = pmin[k, j] - a
            why.append(f"offer floored by {t}")
    return cap, floor, why


def plan_quotes(st, feas, fair, half, tops, by_team, positions, ctl, blocked=None, now=None, pairs=None):
    """Every order we want resting: {contract id: [(side, price, qty)]} - one 1-lot bid and offer each -
    plus a row per team (and row["take"]: a price someone else shows that we trade on now).

    fair is the PA theo. Width = width_now(): wide after a result, narrowing to the MM SPREAD (the N
    TIGHT smallest-model-spread markets) or to 2 x (k * sigma_next + c) (the rest).
      PA inside everyone else's top of book -> copy it (or sit inside it once our width is tighter);
      otherwise our own market around PA, and take their price if it gives TAKE EDGE against PA.
    Never narrower than the MM spread, never crossing anyone, never outside the payouts still possible.
    Known payout: quoted KNOWN_WIDTH around it. Always taken (TAKE on): a known payout offered below /
    bid above it, and any offer below / bid above the range a team can still pay.
    blocked {contract id: {"bid"/"ask"}}: sides we were traded on - nothing there until re-enabled."""
    blocked, now = blocked or {}, time.time() if now is None else now
    plan, rows = {}, []
    left = ctl["budget"] - risk_used(st, feas, fair, by_team, positions)
    o_bid = {iid: t["oBids"][0] for iid, t in tops.items() if t["oBids"]}
    o_ask = {iid: t["oAsks"][0] for iid, t in tops.items() if t["oAsks"]}
    minw = ctl["rebateWidth"]
    width, known, tight = market_widths(st, feas, half, tops, by_team, ctl, now)
    for k, t in enumerate(st["teams"]):
        iid = by_team.get(t)
        lo, hi = feas.get(t, [0, 100])
        f = float(fair[k])
        pos = positions.get(iid, 0) if iid else 0
        row = {"team": t, "iid": iid, "fair": f, "orders": [], "why": "", "pos": pos, "note": ""}
        rows.append(row)
        if not iid:
            row["why"] = "no contract"
            continue
        if tops[iid].get("closed") or t in ctl["off"]:
            row["why"] = "pulled" if t in ctl["off"] else "contract closed / settled"
            plan[iid] = []
            continue
        tb, tick = tops[iid], tops[iid]["tick"]
        mp = max_pos(ctl, t)
        ob, oa = o_bid.get(iid), o_ask.get(iid)
        notes, take, cap_bid, cap_ask = [], None, 1e9, -1e9
        if k in known:
            bid, ask = known_quotes(lo, tick)
            mode, notes = "known", [f"payout known: {lo:g}"]
        else:
            w, floor = float(width[k]), 0.0 if t in ctl["edgeAdj"] else minw     # the trader's own edge: no floor
            if ob is not None and oa is not None and ob < f < oa:
                bid, ask, mode = max(ob, f - w / 2), min(oa, f + w / 2), "copy"
            else:
                bid, ask, mode = f - w / 2, f + w / 2, "own"
            bid, ask = pricing.round_down(bid, tick), pricing.round_up(ask, tick)
            if ask - bid < floor - 1e-9:                         # never tighter than the MM spread
                m = (bid + ask) / 2
                bid, ask = pricing.round_down(m - floor / 2, tick), pricing.round_up(m + floor / 2, tick)
            others = [x for x in by_team.values() if x != iid]
            # 310 self-check: nobody may sell the whole field for more than 310 by hitting our bid and
            # everyone else's bids, or buy it for less than 310 by lifting our offer and everyone else's.
            if len(others) == len(st["teams"]) - 1 and all(x in o_bid for x in others):
                cap_bid = pricing.round_down(tpricer.TOTAL - sum(o_bid[x] for x in others), tick)
            if len(others) == len(st["teams"]) - 1 and all(x in o_ask for x in others):
                cap_ask = pricing.round_up(tpricer.TOTAL - sum(o_ask[x] for x in others), tick)
            pcap, pfloor, pwhy = pair_limits(k, st, by_team, tops, pairs, iid)
            if bid > pcap + 1e-9 or ask < pfloor - 1e-9:
                notes.append("pair check: " + ", ".join(w for w in pwhy if (w.startswith("bid") and bid > pcap) or (w.startswith("offer") and ask < pfloor))[:80])
            cap_bid = min(cap_bid, pricing.round_down(pcap, tick))
            cap_ask = max(cap_ask, pricing.round_up(pfloor, tick))
            bid, ask = min(bid, cap_bid), max(ask, cap_ask)      # (an offer pushed above the max payout is dropped below)
            if mode == "own" and ctl["take"]:                    # their price beats our PA by enough: trade it
                if ob is not None and ob >= f + ctl["takeEdge"] - 1e-9:
                    take = ("ask", ob, min(tb["oBidQty"], ctl["takeMax"], mp + pos))
                elif oa is not None and oa <= f - ctl["takeEdge"] + 1e-9:
                    take = ("bid", oa, min(tb["oAskQty"], ctl["takeMax"], mp - pos))
            # inside what the team can still pay: never bid below its lowest payout or offer above its
            # highest (both are risk-free - and keep us tight for the rebate once results narrow the range)
            bid, ask = max(min(bid, hi - tick), lo), max(min(ask, hi), lo + tick)
        if ctl["take"]:                                          # free money: outside what it can still pay
            hard = mp + ctl["hardExtra"]
            if ob is not None and ob > hi + 1e-9:
                take = ("ask", ob, min(tb["oBidQty"], ctl["takeMax"], hard + pos))
                notes.append(f"bid {ob:g} above max payout {hi:g}: selling")
            elif oa is not None and oa < lo - 1e-9:
                take = ("bid", oa, min(tb["oAskQty"], ctl["takeMax"], hard - pos))
                notes.append(f"offer {oa:g} below min payout {lo:g}: buying")
        if oa is not None:                                       # passive only: never cross anyone
            bid = min(bid, round(oa - tick, 9))
        if ob is not None:
            ask = max(ask, round(ob + tick, 9))
        orders = []
        if bid > cap_bid + 1e-9 or ask < cap_ask - 1e-9:
            notes.append("lock-in check: our price + others' would lock in a profit against us")
        if bid < 0 or bid > cap_bid + 1e-9:
            pass
        elif pos + QUOTE_SIZE <= mp and (left > 0 or pos < 0):
            orders.append(("bid", round(bid, 9), QUOTE_SIZE))
        else:
            notes.append("no bid (max position / budget)")
        if ask <= 0 or ask < cap_ask - 1e-9 or ask > max(hi, ob or 0) + tick:
            pass
        elif -pos + QUOTE_SIZE <= mp and (left > 0 or pos > 0):
            orders.append(("ask", round(ask, 9), QUOTE_SIZE))
        else:
            notes.append("no offer (max position / budget)")
        off_sides = blocked.get(iid, set())
        if off_sides:
            orders = [o for o in orders if o[0] not in off_sides]
            if take and take[0] in off_sides:
                take = None
            notes.insert(0, f"traded on {'/'.join(sorted(off_sides))}: off until quotes ON")
        if take and take[2] >= 1:
            row["take"] = (take[0], take[1], int(take[2]))
        plan[iid] = orders
        row.update(orders=orders, bid=bid, ask=ask, width=round(ask - bid, 9), mode=mode, tight=k in tight,
                   note="; ".join(notes))
    return plan, rows


def close_enough(want, live, tol):
    """Same orders per side (count and sizes) and every price within tol: leave them resting."""
    for side in ("bid", "ask"):
        w = sorted((p, q) for s, p, q in want if s == side)
        l = sorted((o["price"], o["qty"]) for o in live if o["side"] == side)
        if len(w) != len(l) or any(qa != qb or abs(pa - pb) > tol + 1e-9 for (pa, qa), (pb, qb) in zip(w, l)):
            return False
    return True


# --- positions: average price, market value, hedges -----------------------------------------------
def avg_cost(trades):
    """trades [(signed qty, price)] oldest first -> (position, average price of the shares still open)."""
    pos, avg = 0, None
    for q, p in trades:
        if pos == 0 or (pos > 0) == (q > 0):
            avg = p if pos == 0 else (avg * abs(pos) + p * abs(q)) / (abs(pos) + abs(q))
            pos += q
        elif abs(q) <= abs(pos):
            pos += q
        else:                                      # through zero: the rest opens at this price
            pos, avg = pos + q, p
        if pos == 0:
            avg = None
    return pos, avg


def market_value(pos, o_bid, o_ask, width):
    """What the position is worth in the market: long -> best bid we could sell into, short -> best
    offer we could buy back, None if we can't exit. Flat: mid; one-sided: that price +/- width/2."""
    if pos > 0:
        return o_bid
    if pos < 0:
        return o_ask
    if o_bid is not None and o_ask is not None:
        return (o_bid + o_ask) / 2
    if o_bid is not None:
        return o_bid + width / 2
    if o_ask is not None:
        return o_ask - width / 2
    return None


HEDGE_MIN_CORR = 0.08                    # ignore teams less correlated with ours (group stage: the same half
                                         # of the bracket passes; semis: the other two semi-finalists pass)


def hedges(k, rows, cov, theo, M, n=3):
    """The n best single-lot trades that take us one share closer to flat on team k: the closing trade
    itself, or the same direction of risk taken off through a team whose payout moves against ours
    (long Team1 in group B -> buy its group-mates; short a semi-finalist -> short its opponent first,
    then the other two). Ranked by edge = what the hedge trade gives away against our fair (best, or
    least bad, first). lockPnl: when 1 share of k and the hedge pay the same whatever happens, what
    that pair locks in against k's average price."""
    r = rows[k]
    if not r["pos"] or r["avg"] is None:
        return []
    s = 1 if r["pos"] > 0 else -1
    sd = np.sqrt(np.maximum(np.diag(cov), 0))
    corr = np.array([cov[k, j] / (sd[k] * sd[j]) if sd[k] > 1e-9 and sd[j] > 1e-9 else 0.0 for j in range(len(rows))])
    others = [j for j, y in enumerate(rows) if j != k and y["iid"] and not y["settled"] and abs(corr[j]) >= HEDGE_MIN_CORR]
    out = []
    for j in [k] + others:
        y = rows[j]
        if j == k and (not y["iid"] or y["settled"]):
            continue
        d = -s if j == k or corr[j] > 0 else s           # sell what moves with us, buy what moves against us
        price = y["oAsk"] if d > 0 else y["oBid"]
        if price is None:
            continue
        if j == k:
            lock = s * (price - r["avg"])                  # exact: sell above / buy below what we paid
        elif M["lockSum" if s == d else "lockDiff"][k, j]:
            lock = s * float(M["sumVal" if s == d else "diffVal"][k, j]) - s * r["avg"] - d * price
        else:
            lock = None
        out.append({"team": y["team"], "iid": y["iid"], "side": "bid" if d > 0 else "ask", "price": price,
                    "edge": float(d * (theo[j] - price)), "corr": float(corr[j]) if j != k else 1.0,
                    "lockPnl": None if lock is None else float(lock)})
    out.sort(key=lambda h: (-h["edge"], -abs(h["corr"])))
    return out[:n]


# --- session history for the charts (REST events: read-only, no websocket) -----------------------
class History:
    MAX = 20000

    def __init__(self):
        self.lock = threading.Lock()
        self.reset(None)

    def reset(self, mid):
        self.mid, self.seq = mid, 0
        self.tob, self.trades, self.pa = defaultdict(list), defaultdict(list), defaultdict(list)
        self.parts = defaultdict(lambda: defaultdict(int))   # contract id -> {other participant: position}
        self.orders = {}          # order id -> [contract id, side, price, qty left, owner ("us" = ours)]

    def owners(self, iid):
        """{(side, price): {owner: qty}} for every order resting on this contract, from the event feed."""
        out = defaultdict(lambda: defaultdict(int))
        with self.lock:
            for i, side, px, q, who in self.orders.values():
                if i == iid:
                    out[(side, round(px, 6))][who] += q
        return out

    def others(self, iid):
        with self.lock:
            return sorted(([w, q] for w, q in self.parts.get(iid, {}).items() if q), key=lambda x: -abs(x[1]))

    def update(self, mid, view, pa_by_iid):
        with self.lock:
            if mid != self.mid:
                self.reset(mid)
            me = view.get("you")
            if view.get("seq", 0) > self.seq:
                for e in ps.events(mid, self.seq):
                    if e["seq"] <= self.seq:
                        continue
                    self.seq = e["seq"]
                    at = e.get("at") or time.time() * 1000
                    for t in e.get("tob", []):
                        pt = (at, from_px(t["bid"]), from_px(t["ask"]))
                        lst = self.tob[t["instrumentId"]]
                        if not lst or lst[-1][1:] != pt[1:]:
                            lst.append(pt)
                            del lst[:-self.MAX]
                    if e.get("type") == "TRADE":
                        if me in (e.get("buyer"), e.get("seller")):
                            kind = "ourBuy" if e.get("buyer") == me else "ourSell"
                        else:
                            kind = "mktBuy" if e.get("aggressorSide") == "bid" else "mktSell"
                        self.trades[e["instrumentId"]].append((at, from_px(e["price"]), e["qty"], kind))
                        for who, sgn in ((e.get("buyer"), 1), (e.get("seller"), -1)):
                            if who and who != me:
                                self.parts[e["instrumentId"]][who] += sgn * e["qty"]
                        for oid in (e.get("restingOrderId"), e.get("aggressingOrderId")):
                            o = self.orders.get(oid)
                            if o:
                                o[3] -= e["qty"]
                                if o[3] <= 0:
                                    del self.orders[oid]
                    elif e.get("type") == "ORDER_PLACED":
                        self.orders[e["orderId"]] = [e["instrumentId"], e["side"], from_px(e["price"]), e["qty"],
                                                     "us" if e.get("owner") == me else e.get("owner") or "?"]
                    elif e.get("type") == "ORDER_CANCELLED":
                        self.orders.pop(e.get("orderId"), None)
                    elif e.get("type") == "RESET":
                        self.parts = defaultdict(lambda: defaultdict(int))
                        self.orders = {}
            now = time.time() * 1000
            for iid, v in pa_by_iid.items():
                lst = self.pa[iid]
                if not lst or abs(lst[-1][1] - v) > 1e-6:
                    lst.append((now, round(float(v), 4)))
                    del lst[:-self.MAX]

    def dump(self):
        with self.lock:
            return {iid: {"tob": self.tob.get(iid, []), "trades": self.trades.get(iid, []), "pa": self.pa.get(iid, [])}
                    for iid in set(self.tob) | set(self.trades) | set(self.pa)}


HISTORY = History()


# --- gateway: make the resting orders match the plan, paced under the rate limit -----------------
class DeskGateway(ps.Gateway):
    def __init__(self, enabled):
        super().__init__(enabled)
        self.plan = {}            # market id -> {contract id: [(side, price, qty)]}
        self.plan_at = {}
        self.quoting = {}         # market id -> bool
        self.last = {}            # (market id, contract id) -> when we last changed it
        self.pulled = {}          # market id -> contract ids pulled one by one
        self.blocked = {}         # market id -> {contract id: {"bid"/"ask"}} sides we were traded on
        self.tol = {}             # market id -> re-quote only if a price is off by more than this
        self.news_pause = 5.0     # seconds out after a market event (tick change, news ...)
        self.takes = {}           # market id -> {contract id: (side, price, qty)} crossing our prices
        self.took = {}            # (market, contract, side, price) -> when we last took it
        self.hard = {}            # market id -> {contract id: hard position limit (max pos + hard extra)}
        self.hits = {}            # market id -> {contract id: lots traded on our quotes since quoting (re)started}
        self.defl_cfg = {}        # market id -> (auto deflection on, deflection count, {contract id: max position})
        self.rest_pos = {}        # market id -> positions from the last exchange snapshot
        self.manual = {}          # market id -> order ids typed on the page (the quoter leaves them alone)
        self.rejected = {}        # (market, contract, side, price) -> when the exchange last rejected it

    def pump(self):
        super().pump()                                      # news / market events pull everything ...
        for mid, c in list(self.conns.items()):             # ... but we only sit out NEWS_PAUSE here (not 60 s):
            c["paused_until"] = min(c["paused_until"], time.time() + self.news_pause)   # results have their own button
            self.on_fills(mid, c)
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

    def on_fills(self, mid, c):
        """Someone traded on one of our resting orders.
        AUTO DEFLECTION on: we keep quoting (the position moves our PA, so the next quote is deflected)
            until DEFL COUNT lots have been traded on this market or the position limit is reached -
            then the whole market goes off.
        AUTO DEFLECTION off: the side we were traded on goes off.
        Either way it stays off until the trader clicks "start quoting" for that market."""
        book = c["book"]
        new, c["seen"] = book.fills[c.get("seen", 0):], len(book.fills)
        auto, count, limits = self.defl_cfg.get(mid, (False, 0, {}))
        for f in new:
            if f["aggressor"]:
                continue
            iid, side = f["iid"], ("bid" if f["qty"] > 0 else "ask")
            n = self.hits.setdefault(mid, {}).get(iid, 0) + abs(f["qty"])
            self.hits[mid][iid] = n
            at_limit = abs(book.pos.get(iid, 0)) >= limits.get(iid, 10 ** 9)
            if auto and n < count and not at_limit:
                print(f"{time.strftime('%H:%M:%S')}  {mid}: traded on our {side} in {book.name(iid)} - deflected, "
                      f"quoting on ({n}/{count})")
                continue
            sides = {"bid", "ask"} if auto else {side}
            self.blocked.setdefault(mid, {}).setdefault(iid, set()).update(sides)
            man = self.manual.get(mid, set())
            for oid, o in list(book.my_orders.items()):
                if o["iid"] == iid and o["side"] in sides and oid not in man:
                    c["fm"].cancel(oid)
            why = "position limit" if at_limit else f"{n} deflections" if auto else "traded on"
            print(f"{time.strftime('%H:%M:%S')}  {mid}: {book.name(iid)} {'/'.join(sorted(sides))} off ({why}) until start quoting")

    def cancel_level(self, mid, iid, side, price):
        """Cancel our orders at one price (clicked on the page) and keep the quoter off that side
        until the trader clicks "start quoting" - otherwise it would put the order straight back."""
        if not self.enabled:
            return {"ok": False, "reason": "paper mode: restart with --trade to send orders"}
        c = self.conn(mid)
        oids = [oid for oid, o in c["book"].my_orders.items()
                if o["iid"] == iid and o["side"] == side and abs(o["price"] - float(price)) < 1e-9]
        if not oids:
            return {"ok": False, "reason": "no order of ours at that price"}
        self.blocked.setdefault(mid, {}).setdefault(iid, set()).add(side)
        for oid in oids:
            c["fm"].cancel(oid)
            self.manual.get(mid, set()).discard(oid)
        print(f"{time.strftime('%H:%M:%S')}  {mid}: cancelled our {side} at {float(price):g} in {c['book'].name(iid)} - "
              f"that side off until start quoting")
        return {"ok": True, "n": len(oids)}

    def all_on(self, mid):
        self.pulled.pop(mid, None)
        return self.unblock(mid)

    def unblock(self, mid, iid=None):
        if iid is None:
            self.blocked.pop(mid, None)
            self.hits.pop(mid, None)
        else:
            self.blocked.get(mid, {}).pop(iid, None)
            self.hits.get(mid, {}).pop(iid, None)
        return {"ok": True}

    def hard_limit(self, mid, iid):
        h = self.hard.get(mid, {})
        return h.get(iid, 10 ** 9) if isinstance(h, dict) else h

    def reconcile(self, mid, c):
        """Make our resting orders match the plan, within the rate budget. Contracts with nothing
        resting go first, then the longest-untouched, so every market gets quoted; a contract
        whose orders are all within TOL of the plan is left alone (no churn)."""
        book, fm, now = c["book"], c["fm"], time.time()
        tol = self.tol.get(mid, 0.0)
        live_by = defaultdict(dict)
        manual = self.manual.get(mid, set())
        blocked = self.blocked.get(mid, {})
        for oid, o in book.my_orders.items():
            if oid not in manual:                           # manual orders are the trader's, not ours to move
                live_by[o["iid"]][oid] = o
        items = sorted(self.plan.get(mid, {}).items(),
                       key=lambda it: (bool(live_by.get(it[0])), self.last.get((mid, it[0]), 0)))
        for iid, want in items:
            if now - self.last.get((mid, iid), 0) < 0.5:
                continue
            want = [o for o in want if o[0] not in blocked.get(iid, ())]     # a side we were traded on stays off
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
                    if self.long_short(mid, book, iid, s) + exposure[s] + q > self.hard_limit(mid, iid):
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
        """Pull (or re-allow) one contract's quotes, straight away. Re-allowing clears its traded-on sides."""
        p = self.pulled.setdefault(mid, set())
        if off:
            p.add(iid)
            self.plan.setdefault(mid, {})[iid] = []
            if self.enabled:
                self.conn(mid)["fm"].post({"type": "CANCEL_ALL_ON_INSTRUMENT", "instrumentId": iid}, force=True)
                print(f"{time.strftime('%H:%M:%S')}  {mid}: pulled quotes on {self.conn(mid)['book'].name(iid)}")
        else:
            p.discard(iid)
            self.unblock(mid, iid)
        return {"ok": True}

    def manual_quote(self, mid, iid, bid, ask, qty):
        """A resting bid and/or offer typed on the page. The quoter leaves these alone (it only
        manages its own orders); quotes OFF and a contract's off button still cancel them."""
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
            if (now - self.took.get(key, 0) < 2.0 or iid in self.pulled.get(mid, set()) or fm.budget() < 2
                    or side in self.blocked.get(mid, {}).get(iid, ())):
                continue
            qty = min(qty, self.hard_limit(mid, iid) - self.long_short(mid, c["book"], iid, side))
            if qty < 1:
                continue
            self.took[key] = now

            def done(ack, mid=mid):
                if ack.get("ok") and ack.get("seqs"):
                    self.cancel_now.append((mid, f"o{ack['seqs'][0]}"))      # whatever didn't fill
            fm.order(iid, side, price, qty, on_ack=done)
            print(f"{time.strftime('%H:%M:%S')}  {mid}: TAKE {'BUY' if side == 'bid' else 'SELL'} {qty} "
                  f"{c['book'].name(iid)} @ {price:g} (crossed our price)")

    def set_plan(self, mid, plan, on, tol=0.0, news_pause=5.0, takes=None, hard=10 ** 9, positions=None, defl=None):
        """hard: {contract id: hard position limit} (or one number for all); defl: (auto deflection on,
        deflection count, {contract id: max position}) for on_fills."""
        self.tol[mid], self.news_pause, self.hard[mid] = tol, news_pause, hard
        if defl is not None:
            self.defl_cfg[mid] = defl
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
STATUS = {"rows": [], "tops": {}, "error": None, "view": None, "at": 0, "q": None}


def setup(body):
    """Teams (seed order), grid columns, tiers, regions, seeds, strength, series lengths, contract
    mapping. A team keeps its results, edits, pulls and layout when it's renamed: everything follows
    the new name."""
    st = tpricer.load_state()
    old = list(st["teams"])
    new = [str(t).strip() for t in body.get("teams", old)]
    ren = dict(zip(old, new)) if len(new) == len(old) else {}
    rn = lambda t: ren.get(t, t)
    if st.get("results"):
        same = {g: [rn(t) for t in st["groups"][g]] for g in st["groups"]} == body.get("groups", {g: [rn(t) for t in st["groups"][g]] for g in st["groups"]})
        if not same:
            raise ValueError("results are entered: teams can be renamed, but not moved between tiers")
    res = {}
    for mid, r in st.get("results", {}).items():
        r = {**r, **{f: rn(r[f]) for f in ("a", "b", "winner") if r.get(f)}}
        if ":" in mid:
            mid = f"{mid.split(':')[0]}:{r['a']}-{r['b']}" if r.get("a") else mid
        res[mid] = r
    st["results"] = res
    for k in ("ratings", "tiers", "regions", "ranks"):
        st[k] = {rn(t): v for t, v in (st.get(k) or {}).items()}
    for k in ("teams", "groups", "pBeat", "ratings", "tiers", "regions", "ranks", "bo"):
        if k in body:
            st[k] = body[k]
    tpricer.save_state(st)
    if new != old:
        ctl = controls()
        lay = ctl["layout"]
        save_controls({**{k: {rn(t): v for t, v in ctl[k].items()} for k in ("theoAdj", "paAdj", "paKeep", "edgeAdj", "deflPos", "deflAmt", "maxPosAdj")},
                       "off": [rn(t) for t in ctl["off"]], "hidden": [rn(t) for t in ctl["hidden"]],
                       "mapping": {c: rn(t) for c, t in ctl["mapping"].items()},
                       "layout": {**lay, "order": [rn(t) for t in lay.get("order") or []]}})
    if "mapping" in body:
        save_controls({"mapping": body["mapping"], "mappingOk": True})
    model()


def fat_finger_band(fair, ctl):
    """How far from fair a typed price may be: FAT PCT of fair or FAT MIN points, whichever is bigger."""
    return max(ctl["fatPct"] * abs(fair), ctl["fatMin"])


def fat_finger(iid, bid, ask):
    """None if a typed bid / offer is near enough to that team's fair, else the reason it's refused."""
    team = next((t for t, i in (STATUS.get("byTeam") or {}).items() if i == iid), None)
    if team is None:
        return "unknown contract"
    st, M = model()
    ctl = controls()
    F = fairs(st, M, ctl, STATUS.get("mkt"))
    f = float(F["theo"][st["teams"].index(team)])
    band = fat_finger_band(f, ctl)
    for side, px in (("bid", bid), ("offer", ask)):
        if px is not None and abs(px - f) > band + 1e-9:
            return (f"fat-finger guard: {side} {px:g} is {abs(px - f):.2f} from {team}'s fair {f:.2f} "
                    f"(allowed {max(f - band, 0):.2f} - {min(f + band, 100):.2f})")
    return None


def current_book(ctl):
    """(our positions, other market makers' mids) from the last tick, or flat / none without a market."""
    if ctl["market"] and STATUS.get("q") is not None:
        return STATUS["q"], STATUS.get("mkt")
    return np.zeros(len(tpricer.load_state()["teams"])), None


def apply_results(results):
    """Enter / clear results ({id: result or None}). New information: our theo edits go (the result
    replaces them), each team's PA keeps its gap to theo, and every market starts wide again."""
    st, M = model()
    ctl = controls()
    _, mkt = current_book(ctl)
    old = fairs(st, M, ctl, mkt)
    for k, r in results.items():
        if r is None:
            st["results"].pop(k, None)
        else:
            st["results"][k] = r
    tpricer.save_state(st)
    st, M = model()
    new = fairs(st, M, {**ctl, "theoAdj": {}, "paAdj": {}, "paKeep": {}}, mkt)
    keep = (old["pa"] - old["theo"]) - (new["pa"] - new["theo"])
    known = np.array([new["base"][k] == M["feasible"][t][0] == M["feasible"][t][1] for k, t in enumerate(st["teams"])])
    hidden = set(ctl["hidden"])
    q = st["results"].get("Q") or {}
    if q.get("winner") and q["winner"] in (q.get("a"), q.get("b")):                 # the qualifier's loser is out: off the screen
        hidden.add(q["b"] if q["winner"] == q["a"] else q["a"])
    save_controls({"theoAdj": {}, "paAdj": {}, "edgeAdj": {}, "wideSince": time.time(), "hidden": hidden,
                   "paKeep": {t: float(keep[k]) for k, t in enumerate(st["teams"]) if abs(keep[k]) > 1e-9 and not known[k]}})


def tick():
    """Read the book, work out the plan, hand it to the gateway."""
    ctl = controls()
    mid = ctl["market"]
    if not mid:
        return
    st, M = model()
    view = ps.snapshot(mid, max_age=0.5)
    tops = book_tops(view)
    names = {i["id"]: i["name"] for i in view["instruments"]}
    by_team = {ctl["mapping"][n]: iid for iid, n in names.items() if n in ctl["mapping"] and ctl["mapping"][n] in st["teams"]}
    positions = {i["id"]: i["position"] for i in view["instruments"]}
    q = np.array([positions.get(by_team.get(t), 0) for t in st["teams"]], dtype=float)
    mkt = market_mids(st, tops, by_team, view)
    model_half = ctl["k"] * M["sim"]["sigmaNext"] + ctl["c"]
    width, _, _ = market_widths(st, M["feasible"], model_half, tops, by_team, ctl, time.time())
    ctl = {**ctl, "deflAmt": update_defl_book(st, ctl, q, width)}
    pa = fairs(st, M, ctl, mkt)["pa"]
    lim = view.get("positionLimits") or {}
    if lim.get("enabled"):                              # the exchange's own position limit wins
        x = int(lim.get("defaultLimit", ctl["maxPos"]))
        ctl = {**ctl, "maxPos": min(ctl["maxPos"], x), "maxPosAdj": {t: min(v, x) for t, v in ctl["maxPosAdj"].items()}}
    blocked = {iid: set(s) for iid, s in gateway.blocked.get(mid, {}).items()} if gateway else {}
    plan, rows = plan_quotes(st, M["feasible"], pa, model_half, tops, by_team, positions, ctl, blocked,
                             pairs=(M["pairMin"], M["pairMax"]))
    ok = not check_mapping(ctl["mapping"], st["teams"], list(names.values()))   # Quoting ON is the confirmation
    if gateway.enabled:
        takes = {r["iid"]: r["take"] for r in rows if r.get("take") and ok}
        limits = {iid: max_pos(ctl, t) for t, iid in by_team.items()}
        gateway.call(gateway.set_plan, mid, plan if ok else {}, bool(ctl["quoting"] and ok), ctl["requoteMin"],
                     ctl["newsPause"], takes, {iid: m + ctl["hardExtra"] for iid, m in limits.items()}, positions,
                     (bool(ctl["autoDefl"]), int(ctl["deflCount"]), limits))
    try:
        HISTORY.update(mid, view, {by_team[t]: pa[k] for k, t in enumerate(st["teams"]) if t in by_team})
    except ApiError as e:
        print("history:", e)
    STATUS.update(rows=rows, tops=tops, view=view, byTeam=by_team, error=None, at=time.time(), q=q, mkt=mkt,
                  blocked=blocked, risk=risk_used(st, M["feasible"], pa, by_team, positions))


def brain():
    """Twice a second."""
    while True:
        t0 = time.time()
        try:
            tick()
        except ApiError as e:
            STATUS["error"] = str(e)
        except Exception as e:
            traceback.print_exc()
            STATUS["error"] = f"{type(e).__name__}: {e}"
        time.sleep(max(0.1, 0.5 - (time.time() - t0)))


def instruments(view):
    """Per contract id: book levels (ours marked), position, average price of the open shares, cash."""
    out = {}
    trades = defaultdict(list)
    for t in sorted(view.get("myTrades", []), key=lambda t: t.get("seq", 0)):
        trades[t["instrumentId"]].append((t["signedQty"], from_px(t["price"])))
    for inst in view["instruments"]:
        iid = inst["id"]
        known = HISTORY.owners(iid)

        def side(levels, s):
            """[price, qty, our qty, "who×qty, who×qty"] per level: the owners from the snapshot if it names
            them, else from the event feed; whatever's left unnamed shows as "?"."""
            out = []
            for l in levels[:5]:
                px = from_px(l["price"])
                who = defaultdict(int)
                for o in l["orders"]:
                    if o.get("owner"):
                        who["us" if o.get("mine") else o["owner"]] += o["qty"]
                if not who:
                    who.update(known.get((s, round(px, 6)), {}))
                mine = sum(o["qty"] for o in l["orders"] if o.get("mine"))
                if mine and "us" not in who:
                    who["us"] = mine
                if sum(who.values()) < l["qty"]:
                    who["?"] += l["qty"] - sum(who.values())
                out.append([px, l["qty"], mine, " · ".join(f"{w}×{q}" if q > 1 else w for w, q in who.items())])
            return out
        last_hit = max((t["at"] for t in view.get("myTrades", []) if t["instrumentId"] == iid and not t.get("aggressor")), default=None)
        _, avg = avg_cost(trades[iid])
        out[iid] = {"name": inst["name"], "bids": side(inst["bids"], "bid"), "asks": side(inst["asks"], "ask"), "pos": inst["position"],
                    "avg": avg, "cash": -sum(q * p for q, p in trades[iid]), "settle": from_px(inst.get("settlementValue")),
                    "closed": bool(inst.get("closed")), "hitAt": last_hit, "last": from_px(inst.get("lastTrade"))}
    return out


def state_json():
    st, M = model()
    ctl = controls()
    sim = M["sim"]
    view = STATUS.get("view") if ctl["market"] else None
    names = [i["name"] for i in view["instruments"]] if view else []
    srows = {r["team"]: r for r in STATUS.get("rows") or []} if view else {}
    tops = STATUS.get("tops") or {}
    by_team = (STATUS.get("byTeam") or {}) if view else {}
    insts = instruments(view) if view else {}
    F = fairs(st, M, ctl, STATUS.get("mkt") if view else None)
    theo, pa = F["theo"], F["pa"]
    seeds, dseed = tpricer.seed_values(st), tpricer.default_seeds(st)
    table = tpricer.standings(st)
    blocked = STATUS.get("blocked") or {}
    rows = []
    for k, t in enumerate(st["teams"]):
        r, iid = srows.get(t, {}), by_team.get(t)
        im, tb = insts.get(iid, {}), tops.get(iid, {})
        lo, hi = M["feasible"].get(t, [0, 100])
        if im.get("settle") is not None:                 # the exchange has settled it: that's the value
            lo = hi = im["settle"]
            for x in ("base", "our", "theo", "pa"):     # (fresh arrays from fairs(): safe to change here)
                F[x][k] = lo
        o_bid = tb["oBids"][0] if tb.get("oBids") else None
        o_ask = tb["oAsks"][0] if tb.get("oAsks") else None
        pos = im.get("pos", 0)
        mv = im["settle"] if im.get("settle") is not None else market_value(pos, o_bid, o_ask, ctl["rebateWidth"])
        mark = mv if mv is not None else float(theo[k])
        rows.append({"team": t, "seed": dseed[t], "rank": (st.get("ranks") or {}).get(t),"group": next((g for g, ts in st["groups"].items() if t in ts), ""),
                     "tier": st["tiers"].get(t), "region": st["regions"].get(t),
                     "maxPos": max_pos(ctl, t), "maxPosEdited": t in ctl["maxPosAdj"],
                     "hits": (gateway.hits.get(ctl["market"], {}).get(iid, 0) if gateway and iid else 0),
                     "iid": iid, "contract": im.get("name"), "range": [lo, hi], "known": lo == hi,
                     "settled": lo == hi or bool(im.get("settle") is not None or im.get("closed")),
                     "base": float(F["base"][k]), "our": float(F["our"][k]), "theo": float(theo[k]), "pa": float(pa[k]),
                     "nMM": int(F["n"][k]), "mktMid": None if np.isnan(F["mid"][k]) else float(F["mid"][k]),
                     "mode": r.get("mode"), "width": r.get("width"), "others": HISTORY.others(iid) if iid else [],
                     "edge": None if r.get("width") is None else r["width"] / 2, "edgeEdited": t in ctl["edgeAdj"],
                     "fatLo": max(float(theo[k]) - fat_finger_band(float(theo[k]), ctl), 0.0),
                     "fatHi": min(float(theo[k]) + fat_finger_band(float(theo[k]), ctl), 100.0),
                     "seedVal": float(seeds[k]), "seedEdited": t in (st.get("ratings") or {}),
                     "hidden": t in ctl["hidden"],
                     "theoEdited": t in ctl["theoAdj"], "paEdited": t in ctl["paAdj"],
                     "pWin": float(sim["pWin"][k]), "sd": float(sim["sd"][k]), "rec": table.get(t),
                     "pos": pos, "avg": im.get("avg"), "mktValue": mv, "pnl": im.get("cash", 0.0) + pos * mark if im else 0.0,
                     "pnlMarked": mv is not None, "bids": im.get("bids", []), "asks": im.get("asks", []),
                     "oBid": o_bid, "oAsk": o_ask, "off": t in ctl["off"], "blocked": sorted(blocked.get(iid, ())),
                     "hitAt": im.get("hitAt"), "ourBid": r.get("bid"), "ourAsk": r.get("ask"), "tight": bool(r.get("tight")),
                     "note": r.get("note") or r.get("why", ""), "last": im.get("last")})
    cov = sim["cov"]
    for k, r in enumerate(rows):
        r["hedges"] = hedges(k, rows, cov, theo, M) if r["pos"] else []
    auto, auto_probs = auto_map(names, st["teams"]) if names else ({}, [])
    cover = None
    if view:                                             # what's actually resting right now vs the MM requirement
        op = [r for r in rows if r["iid"] and not insts[r["iid"]]["closed"] and insts[r["iid"]]["settle"] is None]
        def mine(r, s):
            return [lvl[0] for lvl in r[s] if lvl[2]]
        within = [r for r in op if mine(r, "bids") and mine(r, "asks")
                  and min(mine(r, "asks")) - max(mine(r, "bids")) <= ctl["rebateWidth"] + 1e-9]
        cover = {"open": len(op), "within": len(within), "need": min(ctl["nTight"], len(op)),
                 "exchange": (view.get("rebate") or {}).get("mine")}
    return {"rows": rows, "controls": ctl, "stage": tpricer.stage(st), "cover": cover,
            "state": {k: st.get(k) for k in ("teams", "groups", "results", "pBeat", "ratings", "tiers", "regions", "ranks", "bo")},
            "fixtures": tpricer.fixtures(st), "presets": tpricer.PRESETS, "simMs": M["ms"],
            "api": {"read": bool(view) and not STATUS.get("error") and time.time() - STATUS["at"] < 5,
                    "orders": bool(gateway and gateway.enabled and ctl["market"] in gateway.conns)},
            "contracts": names, "autoMap": auto, "autoProblems": auto_probs,
            "mappingProblems": check_mapping(ctl["mapping"], st["teams"], names) if names else ["no market loaded"],
            "error": STATUS.get("error") if ctl["market"] else None,
            "trading": bool(gateway and gateway.enabled),
            "quotingLive": bool(gateway and gateway.quoting.get(ctl["market"])),
            "market": view["name"] if view else None, "status": view.get("status") if view else None,
            "units": view.get("units", "") if view else "", "now": time.time() * 1000,
            "narrowLeft": max(0.0, ctl["wideSince"] + ctl["narrowSecs"] - time.time()),
            "totalPnl": float(sum(r["pnl"] for r in rows if r["iid"]))}


class Handler(BaseHTTPRequestHandler):
    def send_json(self, obj, code=200):
        body = json.dumps(obj, default=float).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def allowed(self, url=None):
        """This laptop: always (POSTs still need the page token). Shared (--share): anyone on the
        network with the link, which carries the token (?k=...); the page then sends it as a header."""
        token = self.headers.get("X-Trade-Token") or (parse_qs(url.query).get("k", [""])[0] if url else "")
        if self.client_address[0] in ("127.0.0.1", "::1"):
            return self.headers.get("Host") in ALLOWED_HOSTS or token == TOKEN
        return SHARE and secrets.compare_digest(token, TOKEN)

    def do_GET(self):
        url = urlparse(self.path)
        if not self.allowed(url):
            return self.send_json({"ok": False, "reason": "forbidden: use the link the desk printed (it has ?k=...)"}, 403)
        if url.path in ("/api/state", "/api/history"):
            try:
                return self.send_json(state_json() if url.path == "/api/state" else HISTORY.dump())
            except Exception as e:
                traceback.print_exc()
                return self.send_json({"error": f"{type(e).__name__}: {e}"}, 500)
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
        if not self.allowed() or self.headers.get("X-Trade-Token") != TOKEN:
            return self.send_json({"ok": False, "reason": "forbidden"}, 403)
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        except ValueError:
            return self.send_json({"ok": False, "reason": "bad request"}, 400)
        path = urlparse(self.path).path
        mid = controls()["market"]
        live = gateway.enabled and mid
        try:
            if path == "/api/setup":                       # teams (seed order), groups, strength, contract mapping
                with STATE_LOCK:
                    setup(body)
                return self.send_json({"ok": True})
            if path == "/api/results":                     # {id: result or null}, several at once
                with STATE_LOCK:
                    apply_results(body.get("results") or {})
                return self.send_json({"ok": True})
            if path == "/api/seed":                        # {team, value (None = back to its seed position)}
                with STATE_LOCK:
                    st = tpricer.load_state()
                    team, v = str(body["team"]), body.get("value")
                    if team not in st["teams"]:
                        raise ValueError(f"unknown team {team}")
                    r = dict(st.get("ratings") or {})
                    if v is None or abs(float(v) - tpricer.default_seeds(st)[team]) < 1e-9:
                        r.pop(team, None)
                    elif not 0.55 <= float(v) <= len(st["teams"]) + 0.45:
                        raise ValueError(f"seed must be between 0.6 and {len(st['teams']) + 0.4:g}")
                    else:
                        r[team] = round(float(v), 4)
                    st["ratings"] = r
                    tpricer.save_state(st)
                model()                                    # re-price now (a second or two)
                return self.send_json({"ok": True})
            if path == "/api/snap":                        # the MM-requirement markets: exactly the MM spread
                with STATE_LOCK:
                    ctl = controls()
                    tight = [r["team"] for r in STATUS.get("rows") or [] if r.get("tight") and r.get("mode") != "known"]
                    if not tight:
                        raise ValueError("no MM-requirement markets being quoted yet")
                    save_controls({"edgeAdj": {**ctl["edgeAdj"], **{t: ctl["rebateWidth"] / 2 for t in tight}}})
                return self.send_json({"ok": True, "n": len(tight)})
            if path == "/api/hide":                        # minimise / restore a market: {team, hidden}
                with STATE_LOCK:
                    h = set(controls()["hidden"])
                    save_controls({"hidden": (h | {str(body["team"])}) if body.get("hidden") else (h - {str(body["team"])})})
                return self.send_json({"ok": True})
            if path == "/api/pull":                        # one team's quotes off / back on (on = clears traded-on sides)
                team, off = str(body["team"]), bool(body.get("off", True))
                with STATE_LOCK:
                    ctl = controls()
                    save_controls({"off": (set(ctl["off"]) | {team}) if off else (set(ctl["off"]) - {team})})
                iid = (STATUS.get("byTeam") or {}).get(team)
                if live and iid:
                    return self.send_json(gateway.call(gateway.pull_one, mid, iid, off))
                return self.send_json({"ok": True})
            if path == "/api/allquotes":                   # everything on (clears every pull) / everything off
                on = bool(body.get("on"))
                save_controls({"quoting": on, **({"off": []} if on else {})})
                if live:
                    if on:
                        return self.send_json(gateway.call(gateway.all_on, mid))
                    return self.send_json(gateway.call(gateway.stop, mid))
                return self.send_json({"ok": True})
            if path == "/api/theo":                        # {team, kind theo|pa|edge, value (None = undo the edit)}
                with STATE_LOCK:
                    st, M = model()
                    ctl = controls()
                    team, kind = str(body["team"]), body.get("kind")
                    k = st["teams"].index(team)
                    F = fairs(st, M, ctl, current_book(ctl)[1])
                    key = {"theo": "theoAdj", "pa": "paAdj", "edge": "edgeAdj", "maxpos": "maxPosAdj"}[kind]   # theo: OUR fair
                    adj = dict(ctl[key])
                    if body.get("value") is None:
                        adj.pop(team, None)
                    elif kind == "maxpos":
                        adj[team] = max(0, int(round(float(body["value"]))))
                    elif kind == "edge":
                        adj[team] = max(0.0, float(body["value"]))
                    else:
                        adj[team] = float(body["value"]) - float(F["base"][k] if kind == "theo" else F["pa0"][k])
                    return self.send_json({"ok": True, "controls": save_controls({key: adj})})
            if path == "/api/controls":
                return self.send_json({"ok": True, "controls": save_controls(body.get("set") or {})})
            if path == "/api/quote":                       # manual resting bid and/or offer
                num = lambda v: None if v in (None, "") else float(v)
                qty = body.get("qty")
                bad = fat_finger(body.get("iid"), num(body.get("bid")), num(body.get("ask")))
                if bad:
                    return self.send_json({"ok": False, "reason": bad})
                return self.send_json(gateway.call(gateway.manual_quote, mid, body.get("iid"), num(body.get("bid")),
                                                   num(body.get("ask")), qty if isinstance(qty, int) else -1))
            if path == "/api/cancel_level":                # click on our own top-of-book price: cancel it
                return self.send_json(gateway.call(gateway.cancel_level, mid, body.get("iid"), body.get("side"),
                                                   float(body.get("price"))))
            if path == "/api/order":                       # click on a price: fill now at that price or cancel
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
    if not controls()["wideSince"]:                        # first session: start wide too
        save_controls({"wideSince": time.time()})
    model()
    threading.Thread(target=brain, daemon=True).start()
    server = ThreadingHTTPServer(("0.0.0.0" if SHARE else "127.0.0.1", PORT), Handler)
    print(f"tournament desk on http://localhost:{PORT}  ({'LIVE' if trading else 'paper'}; Ctrl+C to stop)")
    if SHARE:
        ips = sorted({ip for ip in socket.gethostbyname_ex(socket.gethostname())[2] if not ip.startswith("127.")})
        print("SHARED: the others open this link (it lets them trade - only give it to the desk):")
        for ip in ips:
            print(f"    http://{ip}:{PORT}/?k={TOKEN}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("stopped - the exchange will pull any API orders this server placed (~5 s)")


if __name__ == "__main__":
    main()
