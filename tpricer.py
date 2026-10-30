"""Tournament pricer: 17 teams - a qualifier, a 16-team Swiss stage, an 8-team knockout.

    qualifier  the two Tier "Q" teams (Na #4 v Sa #4), Best-of-5: winner joins the Swiss stage as a
               Tier 3 team, loser is out (0)
    Swiss      16 teams, 5 rounds at most, paired with teams on the same win/loss record:
               3 wins = knockout, 3 losses = out (0). Round 1: Tier 1 v Tier 3, Tier 2 v Tier 2,
               never two teams from the same region. Later rounds: random among the same record
               (rematches are not avoided - a small approximation)
    knockout   QF1 3-1 v 3-1 · QF2 3-2 v 3-0 · QF3 3-1 v 3-2 · QF4 3-0 v 3-2 (drawn within each record)
               SF1 QF1 v QF2 · SF2 QF3 v QF4 · Grand Final
    payouts    1st 100 · 2nd 60 · 3rd-4th 35 · 5th-8th 20 · 9th-17th 0   (pot 310)
    matches    each game: P(i beats j) = 1 / (1 + e^-(S_i - S_j)); a Best-of-k series is played from
               that (lengths in st["bo"]: qualifier 5, Swiss / knockout / final configurable)
    strengths  prior S = σ·Φ⁻¹(1 - (r - 0.5)/N) for each team's seed value r (1 = strongest; any
               number, e.g. 2.4, to say how close two teams are), σ set so seed 1 beats seed N in one
               game with probability pBeat; updated after every result by MAP (Gaussian prior +
               the game-by-game likelihood), covariance from the Hessian.

Monte Carlo (numpy, vectorised): draw strengths from the posterior, play everything not entered yet,
record payouts. theo = mean payout (always 310 in total). σ_next = how much a team's theo moves on
its NEXT match's result, from the same simulations.

    python tpricer.py      # self-tests
"""
import itertools
import json
import math
import os
import time
from statistics import NormalDist

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
STATE_FILE = os.path.join(HERE, "tdesk_state.json")
FORMAT = "swiss17"
TOTAL = 310
PAY = {"champion": 100, "final": 60, "semi": 35, "quarter": 20, "out": 0}
ROSTER = [("Nova", "Na", 1, 1), ("Rogue", "Sa", 1, 1), ("Echo", "eu", 1, 1), ("Pulse", "Am", 1, 1), ("Vertex", "au", 1, 1),
          ("Kinetik", "Na", 2, 2), ("Drift", "Sa", 2, 2), ("Vesper", "eu", 2, 2), ("Cipher", "Am", 2, 2), ("Onyx", "au", 2, 2),
          ("Zenith", "Na", 2, 3), ("Strata", "Sa", 3, 3), ("Omni", "eu", 3, 3), ("Flux", "Am", 3, 3), ("Null", "au", 3, 3),
          ("Apex", "Na", "Q", 4), ("Glaive", "Sa", "Q", 4)]          # (team, region, pot, rank within its region)
POTS = {1: "Tier 1", 2: "Tier 2", 3: "Tier 3", "Q": "Qualifier"}
QF_DRAW = [("QF1", "3-1", "3-1"), ("QF2", "3-2", "3-0"), ("QF3", "3-1", "3-2"), ("QF4", "3-0", "3-2")]
SF = [("SF1", "QF1", "QF2"), ("SF2", "QF3", "QF4")]
KO_ORDER = ["QF1", "QF2", "QF3", "QF4", "SF1", "SF2", "F"]
SWISS_ROUNDS = 5
PRESETS = {"Even field": 0.60, "Moderate": 0.75, "Strong favourites": 0.85}
CONFIDENCE = {"Low": 0.35, "Medium": 0.30, "High": 0.25}    # prior sd of a strength, x sigma
BO = {"qual": 5, "swiss": 1, "ko": 3, "final": 5}


def default_state():
    teams = [t for t, _, _, _ in ROSTER]
    tier = {t: k for t, _, k, _ in ROSTER}
    return {"format": FORMAT, "teams": teams, "regions": {t: r for t, r, _, _ in ROSTER}, "tiers": tier,
            "ranks": {t: k for t, _, _, k in ROSTER},
            "groups": pots(teams, tier), "pBeat": PRESETS["Moderate"], "bo": dict(BO),
            "confidence": "Low", "results": {}, "sims": 20000, "ratings": {}}


def pots(teams, tier):
    """The seeding pots (the tiers), as {pot name: [teams]}."""
    return {name: [t for t in teams if tier.get(t) == k] for k, name in POTS.items()}


# keyed by team name in the dashboard's controls: no meaning once the teams change
TEAM_CONTROLS = ("theoAdj", "paAdj", "paKeep", "edgeAdj", "off", "hidden", "layout", "mapping", "deflPos", "deflAmt",
                 "maxPosAdj")


def load_state():
    """The saved state. A state from an older tournament format is kept as tdesk_state.json.old.json
    and replaced by this format's defaults (settings that don't depend on the teams are kept)."""
    try:
        with open(STATE_FILE, encoding="utf-8") as f:
            saved = json.load(f)
    except (OSError, ValueError):
        return default_state()
    if saved.get("format") == FORMAT:
        return {**default_state(), **saved}
    with open(STATE_FILE + ".old.json", "w", encoding="utf-8") as f:
        json.dump(saved, f, indent=1)
    ctl = {k: v for k, v in saved.get("controls", {}).items() if k not in TEAM_CONTROLS}
    st = {**default_state(), "controls": {**ctl, "mappingOk": False, "quoting": False, "maxPos": 10}}
    _write(st)
    return st


def save_state(st):
    teams = [str(t).strip() for t in st["teams"]]
    if len(teams) < 4 or len(set(t.lower() for t in teams)) != len(teams) or not all(teams):
        raise ValueError("team names must all be different")
    groups = {g: [str(t).strip() for t in ts] for g, ts in st["groups"].items()}
    if sorted(t for ts in groups.values() for t in ts) != sorted(teams):
        raise ValueError("the pots must hold each team exactly once")
    if any(t not in st["tiers"] or t not in st["regions"] for t in teams):
        raise ValueError("every team needs a tier and a region")
    st = {**st, "format": FORMAT, "teams": teams, "groups": groups}
    _write(st)
    return st


def _write(st):
    with open(STATE_FILE + ".tmp", "w", encoding="utf-8") as f:
        json.dump(st, f, indent=1)
    for attempt in range(40):                  # Windows refuses the swap while another thread is reading the
        try:                                   # file: wait a moment and try again (up to ~2 s)
            os.replace(STATE_FILE + ".tmp", STATE_FILE)
            return
        except PermissionError:
            if attempt == 39:
                raise
            time.sleep(0.05)


# --- match model ----------------------------------------------------------------------------
def game_p(d):
    """P(win one game) for strength difference d."""
    return 1.0 / (1.0 + np.exp(-d))


def series_p(p, bo):
    """P(win a Best-of-bo series) from the single-game probability p."""
    k = bo // 2 + 1
    return sum(math.comb(k - 1 + j, j) * p ** k * (1 - p) ** j for j in range(k))


def default_seeds(st):
    """{team: seed} before any nudges. We only know each team's rank within its region, so teams with
    the same rank share a seed: the middle of the places they'd take between them (five regional #1s
    -> 3.0, the #2s -> 8.0, ...). Without ranks: seed order."""
    teams, ranks = st["teams"], st.get("ranks") or {}
    if not all(isinstance(ranks.get(t), (int, float)) for t in teams):
        return {t: k + 1.0 for k, t in enumerate(teams)}
    order = sorted(teams, key=lambda t: ranks[t])
    out = {}
    for rk in sorted({ranks[t] for t in teams}):
        places = [k + 1 for k, t in enumerate(order) if ranks[t] == rk]
        for t in teams:
            if ranks[t] == rk:
                out[t] = sum(places) / len(places)
    return out


def seed_values(st):
    """Each team's seed as a number (1 = strongest): st["ratings"][team] if set (any value, e.g. 2.4 =
    between seeds 2 and 3, to say how close two teams are or give one a buff), else its default seed."""
    r, d = st.get("ratings") or {}, default_seeds(st)
    return np.array([float(r[t]) if isinstance(r.get(t), (int, float)) else d[t] for t in st["teams"]])


def _z(r, n):
    return NormalDist().inv_cdf(1 - (min(max(r, 0.55), n + 0.45) - 0.5) / n)


def seed_strength(sigma, seed, n):
    """The seed curve S_r = σ·Φ⁻¹(1 - (r - 0.5)/n) for any seed value r (kept inside 0.55 .. n + 0.45)."""
    return np.array([sigma * _z(r, n) for r in seed])


def solve_sigma(p_beat, n, best=1.0, worst=None):
    """σ so that a team seeded `best` beats one seeded `worst` (default: seed 1 v seed n) in one game
    with probability p_beat."""
    return math.log(p_beat / (1 - p_beat)) / (_z(best, n) - _z(worst or n, n))


# --- strengths: prior + MAP update ---------------------------------------------------------
def prior(st):
    n = len(st["teams"])
    d = default_seeds(st).values()
    sigma = solve_sigma(st["pBeat"], n, min(d), max(d))          # P(best-ranked beats worst-ranked) = pBeat
    return seed_strength(sigma, seed_values(st), n), CONFIDENCE.get(st.get("confidence", "Low"), 0.35) * sigma, sigma


def played(st):
    """[(i, j, games won by i, games won by j)] for every result entered (the series score if given,
    else 1-0 to the winner)."""
    idx = {t: k for k, t in enumerate(st["teams"])}
    out = []
    for r in st.get("results", {}).values():
        a, b, w = r.get("a"), r.get("b"), r.get("winner")
        if a not in idx or b not in idx or w not in (a, b):
            continue
        ga, gb = r.get("ga"), r.get("gb")
        if not (isinstance(ga, (int, float)) and isinstance(gb, (int, float))) or (ga > gb) != (w == a):
            ga, gb = (1, 0) if w == a else (0, 1)
        out.append((idx[a], idx[b], int(ga), int(gb)))
    return out


def posterior(st):
    """MAP strengths and their covariance (Laplace). Newton steps on the log-posterior."""
    m0, sd0, sigma = prior(st)
    n = len(m0)
    s, prec0 = m0.copy(), 1.0 / sd0 ** 2
    hess = -np.eye(n) * prec0
    for _ in range(50):
        grad = -(s - m0) * prec0
        hess = -np.eye(n) * prec0
        for i, j, wi, wj in played(st):
            p = game_p(s[i] - s[j])
            g, h = wi - (wi + wj) * p, (wi + wj) * p * (1 - p)
            grad[i] += g
            grad[j] -= g
            hess[i, i] -= h
            hess[j, j] -= h
            hess[i, j] += h
            hess[j, i] += h
        step = np.linalg.solve(hess, -grad)
        s = s + step
        if np.max(np.abs(step)) < 1e-10:
            break
    cov = np.linalg.inv(-hess)
    return s, (cov + cov.T) / 2, sigma


# --- the format ------------------------------------------------------------------------------
def qualifier_teams(st):
    return [t for t in st["teams"] if st["tiers"].get(t) == "Q"]


def round1_pairings(st, qwin, fixed=()):
    """Every allowed round-1 draw given the qualifier winner, as lists of (a, b) team pairs: Tier 1 v
    Tier 3 (+ the qualifier winner), Tier 2 v Tier 2, never the same region. Pairs already entered
    (fixed) must be in it and are left out of what's returned. [] if the tiers don't fit that shape."""
    tier, reg = st["tiers"], st["regions"]
    t1 = [t for t in st["teams"] if tier.get(t) == 1]
    t3 = [t for t in st["teams"] if tier.get(t) == 3] + ([qwin] if qwin else [])
    t2 = [t for t in st["teams"] if tier.get(t) == 2]
    if len(t1) != len(t3) or len(t2) % 2:
        return []

    def matchings(ts):
        if not ts:
            yield []
            return
        a = ts[0]
        for k in range(1, len(ts)):
            if reg[a] != reg[ts[k]]:
                for m in matchings(ts[1:k] + ts[k + 1:]):
                    yield [(a, ts[k])] + m
    m2 = list(matchings(t2))
    out = []
    want = [frozenset(p) for p in fixed]
    for perm in itertools.permutations(t3):
        if any(reg[a] == reg[b] for a, b in zip(t1, perm)):
            continue
        for m in m2:
            draw = list(zip(t1, perm)) + m
            sets = [frozenset(p) for p in draw]
            if all(w in sets for w in want):
                out.append([p for p, s in zip(draw, sets) if s not in want])
    return out


def swiss_matches(st, rnd):
    """Entered matches of Swiss round rnd: [(id, result)]."""
    return [(k, r) for k, r in st.get("results", {}).items() if k.startswith(f"S{rnd}:")]


def records(st, upto=SWISS_ROUNDS):
    """Swiss win/loss record of every team from the results entered for rounds 1..upto."""
    rec = {}
    for rnd in range(1, upto + 1):
        for _, r in swiss_matches(st, rnd):
            if r.get("winner") and r["winner"] in (r.get("a"), r.get("b")):
                w, l = r["winner"], r["b"] if r["winner"] == r["a"] else r["a"]
                rec.setdefault(w, [0, 0])[0] += 1
                rec.setdefault(l, [0, 0])[1] += 1
    return rec


def swiss_field(st):
    """The 16 Swiss teams (None in the qualifier winner's place until it's entered)."""
    q = qualifier_teams(st)
    r = st.get("results", {}).get("Q") or {}
    return [t for t in st["teams"] if t not in q] + [r.get("winner") if r.get("winner") in q else None]


def swiss_done(st):
    field = swiss_field(st)
    rec = records(st)
    return None not in field and all(max(rec.get(t, [0, 0])) >= 3 for t in field)


# --- the simulation ---------------------------------------------------------------------------
def simulate(st, n=None, seed=7, shift=None, post=None):
    n = int(n or st.get("sims", 20000))
    rng = np.random.default_rng(seed)
    teams = st["teams"]
    N = len(teams)
    idx = {t: k for k, t in enumerate(teams)}
    mean, cov, sigma = post or posterior(st)
    if shift is not None:
        mean = mean + shift
    S = rng.multivariate_normal(mean, cov, size=n, method="cholesky")
    res, bo = st.get("results", {}), {**BO, **st.get("bo", {})}
    rows = np.arange(n)
    nxt = np.full((n, N), -1, dtype=np.int8)            # each team's next unplayed match: 1 won, 0 lost
    pay = np.zeros((n, N))

    def play(a, b, k, mid=None):
        """Best-of-k between team arrays a and b; an entered result decides it where it fits."""
        win = np.zeros(n, dtype=bool)
        free = np.ones(n, dtype=bool)
        r = res.get(mid) if mid else None
        if r and r.get("winner") in idx:
            w = idx[r["winner"]]
            free = (a != w) & (b != w)
            win = a == w
        if free.any():
            p = series_p(game_p(S[rows, a] - S[rows, b]), k)
            sim = rng.random(n) < p
            win = np.where(free, sim, win)
            for x, xw in ((a, sim), (b, ~sim)):
                m = free & (nxt[rows, x] == -1)
                nxt[rows[m], x[m]] = xw[m]
        return np.where(win, a, b), np.where(win, b, a)

    # qualifier
    q = qualifier_teams(st)
    field = [idx[t] for t in teams if t not in q]
    if len(q) == 2:
        qw, ql = play(np.full(n, idx[q[0]]), np.full(n, idx[q[1]]), bo["qual"], "Q")
        swiss = np.column_stack([np.tile(field, (n, 1)), qw])
    else:
        qw = None
        swiss = np.tile(field, (n, 1))
    W, L = np.zeros((n, N), dtype=int), np.zeros((n, N), dtype=int)
    inswiss = np.zeros((n, N), dtype=bool)
    inswiss[rows[:, None], swiss] = True

    def record(w, l):
        W[rows, w] += 1
        L[rows, l] += 1

    for rnd in range(1, SWISS_ROUNDS + 1):
        busy = np.zeros((n, N), dtype=bool)
        for mid, r in swiss_matches(st, rnd):                       # entered pairings (their winners if known)
            if r.get("a") in idx and r.get("b") in idx:
                a, b = np.full(n, idx[r["a"]]), np.full(n, idx[r["b"]])
                record(*play(a, b, bo["swiss"], mid))
                busy[rows, a] = busy[rows, b] = True
        active = inswiss & (W < 3) & (L < 3) & ~busy
        pairs = None
        if rnd == 1:
            fixed = [(r["a"], r["b"]) for _, r in swiss_matches(st, 1) if r.get("a") in idx and r.get("b") in idx]
            qr = res.get("Q") or {}
            qws = [qr["winner"]] if qr.get("winner") in q else q if len(q) == 2 else [None]
            cands = {t: round1_pairings(st, t, fixed) for t in qws}
            if all(cands.values()):
                pairs = np.zeros((n, int(active[0].sum()) // 2, 2), dtype=int)
                for t, draws in cands.items():
                    arr = np.array([[(idx[a], idx[b]) for a, b in d] for d in draws], dtype=int).reshape(len(draws), -1, 2)
                    pick = arr[rng.integers(0, len(arr), n)]
                    m = np.ones(n, dtype=bool) if t is None or len(qws) == 1 else (qw == idx[t])
                    pairs[m] = pick[m]
        if pairs is None:                                           # same record, random within it
            key = np.where(active, (W * 4 + L) + rng.random((n, N)) * 0.5, np.inf)
            m = int(active.sum(axis=1).min()) // 2 * 2
            pairs = np.argsort(key, axis=1)[:, :m].reshape(n, m // 2, 2)
        for s in range(pairs.shape[1]):
            record(*play(pairs[:, s, 0], pairs[:, s, 1], bo["swiss"]))

    # knockout draw: QF slots from the 3-0 / 3-1 / 3-2 pools; entered QFs keep their teams
    adv = inswiss & (W >= 3)
    taken = np.zeros((n, N), dtype=bool)
    qf_fixed = {}
    if swiss_done(st):
        for mid, _, _ in QF_DRAW:
            r = res.get(mid) or {}
            if r.get("a") in idx and r.get("b") in idx:
                qf_fixed[mid] = (np.full(n, idx[r["a"]]), np.full(n, idx[r["b"]]))
                taken[:, idx[r["a"]]] = taken[:, idx[r["b"]]] = True
    pools = {}
    for rec, losses in (("3-0", 0), ("3-1", 1), ("3-2", 2)):
        key = np.where(adv & (L == losses) & ~taken, rng.random((n, N)), np.inf)
        pools[rec] = [np.argsort(key, axis=1), 0]

    def draw(rec):
        arr, k = pools[rec]
        pools[rec][1] += 1
        return arr[:, k]
    winners = {}
    for mid, ra, rb in QF_DRAW:
        a, b = qf_fixed.get(mid) or (draw(ra), draw(rb))
        w, l = play(a, b, bo["ko"], mid)
        winners[mid] = w
        pay[rows, l] = PAY["quarter"]
    sf = {}
    for mid, x, y in SF:
        sf[mid] = play(winners[x], winners[y], bo["ko"], mid)
        pay[rows, sf[mid][1]] = PAY["semi"]
    champ, second = play(sf["SF1"][0], sf["SF2"][0], bo["final"], "F")
    pay[rows, champ], pay[rows, second] = PAY["champion"], PAY["final"]

    theo = pay.mean(axis=0)
    return {"teams": teams, "theo": theo, "sd": pay.std(axis=0), "cov": np.cov(pay, rowvar=False), "pay": pay,
            "pWin": (pay == PAY["champion"]).mean(axis=0), "pKO": (pay >= PAY["quarter"]).mean(axis=0),
            "mean": mean, "sdS": np.sqrt(np.diag(cov)), "sigma": sigma, "n": n,
            "sigmaNext": _sigma_next(pay, nxt), "totals": np.unique(pay.sum(axis=1))}


def _sigma_next(pay, nxt):
    """Std dev of each team's theo across the outcomes of ITS next match (won / lost / no match left)."""
    out = np.zeros(pay.shape[1])
    for i in range(pay.shape[1]):
        mean = pay[:, i].mean()
        v = 0.0
        for b in (1, 0, -1):
            m = nxt[:, i] == b
            if m.any():
                v += m.mean() * (pay[m, i].mean() - mean) ** 2
        out[i] = math.sqrt(v)
    return out


def readouts(st, sim=None):
    mean, _, sigma = posterior(st)
    return {"pTopBeatsBottom": float(game_p(mean[0] - mean[-1])), "sigma": sigma,
            "pSeed1Wins": None if sim is None else float(sim["pWin"][0])}


# --- what the results say for sure ---------------------------------------------------------------
def feasible(st):
    """(min, max) payout still possible for each team, from the results entered so far."""
    res = st.get("results", {})
    rng_ = {t: [0, 100] for t in st["teams"]}
    q = res.get("Q") or {}
    if q.get("winner") and q["winner"] in (q.get("a"), q.get("b")):
        rng_[q["b"] if q["winner"] == q["a"] else q["a"]] = [0, 0]
    for t, (w, l) in records(st).items():
        if t in rng_:
            if l >= 3:
                rng_[t] = [0, 0]
            elif w >= 3:
                rng_[t] = [PAY["quarter"], 100]
    for mid in KO_ORDER:
        r = res.get(mid) or {}
        w = r.get("winner")
        if w not in (r.get("a"), r.get("b")) or w not in rng_:
            continue
        l = r["b"] if w == r["a"] else r["a"]
        if mid.startswith("QF"):
            rng_[w][0], rng_[l] = max(rng_[w][0], PAY["semi"]), [PAY["quarter"]] * 2
        elif mid.startswith("SF"):
            rng_[w], rng_[l] = [PAY["final"], 100], [PAY["semi"]] * 2
        else:
            rng_[w], rng_[l] = [100, 100], [PAY["final"]] * 2
    return rng_


def standings(st):
    """{team: {"w", "l", "status"}}: the Swiss record so far, status qual / swiss / adv / out."""
    rec = records(st)
    q = qualifier_teams(st)
    qr = st.get("results", {}).get("Q") or {}
    out = {}
    for t in st["teams"]:
        w, l = rec.get(t, [0, 0])
        if t in q and not qr.get("winner"):
            status = "qual"
        elif t in q and qr.get("winner") != t:
            status = "out"
        else:
            status = "adv" if w >= 3 else "out" if l >= 3 else "swiss"
        out[t] = {"w": w, "l": l, "status": status}
    return out


def fixtures(st):
    """Everything the results popup needs: the qualifier, each Swiss round's record groups and entered
    matches, the knockout slots (the record pools once the Swiss stage is over)."""
    res = st.get("results", {})
    q = qualifier_teams(st)
    field = swiss_field(st)
    rounds = []
    for rnd in range(1, SWISS_ROUNDS + 1):
        rec = records(st, rnd - 1)
        groups = {}
        for t in field:
            w, l = rec.get(t, [0, 0]) if t else (0, 0)
            if w < 3 and l < 3:
                groups.setdefault(f"{w}-{l}", []).append(t)
        matches = [{"id": k, **r} for k, r in swiss_matches(st, rnd)]
        n_teams = sum(len(ts) for ts in groups.values())
        # a round opens once the one before is complete (its pairings depend on the records)
        is_open = rnd == 1 or rounds[-1]["complete"]
        rounds.append({"round": rnd, "groups": groups if is_open else {}, "matches": matches, "open": is_open,
                       "complete": is_open and None not in field and 2 * len(matches) >= n_teams > 0})
    done = swiss_done(st)
    rec = records(st)
    pools = {f"3-{l}": [t for t in field if t and rec.get(t, [0, 0]) == [3, l]] for l in range(3)} if done else {}
    ko, win = [], {}
    for mid, ra, rb in QF_DRAW:
        r = res.get(mid) or {}
        ko.append({"id": mid, "need": [ra, rb], "a": r.get("a"), "b": r.get("b"), "res": r or None})
        win[mid] = r.get("winner")
    for mid, x, y in SF + [("F", "SF1", "SF2")]:
        r = res.get(mid) or {}
        ko.append({"id": mid, "a": win.get(x), "b": win.get(y), "res": r or None, "srcA": f"winner {x}", "srcB": f"winner {y}"})
        win[mid] = r.get("winner")
    return {"qual": {"id": "Q", "a": q[0] if q else None, "b": q[1] if len(q) > 1 else None, "res": res.get("Q")},
            "rounds": rounds, "swissDone": done, "pools": pools, "ko": ko}


def stage(st):
    res = st.get("results", {})
    if not (res.get("Q") or {}).get("winner") and len(qualifier_teams(st)) == 2:
        return "qualifier"
    if not swiss_done(st):
        return "swiss"
    return "done" if (res.get("F") or {}).get("winner") else "knockout"


# --- propagation: move some theos, the rest follow by covariance -----------------------------
def propagate(cov, targets):
    """targets {team index: wanted change}. Every target hits exactly (all at once) and the other
    teams move by their regression on the targets (Gaussian conditional mean): the most correlated
    teams move most. Any set whose total payout is fixed (all = 310, the two finalists = 160 ...)
    has zero covariance with everything, so its total never moves."""
    out = np.zeros(cov.shape[0])
    T = sorted(i for i, d in targets.items() if d and cov[i, i] > 1e-9)
    if not T:
        return out
    s = np.linalg.lstsq(cov[np.ix_(T, T)], np.array([targets[i] for i in T]), rcond=None)[0]
    return cov[:, T] @ s


# --- tests --------------------------------------------------------------------------------------
def _test():
    import time
    st = default_state()
    n = len(st["teams"])
    assert n == 17 and len(qualifier_teams(st)) == 2
    # seed curve: integer seeds reproduce it, 1.5 sits between seeds 1 and 2; σ hits pBeat exactly
    sg = solve_sigma(0.75, n)
    s = seed_strength(sg, range(1, n + 1), n)
    assert abs(game_p(s[0] - s[-1]) - 0.75) < 1e-12 and s[1] < seed_strength(sg, [1.5], n)[0] < s[0]
    assert abs(series_p(0.6, 1) - 0.6) < 1e-12 and abs(series_p(0.6, 3) - 0.648) < 1e-12 and abs(series_p(0.5, 5) - 0.5) < 1e-12
    # same regional rank = same seed; P(a regional #1 beats a regional #4) = pBeat
    d = default_seeds(st)
    assert d["Nova"] == d["Vertex"] == 3.0 and d["Kinetik"] == 8.0 and d["Zenith"] == d["Null"] == 13.0 and d["Apex"] == 16.5
    m0p = prior(st)[0]
    assert abs(game_p(m0p[0] - m0p[-1]) - st["pBeat"]) < 1e-12 and m0p[0] == m0p[4] and m0p[10] == m0p[11]
    # equal strengths: the 15 seeded teams 310/16 each, the two qualifiers half that; every sim pays 310
    t0 = time.time()
    flat = (np.zeros(n), np.eye(n) * 1e-12, 0.0)
    r = simulate(st, n=40000, post=flat)
    ms = (time.time() - t0) * 1000
    assert list(r["totals"]) == [310], r["totals"]
    assert np.all(np.abs(r["theo"][:15] - 19.375) < 0.6) and np.all(np.abs(r["theo"][15:] - 9.6875) < 0.5), r["theo"]
    pays = np.sort(r["pay"], axis=1)[:, ::-1]
    assert (pays[:, :8] == [100, 60, 35, 35, 20, 20, 20, 20]).all() and (pays[:, 8:] == 0).all()
    # round-1 draws: Tier 1 v Tier 3, Tier 2 v Tier 2, never the same region
    for qw in qualifier_teams(st):
        draws = round1_pairings(st, qw)
        assert draws and all(len(d) == 8 for d in draws)
        for d in draws[:200]:
            for a, b in d:
                assert st["regions"][a] != st["regions"][b] and {st["tiers"][a], st["tiers"][b]} in ({1, 3}, {1, "Q"}, {2})
    # results: Nova beats Strata in round 1 -> Nova 1-0, re-priced, Strata can still go through
    st2 = {**st, "results": {"Q": {"a": "Apex", "b": "Glaive", "winner": "Glaive", "ga": 3, "gb": 1},
                             "S1:Nova-Strata": {"a": "Nova", "b": "Strata", "winner": "Strata"}}}
    f = feasible(st2)
    assert f["Apex"] == [0, 0] and f["Strata"] == [0, 100] and records(st2) == {"Strata": [1, 0], "Nova": [0, 1]}
    m0, m1 = posterior(st)[0], posterior(st2)[0]
    k = st["teams"].index
    assert m1[k("Strata")] > m0[k("Strata")] and m1[k("Nova")] < m0[k("Nova")]          # an upset moves strengths ...
    assert m1[k("Strata")] < m1[k("Nova")]                                               # ... but the prior holds
    r2 = simulate(st2, n=20000)
    assert r2["theo"][k("Apex")] == 0 and abs(r2["theo"].sum() - 310) < 1e-9 and r2["sigmaNext"][k("Nova")] > 0
    assert stage(st) == "qualifier" and stage(st2) == "swiss"
    fx = fixtures(st2)
    assert fx["rounds"][0]["groups"]["0-0"][-1] == "Glaive" and len(fx["rounds"][0]["matches"]) == 1 and not fx["swissDone"]
    assert not fx["rounds"][1]["open"] and fx["rounds"][1]["groups"] == {}           # round 2 waits for round 1
    assert records(st2, 1) == {"Strata": [1, 0], "Nova": [0, 1]}
    # a whole legal round 1 entered (Tier 1 v Tier 3, Tier 2 v Tier 2, no shared region): prices fine
    r1 = {"Q": st2["results"]["Q"]}
    for a, b in round1_pairings(st, "Glaive")[0]:
        r1[f"S1:{a}-{b}"] = {"a": a, "b": b, "winner": a}
    st3 = {**st, "results": r1}
    r3 = simulate(st3, n=5000)
    assert abs(r3["theo"].sum() - 310) < 1e-9 and fixtures(st3)["rounds"][1]["open"]
    # a whole Swiss stage + knockout entered: payouts fixed for everyone
    full = dict(st2["results"])
    sim_rng = np.random.default_rng(3)
    rec = {}
    for rnd in range(1, SWISS_ROUNDS + 1):
        stx = {**st, "results": full}
        live = [t for t in swiss_field(stx) if max(records(stx).get(t, [0, 0])) < 3 and not any(
            t in (r["a"], r["b"]) for _, r in swiss_matches(stx, rnd))]
        live.sort(key=lambda t: (records(stx).get(t, [0, 0]), sim_rng.random()))
        for a, b in zip(live[::2], live[1::2]):
            full[f"S{rnd}:{a}-{b}"] = {"a": a, "b": b, "winner": a if sim_rng.random() < 0.5 else b}
    stx = {**st, "results": full}
    assert swiss_done(stx) and stage(stx) == "knockout", records(stx)
    pools = fixtures(stx)["pools"]
    assert [len(pools[k]) for k in ("3-0", "3-1", "3-2")] == [2, 3, 3], pools
    p = {k: list(v) for k, v in pools.items()}
    for mid, ra, rb in QF_DRAW:
        a, b = p[ra].pop(), p[rb].pop()
        full[mid] = {"a": a, "b": b, "winner": a}
    full["SF1"] = {"a": full["QF1"]["a"], "b": full["QF2"]["a"], "winner": full["QF1"]["a"]}
    full["SF2"] = {"a": full["QF3"]["a"], "b": full["QF4"]["a"], "winner": full["QF4"]["a"]}
    full["F"] = {"a": full["QF1"]["a"], "b": full["QF4"]["a"], "winner": full["QF4"]["a"]}
    stx = {**st, "results": full}
    fz = feasible(stx)
    assert all(lo == hi for lo, hi in fz.values()) and sum(lo for lo, _ in fz.values()) == 310 and stage(stx) == "done"
    rz = simulate(stx, n=2000)
    assert np.allclose(rz["theo"], [fz[t][0] for t in st["teams"]])
    # propagation: lowering one team's theo raises the others, total stays 310
    mv = propagate(r2["cov"], {0: -5.0})
    assert abs(mv[0] + 5) < 1e-9 and abs(mv.sum()) < 1e-6
    print(f"tpricer.py: all tests passed (40k sims {ms:.0f} ms; equal strengths: seeded "
          f"{r['theo'][:15].mean():.2f}, qualifiers {r['theo'][15:].mean():.2f})")


if __name__ == "__main__":
    _test()
