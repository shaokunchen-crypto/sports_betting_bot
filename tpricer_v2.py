"""Tournament pricer (brief 1): 16 players, groups A-D, crossovers, QF, SF, final + 3rd place.

    payouts   group exit 0 · crossover exit 5 · QF exit 10 · 4th 25 · 3rd 50 · 2nd 75 · 1st 100  (pot 310)
    bracket   C1 2A v 3B · C2 2B v 3A · C3 2C v 3D · C4 2D v 3C
              QF A 1A v C2 · QF B 1D v C3 · QF C 1B v C1 · QF D 1C v C4
              SF1 QF A v QF B · SF2 QF C v QF D · final SF winners · 3rd place SF losers
    groups    round robin, 3/1/0; tiebreak: points, goal difference, goals for, head-to-head, random
    matches   Poisson goals: log λ_i = μ + g(d)/2, d = S_i - S_j, g(d) = d + κ·d·|d|  (β folded into S)
    knockout  level after 90': extra time at λ/3, then a 50/50 shootout
    strengths prior S_r = σ·Φ⁻¹(1 - (r - 0.5)/16) by seed (σ solved from P(seed 1 beats seed 16)),
              optional nudges blended in by confidence; updated after every result by MAP
              (Gaussian prior + Poisson likelihood of every score), covariance from the Hessian.

Monte Carlo (numpy, vectorised): draw strengths from the posterior, play everything not yet
played, apply tiebreakers, play the bracket, record payouts. theo = mean payout (always 310 in
total). σ_next = how much a player's theo would move on their NEXT match's result (W/D/L or
win/lose), from the same simulations.

    python tpricer_v2.py      # self-tests
"""
import json
import math
import os
from statistics import NormalDist

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
STATE_FILE = os.path.join(HERE, "tdesk_state.json")
GROUPS = "ABCD"
PAY = {"group": 0, "cross": 5, "qf": 10, "fourth": 25, "third": 50, "second": 75, "first": 100}
TOTAL = 310
ROUNDS = [[(0, 1), (2, 3)], [(0, 2), (1, 3)], [(0, 3), (1, 2)]]      # group schedule by position in the group list
CROSS = [("C1", "2A", "3B"), ("C2", "2B", "3A"), ("C3", "2C", "3D"), ("C4", "2D", "3C")]
QF = [("QFA", "1A", "C2"), ("QFB", "1D", "C3"), ("QFC", "1B", "C1"), ("QFD", "1C", "C4")]
SF = [("SF1", "QFA", "QFB"), ("SF2", "QFC", "QFD")]
KO_ORDER = [m for m, _, _ in CROSS] + [m for m, _, _ in QF] + ["SF1", "SF2", "F", "T"]
PRESETS = {"Even field": 0.40, "Moderate": 0.60, "Strong favourites": 0.75}
CONFIDENCE = {"Low": (0.25, 0.35), "Medium": (0.5, 0.30), "High": (0.75, 0.25)}   # blend weight, prior sd


def default_state():
    teams = [f"Team{i}" for i in range(1, 17)]
    return {"teams": teams, "groupMode": "snake", "groups": snake(teams), "pBeat": 0.60, "goals": 2.5,
            "kappa": 0.0, "nudges": {}, "confidence": "Low", "results": {}, "sims": 50000, "ratings": {}}


def snake(teams):
    """1-4 -> A-D, 5-8 -> D-A, 9-12 -> A-D, 13-16 -> D-A."""
    g = {x: [] for x in GROUPS}
    for r, t in enumerate(teams):
        row, col = divmod(r, 4)
        g[GROUPS[col] if row % 2 == 0 else GROUPS[3 - col]].append(t)
    return g


def load_state():
    try:
        with open(STATE_FILE, encoding="utf-8") as f:
            return {**default_state(), **json.load(f)}
    except (OSError, ValueError):
        return default_state()


def save_state(st):
    teams = [str(t).strip() for t in st["teams"]]
    if len(teams) != 16 or len(set(t.lower() for t in teams)) != 16 or not all(teams):
        raise ValueError("need 16 different team names in seed order")
    groups = snake(teams) if st.get("groupMode") == "snake" else {g: [str(t).strip() for t in st["groups"][g]] for g in GROUPS}
    if sorted(t for g in GROUPS for t in groups[g]) != sorted(teams) or any(len(groups[g]) != 4 for g in GROUPS):
        raise ValueError("groups must hold each of the 16 teams exactly once, 4 per group")
    st = {**st, "teams": teams, "groups": groups}
    with open(STATE_FILE + ".tmp", "w", encoding="utf-8") as f:
        json.dump(st, f, indent=1)
    os.replace(STATE_FILE + ".tmp", STATE_FILE)
    return st


# --- match model ----------------------------------------------------------------------------
def lam(si, sj, mu, kappa=0.0):
    d = si - sj
    g = d + kappa * d * abs(d)
    return math.exp(mu + g / 2), math.exp(mu - g / 2)


def _pmf(l, kmax=None):
    """Poisson probabilities for 0..kmax goals; kmax reaches far enough for any rate (log space)."""
    kmax = kmax or int(l + 12 * math.sqrt(l) + 15)
    return [math.exp(k * math.log(l) - l - math.lgamma(k + 1)) if l > 0 else float(k == 0) for k in range(kmax + 1)]


def wdl(l1, l2):
    k = int(max(l1, l2) + 12 * math.sqrt(max(l1, l2)) + 15)
    a, b = _pmf(l1, k), _pmf(l2, k)
    win = sum(a[i] * sum(b[:i]) for i in range(len(a)))
    draw = sum(a[i] * b[i] for i in range(len(a)))
    return win, draw, 1 - win - draw


def mu_of(goals):
    return math.log(goals / 2)


def seed_strengths(sigma, n=16):
    nd = NormalDist()
    return np.array([sigma * nd.inv_cdf(1 - (r - 0.5) / n) for r in range(1, n + 1)])


def solve_sigma(p_target, goals, kappa=0.0):
    """σ so that seed 1 beats seed 16 in one (group) match with probability p_target."""
    mu = mu_of(goals)
    lo, hi = 0.0, 5.0
    for _ in range(60):
        mid = (lo + hi) / 2
        s = seed_strengths(mid)
        if wdl(*lam(s[0], s[15], mu, kappa))[0] < p_target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def seed_values(st):
    """Each team's seed as a number (1 = strongest): st["ratings"][team] if set (any value, e.g. 2.4 =
    between seeds 2 and 3, to say how close two teams are or give one a buff), else its seed position."""
    r = st.get("ratings") or {}
    return np.array([float(r[t]) if isinstance(r.get(t), (int, float)) else k + 1.0 for k, t in enumerate(st["teams"])])


def seed_strength(sigma, seed, n=16):
    """The seed curve S_r = σ·Φ⁻¹(1 - (r - 0.5)/n) for any seed value r (kept inside 0.55 .. n + 0.45)."""
    nd = NormalDist()
    return np.array([sigma * nd.inv_cdf(1 - (min(max(r, 0.55), n + 0.45) - 0.5) / n) for r in seed])


# --- strengths: prior + MAP update ---------------------------------------------------------
def prior(st):
    sigma = solve_sigma(st["pBeat"], st["goals"], st.get("kappa", 0.0))
    s0 = seed_strength(sigma, seed_values(st))           # integer seeds: exactly seed_strengths(sigma)
    w, sd = CONFIDENCE.get(st.get("confidence", "Low"), CONFIDENCE["Low"])
    tier = 2 * (s0.max() - s0.min()) / 15              # gap between neighbouring seeds, x2
    nudge = np.array([float(st.get("nudges", {}).get(t, 0)) for t in st["teams"]])
    return s0 + w * nudge * tier, sd * max(sigma, 0.05), sigma


def played(st):
    """[(i, j, gi, gj)] for every match with a score entered (regulation score)."""
    idx = {t: k for k, t in enumerate(st["teams"])}
    out = []
    for mid, r in st.get("results", {}).items():
        if r.get("a") in idx and r.get("b") in idx and r.get("ga") is not None and r.get("gb") is not None:
            out.append((idx[r["a"]], idx[r["b"]], int(r["ga"]), int(r["gb"])))
    return out


def posterior(st):
    """MAP strengths and their covariance (Laplace). Newton steps on the log-posterior."""
    m0, sd0, sigma = prior(st)
    mu, games = mu_of(st["goals"]), played(st)
    s = m0.copy()
    prec0 = 1.0 / sd0 ** 2
    for _ in range(50):
        grad = -(s - m0) * prec0
        hess = -np.eye(16) * prec0
        for i, j, gi, gj in games:
            li, lj = lam(s[i], s[j], mu)         # κ ignored in the fit (small; keeps the Hessian simple)
            gr = (gi - li) / 2 - (gj - lj) / 2
            grad[i] += gr
            grad[j] -= gr
            h = (li + lj) / 4
            hess[i, i] -= h
            hess[j, j] -= h
            hess[i, j] += h
            hess[j, i] += h
        step = np.linalg.solve(hess, -grad)
        s = s + step
        if np.max(np.abs(step)) < 1e-9:
            break
    cov = np.linalg.inv(-hess)
    return s, (cov + cov.T) / 2, sigma


# --- the simulation ---------------------------------------------------------------------------
def simulate(st, n=None, seed=7, shift=None, post=None):
    n = int(n or st.get("sims", 50000))
    rng = np.random.default_rng(seed)
    teams = st["teams"]
    idx = {t: k for k, t in enumerate(teams)}
    mean, cov, sigma = post or posterior(st)
    if shift is not None:
        mean = mean + shift
    S = rng.multivariate_normal(mean, cov, size=n, method="cholesky")
    mu, kappa = mu_of(st["goals"]), st.get("kappa", 0.0)
    res = st.get("results", {})
    rows = np.arange(n)

    def lams(a, b):                                   # a, b: arrays (n,) of team indices
        d = S[rows, a] - S[rows, b]
        g = d + kappa * d * np.abs(d)
        return np.exp(mu + g / 2), np.exp(mu - g / 2)

    pts = np.zeros((n, 16))
    gd = np.zeros((n, 16))
    gf = np.zeros((n, 16))
    gm = {}                                           # group match id -> (a, b, ga, gb)
    for g in GROUPS:
        members = [idx[t] for t in st["groups"][g]]
        for rnd in ROUNDS:
            for p, q in rnd:
                a, b = members[p], members[q]
                mid = f"{g}:{teams[a]}-{teams[b]}"
                r = res.get(mid)
                if r and r.get("ga") is not None:
                    ga, gb = np.full(n, int(r["ga"])), np.full(n, int(r["gb"]))
                else:
                    la, lb = lams(np.full(n, a), np.full(n, b))
                    ga, gb = rng.poisson(la), rng.poisson(lb)
                gm[mid] = (a, b, ga, gb)
                pts[:, a] += np.where(ga > gb, 3, np.where(ga == gb, 1, 0))
                pts[:, b] += np.where(gb > ga, 3, np.where(ga == gb, 1, 0))
                gd[:, a] += ga - gb
                gd[:, b] += gb - ga
                gf[:, a] += ga
                gf[:, b] += gb
    key = pts * 1e6 + (gd + 500) * 1e3 + gf           # points, then GD, then goals for
    h2h = np.zeros((n, 16))                           # head-to-head points among teams level on all of that
    for a, b, ga, gb in gm.values():
        tied = key[:, a] == key[:, b]
        h2h[:, a] += tied * np.where(ga > gb, 3, np.where(ga == gb, 1, 0))
        h2h[:, b] += tied * np.where(gb > ga, 3, np.where(ga == gb, 1, 0))
    full = key * 100 + h2h * 10 + rng.random((n, 16))  # ... then random
    place = {}                                        # "1A" -> array of team index per sim
    finish = np.zeros((n, 16), dtype=int)
    for g in GROUPS:
        members = np.array([idx[t] for t in st["groups"][g]])
        order = np.argsort(-full[:, members], axis=1)
        for k in range(4):
            tm = members[order[:, k]]
            place[f"{k + 1}{g}"] = tm
            finish[rows, tm] = k + 1

    pay = np.zeros((n, 16))
    ko = {}

    def play(mid, a, b):
        r = res.get(mid)
        if r and r.get("winner") in idx:              # entered result: that team goes through
            w = idx[r["winner"]]
            win_a = a == w
        else:
            la, lb = lams(a, b)
            ga, gb = rng.poisson(la), rng.poisson(lb)
            et_a, et_b = rng.poisson(la / 3), rng.poisson(lb / 3)
            level = ga == gb
            ga2, gb2 = ga + np.where(level, et_a, 0), gb + np.where(level, et_b, 0)
            win_a = np.where(ga2 != gb2, ga2 > gb2, rng.random(n) < 0.5)
        w, l = np.where(win_a, a, b), np.where(win_a, b, a)
        ko[mid] = (a, b, w)
        return w, l

    for mid, x, y in CROSS:
        w, l = play(mid, place[x], place[y])
        place[mid] = w
        pay[rows, l] = PAY["cross"]
    for mid, x, y in QF:
        w, l = play(mid, place[x], place[y])
        place[mid] = w
        pay[rows, l] = PAY["qf"]
    sf = {}
    for mid, x, y in SF:
        sf[mid] = play(mid, place[x], place[y])
    champ, second = play("F", sf["SF1"][0], sf["SF2"][0])
    third, fourth = play("T", sf["SF1"][1], sf["SF2"][1])
    pay[rows, champ], pay[rows, second] = PAY["first"], PAY["second"]
    pay[rows, third], pay[rows, fourth] = PAY["third"], PAY["fourth"]

    theo = pay.mean(axis=0)
    out = {"teams": teams, "theo": theo, "sd": pay.std(axis=0), "cov": np.cov(pay, rowvar=False), "pay": pay,
           "finish": np.stack([(finish == k).mean(axis=0) for k in range(1, 5)], axis=1),
           "pQF": (pay >= PAY["qf"]).mean(axis=0), "pSF": (pay >= PAY["fourth"]).mean(axis=0),
           "pF": (pay >= PAY["second"]).mean(axis=0), "pWin": (pay == PAY["first"]).mean(axis=0),
           "mean": mean, "sdS": np.sqrt(np.diag(cov)), "sigma": sigma, "n": n,
           "sigmaNext": _sigma_next(st, pay, gm, ko, idx, n), "totals": np.unique(pay.sum(axis=1))}
    return out


def _sigma_next(st, pay, gm, ko, idx, n):
    """Std dev of each player's theo across the outcomes of THEIR next match (same simulations)."""
    res = st.get("results", {})
    out = np.zeros(16)
    for i in range(16):
        buckets = None
        for g in GROUPS:                              # next unplayed group match, in round order
            if buckets is not None:
                break
            for mid, (a, b, ga, gb) in gm.items():
                if mid.startswith(g + ":") and i in (a, b) and not (res.get(mid) or {}).get("ga") is not None:
                    mine, theirs = (ga, gb) if i == a else (gb, ga)
                    buckets = [mine > theirs, mine == theirs, mine < theirs]
                    break
        if buckets is None:                           # group stage done: next knockout match they're in
            for mid in KO_ORDER:
                if mid in ko and mid not in res:
                    a, b, w = ko[mid]
                    inn = (a == i) | (b == i)
                    if inn.mean() > 0.999:
                        buckets = [w == i, w != i]
                        break
        if buckets is None:
            continue
        p = np.array([bk.mean() for bk in buckets])
        m = np.array([pay[bk, i].mean() if bk.any() else 0.0 for bk in buckets])
        mean = (p * m).sum()
        out[i] = math.sqrt(max(0.0, (p * (m - mean) ** 2).sum()))
    return out


def readouts(st, sim=None):
    mean, _, sigma = posterior(st)
    mu = mu_of(st["goals"])
    l1, l16 = lam(mean[0], mean[15], mu, st.get("kappa", 0.0))
    eq = wdl(math.exp(mu), math.exp(mu))
    return {"p1v16": wdl(l1, l16)[0], "p1v2": wdl(*lam(mean[0], mean[1], mu, st.get("kappa", 0.0)))[0],
            "pDrawEq": eq[1], "xg1v16": [l1, l16], "pSeed1Wins": None if sim is None else float(sim["pWin"][0]),
            "sigma": sigma}


def feasible(st):
    """(min, max) payout still possible for each team, from the results entered so far."""
    res, teams = st.get("results", {}), st["teams"]
    rng_ = {t: [0, 100] for t in teams}
    ko_res = {m: r for m, r in res.items() if m in KO_ORDER and r.get("winner")}
    groups_done = all(f"{g}:" in " ".join(res) and sum(1 for m in res if m.startswith(g + ":") and res[m].get("ga") is not None) == 6 for g in GROUPS)
    if groups_done:
        sim = simulate(st, n=200, seed=1)             # group stage fully entered: places are fixed
        for k, t in enumerate(teams):
            f = int(np.round(sim["finish"][k].argmax())) + 1
            rng_[t] = [10, 100] if f == 1 else [5, 100] if f in (2, 3) else [0, 0]
    for m, r in ko_res.items():
        w, a, b = r["winner"], r.get("a"), r.get("b")
        l = b if w == a else a
        if m.startswith("C"):
            rng_[w][0], rng_[l] = max(rng_[w][0], 10), [5, 5]
        elif m.startswith("QF"):
            rng_[w][0], rng_[l] = max(rng_[w][0], 25), [10, 10]
        elif m.startswith("SF"):
            rng_[w], rng_[l] = [75, 100], [25, 50]
        elif m == "F":
            rng_[w], rng_[l] = [100, 100], [75, 75]
        elif m == "T":
            rng_[w], rng_[l] = [50, 50], [25, 25]
    return rng_


def standings(st):
    """Group tables from the scores entered so far: {team: {rank, pts, p, gd, gf}} (points, GD, GF, seed)."""
    res, out = st.get("results", {}), {}
    for g in GROUPS:
        tab = {t: {"pts": 0, "p": 0, "gd": 0, "gf": 0} for t in st["groups"][g]}
        for mid, r in res.items():
            if mid.startswith(g + ":") and r.get("ga") is not None and r.get("a") in tab and r.get("b") in tab:
                for t, f, a in ((r["a"], r["ga"], r["gb"]), (r["b"], r["gb"], r["ga"])):
                    tab[t]["p"] += 1
                    tab[t]["gd"] += f - a
                    tab[t]["gf"] += f
                    tab[t]["pts"] += 3 if f > a else 1 if f == a else 0
        order = sorted(tab, key=lambda t: (-tab[t]["pts"], -tab[t]["gd"], -tab[t]["gf"], st["teams"].index(t)))
        for k, t in enumerate(order):
            out[t] = {**tab[t], "rank": k + 1}
    return out


# --- propagation: move some theos, the rest follow by covariance -----------------------------
def propagate(cov, targets):
    """targets {team index: wanted change}. Every target hits exactly (all at once) and the other
    teams move by their regression on the targets (Gaussian conditional mean): the most correlated
    teams move most. Any set whose total payout is fixed (all 16 = 310, the two finalists = 175 ...)
    has zero covariance with everything, so its total never moves."""
    out = np.zeros(cov.shape[0])
    T = sorted(i for i, d in targets.items() if d and cov[i, i] > 1e-9)
    if not T:
        return out
    s = np.linalg.lstsq(cov[np.ix_(T, T)], np.array([targets[i] for i in T]), rcond=None)[0]
    return cov[:, T] @ s


# --- tests --------------------------------------------------------------------------------------
def _test():
    st = default_state()
    assert snake(st["teams"])["A"] == ["Team1", "Team8", "Team9", "Team16"]
    for g, ms in snake(list(range(1, 17))).items():
        assert sum(ms) == 34, (g, ms)
    for name, p in PRESETS.items():
        sig = solve_sigma(p, 2.5)
        s = seed_strengths(sig)
        got = wdl(*lam(s[0], s[15], mu_of(2.5)))[0]
        assert abs(got - p) < 0.01, (name, got)
    # equal strengths: every theo 310/16; 1X = 36.25, 2X = 3X = 20.625
    import time
    flat = {**st}
    post = (np.zeros(16), np.eye(16) * 1e-12, 0.0)
    t0 = time.time()
    r = simulate(flat, n=100000, post=post)
    ms = (time.time() - t0) * 1000
    assert list(r["totals"]) == [310], r["totals"]
    assert np.all(np.abs(r["theo"] - 310 / 16) < 0.3), r["theo"]
    ab = sum(r["theo"][[st["teams"].index(t) for t in st["groups"]["A"] + st["groups"]["B"]]])
    assert abs(ab - 155) < 0.5, ab
    pay, finish = _raw_sim(flat, 200000, post)       # knockout as coin flips = equal strengths, exactly
    win_theo, two_theo = pay[finish == 1].mean(), pay[finish == 2].mean()
    assert abs(win_theo - 36.25) < 0.3 and abs(two_theo - 20.625) < 0.3 and pay[finish == 4].max() == 0, (win_theo, two_theo)
    # Poisson draw rate matches the analytic value
    rng = np.random.default_rng(0)
    l = math.exp(mu_of(2.5))
    sim_draw = (rng.poisson(l, 400000) == rng.poisson(l, 400000)).mean()
    assert abs(sim_draw - wdl(l, l)[1]) < 0.003
    # tiebreakers: points, then GD, then goals for, then head-to-head
    t = default_state()
    g = t["groups"]["A"]                              # Team1, Team8, Team9, Team16
    def res(a, b, ga, gb):
        return {f"A:{a}-{b}": {"a": a, "b": b, "ga": ga, "gb": gb}}
    t["results"] = {**res(g[0], g[1], 1, 0), **res(g[2], g[3], 0, 0), **res(g[0], g[2], 0, 1), **res(g[1], g[3], 2, 0),
                    **res(g[0], g[3], 2, 0), **res(g[1], g[2], 1, 0)}
    # points: g0 6, g1 6, g2 4, g3 1; g0 GD +2 (1-0, 0-1, 2-0) vs g1 GD +2 (0-1, 2-0, 1-0)... GF g0 3, g1 3 -> h2h: g0 beat g1
    fin = simulate(t, n=2000)["finish"]
    assert fin[t["teams"].index(g[0])][0] > 0.999 and fin[t["teams"].index(g[1])][1] > 0.999, fin
    # fractional seeds: integer values reproduce the seed curve; 1.5 sits between seeds 1 and 2
    sg = solve_sigma(0.6, 2.5)
    assert np.allclose(seed_strength(sg, np.arange(1, 17)), seed_strengths(sg))
    s15 = seed_strength(sg, [1.5])[0]
    assert seed_strengths(sg)[1] < s15 < seed_strengths(sg)[0]
    assert np.allclose(prior({**st, "ratings": {}})[0], prior(st)[0])
    # MAP: a big upset moves strengths, but the prior limits it
    m0 = posterior(st)[0]
    up = {**st, "results": {f"A:Team16-Team1": {"a": "Team16", "b": "Team1", "ga": 5, "gb": 0}}}
    m1 = posterior(up)[0]
    assert m1[15] > m0[15] and m1[0] < m0[0] and m1[15] < m0[0], (m0[[0, 15]], m1[[0, 15]])
    # propagation: lowering one team's theo raises every other (its closest rivals most), total stays 310
    cov = simulate(st, n=20000, seed=11)["cov"]
    mv = propagate(cov, {0: -5.0})
    assert abs(mv[0] + 5) < 1e-9 and abs(mv.sum()) < 1e-6 and np.all(mv[1:] > 0) and mv[1] == mv[1:].max(), mv
    print(f"tpricer_v2.py: all tests passed (100k-equivalent sim {ms:.0f} ms for 200k; equal: 1X {win_theo:.2f}, 2X {two_theo:.2f}; "
          f"A+B {ab:.1f}; Team1 -5 -> others +{mv[1:].sum():.2f})")


def _raw_sim(st, n, post):
    # equal-strength version that keeps the raw arrays (knockout = coin flips) - used only by the tests
    rng = np.random.default_rng(13)
    teams = st["teams"]
    idx = {t: k for k, t in enumerate(teams)}
    mu = mu_of(st["goals"])
    pts = np.zeros((n, 16)); gd = np.zeros((n, 16)); gf = np.zeros((n, 16))
    for g in GROUPS:
        m = [idx[t] for t in st["groups"][g]]
        for rnd in ROUNDS:
            for p, q in rnd:
                a, b = m[p], m[q]
                ga, gb = rng.poisson(math.exp(mu), n), rng.poisson(math.exp(mu), n)
                pts[:, a] += np.where(ga > gb, 3, np.where(ga == gb, 1, 0)); pts[:, b] += np.where(gb > ga, 3, np.where(ga == gb, 1, 0))
                gd[:, a] += ga - gb; gd[:, b] += gb - ga; gf[:, a] += ga; gf[:, b] += gb
    full = (pts * 1e6 + (gd + 500) * 1e3 + gf) * 100 + rng.random((n, 16))
    rows = np.arange(n)
    finish = np.zeros((n, 16), dtype=int)
    place = {}
    for g in GROUPS:
        m = np.array([idx[t] for t in st["groups"][g]])
        order = np.argsort(-full[:, m], axis=1)
        for k in range(4):
            place[f"{k + 1}{g}"] = m[order[:, k]]
            finish[rows, m[order[:, k]]] = k + 1
    pay = np.zeros((n, 16))
    def coin(a, b):
        w = rng.random(n) < 0.5
        return np.where(w, a, b), np.where(w, b, a)
    for mid, x, y in CROSS:
        w, l = coin(place[x], place[y]); place[mid] = w; pay[rows, l] = 5
    for mid, x, y in QF:
        w, l = coin(place[x], place[y]); place[mid] = w; pay[rows, l] = 10
    s1 = coin(place["QFA"], place["QFB"]); s2 = coin(place["QFC"], place["QFD"])
    c, r2 = coin(s1[0], s2[0]); t3, t4 = coin(s1[1], s2[1])
    pay[rows, c], pay[rows, r2], pay[rows, t3], pay[rows, t4] = 100, 75, 50, 25
    return pay, finish


if __name__ == "__main__":
    _test()
