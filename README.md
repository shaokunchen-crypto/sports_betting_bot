# Tournament desk

A pricing and market-making dashboard for the 16-team tournament on mock-exchange.fly.dev.
It simulates the tournament (groups, crossovers, knockout) to get a theo for every team's payout,
then quotes all 16 contracts around it. The page runs on your own laptop at http://localhost:8003.

## Setup (once, about 2 minutes)

You need Python 3.12 (python.org). In a terminal in this folder:

```
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
copy my_keys_template.py my_keys.py
```

Open `my_keys.py` and paste the API key in. **Never send `my_keys.py` to anyone or upload it.**

Check it works: `.venv\Scripts\python.exe test_tdesk.py` should end with "all tests passed".

## Running it

- **`start_view_only.bat`**: everyone else. You see the board, prices, positions and P&L,
  but nothing you click can send or cancel orders.
- **`start_trading.bat`**: the ONE person running the market maker. It asks you to type `LIVE`.

Then, on the page: type the market id → Load. Teams, groups, strength ratings and results are
in Setup at the bottom. Contract mapping must show "mapping OK". Click **Quoting ON** to start.

## Sharing one API key: read this

Everyone with the same key is the **same trader** on the exchange: same positions, same orders,
same P&L, and one rate limit (20 commands a second, shared). So:

1. **Only ONE laptop may run `start_trading.bat` at a time.** Two quoters on one key will cancel
   each other's orders (each one sees the other's orders as "ours but not in my plan") and burn
   the rate limit, and their position limits won't add up.
2. **When the trading laptop's desk stops or loses its connection, the exchange pulls ALL API
   orders on that key a few seconds later.** That includes manual quotes sent from this page.
3. View-only laptops are safe to run alongside: they only read the market.
4. Results, setup, deflection and every setting are saved per laptop, in `tdesk_state.json`.
   The trading laptop's settings are the ones that trade. If someone else enters a result on
   their copy, the quoter doesn't see it.

## The controls that matter most during a game

| Control | What it does |
|---|---|
| Quoting ON / OFF | start / stop the market maker (OFF pulls everything) |
| Result incoming - pull | pull everything before a result; enter the result; Quoting ON again |
| KILL | quoting off and every order pulled |
| pull / quote (per team) | that one contract only |
| Deflection % | how far our fair moves per share we hold (x half-spread); spreads to the other teams |
| deflect − / + (per team) | move one team's fair by hand; also spreads to the others; stays until reset |
| our weight vs market mid | fair = this x our theo + rest x the market mid |
| rebate settings | how many markets to keep within the rebate width (80% + spare) |
| trade on crossing prices | sell into bids above where we'd offer / buy offers below where we'd bid |
| max position / hard limit / max budget | risk limits, checked against the live position |

## The page: three tabs + a sidebar (one file each, so people can work on them separately)

| File | What's in it |
|---|---|
| `tdesk.html` | the shell: shared styles, top bar (market, Quoting ON, Result incoming, KILL, MTM P&L), layout |
| `static/core.js` | shared: refreshes `S` (= `/api/state`) every second, helpers (`post`, `setCtl`, `toast`, `hit`, `pull`, `f2`, `esc` ...), tab switching |
| `static/sidebar.js` | always visible on the right: bracket, results & fixtures (enter scores here) |
| `static/tab_model.js` | **Model theos**: deflection %, market blend, manual deflection, board, pricing table, setup & mapping |
| `static/tab_positions.js` | **Positions**: instruments & MTM P&L, trades, manual quotes, resting orders, quoting settings |
| `static/tab_hedging.js` | **Hedging**: P&L by finishing position per contract (starter - build hedges here) |

A tab file is self-contained: it calls `TABS.register({id, title, html, render})`, where `render(S)`
runs every second while the tab is open. Keep element ids prefixed per tab (`m...`, `p...`, `h...`)
so tabs don't clash. Any `<input data-ctl="name">` saves that setting automatically.
Edit a tab, save, reload the page: no server restart needed for `.html`/`.js` changes.
New data needs a change in `state_json()` in `tdesk_server.py` (and a restart).

Files: `tdesk_server.py` (server + quoter), `tdesk.html` + `static/` (page), `tpricer.py` (model),
`deskgate.py` (exchange connection), `book.py`, `fastmarket.py`, `market_events.py`,
`pricing.py`, `mockx.py` (exchange client), `test_tdesk.py` (tests),
`tournament_pricer_brief_1.md` (the spec).
