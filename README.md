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

Then, on the page: type the market id (or paste its URL) → Load. **Add result → Teams & seeds**:
names, seeds, groups and which contract each team trades as. Click **All quotes ON** to start.

## Several people on one desk (the right way)

Run ONE desk and let the others open it in their browser: **`start_trading_shared.bat`** on the
trading laptop prints a link like `http://10.0.0.5:8003/?k=...`; the others open that link (same
network; allow Python through the Windows firewall when it asks). Everyone then sees and changes the
same desk: quotes, results, fair/edge edits, minimised markets, pulls. Each person keeps their own tab,
charts and zoom. The link lets its holder trade - only give it to the desk. Nobody else runs a server.

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

## The page: two tabs, one screen, no scrolling (`tdesk.html`)

Both tabs show a grid: one column per group, one cell per team (drag a team name to reorder it
within its group, drag a group header to move the column). Settled teams are greyed out.

**Markets**: per team, the order book (click a price to trade it at the top-bar qty; light blue =
a price through our position-adjusted fair), our fair (blended with the other market makers' mid),
position-adjusted fair, our width and whether we're copying the top of book. **chart** switches to
the session's top-of-book bid (green) / offer (red), our position-adjusted fair (white), our trades
(filled triangles) and everyone else's (white triangles, aggressor side); wheel zooms, drag pans,
double-click resets. Yellow border = we were just traded on: that side stays off until **quotes ON**.

**Position management**: our fair and position-adjusted fair with ▲▼ 0.1 (or type a value): an edit
propagates to every correlated team by payout covariance. Position, average price of the open shares,
market value, MTM P&L, the range a team can still pay, everyone else's position, and the 3 best
1-lot hedges with their edge (🔒 = locked whatever happens); click one to trade it.

| Top bar | What it does |
|---|---|
| MM: spread ≤ X in N markets | the market-making requirement; ▲▼ on both |
| qty | lots per click-trade |
| Add result | first time: names, seeds, groups, contracts. Then the group diagrams (a score box on each line between two teams), then the knockout bracket |
| ⚙ | start width, narrowing time, deflection, limits, take settings |
| All quotes ON / OFF | ON re-enables every market; OFF pulls everything |

Each result drops our fair edits (the result replaces them), keeps every team's PA − fair gap, and
starts every market wide again (START WIDTH, narrowing to the MM spread over NARROW SECS).

Files: `tdesk_server.py` (server + quoter), `tdesk.html` (page), `tpricer.py` (model),
`deskgate.py` (exchange connection), `book.py`, `fastmarket.py`, `market_events.py`,
`pricing.py` (ticks), `mockx.py` (exchange client), `test_tdesk.py` (tests),
`tournament_pricer_brief_1.md` (the spec).
