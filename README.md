# Tournament desk

A pricing and market-making dashboard for the 17-team tournament on mock-exchange.fly.dev: a Bo5
qualifier (Apex v Glaive), a 16-team Swiss stage (3 wins through, 3 losses out), an 8-team knockout;
payouts 100 / 60 / 35 / 35 / 20 x4 / 0 (pot 310). It simulates the tournament to get a theo for every
team's payout, then quotes every contract around it. The page runs on your own laptop at http://localhost:8003.

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
names, regions and which contract each team trades as; seeds (same rank in a region = same seed)
nudge in 0.1 steps. Click **Turn Quotes On** to start.

## Several people on one desk (the right way)

Run ONE desk and let the others open it in their browser: **`start_trading_shared.bat`** on the
trading laptop prints a link like `http://10.0.0.5:8003/?k=...`; the others open that link (same
network; allow Python through the Windows firewall when it asks). Everyone then sees and changes the
same desk: quotes, results, fair/edge edits, minimised markets, pulls. Each person keeps their own tab,
charts and zoom. The link lets its holder trade - only give it to the desk. Nobody else runs a server.

## Two desks, two API keys (each person runs their own)

Each person makes their own API key under the same account, puts it in their own `my_keys.py`, and runs
`start_trading.bat` on their own laptop (http://localhost:8003). What the desk does about it:
- Orders from `TEAM_OWNERS` (top of `tdesk_server.py`, `{"julian"}`) are our team's, not the market's: they
  are left out of the market mid, copying the top of book and takes, and show as **team** in the book.
- Positions are the team's: ours + every `TEAM_OWNERS` desk's net position (from the market's trade feed)
  drive pos-adj fair, max position, risk and hedges; the page shows the split "(me x · team y)". P&L is this desk's own.
- **One prices, both quote.** The pricer runs `start_trading.bat` (sets theos, quotes). The other runs
  `start_follow.bat`: its fair and pos-adj fair are the mid of the pricer's best bid / offer, and it quotes (and
  takes) on its own key only where the pricer quotes both sides - when the pricer pulls a market, so does it.
  Position management (hedges, click trades) on the team's combined position works on both.
- `--rate 9` in `start_trading.bat`: each desk sends at most 9 commands a second, so two desks stay under 20.
- If another key's orders ever show under OUR owner name, quoting is blocked (red error at the top): the
  quoter tracks its orders by owner and would otherwise cancel the teammate's.
- Still shared between desks: **Turn Quotes Off** / news pulls cancel everything on the market if the exchange
  scopes cancel-all to the account rather than the key - test this before relying on it.

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

Both tabs show a 4 x 4 grid, one square per team (the two qualifiers share one until one of them
wins; the loser is minimised). Drag a team name onto another square to move it. Before the first
result each pot has one coloured outline; after it, each square's border shows its Swiss record
(red losing .. green winning, key at the bottom). Settled teams are greyed out; ✕ minimises them.

**Markets**: per team, the order book (click a price to trade it at the top-bar qty; light blue =
a price through our position-adjusted fair), our fair (blended with the other market makers' mid),
position-adjusted fair, our width and whether we're copying the top of book. **chart** switches to
the session's top-of-book bid (green) / offer (red), our position-adjusted fair (white), our trades
(filled triangles) and everyone else's (white triangles, aggressor side); wheel zooms, drag pans,
double-click resets. Fair / adj (position-adjusted fair) / Edge can be set beside each book.
Yellow border = we were just traded on. Auto-deflection ON: we keep quoting (PA moved by the
deflection % of our edge per share) for the deflection count, or until the position limit, then the
market switches off until **start quoting**. Auto-deflection OFF: the side we were traded on goes off.

**Position management**: our fair, the pos-adj − fair gap, edge (▲▼ 0.1) and max position (▲▼ 1);
fair edits propagate to every correlated team by payout covariance. Position, average price of the open shares,
market value, MTM P&L, the range a team can still pay, everyone else's position, and the 3 best
1-lot hedges with their edge (🔒 = locked whatever happens); click one to trade it.

| Top bar | What it does |
|---|---|
| MM ≤ X in N | the market-making requirement; ▲▼ on both; **snap to edge** sets those markets to exactly X |
| deflect switch, %, × | auto-deflection on/off, % of our edge per share, how many times before the market switches off |
| max pos | max position per market (each market can have its own in Position management) |
| qty | lots per click-trade |
| Add result | Teams & seeds (names, regions, contracts, seed nudges, series lengths); Qualifier & Swiss (each round's record groups: pick the two teams, click the winner's side); Knockout (QF teams from the 3-0 / 3-1 / 3-2 pools, then click winners) |
| ⚙ | start width, narrowing time, deflection, limits, take settings |
| Turn Quotes On / Off | On re-enables every market; Off pulls everything |

Each result drops our fair edits (the result replaces them), keeps every team's PA − fair gap, and
starts every market wide again (START WIDTH, narrowing to the MM spread over NARROW SECS).

Files: `tdesk_server.py` (server + quoter), `tdesk.html` (page), `tpricer.py` (model),
`deskgate.py` (exchange connection), `book.py`, `fastmarket.py`, `market_events.py`,
`pricing.py` (ticks), `mockx.py` (exchange client), `test_tdesk.py` (tests),
`tournament_pricer_brief_1.md` (the spec).
