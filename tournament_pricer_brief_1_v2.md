# Brief: tournament player pricer + live dashboard

## Goal
We make markets on each player's **final payout** in an office tournament. Build a model plus a **local dashboard** (Streamlit or similar) that outputs, for every player: a theo, our theo blended with the market's, a spread and an inventory skew. During the game we will have **no AI access, only the dashboard**. Every choice that's still unknown, especially the seeding model, must therefore be a control in the dashboard (dropdown, slider or input), never hard-coded.

## Confirmed format
- **16 players in 4 groups (A–D) of 4.** Each group is a round robin, so each player plays 3 games. Scoring is 3/1/0.
- **Group winners** go straight to the quarter-finals. **2nd and 3rd place** play a crossover, and the crossover winners go into the quarter-finals. **4th place** is out.
- Bracket (from the image):

| Round | Matches |
|---|---|
| Crossovers | C1: 2A v 3B · C2: 2B v 3A · C3: 2C v 3D · C4: 2D v 3C |
| Quarter-finals | QF A: 1A v C2 · QF B: 1D v C3 · QF C: 1B v C1 · QF D: 1C v C4 |
| Semi-finals | QF A v QF B · QF C v QF D |
| Final / 3rd-place playoff | SF winners / SF losers |

- **Payouts, one per player:** group exit 0, crossover exit 5, QF exit 10, 4th 25, 3rd 50, 2nd 75, 1st 100. The **pot is fixed at 310**, so player theos must always add up to 310.
- Structural facts, which double as tests:
  - Each 3rd-placed player who wins their crossover meets **their own group winner** in the QF.
  - Every semi-final is one A/B player against one C/D player, so the A+B group total lies in [105, 205] and A+B + C+D = 310.
- **Group tiebreakers (confirmed):** points, then goal difference, then goals scored, then head-to-head, then random as a final fallback.
- **Knockout draws (TBD):** default is extra time at λ × 1/3, then a 50/50 shootout. Make it config.
- **The event is historical, not live.** Results are revealed match by match, and there is no in-play pricing.

## Timeline on the day
- **13:30:** we receive the seeding (teams ranked 1–16, and possibly the groups).
- **13:30–14:00:** set up the dashboard, which must take under 10 minutes.
- **14:00:** trading starts.

Setup has to be quick and simple. Keep anything advanced out of the way.

## Seeding setup: 4 simple steps in the dashboard
Keep two questions separate:
- **Who is in which group** (allocation).
- **How much stronger a top seed is than a low seed** (strength).

**Step 1: Teams and seeds.** Enter or paste 16 team names in seed order, 1 to 16.

**Step 2: Groups.** Choose one:
- **Given by organiser.** Type which group each team is in. This is the default if groups are provided.
- **Snake.** Fill automatically: 1–4 → A–D, 5–8 → D–A, 9–12 → A–D, 13–16 → D–A. That gives A = 1, 8, 9, 16; B = 2, 7, 10, 15; C = 3, 6, 11, 14; D = 4, 5, 12, 13.
- **Pots.** Seeds 1–4, 5–8, 9–12 and 13–16 each form a pot, one team from each pot per group. Groups are then entered manually after the draw.

**Step 3: Strength gap.** A single control with three presets plus a slider. The preset is defined by a number the user can understand, **P(seed 1 beats seed 16 in one match)**:

| Preset | P(1 beats 16) | Meaning |
|---|---|---|
| Even field | ~40% | Seeding barely matters |
| **Moderate** (default) | ~60% | Seeding matters, upsets are common |
| Strong favourites | ~75% | Top seeds clearly better |

- Under the hood, strengths are S_r = σ_S · Φ⁻¹(1 − (r − 0.5)/16). This is evenly spread rank-based scoring, an assumption rather than fitted evidence.
- The code solves for the σ_S that hits the chosen P(1 beats 16).
- The user never sees σ_S.

**Step 4 (optional): our own view.** After seeding is revealed, the user can override with their own information:
- **Re-order:** drag teams into our own strength ranking. The default is the seed order.
- **Nudge:** set each team to −2, −1, 0, +1 or +2 tiers. One tier is the gap between neighbouring seeds' strengths × 2.
- **Confidence in our view:** Low, Medium or High. This sets how much the override moves strengths away from the seeding (blend weight 0.25 / 0.5 / 0.75), and also the prior variance. **Low is the default**, because our information may be weak.

**Readouts next to the controls (must have)**, updating live:
- P(seed 1 beats 16)
- P(seed 1 beats 2)
- P(draw between equals)
- P(seed 1 wins tournament)
- Expected goals in a 1 v 16 match
- Opening theo for each team

The user tunes the controls by feel until these look sensible.

**Hide under "Advanced":** alternative strength shapes (linear rank, log rank, pot-based levels), external ratings input, and fitting strengths to the market mids. These are for later use, not needed on the day.

## Model

### 1. Match model: Poisson goals with a mismatch term
- d = S_i − S_j, and g(d) = β·d + κ·d·|d|
- log λ_i = μ + g(d)/2, and log λ_j = μ − g(d)/2
- κ > 0 makes mismatches between very good and very bad players more one-sided.
- Optional extras:
  - Dixon–Coles ρ, to correct how often low scores (0-0, 1-1) occur.
  - A negative-binomial dispersion α, for when blowouts are more common than Poisson predicts.
- Defaults: κ = ρ = α = 0. μ is set from the expected goals per game. All are sliders, and the model can refit them as data arrives.

### 2. Updating strengths after every match
- After each result, refit all strengths jointly by MAP: the priors from the seeding mode plus the scoreline likelihood of every completed match.
- Take the posterior covariance from the Hessian (Laplace approximation).
- Results against strong opponents count for more automatically, and 3 games can't swamp the prior.

### 3. Monte Carlo engine
Use N ≈ 100k simulations, vectorised, at under ~3 s per run. In each simulation:
1. Draw strengths from the posterior.
2. Simulate all remaining matches.
3. Apply the tiebreakers.
4. Play out the bracket.
5. Record each player's payout.

Outputs per player:
- **Theo** = mean payout
- P(group finish 1–4)
- P(each knockout result)
- Payout standard deviation

Plus the **payout covariance matrix Σ** across players. Assert that Σ theo = 310.

### 4. Luck
Luck = theo − theo of the same strength averaged over random group placements. It's derived, not an input.

### 5. Market blend
- Input: the best bid and offer per player.
- Preferred method: blend at the **strength level**.
  1. Fit the market-implied strengths S_mkt to the market mids.
  2. S_blend = w·S_model + (1 − w)·S_mkt, where w = τ_p/(τ_p + τ_m). τ_m comes from spread and size; τ_p from posterior precision. Show w and allow overriding it.
  3. Re-simulate with S_blend.
- Fallback: a linear blend of theos, rescaled so they add up to 310.

### 6. Spread
- Half-spread_i = k · σ_next,i + c
- σ_next,i is the standard deviation of player i's theo across the outcomes of the **next match or round**: enumerate scoreline buckets, re-run the theo for each, and take the probability-weighted standard deviation.
- Calculate it per player, because an individual's σ can rise.
- Widen before a result is revealed; narrow once it's in. This keeps us tight near the top of the book between events.

### 7. Skew
- Quote centre_i = blended theo_i − γ · (Σq)_i, where q is our positions.
- Because it uses the full Σ, offsetting positions don't get skewed.

### 8. Arbitrage and parity checks
- **Whole field:** alert if Σ bids > 310 or Σ offers < 310.
- **A+B and C+D halves:** alert if Σ offers < 105 or Σ bids > 205.
- **Sub-groups that appear later:**
  - Eliminated players have a known payout.
  - The 4 semi-finalists sum to 250.
  - The 2 finalists sum to 175.
  - The 2 in the 3rd-place playoff sum to 75.
- **Bracket bounds.** Find these by enumerating all 2¹² = 4,096 knockout outcomes. They must hold for every outcome, whatever the strengths:

| Set of slots | Min | Max | Equal-strength value |
|---|---|---|---|
| Single 1X | 10 | 100 | 36.25 |
| Single 2X or 3X | 5 | 100 | 20.625 |
| Crossover pair (e.g. 2A + 3B) | 15 | 105 | 41.25 |
| QF trio (e.g. 1A + 2B + 3A) | 40 | 115 | 77.5 |
| **Semi-final half** (1A, 2B, 3A, 1D, 2C, 3D) or (1B, 2A, 3B, 1C, 2D, 3C) | **130** | **180** | 155 |
| One group (1X + 2X + 3X, with 4X = 0) | 20 | 185 | 77.5 |
| Groups A+B or C+D | 105 | 205 | 155 |
| Groups A+C or A+D | 40 | 270 | 155 |
| All 16 | 310 | 310 | 310 |

- Write a general function: **for any set of contracts, given the results so far, return the min and max achievable total payout.** Enumerate the remaining outcomes, including group finishing orders while the group stage is still running.
- The dashboard alerts when Σ offers < min (buy the whole set) or Σ bids > max (sell the whole set), net of fees. The bot's own quotes must never create such a breach.
- **Self-check:** our bid_i ≤ 310 − Σ(best offers of everyone else), and our offer_i ≥ 310 − Σ(best bids of everyone else).

## Exchange connection and bot
- **Contracts:** the exchange lists one future per team, which settles at that team's final payout score (0/5/10/25/50/75/100). The contract names are only given out in the room, between the seeding and the start of trading.
- **Contract mapping step (setup, needs to take about 2 minutes):**
  - Pull the list of contracts from the exchange.
  - Auto-match each contract symbol to a team name using fuzzy matching (case, abbreviations, country codes).
  - Show the matches in an editable table for us to confirm, and flag anything unmatched or ambiguous.
  - Fail loudly if any contract is unmatched, or if two contracts map to the same team. No quotes are sent until all 16 are confirmed.
  - Save the mapping.
- **Adapter interface:** the exchange API is unknown until we're in the room, so isolate it behind one module with the following methods:
  - `list_contracts()`
  - `get_book(symbol)`: best bid/offer and size
  - `get_positions()`
  - `get_fills()`
  - `place_order(symbol, side, px, qty, tif)`
  - `cancel(order_id)` / `cancel_all()`

  Ship a **manual/paper adapter** too: the market table is typed into the dashboard and orders are only displayed, never sent. That way everything works even if we can't connect.
- **Config, filled in once in the room:** tick size, price limits (0–100), contract size, position limits, fees, rate limits.
- **Bot behaviour:**
  - Quote the bid and offer from section 7 on all 16 contracts, rounded outward to the tick.
  - Re-quote when the theo moves by more than one tick or when a fill changes our position.
  - **Pull all quotes before a result is revealed.** Re-quote after the result has been entered and the model has been recomputed.
  - Arbitrage: when the whole-field or A+B/C+D checks are breached by more than fees plus ticks, offer to hit all the legs at once, after confirmation.
  - **Kill switch:** a big button that runs cancel_all and stops quoting.
- **Settlement checks:** payouts that are already known (eliminated teams) are fixed in the model, and the bot never quotes outside the range of payouts still possible for that team.

## Dashboard layout
0. **Exchange:** connection status, the contract mapping table, the kill switch.
1. **Setup:** the 4 seeding steps above, plus the readouts.
2. **Results:** enter each match score as it's revealed, with a recompute button. Show the next fixtures.
3. **Market and positions:** an editable best bid/offer table, plus our positions.
4. **Pricing table**, one row per player:
   - group, strength ± sd
   - P(group finish 1–4)
   - knockout probabilities
   - theo, luck
   - market mid, blend w, blended theo
   - σ_next
   - our bid / offer
   - position, skew
5. **Alerts:** parity and arbitrage alerts, and the sum checks.
6. **Bracket view:** current probabilities for each slot.

Save state to disk after every change, so nothing is lost if the app restarts.

## Tests
- Equal strengths:
  - Every player's theo = 310/16 = 19.375.
  - Group winner = 36.25, 2nd = 3rd = 20.625, 4th = 0.
  - The A+B total = 155.
- Each strength-gap preset hits its target P(1 beats 16) to within 1%. Snake allocation gives A = {1, 8, 9, 16} and seeds summing to 34 in every group.
- Tiebreakers: build test groups that are level on points, then on goal difference, then on goals scored, and check that each is resolved in the right order.
- The Poisson draw rate matches its analytic value; the knockout simulation matches an analytic bracket recursion.
- MAP updating: a big upset moves strengths but stays limited by the prior.
- Contract mapping: given messy symbols (e.g. "BRA_FUT", "Brazil", "bra"), it maps them correctly, and refuses to quote when a contract is unmatched or two contracts map to the same team.
- Bot: quotes stay within the tick and the range of possible payouts; quotes are pulled before each result; the kill switch cancels everything.
