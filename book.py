"""Everything we know about one market, kept up to date from its event stream.

Works for any market: instruments are described by their legs
[(type, strike, ratio)], never by hard-coded ids, and instruments the conductor
deploys mid-session are added as they appear.
"""
import time

from market_events import pull_reason
from mockx import from_px


def parse_legs(raw):
    return [(l["type"], from_px(l.get("strike")), l.get("ratio", 1)) for l in raw]


class Book:
    def __init__(self, me):
        self.me = me
        self.reset()

    def reset(self):
        self.inst = {}        # id -> {"name", "legs", "tick"}
        self.top = {}         # id -> {"bid", "ask", "bidQty", "askQty"} (human prices)
        self.last = {}        # id -> last trade price
        self.pos = {}         # id -> my position
        self.my_orders = {}   # order id -> {"iid", "side", "price", "qty"}
        self.fills = []       # my trades: {"oid", "iid", "qty" (signed), "price", "aggressor", "at"}
        self.news = []        # reasons to pull quotes, waiting for the bot to act on
        self.changed = set()  # instruments whose top of book moved since the bot last looked

    # --- building it --------------------------------------------------------------
    def load(self, view):
        """Start from a snapshot. Our orders placed before this have no ids we know."""
        self.reset()
        for inst in view["instruments"]:
            iid = inst["id"]
            self.add(iid, inst["name"], inst["legs"], inst["tickSize"])
            b = inst["bids"][0] if inst["bids"] else None
            a = inst["asks"][0] if inst["asks"] else None
            self.top[iid] = {"bid": from_px(b["price"]) if b else None, "bidQty": b["qty"] if b else 0,
                             "ask": from_px(a["price"]) if a else None, "askQty": a["qty"] if a else 0}
            self.last[iid] = from_px(inst.get("lastTrade"))
            self.pos[iid] = inst["position"]
        self.changed = set(self.inst)

    def add(self, iid, name, legs, tick_raw):
        self.inst[iid] = {"name": name, "legs": parse_legs(legs), "tick": tick_raw / 10_000}
        self.top.setdefault(iid, {"bid": None, "ask": None, "bidQty": 0, "askQty": 0})
        self.pos.setdefault(iid, 0)

    def apply(self, ev):
        for t in ev.get("tob", []):   # every event carries the new top of book it caused
            self.top[t["instrumentId"]] = {"bid": from_px(t["bid"]), "bidQty": t["bidQty"],
                                           "ask": from_px(t["ask"]), "askQty": t["askQty"]}
            self.changed.add(t["instrumentId"])
        kind = ev.get("type")
        if kind == "INSTRUMENT_DEPLOYED":
            self.add(ev["instrumentId"], ev["name"], ev["legs"], ev["tickSize"])
        elif kind == "ORDER_PLACED" and ev.get("owner") == self.me:
            self.my_orders[ev["orderId"]] = {"iid": ev["instrumentId"], "side": ev["side"],
                                             "price": from_px(ev["price"]), "qty": ev["qty"]}
        elif kind == "ORDER_CANCELLED":
            self.my_orders.pop(ev.get("orderId"), None)
        elif kind == "TRADE":
            self._trade(ev)
        elif kind == "__error__":
            print("  exchange error:", ev.get("message"))
        reason = pull_reason(ev)
        if reason:
            self.news.append(reason)

    def _trade(self, ev):
        iid, qty, px = ev["instrumentId"], ev["qty"], from_px(ev["price"])
        self.last[iid] = px
        for oid in (ev.get("restingOrderId"), ev.get("aggressingOrderId")):
            o = self.my_orders.get(oid)
            if o:
                o["qty"] -= qty
                if o["qty"] <= 0:
                    del self.my_orders[oid]
        if self.me not in (ev.get("buyer"), ev.get("seller")):
            return
        signed = qty if ev.get("buyer") == self.me else -qty
        self.pos[iid] = self.pos.get(iid, 0) + signed
        took = ev.get("aggressor") == self.me
        self.fills.append({"oid": ev.get("aggressingOrderId") if took else ev.get("restingOrderId"),
                           "iid": iid, "qty": signed, "price": px, "aggressor": took, "at": time.time()})
        print(f"{time.strftime('%H:%M:%S')}  FILL {'BUY ' if signed > 0 else 'SELL'} {qty} "
              f"{self.name(iid)} @ {px:,.2f} -> position {self.pos[iid]}")

    # --- questions the bot asks ---------------------------------------------------
    def name(self, iid):
        return self.inst.get(iid, {}).get("name", iid)

    def legs(self, iid):
        return self.inst.get(iid, {}).get("legs", [])

    def singles(self):
        """(type, strike) -> id, for every one-leg, ratio-1 instrument (plain calls, puts, future)."""
        out = {}
        for iid, inst in self.inst.items():
            legs = inst["legs"]
            if len(legs) == 1 and legs[0][2] == 1:
                out[legs[0][:2]] = iid
        return out

    def future_id(self):
        return self.singles().get(("future", None))

    def trade_price(self, iid, side):
        """Price we'd trade at now taking liquidity. side "bid" = we buy (pay the ask)."""
        t = self.top.get(iid)
        if t is None:
            return None
        return t["ask"] if side == "bid" else t["bid"]

    def my_quotes(self, iid):
        """{"bid": [prices], "ask": [prices]} of my resting orders on this instrument."""
        out = {}
        for o in self.my_orders.values():
            if o["iid"] == iid:
                out.setdefault(o["side"], []).append(o["price"])
        return out
