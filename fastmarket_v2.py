"""Non-blocking sending on top of mockx.Market (quirk 3).

mockx's send() waits a full round trip (~10 ms) for every command, so the 3
legs of an arb took 3 round trips plus waiting. Here post() writes the command
to the socket and returns immediately; replies ("acks") are matched up later
in poll(). All 3 legs leave within a fraction of a millisecond.

Rules that still apply:
  * ONE thread: call post() and poll() from the same thread (mockx isn't thread-safe)
  * a rejected command is reported, never resent automatically
  * an ack that never comes (quirk 5) may still have executed: check, don't resend
"""
import json
import time
from collections import deque
from queue import Empty

from mockx_v2 import to_px


class FastMarket:
    def __init__(self, market, max_per_sec=15, dry_run=False):
        self.m = market
        self.me = market.you
        self.max_per_sec = max_per_sec   # the exchange allows 20/s, shared with your browser
        self.dry_run = dry_run
        self.sent_times = deque()        # when each command in the last second went out
        self.pending = {}                # command id -> {"cmd", "at", "on_ack"}
        self.n = 0
        self.last_ping = time.time()

    @property
    def closed(self):
        return self.m._closed

    def budget(self):
        """Commands we may still send in the current 1-second window."""
        now = time.time()
        while self.sent_times and now - self.sent_times[0] > 1.0:
            self.sent_times.popleft()
        return self.max_per_sec - len(self.sent_times)

    def post(self, command, on_ack=None, force=False):
        """Send without waiting. Returns the command id, or None if over budget.

        force=True ignores our own budget: only for safety commands like pulling quotes.
        """
        if not force and self.budget() <= 0:
            return None
        self.n += 1
        cid = f"f{self.n}"
        self.sent_times.append(time.time())
        if self.dry_run:
            print(f"  (dry run) would send {command}")
            return cid
        if self.m._closed or self.m._ws is None:
            raise ConnectionError("not connected")
        self.m._ws.send(json.dumps({"t": "cmd", "id": cid, "command": command}))
        self.pending[cid] = {"cmd": command, "at": time.time(), "on_ack": on_ack}
        return cid

    def poll(self, timeout=0.005):
        """Wait up to `timeout` for the next event, then take everything queued.

        Queue.get wakes up the moment an event arrives, so there's no polling
        delay. Acks are processed before and after, so callbacks see fresh state.
        """
        self._drain_acks()
        events = []
        try:
            events.append(self.m._events.get(timeout=timeout))
            while True:
                events.append(self.m._events.get_nowait())
        except Empty:
            pass
        self._drain_acks()
        return events

    def _drain_acks(self):
        while True:
            try:
                ack = self.m._acks.get_nowait()
            except Empty:
                return
            p = self.pending.pop(ack.get("id"), None)
            if p is None:
                continue
            if not ack.get("ok"):
                print(f"  rejected {p['cmd'].get('type')}: {ack.get('reason')} {ack.get('detail') or ''}")
            if p["on_ack"]:
                p["on_ack"](ack)

    def overdue(self, secs=5.0):
        """Commands with no reply after `secs` (quirk 5: they may still have executed)."""
        now = time.time()
        return [p["cmd"] for p in self.pending.values() if now - p["at"] > secs]

    def keepalive(self, every=20.0):
        if not self.dry_run and time.time() - self.last_ping >= every:
            self.m._ws.send(json.dumps({"t": "ping"}))   # not a command: no budget, no ack
            self.last_ping = time.time()

    # --- the commands ------------------------------------------------------------
    def order(self, iid, side, price, qty, on_ack=None):
        """Limit order. side "bid" = buy, "ask" = sell. Rests if it doesn't fill."""
        return self.post({"type": "PLACE_ORDER", "instrumentId": iid, "side": side,
                          "price": to_px(price), "qty": qty}, on_ack)

    def quote(self, iid, bid=None, ask=None, qty=1, on_ack=None):
        cmd = {"type": "PLACE_QUOTE", "instrumentId": iid, "qty": qty}
        if bid is not None:
            cmd["bidPrice"] = to_px(bid)
        if ask is not None:
            cmd["askPrice"] = to_px(ask)
        return self.post(cmd, on_ack)

    def cancel(self, order_id, on_ack=None):
        return self.post({"type": "CANCEL_ORDER", "orderId": order_id}, on_ack)

    def cancel_instrument(self, iid, on_ack=None):
        return self.post({"type": "CANCEL_ALL_ON_INSTRUMENT", "instrumentId": iid}, on_ack)

    def cancel_all(self):
        """Pull every order of ours in the market (browser ones too). Always sent."""
        return self.post({"type": "CANCEL_ALL_ON_MARKET"}, force=True)
