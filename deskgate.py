"""The exchange side of the tournament desk: one thread owns the connection and sends everything.

Cut down from the original positions dashboard: REST snapshots for the page, and a gateway
thread that keeps one websocket per market open (the exchange pulls ALL your API orders a few
seconds after any API connection closes), applies its events to a Book, pulls everything on
news, and runs order jobs for the HTTP threads (the mockx client is not thread-safe).
"""
import os
import secrets  # noqa: F401  (tdesk_server makes its page token with it)
import threading
import time
import traceback
from queue import Empty, Queue

from book import Book
from fastmarket import FastMarket
from mockx import ApiError, Exchange

try:                                      # the API key lives in my_keys.py (never commit or share that file)
    import my_keys
    if getattr(my_keys, "MOCKX_KEY", "") and my_keys.MOCKX_KEY != "PASTE_YOUR_KEY_HERE":
        os.environ.setdefault("MOCKX_KEY", my_keys.MOCKX_KEY)
except ImportError:
    pass

BASE = os.environ.get("MOCKX_BASE", "https://mock-exchange.fly.dev")
MAX_QTY = 10                 # per order
FAT_FINGER_FRAC = 0.05       # refuse buying more than 5% above the best offer / selling 5% below the bid ...
FAT_FINGER_TICKS = 20        # ... (or 20 ticks, whichever is wider)
MAX_CMDS_PER_SEC = 15        # the exchange allows 20/s per account, shared with the browser
ACK_WAIT = 2.0               # seconds to wait for the exchange's reply
PULL_ON_NEWS = True          # pull all my orders in a market when news / a market event hits it
NEWS_PAUSE_SECS = 60         # (the desk shortens this to its own setting)


def api_key():
    key = os.environ.get("MOCKX_KEY")
    if not key:
        raise SystemExit("No API key: copy my_keys_template.py to my_keys.py and paste your key into it.")
    return key


_ex = None
_cache = {}                  # market id -> (time, snapshot): several tabs polling share one REST call
_cache_lock = threading.Lock()


def snapshot(market_id, max_age=0.3):
    global _ex
    with _cache_lock:
        hit = _cache.get(market_id)
        if hit and time.time() - hit[0] < max_age:
            return hit[1]
        if _ex is None:
            _ex = Exchange(BASE, api_key())
    view = _ex.snapshot(market_id)
    with _cache_lock:
        _cache[market_id] = (time.time(), view)
    return view


class Gateway(threading.Thread):
    def __init__(self, enabled):
        super().__init__(daemon=True)
        self.enabled = enabled
        self.jobs = Queue()
        self.conns = {}      # market id -> {"m", "fm", "book", "paused_until"}
        self.cancel_now = [] # (market id, order id): leftovers of fill-now-or-cancel orders

    def call(self, fn, *args, timeout=10):
        """Run fn(*args) on the gateway thread and return its result (called from HTTP threads)."""
        done = Queue(maxsize=1)
        self.jobs.put((fn, args, done))
        try:
            ok, val = done.get(timeout=timeout)
        except Empty:
            return {"ok": False, "reason": "gateway busy - check your orders before trying again"}
        return val if ok else {"ok": False, "reason": val}

    def run(self):
        while True:
            try:
                fn, args, done = self.jobs.get(timeout=0.002)
                try:
                    done.put((True, fn(*args)))
                except ApiError as e:
                    done.put((False, str(e)))
                except Exception as e:
                    done.put((False, f"{type(e).__name__}: {e}"))
            except Empty:
                pass
            try:
                self.pump()
            except Exception:
                traceback.print_exc()        # never let one bad event kill the gateway
                time.sleep(0.5)

    def pump(self):
        for mid, c in list(self.conns.items()):
            for ev in c["fm"].poll(timeout=0):
                c["book"].apply(ev)
            if c["fm"].closed:
                print(f"{time.strftime('%H:%M:%S')}  connection to {mid} dropped - the exchange pulls "
                      f"our API orders there; reconnecting when needed")
                del self.conns[mid]
                continue
            c["fm"].keepalive()
            if c["book"].news:
                for reason in c["book"].news:
                    print(f"{time.strftime('%H:%M:%S')}  {mid}: {reason}")
                c["book"].news.clear()
                c["paused_until"] = time.time() + NEWS_PAUSE_SECS
                if PULL_ON_NEWS and self.enabled:
                    c["fm"].cancel_all()
                    print(f"{time.strftime('%H:%M:%S')}  PULLED all my orders in {mid}")
            c["book"].changed.clear()
        while self.cancel_now:
            mid, oid = self.cancel_now.pop()
            c = self.conns.get(mid)
            if c and oid in c["book"].my_orders:
                c["fm"].cancel(oid)

    # --- jobs (all run on the gateway thread) ----------------------------------------
    def conn(self, mid):
        c = self.conns.get(mid)
        if c:
            return c
        m = Exchange(BASE, api_key()).join(mid)
        deadline = time.time() + 5
        while m.view is None and time.time() < deadline:
            time.sleep(0.01)
        if m.view is None:
            m.close()
            raise ApiError("no snapshot from the exchange")
        book = Book(m.you)
        book.load(m.view)
        book.changed.clear()
        c = {"m": m, "fm": FastMarket(m, MAX_CMDS_PER_SEC), "book": book, "paused_until": 0.0}
        self.conns[mid] = c
        print(f"{time.strftime('%H:%M:%S')}  connected to {mid} (orders can be sent)")
        return c

    def wait(self, c, cond, secs=ACK_WAIT):
        end = time.time() + secs
        while not cond() and time.time() < end:
            for ev in c["fm"].poll(timeout=0.002):
                c["book"].apply(ev)

    def check(self, book, iid, side, price, qty, override):
        """None if the order is fine, else {"ok": False, "reason"}."""
        inst = book.inst.get(iid)
        if inst is None:
            return {"ok": False, "reason": "unknown instrument"}
        if not isinstance(qty, int) or not 1 <= qty <= MAX_QTY:
            return {"ok": False, "reason": f"quantity must be 1 to {MAX_QTY}"}
        tick = inst["tick"]
        if abs(price / tick - round(price / tick)) > 1e-6:
            return {"ok": False, "reason": f"price must be a multiple of the tick ({tick:g})"}
        # fat-finger check, only in the direction that can hurt
        top = book.top.get(iid, {})
        if side == "bid":
            ref = next((x for x in (top.get("ask"), book.last.get(iid), top.get("bid")) if x is not None), None)
        else:
            ref = next((x for x in (top.get("bid"), book.last.get(iid), top.get("ask")) if x is not None), None)
        if ref is not None and not override:
            limit = max(FAT_FINGER_FRAC * abs(ref), FAT_FINGER_TICKS * tick)
            beyond = price - ref if side == "bid" else ref - price
            if beyond > limit:
                return {"ok": False, "reason": f"fat-finger check: {price:g} is {beyond:g} "
                        f"{'above the best offer' if side == 'bid' else 'below the best bid'} ({ref:g})"}
        return None

    def place(self, mid, iid, side, qty, price, mode, override):
        """One order. mode "ioc" = fill now at this price or better, cancel the rest; "limit" = rest."""
        if not self.enabled:
            return {"ok": False, "reason": "paper mode: restart with --trade to send orders"}
        c = self.conn(mid)
        book, fm = c["book"], c["fm"]
        if side not in ("bid", "ask") or mode not in ("limit", "ioc"):
            return {"ok": False, "reason": "bad side or mode"}
        try:
            price = float(price)
        except (TypeError, ValueError):
            return {"ok": False, "reason": "enter a price"}
        bad = self.check(book, iid, side, price, qty, override)
        if bad:
            return bad
        res = {}
        t0 = time.perf_counter()
        if fm.order(iid, side, price, qty, on_ack=lambda a: res.update(ack=a, ms=(time.perf_counter() - t0) * 1000)) is None:
            return {"ok": False, "reason": "over our rate budget - wait a second and try again"}
        self.wait(c, lambda: "ack" in res)
        if "ack" not in res:
            return {"ok": False, "reason": "no reply within 2s - it MAY have gone through. Check before sending again."}
        ack = res["ack"]
        if not ack.get("ok"):
            return {"ok": False, "reason": f"rejected: {ack.get('reason')} {ack.get('detail') or ''}".strip()}
        oid = f"o{ack['seqs'][0]}" if ack.get("seqs") else None
        self.wait(c, lambda: False, secs=0.05)            # let immediate fills arrive
        filled = sum(abs(f["qty"]) for f in book.fills if f["oid"] == oid)
        if mode == "ioc" and filled < qty and oid:
            self.cancel_now.append((mid, oid))
        print(f"{time.strftime('%H:%M:%S')}  {mid}: {'BUY' if side == 'bid' else 'SELL'} {qty} "
              f"{book.name(iid)} @ {price:g} ({mode}) -> {oid}, filled {filled}, ack {res['ms']:.1f} ms")
        return {"ok": True, "oid": oid, "price": price, "filled": filled, "ms": round(res["ms"], 1)}

    def _simple(self, mid, command, what):
        if not self.enabled:
            return {"ok": False, "reason": "paper mode"}
        c = self.conn(mid)
        res = {}
        if c["fm"].post(command, on_ack=lambda a: res.update(ack=a), force=True) is None:
            return {"ok": False, "reason": "over budget"}
        self.wait(c, lambda: "ack" in res)
        ack = res.get("ack")
        if ack is None:
            return {"ok": False, "reason": "no reply within 2s - check your orders"}
        print(f"{time.strftime('%H:%M:%S')}  {mid}: {what} -> {'ok' if ack.get('ok') else ack.get('reason')}")
        return {"ok": bool(ack.get("ok")), "reason": ack.get("reason")}

    def cancel(self, mid, oid):
        return self._simple(mid, {"type": "CANCEL_ORDER", "orderId": oid}, f"cancel {oid}")

    def cancel_all(self, mid):
        return self._simple(mid, {"type": "CANCEL_ALL_ON_MARKET"}, "cancel ALL my orders")


gateway = None
