"""
A Python client for the mock exchange.

One file, no dependencies beyond `websocket-client`, and deliberately small
enough to read in one sitting — a trainee writing their first bot should be
able to see everything that happens between their code and the exchange.

    pip install websocket-client

    from mockx import Exchange

    with Exchange("http://localhost:8787", "mx_...") as ex:
        market = ex.join("top-speed-of-a-fish-1a2b")
        market.quote("F", bid=95, ask=105, qty=2)
        for event in market.stream():
            print(event["type"])
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from queue import Empty, Queue
from typing import Any, Callable, Iterator
from urllib import error, request

import websocket  # type: ignore[import-untyped]

SCALE = 10_000
"""Prices are whole numbers of 1e-4 price points. Nothing here is a float."""


def to_px(value: float | int) -> int:
    """A human price to the wire's integer. Rounds once, here, and nowhere else."""
    return int(round(float(value) * SCALE))


def from_px(px: int | None) -> float | None:
    """The wire's integer back to something you can print."""
    return None if px is None else px / SCALE


class ApiError(RuntimeError):
    def __init__(self, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


class Rejected(RuntimeError):
    """A command the engine refused. `reason` is the machine-readable code."""

    def __init__(self, reason: str, detail: str | None = None) -> None:
        super().__init__(detail or reason)
        self.reason = reason
        self.detail = detail


@dataclass
class Exchange:
    """REST half of the protocol, plus the way in to a market."""

    base: str
    key: str
    timeout: float = 10.0

    def __post_init__(self) -> None:
        self.base = self.base.rstrip("/")

    def _call(self, method: str, path: str, body: Any = None) -> dict[str, Any]:
        data = None if body is None else json.dumps(body).encode()
        req = request.Request(f"{self.base}/api/v1{path}", data=data, method=method)
        req.add_header("authorization", f"Bearer {self.key}")
        if data is not None:
            req.add_header("content-type", "application/json")
        try:
            with request.urlopen(req, timeout=self.timeout) as res:
                return json.loads(res.read() or b"{}")
        except error.HTTPError as e:
            payload: dict[str, Any] = {}
            try:
                payload = json.loads(e.read() or b"{}")
            except Exception:
                pass
            # The server's own sentence, which is already written to be read.
            raise ApiError(payload.get("error", f"HTTP {e.code}"), e.code) from None
        except error.URLError as e:
            raise ApiError(f"could not reach {self.base}: {e.reason}") from None

    def health(self) -> dict[str, Any]:
        req = request.Request(f"{self.base}/health")
        with request.urlopen(req, timeout=self.timeout) as res:
            return json.loads(res.read())

    def markets(self) -> list[dict[str, Any]]:
        return self._call("GET", "/markets")["markets"]

    def snapshot(self, market_id: str) -> dict[str, Any]:
        return self._call("GET", f"/markets/{market_id}/snapshot")["view"]

    def events(self, market_id: str, from_seq: int = 0, limit: int = 5_000) -> list[dict[str, Any]]:
        """Every event from `from_seq`, paging until the server says there is no more."""
        out: list[dict[str, Any]] = []
        cursor = from_seq
        while True:
            page = self._call("GET", f"/markets/{market_id}/events?fromSeq={cursor}&limit={limit}")
            out.extend(page["events"])
            if not page.get("more"):
                return out
            cursor = page["lastSeq"]

    def join(self, market_id: str, fold: bool = False) -> "Market":
        """
        Open a market.

        `fold=False` asks for a rendered snapshot and then the live stream, which
        is what a bot reacting to trades or information wants. `fold=True` asks
        for the whole event log instead, for a bot that maintains the book.
        """
        market = Market(self, market_id, fold=fold)
        market.connect()
        return market

    def __enter__(self) -> "Exchange":
        return self

    def __exit__(self, *_: object) -> None:
        for m in list(self._open):
            m.close()

    _open: list["Market"] = field(default_factory=list)


class Market:
    """One market's websocket: send commands, receive events."""

    def __init__(self, exchange: Exchange, market_id: str, fold: bool = False) -> None:
        self.exchange = exchange
        self.market_id = market_id
        self.fold = fold
        self.you: str | None = None
        self.info: dict[str, Any] = {}
        self.last_seq = 0
        self.view: dict[str, Any] | None = None

        self._ws: websocket.WebSocket | None = None
        self._events: "Queue[dict[str, Any]]" = Queue()
        self._acks: "Queue[dict[str, Any]]" = Queue()
        self._n = 0
        self._closed = False
        self._reader: threading.Thread | None = None
        exchange._open.append(self)

    # --------------------------------------------------------------- lifecycle

    def connect(self) -> None:
        url = self.exchange.base.replace("https://", "wss://").replace("http://", "ws://") + "/ws"
        self._ws = websocket.create_connection(url, timeout=30)
        # The timeout is for connecting only. Left on, it applies to every read,
        # and a market that is quiet for 30 seconds silently disconnected every
        # bot. Quiet is normal: everyone waiting on a hint is quiet.
        self._ws.settimeout(None)
        hello: dict[str, Any] = {
            "t": "hello",
            "token": self.exchange.key,
            "marketId": self.market_id,
        }
        if self.fold:
            hello["fromSeq"] = self.last_seq
        self._ws.send(json.dumps(hello))
        self._reader = threading.Thread(target=self._read, daemon=True)
        self._reader.start()

        # Wait for the handshake, so a caller that immediately places an order
        # is not racing the welcome.
        deadline = time.time() + 10
        while self.you is None and time.time() < deadline:
            if self._closed:
                raise ApiError("the exchange closed the connection during the handshake")
            time.sleep(0.005)
        if self.you is None:
            raise ApiError("no welcome from the exchange")

    def close(self) -> None:
        self._closed = True
        if self._ws is not None:
            try:
                self._ws.close()
            except Exception:
                pass
            self._ws = None

    def __enter__(self) -> "Market":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    # ------------------------------------------------------------------ reading

    def _read(self) -> None:
        while not self._closed and self._ws is not None:
            try:
                raw = self._ws.recv()
            except Exception:
                break
            if not raw:
                break
            msg = json.loads(raw)
            kind = msg.get("t")

            if kind == "welcome":
                self.you = msg["you"]
                self.info = msg.get("market", {})
            elif kind == "snapshot" or kind == "resync":
                self.view = msg["view"]
                self.last_seq = self.view.get("seq", self.last_seq)
            elif kind == "events":
                for event in msg["events"]:
                    seq = event["seq"]
                    if seq > self.last_seq + 1 and self.last_seq > 0:
                        # Never guess at a gap. Ask for everything again.
                        self._resync()
                        break
                    self.last_seq = seq
                    self._events.put(event)
            elif kind == "ack":
                self._acks.put(msg)
            elif kind == "pong":
                pass
            elif kind == "error":
                self._events.put({"type": "__error__", "message": msg.get("message", "")})
        self._closed = True

    def _resync(self) -> None:
        if self._ws is None:
            return
        self.last_seq = 0
        self._ws.send(json.dumps({"t": "resync", "fromSeq": 0}))

    def stream(self, timeout: float | None = None) -> Iterator[dict[str, Any]]:
        """
        Events as they arrive, oldest first.

        Stops when the connection closes. With a timeout it also stops when
        nothing has happened for that long, which is how you write a bot that
        does something on a quiet market.
        """
        while not self._closed or not self._events.empty():
            try:
                yield self._events.get(timeout=timeout if timeout is not None else 0.25)
            except Empty:
                if timeout is not None:
                    return
                continue

    # ------------------------------------------------------------------ sending

    def send(self, command: dict[str, Any], wait: float = 5.0) -> dict[str, Any]:
        """
        Send one command and wait for its acknowledgement.

        Raises `Rejected` with the engine's own reason, so a bot can branch on
        `price-band` or `position-limit` rather than parsing prose.
        """
        if self._ws is None or self._closed:
            raise ApiError("not connected")
        self._n += 1
        cid = f"py-{self._n}"
        self._ws.send(json.dumps({"t": "cmd", "id": cid, "command": command}))

        deadline = time.time() + wait
        held: list[dict[str, Any]] = []
        try:
            while time.time() < deadline:
                try:
                    ack = self._acks.get(timeout=0.05)
                except Empty:
                    if self._closed:
                        raise ApiError("the connection closed before the acknowledgement")
                    continue
                if ack.get("id") != cid:
                    held.append(ack)
                    continue
                if not ack.get("ok"):
                    raise Rejected(ack.get("reason", "unknown"), ack.get("detail"))
                return ack
            raise ApiError("no acknowledgement from the exchange")
        finally:
            for ack in held:
                self._acks.put(ack)

    # --- the things a bot actually does ---------------------------------------

    def order(
        self,
        instrument: str,
        side: str,
        price: float,
        qty: int,
        override: bool = False,
        client_order_id: str | None = None,
    ) -> dict[str, Any]:
        command: dict[str, Any] = {
            "type": "PLACE_ORDER",
            "instrumentId": instrument,
            "side": side,
            "price": to_px(price),
            "qty": qty,
        }
        if override:
            command["override"] = True
        if client_order_id:
            command["clientOrderId"] = client_order_id
        return self.send(command)

    def quote(
        self,
        instrument: str,
        bid: float | None = None,
        ask: float | None = None,
        qty: int = 1,
        override: bool = False,
    ) -> dict[str, Any]:
        command: dict[str, Any] = {"type": "PLACE_QUOTE", "instrumentId": instrument, "qty": qty}
        if bid is not None:
            command["bidPrice"] = to_px(bid)
        if ask is not None:
            command["askPrice"] = to_px(ask)
        if override:
            command["override"] = True
        return self.send(command)

    def take_one(self, instrument: str, side: str) -> dict[str, Any]:
        """`side` is the side YOU are on: "bid" buys at the best offer."""
        return self.send({"type": "TAKE_ONE", "instrumentId": instrument, "side": side})

    def cancel(self, order_id: str) -> dict[str, Any]:
        return self.send({"type": "CANCEL_ORDER", "orderId": order_id})

    def off_quotes(self, instrument: str | None = None) -> dict[str, Any]:
        if instrument is None:
            return self.send({"type": "CANCEL_ALL_ON_MARKET"})
        return self.send({"type": "CANCEL_ALL_ON_INSTRUMENT", "instrumentId": instrument})

    # --- reading the book -----------------------------------------------------

    def refresh(self) -> dict[str, Any]:
        """Pull a fresh rendered view over REST."""
        self.view = self.exchange.snapshot(self.market_id)
        return self.view

    def instrument(self, instrument_id: str) -> dict[str, Any] | None:
        for inst in (self.view or {}).get("instruments", []):
            if inst["id"] == instrument_id:
                return inst
        return None

    def best(self, instrument_id: str) -> tuple[float | None, float | None]:
        """Best bid and best offer, as prices you can print."""
        inst = self.instrument(instrument_id)
        if inst is None:
            return (None, None)
        bid = inst["bids"][0]["price"] if inst["bids"] else None
        ask = inst["asks"][0]["price"] if inst["asks"] else None
        return (from_px(bid), from_px(ask))

    def mid(self, instrument_id: str) -> float | None:
        bid, ask = self.best(instrument_id)
        if bid is None or ask is None:
            return None
        return (bid + ask) / 2

    def on(self, event_type: str, handler: Callable[[dict[str, Any]], None]) -> None:
        """Run `handler` for every event of one type, until the connection ends."""
        for event in self.stream():
            if event.get("type") == event_type:
                handler(event)
