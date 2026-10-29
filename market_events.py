"""Shared rules for reacting to exchange events. Used by every bot, any market.

News arrives as INFO_PUBLISHED (with a "text" field). We treat ANYTHING that
isn't routine order flow as a reason to pull our quotes: news, a new
instrument, a pause, settlement, API access changing, or an event type we have
never seen. Being out of the market for a few seconds is cheap; quoting stale
prices into news is not.
"""
from mockx import ApiError, Rejected

ROUTINE = {
    "ORDER_PLACED", "ORDER_CANCELLED", "TRADE",
    "CONDUCTOR_ADDED", "COUNTERPARTY_VISIBILITY_CHANGED",
    "__error__",   # mockx's own wrapper for error messages
}


def pull_reason(event):
    """None for routine events, otherwise a short human reason to pull quotes."""
    kind = event.get("type")
    if kind in ROUTINE:
        return None
    if kind == "INFO_PUBLISHED":
        return f"NEWS: {event.get('text', '')}"
    return f"market event {kind}"


def pull_all(market):
    """Cancel every order of ours in this market (browser ones too). One command."""
    try:
        market.off_quotes()
        return True
    except (Rejected, ApiError) as e:
        # Quirk 5: a timeout may still have worked. Don't loop on it.
        print(f"  pull_all failed: {e} - check your orders by hand")
        return False
