"""Splendor card and noble definitions (base game).

90 development cards (Lv1:40, Lv2:30, Lv3:20) and 10 noble tiles.

Every card carries a *stable* string ID (e.g. ``L1-00``) so that game records
and analysis reference a card by identity rather than by its transient board
slot. IDs are assigned tier-major then in the fixed authoring order below, so
they never change between runs.

Gem color codes (single letters, used everywhere in the engine and notation):
    w = white / diamond
    u = blue  / sapphire   (``u`` avoids clashing with ``k`` = black)
    g = green / emerald
    r = red   / ruby
    k = black / onyx
    y = gold  / joker       (obtained only by reserving a card)
"""
from __future__ import annotations

from dataclasses import dataclass

COLORS = ("w", "u", "g", "r", "k")
GOLD = "y"
ALL_TOKENS = COLORS + (GOLD,)


@dataclass(frozen=True)
class Card:
    id: str
    tier: int
    bonus: str          # one of COLORS
    points: int
    cost: dict          # color -> count (nonzero entries only)


@dataclass(frozen=True)
class Noble:
    id: str
    points: int
    requirement: dict   # color -> required bonus count


# (tier, bonus, points, cost) in fixed authoring order -> drives stable IDs.
_RAW_CARDS = [
    # ---- Tier 1 (40) ----
    (1, "w", 0, {"k": 1, "u": 2, "g": 2}),
    (1, "w", 0, {"k": 1, "r": 2}),
    (1, "w", 0, {"k": 1, "r": 1, "u": 1, "g": 1}),
    (1, "w", 0, {"u": 3}),
    (1, "w", 0, {"k": 2, "u": 2}),
    (1, "w", 0, {"k": 1, "r": 1, "u": 1, "g": 2}),
    (1, "w", 0, {"k": 1, "w": 3, "u": 1}),
    (1, "w", 1, {"g": 4}),
    (1, "u", 0, {"k": 2, "w": 1}),
    (1, "u", 0, {"k": 1, "w": 1, "r": 2, "g": 1}),
    (1, "u", 0, {"k": 1, "w": 1, "r": 1, "g": 1}),
    (1, "u", 0, {"r": 1, "u": 1, "g": 3}),
    (1, "u", 0, {"k": 3}),
    (1, "u", 0, {"w": 1, "r": 2, "g": 2}),
    (1, "u", 0, {"k": 2, "g": 2}),
    (1, "u", 1, {"r": 4}),
    (1, "g", 0, {"w": 2, "u": 1}),
    (1, "g", 0, {"r": 2, "u": 2}),
    (1, "g", 0, {"w": 1, "r": 3, "g": 1}),
    (1, "g", 0, {"k": 1, "w": 1, "r": 1, "u": 1}),
    (1, "g", 0, {"k": 2, "w": 1, "r": 1, "u": 1}),
    (1, "g", 0, {"k": 2, "r": 2, "u": 1}),
    (1, "g", 0, {"r": 3}),
    (1, "g", 1, {"k": 4}),
    (1, "r", 0, {"w": 3}),
    (1, "r", 0, {"k": 3, "w": 1, "r": 1}),
    (1, "r", 0, {"u": 2, "g": 1}),
    (1, "r", 0, {"k": 2, "w": 2, "g": 1}),
    (1, "r", 0, {"k": 1, "w": 2, "u": 1, "g": 1}),
    (1, "r", 0, {"k": 1, "w": 1, "u": 1, "g": 1}),
    (1, "r", 0, {"w": 2, "r": 2}),
    (1, "r", 1, {"w": 4}),
    (1, "k", 0, {"w": 1, "r": 1, "u": 1, "g": 1}),
    (1, "k", 0, {"r": 1, "g": 2}),
    (1, "k", 0, {"w": 2, "g": 2}),
    (1, "k", 0, {"k": 1, "r": 3, "g": 1}),
    (1, "k", 0, {"g": 3}),
    (1, "k", 0, {"w": 1, "r": 1, "u": 2, "g": 1}),
    (1, "k", 0, {"w": 2, "r": 1, "u": 2}),
    (1, "k", 1, {"u": 4}),
    # ---- Tier 2 (30) ----
    (2, "w", 1, {"k": 2, "r": 2, "g": 3}),
    (2, "w", 1, {"w": 2, "r": 3, "u": 3}),
    (2, "w", 2, {"k": 2, "r": 4, "g": 1}),
    (2, "w", 2, {"r": 5}),
    (2, "w", 2, {"k": 3, "r": 5}),
    (2, "w", 3, {"w": 6}),
    (2, "u", 1, {"r": 3, "u": 2, "g": 2}),
    (2, "u", 1, {"k": 3, "u": 2, "g": 3}),
    (2, "u", 2, {"w": 5, "u": 3}),
    (2, "u", 2, {"u": 5}),
    (2, "u", 2, {"k": 4, "w": 2, "r": 1}),
    (2, "u", 3, {"u": 6}),
    (2, "g", 1, {"w": 3, "r": 3, "g": 2}),
    (2, "g", 1, {"k": 2, "w": 2, "u": 3}),
    (2, "g", 2, {"k": 1, "w": 4, "u": 2}),
    (2, "g", 2, {"g": 5}),
    (2, "g", 2, {"u": 5, "g": 3}),
    (2, "g", 3, {"g": 6}),
    (2, "r", 1, {"k": 3, "r": 2, "u": 3}),
    (2, "r", 1, {"k": 3, "w": 2, "r": 2}),
    (2, "r", 2, {"w": 1, "u": 4, "g": 2}),
    (2, "r", 2, {"k": 5, "w": 3}),
    (2, "r", 2, {"k": 5}),
    (2, "r", 3, {"r": 6}),
    (2, "k", 1, {"w": 3, "u": 2, "g": 2}),
    (2, "k", 1, {"k": 2, "w": 3, "g": 3}),
    (2, "k", 2, {"r": 2, "u": 1, "g": 4}),
    (2, "k", 2, {"w": 5}),
    (2, "k", 2, {"r": 3, "g": 5}),
    (2, "k", 3, {"k": 6}),
    # ---- Tier 3 (20) ----
    (3, "w", 3, {"k": 3, "r": 5, "u": 3, "g": 3}),
    (3, "w", 4, {"k": 7}),
    (3, "w", 4, {"k": 6, "w": 3, "r": 3}),
    (3, "w", 5, {"k": 7, "w": 3}),
    (3, "u", 3, {"k": 5, "w": 3, "r": 3, "g": 3}),
    (3, "u", 4, {"w": 7}),
    (3, "u", 4, {"k": 3, "w": 6, "g": 3}),
    (3, "u", 5, {"w": 7, "g": 3}),
    (3, "g", 3, {"k": 3, "w": 5, "r": 3, "u": 3}),
    (3, "g", 4, {"w": 3, "u": 6, "g": 3}),
    (3, "g", 4, {"u": 7}),
    (3, "g", 5, {"u": 7, "g": 3}),
    (3, "r", 3, {"k": 3, "w": 3, "u": 5, "g": 3}),
    (3, "r", 4, {"g": 7}),
    (3, "r", 4, {"r": 3, "u": 3, "g": 6}),
    (3, "r", 5, {"r": 3, "g": 7}),
    (3, "k", 3, {"w": 3, "r": 3, "u": 3, "g": 5}),
    (3, "k", 4, {"r": 7}),
    (3, "k", 4, {"k": 3, "r": 6, "g": 3}),
    (3, "k", 5, {"k": 3, "r": 7}),
]

_RAW_NOBLES = [
    {"w": 4, "u": 4},
    {"u": 4, "g": 4},
    {"g": 4, "r": 4},
    {"r": 4, "k": 4},
    {"k": 4, "w": 4},
    {"w": 3, "u": 3, "g": 3},
    {"u": 3, "g": 3, "r": 3},
    {"g": 3, "r": 3, "k": 3},
    {"r": 3, "k": 3, "w": 3},
    {"k": 3, "w": 3, "u": 3},
]


def _build_cards():
    cards = {}
    counters = {1: 0, 2: 0, 3: 0}
    for tier, bonus, points, cost in _RAW_CARDS:
        idx = counters[tier]
        counters[tier] += 1
        cid = f"L{tier}-{idx:02d}"
        cards[cid] = Card(id=cid, tier=tier, bonus=bonus, points=points, cost=dict(cost))
    return cards


def _build_nobles():
    return {
        f"N{i}": Noble(id=f"N{i}", points=3, requirement=dict(req))
        for i, req in enumerate(_RAW_NOBLES)
    }


CARDS: dict[str, Card] = _build_cards()
NOBLES: dict[str, Noble] = _build_nobles()

# Card IDs grouped by tier, in stable order (the per-tier deck before shuffling).
CARDS_BY_TIER: dict[int, list[str]] = {
    tier: [cid for cid, c in CARDS.items() if c.tier == tier] for tier in (1, 2, 3)
}
NOBLE_IDS: list[str] = list(NOBLES.keys())


def card(cid: str) -> Card:
    return CARDS[cid]


def noble(nid: str) -> Noble:
    return NOBLES[nid]
