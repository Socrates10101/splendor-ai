"""Game constants for base-game Splendor."""
from __future__ import annotations

WINNING_POINTS = 15
MAX_TOKENS_IN_HAND = 10      # excess discarded at end of turn
MAX_RESERVED = 3
BOARD_SLOTS_PER_TIER = 4     # face-up cards per tier
TAKE_SAME_MIN_PILE = 4       # may take 2 of one color only if pile >= 4
GOLD_PER_RESERVE = 1

# Token piles by player count: gem tokens per color, plus gold.
TOKENS_BY_PLAYERS = {
    2: {"gems": 4, "gold": 5},
    3: {"gems": 5, "gold": 5},
    4: {"gems": 7, "gold": 5},
}

# Nobles in play = players + 1.
def nobles_in_play(num_players: int) -> int:
    return num_players + 1
