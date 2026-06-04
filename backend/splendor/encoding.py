"""Canonical action indexing and state feature vector.

The IS-MCTS search (v1) works directly on ``GameState`` and ``Move`` objects, so
this module is not on the critical path yet. It exists so a future
``NNEvaluator`` can map moves to a fixed policy head and states to a fixed input
vector without touching the rest of the engine.

Canonical action space (primary actions only; discard / noble are resolved as
part of the same Move):

    [ 0.. 9]  take 3 different colors   (C(5,3) = 10 combos, colors sorted)
    [10..14]  take 2 of one color       (5)
    [15..26]  reserve a face-up card    (3 tiers x 4 slots = 12)
    [27..29]  reserve from a deck top    (3)
    [30..41]  purchase a face-up card   (3 tiers x 4 slots = 12)
    [42..44]  purchase a reserved card  (3 hand slots)
                                         total = 45
"""
from __future__ import annotations

import itertools

from .cards import CARDS, NOBLES, COLORS, GOLD, ALL_TOKENS
from .state import GameState
from .moves import Move
from . import rules

_TAKE3 = list(itertools.combinations(COLORS, 3))         # 10
_TAKE3_INDEX = {combo: i for i, combo in enumerate(_TAKE3)}
_TAKE2_BASE = 10
_RESERVE_BOARD_BASE = 15
_RESERVE_DECK_BASE = 27
_PURCHASE_BOARD_BASE = 30
_PURCHASE_RESERVE_BASE = 42
ACTION_SPACE = 45


def action_index(move: Move) -> int:
    if move.type == "take3":
        # Late game the bank can offer fewer than 3 colors, so a "take3" may
        # carry 1-2 gems. Pad to a canonical 3-combo bucket (deterministic; a
        # collision with a real 3-take just shares a policy slot, which is fine
        # the same way discard variants already collapse to one index).
        gems = sorted(move.gems, key=COLORS.index)
        if len(gems) < 3:
            for c in COLORS:
                if len(gems) >= 3:
                    break
                if c not in gems:
                    gems.append(c)
            gems = sorted(gems, key=COLORS.index)
        return _TAKE3_INDEX[tuple(gems)]
    if move.type == "take2":
        return _TAKE2_BASE + COLORS.index(move.color)
    if move.type == "reserve":
        if move.source == "deck":
            return _RESERVE_DECK_BASE + (move.tier - 1)
        return _RESERVE_BOARD_BASE + (move.tier - 1) * rules.BOARD_SLOTS_PER_TIER + move.slot
    if move.type == "purchase":
        if move.source == "reserve":
            return _PURCHASE_RESERVE_BASE + move.reserve_index
        return _PURCHASE_BOARD_BASE + (move.tier - 1) * rules.BOARD_SLOTS_PER_TIER + move.slot
    raise ValueError(f"cannot index move type {move.type}")


def feature_vector(state: GameState, perspective: int | None = None) -> list[float]:
    """Flat numeric features for a learned evaluator (perspective-relative)."""
    me = state.to_move if perspective is None else perspective
    order = [me] + [i for i in range(state.num_players) if i != me]
    feats: list[float] = []

    feats += [state.bank[c] / 7.0 for c in ALL_TOKENS]

    for idx in order:
        p = state.players[idx]
        feats += [p.tokens[c] / 7.0 for c in ALL_TOKENS]
        feats += [p.bonuses[c] / 7.0 for c in COLORS]
        feats.append(p.points / rules.WINNING_POINTS)
        feats.append(len(p.reserved) / rules.MAX_RESERVED)

    for tier in (1, 2, 3):
        for cid in state.board[tier]:
            if cid is None:
                feats += [0.0] * (1 + len(COLORS) + 1)
            else:
                c = CARDS[cid]
                feats.append(c.points / 5.0)
                feats += [c.cost.get(col, 0) / 7.0 for col in COLORS]
                feats.append(1.0 if c.bonus else 0.0)

    for nid in NOBLES:
        feats.append(1.0 if nid in state.nobles else 0.0)

    return feats
