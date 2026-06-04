"""Position evaluators.

An ``Evaluator`` maps a state to ``(policy, value)``:
    policy: dict[move.key()] -> prior probability over the legal moves
    value:  float in [-1, 1] from the perspective of the side to move
            (+1 = side to move is winning, -1 = losing)

``HeuristicEvaluator`` needs no training and drives the v1 AI. The same
interface will later accept an ``NNEvaluator`` backed by a learned policy-value
network with zero changes to the IS-MCTS search (:mod:`splendor.ismcts`).
"""
from __future__ import annotations

import math

from splendor.engine.cards import CARDS, NOBLES, COLORS
from splendor.engine.state import GameState, PlayerState
from splendor.engine.moves import legal_moves, payment_for, Move
from splendor.engine import rules


class Evaluator:
    def evaluate(self, state: GameState) -> tuple[dict, float]:
        raise NotImplementedError


def _softmax(scores: list[float]) -> list[float]:
    if not scores:
        return []
    hi = max(scores)
    exps = [math.exp(s - hi) for s in scores]
    total = sum(exps) or 1.0
    return [e / total for e in exps]


def win_prob(value: float) -> float:
    """Map an evaluator value in [-1, 1] to a win probability in [0, 1]."""
    return max(0.0, min(1.0, (value + 1.0) / 2.0))


class HeuristicEvaluator(Evaluator):
    # value-function weights (tuned by hand; the search corrects most mistakes).
    # Points dominate because the game is won on points; W_PROX makes a lead
    # near the 15-point goal decisive, so "opponent reached 15" reads as losing.
    W_POINTS = 1.6
    W_PROX = 1.0             # convex term on points (points^2 / goal)
    W_BONUS = 0.30           # card engine still matters, but less than points
    W_HIGH_BONUS = 0.04      # extra for a deep stack of one color
    W_TOKENS = 0.03
    W_GOLD = 0.08
    W_RESERVED = 0.10
    W_NOBLE_PROG = 0.6
    W_AFFORD = 0.18
    VALUE_SCALE = 10.0       # divides the score gap before tanh

    # policy heuristic weights
    P_BUY_BASE = 1.5
    P_BUY_POINT = 1.4
    P_NOBLE = 3.0
    P_TIER_BIAS = 0.3
    P_RESERVE = -0.3
    P_TAKE = 0.4

    # ---- value -------------------------------------------------------------
    def _noble_progress(self, state: GameState, p: PlayerState) -> float:
        best = 0.0
        for nid in state.nobles:
            req = NOBLES[nid].requirement
            have = sum(min(p.bonuses[c], n) for c, n in req.items())
            need = sum(req.values())
            best = max(best, have / need)
        return best

    def _affordable_points(self, state: GameState, p: PlayerState) -> float:
        pts = 0
        for cid in state.board_card_ids():
            if payment_for(p, cid) is not None:
                pts += CARDS[cid].points
        for cid in p.reserved:
            if payment_for(p, cid) is not None:
                pts += CARDS[cid].points
        return pts

    def _strength(self, state: GameState, idx: int) -> float:
        p = state.players[idx]
        s = self.W_POINTS * p.points
        s += self.W_PROX * (p.points * p.points) / rules.WINNING_POINTS
        total_bonus = sum(p.bonuses.values())
        s += self.W_BONUS * total_bonus
        s += self.W_HIGH_BONUS * sum(b * b for b in p.bonuses.values())
        s += self.W_TOKENS * sum(p.tokens[c] for c in COLORS)
        s += self.W_GOLD * p.tokens.get("y", 0)
        s += self.W_RESERVED * len(p.reserved)
        s += self.W_NOBLE_PROG * self._noble_progress(state, p)
        s += self.W_AFFORD * self._affordable_points(state, p)
        return s

    def _value(self, state: GameState) -> float:
        if state.is_over():
            return 1.0 if state.winner == state.to_move else -1.0
        me = state.to_move
        opp = (me + 1) % state.num_players  # 2-player focus
        gap = self._strength(state, me) - self._strength(state, opp)
        return math.tanh(gap / self.VALUE_SCALE)

    # ---- policy ------------------------------------------------------------
    def _move_score(self, state: GameState, p: PlayerState, m: Move) -> float:
        if m.type == "purchase":
            card = CARDS[m.card_id]
            s = self.P_BUY_BASE + self.P_BUY_POINT * card.points
            s += self.P_TIER_BIAS * card.tier
            if m.noble:
                s += self.P_NOBLE
            return s
        if m.type == "reserve":
            card = CARDS[m.card_id]
            return self.P_RESERVE + 0.3 * card.points
        # take3 / take2
        return self.P_TAKE + (0.1 if m.type == "take3" else 0.0)

    def evaluate(self, state: GameState) -> tuple[dict, float]:
        moves = legal_moves(state)
        p = state.current
        scores = [self._move_score(state, p, m) for m in moves]
        probs = _softmax(scores)
        policy = {m.key(): pr for m, pr in zip(moves, probs)}
        return policy, self._value(state)
