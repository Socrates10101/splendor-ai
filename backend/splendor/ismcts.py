"""Information Set Monte Carlo Tree Search (PUCT), AlphaZero-shaped.

The server's ``GameState`` is fully known internally, but the side to move does
*not* really know the deck order or an opponent's deck-reserved cards. Letting
the AI search the true future would be cheating and would make analysis
unrealistic. So each simulation runs on a *determinized* clone where the unseen
cards are reshuffled from the searcher's perspective.

Selection uses PUCT with priors and a leaf value from a pluggable
:class:`~splendor.evaluator.Evaluator` (heuristic now, neural later).
"""
from __future__ import annotations

import math

from .cards import CARDS
from .state import GameState
from .moves import legal_moves, apply_move, Move
from .evaluator import Evaluator, HeuristicEvaluator, win_prob
from .notation import to_notation
from .rng import Rng

C_PUCT = 1.6
MAX_DEPTH = 36


# ---------------------------------------------------------------------------
# Determinization
# ---------------------------------------------------------------------------
def determinize(state: GameState, root: int, rng: Rng) -> GameState:
    """Clone ``state`` with unseen cards reshuffled from ``root``'s view."""
    g = state.clone()
    for tier in (1, 2, 3):
        pool = list(g.decks[tier])
        spots = []  # (player_idx, reserve_idx) of opponents' hidden reserves
        for pi, pl in enumerate(g.players):
            if pi == root:
                continue
            for ri, (cid, hidden) in enumerate(zip(pl.reserved, pl.reserved_hidden)):
                if hidden and CARDS[cid].tier == tier:
                    pool.append(cid)
                    spots.append((pi, ri))
        if not pool:
            continue
        rng.shuffle(pool)
        for pi, ri in spots:
            g.players[pi].reserved[ri] = pool.pop()
        g.decks[tier] = pool
    return g


# ---------------------------------------------------------------------------
# Tree
# ---------------------------------------------------------------------------
class _Node:
    __slots__ = ("to_move", "expanded", "priors", "N", "W")

    def __init__(self, to_move: int):
        self.to_move = to_move
        self.expanded = False
        self.priors: dict = {}
        self.N: dict = {}   # move.key() -> visit count
        self.W: dict = {}   # move.key() -> summed value (root perspective)


def _select(node: _Node, legal: list[Move], root: int) -> tuple[tuple, Move]:
    orient = 1.0 if node.to_move == root else -1.0
    total = sum(node.N.values())
    sqrt_total = math.sqrt(total) + 1e-8
    best_score = -1e18
    best_key = None
    best_move = None
    for m in legal:
        k = m.key()
        n = node.N.get(k, 0)
        q = (node.W[k] / n) if n else 0.0
        p = node.priors.get(k, 1e-3)
        u = C_PUCT * p * sqrt_total / (1 + n)
        score = orient * q + u
        if score > best_score:
            best_score, best_key, best_move = score, k, m
    return best_key, best_move


def _leaf_value(state: GameState, root: int, ev: Evaluator) -> tuple[dict, float]:
    policy, value = ev.evaluate(state)
    v_root = value if state.to_move == root else -value
    return policy, v_root


def search(state: GameState, sims: int, evaluator: Evaluator | None = None,
           seed: int = 1, progress=None, progress_every: int = 64) -> dict:
    """Run ``sims`` IS-MCTS simulations from ``state``.

    Returns a dict with root statistics. ``progress(done, info)`` is called
    periodically for live streaming, where ``info`` mirrors the final summary.
    """
    ev = evaluator or HeuristicEvaluator()
    rng = Rng(seed)
    root_player = state.to_move
    root = _Node(to_move=root_player)
    nodes = {(): root}

    for i in range(sims):
        det = determinize(state, root_player, rng)
        node = root
        path: list[tuple[_Node, tuple]] = []
        depth = 0
        v = 0.0
        while True:
            if det.is_over():
                v = 1.0 if det.winner == root_player else -1.0
                break
            legal = legal_moves(det)
            if not legal or depth >= MAX_DEPTH:
                _, v = _leaf_value(det, root_player, ev)
                break
            if not node.expanded:
                node.priors, v = _leaf_value(det, root_player, ev)
                node.expanded = True
                break
            key, move = _select(node, legal, root_player)
            path.append((node, key))
            apply_move(det, move)
            childpath = tuple(k for _, k in path)
            child = nodes.get(childpath)
            if child is None:
                child = _Node(to_move=det.to_move)
                nodes[childpath] = child
            node = child
            depth += 1

        for nd, key in path:
            nd.N[key] = nd.N.get(key, 0) + 1
            nd.W[key] = nd.W.get(key, 0.0) + v

        if progress and (i + 1) % progress_every == 0:
            progress(i + 1, _summarize(state, root, sims))

    return _summarize(state, root, sims)


def _summarize(state: GameState, root: _Node, sims: int) -> dict:
    # map root edge keys back to the *true* legal moves for reporting
    true_moves = {m.key(): m for m in legal_moves(state)}
    rows = []
    for key, n in root.N.items():
        if n == 0:
            continue
        q = root.W[key] / n
        m = true_moves.get(key)
        if m is None:
            continue
        rows.append({
            "notation": to_notation(m),
            "action": key,
            "visits": n,
            "value": q,
            "winProb": win_prob(q),
        })
    rows.sort(key=lambda r: r["visits"], reverse=True)
    best = rows[0] if rows else None
    # position win prob = value of the principal variation (best line)
    root_win_prob = best["winProb"] if best else 0.5
    return {
        "sims": sum(root.N.values()),
        "requestedSims": sims,
        "bestMove": (None if best is None
                     else {"notation": best["notation"], "value": best["value"],
                           "winProb": best["winProb"]}),
        "winProb": root_win_prob,
        "topMoves": [{k: r[k] for k in ("notation", "visits", "value", "winProb")}
                     for r in rows[:8]],
        "moveStats": {r["notation"]: {"visits": r["visits"], "value": r["value"],
                                      "winProb": r["winProb"]} for r in rows},
    }


def choose_move(state: GameState, sims: int, evaluator: Evaluator | None = None,
                seed: int = 1) -> Move:
    """Pick the AI's move (the most-visited root action) as a true legal Move."""
    summary = search(state, sims, evaluator, seed=seed)
    if summary["bestMove"] is None:
        return legal_moves(state)[0]
    best_key = None
    best_n = -1
    # recover the exact move by matching notation against true legal moves
    target = summary["bestMove"]["notation"]
    for m in legal_moves(state):
        if to_notation(m) == target:
            return m
    return legal_moves(state)[0]
