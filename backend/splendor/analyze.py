"""Post-game analysis (感想戦): annotate a record with per-ply evaluations.

For each ply we run an IS-MCTS search on the position *before* the move:
  - winProbBefore : position value for the player to move (best-play proxy)
  - bestMove      : the search's preferred move
  - playedWinProb : the win prob of the move actually played
  - delta         : playedWinProb - winProbBefore (negative = worse than best)
  - blunder       : delta is below -threshold
The graph in the UI reads ``winProbP0`` (always player-0 perspective).
"""
from __future__ import annotations

from .notation import states_through
from .evaluator import HeuristicEvaluator, Evaluator
from .ismcts import search

DEFAULT_BLUNDER_THRESHOLD = 0.15


def analyze_record(record: dict, sims: int = 400, evaluator: Evaluator | None = None,
                   blunder_threshold: float = DEFAULT_BLUNDER_THRESHOLD,
                   progress=None) -> dict:
    ev = evaluator or HeuristicEvaluator()
    positions = list(states_through(record))      # (i, state_before), last = final
    moves = record["moves"]

    for i, entry in enumerate(moves):
        state_before = positions[i][1]
        mover = state_before.to_move
        summary = search(state_before, sims, ev, seed=1000 + i)

        wpb = summary["winProb"]                   # mover perspective
        best = summary["bestMove"]
        played_note = entry["notation"]
        played_stat = summary["moveStats"].get(played_note)
        played_wp = played_stat["winProb"] if played_stat else wpb
        delta = played_wp - wpb

        entry["eval"] = {
            "winProbBefore": round(wpb, 4),
            "winProbP0": round(wpb if mover == 0 else 1.0 - wpb, 4),
            "bestMove": best,
            "playedWinProb": round(played_wp, 4),
            "delta": round(delta, 4),
            "blunder": delta <= -blunder_threshold,
            "topMoves": summary["topMoves"],
            "sims": summary["sims"],
        }
        if progress:
            progress(i + 1, len(moves), entry["eval"])

    return record
