"""Strength evaluation: play two evaluators head-to-head and report a win rate.

Progress in AlphaZero training is not readable from the loss alone — what
matters is whether the agent actually wins more games. ``play_match`` pits two
:class:`~splendor.evaluator.Evaluator` instances against each other through the
existing IS-MCTS search (:func:`splendor.ismcts.choose_move`), alternating who
moves first to cancel the first-player edge, and returns win/draw/loss counts
plus an Elo difference.

The natural fixed anchor is :class:`~splendor.evaluator.HeuristicEvaluator`
(the v1 AI): "win rate vs heuristic" starts near 50% (or below) for a random
net and should climb as training proceeds.

CLI::

    python -m splendor.arena --a checkpoints/latest.pt --b heuristic --games 40
    python -m splendor.arena --a ckptA.pt --b ckptB.pt --sims 120
    python -m splendor.arena --report checkpoints/metrics.jsonl
"""
from __future__ import annotations

import argparse
import json
import math

from .evaluator import Evaluator, HeuristicEvaluator
from .ismcts import choose_move
from .moves import apply_move, legal_moves
from .state import GameState
from . import rules


def _resolve_winner(state: GameState) -> int | None:
    """Winner index for a finished/truncated game (None == draw)."""
    if state.winner is not None:
        return state.winner
    # truncated or stuck: higher points, then fewer development cards
    best_idx, best_key, tie = None, None, False
    for idx, p in enumerate(state.players):
        key = (p.points, -len(p.cards))
        if best_key is None or key > best_key:
            best_idx, best_key, tie = idx, key, False
        elif key == best_key:
            tie = True
    return None if tie else best_idx


def play_one(eval_a: Evaluator, eval_b: Evaluator, a_first: bool, sims: int,
             seed: int, max_moves: int = 80) -> int | None:
    """Play a single game. Returns 0 if A wins, 1 if B wins, None if draw."""
    state = GameState.new(
        [{"id": "p0", "type": "ismcts"}, {"id": "p1", "type": "ismcts"}], seed=seed)
    # which player index each side controls
    by_player = {0: eval_a, 1: eval_b} if a_first else {0: eval_b, 1: eval_a}
    side_of_player = {0: 0, 1: 1} if a_first else {0: 1, 1: 0}  # player -> A/B

    moves = 0
    while not state.is_over() and moves < max_moves:
        if not legal_moves(state):  # stuck: no pass move in Splendor
            break
        ev = by_player[state.to_move]
        move = choose_move(state, sims, ev, seed=seed * 131 + moves)
        apply_move(state, move)
        moves += 1

    winner_player = _resolve_winner(state)
    if winner_player is None:
        return None
    return side_of_player[winner_player]


def play_match(eval_a: Evaluator, eval_b: Evaluator, games: int = 40,
               sims: int = 100, base_seed: int = 1, max_moves: int = 80,
               progress=None) -> dict:
    a_wins = b_wins = draws = 0
    for i in range(games):
        res = play_one(eval_a, eval_b, a_first=(i % 2 == 0), sims=sims,
                       seed=base_seed + i * 7919, max_moves=max_moves)
        if res == 0:
            a_wins += 1
        elif res == 1:
            b_wins += 1
        else:
            draws += 1
        if progress:
            progress(i + 1, games)

    score = (a_wins + 0.5 * draws) / games if games else 0.0
    return {
        "games": games, "sims": sims,
        "a_wins": a_wins, "b_wins": b_wins, "draws": draws,
        "a_winrate": score,
        "elo_diff": _elo_diff(score),
    }


def _elo_diff(score: float) -> float:
    """Elo of A relative to B from A's score (clamped away from 0/1)."""
    s = min(max(score, 1e-4), 1 - 1e-4)
    return 400.0 * math.log10(s / (1 - s))


def load_evaluator(spec: str, device: str = "auto") -> Evaluator:
    """``spec`` is "heuristic" or a path to a training checkpoint."""
    if spec == "heuristic":
        return HeuristicEvaluator()
    from .nn import NNEvaluator  # local import: torch optional
    return NNEvaluator.from_checkpoint(spec, device=device)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _print_report(path: str) -> None:
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    evals = [r for r in rows if r.get("eval_winrate") is not None]
    print(f"{path}: {len(rows)} iters logged, {len(evals)} evaluations")
    if not evals:
        print("(no eval points yet — run with --eval-every > 0)")
        return
    print(f"{'iter':>6} {'samples':>10} {'vs':>10} {'winrate':>8} {'elo':>7} "
          f"{'ploss':>7} {'vloss':>7}")
    spark = "▁▂▃▄▅▆▇█"
    for r in evals:
        wr = r["eval_winrate"]
        bar = spark[min(len(spark) - 1, int(wr * len(spark)))]
        print(f"{r['iter']:>6} {r.get('samples_seen', 0):>10} "
              f"{r.get('eval_opponent', '?'):>10} {wr:>7.1%}{bar} "
              f"{r.get('eval_elo', 0):>6.0f} {r.get('ploss', 0):>7.3f} "
              f"{r.get('vloss', 0):>7.3f}")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Splendor strength arena / progress report")
    p.add_argument("--a", help='checkpoint path or "heuristic"')
    p.add_argument("--b", default="heuristic", help='checkpoint path or "heuristic"')
    p.add_argument("--games", type=int, default=40)
    p.add_argument("--sims", type=int, default=100)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--device", default="auto")
    p.add_argument("--report", metavar="METRICS_JSONL",
                   help="print a progress table from a metrics.jsonl and exit")
    args = p.parse_args(argv)

    if args.report:
        _print_report(args.report)
        return
    if not args.a:
        p.error("--a is required (a checkpoint path or 'heuristic')")

    ev_a = load_evaluator(args.a, args.device)
    ev_b = load_evaluator(args.b, args.device)

    def prog(done, total):
        print(f"\r  playing {done}/{total}...", end="", flush=True)

    res = play_match(ev_a, ev_b, games=args.games, sims=args.sims,
                     base_seed=args.seed, progress=prog)
    print("\r" + " " * 40 + "\r", end="")
    print(f"A={args.a}  vs  B={args.b}   ({res['games']} games, sims={res['sims']})")
    print(f"  A wins {res['a_wins']}  |  B wins {res['b_wins']}  |  draws {res['draws']}")
    print(f"  A win rate: {res['a_winrate']:.1%}   Elo(A-B): {res['elo_diff']:+.0f}")


if __name__ == "__main__":
    main()
