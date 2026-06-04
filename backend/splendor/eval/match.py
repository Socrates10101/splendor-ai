"""Evaluate trained models head-to-head (arena) to check training results.

Specs accepted for --a/--b/--vs:
  heuristic        the v1 hand-tuned AI (fixed baseline)
  current          the store's current (latest promoted) model
  genNNNN          a specific store generation
  path/to/x.pt     a local checkpoint file

Examples:
  # is the latest model better than the v1 AI?
  python -m splendor.eval --a current --b heuristic --games 40 --sims 96
  # did training improve over where it started?
  python -m splendor.eval --a current --b gen0002 --games 40
  # strength progression across generations vs the heuristic baseline:
  python -m splendor.eval --sweep --every 5 --vs heuristic --games 24 --sims 48
  python -m splendor.eval --sweep --versions gen0002,gen0010,current --vs heuristic
"""
from __future__ import annotations

import argparse
import os
import sys

from splendor.eval.arena import play_match
from splendor.engine.evaluator import HeuristicEvaluator
from splendor.store import load_checkpoint_bytes, open_store, resolve_store_uri


def resolve_evaluator(store, spec: str, device: str = "cpu"):
    """(evaluator, label) from a spec: heuristic | current | genNNNN | *.pt."""
    if spec == "heuristic":
        return HeuristicEvaluator(), "heuristic"
    if spec == "current":
        cur = store.current_model()
        if cur is None:
            return HeuristicEvaluator(), "heuristic(no model)"
        spec = cur
    import torch
    from splendor.nn import NNEvaluator, PolicyValueNet
    if spec.endswith(".pt"):
        ckpt = torch.load(spec, map_location=device, weights_only=False)
    else:
        ckpt = load_checkpoint_bytes(store.get_model_bytes(spec), device=device)
    net = PolicyValueNet.from_config(ckpt["config"])
    net.load_state_dict(ckpt["model"])
    return NNEvaluator(net, device=torch.device(device)), spec


def _wr_cell(wr: float, color: bool, width: int = 7) -> str:
    s = f"{wr:.1%}".rjust(width)
    if not color:
        return s
    c = "\033[32m" if wr > 0.52 else "\033[31m" if wr < 0.48 else "\033[33m"
    return f"{c}{s}\033[0m"


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Evaluate models head-to-head (arena)")
    p.add_argument("--store", default=None,
                   help="local dir or gs://bucket/prefix (default: $SPLENDOR_STORE)")
    p.add_argument("--a", help="A spec: heuristic | current | genNNNN | *.pt")
    p.add_argument("--b", default="heuristic", help="B spec (default: heuristic)")
    p.add_argument("--games", type=int, default=40)
    p.add_argument("--sims", type=int, default=96)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=777,
                   help="fixed so comparisons use the same starting positions")
    p.add_argument("--sweep", action="store_true",
                   help="evaluate several generations vs a fixed opponent (--vs)")
    p.add_argument("--versions", default=None, help="comma list of specs for --sweep")
    p.add_argument("--every", type=int, default=1, help="sweep every Nth generation")
    p.add_argument("--vs", default="heuristic", help="sweep opponent")
    p.add_argument("--no-color", action="store_true")
    args = p.parse_args(argv)

    color = (not args.no_color and sys.stdout.isatty()
             and os.environ.get("NO_COLOR") is None)
    store = open_store(resolve_store_uri(args.store))

    def prog(done, total):
        print(f"\r  対戦中 {done}/{total} …", end="", flush=True)

    def clear():
        print("\r" + " " * 40 + "\r", end="")

    if args.sweep:
        if args.versions:
            versions = [v.strip() for v in args.versions.split(",")]
        else:
            allm = store.list_models()
            versions = allm[:: max(1, args.every)]
            cur = store.current_model()
            if cur and cur not in versions:
                versions.append(cur)
        opp, opp_label = resolve_evaluator(store, args.vs, args.device)
        print(f"強さ推移  vs {opp_label}   ({args.games}局 / sims{args.sims}, 固定seed)")
        print(f"  {'世代':<10} {'勝率':>7}  {'Elo':>6}  W-L-D")
        for v in versions:
            ev, label = resolve_evaluator(store, v, args.device)
            r = play_match(ev, opp, games=args.games, sims=args.sims,
                           base_seed=args.seed, progress=prog)
            clear()
            print(f"  {label:<10} {_wr_cell(r['a_winrate'], color)}  "
                  f"{r['elo_diff']:>+6.0f}  {r['a_wins']}-{r['b_wins']}-{r['draws']}")
        return

    if not args.a:
        p.error("--a is required (or use --sweep)")
    ea, la = resolve_evaluator(store, args.a, args.device)
    eb, lb = resolve_evaluator(store, args.b, args.device)
    r = play_match(ea, eb, games=args.games, sims=args.sims,
                   base_seed=args.seed, progress=prog)
    clear()
    wr = r["a_winrate"]
    verdict = "→ A 優勢" if wr > 0.5 else "→ B 優勢" if wr < 0.5 else "→ 互角"
    print(f"A = {la}   vs   B = {lb}    ({r['games']}局 / sims{r['sims']})")
    print(f"  A {r['a_wins']}勝 / B {r['b_wins']}勝 / 分 {r['draws']}")
    print(f"  A 勝率 {_wr_cell(wr, color).strip()}   Elo(A-B) {r['elo_diff']:+.0f}   {verdict}")


if __name__ == "__main__":
    main()
