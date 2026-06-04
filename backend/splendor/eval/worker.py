"""Evaluator worker (the 検証 process): scores models and logs the results.

Polls the store for models that have no eval record yet, plays each vs the
heuristic baseline (fixed seed = comparable across generations), and writes
``eval/<version>.json``. The learning curve (``splendor.curve``) reads these.

Decoupled from training so it never blocks the learner. Run it on any spare
CPU (the mini-PC or the laptop):

    python -m splendor.evaluate --loop --games 40 --sims 64
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import time

from splendor.eval.arena import play_match
from splendor.engine.evaluator import HeuristicEvaluator
from splendor.store import load_checkpoint_bytes, open_store, resolve_store_uri


def _nn_evaluator(store, version: str, device: str = "cpu"):
    import torch
    from splendor.nn import NNEvaluator, PolicyValueNet
    ckpt = load_checkpoint_bytes(store.get_model_bytes(version), device=device)
    net = PolicyValueNet.from_config(ckpt["config"])
    net.load_state_dict(ckpt["model"])
    return NNEvaluator(net, device=torch.device(device))


def _evaluated_versions(store) -> set[str]:
    out = set()
    for k in store.list_keys("eval/"):
        name = k.split("/")[-1]
        if name.endswith(".json"):
            out.add(name[:-5])
    return out


def evaluate_one(store, version: str, args) -> dict:
    ev = _nn_evaluator(store, version, args.device)
    r = play_match(ev, HeuristicEvaluator(), games=args.games, sims=args.sims,
                   base_seed=args.seed)
    gen = int(version[3:]) if version.startswith("gen") and version[3:].isdigit() else None
    rec = {"version": version, "gen": gen, "vs": "heuristic",
           "winrate": r["a_winrate"], "elo": round(r["elo_diff"], 1),
           "wins": r["a_wins"], "losses": r["b_wins"], "draws": r["draws"],
           "games": args.games, "sims": args.sims, "ts": time.time()}
    store.put_bytes(f"eval/{version}.json", json.dumps(rec).encode())
    return rec


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Evaluator worker (logs eval/<gen>.json)")
    p.add_argument("--store", default=None,
                   help="local dir or gs://bucket/prefix (default: $SPLENDOR_STORE)")
    p.add_argument("--games", type=int, default=40)
    p.add_argument("--sims", type=int, default=64)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=777,
                   help="fixed so winrates are comparable across generations")
    p.add_argument("--every", type=int, default=1,
                   help="evaluate every Nth generation (1 = all)")
    p.add_argument("--loop", action="store_true", help="keep evaluating new models")
    p.add_argument("--interval", type=float, default=60)
    args = p.parse_args(argv)

    store = open_store(resolve_store_uri(args.store))
    worker = f"{socket.gethostname()}-eval"

    def beat(state, detail):
        try:
            store.put_heartbeat(worker, {"kind": "evaluate", "state": state,
                "host": socket.gethostname(), "pid": os.getpid(), "detail": detail})
        except Exception:  # noqa: BLE001
            pass

    stop = {"v": False}
    signal.signal(signal.SIGINT, lambda *_: stop.update(v=True))
    signal.signal(signal.SIGTERM, lambda *_: stop.update(v=True))

    try:
        while not stop["v"]:
            done = _evaluated_versions(store)
            models = store.list_models()
            todo = [m for i, m in enumerate(models)
                    if m not in done and (args.every <= 1 or i % args.every == 0)]
            for ver in todo:
                if stop["v"]:
                    break
                beat("evaluating", ver)
                rec = evaluate_one(store, ver, args)
                print(f"[eval] {ver}: vs heuristic {rec['winrate']:.1%} "
                      f"(Elo {rec['elo']:+.0f}, {rec['wins']}-{rec['losses']}-{rec['draws']})")
            if not args.loop:
                break
            beat("idle", f"{len(done) + len(todo)} evaluated, sleeping {int(args.interval)}s")
            slept = 0.0
            while slept < args.interval and not stop["v"]:
                time.sleep(min(2.0, args.interval - slept))
                slept += 2.0
    finally:
        beat("stopped", "evaluator stopped")


if __name__ == "__main__":
    main()
