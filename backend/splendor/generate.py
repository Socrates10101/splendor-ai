"""Self-play data *generator* (stateless worker).

Phase 1 of the platform design: the generator is decoupled from training. It
reads the current model from the store (or falls back to the HeuristicEvaluator
to bootstrap), plays games, and appends a shard of game records + dense policy
targets. It keeps no state of its own; restart-safe.

Run continuously on a CPU box (the always-on mini-PC):
    python -m splendor.generate --store DATA_DIR --loop --games 16 --sims 64

It uses the per-move IS-MCTS search (``ismcts.search``) so it works uniformly
with the heuristic and with any NN evaluator, and reads visit counts directly.
(Throughput is fine for a continuous CPU generator; the leaf-batched path can be
wired in later if needed.)
"""
from __future__ import annotations

import argparse
import os
import signal
import socket
import time

import numpy as np

from .encoding import ACTION_SPACE, action_index
from .evaluator import HeuristicEvaluator
from .ismcts import search
from .moves import apply_move, find_legal, legal_moves
from .notation import from_notation, to_notation
from .state import GameState
from .store import (SCHEMA_VERSION, load_checkpoint_bytes, make_provenance,
                    open_store, resolve_store_uri)


def generate_game(evaluator, seed: int, sims: int, temp: float,
                  temp_moves: int, max_moves: int) -> dict:
    specs = [{"id": "p0", "type": "ismcts"}, {"id": "p1", "type": "ismcts"}]
    state = GameState.new(specs, seed=seed)
    rng = np.random.default_rng(seed & 0xFFFFFFFF)
    moves: list[str] = []
    targets: list[dict] = []
    ply = 0

    while not state.is_over() and ply < max_moves:
        legal = legal_moves(state)
        if not legal:  # stuck (no pass move) -> capped game, decided by score
            break
        summary = search(state, sims, evaluator, seed=(seed * 131 + ply) & 0xFFFFFFFF)
        stats = summary.get("moveStats", {})

        counts = [0] * ACTION_SPACE
        mask = [0] * ACTION_SPACE
        best_vis: dict[int, int] = {}
        best_notation: dict[int, str] = {}
        for m in legal:
            a = action_index(m)
            mask[a] = 1
            notation = to_notation(m)
            v = int(stats.get(notation, {}).get("visits", 0))
            counts[a] += v
            if v > best_vis.get(a, -1):
                best_vis[a], best_notation[a] = v, notation

        total = sum(counts)
        if total <= 0:
            chosen_notation = to_notation(legal[0])
        else:
            c = np.asarray(counts, dtype=float)
            if ply < temp_moves and temp > 0:
                scaled = c ** (1.0 / temp)
                p = scaled / scaled.sum()
                a = int(rng.choice(ACTION_SPACE, p=p))
                if a not in best_notation:
                    a = int(c.argmax())
            else:
                a = int(c.argmax())
            chosen_notation = best_notation[a]

        targets.append({"ply": ply, "to_move": state.to_move,
                        "action": chosen_notation,
                        "visit_counts": counts, "legal_mask": mask})
        moves.append(chosen_notation)
        mv = find_legal(state, from_notation(state, chosen_notation))
        apply_move(state, mv)
        ply += 1

    result = {
        "natural_winner": state.winner,
        "points": [p.points for p in state.players],
        "cards": [len(p.cards) for p in state.players],
        "plies": ply,
    }
    return {"seed": seed, "players": [{"id": s["id"]} for s in specs],
            "moves": moves, "result": result, "targets": targets}


def _load_evaluator(store, version: str | None):
    """Return (evaluator, version, kind). version None -> heuristic bootstrap.

    Loads weights via the backend (bytes), so it works for local and GCS alike.
    """
    if version is None:
        return HeuristicEvaluator(), None, "heuristic-bootstrap"
    import torch
    # one core per generator process: run MANY processes in parallel instead of
    # letting each process's torch grab all cores (oversubscription thrash).
    torch.set_num_threads(1)
    from .nn import NNEvaluator, PolicyValueNet  # local import: torch optional
    ckpt = load_checkpoint_bytes(store.get_model_bytes(version), device="cpu")
    net = PolicyValueNet.from_config(ckpt["config"])
    net.load_state_dict(ckpt["model"])
    ev = NNEvaluator(net, device=torch.device("cpu"))
    return ev, version, "selfplay"


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Self-play data generator (stateless)")
    p.add_argument("--store", default=None,
                   help="data store: local dir or gs://bucket/prefix (default: $SPLENDOR_STORE)")
    p.add_argument("--games", type=int, default=16, help="games per shard")
    p.add_argument("--sims", type=int, default=64)
    p.add_argument("--temp", type=float, default=1.0)
    p.add_argument("--temp-moves", type=int, default=12)
    p.add_argument("--max-moves", type=int, default=80)
    p.add_argument("--seed", type=int, default=-1, help="base seed (-1 = random)")
    p.add_argument("--worker", type=str, default=os.uname().nodename.split(".")[0])
    p.add_argument("--loop", action="store_true", help="generate forever")
    p.add_argument("--shards", type=int, default=1, help="shards to write (if not --loop)")
    args = p.parse_args(argv)

    store = open_store(resolve_store_uri(args.store))
    base = args.seed if args.seed >= 0 else int.from_bytes(os.urandom(4), "big")
    ev, ver, kind = _load_evaluator(store, store.current_model())
    print(f"[gen] worker={args.worker} model={ver or 'heuristic'} "
          f"sims={args.sims} games/shard={args.games}")

    host, pid = socket.gethostname(), os.getpid()

    def beat(state, detail):
        try:
            store.put_heartbeat(args.worker, {"kind": "generate", "state": state,
                "host": host, "pid": pid, "model_version": ver, "detail": detail})
        except Exception:  # noqa: BLE001 - telemetry must never break generation
            pass

    # SIGTERM (systemd stop / kill) -> raise so the finally below runs
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))

    written = 0
    try:
        while True:
            cur = store.current_model()        # cheap pointer check each shard
            if cur != ver:                      # reload weights only on change
                ev, ver, kind = _load_evaluator(store, cur)
                print(f"[gen] model switched -> {ver or 'heuristic'}")

            beat("generating", f"{args.games} games @sims{args.sims}")
            t0 = time.time()
            games = [generate_game(ev, base + i, args.sims, args.temp,
                                   args.temp_moves, args.max_moves)
                     for i in range(args.games)]
            base += args.games
            prov = make_provenance(
                kind, ver,
                {"sims": args.sims, "temp": args.temp, "temp_moves": args.temp_moves,
                 "max_moves": args.max_moves},
                args.worker)
            shard = {"schema_version": SCHEMA_VERSION, "provenance": prov, "games": games}
            key = store.put_shard(shard, worker=args.worker)
            positions = sum(len(g["targets"]) for g in games)
            written += 1
            dt = time.time() - t0
            beat("idle", f"{len(games)}g/{positions}pos {dt:.1f}s -> {key.split('/')[-1]}")
            print(f"[gen] {key}: {len(games)} games, {positions} positions, "
                  f"{dt:.1f}s, model={ver or 'heuristic'}")

            if not args.loop and written >= args.shards:
                break
    except KeyboardInterrupt:
        print("\n[gen] interrupted")
    finally:
        beat("stopped", f"exited after {written} shards")


if __name__ == "__main__":
    main()
