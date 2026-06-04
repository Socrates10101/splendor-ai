"""Trainer that *consumes* the append-only store (the decoupled learner).

Reads recent positions from the store (POSITIONS budget, not shard count),
regenerates features + value targets via the data contract, trains the
policy-value net, writes a new model + manifest, and (optionally) appends a
promote event so generators pick it up.

One-shot:
    python -m splendor.train_store --warm-start --promote --steps 400

Run forever (continuous learner — net stays in memory, trains on the growing
data, publishes a new generation every --interval seconds and promotes it so
the generators keep upgrading):
    python -m splendor.train_store --loop --interval 300 --steps 400

NOTE: promotion here is currently UNCONDITIONAL (no strength gating yet). A
later step should arena-gate it (promote only if it beats the current best).
"""
from __future__ import annotations

import argparse
import os
import signal
import socket
import time

import numpy as np
import torch
import torch.nn.functional as F

from splendor.engine.encoding import feature_vector
from splendor.nn import PolicyValueNet, pick_device
from splendor.engine.state import GameState
from splendor.store import (git_sha, load_checkpoint_bytes, model_to_bytes, open_store,
                    resolve_store_uri)


def _input_dim() -> int:
    g = GameState.new([{"id": "a"}, {"id": "b"}], seed=1)
    return len(feature_vector(g))


def _load_window(store, max_positions: int):
    samples = store.recent_samples(max_positions)
    if not samples:
        return None
    feats = np.asarray([s[0] for s in samples], dtype=np.float32)
    pis = np.asarray([s[1] for s in samples], dtype=np.float32)
    zs = np.asarray([s[2] for s in samples], dtype=np.float32)
    return feats, pis, zs


def _build_net(store, args, device):
    """Warm-start from the current model (default in --loop), else fresh."""
    parent = store.current_model()
    if (args.warm_start or args.loop) and parent is not None:
        ckpt = load_checkpoint_bytes(store.get_model_bytes(parent), device=device)
        net = PolicyValueNet.from_config(ckpt["config"]).to(device)
        net.load_state_dict(ckpt["model"])
        print(f"[train] warm-started from {parent}")
    else:
        net = PolicyValueNet(_input_dim(), width=args.width,
                             blocks=args.blocks).to(device)
        parent = None
    return net, parent


def _train_and_publish(store, args, device, net, opt, parent: str | None):
    """One training pass over the recent window -> publish (+promote). Returns
    the new version (to chain as the next parent), or None if there is no data."""
    window = _load_window(store, args.max_positions)
    if window is None:
        print("[train] no data in store yet.")
        return None
    feats, pis, zs = window
    n = len(feats)
    feats_t = torch.tensor(feats, device=device)
    pis_t = torch.tensor(pis, device=device)
    zs_t = torch.tensor(zs, device=device)
    bs = min(args.batch_size, n)

    net.train()
    t0 = time.time()
    p_sum = v_sum = 0.0
    for _ in range(args.steps):
        idx = torch.randint(0, n, (bs,), device=device)
        logits, value = net(feats_t[idx])
        logp = F.log_softmax(logits, dim=1)
        policy_loss = -(pis_t[idx] * logp).sum(dim=1).mean()
        value_loss = F.mse_loss(value, zs_t[idx])
        loss = policy_loss + value_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip)
        opt.step()
        p_sum += float(policy_loss.item())
        v_sum += float(value_loss.item())

    gen = store.next_generation()
    # Per-generation training log (the store IS the metrics history): a future
    # training UI / curve.py reads models/<gen>/manifest.json for the loss curve
    # and eval/<gen>.json for the winrate curve.
    manifest = {
        "parent": parent, "git_sha": git_sha(),
        "steps": args.steps, "batch_size": bs, "lr": args.lr,
        "train_positions": n,
        "replay_ratio": round((args.steps * bs) / max(1, n), 2),
        "ploss": round(p_sum / args.steps, 4),
        "vloss": round(v_sum / args.steps, 4),
        "created_ts": time.time(),
    }
    version = store.put_model(gen, model_to_bytes(net.state_dict(), net.config()),
                              manifest)
    print(f"[train] {version}: positions={n} steps={args.steps} "
          f"ploss={manifest['ploss']} vloss={manifest['vloss']} "
          f"replay_ratio={manifest['replay_ratio']} ({time.time() - t0:.1f}s)")

    if args.promote or args.loop:
        store.promote(version, {
            "parent_model": parent, "candidate_model": version,
            "logical_generation": gen, "accepted_by": "train_store",
            "created_at": time.time(),
        })
        print(f"[train] promoted {version} (generation {gen}) -> current")
    return version


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Train from the append-only store")
    p.add_argument("--store", default=None,
                   help="data store: local dir or gs://bucket/prefix (default: $SPLENDOR_STORE)")
    p.add_argument("--steps", type=int, default=400, help="grad steps per cycle")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--max-positions", type=int, default=200_000,
                   help="size of the recent window (positions)")
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--blocks", type=int, default=4)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--warm-start", action="store_true",
                   help="continue from the current model's weights")
    p.add_argument("--promote", action="store_true",
                   help="append a promote event (make it current)")
    p.add_argument("--loop", action="store_true",
                   help="train forever: net persists, publish+promote every --interval")
    p.add_argument("--interval", type=float, default=300,
                   help="seconds between cycles in --loop")
    p.add_argument("--seed", type=int, default=12345)
    args = p.parse_args(argv)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed & 0xFFFFFFFF)
    device = pick_device(args.device)
    store = open_store(resolve_store_uri(args.store))
    worker = f"{socket.gethostname()}-train"

    def beat(state, detail):
        try:
            store.put_heartbeat(worker, {"kind": "train", "state": state,
                "host": socket.gethostname(), "pid": os.getpid(), "detail": detail})
        except Exception:  # noqa: BLE001 - telemetry must never break training
            pass

    net, parent = _build_net(store, args, device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr,
                           weight_decay=args.weight_decay)
    print(f"[train] device={device} params={sum(p.numel() for p in net.parameters()):,} "
          f"{'loop' if args.loop else 'one-shot'}")

    if not args.loop:
        beat("training", f"{args.steps} steps")
        version = _train_and_publish(store, args, device, net, opt, parent)
        beat("stopped", f"published {version}" if version else "no data")
        return

    # continuous learner
    stop = {"v": False}
    signal.signal(signal.SIGINT, lambda *_: stop.update(v=True))
    signal.signal(signal.SIGTERM, lambda *_: stop.update(v=True))
    cycles = 0
    try:
        while not stop["v"]:
            beat("training", f"cycle {cycles + 1} ({args.steps} steps)")
            new = _train_and_publish(store, args, device, net, opt, parent)
            if new:
                parent = new            # chain: next cycle builds on this gen
                cycles += 1
            beat("idle", f"{cycles} gens published, sleeping {int(args.interval)}s")
            slept = 0.0
            while slept < args.interval and not stop["v"]:
                time.sleep(min(2.0, args.interval - slept))
                slept += 2.0
    finally:
        print(f"\n[train] loop stopped after {cycles} generations (last={parent})")
        beat("stopped", f"loop ended; {cycles} gens, last={parent}")


if __name__ == "__main__":
    main()
