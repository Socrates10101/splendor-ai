"""Trainer that *consumes* the append-only store (Phase 1, decoupled learner).

Reads recent positions from the store (POSITIONS budget, not shard count),
regenerates features + value targets via the data contract, trains the
policy-value net, writes a new model + manifest to the store, and (optionally)
appends a promote event so generators pick it up.

    python -m splendor.train_store --store DATA_DIR --steps 200 --promote

This is intentionally minimal (one training pass over a sampled window). The
richer schedule lives in train.py; here the point is the generate/train split
and the store contract.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch
import torch.nn.functional as F

from .nn import PolicyValueNet, pick_device
from .store import git_sha, load_checkpoint_bytes, model_to_bytes, open_store


def _load_window(store, max_positions: int):
    samples = store.recent_samples(max_positions)
    if not samples:
        return None
    feats = np.asarray([s[0] for s in samples], dtype=np.float32)
    pis = np.asarray([s[1] for s in samples], dtype=np.float32)
    zs = np.asarray([s[2] for s in samples], dtype=np.float32)
    return feats, pis, zs


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Train from the append-only store")
    p.add_argument("--store", required=True,
                   help="data store: a local dir or gs://bucket/prefix")
    p.add_argument("--steps", type=int, default=200)
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
    p.add_argument("--seed", type=int, default=12345)
    args = p.parse_args(argv)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed & 0xFFFFFFFF)
    device = pick_device(args.device)
    store = open_store(args.store)

    window = _load_window(store, args.max_positions)
    if window is None:
        print("[train] no data in store; generate some shards first.")
        return
    feats, pis, zs = window
    n = len(feats)
    input_dim = feats.shape[1]

    # build net: warm-start from current model, else fresh
    parent = store.current_model()
    if args.warm_start and parent is not None:
        ckpt = load_checkpoint_bytes(store.get_model_bytes(parent), device=device)
        net = PolicyValueNet.from_config(ckpt["config"]).to(device)
        net.load_state_dict(ckpt["model"])
        print(f"[train] warm-started from {parent}")
    else:
        net = PolicyValueNet(input_dim, width=args.width, blocks=args.blocks).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr,
                           weight_decay=args.weight_decay)

    feats_t = torch.tensor(feats, device=device)
    pis_t = torch.tensor(pis, device=device)
    zs_t = torch.tensor(zs, device=device)

    bs = min(args.batch_size, n)
    net.train()
    t0 = time.time()
    p_sum = v_sum = 0.0
    for step in range(args.steps):
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
    replay_ratio = (args.steps * bs) / max(1, n)  # positions seen / available

    manifest = {
        "parent": parent,
        "git_sha": git_sha(),
        "steps": args.steps, "batch_size": bs,
        "train_positions": n, "replay_ratio": round(replay_ratio, 2),
        "ploss": round(p_sum / args.steps, 4), "vloss": round(v_sum / args.steps, 4),
        "created_ts": time.time(),
    }
    weights = model_to_bytes(net.state_dict(), net.config())
    version = store.put_model(gen, weights, manifest)
    print(f"[train] {version}: positions={n} steps={args.steps} "
          f"ploss={manifest['ploss']} vloss={manifest['vloss']} "
          f"replay_ratio={manifest['replay_ratio']} ({time.time() - t0:.1f}s)")

    if args.promote:
        store.promote(version, {
            "parent_model": parent, "candidate_model": version,
            "logical_generation": gen, "accepted_by": "train_store",
            "created_at": time.time(),
        })
        print(f"[train] promoted {version} (generation {gen}) -> current")


if __name__ == "__main__":
    main()
