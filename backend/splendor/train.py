"""AlphaZero-style training loop for the Splendor policy-value network.

One *iteration* = generate a round of batched self-play games, append the
samples to a replay buffer, then take several gradient steps on minibatches
sampled from the buffer. The latest weights are used for the next round of
self-play, so the agent bootstraps on itself.

Checkpointing is built for "stop any time, keep a recent point":

* ``latest.pt``        -- overwritten after every iteration (resume target)
* ``hourly_NNN.pt``    -- a timestamped snapshot every ``--ckpt-interval-min``

Run::

    python -m splendor.train --iters 0          # train until interrupted
    python -m splendor.train --resume            # continue from latest.pt

The loop installs a SIGINT handler so Ctrl-C finishes the current iteration,
writes a checkpoint, and exits cleanly.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import multiprocessing as mp
import os
import signal
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch
import torch.nn.functional as F

from .encoding import feature_vector
from .nn import PolicyValueNet, pick_device
from .selfplay import play_games
from .state import GameState


def _input_dim() -> int:
    g = GameState.new([{"id": "a"}, {"id": "b"}], seed=1)
    return len(feature_vector(g))


def _worker_play(payload):
    """Run a chunk of self-play in a separate process (CPU inference).

    The bottleneck is the single-threaded Python tree walk, so we fan self-play
    out across CPU cores. Each worker rebuilds the net on CPU from the broadcast
    weights; the GPU stays reserved for the training step in the main process.
    """
    cfg, weights, n_games, base_seed, sp = payload
    torch.set_num_threads(1)  # avoid oversubscription across workers
    net = PolicyValueNet.from_config(cfg)
    net.load_state_dict(weights)
    net.eval()
    return play_games(net, torch.device("cpu"), n_games, sp["sims"], base_seed,
                      temp_moves=sp["temp_moves"], max_moves=sp["max_moves"],
                      temp=sp.get("temp", 1.0))


def _format_secs(s: float) -> str:
    h, rem = divmod(int(s), 3600)
    m, sec = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{sec:02d}"


class Trainer:
    def __init__(self, args):
        self.args = args
        self.run_args = vars(args)
        self.device = pick_device(args.device)
        # reproducibility: seed weight init, minibatch sampling, self-play RNG
        torch.manual_seed(args.seed)
        np.random.seed(args.seed & 0xFFFFFFFF)
        self.input_dim = _input_dim()

        # warm-start: continue from an existing checkpoint's weights (fresh
        # optimizer + schedule). The architecture follows the checkpoint, so the
        # loaded weights always fit; --width/--blocks are ignored in this case.
        init_ckpt = None
        width, blocks = args.width, args.blocks
        if args.init_from and os.path.exists(args.init_from):
            init_ckpt = torch.load(args.init_from, map_location=self.device,
                                   weights_only=False)
            cfg = init_ckpt["config"]
            width, blocks = cfg["width"], cfg["blocks"]
        elif args.init_from:
            print(f"[init] WARNING: --init-from {args.init_from} not found; "
                  "starting from random weights")

        self.net = PolicyValueNet(self.input_dim, width=width,
                                  blocks=blocks).to(self.device)
        if init_ckpt is not None:
            self.net.load_state_dict(init_ckpt["model"])
            print(f"[init] warm-started from {args.init_from} "
                  f"(width={width} blocks={blocks})")
        self.opt = torch.optim.Adam(self.net.parameters(), lr=args.lr,
                                    weight_decay=args.weight_decay)
        self.buffer: deque = deque(maxlen=args.buffer)
        self.iteration = 0
        self.samples_seen = 0
        self.elapsed_offset = 0.0  # wall time accumulated in prior runs
        self.hourly_count = 0
        self.best_winrate = -1.0     # best (smoothed) eval winrate -> best.pt
        self.best_iter = 0
        self.eval_ema = None         # EMA of eval winrate (denoised best-gating)
        self._last_eval = None       # most recent raw eval winrate (for sidecars)
        self.cur_lr = args.lr
        self.pool: ProcessPoolExecutor | None = None
        self.metrics_path = os.path.join(args.ckpt_dir, "metrics.jsonl")
        self.status_path = os.path.join(args.ckpt_dir, "status.json")
        self.started_at = time.time()

        os.makedirs(args.ckpt_dir, exist_ok=True)
        if args.resume:
            self._resume()

        self._stop = False
        signal.signal(signal.SIGINT, self._on_sigint)

    # ---- checkpointing -------------------------------------------------
    def _ckpt_payload(self, elapsed: float) -> dict:
        return {
            "model": self.net.state_dict(),
            "optimizer": self.opt.state_dict(),
            "config": self.net.config(),
            "iteration": self.iteration,
            "samples_seen": self.samples_seen,
            "elapsed": self.elapsed_offset + elapsed,
            "hourly_count": self.hourly_count,
            "best_winrate": self.best_winrate,
            "best_iter": self.best_iter,
            "eval_ema": self.eval_ema,
            "run_args": self.run_args,         # how this run was configured
            "created": time.time(),
        }

    def _save(self, path: str, elapsed: float) -> None:
        tmp = path + ".tmp"
        torch.save(self._ckpt_payload(elapsed), tmp)
        os.replace(tmp, path)  # atomic: never leave a half-written latest.pt
        # lightweight sidecar so the UI can list checkpoints (iter / winrate)
        # without loading multi-MB weight files.
        try:
            meta = {
                "iteration": self.iteration,
                "eval_winrate": self._last_eval,
                "eval_winrate_ema": self.eval_ema,
                "best_winrate": self.best_winrate if self.best_winrate >= 0 else None,
                "samples_seen": self.samples_seen,
                "created": time.time(),
            }
            with open(path + ".json", "w") as fh:
                json.dump(meta, fh)
        except OSError:
            pass

    def _keep_top_best(self, elapsed: float) -> None:
        """Snapshot this new peak and keep only the top-K best_* by winrate.

        Self-play trains through dips (it always uses the LATEST net), so a
        "dip then soar" is possible; keeping several near-peak snapshots means a
        transient or noisily-measured peak is never lost to a single best.pt.
        """
        snap = os.path.join(self.args.ckpt_dir, f"best_i{self.iteration:04d}.pt")
        self._save(snap, elapsed)
        cands = []
        for p in glob.glob(os.path.join(self.args.ckpt_dir, "best_i*.pt")):
            wr = -1.0
            try:
                with open(p + ".json") as fh:
                    wr = json.load(fh).get("eval_winrate_ema") or -1.0
            except (OSError, json.JSONDecodeError):
                pass
            cands.append((wr, p))
        cands.sort(reverse=True)
        for _wr, p in cands[self.args.keep_best:]:
            for f in (p, p + ".json"):
                try:
                    os.remove(f)
                except OSError:
                    pass

    def _resume(self) -> None:
        path = os.path.join(self.args.ckpt_dir, "latest.pt")
        if not os.path.exists(path):
            print(f"[resume] no checkpoint at {path}; starting fresh")
            return
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.net.load_state_dict(ckpt["model"])
        self.opt.load_state_dict(ckpt["optimizer"])
        self.iteration = ckpt.get("iteration", 0)
        self.samples_seen = ckpt.get("samples_seen", 0)
        self.elapsed_offset = ckpt.get("elapsed", 0.0)
        self.hourly_count = ckpt.get("hourly_count", 0)
        self.best_winrate = ckpt.get("best_winrate", -1.0)
        self.best_iter = ckpt.get("best_iter", 0)
        self.eval_ema = ckpt.get("eval_ema", None)
        print(f"[resume] iter={self.iteration} samples_seen={self.samples_seen} "
              f"elapsed={_format_secs(self.elapsed_offset)} "
              f"best={self.best_winrate:.1%}@{self.best_iter}")

    def _on_sigint(self, *_):
        if self._stop:
            raise KeyboardInterrupt
        print("\n[signal] finishing current iteration, then checkpointing "
              "(Ctrl-C again to force quit)...")
        self._stop = True

    # ---- learning-rate schedule ---------------------------------------
    def _lr_for_iter(self, it: int) -> float:
        a = self.args
        peak = a.lr
        if a.lr_schedule == "constant":
            return peak
        # linear warmup then cosine decay to peak*lr_final_frac over a horizon
        if it < a.lr_warmup_iters:
            return peak * (it + 1) / max(1, a.lr_warmup_iters)
        horizon = a.lr_decay_iters or (a.iters if a.iters > 0 else 500)
        floor = peak * a.lr_final_frac
        progress = min(1.0, (it - a.lr_warmup_iters) /
                       max(1, horizon - a.lr_warmup_iters))
        return floor + 0.5 * (peak - floor) * (1 + math.cos(math.pi * progress))

    # ---- training step -------------------------------------------------
    def _train_steps(self) -> tuple[float, float]:
        if len(self.buffer) < self.args.batch_size:
            return 0.0, 0.0
        lr = self._lr_for_iter(self.iteration)
        for g in self.opt.param_groups:
            g["lr"] = lr
        self.cur_lr = lr
        self.net.train()
        bs = self.args.batch_size
        idxs_pool = len(self.buffer)
        p_loss_sum = v_loss_sum = 0.0
        for _ in range(self.args.train_steps):
            idx = np.random.randint(0, idxs_pool, size=bs)
            feats = torch.tensor(
                np.stack([self.buffer[i][0] for i in idx]),
                dtype=torch.float32, device=self.device)
            pis = torch.tensor(
                np.stack([self.buffer[i][1] for i in idx]),
                dtype=torch.float32, device=self.device)
            zs = torch.tensor(
                np.array([self.buffer[i][2] for i in idx]),
                dtype=torch.float32, device=self.device)

            logits, value = self.net(feats)
            logp = F.log_softmax(logits, dim=1)
            policy_loss = -(pis * logp).sum(dim=1).mean()
            value_loss = F.mse_loss(value, zs)
            # entropy bonus: (p·logp).sum = -H, so adding +coef·(-H) and
            # minimising it raises the policy's entropy → less mode collapse.
            ent_term = (logp.exp() * logp).sum(dim=1).mean()
            loss = (policy_loss + self.args.value_coef * value_loss
                    + self.args.entropy_coef * ent_term)

            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            if self.args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.net.parameters(),
                                               self.args.grad_clip)
            self.opt.step()
            p_loss_sum += float(policy_loss.item())
            v_loss_sum += float(value_loss.item())
        n = self.args.train_steps
        return p_loss_sum / n, v_loss_sum / n

    # ---- self-play generation (single- or multi-process) --------------
    def _generate(self, seed: int):
        a = self.args
        if a.workers <= 1:
            return play_games(self.net, self.device, a.games_per_iter, a.sims,
                              seed, temp_moves=a.temp_moves, max_moves=a.max_moves,
                              temp=a.temp)

        # split games across workers, each with a disjoint per-game seed range
        sp = {"sims": a.sims, "temp_moves": a.temp_moves, "max_moves": a.max_moves,
              "temp": a.temp}
        weights = {k: v.detach().cpu() for k, v in self.net.state_dict().items()}
        cfg = self.net.config()
        base = a.games_per_iter // a.workers
        rem = a.games_per_iter % a.workers
        payloads, off = [], 0
        for w in range(a.workers):
            n = base + (1 if w < rem else 0)
            if n == 0:
                continue
            payloads.append((cfg, weights, n, seed + off, sp))
            off += n

        results = list(self.pool.map(_worker_play, payloads))
        samples = []
        lengths, batches, maxb, total_games = [], [], 0, 0
        natural = decided = p0_wins = true_draws = 0
        ent_sum = 0.0
        ent_n = 0
        for s, st in results:
            samples.extend(s)
            total_games += st["games"]
            lengths.append(st["avg_game_len"] * st["games"])
            batches.append(st["avg_batch"])
            maxb = max(maxb, st["max_batch"])
            natural += st.get("_natural", 0)
            decided += st.get("_decided", 0)
            p0_wins += st.get("_p0_wins", 0)
            true_draws += st.get("_true_draws", 0)
            ent_sum += st.get("_ent_sum", 0.0)
            ent_n += st.get("_ent_n", 0)
        stats = {
            "games": total_games, "samples": len(samples),
            "avg_game_len": (sum(lengths) / total_games) if total_games else 0.0,
            "avg_batch": (sum(batches) / len(batches)) if batches else 0.0,
            "max_batch": maxb,
            "capped_rate": 1.0 - (natural / total_games) if total_games else 0.0,
            "draw_rate": (true_draws / total_games) if total_games else 0.0,
            "p0_winrate": (p0_wins / decided) if decided else 0.0,
            "policy_entropy": (ent_sum / ent_n) if ent_n else 0.0,
        }
        return samples, stats

    # ---- strength evaluation ------------------------------------------
    def _evaluate(self) -> tuple[float, float]:
        from .arena import play_match, load_evaluator
        from .nn import NNEvaluator
        a = self.args
        candidate = NNEvaluator(self.net, device=self.device)
        opponent = (load_evaluator("heuristic") if a.eval_opponent == "heuristic"
                    else load_evaluator(a.eval_opponent, device=str(self.device)))
        # FIXED seed set every eval: the same starting positions each time, so
        # the winrate curve tracks the net improving, not lucky/unlucky deals.
        res = play_match(candidate, opponent, games=a.eval_games, sims=a.eval_sims,
                         base_seed=(a.seed ^ 0x5151A1))
        return res["a_winrate"], res["elo_diff"]

    def _log_metrics(self, row: dict) -> None:
        with open(self.metrics_path, "a") as fh:
            fh.write(json.dumps(row) + "\n")

    def _write_status(self, state: str, row: dict, sps: float) -> None:
        a = self.args
        now = time.time()
        target_end = (self.started_at + a.hours * 3600) if a.hours else None
        status = {
            "state": state,                       # "running" | "stopped"
            "pid": os.getpid(),
            "started_at": self.started_at,
            "now": now,
            "target_end": target_end,
            "hours": a.hours or None,
            "iteration": self.iteration,
            "samples_seen": self.samples_seen,
            "buffer": len(self.buffer),
            "samples_per_s": round(sps, 1),
            "ploss": row.get("ploss"),
            "vloss": row.get("vloss"),
            "lr": row.get("lr"),
            "capped_rate": row.get("capped_rate"),
            "draw_rate": row.get("draw_rate"),
            "p0_winrate": row.get("p0_winrate"),
            "policy_entropy": row.get("policy_entropy"),
            "eval_winrate": row.get("eval_winrate"),
            "eval_winrate_ema": row.get("eval_winrate_ema"),
            "eval_elo": row.get("eval_elo"),
            "best_winrate": self.best_winrate if self.best_winrate >= 0 else None,
            "best_iter": self.best_iter,
            "workers": a.workers,
            "device": str(self.device),
            "ckpt_dir": os.path.abspath(a.ckpt_dir),
        }
        tmp = self.status_path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(status, fh)
        os.replace(tmp, self.status_path)

    # ---- main loop -----------------------------------------------------
    def run(self) -> None:
        a = self.args
        start = time.time()
        last_ckpt = start
        print(f"[train] device={self.device} input_dim={self.input_dim} "
              f"width={self.net.width} blocks={self.net.blocks} params="
              f"{sum(p.numel() for p in self.net.parameters()):,}")
        print(f"[train] games/iter={a.games_per_iter} sims={a.sims} "
              f"train_steps={a.train_steps} batch={a.batch_size} "
              f"buffer={a.buffer} ckpt_every={a.ckpt_interval_min}min "
              f"workers={a.workers} eval_every={a.eval_every}")

        if a.workers > 1:
            ctx = mp.get_context("spawn")
            self.pool = ProcessPoolExecutor(max_workers=a.workers, mp_context=ctx)

        try:
            self._loop(start, last_ckpt)
        finally:
            if self.pool is not None:
                self.pool.shutdown(wait=True)

    def _time_up(self, start: float) -> bool:
        return bool(self.args.hours) and (time.time() - start) >= self.args.hours * 3600

    def _loop(self, start: float, last_ckpt: float) -> None:
        a = self.args
        target_iters = a.iters  # 0 == run forever (until Ctrl-C / --hours)
        if a.hours:
            import datetime as _dt
            end = _dt.datetime.fromtimestamp(start + a.hours * 3600)
            print(f"[train] running for {a.hours}h, auto-stop ~{end:%Y-%m-%d %H:%M}")
        while (not self._stop and not self._time_up(start)
               and (target_iters == 0 or self.iteration < target_iters)):
            t0 = time.time()
            seed = a.seed + self.iteration * a.games_per_iter
            samples, stats = self._generate(seed)
            self.buffer.extend(samples)
            self.samples_seen += len(samples)
            gen_dt = time.time() - t0

            t1 = time.time()
            p_loss, v_loss = self._train_steps()
            train_dt = time.time() - t1

            self.iteration += 1
            elapsed = time.time() - start
            total_elapsed = self.elapsed_offset + elapsed

            sps = len(samples) / gen_dt if gen_dt else 0.0
            print(
                f"[iter {self.iteration:4d}] "
                f"buf={len(self.buffer):6d} "
                f"games={stats['games']} len~{stats['avg_game_len']:.1f} "
                f"cap={stats.get('capped_rate', 0):.0%} draw={stats.get('draw_rate', 0):.0%} "
                f"H={stats.get('policy_entropy', 0):.2f} p0={stats.get('p0_winrate', 0):.0%} "
                f"gen={gen_dt:5.1f}s({sps:4.0f} smp/s) train={train_dt:4.1f}s "
                f"lr={self.cur_lr:.1e} ploss={p_loss:.3f} vloss={v_loss:.3f} "
                f"up={_format_secs(total_elapsed)}"
            )

            # periodic strength check against a fixed anchor (guarded so a flaky
            # eval can never kill an overnight run)
            winrate = elo = wr_ema = None
            if a.eval_every and self.iteration % a.eval_every == 0:
                t2 = time.time()
                try:
                    winrate, elo = self._evaluate()
                    self._last_eval = winrate
                    # EMA-smooth the (still somewhat noisy) winrate so best.pt is
                    # not crowned by a single lucky match.
                    alpha = a.eval_ema_alpha
                    self.eval_ema = (winrate if self.eval_ema is None
                                     else alpha * winrate + (1 - alpha) * self.eval_ema)
                    wr_ema = self.eval_ema
                    print(f"[eval {self.iteration:4d}] vs {a.eval_opponent}: "
                          f"winrate={winrate:.1%} (ema {wr_ema:.1%}) elo={elo:+.0f} "
                          f"({a.eval_games}g/{a.eval_sims}sims, {time.time()-t2:.0f}s)")
                except Exception as exc:  # noqa: BLE001 - keep training alive
                    print(f"[eval {self.iteration:4d}] skipped (error: {exc})")

            row = {
                "iter": self.iteration, "samples_seen": self.samples_seen,
                "buffer": len(self.buffer), "elapsed": total_elapsed,
                "ploss": p_loss, "vloss": v_loss, "lr": self.cur_lr,
                "gen_s": round(gen_dt, 1), "train_s": round(train_dt, 1),
                "avg_game_len": round(stats["avg_game_len"], 1),
                "capped_rate": round(stats.get("capped_rate", 0.0), 3),
                "draw_rate": round(stats.get("draw_rate", 0.0), 3),
                "p0_winrate": round(stats.get("p0_winrate", 0.0), 3),
                "policy_entropy": round(stats.get("policy_entropy", 0.0), 3),
                "eval_winrate": winrate, "eval_elo": elo,
                "eval_winrate_ema": round(wr_ema, 3) if wr_ema is not None else None,
                "eval_opponent": a.eval_opponent if winrate is not None else None,
            }
            self._log_metrics(row)

            # latest.pt after every iteration -> stopping is always safe
            self._save(os.path.join(a.ckpt_dir, "latest.pt"), elapsed)

            # best.pt: gate on the smoothed winrate, not a single noisy match.
            # Also retain the top-K near-peak snapshots (best_iNNNN.pt).
            if wr_ema is not None and wr_ema > self.best_winrate:
                self.best_winrate = wr_ema
                self.best_iter = self.iteration
                self._save(os.path.join(a.ckpt_dir, "best.pt"), elapsed)
                self._keep_top_best(elapsed)
                print(f"[best] new best (ema {wr_ema:.1%}, raw {winrate:.1%}) "
                      f"-> best.pt (+top-{a.keep_best} snapshot)")

            # hourly snapshot
            if time.time() - last_ckpt >= a.ckpt_interval_min * 60:
                self.hourly_count += 1
                snap = os.path.join(a.ckpt_dir, f"hourly_{self.hourly_count:03d}.pt")
                self._save(snap, elapsed)
                last_ckpt = time.time()
                print(f"[ckpt] wrote {snap}")

            self._write_status("running", row, sps)

        # final checkpoint + status on exit
        self._save(os.path.join(a.ckpt_dir, "latest.pt"), time.time() - start)
        reason = ("time limit" if self._time_up(start)
                  else "interrupted" if self._stop else "target iters")
        self._write_status("stopped", {"ploss": None, "vloss": None}, 0.0)
        best = (f"best={self.best_winrate:.1%}@{self.best_iter}"
                if self.best_winrate >= 0 else "best=n/a (no eval)")
        print(f"[train] stopped ({reason}) at iter={self.iteration}; "
              f"latest.pt saved. {best}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train the Splendor policy-value net")
    # loop size
    p.add_argument("--iters", type=int, default=0,
                   help="number of iterations (0 = run until Ctrl-C / --hours)")
    p.add_argument("--hours", type=float, default=0.0,
                   help="wall-clock budget; auto-stop & checkpoint after N hours "
                        "(0 = no time limit)")
    p.add_argument("--games-per-iter", type=int, default=128,
                   help="self-play games per iteration (also the leaf batch width)")
    p.add_argument("--sims", type=int, default=80,
                   help="IS-MCTS simulations per move")
    p.add_argument("--train-steps", type=int, default=200,
                   help="gradient steps per iteration")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--buffer", type=int, default=200_000,
                   help="replay buffer capacity (samples)")
    # self-play knobs
    p.add_argument("--temp-moves", type=int, default=16,
                   help="opening plies chosen by sampling (then greedy)")
    p.add_argument("--temp", type=float, default=1.0,
                   help="sampling temperature on visit counts (N^(1/temp)); "
                        "higher = more exploration diversity")
    p.add_argument("--max-moves", type=int, default=80,
                   help="hard cap on plies per game")
    # optimisation
    p.add_argument("--lr", type=float, default=1e-3, help="peak learning rate")
    p.add_argument("--lr-schedule", choices=["cosine", "constant"], default="constant",
                   help="constant (proven) | cosine = warmup then decay to lr*final-frac")
    p.add_argument("--lr-warmup-iters", type=int, default=20)
    p.add_argument("--lr-decay-iters", type=int, default=0,
                   help="cosine horizon (0 = use --iters, else 500)")
    p.add_argument("--lr-final-frac", type=float, default=0.1,
                   help="cosine floor as a fraction of peak lr")
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0,
                   help="max gradient norm (0 = disabled); guards against spikes")
    p.add_argument("--value-coef", type=float, default=1.0,
                   help="weight on the value MSE term")
    p.add_argument("--entropy-coef", type=float, default=0.0,
                   help="policy-entropy bonus (opt-in); higher = softer policy")
    # model
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--blocks", type=int, default=4)
    # parallelism: workers run self-play on CPU cores; the GPU does training
    p.add_argument("--workers", type=int, default=1,
                   help="self-play worker processes (>1 fans out across CPU cores)")
    # strength evaluation / progress tracking
    p.add_argument("--eval-every", type=int, default=0,
                   help="run an arena match every N iterations (0 = off)")
    p.add_argument("--eval-games", type=int, default=40,
                   help="games per arena eval (more = less noisy winrate)")
    p.add_argument("--eval-sims", type=int, default=60)
    p.add_argument("--eval-ema-alpha", type=float, default=0.5,
                   help="EMA weight on the latest eval when gating best.pt")
    p.add_argument("--keep-best", type=int, default=5,
                   help="how many near-peak best_iNNNN.pt snapshots to retain")
    p.add_argument("--eval-opponent", type=str, default="heuristic",
                   help='"heuristic" or a checkpoint path to measure progress against')
    # checkpointing / device
    p.add_argument("--ckpt-dir", type=str, default="checkpoints")
    p.add_argument("--ckpt-interval-min", type=float, default=60.0)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--init-from", type=str, default="",
                   help="warm-start from a checkpoint's weights (architecture "
                        "follows the checkpoint; fresh optimizer/schedule)")
    p.add_argument("--device", type=str, default="auto",
                   help="auto | mps | cuda | cpu")
    p.add_argument("--seed", type=int, default=12345)
    return p


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    Trainer(args).run()


if __name__ == "__main__":
    main()
