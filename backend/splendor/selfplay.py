"""Batched, parallel self-play to generate AlphaZero training data.

Each game owns an IS-MCTS tree (the same determinize + PUCT search as
:mod:`splendor.ismcts`, restructured so a single simulation can pause at its
leaf). Many games are advanced in lockstep: every game contributes one leaf per
round, all leaves are evaluated by the network in **one batched forward pass**,
and the values are scattered back. This keeps the GPU fed even though the tree
walk itself is pure-Python and CPU-bound.

A finished game yields a list of training samples::

    (features, pi, player)

where ``features`` is the perspective-relative input vector for the position,
``pi`` is the MCTS visit distribution over the 45 canonical actions (the policy
target), and ``player`` is the side to move (used to assign the final outcome
``z`` once the game ends).
"""
from __future__ import annotations

import math

import numpy as np
import torch

from .encoding import ACTION_SPACE, action_index, feature_vector
from .ismcts import C_PUCT, MAX_DEPTH, _Node, determinize
from .moves import apply_move, legal_moves
from .rng import Rng
from .state import GameState


# ---------------------------------------------------------------------------
# A single self-play game with a pausable IS-MCTS search
# ---------------------------------------------------------------------------
class _Game:
    def __init__(self, seed: int, sims: int, temp_moves: int, max_moves: int,
                 dirichlet_alpha: float, dirichlet_frac: float, temp: float = 1.0):
        self.state = GameState.new(
            [{"id": "p0", "type": "ismcts"}, {"id": "p1", "type": "ismcts"}],
            seed=seed,
        )
        self.rng = Rng(seed ^ 0x5DEECE66D)
        self.np_rng = np.random.default_rng(seed & 0xFFFFFFFF)
        self.sims = sims
        self.temp_moves = temp_moves
        self.temp = temp
        self.max_moves = max_moves
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_frac = dirichlet_frac

        self.move_number = 0
        self.samples: list[tuple] = []
        self.done = self._is_terminal()
        if not self.done:
            self._start_search()

    def _is_terminal(self) -> bool:
        # Splendor has no pass move, so a position with no legal moves (e.g. a
        # full reserve, an empty bank, nothing affordable) is a dead end. Treat
        # it as game over (a draw if no one has reached the goal).
        return self.state.is_over() or not legal_moves(self.state)

    # ---- per-move MCTS lifecycle --------------------------------------
    def _start_search(self) -> None:
        self.root_player = self.state.to_move
        self.root = _Node(to_move=self.root_player)
        self.nodes = {(): self.root}
        self.sims_done = 0

    def search_complete(self) -> bool:
        return self.sims_done >= self.sims

    # ---- one simulation, paused at the leaf ---------------------------
    def select_to_leaf(self):
        """Descend to a leaf. Returns one of:

        ("value", v, None)        -- terminal leaf, ready to back up
        ("expand", det, path)     -- unexpanded node, needs (priors, value)
        ("leaf",   det, path)     -- depth-capped node, needs value only
        """
        det = determinize(self.state, self.root_player, self.rng)
        node = self.root
        path: list[tuple[_Node, tuple]] = []
        depth = 0
        while True:
            if det.is_over():
                v = 1.0 if det.winner == self.root_player else -1.0
                return ("value", v, path)
            legal = legal_moves(det)
            if not legal or depth >= MAX_DEPTH:
                return ("leaf", det, path)
            if not node.expanded:
                return ("expand", det, path)
            key, move = _select_edge(node, legal, self.root_player)
            path.append((node, key))
            apply_move(det, move)
            childpath = tuple(k for _, k in path)
            child = self.nodes.get(childpath)
            if child is None:
                child = _Node(to_move=det.to_move)
                self.nodes[childpath] = child
            node = child
            depth += 1

    def expand(self, det: GameState, path, priors: dict, value: float) -> None:
        node = self._node_at(path)
        node.priors = priors
        node.expanded = True
        if node is self.root:
            self._add_dirichlet_noise(node, det)
        self._backup(path, _orient(value, det.to_move, self.root_player))

    def add_value(self, det: GameState, path, value: float) -> None:
        self._backup(path, _orient(value, det.to_move, self.root_player))

    def add_terminal(self, path, v_root: float) -> None:
        self._backup(path, v_root)

    def _node_at(self, path) -> _Node:
        if not path:
            return self.root
        return self.nodes[tuple(k for _, k in path)]

    def _backup(self, path, v_root: float) -> None:
        for nd, key in path:
            nd.N[key] = nd.N.get(key, 0) + 1
            nd.W[key] = nd.W.get(key, 0.0) + v_root
        self.sims_done += 1

    def _add_dirichlet_noise(self, node: _Node, det: GameState) -> None:
        keys = list(node.priors.keys())
        if not keys:
            return
        noise = self.np_rng.dirichlet([self.dirichlet_alpha] * len(keys))
        frac = self.dirichlet_frac
        for k, n in zip(keys, noise):
            node.priors[k] = (1 - frac) * node.priors[k] + frac * float(n)

    # ---- commit a move once the search budget is spent ----------------
    def play_move(self) -> None:
        legal = legal_moves(self.state)
        if not legal:  # stuck position; end the game without a sample
            self.done = True
            return
        moves = {m.key(): m for m in legal}
        counts = np.zeros(ACTION_SPACE, dtype=np.float64)
        best_key_for_action: dict[int, tuple] = {}
        best_n_for_action: dict[int, int] = {}
        for key, n in self.root.N.items():
            m = moves.get(key)
            if m is None or n <= 0:
                continue
            a = action_index(m)
            counts[a] += n
            if n > best_n_for_action.get(a, -1):
                best_n_for_action[a] = n
                best_key_for_action[a] = key

        total = counts.sum()
        if total <= 0:
            # search produced nothing usable (e.g. all sims terminal); play a
            # legal move and skip storing a degenerate target.
            any_move = next(iter(moves.values()))
            apply_move(self.state, any_move)
            self._after_move()
            return

        pi = counts / total  # policy TARGET is always the raw visit distribution
        self.samples.append((
            feature_vector(self.state), pi.astype(np.float32), self.state.to_move,
        ))

        # action SELECTION uses a temperature on the visit counts: sample
        # N^(1/τ) for the opening (τ>0), then play greedily. Higher τ = more
        # exploration diversity in self-play (counters mode collapse).
        if self.move_number < self.temp_moves and self.temp > 0:
            scaled = counts ** (1.0 / self.temp)
            s = scaled.sum()
            sel_p = (scaled / s) if s > 0 else pi
            action = int(self.np_rng.choice(ACTION_SPACE, p=sel_p))
            if action not in best_key_for_action:
                action = int(counts.argmax())
        else:
            action = int(counts.argmax())

        chosen = moves[best_key_for_action[action]]
        apply_move(self.state, chosen)
        self._after_move()

    def _after_move(self) -> None:
        self.move_number += 1
        if self._is_terminal() or self.move_number >= self.max_moves:
            self.done = True
        else:
            self._start_search()

    # ---- outcome ------------------------------------------------------
    def resolved_winner(self) -> tuple[int | None, bool]:
        """Return ``(winner_index_or_None, natural)``.

        ``natural`` is True when a player actually reached the points goal.
        Otherwise (the game hit the move cap or got stuck) the result is
        resolved by score — higher points, then fewer development cards — so a
        truncated game still teaches "make progress" instead of collapsing to a
        z=0 draw. ``None`` means a genuine tie.
        """
        if self.state.winner is not None:
            return self.state.winner, True
        best_idx, best_key, tie = None, None, False
        for idx, p in enumerate(self.state.players):
            key = (p.points, -len(p.cards))
            if best_key is None or key > best_key:
                best_idx, best_key, tie = idx, key, False
            elif key == best_key:
                tie = True
        return (None if tie else best_idx), False

    # ---- finalise: assign z to every stored sample --------------------
    def finalize(self) -> list[tuple]:
        winner, _natural = self.resolved_winner()
        out = []
        for feat, pi, player in self.samples:
            if winner is None:
                z = 0.0
            elif winner == player:
                z = 1.0
            else:
                z = -1.0
            out.append((feat, pi, z))
        return out


def _orient(value: float, to_move: int, root_player: int) -> float:
    return value if to_move == root_player else -value


def _select_edge(node: _Node, legal, root: int):
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


# ---------------------------------------------------------------------------
# Network-side priors from raw logits over the legal moves at a node
# ---------------------------------------------------------------------------
def _priors_from_logits(logits_row: np.ndarray, det: GameState) -> dict:
    moves = legal_moves(det)
    if not moves:
        return {}
    idx = [action_index(m) for m in moves]
    sel = logits_row[idx]
    sel = sel - sel.max()
    exps = np.exp(sel)
    probs = exps / (exps.sum() or 1.0)
    share: dict[int, int] = {}
    for a in idx:
        share[a] = share.get(a, 0) + 1
    return {m.key(): float(pr) / share[a] for m, a, pr in zip(moves, idx, probs)}


# ---------------------------------------------------------------------------
# Driver: run N games in parallel, batching all leaf evaluations
# ---------------------------------------------------------------------------
@torch.no_grad()
def play_games(net, device, n_games: int, sims: int, base_seed: int,
               temp_moves: int = 12, max_moves: int = 80,
               dirichlet_alpha: float = 0.3, dirichlet_frac: float = 0.25,
               temp: float = 1.0, progress=None) -> tuple[list[tuple], dict]:
    """Self-play ``n_games`` games concurrently and return all training samples.

    Returns ``(samples, stats)`` where ``stats`` reports game lengths and the
    average network batch size (a proxy for how well the GPU was fed).
    """
    net.eval()
    games = [
        _Game(base_seed + i, sims, temp_moves, max_moves,
              dirichlet_alpha, dirichlet_frac, temp=temp)
        for i in range(n_games)
    ]

    batch_sizes: list[int] = []
    while True:
        active = [g for g in games if not g.done]
        if not active:
            break

        # spend the full simulation budget for the current move of every game,
        # one leaf per game per round, evaluated as a single batch.
        while any(not g.search_complete() for g in active):
            requests = []  # (game, kind, det, path)
            for g in active:
                if g.search_complete():
                    continue
                kind, payload, path = g.select_to_leaf()
                if kind == "value":
                    g.add_terminal(path, payload)
                else:
                    requests.append((g, kind, payload, path))

            if not requests:
                continue

            feats = torch.tensor(
                np.stack([feature_vector(det) for _, _, det, _ in requests]),
                dtype=torch.float32, device=device,
            )
            logits, values = net(feats)
            logits = logits.detach().cpu().numpy()
            values = values.detach().cpu().numpy()
            batch_sizes.append(len(requests))

            for row, (g, kind, det, path) in enumerate(requests):
                v = float(values[row])
                if kind == "expand":
                    priors = _priors_from_logits(logits[row], det)
                    g.expand(det, path, priors, v)
                else:  # depth-capped leaf
                    g.add_value(det, path, v)

        # every active game has finished searching this move -> commit it
        for g in active:
            g.play_move()

        if progress:
            progress(sum(1 for g in games if g.done), len(games))

    samples: list[tuple] = []
    lengths = []
    natural = decided = p0_wins = true_draws = 0
    ent_sum = 0.0
    ent_n = 0
    for g in games:
        gs = g.finalize()
        samples.extend(gs)
        lengths.append(g.move_number)
        w, was_natural = g.resolved_winner()
        if was_natural:
            natural += 1
        if w is None:
            true_draws += 1
        else:
            decided += 1
            if w == 0:
                p0_wins += 1
        for _f, pi, _z in gs:  # policy-target entropy = exploration diversity
            p = pi[pi > 0]
            if p.size:
                ent_sum += float(-(p * np.log(p)).sum())
                ent_n += 1

    stats = {
        "games": n_games,
        "samples": len(samples),
        "avg_game_len": (sum(lengths) / len(lengths)) if lengths else 0.0,
        "avg_batch": (sum(batch_sizes) / len(batch_sizes)) if batch_sizes else 0.0,
        "max_batch": max(batch_sizes) if batch_sizes else 0,
        # B9 diagnostics (rates for direct logging) ...
        "capped_rate": 1.0 - (natural / n_games) if n_games else 0.0,
        "draw_rate": (true_draws / n_games) if n_games else 0.0,
        "p0_winrate": (p0_wins / decided) if decided else 0.0,
        "policy_entropy": (ent_sum / ent_n) if ent_n else 0.0,
        # ... and raw counts so multiprocess chunks can be combined exactly
        "_natural": natural, "_decided": decided, "_p0_wins": p0_wins,
        "_true_draws": true_draws, "_ent_sum": ent_sum, "_ent_n": ent_n,
    }
    return samples, stats
