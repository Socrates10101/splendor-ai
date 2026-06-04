"""Policy-value network and the learned :class:`NNEvaluator`.

The network maps a perspective-relative :func:`splendor.encoding.feature_vector`
to a policy over the 45-action canonical space (:data:`encoding.ACTION_SPACE`)
and a scalar value in ``[-1, 1]`` from the side-to-move's perspective.

``NNEvaluator`` adapts that network to the :class:`splendor.evaluator.Evaluator`
interface, so a trained checkpoint drops straight into the existing IS-MCTS
search and the post-game analysis with no other changes.

torch is an optional dependency: importing this module without torch raises a
clear error, but the rest of the engine (heuristic AI, server, tests) keeps
working.
"""
from __future__ import annotations

from collections import defaultdict

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ModuleNotFoundError as exc:  # pragma: no cover - guard for torch-less envs
    raise ModuleNotFoundError(
        "splendor.nn requires PyTorch. Install it with `pip install torch`."
    ) from exc

from .encoding import ACTION_SPACE, action_index, feature_vector
from .evaluator import Evaluator
from .moves import legal_moves
from .state import GameState


def pick_device(name: str | None = None) -> "torch.device":
    """Resolve the best available device (mps on Apple Silicon, else cpu)."""
    if name and name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class _ResidualBlock(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.fc1 = nn.Linear(width, width)
        self.fc2 = nn.Linear(width, width)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = self.fc2(h)
        return F.relu(x + h)


class PolicyValueNet(nn.Module):
    """Small fully-connected policy-value network.

    Width and depth are the main knobs for trading speed against capacity (and
    for pushing GPU utilisation higher when leaf batches are large).
    """

    def __init__(self, input_dim: int, width: int = 256, blocks: int = 4,
                 actions: int = ACTION_SPACE):
        super().__init__()
        self.input_dim = input_dim
        self.width = width
        self.blocks = blocks
        self.actions = actions

        self.stem = nn.Linear(input_dim, width)
        self.body = nn.ModuleList(_ResidualBlock(width) for _ in range(blocks))
        self.policy_head = nn.Linear(width, actions)
        self.value_head = nn.Linear(width, 1)

    def forward(self, x):
        h = F.relu(self.stem(x))
        for block in self.body:
            h = block(h)
        policy_logits = self.policy_head(h)
        value = torch.tanh(self.value_head(h)).squeeze(-1)
        return policy_logits, value

    # ---- (de)serialisation helpers ------------------------------------
    def config(self) -> dict:
        return {"input_dim": self.input_dim, "width": self.width,
                "blocks": self.blocks, "actions": self.actions}

    @classmethod
    def from_config(cls, cfg: dict) -> "PolicyValueNet":
        return cls(cfg["input_dim"], cfg["width"], cfg["blocks"], cfg["actions"])


def load_checkpoint(path: str, device: "torch.device | str | None" = None):
    """Load a checkpoint saved by :mod:`splendor.train`. Returns (net, ckpt)."""
    dev = pick_device(device if isinstance(device, str) else None) \
        if not isinstance(device, torch.device) else device
    ckpt = torch.load(path, map_location=dev, weights_only=False)
    net = PolicyValueNet.from_config(ckpt["config"])
    net.load_state_dict(ckpt["model"])
    net.to(dev)
    net.eval()
    return net, ckpt


class NNEvaluator(Evaluator):
    """:class:`Evaluator` backed by a trained :class:`PolicyValueNet`.

    Priors are produced over the *legal* moves only: net logits are gathered at
    each legal move's canonical action index, softmaxed, then split evenly among
    distinct moves that collapse to the same index (e.g. take-3 variants that
    differ only by which tokens are discarded).
    """

    def __init__(self, net: PolicyValueNet, device: "torch.device | None" = None):
        self.net = net
        self.device = device or next(net.parameters()).device
        self.net.to(self.device)
        self.net.eval()

    @classmethod
    def from_checkpoint(cls, path: str, device: str | None = None) -> "NNEvaluator":
        net, _ = load_checkpoint(path, device)
        return cls(net)

    @torch.no_grad()
    def evaluate(self, state: GameState) -> tuple[dict, float]:
        moves = legal_moves(state)
        feat = torch.tensor([feature_vector(state)], dtype=torch.float32,
                            device=self.device)
        logits, value = self.net(feat)
        logits = logits[0]
        v = float(value.item())

        if not moves:
            return {}, v

        # gather logits at the legal action indices, then softmax over them
        idx = [action_index(m) for m in moves]
        sel = logits[torch.tensor(idx, dtype=torch.long, device=self.device)]
        probs = torch.softmax(sel, dim=0).tolist()

        # split each action index's probability across moves sharing it
        share = defaultdict(int)
        for a in idx:
            share[a] += 1
        policy = {}
        for m, a, pr in zip(moves, idx, probs):
            policy[m.key()] = pr / share[a]
        return policy, v
