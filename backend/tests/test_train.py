"""Smoke tests for the self-play + training pipeline (skipped without torch)."""
import os

import pytest

torch = pytest.importorskip("torch")

from splendor.encoding import ACTION_SPACE, feature_vector
from splendor.nn import NNEvaluator, PolicyValueNet, pick_device
from splendor.selfplay import play_games
from splendor.state import GameState


def _input_dim():
    g = GameState.new([{"id": "a"}, {"id": "b"}], seed=1)
    return len(feature_vector(g))


def test_net_shapes():
    dim = _input_dim()
    net = PolicyValueNet(dim, width=32, blocks=2)
    x = torch.zeros((4, dim))
    logits, value = net(x)
    assert logits.shape == (4, ACTION_SPACE)
    assert value.shape == (4,)
    assert torch.all(value.abs() <= 1.0)


def test_nn_evaluator_matches_evaluator_contract():
    dim = _input_dim()
    net = PolicyValueNet(dim, width=32, blocks=2)
    ev = NNEvaluator(net, device=torch.device("cpu"))
    g = GameState.new([{"id": "a"}, {"id": "b"}], seed=7)
    policy, value = ev.evaluate(g)
    assert -1.0 <= value <= 1.0
    assert policy, "expected priors over the opening's legal moves"
    assert abs(sum(policy.values()) - 1.0) < 1e-5


def test_self_play_produces_consistent_samples():
    dim = _input_dim()
    net = PolicyValueNet(dim, width=32, blocks=2)
    samples, stats = play_games(net, torch.device("cpu"), n_games=4, sims=12,
                                base_seed=1, max_moves=30)
    assert samples, "self-play should yield training samples"
    feat, pi, z = samples[0]
    assert len(feat) == dim
    assert pi.shape == (ACTION_SPACE,)
    assert abs(float(pi.sum()) - 1.0) < 1e-5
    assert z in (-1.0, 0.0, 1.0)
    assert stats["avg_batch"] > 0
    # B9 diagnostics present and in range
    for k in ("capped_rate", "draw_rate", "p0_winrate", "policy_entropy"):
        assert k in stats
    assert 0.0 <= stats["capped_rate"] <= 1.0
    assert stats["policy_entropy"] >= 0.0


def test_capped_game_resolved_by_points_not_draw():
    # B1: a game with no natural winner is decided by score, not z=0.
    from splendor.selfplay import _Game
    g = _Game(seed=1, sims=4, temp_moves=0, max_moves=80,
              dirichlet_alpha=0.3, dirichlet_frac=0.25)
    g.state.winner = None                 # cap hit, nobody reached the goal
    g.state.players[0].points = 9
    g.state.players[1].points = 4
    winner, natural = g.resolved_winner()
    assert winner == 0 and natural is False
    # the stored samples now carry a real ±1 signal instead of all-zero z
    g.samples = [("f", None, 0), ("f", None, 1)]
    g.finalize_z = [z for _f, _pi, z in g.finalize()]
    assert g.finalize_z == [1.0, -1.0]

    # a true tie stays a draw
    g.state.players[1].points = 9
    g.state.players[0].cards = [1]        # equal points, p1 has fewer cards
    g.state.players[1].cards = []
    w2, _ = g.resolved_winner()
    assert w2 == 1


def test_arena_match_reports_winrate():
    from splendor.arena import play_match
    from splendor.evaluator import HeuristicEvaluator
    res = play_match(HeuristicEvaluator(), HeuristicEvaluator(), games=4,
                     sims=16, max_moves=40)
    assert res["games"] == 4
    assert res["a_wins"] + res["b_wins"] + res["draws"] == 4
    assert 0.0 <= res["a_winrate"] <= 1.0


def test_lr_schedule_warmup_then_decay():
    from splendor.train import build_parser, Trainer
    args = build_parser().parse_args([
        "--iters", "100", "--lr", "1e-3", "--lr-schedule", "cosine",
        "--lr-warmup-iters", "10", "--lr-final-frac", "0.1",
        "--device", "cpu", "--width", "16", "--blocks", "1",
    ])
    t = Trainer(args)
    assert t._lr_for_iter(0) < t._lr_for_iter(9)        # warming up
    assert abs(t._lr_for_iter(10) - 1e-3) < 1e-9        # peak at end of warmup
    assert t._lr_for_iter(99) < t._lr_for_iter(10)      # decayed below peak
    assert t._lr_for_iter(99) >= 1e-3 * 0.1 - 1e-9      # not below the floor


def test_init_from_warm_starts_with_checkpoint_architecture(tmp_path):
    # save a tiny net, then warm-start a Trainer from it with DIFFERENT --width
    from splendor.train import build_parser, Trainer
    ckpt = PolicyValueNet(_input_dim(), width=24, blocks=1)
    path = tmp_path / "seed.pt"
    torch.save({"model": ckpt.state_dict(), "config": ckpt.config()}, str(path))

    args = build_parser().parse_args([
        "--iters", "0", "--device", "cpu", "--width", "99", "--blocks", "9",
        "--init-from", str(path), "--ckpt-dir", str(tmp_path / "ck"),
    ])
    t = Trainer(args)
    # architecture follows the checkpoint (24/1), not the args (99/9)
    assert t.net.width == 24 and t.net.blocks == 1
    # weights actually copied
    assert torch.equal(t.net.stem.weight, ckpt.stem.weight)


def test_train_loop_runs_and_checkpoints(tmp_path):
    from splendor.train import build_parser, Trainer
    args = build_parser().parse_args([
        "--iters", "1", "--games-per-iter", "4", "--sims", "12",
        "--train-steps", "5", "--batch-size", "16", "--width", "32",
        "--blocks", "2", "--max-moves", "30", "--device", "cpu",
        "--ckpt-dir", str(tmp_path / "ck"),
    ])
    Trainer(args).run()
    assert os.path.exists(tmp_path / "ck" / "latest.pt")

    # the checkpoint loads back into an evaluator usable by the engine
    ev = NNEvaluator.from_checkpoint(str(tmp_path / "ck" / "latest.pt"), device="cpu")
    g = GameState.new([{"id": "a"}, {"id": "b"}], seed=3)
    policy, value = ev.evaluate(g)
    assert policy and -1.0 <= value <= 1.0


def test_train_loop_eval_and_metrics(tmp_path):
    from splendor.train import build_parser, Trainer
    ck = tmp_path / "ck"
    args = build_parser().parse_args([
        "--iters", "1", "--games-per-iter", "4", "--sims", "10",
        "--train-steps", "3", "--batch-size", "16", "--width", "16",
        "--blocks", "1", "--max-moves", "24", "--device", "cpu",
        "--eval-every", "1", "--eval-games", "2", "--eval-sims", "8",
        "--ckpt-dir", str(ck),
    ])
    Trainer(args).run()
    metrics = ck / "metrics.jsonl"
    assert metrics.exists()
    import json
    rows = [json.loads(l) for l in metrics.read_text().splitlines() if l.strip()]
    assert rows and rows[-1]["eval_winrate"] is not None
