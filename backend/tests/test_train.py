"""Smoke tests for the shared neural net (nn.py) and the arena (eval/arena.py).

(The legacy local-training tests were removed with the legacy flow; the current
data-lake flow is covered by test_store.py.)
"""
import pytest

torch = pytest.importorskip("torch")

from splendor.engine.encoding import ACTION_SPACE, feature_vector
from splendor.engine.evaluator import HeuristicEvaluator
from splendor.engine.state import GameState
from splendor.eval.arena import play_match
from splendor.nn import NNEvaluator, PolicyValueNet


def _input_dim():
    g = GameState.new([{"id": "a"}, {"id": "b"}], seed=1)
    return len(feature_vector(g))


def test_net_shapes():
    dim = _input_dim()
    net = PolicyValueNet(dim, width=32, blocks=2)
    logits, value = net(torch.zeros((4, dim)))
    assert logits.shape == (4, ACTION_SPACE)
    assert value.shape == (4,)
    assert torch.all(value.abs() <= 1.0)


def test_nn_evaluator_matches_evaluator_contract():
    dim = _input_dim()
    ev = NNEvaluator(PolicyValueNet(dim, width=32, blocks=2),
                     device=torch.device("cpu"))
    policy, value = ev.evaluate(GameState.new([{"id": "a"}, {"id": "b"}], seed=7))
    assert -1.0 <= value <= 1.0
    assert policy and abs(sum(policy.values()) - 1.0) < 1e-5


def test_arena_match_reports_winrate():
    res = play_match(HeuristicEvaluator(), HeuristicEvaluator(), games=4,
                     sims=16, max_moves=40)
    assert res["games"] == 4
    assert res["a_wins"] + res["b_wins"] + res["draws"] == 4
    assert 0.0 <= res["a_winrate"] <= 1.0
