import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from splendor.state import GameState
from splendor.moves import legal_moves, apply_move
from splendor.evaluator import HeuristicEvaluator, win_prob
from splendor.ismcts import search, choose_move, determinize
from splendor import notation, analyze
from splendor.rng import Rng

SPECS = [{"id": "p0"}, {"id": "p1"}]


def _play_game(p0_sims, p1_sims, seed, max_plies=300):
    """Return (winner_index_or_None, record). sims=0 -> random policy."""
    g = GameState.new(SPECS, seed=seed)
    record = notation.new_record(SPECS, seed=seed)
    rng = Rng(seed ^ 0xABCD)
    while not g.is_over() and g.ply < max_plies:
        sims = p0_sims if g.to_move == 0 else p1_sims
        if sims == 0:
            moves = legal_moves(g)
            m = moves[rng.randint(len(moves))]
        else:
            m = choose_move(g, sims, HeuristicEvaluator(), seed=seed + g.ply)
        notation.append_move(record, g, m)
        apply_move(g, m)
    notation.finalize_record(record, g)
    return g.winner, record


def test_determinize_preserves_visible_state():
    g = GameState.new(SPECS, seed=4)
    # give p1 a hidden deck-reserved card
    res = [m for m in legal_moves(g) if m.type == "reserve" and m.source == "deck"]
    apply_move(g, res[0])                 # p0 reserves (now p1)
    res = [m for m in legal_moves(g) if m.type == "reserve" and m.source == "deck"]
    apply_move(g, res[0])                 # p1 reserves
    rng = Rng(1)
    d = determinize(g, root=0, rng=rng)
    # visible things unchanged
    assert d.bank == g.bank
    assert d.board == g.board
    assert [c for c in d.players[0].reserved] == g.players[0].reserved  # own kept
    # deck counts preserved
    for t in (1, 2, 3):
        assert len(d.decks[t]) == len(g.decks[t])


def test_search_returns_legal_best_move():
    g = GameState.new(SPECS, seed=2)
    summary = search(g, sims=120, evaluator=HeuristicEvaluator(), seed=7)
    assert summary["bestMove"] is not None
    assert 0.0 <= summary["winProb"] <= 1.0
    legal_notations = {notation.to_notation(m) for m in legal_moves(g)}
    assert summary["bestMove"]["notation"] in legal_notations
    m = choose_move(g, sims=120, seed=7)
    assert notation.to_notation(m) in legal_notations


def test_mcts_beats_random():
    wins = 0
    games = 6
    for s in range(games):
        # alternate colors to cancel first-move advantage
        if s % 2 == 0:
            winner, _ = _play_game(p0_sims=80, p1_sims=0, seed=100 + s)
            wins += 1 if winner == 0 else 0
        else:
            winner, _ = _play_game(p0_sims=0, p1_sims=80, seed=100 + s)
            wins += 1 if winner == 1 else 0
    assert wins >= 5, f"MCTS only won {wins}/{games} vs random"


def test_analyze_annotates_record():
    _, record = _play_game(p0_sims=0, p1_sims=0, seed=55, max_plies=24)
    analyze.analyze_record(record, sims=40)
    assert all(mv["eval"] is not None for mv in record["moves"])
    ev0 = record["moves"][0]["eval"]
    assert 0.0 <= ev0["winProbBefore"] <= 1.0
    assert "delta" in ev0 and "blunder" in ev0
    assert ev0["bestMove"] is not None
