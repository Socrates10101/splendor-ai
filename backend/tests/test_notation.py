import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from splendor.engine.state import GameState
from splendor.engine.moves import legal_moves, apply_move
from splendor.engine import notation
from splendor.engine.rng import Rng

SPECS = [{"id": "p0"}, {"id": "p1"}]


def test_notation_round_trip_all_legal_moves():
    """Every legal move -> notation -> parse must recover the same move."""
    g = GameState.new(SPECS, seed=5)
    # walk a few plies of pseudo-random play, checking round-trip at each
    rng = Rng(123)
    for _ in range(40):
        if g.is_over():
            break
        moves = legal_moves(g)
        for m in moves:
            text = notation.to_notation(m)
            parsed = notation.from_notation(g, text)
            assert parsed.key() == m.key(), f"{text}: {parsed.key()} != {m.key()}"
        apply_move(g, moves[rng.randint(len(moves))])


def test_record_replay_is_deterministic():
    g = GameState.new(SPECS, seed=99)
    record = notation.new_record(SPECS, seed=99)
    rng = Rng(99)
    for _ in range(60):
        if g.is_over():
            break
        moves = legal_moves(g)
        m = moves[rng.randint(len(moves))]
        notation.append_move(record, g, m)
        apply_move(g, m)
    notation.finalize_record(record, g)

    final = notation.replay(record)
    assert final.view_state() == g.view_state()


def test_states_through_yields_each_position():
    g = GameState.new(SPECS, seed=11)
    record = notation.new_record(SPECS, seed=11)
    rng = Rng(11)
    for _ in range(20):
        moves = legal_moves(g)
        m = moves[rng.randint(len(moves))]
        notation.append_move(record, g, m)
        apply_move(g, m)

    seq = list(notation.states_through(record))
    assert len(seq) == len(record["moves"]) + 1
    assert seq[-1][1].view_state() == g.view_state()


def test_move_dict_round_trip():
    g = GameState.new(SPECS, seed=2)
    for m in legal_moves(g):
        d = notation.move_to_dict(m)
        back = notation.move_from_dict(d)
        assert back.key() == m.key()
