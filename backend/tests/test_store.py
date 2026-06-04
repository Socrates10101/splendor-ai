"""Phase 1: data contract + append-only store + generate/train decoupling."""
import pytest

from splendor.encoding import ACTION_SPACE
from splendor.evaluator import HeuristicEvaluator
from splendor.generate import generate_game
from splendor.store import (LocalStore, game_to_samples, make_provenance,
                            z_from_result, SCHEMA_VERSION)


def test_z_from_result_rule():
    # natural winner
    assert z_from_result({"natural_winner": 0, "points": [15, 8], "cards": [9, 7]}, 0) == 1.0
    assert z_from_result({"natural_winner": 0, "points": [15, 8], "cards": [9, 7]}, 1) == -1.0
    # capped: decided by points
    r = {"natural_winner": None, "points": [9, 4], "cards": [6, 5]}
    assert z_from_result(r, 0) == 1.0 and z_from_result(r, 1) == -1.0
    # capped tie -> draw
    r2 = {"natural_winner": None, "points": [9, 9], "cards": [6, 6]}
    assert z_from_result(r2, 0) == 0.0


def test_generate_game_record_shape():
    g = generate_game(HeuristicEvaluator(), seed=1, sims=12, temp=1.0,
                      temp_moves=4, max_moves=24)
    assert g["seed"] == 1 and g["targets"]
    assert len(g["moves"]) == len(g["targets"])
    t = g["targets"][0]
    assert len(t["visit_counts"]) == ACTION_SPACE
    assert len(t["legal_mask"]) == ACTION_SPACE
    # at least one legal action flagged
    assert sum(t["legal_mask"]) >= 1


def test_game_to_samples_regenerates_features():
    g = generate_game(HeuristicEvaluator(), seed=3, sims=12, temp=1.0,
                      temp_moves=4, max_moves=24)
    samples = list(game_to_samples(g))
    assert samples
    feats, pi, z = samples[0]
    assert len(pi) == ACTION_SPACE
    assert abs(sum(pi) - 1.0) < 1e-5
    assert z in (-1.0, 0.0, 1.0)
    assert len(feats) == 126  # 2-player feature vector


def test_store_roundtrip_and_current_pointer(tmp_path):
    store = LocalStore(str(tmp_path / "data"))
    g = generate_game(HeuristicEvaluator(), seed=5, sims=12, temp=1.0,
                      temp_moves=4, max_moves=24)
    shard = {"schema_version": SCHEMA_VERSION,
             "provenance": make_provenance("heuristic-bootstrap", None,
                                           {"sims": 12}, "tester"),
             "games": [g]}
    key = store.put_shard(shard)
    assert key in store.list_shards()
    assert store.recent_samples(1000)  # yields samples

    # promote pointer uses logical_generation, not wall-clock
    assert store.current_model() is None
    store.promote("gen0001", {"logical_generation": 1})
    store.promote("gen0002", {"logical_generation": 2})
    assert store.current_model() == "gen0002"


def test_open_store_dispatch(tmp_path):
    from splendor.store import open_store, LocalStore
    assert isinstance(open_store(str(tmp_path)), LocalStore)
    # gs:// would construct a GCSBackend; we only check the path branch here so
    # the test doesn't require google-cloud-storage / credentials.


def test_model_bytes_roundtrip_via_backend(tmp_path):
    pytest.importorskip("torch")
    from splendor.store import LocalStore, model_to_bytes, load_checkpoint_bytes
    from splendor.nn import PolicyValueNet
    store = LocalStore(str(tmp_path / "data"))
    net = PolicyValueNet(126, width=16, blocks=1)
    store.put_model(1, model_to_bytes(net.state_dict(), net.config()),
                    {"parent": None})
    assert store.list_models() == ["gen0001"]
    ckpt = load_checkpoint_bytes(store.get_model_bytes("gen0001"))
    assert ckpt["config"]["width"] == 16
    assert store.read_manifest("gen0001")["logical_generation"] == 1


def test_status_collect_and_render(tmp_path):
    from splendor.status import collect, render
    store = LocalStore(str(tmp_path / "data"))
    g = generate_game(HeuristicEvaluator(), seed=9, sims=10, temp=1.0,
                      temp_moves=3, max_moves=20)
    store.put_shard({"schema_version": SCHEMA_VERSION,
                     "provenance": make_provenance("heuristic-bootstrap", None,
                                                   {"sims": 10}, "minipc"),
                     "games": [g]}, worker="minipc")
    store.put_model(1, b"weights", {"parent": None, "ploss": 1.2})
    store.promote("gen0001", {"logical_generation": 1})
    store.put_heartbeat("minipc", {"kind": "generate", "state": "idle",
                                   "model_version": "gen0001", "detail": "live"})

    s = collect(store, recent=50)
    assert s["shards"] == 1 and s["games"] == 1
    assert s["positions"] == len(g["targets"])
    assert s["current"] == "gen0001"
    assert "minipc" in s["workers"]
    assert any(h["worker"] == "minipc" for h in s["heartbeats"])
    text = render("local", s, ttl=180, color=False)
    assert "現役モデル" in text and "gen0001" in text
    assert "Splendor 学習状況" in text
    assert "● minipc" in text  # shown as a live worker


def test_end_to_end_generate_then_train(tmp_path):
    torch = pytest.importorskip("torch")
    from splendor.train_store import main as train_main
    from splendor.generate import main as gen_main
    from splendor.nn import NNEvaluator

    d = str(tmp_path / "data")
    gen_main(["--store", d, "--shards", "1", "--games", "3", "--sims", "10",
              "--temp-moves", "3", "--max-moves", "24", "--seed", "1"])
    store = LocalStore(d)
    assert store.list_shards()

    train_main(["--store", d, "--steps", "5", "--batch-size", "16",
                "--width", "32", "--blocks", "1", "--device", "cpu", "--promote"])
    assert store.current_model() == "gen0001"

    # the published model loads into an evaluator usable by the engine/generator
    ev = NNEvaluator.from_checkpoint(store.model_weights_path("gen0001"), device="cpu")
    from splendor.state import GameState
    policy, value = ev.evaluate(GameState.new([{"id": "a"}, {"id": "b"}], seed=2))
    assert policy and -1.0 <= value <= 1.0
