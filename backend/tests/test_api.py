import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

import server

client = TestClient(server.app)


def _new_game(seed=12345, ai_sims=40):
    r = client.post("/api/games", json={"seed": seed, "aiSims": ai_sims})
    assert r.status_code == 200
    return r.json()


def test_healthz():
    assert client.get("/healthz").json()["ok"] is True


def test_create_and_view_game():
    data = _new_game()
    assert "gameId" in data
    view = data["view"]
    assert view["numPlayers"] == 2
    assert len(data["legalMoves"]) > 0
    assert view["bank"]["w"] == 4


def test_play_human_move_by_notation():
    data = _new_game()
    gid = data["gameId"]
    note = data["legalMoves"][0]["notation"]
    r = client.post(f"/api/games/{gid}/moves", json={"notation": note})
    assert r.status_code == 200
    assert r.json()["view"]["toMove"] == 1


def test_illegal_move_rejected():
    data = _new_game()
    gid = data["gameId"]
    r = client.post(f"/api/games/{gid}/moves", json={"notation": "P:L3#0"})
    # buying a Lv3 card on turn 1 is unaffordable -> 422
    assert r.status_code == 422


def test_ai_move_and_record():
    data = _new_game(ai_sims=30)
    gid = data["gameId"]
    # human plays first, then AI
    client.post(f"/api/games/{gid}/moves", json={"notation": data["legalMoves"][0]["notation"]})
    r = client.post(f"/api/games/{gid}/ai-move", json={"sims": 30})
    assert r.status_code == 200
    body = r.json()
    assert "aiMove" in body
    assert body["view"]["toMove"] == 0
    rec = client.get(f"/api/games/{gid}/record").json()
    assert len(rec["moves"]) == 2


def test_position_analyze():
    data = _new_game()
    gid = data["gameId"]
    r = client.post("/api/position/analyze", json={"gameId": gid, "sims": 60})
    assert r.status_code == 200
    summary = r.json()["summary"]
    assert summary["bestMove"] is not None
    assert 0.0 <= summary["winProb"] <= 1.0


def test_full_game_analyze_and_import():
    data = _new_game(ai_sims=20)
    gid = data["gameId"]
    # play a handful of plies via ai-move on both sides
    for _ in range(6):
        view = client.get(f"/api/games/{gid}").json()["view"]
        if view["winner"] is not None:
            break
        client.post(f"/api/games/{gid}/ai-move", json={"sims": 20})
    analyzed = client.post(f"/api/games/{gid}/analyze", json={"sims": 20}).json()
    assert all(m["eval"] is not None for m in analyzed["moves"])
    # round-trip import of the analyzed record
    imp = client.post("/api/import", json={"record": analyzed}).json()
    assert imp["view"]["ply"] == len(analyzed["moves"])


def test_ws_analyze_streams_progress():
    data = _new_game(ai_sims=20)
    gid = data["gameId"]
    for _ in range(4):
        client.post(f"/api/games/{gid}/ai-move", json={"sims": 20})
    record = client.get(f"/api/games/{gid}/record").json()
    progress_seen = 0
    done = None
    with client.websocket_connect("/ws/analyze") as ws:
        ws.send_json({"record": record, "sims": 20})
        while True:
            msg = ws.receive_json()
            if msg["type"] == "progress":
                progress_seen += 1
            elif msg["type"] == "done":
                done = msg
                break
            elif msg["type"] == "error":
                raise AssertionError(msg["message"])
    assert done is not None
    assert progress_seen >= 1
    assert all(m["eval"] is not None for m in done["record"]["moves"])


def test_training_metrics_endpoint(tmp_path):
    import json
    d = tmp_path / "ck"
    d.mkdir()
    (d / "metrics.jsonl").write_text(
        json.dumps({"iter": 1, "ploss": 2.0, "vloss": 0.1,
                    "eval_winrate": 0.3, "eval_elo": -50.0}) + "\n")
    (d / "status.json").write_text(
        json.dumps({"state": "running", "iteration": 1, "best_winrate": 0.3}))
    (d / "best.pt").write_bytes(b"weights")

    saved = server.TRAIN["dir"]
    server.TRAIN["dir"] = str(d)
    try:
        r = client.get("/api/training/metrics")
        assert r.status_code == 200
        j = r.json()
        assert j["status"]["state"] == "running"
        assert j["totalIters"] == 1
        assert j["metrics"][0]["eval_winrate"] == 0.3
        assert any(c["name"] == "best.pt" for c in j["checkpoints"])
        assert "managed" in j
        assert client.get("/training").status_code == 200
    finally:
        server.TRAIN["dir"] = saved


def test_training_options_endpoint():
    r = client.get("/api/training/options")
    assert r.status_code == 200
    j = r.json()
    assert "defaults" in j and "hours" in j["defaults"]
    assert any(p["id"] == "standard" for p in j["presets"])
    assert "checkpoints" in j
    assert j["running"] is False


def test_arena_endpoint_runs_heuristic_match():
    import time
    r = client.post("/api/training/arena",
                    json={"a": "heuristic", "b": "heuristic", "games": 2, "sims": 8})
    assert r.status_code == 200
    for _ in range(60):
        s = client.get("/api/training/arena").json()
        if s["state"] in ("done", "error"):
            break
        time.sleep(0.5)
    assert s["state"] == "done", s
    assert s["result"]["games"] == 2
    assert 0.0 <= s["result"]["a_winrate"] <= 1.0
