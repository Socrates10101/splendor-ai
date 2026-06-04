"""Splendor backend: REST API + WebSocket, serving the static frontend.

Run:  uvicorn server:app --reload --port 8000   (from the backend/ directory)
Then open http://127.0.0.1:8000/

Heavy search runs in a threadpool (sync endpoints) or via asyncio.to_thread
(WebSocket), so the event loop stays responsive during analysis.
"""
from __future__ import annotations

import asyncio
import json
import os
import time
import uuid

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from splendor.engine.state import GameState
from splendor.engine.moves import legal_moves, apply_move, find_legal
from splendor.engine import notation
from splendor.engine.notation import to_notation, move_to_dict, move_from_dict, from_notation
from splendor.engine.evaluator import HeuristicEvaluator
from splendor.engine.ismcts import search, choose_move
from splendor.engine.analyze import analyze_record
from splendor.engine.rng import Rng

app = FastAPI(title="Splendor Trainer")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

# In-memory session store (single-user local tool).
GAMES: dict[str, dict] = {}

DEFAULT_PLAYERS = [
    {"id": "p0", "name": "You", "type": "human"},
    {"id": "p1", "name": "AI", "type": "ismcts"},
]
DEFAULT_AI_SIMS = 600


def _random_seed() -> int:
    return int.from_bytes(os.urandom(8), "big")


def _legal_list(state: GameState) -> list[dict]:
    return [{"notation": to_notation(m), "action": move_to_dict(m)}
            for m in legal_moves(state)]


def _game_response(game_id: str, perspective: int | None = 0,
                   include_record: bool = False) -> dict:
    g = GAMES[game_id]
    state: GameState = g["state"]
    out = {
        "gameId": game_id,
        "view": state.view_state(perspective),
        "legalMoves": _legal_list(state),
        "aiSims": g["aiSims"],
        "players": g["record"]["players"],
    }
    if include_record:
        out["record"] = g["record"]
    return out


def _require(game_id: str) -> dict:
    g = GAMES.get(game_id)
    if g is None:
        raise HTTPException(404, f"unknown game {game_id}")
    return g


def _resolve_move(state: GameState, body: dict):
    if "notation" in body and body["notation"]:
        m = from_notation(state, body["notation"])
    elif "action" in body and body["action"]:
        m = move_from_dict(body["action"])
    else:
        raise HTTPException(400, "move requires 'notation' or 'action'")
    canonical = find_legal(state, m)
    if canonical is None:
        raise HTTPException(422, "illegal move for current position")
    return canonical


# ---------------------------------------------------------------------------
# Game lifecycle
# ---------------------------------------------------------------------------
@app.post("/api/games")
def create_game(body: dict | None = None):
    body = body or {}
    players = body.get("players") or DEFAULT_PLAYERS
    seed = body.get("seed")
    if seed is None:
        seed = _random_seed()
    ai_sims = int(body.get("aiSims", DEFAULT_AI_SIMS))
    state = GameState.new(players, seed)
    record = notation.new_record(players, seed)
    game_id = uuid.uuid4().hex[:12]
    GAMES[game_id] = {"state": state, "record": record, "aiSims": ai_sims,
                      "players": players}
    return _game_response(game_id)


@app.get("/api/games/{game_id}")
def get_game(game_id: str, perspective: int = 0):
    _require(game_id)
    return _game_response(game_id, perspective)


@app.get("/api/games/{game_id}/record")
def get_record(game_id: str):
    g = _require(game_id)
    return g["record"]


@app.post("/api/games/{game_id}/moves")
def play_move(game_id: str, body: dict):
    g = _require(game_id)
    state: GameState = g["state"]
    if state.is_over():
        raise HTTPException(409, "game is over")
    move = _resolve_move(state, body)
    notation.append_move(g["record"], state, move)
    apply_move(state, move)
    notation.finalize_record(g["record"], state)
    return _game_response(game_id)


@app.post("/api/games/{game_id}/ai-move")
def ai_move(game_id: str, body: dict | None = None):
    body = body or {}
    g = _require(game_id)
    state: GameState = g["state"]
    if state.is_over():
        raise HTTPException(409, "game is over")
    sims = int(body.get("sims", g["aiSims"]))
    summary = search(state, sims, HeuristicEvaluator(), seed=_random_seed() & 0xFFFFFFFF)
    # recover the exact legal move from the recommended notation
    target = summary["bestMove"]["notation"]
    move = next((m for m in legal_moves(state) if to_notation(m) == target),
                legal_moves(state)[0])
    notation.append_move(g["record"], state, move)
    apply_move(state, move)
    notation.finalize_record(g["record"], state)
    resp = _game_response(game_id)
    resp["aiMove"] = {"notation": to_notation(move), "summary": summary}
    return resp


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
@app.post("/api/games/{game_id}/analyze")
def analyze_game(game_id: str, body: dict | None = None):
    body = body or {}
    g = _require(game_id)
    sims = int(body.get("sims", 300))
    record = analyze_record(dict(g["record"]), sims=sims)  # shallow copy ok; moves mutated in place
    return record


@app.post("/api/analyze")
def analyze_arbitrary(body: dict):
    record = body.get("record")
    if not record:
        raise HTTPException(400, "missing 'record'")
    sims = int(body.get("sims", 300))
    return analyze_record(record, sims=sims)


@app.post("/api/position/analyze")
def analyze_position(body: dict):
    """Analyze a single position: either a live game, or a record at a ply."""
    sims = int(body.get("sims", 800))
    if "gameId" in body:
        state = _require(body["gameId"])["state"]
    elif "record" in body:
        state = notation.replay(body["record"], body.get("ply"))
    else:
        raise HTTPException(400, "provide 'gameId' or 'record'")
    summary = search(state, sims, HeuristicEvaluator(), seed=_random_seed() & 0xFFFFFFFF)
    return {"view": state.view_state(state.to_move), "summary": summary}


@app.post("/api/replay-view")
def replay_view(body: dict):
    """Board view at a given ply of a record (no search) for the review UI."""
    record = body.get("record")
    if not record:
        raise HTTPException(400, "missing 'record'")
    ply = body.get("ply")
    state = notation.replay(record, ply)
    return {"view": state.view_state(state.to_move), "legalMoves": _legal_list(state)}


@app.post("/api/import")
def import_record(body: dict):
    record = body.get("record")
    if not record:
        raise HTTPException(400, "missing 'record'")
    state = notation.replay(record)
    game_id = uuid.uuid4().hex[:12]
    GAMES[game_id] = {"state": state, "record": record,
                      "aiSims": DEFAULT_AI_SIMS, "players": record["players"]}
    return _game_response(game_id, include_record=True)


# ---------------------------------------------------------------------------
# WebSocket: stream analysis progress
# ---------------------------------------------------------------------------
@app.websocket("/ws/analyze")
async def ws_analyze(ws: WebSocket):
    await ws.accept()
    try:
        req = await ws.receive_json()
        record = req.get("record")
        if record is None and "gameId" in req:
            record = GAMES[req["gameId"]]["record"]
        sims = int(req.get("sims", 300))
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()

        def on_progress(done, total, ev):
            loop.call_soon_threadsafe(
                queue.put_nowait,
                {"type": "progress", "done": done, "total": total, "eval": ev})

        def run():
            result = analyze_record(dict(record), sims=sims, progress=on_progress)
            loop.call_soon_threadsafe(queue.put_nowait,
                                      {"type": "done", "record": result})

        task = asyncio.create_task(asyncio.to_thread(run))
        while True:
            msg = await queue.get()
            await ws.send_json(msg)
            if msg["type"] == "done":
                break
        await task
    except WebSocketDisconnect:
        pass
    except Exception as e:  # surface errors to the client for debugging
        try:
            await ws.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass


@app.websocket("/ws/position")
async def ws_position(ws: WebSocket):
    """Stream live MCTS thinking for a single position (best move so far)."""
    await ws.accept()
    try:
        req = await ws.receive_json()
        if "gameId" in req:
            state = GAMES[req["gameId"]]["state"]
        else:
            state = notation.replay(req["record"], req.get("ply"))
        sims = int(req.get("sims", 1500))
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()

        def on_progress(done, info):
            loop.call_soon_threadsafe(
                queue.put_nowait, {"type": "progress", "done": done, "summary": info})

        def run():
            result = search(state, sims, HeuristicEvaluator(),
                            seed=12345, progress=on_progress)
            loop.call_soon_threadsafe(queue.put_nowait,
                                      {"type": "done", "summary": result})

        asyncio.create_task(asyncio.to_thread(run))
        while True:
            msg = await queue.get()
            await ws.send_json(msg)
            if msg["type"] == "done":
                break
    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await ws.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass


@app.get("/healthz")
def healthz():
    return JSONResponse({"ok": True, "games": len(GAMES)})


# ---------------------------------------------------------------------------
# Static frontend (mounted last so /api, /ws, /healthz take precedence)
# ---------------------------------------------------------------------------
_FRONTEND = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend")
if os.path.isdir(_FRONTEND):
    app.mount("/", StaticFiles(directory=_FRONTEND, html=True), name="frontend")
