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
import signal
import subprocess
import sys
import threading
import time
import uuid

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from splendor.state import GameState
from splendor.moves import legal_moves, apply_move, find_legal
from splendor import notation
from splendor.notation import to_notation, move_to_dict, move_from_dict, from_notation
from splendor.evaluator import HeuristicEvaluator
from splendor.ismcts import search, choose_move
from splendor.analyze import analyze_record
from splendor.rng import Rng

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
# Training control + monitor
#
# The server launches `python -m splendor.train` as a child process so the UI
# can start/stop training without the user touching the CLI, then tails the
# run directory (status.json / metrics.jsonl) for the live view.
# ---------------------------------------------------------------------------
_BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
_RUNS_ROOT = os.path.join(_BACKEND_DIR, "checkpoints_runs")
_METRICS_TAIL = 1500  # cap rows returned for the chart

# Friendly field -> (cli flag, type, default). These are the only args the UI
# can set; the command is built from this whitelist (no shell, no injection).
TRAIN_FIELDS: dict = {
    "hours": ("--hours", float, 1.0),
    "workers": ("--workers", int, 10),
    "games_per_iter": ("--games-per-iter", int, 256),
    "sims": ("--sims", int, 96),
    "eval_every": ("--eval-every", int, 10),
    "eval_games": ("--eval-games", int, 40),
    "train_steps": ("--train-steps", int, 200),
    "lr": ("--lr", float, 1e-3),
    "temp": ("--temp", float, 1.0),
    "entropy_coef": ("--entropy-coef", float, 0.0),
    "width": ("--width", int, 256),
    "blocks": ("--blocks", int, 4),
    "device": ("--device", str, "mps"),
    "seed": ("--seed", int, 12345),
}

# Presets spell out EVERY tunable so switching presets never leaves a stale
# value behind. "standard" is the proven run-1 config (reached ~54% vs the
# heuristic) plus the B1 collapse fix: 256/4 net, constant lr, no entropy reg.
_BASE = {"workers": 10, "eval_every": 10, "eval_games": 40, "train_steps": 200,
         "lr": 1e-3, "temp": 1.0, "entropy_coef": 0.0, "device": "mps",
         "seed": 12345}
TRAIN_PRESETS = [
    {"id": "quick", "label": "お試し（軽め・30分）",
     "values": {**_BASE, "hours": 0.5, "workers": 6, "games_per_iter": 96,
                "sims": 48, "eval_every": 5, "width": 256, "blocks": 4}},
    {"id": "standard", "label": "標準（実証済み・6時間）",
     "values": {**_BASE, "hours": 6, "games_per_iter": 256, "sims": 96,
                "width": 256, "blocks": 4}},
    {"id": "deep", "label": "大きめネット実験（12時間）",
     "values": {**_BASE, "hours": 12, "games_per_iter": 384, "sims": 128,
                "width": 384, "blocks": 6}},
]

# Live state of the managed training subprocess.
TRAIN: dict = {
    "proc": None,            # subprocess.Popen | None
    "dir": os.path.abspath(os.environ.get("SPLENDOR_CKPT_DIR", "checkpoints")),
    "run": None,
    "logfile": None,
    "cmd": None,
    "started_at": None,
}


def _train_running() -> bool:
    p = TRAIN["proc"]
    return p is not None and p.poll() is None


def _tail(path: str | None, n: int = 30) -> list[str]:
    if not path or not os.path.exists(path):
        return []
    try:
        with open(path, "r", errors="replace") as fh:
            return [ln.rstrip("\n") for ln in fh.readlines()[-n:]]
    except OSError:
        return []


def _ckpt_tag(name: str) -> str:
    if name == "best.pt":
        return "best"
    if name == "latest.pt":
        return "latest"
    if name.startswith("best_i"):
        return "peak"
    if name.startswith("hourly_"):
        return "hourly"
    return "ckpt"


def _ckpt_meta(path: str) -> tuple:
    """(iteration, winrate) from the sidecar JSON, or (None, None)."""
    sc = path + ".json"
    if os.path.exists(sc):
        try:
            with open(sc) as fh:
                m = json.load(fh)
            wr = (m.get("eval_winrate_ema") if m.get("eval_winrate_ema") is not None
                  else m.get("eval_winrate") if m.get("eval_winrate") is not None
                  else m.get("best_winrate"))
            return m.get("iteration"), wr
        except (OSError, json.JSONDecodeError):
            pass
    return None, None


def _list_all_checkpoints() -> list[dict]:
    """Every *.pt under checkpoints/ and checkpoints_runs/*, enriched for the UI."""
    roots = [os.path.join(_BACKEND_DIR, "checkpoints")]
    if os.path.isdir(_RUNS_ROOT):
        roots += [os.path.join(_RUNS_ROOT, d) for d in sorted(os.listdir(_RUNS_ROOT))]
    out = []
    for root in roots:
        if not os.path.isdir(root):
            continue
        run = os.path.relpath(root, _BACKEND_DIR)
        for name in sorted(os.listdir(root)):
            if name.endswith(".pt"):
                p = os.path.join(root, name)
                it, wr = _ckpt_meta(p)
                out.append({
                    "path": os.path.relpath(p, _BACKEND_DIR),
                    "run": run, "name": name, "label": f"{run}/{name}",
                    "tag": _ckpt_tag(name), "iteration": it, "winrate": wr,
                    "size": os.path.getsize(p), "mtime": os.path.getmtime(p),
                })
    out.sort(key=lambda c: c["mtime"], reverse=True)
    return out


@app.get("/api/training/options")
def training_options():
    """Defaults + presets so the UI can render the form without hardcoding."""
    return {
        "defaults": {k: d for k, (_f, _t, d) in TRAIN_FIELDS.items()},
        "presets": TRAIN_PRESETS,
        "checkpoints": _list_all_checkpoints(),
        "running": _train_running(),
    }


@app.post("/api/training/start")
def training_start(body: dict | None = None):
    body = body or {}
    if _train_running():
        raise HTTPException(409, "training already running")

    raw = str(body.get("run") or "run1")
    run = "".join(c for c in raw if c.isalnum() or c in "-_") or "run1"
    run_dir = os.path.join(_RUNS_ROOT, run)
    os.makedirs(run_dir, exist_ok=True)

    argv = [sys.executable, "-u", "-m", "splendor.train", "--ckpt-dir", run_dir]
    if body.get("resume"):
        argv.append("--resume")
    init_from = str(body.get("init_from") or "").strip()
    if init_from:
        abs_init = os.path.abspath(os.path.join(_BACKEND_DIR, init_from))
        if not (abs_init.startswith(_BACKEND_DIR) and abs_init.endswith(".pt")
                and os.path.exists(abs_init)):
            raise HTTPException(400, f"invalid init_from checkpoint: {init_from}")
        argv += ["--init-from", abs_init]
    for key, (flag, typ, default) in TRAIN_FIELDS.items():
        val = body.get(key, default)
        try:
            val = typ(val)
        except (TypeError, ValueError):
            val = default
        argv += [flag, str(val)]

    logpath = os.path.join(run_dir, "train.log")
    logf = open(logpath, "ab")
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    proc = subprocess.Popen(argv, cwd=_BACKEND_DIR, stdout=logf,
                            stderr=subprocess.STDOUT, env=env)
    TRAIN.update(proc=proc, dir=run_dir, run=run, logfile=logpath,
                 cmd=" ".join(argv), started_at=time.time())
    return {"ok": True, "pid": proc.pid, "run": run, "ckptDir": run_dir,
            "cmd": TRAIN["cmd"]}


@app.post("/api/training/stop")
def training_stop():
    if not _train_running():
        return {"ok": True, "running": False}
    # SIGINT = graceful: the trainer finishes the current iteration, writes a
    # checkpoint, then exits.
    TRAIN["proc"].send_signal(signal.SIGINT)
    return {"ok": True, "running": True,
            "message": "現在のイテレーションを終えてから停止します"}


@app.get("/api/training/metrics")
def training_metrics(limit: int = _METRICS_TAIL):
    d = TRAIN["dir"]
    status = None
    status_path = os.path.join(d, "status.json")
    if os.path.exists(status_path):
        try:
            with open(status_path) as fh:
                status = json.load(fh)
        except (json.JSONDecodeError, OSError):
            status = None  # mid-write; the next poll will pick it up

    metrics: list[dict] = []
    metrics_path = os.path.join(d, "metrics.jsonl")
    total = 0
    if os.path.exists(metrics_path):
        with open(metrics_path) as fh:
            lines = fh.readlines()
        total = len(lines)
        for line in lines[-limit:]:
            line = line.strip()
            if line:
                try:
                    metrics.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    checkpoints = []
    if os.path.isdir(d):
        for name in sorted(os.listdir(d)):
            if name.endswith(".pt"):
                p = os.path.join(d, name)
                checkpoints.append({"name": name, "size": os.path.getsize(p),
                                    "mtime": os.path.getmtime(p)})

    return {
        "ckptDir": d,
        "status": status,
        "totalIters": total,
        "metrics": metrics,
        "checkpoints": checkpoints,
        "allCheckpoints": _list_all_checkpoints(),
        "managed": {
            "running": _train_running(),
            "run": TRAIN["run"],
            "cmd": TRAIN["cmd"],
            "pid": TRAIN["proc"].pid if _train_running() else None,
        },
        "log": _tail(TRAIN["logfile"], 30),
    }


# ---------------------------------------------------------------------------
# Arena compare: play two checkpoints (or vs heuristic) head-to-head so the
# user can pick the TRUE strongest among candidates instead of trusting the
# single noisy in-loop best. Runs in a background thread (one at a time).
# ---------------------------------------------------------------------------
ARENA: dict = {"thread": None, "id": None, "state": "idle", "result": None,
               "progress": None, "a": None, "b": None}


def _resolve_ckpt_spec(spec: str) -> str:
    if spec == "heuristic":
        return "heuristic"
    abs_p = os.path.abspath(os.path.join(_BACKEND_DIR, spec))
    if not (abs_p.startswith(_BACKEND_DIR) and abs_p.endswith(".pt")
            and os.path.exists(abs_p)):
        raise HTTPException(400, f"invalid checkpoint: {spec}")
    return abs_p


@app.post("/api/training/arena")
def arena_start(body: dict):
    if ARENA["thread"] is not None and ARENA["thread"].is_alive():
        raise HTTPException(409, "an arena match is already running")
    a_spec = str(body.get("a") or "")
    b_spec = str(body.get("b") or "heuristic")
    if not a_spec:
        raise HTTPException(400, "missing 'a'")
    a_path = _resolve_ckpt_spec(a_spec)
    b_path = _resolve_ckpt_spec(b_spec)
    games = max(2, min(int(body.get("games", 40)), 400))
    sims = max(8, min(int(body.get("sims", 96)), 2000))

    from splendor.arena import play_match, load_evaluator
    ARENA.update(id=uuid.uuid4().hex[:8], state="running", result=None,
                 progress={"done": 0, "total": games}, a=a_spec, b=b_spec)

    def run():
        try:
            ev_a = load_evaluator(a_path, device="cpu")
            ev_b = load_evaluator(b_path, device="cpu")
            res = play_match(ev_a, ev_b, games=games, sims=sims,
                             progress=lambda d, t: ARENA.update(
                                 progress={"done": d, "total": t}))
            ARENA.update(state="done", result=res)
        except Exception as exc:  # noqa: BLE001 - surface to UI
            ARENA.update(state="error", result={"error": str(exc)})

    t = threading.Thread(target=run, daemon=True)
    ARENA["thread"] = t
    t.start()
    return {"ok": True, "id": ARENA["id"]}


@app.get("/api/training/arena")
def arena_status():
    return {k: ARENA[k] for k in ("id", "state", "result", "progress", "a", "b")}


@app.get("/training")
def training_page():
    path = os.path.join(_FRONTEND, "training.html")
    if not os.path.exists(path):
        raise HTTPException(404, "training.html not found")
    return FileResponse(path)


# ---------------------------------------------------------------------------
# Static frontend (mounted last so /api, /ws, /healthz take precedence)
# ---------------------------------------------------------------------------
_FRONTEND = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend")
if os.path.isdir(_FRONTEND):
    app.mount("/", StaticFiles(directory=_FRONTEND, html=True), name="frontend")
