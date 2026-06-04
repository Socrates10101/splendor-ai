"""Append-only store + data contract for the decoupled training platform.

Phase 1 introduced the data contract and a local store. Phase 2 splits that into:

  * ``StorageBackend`` — a tiny primitive interface (``put_bytes`` / ``get_bytes``
    / ``list_keys`` / ``exists``). The append-only, never-overwrite, unique-key
    discipline lives here.
  * High-level methods (shards, models, promote events, "current" pointer) are
    implemented ONCE on the base class on top of those primitives, so every
    backend behaves identically.
  * Concrete backends: ``LocalStore`` (filesystem; dev/test) and ``GCSBackend``
    (Google Cloud Storage, native SDK). ``open_store("gs://bucket/prefix")``
    vs ``open_store("/path")`` dispatches.

Design tenets: append-only (unique keys, no rewrite), state lives in the data
layer (``current_model`` folds promote events by ``logical_generation``), data
is the asset (game = ``seed + moves + raw result`` + dense per-move policy
target; features and ``z`` are regenerated on the training side).
"""
from __future__ import annotations

import io
import json
import os
import subprocess
import time
import uuid

from .encoding import ACTION_SPACE, feature_vector
from .moves import apply_move, find_legal
from .notation import from_notation
from .state import GameState

SCHEMA_VERSION = 1
ACTION_SPACE_VERSION = 1            # the 45-action canonical layout
POLICY_TARGET_FORMAT = "dense_visit_counts_v1"   # visit_counts[45] + legal_mask[45]
RESULT_RULE_VERSION = 1             # how raw result -> z is computed (B1 points)
ENCODING_VERSION = 1               # feature_vector layout
ENGINE_VERSION = 1                 # rules/state/moves


def git_sha() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=5)
        if out.returncode == 0:
            return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return "nogit"


# ---------------------------------------------------------------------------
# Value target: raw result -> z  (RESULT_RULE_VERSION = 1, the B1 rule)
# ---------------------------------------------------------------------------
def z_from_result(result: dict, player: int) -> float:
    """Outcome from ``player``'s perspective. Capped games decided by score (B1)."""
    w = result.get("natural_winner")
    if w is not None:
        return 1.0 if w == player else -1.0
    points, cards = result["points"], result["cards"]
    keys = [(points[i], -cards[i]) for i in range(len(points))]
    best = max(keys)
    winners = [i for i, k in enumerate(keys) if k == best]
    if len(winners) == 1:
        return 1.0 if winners[0] == player else -1.0
    return 0.0  # genuine tie


def game_to_samples(game: dict):
    """Yield (features, pi[45], z) by replaying ``seed + moves`` through the engine."""
    specs = game.get("players") or [{"id": "p0"}, {"id": "p1"}]
    state = GameState.new(specs, seed=game["seed"])
    result = game["result"]
    for i, tgt in enumerate(game["targets"]):
        counts, mask = tgt["visit_counts"], tgt["legal_mask"]
        total = sum(counts)
        if total > 0:
            pi = [counts[a] / total if mask[a] else 0.0 for a in range(ACTION_SPACE)]
            feats = feature_vector(state)
            yield feats, pi, z_from_result(result, state.to_move)
        mv = find_legal(state, from_notation(state, game["moves"][i]))
        if mv is None:
            break
        apply_move(state, mv)


def make_provenance(generator_kind: str, model_version, mcts_config: dict,
                    worker_id: str, model_hash: str = "") -> dict:
    return {
        "generator_id": worker_id,
        "generator_kind": generator_kind,
        "model_version": model_version,
        "model_hash": model_hash,
        "mcts_config": mcts_config,
        "action_space_version": ACTION_SPACE_VERSION,
        "policy_target_format": POLICY_TARGET_FORMAT,
        "result_rule_version": RESULT_RULE_VERSION,
        "encoding_version": ENCODING_VERSION,
        "engine_version": ENGINE_VERSION,
        "git_sha": git_sha(),
        "worker_id": worker_id,
        "created_ts": time.time(),
    }


# ---------------------------------------------------------------------------
# StorageBackend: primitives + shared high-level logic
# ---------------------------------------------------------------------------
class StorageBackend:
    """Append-only object store. Subclasses implement the 4 byte primitives."""

    # ---- primitives (override) ----
    def put_bytes(self, key: str, data: bytes, if_absent: bool = False) -> bool:
        raise NotImplementedError

    def get_bytes(self, key: str) -> bytes:
        raise NotImplementedError

    def list_keys(self, prefix: str) -> list[str]:
        raise NotImplementedError

    def exists(self, key: str) -> bool:
        raise NotImplementedError

    # ---- shards ----
    def put_shard(self, shard: dict, worker: str = "local") -> str:
        key = f"shards/{int(time.time() * 1000)}_{worker}_{uuid.uuid4().hex[:8]}.json"
        self.put_bytes(key, json.dumps(shard).encode())
        return key

    def list_shards(self) -> list[str]:
        return sorted(k for k in self.list_keys("shards/") if k.endswith(".json"))

    def read_shard(self, key: str) -> dict:
        return json.loads(self.get_bytes(key))

    def recent_samples(self, max_positions: int):
        """Newest-first samples up to a POSITIONS budget (never a shard count)."""
        out = []
        for key in reversed(self.list_shards()):  # newest first
            shard = self.read_shard(key)
            gen = shard.get("provenance", {}).get("model_version")
            for game in shard.get("games", []):
                for feats, pi, z in game_to_samples(game):
                    out.append((feats, pi, z, gen))
                    if len(out) >= max_positions:
                        return out
        return out

    # ---- models ----
    def _model_generations(self) -> list[int]:
        gens = set()
        for k in self.list_keys("models/"):
            parts = k.split("/")
            if len(parts) >= 2 and parts[1].startswith("gen") and parts[1][3:].isdigit():
                gens.add(int(parts[1][3:]))
        return sorted(gens)

    def next_generation(self) -> int:
        gens = self._model_generations()
        return (gens[-1] + 1) if gens else 1

    def list_models(self) -> list[str]:
        return [f"gen{g:04d}" for g in self._model_generations()]

    def put_model(self, generation: int, weights_bytes: bytes, manifest: dict) -> str:
        version = f"gen{generation:04d}"
        self.put_bytes(f"models/{version}/weights.pt", weights_bytes)
        self.put_bytes(f"models/{version}/manifest.json",
                       json.dumps({**manifest, "version": version,
                                   "logical_generation": generation}).encode())
        return version

    def get_model_bytes(self, version: str) -> bytes:
        return self.get_bytes(f"models/{version}/weights.pt")

    def read_manifest(self, version: str) -> dict | None:
        try:
            return json.loads(self.get_bytes(f"models/{version}/manifest.json"))
        except Exception:  # noqa: BLE001 - missing/corrupt manifest -> None
            return None

    # ---- promote events: the "current" pointer (append-only) ----
    def promote(self, version: str, meta: dict) -> None:
        gen = meta.get("logical_generation", 0)
        key = f"events/promote/{int(time.time() * 1000)}_{version}.json"
        self.put_bytes(key, json.dumps({**meta, "version": version,
                                        "logical_generation": gen}).encode())

    # ---- liveness telemetry (segregated under _live/, NOT the immutable
    # lineage). One overwrite-OK object per worker (latest-wins, keyed by host
    # so it never collides/accumulates). This is OBSERVATION only — no worker
    # ever reads it to make a decision, so it doesn't make the store hold
    # "decision state". If the store is local, liveness stays local; if gs://,
    # it's visible cross-machine. ----
    def put_heartbeat(self, worker: str, info: dict) -> None:
        self.put_bytes(f"_live/{worker}.json",
                       json.dumps({**info, "worker": worker, "ts": time.time()}).encode())

    def read_heartbeats(self) -> list[dict]:
        out = []
        for k in self.list_keys("_live/"):
            if not k.endswith(".json"):
                continue
            try:
                out.append(json.loads(self.get_bytes(k)))
            except (json.JSONDecodeError, OSError):
                pass
        return out

    def current_model(self) -> str | None:
        """Fold promote events; pick the highest logical_generation (NOT ts)."""
        best = None
        for k in self.list_keys("events/promote/"):
            if not k.endswith(".json"):
                continue
            try:
                ev = json.loads(self.get_bytes(k))
            except (json.JSONDecodeError, OSError):
                continue
            cand = (ev.get("logical_generation", 0), ev.get("version", ""))
            if best is None or cand > best[0]:
                best = (cand, ev.get("version"))
        return best[1] if best else None


# ---------------------------------------------------------------------------
# Local filesystem backend
# ---------------------------------------------------------------------------
class LocalStore(StorageBackend):
    def __init__(self, root: str):
        self.root = os.path.abspath(root)
        os.makedirs(self.root, exist_ok=True)

    def _path(self, key: str) -> str:
        return os.path.join(self.root, *key.split("/"))

    def put_bytes(self, key: str, data: bytes, if_absent: bool = False) -> bool:
        path = self._path(key)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if if_absent and os.path.exists(path):
            return False
        tmp = path + ".tmp"
        with open(tmp, "wb") as fh:
            fh.write(data)
        os.replace(tmp, path)  # atomic publish; final object never mutated
        return True

    def get_bytes(self, key: str) -> bytes:
        with open(self._path(key), "rb") as fh:
            return fh.read()

    def list_keys(self, prefix: str) -> list[str]:
        out = []
        for dirpath, _dirs, files in os.walk(self.root):
            for f in files:
                if f.endswith(".tmp"):
                    continue
                full = os.path.join(dirpath, f)
                key = os.path.relpath(full, self.root).replace(os.sep, "/")
                if key.startswith(prefix):
                    out.append(key)
        return out

    def exists(self, key: str) -> bool:
        return os.path.exists(self._path(key))

    # local-only convenience (a real path exists; GCS has none)
    def model_weights_path(self, version: str) -> str:
        return self._path(f"models/{version}/weights.pt")


# ---------------------------------------------------------------------------
# Google Cloud Storage backend (native SDK; lazy import so it stays optional)
# ---------------------------------------------------------------------------
class GCSBackend(StorageBackend):
    def __init__(self, bucket: str, prefix: str = ""):
        try:
            from google.cloud import storage
        except ModuleNotFoundError as exc:  # pragma: no cover
            raise ModuleNotFoundError(
                "GCSBackend needs google-cloud-storage: pip install google-cloud-storage"
            ) from exc
        self._client = storage.Client()
        self._bucket = self._client.bucket(bucket)
        self._prefix = prefix.strip("/")

    def _full(self, key: str) -> str:
        return f"{self._prefix}/{key}" if self._prefix else key

    def put_bytes(self, key: str, data: bytes, if_absent: bool = False) -> bool:
        from google.api_core import exceptions as gexc
        blob = self._bucket.blob(self._full(key))
        try:
            # if_generation_match=0 => create only, fail if it already exists
            blob.upload_from_string(data, **({"if_generation_match": 0} if if_absent else {}))
            return True
        except gexc.PreconditionFailed:
            return False

    def get_bytes(self, key: str) -> bytes:
        return self._bucket.blob(self._full(key)).download_as_bytes()

    def list_keys(self, prefix: str) -> list[str]:
        base = (self._prefix + "/") if self._prefix else ""
        return sorted(b.name[len(base):]
                      for b in self._client.list_blobs(self._bucket, prefix=self._full(prefix)))

    def exists(self, key: str) -> bool:
        return self._bucket.blob(self._full(key)).exists()


def open_store(uri: str) -> StorageBackend:
    """``gs://bucket/prefix`` -> GCSBackend, otherwise a local filesystem path."""
    if uri.startswith("gs://"):
        rest = uri[len("gs://"):]
        bucket, _, prefix = rest.partition("/")
        return GCSBackend(bucket, prefix)
    return LocalStore(uri)


DEFAULT_STORE_ENV = "SPLENDOR_STORE"
# Overridable hardcoded default: env SPLENDOR_STORE and --store both win over
# this. Lets every command run truly arg-free; change/override freely.
DEFAULT_STORE = "gs://splendor-bucket-0605/main"


def default_store_uri() -> str | None:
    """Store used when ``--store`` is omitted: ``$SPLENDOR_STORE`` else DEFAULT_STORE."""
    return os.environ.get(DEFAULT_STORE_ENV) or DEFAULT_STORE or None


def resolve_store_uri(arg_value: str | None) -> str:
    uri = arg_value or default_store_uri()
    if not uri:
        raise SystemExit(
            "no store given: pass --store <dir|gs://...> or set "
            f"${DEFAULT_STORE_ENV} (e.g. export {DEFAULT_STORE_ENV}=gs://bucket/prefix)")
    return uri


# ---------------------------------------------------------------------------
# Model (de)serialisation helpers (backend-agnostic: bytes in/out)
# ---------------------------------------------------------------------------
def model_to_bytes(state_dict: dict, config: dict) -> bytes:
    import torch
    buf = io.BytesIO()
    torch.save({"model": state_dict, "config": config}, buf)
    return buf.getvalue()


def load_checkpoint_bytes(data: bytes, device=None):
    import torch
    return torch.load(io.BytesIO(data), map_location=device, weights_only=False)
