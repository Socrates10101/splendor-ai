"""CLI status view for the training data store (headless / over SSH).

A terminal dashboard for the append-only store — current model, generations,
data volume, generation rate, recent promotes — for either a local path or a
``gs://bucket/prefix`` store. Read-only and torch-free (parses shard keys for
cheap totals; reads only the most recent shards for the detailed breakdown).

    python -m splendor.status --store gs://splendor-bucket-0605/main
    python -m splendor.status --store gs://splendor-bucket-0605/main --watch 10
"""
from __future__ import annotations

import argparse
import datetime as dt
import time

from .store import open_store, resolve_store_uri


def _parse_shard_key(key: str):
    """shards/<ms>_<worker>_<uid>.json -> (epoch_seconds, worker)."""
    name = key.split("/", 1)[1] if "/" in key else key
    if name.endswith(".json"):
        name = name[:-5]
    parts = name.split("_")
    try:
        ts = int(parts[0]) / 1000.0
    except (ValueError, IndexError):
        ts = 0.0
    worker = "_".join(parts[1:-1]) if len(parts) >= 3 else "?"
    return ts, worker


def _ago(seconds: float) -> str:
    if seconds < 0:
        return "—"
    if seconds < 90:
        return f"{int(seconds)}s ago"
    if seconds < 5400:
        return f"{seconds / 60:.0f}m ago"
    if seconds < 172800:
        return f"{seconds / 3600:.1f}h ago"
    return f"{seconds / 86400:.1f}d ago"


def collect(store, recent: int) -> dict:
    now = time.time()
    shard_keys = store.list_shards()
    times, workers = [], {}
    for k in shard_keys:
        ts, w = _parse_shard_key(k)
        times.append(ts)
        workers[w] = workers.get(w, 0) + 1
    times = [t for t in times if t > 0]
    last_hour = sum(1 for t in times if now - t <= 3600)

    # detailed breakdown from the most recent `recent` shards only (bounded I/O)
    sample_keys = sorted(shard_keys)[-recent:]
    games = positions = 0
    by_model: dict[str, int] = {}
    for k in sample_keys:
        try:
            shard = store.read_shard(k)
        except Exception:  # noqa: BLE001
            continue
        mv = shard.get("provenance", {}).get("model_version") or "heuristic"
        for g in shard.get("games", []):
            games += 1
            n = len(g.get("targets", []))
            positions += n
            by_model[mv] = by_model.get(mv, 0) + n

    heartbeats = sorted(store.read_heartbeats(), key=lambda h: h.get("ts", 0),
                        reverse=True)
    current = store.current_model()
    manifest = store.read_manifest(current) if current else None
    models = store.list_models()

    # recent promotes (newest first by logical_generation)
    promotes = []
    for pk in store.list_keys("events/promote/"):
        try:
            import json
            ev = json.loads(store.get_bytes(pk))
            promotes.append((ev.get("logical_generation", 0), ev.get("version", "?")))
        except Exception:  # noqa: BLE001
            pass
    promotes.sort(reverse=True)

    return {
        "now": now,
        "shards": len(shard_keys),
        "newest_ts": max(times) if times else 0,
        "last_hour": last_hour,
        "workers": workers,
        "sampled": len(sample_keys),
        "games": games,
        "positions": positions,
        "by_model": by_model,
        "current": current,
        "manifest": manifest or {},
        "models": models,
        "promotes": promotes[:5],
        "heartbeats": heartbeats,
    }


def render(store_uri: str, s: dict, ttl: float = 180) -> str:
    now = s["now"]
    L = []
    stamp = dt.datetime.fromtimestamp(now).strftime("%Y-%m-%d %H:%M:%S")
    L.append(f"Splendor training status  —  {store_uri}   ({stamp})")
    L.append("─" * 72)

    m = s["manifest"]
    cur = s["current"] or "—"
    detail = ""
    if m:
        bits = []
        if m.get("logical_generation") is not None:
            bits.append(f"gen{m['logical_generation']}")
        if m.get("parent"):
            bits.append(f"parent={m['parent']}")
        if m.get("ploss") is not None:
            bits.append(f"ploss={m['ploss']}")
        if m.get("vloss") is not None:
            bits.append(f"vloss={m['vloss']}")
        if m.get("eval_winrate") is not None:
            bits.append(f"eval={m['eval_winrate']}")
        if m.get("source"):
            bits.append(str(m["source"]))
        detail = "  " + "  ".join(bits)
    L.append(f"current model : {cur}{detail}")
    L.append(f"models        : {len(s['models'])}   {s['models'][-6:]}")

    newest_ago = _ago(now - s["newest_ts"]) if s["newest_ts"] else "—"
    L.append(f"shards        : {s['shards']}   (newest {newest_ago})")
    L.append(f"data (recent {s['sampled']} shards): games={s['games']}  positions={s['positions']}")
    if s["by_model"]:
        bym = "  ".join(f"{k}:{v}" for k, v in sorted(s["by_model"].items()))
        L.append(f"positions/model: {bym}")
    if s["workers"]:
        wk = "  ".join(f"{k}:{v}" for k, v in sorted(s["workers"].items()))
        L.append(f"generators    : {wk}")
    L.append(f"rate          : {s['last_hour']} shards in last 1h")
    if s["promotes"]:
        pr = "  ".join(f"{v}@gen{g}" for g, v in s["promotes"])
        L.append(f"recent promote: {pr}")

    hbs = s.get("heartbeats", [])

    def _is_live(h):
        return (now - h.get("ts", 0)) <= ttl and h.get("state") != "stopped"

    live = sum(1 for h in hbs if _is_live(h))
    L.append("─" * 72)
    L.append(f"workers       : {live} live / {len(hbs)} known  (live = seen ≤{int(ttl)}s)")
    for h in hbs:
        age = now - h.get("ts", 0)
        mark = "●" if _is_live(h) else "○"
        L.append(f"  {mark} {h.get('worker', '?'):16} {h.get('kind', '?'):8} "
                 f"{h.get('state', '?'):11} {str(h.get('model_version') or '-'):8} "
                 f"{_ago(age):>9}  {h.get('detail', '')}")
    return "\n".join(L)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Status view for the training store")
    p.add_argument("--store", default=None,
                   help="local dir or gs://bucket/prefix (default: $SPLENDOR_STORE)")
    p.add_argument("--recent", type=int, default=50,
                   help="how many recent shards to read for the detailed breakdown")
    p.add_argument("--watch", type=float, default=0,
                   help="refresh every N seconds (0 = print once)")
    p.add_argument("--ttl", type=float, default=180,
                   help="seconds since last heartbeat to still count a worker live")
    args = p.parse_args(argv)

    store_uri = resolve_store_uri(args.store)
    store = open_store(store_uri)
    while True:
        s = collect(store, args.recent)
        out = render(store_uri, s, ttl=args.ttl)
        if args.watch > 0:
            print("\033[2J\033[H", end="")  # clear screen for a live view
        print(out, flush=True)
        if args.watch <= 0:
            break
        time.sleep(args.watch)


if __name__ == "__main__":
    main()
