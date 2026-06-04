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
import os
import sys
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
        return "たった今"   # worker clock slightly ahead of this host (skew)
    if seconds < 90:
        return f"{int(seconds)}秒前"
    if seconds < 5400:
        return f"{seconds / 60:.0f}分前"
    if seconds < 172800:
        return f"{seconds / 3600:.1f}時間前"
    return f"{seconds / 86400:.1f}日前"


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


_ANSI = {"reset": "\033[0m", "bold": "\033[1m", "dim": "\033[2m",
         "red": "\033[31m", "green": "\033[32m", "yellow": "\033[33m",
         "blue": "\033[34m", "magenta": "\033[35m", "cyan": "\033[36m",
         "gray": "\033[90m"}
# state -> (Japanese label, colour)
_STATE_JA = {"generating": ("生成中", "cyan"), "idle": ("待機", "green"),
             "training": ("学習中", "magenta"), "stopped": ("停止", "red")}


def _painter(enabled: bool):
    def paint(text, *styles):
        if not enabled or not styles:
            return str(text)
        return "".join(_ANSI[s] for s in styles) + str(text) + _ANSI["reset"]
    return paint


def render(store_uri: str, s: dict, ttl: float = 180, color: bool = True) -> str:
    now = s["now"]
    p = _painter(color)
    L = []
    stamp = dt.datetime.fromtimestamp(now).strftime("%H:%M:%S")
    L.append(p("  Splendor 学習状況  ", "bold", "cyan")
             + "  " + p(f"{store_uri} · {stamp}", "dim"))
    L.append(p("─" * 60, "dim"))

    # --- current model ---
    m = s["manifest"]
    cur = s["current"] or "—"
    info = []
    if m:
        if m.get("logical_generation") is not None:
            info.append(f"世代{m['logical_generation']}")
        if m.get("parent"):
            info.append(f"← {m['parent']}")
        if m.get("ploss") is not None:
            info.append(f"loss p{m['ploss']}/v{m.get('vloss', '?')}")
        if m.get("eval_winrate") is not None:
            info.append(f"勝率{m['eval_winrate']}")
        if m.get("source"):
            info.append(str(m["source"]))
    tail = ("  " + p(" · ".join(info), "dim")) if info else ""
    L.append("  " + p("現役モデル", "bold") + "   " + p(cur, "green", "bold") + tail)
    L.append("  " + p("モデル数", "bold") + f"     {len(s['models'])} "
             + p(f"({', '.join(s['models'][-6:])})", "dim"))
    L.append("")

    # --- live summary ---
    hbs = s.get("heartbeats", [])

    def is_live(h):
        return (now - h.get("ts", 0)) <= ttl and h.get("state") != "stopped"

    live = sum(1 for h in hbs if is_live(h))
    live_c = "green" if live > 0 else "yellow"
    L.append("  " + p("稼働ワーカー", "bold") + " " + p(str(live), live_c, "bold")
             + " 台 稼働中 " + p(f"/ 既知 {len(hbs)}", "dim"))

    rate = s["last_hour"]
    gph = round(rate * s["games"] / s["sampled"]) if s["sampled"] else 0
    rate_c = "green" if rate > 0 else "yellow"
    L.append("  " + p("生成ペース", "bold") + "   " + p(f"{rate:,}", rate_c, "bold")
             + " ファイル/時 " + p(f"(≈ {gph:,} 局/時)", "dim"))
    L.append("  " + p("学習データ", "bold") + "   "
             + p(f"{s['games']:,}", "bold") + " 局 / "
             + p(f"{s['positions']:,}", "bold") + " 局面 "
             + p(f"(直近 {s['sampled']} ファイル)", "dim"))
    newest = _ago(now - s["newest_ts"]) if s["newest_ts"] else "—"
    L.append("  " + p("最新データ", "bold") + f"   {newest}   "
             + p("累計ファイル", "dim") + p(f" {s['shards']:,}", "dim"))
    if s["by_model"]:
        bym = " · ".join(f"{k}:{v:,}" for k, v in sorted(s["by_model"].items()))
        L.append("  " + p(f"モデル別局面  {bym}", "dim"))
    if s["promotes"]:
        pr = " ← ".join(f"{v}(世代{g})" for g, v in s["promotes"])
        L.append("  " + p(f"昇格履歴  {pr}", "dim"))

    # --- per-worker ---
    L.append("")
    L.append("  " + p("ワーカー稼働状況", "bold")
             + p(f"   (● 稼働中 ≤{int(ttl)}s / ○ 停止・無応答)", "dim"))
    for h in hbs:
        age = now - h.get("ts", 0)
        liveh = is_live(h)
        mark = p("●", "green") if liveh else p("○", "gray")
        ja, col = _STATE_JA.get(h.get("state", "?"), (h.get("state", "?"), "yellow"))
        st = p(ja, col) if (liveh or h.get("state") == "stopped") else p(ja, "dim")
        name = p(h.get("worker", "?").ljust(18), "bold" if liveh else "dim")
        mv = p(str(h.get("model_version") or "-"), "dim")
        detail = p(h.get("detail", ""), "dim")
        L.append(f"  {mark} {name} {st}  {mv}  {p(_ago(age), 'dim')}  {detail}")
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
    p.add_argument("--no-color", action="store_true", help="disable ANSI colours")
    args = p.parse_args(argv)

    use_color = (not args.no_color and sys.stdout.isatty()
                 and os.environ.get("NO_COLOR") is None)
    store_uri = resolve_store_uri(args.store)
    store = open_store(store_uri)
    while True:
        s = collect(store, args.recent)
        out = render(store_uri, s, ttl=args.ttl, color=use_color)
        if args.watch > 0:
            print("\033[2J\033[H", end="")  # clear screen for a live view
        print(out, flush=True)
        if args.watch <= 0:
            break
        time.sleep(args.watch)


if __name__ == "__main__":
    main()
