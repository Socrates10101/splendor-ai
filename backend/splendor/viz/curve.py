"""Print the training learning curve from the store's logs (cheap, no arena).

  * loss (ploss / vloss) comes from each generation's model manifest, written
    by the trainer every generation;
  * strength (winrate vs heuristic) comes from the evaluator worker's
    ``eval/<version>.json`` records.

    python -m splendor.curve
    python -m splendor.curve --no-color
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from splendor.store import open_store, resolve_store_uri

_SPARK = "▁▂▃▄▅▆▇█"


def _spark(vals, lo=None, hi=None) -> str:
    xs = [v for v in vals if v is not None]
    if not xs:
        return ""
    lo = min(xs) if lo is None else lo
    hi = max(xs) if hi is None else hi
    if hi <= lo:
        hi = lo + 1e-9
    out = []
    for v in vals:
        if v is None:
            out.append(" ")
        else:
            out.append(_SPARK[min(len(_SPARK) - 1, int((v - lo) / (hi - lo) * (len(_SPARK) - 1)))])
    return "".join(out)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Training learning curve from store logs")
    p.add_argument("--store", default=None,
                   help="local dir or gs://bucket/prefix (default: $SPLENDOR_STORE)")
    p.add_argument("--rows", type=int, default=30, help="max table rows")
    p.add_argument("--no-color", action="store_true")
    args = p.parse_args(argv)
    color = (not args.no_color and sys.stdout.isatty()
             and os.environ.get("NO_COLOR") is None)

    def paint(s, c):
        return f"\033[{c}m{s}\033[0m" if color else str(s)

    store = open_store(resolve_store_uri(args.store))

    evals = {}
    for k in store.list_keys("eval/"):
        if k.endswith(".json"):
            try:
                e = json.loads(store.get_bytes(k))
                evals[e.get("version")] = e
            except (json.JSONDecodeError, OSError):
                pass

    rows = []
    for ver in store.list_models():
        m = store.read_manifest(ver) or {}
        e = evals.get(ver, {})
        rows.append({"ver": ver, "gen": m.get("logical_generation") or 0,
                     "ploss": m.get("ploss"), "vloss": m.get("vloss"),
                     "wr": e.get("winrate")})
    rows.sort(key=lambda r: r["gen"])
    if not rows:
        print("モデルがありません。"); return

    pl = [r["ploss"] for r in rows]
    vl = [r["vloss"] for r in rows]
    wr = [r["wr"] for r in rows]
    n_eval = sum(1 for x in wr if x is not None)

    cur = store.current_model()
    best_wr = max(((r["wr"], r["ver"]) for r in rows if r["wr"] is not None),
                  default=(None, None))

    print(paint("  学習曲線", "1", ) + paint(f"   {len(rows)}世代  評価済み{n_eval}  現役={cur}", "2"))
    print(paint("─" * 60, "2"))
    if any(x is not None for x in pl):
        print("  ploss : " + paint(_spark(pl), "36") + paint(f"   {pl[0]} → {pl[-1]}", "2"))
    if any(x is not None for x in vl):
        print("  vloss : " + paint(_spark(vl), "35") + paint(f"   {vl[0]} → {vl[-1]}", "2"))
    if n_eval:
        print("  勝率  : " + paint(_spark(wr, lo=0.0, hi=1.0), "32")
              + paint(f"   最高 {best_wr[0]:.0%}@{best_wr[1]}", "33"))
    else:
        print(paint("  勝率  : (未評価。`python -m splendor.evaluate --loop` で記録)", "2"))

    # sampled table
    step = max(1, len(rows) // max(1, args.rows))
    print(paint(f"\n  {'世代':>4}  {'ploss':>6}  {'vloss':>6}  {'vs heuristic':>12}", "2"))
    for r in rows[::step] + ([rows[-1]] if (len(rows) - 1) % step else []):
        wrs = (paint(f"{r['wr']:.0%}", "32" if r['wr'] and r['wr'] > 0.5 else "31")
               if r["wr"] is not None else paint("—", "2"))
        pls = f"{r['ploss']:.3f}" if r["ploss"] is not None else "—"
        vls = f"{r['vloss']:.3f}" if r["vloss"] is not None else "—"
        print(f"  {r['gen']:>4}  {pls:>6}  {vls:>6}  {wrs:>12}")


if __name__ == "__main__":
    main()
