#!/usr/bin/env bash
# Launch N parallel Splendor self-play generators (append-only data lake).
#
# Each generator is single-threaded (we pin torch to 1 thread), so parallelism
# comes from running many processes — one per core. They append shards
# independently (no coordination needed). Ctrl-C stops them all gracefully, so
# each writes a "stopped" heartbeat (visible in `splendor.viz.status`).
#
# Usage:
#   scripts/run-generators.sh [N]
#
#   N   number of parallel generators (default: $GEN_N, else nproc-4, min 1)
#
# Env overrides:
#   GEN_GAMES         games per shard          (default 16)
#   GEN_SIMS          IS-MCTS sims per move     (default 64)
#   GEN_WORKER_PREFIX heartbeat worker prefix   (default: short hostname)
#   GEN_LOGDIR        per-process log dir       (default: ~/splendor-gen-logs)
#   SPLENDOR_STORE    gs://bucket/prefix or dir (else the code's DEFAULT_STORE)
#   GOOGLE_CLOUD_PROJECT  GCP project for the GCS store (if using gs://)
#
# Monitor while running:  python -m splendor.viz.status --watch 10
set -uo pipefail

# ---- resolve paths (works from anywhere; backend is ../backend of this script)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND="$(cd "$SCRIPT_DIR/.." && pwd)/backend"
if [ ! -d "$BACKEND/splendor" ]; then
  echo "error: backend/splendor not found under $BACKEND" >&2
  exit 1
fi
PY="$BACKEND/.venv/bin/python"
[ -x "$PY" ] || PY="python"

# ---- config
default_n="$(( $(nproc 2>/dev/null || echo 4) - 4 ))"
[ "$default_n" -ge 1 ] 2>/dev/null || default_n=1
N="${1:-${GEN_N:-$default_n}}"
GAMES="${GEN_GAMES:-16}"
SIMS="${GEN_SIMS:-64}"
WORKER_PREFIX="${GEN_WORKER_PREFIX:-$(hostname -s 2>/dev/null || hostname)}"
LOGDIR="${GEN_LOGDIR:-$HOME/splendor-gen-logs}"
mkdir -p "$LOGDIR"

cd "$BACKEND"

pids=()
cleanup() {
  echo
  echo "[run-generators] stopping ${#pids[@]} generators (SIGTERM -> 'stopped' heartbeat)..."
  kill -TERM "${pids[@]}" 2>/dev/null || true
  wait 2>/dev/null || true
  echo "[run-generators] all stopped."
}
trap cleanup INT TERM

echo "[run-generators] N=$N games=$GAMES sims=$SIMS prefix=$WORKER_PREFIX"
echo "[run-generators] store=${SPLENDOR_STORE:-<DEFAULT_STORE>}  logs=$LOGDIR"
for i in $(seq 0 $((N - 1))); do
  OMP_NUM_THREADS=1 "$PY" -m splendor.gen.generate --loop \
    --games "$GAMES" --sims "$SIMS" --worker "${WORKER_PREFIX}-$i" \
    > "$LOGDIR/gen-$i.log" 2>&1 &
  pids+=($!)
  echo "  started ${WORKER_PREFIX}-$i  pid=$!  log=$LOGDIR/gen-$i.log"
done

echo "[run-generators] running. Ctrl-C to stop all.  Monitor: $PY -m splendor.viz.status --watch 10"
wait
