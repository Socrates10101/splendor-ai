# Splendor Trainer

*日本語の開発メモは [README.ja.md](README.ja.md)。*

A local trainer for the board game Splendor. Play against an AlphaZero-style AI
(policy-value evaluator + information-set MCTS) in your browser, then review the
game move by move: win-rate curve, best move at every position, blunder markers.
The whole engine is pure Python with no third-party dependencies; the optional
self-play / training pipeline adds PyTorch.

## Quick start

Requires Python 3.10+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/Socrates10101/splendor-ai.git
cd splendor-ai/backend
uv sync --extra serve          # web server only (no torch)
uv run uvicorn server:app --port 8000
```

Open <http://127.0.0.1:8000/> and play. You are player 1; the AI is the
built-in `HeuristicEvaluator` driven by IS-MCTS, so no model download or
training is needed. Click a card to buy or reserve it, click tokens to take
them. The UI text is currently Japanese: 新規対局 = new game, この局面を解析 =
analyze this position (live search on the current position), 感想戦へ =
open the post-game review once the game ends.

## Features

- **Play**: 2-player Splendor versus the AI with full rules (token limits,
  reserving from the deck, nobles, end-of-round scoring).
- **Post-game review**: win-rate curve over the whole game, a move list with
  blunder markers, and for every position the engine's best move and
  candidate moves with their visit counts.
- **Position analysis**: run the search on the current position mid-game, or
  import a game record and analyze it.
- **Compact game notation**: a game is `seed + move list`, fully reproducible.

  ```
  T:wug   take 3 tokens (white, blue, green)    R:L2#3  reserve level-2 card in slot 3
  T:rr    take 2 of one color (red)             R:L1^   reserve from the level-1 deck
  P:L3#1  buy level-3 card in slot 1            P:H#2   buy reserved card #2
  /D:rk   discard tokens over the limit         /N:N3   receive noble N3
  ```

- **AI**: AlphaZero-shaped. The search (IS-MCTS over hidden deck order) talks
  to an `Evaluator` interface; `HeuristicEvaluator` ships and works
  out of the box, `NNEvaluator` wraps a trained policy-value net with the same
  interface.

## Training pipeline

Training is optional and split into three stateless stages that communicate
only through an append-only data store (local directory or `gs://` bucket).
All commands run from `backend/`.

```bash
uv sync --extra dev                       # torch + numpy + GCS client + server + tests
```

| Stage | Command | What it does |
|---|---|---|
| gen | `uv run python -m splendor.gen.generate --loop` | Self-play with the current model (or the heuristic when no model exists yet) and append game shards to `shards/`. `scripts/run-generators.sh N` runs N of these in parallel. |
| train | `uv run python -m splendor.train.store --loop --promote` | Pull recent positions, train, publish a new generation to `models/`, and promote it to "current". |
| eval | `uv run python -m splendor.eval.worker --loop` | Play every unevaluated generation against the heuristic and record win rates in `eval/`. |
| viz | `uv run python -m splendor.viz.status --watch 10` | Live dashboard: current model, data volume, generation rate, worker heartbeats. |
| viz | `uv run python -m splendor.viz.curve` | Loss and win-rate curves per generation as terminal sparklines. |

One-off comparisons:

```bash
uv run python -m splendor.eval.match --a current --b heuristic --games 80
uv run python -m splendor.eval.match --sweep --every 2        # several generations vs the heuristic
uv run python -m splendor.eval.arena --a path/to/a.pt --b heuristic --games 40   # raw checkpoints
```

Every command takes `--store <dir|gs://bucket/prefix>`. When omitted it uses
`$SPLENDOR_STORE`, and failing that the local directory `data/` (relative to
where you run the command). For a shared `gs://` store install the `gcs`
extra and authenticate with `gcloud auth application-default login`.

Design notes for the pipeline live in
[docs/training-platform-design.md](docs/training-platform-design.md) (Japanese).

## Models and data

**No trained model ships with this repository.** The play and review UI uses
`HeuristicEvaluator` exclusively, so it is fully functional without training.
If you run the pipeline, everything it produces lands in the store:

```
<store>/
├─ shards/            self-play game records + policy targets (append-only)
├─ models/<gen>/      weights.pt + manifest.json (loss, lr, positions, parent)
├─ events/promote/    which generation is "current"
├─ eval/<gen>.json    win rate vs the heuristic
└─ _live/             worker heartbeats (telemetry only)
```

`data/`, `data_*/`, `checkpoints*/`, `*.pt` and `*.log` are gitignored, so a
local store never ends up in a commit.

To use a trained model from your own code:

```python
from splendor.nn import NNEvaluator
from splendor.engine.ismcts import choose_move

ev = NNEvaluator.from_checkpoint("data/models/gen0003/weights.pt", device="cpu")
move = choose_move(state, sims=200, evaluator=ev)
```

## Project layout

```
backend/splendor/engine/   rules, state, moves, notation, encoding, evaluator, IS-MCTS, analysis (stdlib only)
backend/splendor/nn.py     policy-value network + NNEvaluator (torch)
backend/splendor/store.py  append-only data store (local / GCS) and data contract
backend/splendor/gen|train|eval|viz/   the pipeline stages above
backend/server.py          FastAPI app: game API + static frontend
frontend/                  vanilla JS play + review UI
scripts/run-generators.sh  launch N parallel generators
docs/                      code map and design notes
```

[docs/CODEMAP.md](docs/CODEMAP.md) explains each folder, the data flow, and
where new code should go.

## Development

```bash
cd backend
uv sync --extra dev
uv run pytest
```

The test suite covers the engine, notation, store, training step, IS-MCTS and
the HTTP API. The engine stays dependency-free; keep torch and cloud imports
inside `nn.py`, `store.py` and the pipeline stages.

## Disclaimer

Splendor is a trademark of Space Cowboys. This is an unofficial fan project,
not affiliated with or endorsed by the publisher, and it contains no
copyrighted artwork or assets from the game.

## License

[MIT](LICENSE)
