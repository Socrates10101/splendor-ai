# Splendor Trainer（日本語）

> **English README is the maintained one:** [README.md](README.md).
> このファイルは開発時の日本語メモを残したものです。`train/local.py`・`gen/selfplay.py`・
> `/training` 画面に関する記述は、削除済みの旧・単機学習ループのものです（現行の構成は
> [`docs/CODEMAP.md`](docs/CODEMAP.md) を参照）。セットアップは `uv sync --extra dev` を推奨します。

ローカルで動く Splendor の AI 対戦 + **感想戦**（各手の勝率推移・最善手・悪手指摘）+ 局面解析ツール。
Splendor が上手くなるための学習環境です。

## 構成

```
backend/splendor/      # フォルダ＝工程（全体地図は docs/CODEMAP.md）
├─ engine/   不変エンジン（cards rules rng state moves notation encoding evaluator ismcts analyze）
├─ nn.py     共有: policy-value ネット + NNEvaluator
├─ store.py  共有: 追記専用データレイク + データ契約
├─ gen/      🟢 生成（generate.py / selfplay.py）
├─ train/    🟣 学習（store.py=現行 / local.py=旧UI用）
├─ eval/     🔵 検証（arena.py / match.py / worker.py）
└─ viz/      📊 可視化（status.py / curve.py）
backend/server.py + frontend/   🌐 対局UI(/) + 学習UI(/training)
```

> **コードの全体像・各ファイルの役割・どこに何を足すかは [`docs/CODEMAP.md`](docs/CODEMAP.md) に集約**（1枚地図）。下の「コード一覧（資産）」も参照。

AI は **AlphaZero 形（policy-value 評価器 + IS-MCTS）**。v1 は学習不要の
`HeuristicEvaluator` で動作し、後から同 interface の `NNEvaluator`（学習済み重み）に
差し替えるだけで強化できる設計です。

## コード一覧（資産）：生成 / 学習 / 検証

分散・データ駆動プラットフォーム（詳細は `docs/training-platform-design.md`）。
共通エンジンの上で、**生成・学習・検証**の3工程＋インフラに分かれる。すべて `--store`
（既定 `$SPLENDOR_STORE` → `store.DEFAULT_STORE`）でローカル/`gs://` 両対応。

| 工程 | モジュール | 役割 |
|---|---|---|
| **共通エンジン** | `cards/rules/rng/state/moves/notation/encoding/evaluator/ismcts/analyze` | 規則・状態・着手・seed+moves記法・45アクション/特徴・IS-MCTS・heuristic評価器（純stdlib・不変） |
| **共通NN** | `nn.py` | policy-valueネット + `NNEvaluator`（学習済み重みを同 interface で供給） |
| **共通ストア** | `store.py` | 追記専用データレイク（`LocalStore`/`GCSBackend`、`open_store`）＋データ契約＋best/promote/heartbeat |
| 🟢 **生成** | `gen/generate.py` | ステートレス自己対戦生成器。現役モデルを取得→棋譜+π+provenance を `shards/` へ追記（`-m splendor.gen.generate --loop`） |
| 🟢 **生成** | `gen/selfplay.py` | 並列leaf-batchingの自己対戦コア（`train/local.py`が使用） |
| 🟣 **学習** | `train/store.py` | ストア消費の学習器。直近データ→学習→`models/`へpublish(+promote)。`-m splendor.train.store --loop` |
| 🟣 **学習(旧)** | `train/local.py` | 自己完結のローカル学習ループ（生成も自前。UI(/training)が起動・`-m splendor.train.local`） |
| 🔵 **検証** | `eval/worker.py` | 評価ワーカー。未評価モデルを heuristic と対戦→`eval/<gen>.json` を記録（`-m splendor.eval.worker --loop`） |
| 🔵 **検証** | `eval/arena.py` | 2評価器の head-to-head 対戦ライブラリ（勝率/Elo） |
| 🔵 **検証** | `eval/match.py` | 対戦CLI。`current/genNNNN/heuristic/*.pt` を選んで対戦・世代sweep（`-m splendor.eval.match`） |
| 📊 **可視化** | `viz/status.py` | ストアの稼働状況CLI（色付き・日本語・heartbeat、`-m splendor.viz.status`） |
| 📊 **可視化** | `viz/curve.py` | 学習曲線CLI（manifestのloss＋`eval/`の勝率をスパークライン、`-m splendor.viz.curve`） |
| 🌐 **インフラ** | `server.py` + `frontend/` | 対局/感想戦UI（`/`）＋学習モニター/操作UI（`/training`） |
| 🛠 **スクリプト** | `scripts/run-generators.sh` | N並列の生成器をまとめて起動（1コア/プロセス） |

**運用の3プロセス（各マシンで `--loop`）**：
```bash
python -m splendor.gen.generate --loop      # 生成（ミニPC・CPU、scripts/run-generators.sh で並列）
python -m splendor.train.store --loop   # 学習（ラップトップ・GPU）
python -m splendor.eval.worker --loop      # 検証（任意のCPU。eval/ にログ→curveで曲線）
python -m splendor.viz.status --watch 10    # 監視
python -m splendor.viz.curve                # 学習曲線（loss + 勝率）
```

## セットアップ & 起動

```bash
cd backend
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
uvicorn server:app --reload --port 8000
```

ブラウザで http://127.0.0.1:8000/ を開く。

- **対局**: AIと対戦。カードをクリックで購入/予約、トークンをクリックで取得。
  「この局面を解析」でリアルタイム MCTS の最善手をライブ表示。
- **感想戦**: 終局後「感想戦へ →」。勝率推移グラフ・手リスト（悪手⚠マーク）・
  各局面の最善手と候補手を確認。

## テスト

```bash
cd backend && . .venv/bin/activate && pytest
```

## 記法（棋譜）

```
T:wug   3色取得（白青緑）        R:L2#3  Lv2スロット3を予約
T:rr    同色2枚取得（赤）        R:L1^   Lv1山札トップを予約
P:L3#1  Lv3スロット1を購入       P:H#2   予約カード#2を購入
/D:rk   トークン超過の破棄        /N:N3   貴族N3を獲得
```

棋譜は `seed + 着手列` で完全再現でき、解析結果は各手の `eval` として後付けされます。

## 学習（self-play / AlphaZero）

GPU（Apple Silicon は MPS バックエンド）で自己対戦学習を回し、`NNEvaluator` 用の
チェックポイントを作る。多数局を並列に進めて各シミュレーションのリーフ評価を1回の
バッチ推論にまとめるため（leaf-batching）、小さなネットでも GPU を継続的に使える。

### いちばん簡単：UI から開始（引数の知識不要）

```bash
cd backend && . .venv/bin/activate
pip install -r requirements.txt              # torch / numpy を含む
uvicorn server:app --port 8000
# → http://127.0.0.1:8000/training を開く
```

`/training` の **操作パネル**でプリセット（お試し/標準/大きめ実験）を選び「▶ 学習開始」を
押すだけ。サーバが `python -m splendor.train.local …` を子プロセスとして起動し、同じ画面で
進捗・損失・勝率・崩壊シグナル・ログをライブ表示する。「■ 停止」で現在のイテレーションを
締めてから安全に終了。run ごとに `checkpoints_runs/<run名>/` に保存される。

- **プリセット**は全項目を明示（切替時に前の設定が残らない）。「標準」＝実証済み構成
  （256/4・定数lr・エントロピー無）＋崩壊対策(B1)。
- **起点チェックポイント（warm-start）**：ドロップダウンで既存の `best.pt` 等を選ぶと、
  その重みから継続学習する（最強を捨てずに伸ばす）。アーキはチェックポイントに合わせる。
- **チェックポイント一覧**：run ごとにグループ表示（タグ★best/peak/latest/hourly・iter・勝率・
  サイズ・更新）。`best.pt` は最高の平滑勝率、`best_iNNNN.pt` は近ピーク上位K個の控え。
- **アリーナ比較**：候補2つ（または heuristic）を対戦させ、**ノイズの少ない勝率で本当の最強を選ぶ**。
  学習中の評価は局数が少なくブレるので、最終判断はここで行う。

### 「best」の扱い — なぜ複数残すのか

自己対戦は常に **最新ネット**で生成する（best 固定ではない）ので、**一度下がってから大きく上がる**
軌道も起こり得る（実際 run1 はそれで強くなった）。`best.pt` は後からより高いピークが来れば
更新されるので最高点は失われないが、評価勝率にはノイズがあるため、**近ピークの上位K個
（`best_iNNNN.pt`）も残し、アリーナ比較で確定**する。`--keep-best`（既定5）で控え数を調整。
best-gated self-play（best でしか生成しない）は dip→soar を潰すため**採用していない**。

### CLI から（細かく制御したい場合）

```bash
cd backend && . .venv/bin/activate
pip install -r requirements.txt              # torch / numpy を含む

# 推奨: 自己対戦を CPU 全コアに分散（--workers）、GPU は学習に使う。
# --hours で時間を区切ると、その時点で安全に締めて自動終了する。
# 1時間ごとに hourly_NNN.pt、毎イテレーション latest.pt、評価更新ごとに best.pt を保存。
python -m splendor.train.local --device mps --workers 8 \
    --games-per-iter 256 --sims 96 --eval-every 10 --hours 6

python -m splendor.train.local --resume            # latest.pt から再開（best/経過時間も復元）
```

チェックポイントの役割：

- **`latest.pt`** — 毎イテレーション上書き。`--resume` の対象。いつ止めても直近が残る。
- **`best.pt`** — 評価勝率が更新されるたびに保存。**6時間後はこれが最強の重み**（latest が一時的に退化しても安全）。
- **`hourly_NNN.pt`** — `--ckpt-interval-min`（既定60分）ごとのスナップショット。

> `--workers` がボトルネック解消の本命：IS-MCTS 木探索は Python・シングルスレッドで
> CPU律速なので、ワーカー数を物理コア数程度まで上げると生成スループットが数倍になる。
> `--workers 1`（既定）はワーカーを使わず本体プロセスで生成する。

主なノブ：

| flag | 既定 | 意味 |
|---|---|---|
| `--games-per-iter` | 128 | 並列自己対戦数 = リーフのバッチ幅（上げると GPU 稼働↑） |
| `--sims` | 80 | 1手あたり IS-MCTS シミュレーション数（強さ↑／生成は遅く） |
| `--train-steps` / `--batch-size` | 200 / 512 | 1イテレーションの勾配ステップ数とミニバッチ |
| `--buffer` | 200000 | リプレイバッファ容量（サンプル数） |
| `--lr-schedule` | cosine | `cosine`=warmup→cosine減衰。序盤の急吸い込みを抑え終盤に細かく収束。`constant`も可 |
| `--lr-warmup-iters` / `--lr-final-frac` | 20 / 0.1 | warmup区間と減衰の下限（peakの何割まで） |
| `--temp` / `--temp-moves` | 1.0 / 16 | 序盤の着手温度 `N^(1/τ)` と確率着手する手数（探索多様性＝崩壊対策） |
| `--entropy-coef` | 0.001 | policyエントロピー正則化。尖りすぎ(崩壊方向)を抑える |
| `--value-coef` | 1.0 | value MSE項の重み |
| `--width` / `--blocks` | 256 / 4 | ネット幅・残差ブロック数（容量と GPU 負荷） |
| `--workers` | 1 | 自己対戦ワーカープロセス数（CPUコアに分散。生成スループットの本命） |
| `--grad-clip` | 1.0 | 勾配ノルム上限（0=無効）。発散の保険 |
| `--eval-every` | 0 | Nイテレーションごとに arena 評価（0=無効） |
| `--eval-games` / `--eval-sims` | 40 / 60 | 評価1回の対局数・1手sims（多いほど勝率のノイズ減） |
| `--eval-ema-alpha` | 0.5 | best.pt 選定に使う勝率EMAの最新重み |
| `--eval-opponent` | heuristic | 進捗を測る固定相手（`heuristic` か チェックポイントpath） |
| `--ckpt-interval-min` | 60 | スナップショット間隔（分） |
| `--seed` | 12345 | 乱数シード。weight初期化・ミニバッチ・自己対戦を固定し**再現可能** |

> **第1回ランの反省を反映済み**：①評価は毎回**同じ初期配置(seed固定)**で行い、勝率曲線が
> 「引き運」でなく棋力を映す ②best.pt は単発のラッキー勝率でなく**EMA(平滑値)で選定**
> （前回 20局の偽の80%で best を選んでしまった反省）③`--seed` で全乱数を固定し再現可能
> ④勾配クリッピングで発散を予防 ⑤学習設定(args)を checkpoint に保存し後から追える。

### 学習の進捗を見る

進捗は loss ではなく**固定相手との勝率**で測る。学習中は `--eval-every` で
`checkpoints/metrics.jsonl` に毎イテレーションのロスと評価勝率が追記される。

```bash
# 進捗テーブル（勝率の推移をスパークライン付きで表示）
python -m splendor.eval.arena --report checkpoints/metrics.jsonl

# 任意の2チェックポイント / vs heuristic を直接対戦させて強さを比較
python -m splendor.eval.arena --a checkpoints/hourly_005.pt --b heuristic --games 40
python -m splendor.eval.arena --a checkpoints/latest.pt --b checkpoints/hourly_001.pt
```

ランダム初期ネットは heuristic に大きく負ける（勝率〜0%）。学習が進むと
この勝率が上がっていくのが「どれだけ強くなったか」の一次指標。

### 学習モニター（Web view・対局UIとは別画面）

学習プロセスとは別にサーバを立てると、`/training` で進捗をライブ表示できる
（`checkpoints/` の `status.json` / `metrics.jsonl` を読むだけなので、両者は疎結合）。

```bash
# ターミナル1: 学習（ファイルを書き出す）
python -m splendor.train.local --device mps --workers 8 --eval-every 10 --hours 6

# ターミナル2: サーバ（読み取って表示）。学習を別ディレクトリに出した場合は環境変数で指定
uvicorn server:app --port 8000          # SPLENDOR_CKPT_DIR=path で参照先を変更可
# → http://127.0.0.1:8000/training を開く
```

表示内容：稼働状態（稼働中/停止）・開始時刻・経過・**終了予定と残り時間**・
イテレーション・生成速度(smp/s)・最高勝率(best.pt)、損失曲線、vs-heuristic 勝率曲線、
**崩壊検知（未決着率・引き分け率・policyエントロピー）**、チェックポイント一覧。
対局画面（`/`）とは完全に独立。

### value 崩壊と崩壊検知

純粋な self-play は「決めにいかず引き分けに逃げる」方策に収束し、value のターゲット
が z=0 ばかりになって `vloss→0` で崩壊することがある（実測で約2時間その状態に陥った）。
対策として **上限切れ局は z=0 にせず点数で決着**させ（15点に近い方を勝ち）、常に
「前進する」勾配を残している。崩壊の早期検知のため次を毎イテレーション記録する：

| 指標 | 計算 | 崩壊シグナル |
|---|---|---|
| `capped_rate` | 自然決着しなかった（上限切れ）局の割合 | 上昇しっぱなしは要注意 |
| `draw_rate` | 点数でも決着しない真の引き分け割合 | 上昇＝value 信号が痩せる |
| `policy_entropy` | MCTS訪問分布 π の平均エントロピー `−Σπlogπ` | 低下し続け＝方策が尖りすぎ |
| `p0_winrate` | 先手(player0)の勝率 | 0/1 へ偏ると自己対戦が単調化 |

`vloss` が 0 に張り付く・`capped_rate` が高止まり・`policy_entropy` が下がり続ける、の
3点が揃ったら崩壊。健全帯は `vloss≈0.5–0.7` / `ploss≈2.2`（実戦は結果が不確実なので
ある程度の損失が“正常”）。

学習済みチェックポイントは既存の探索・感想戦にそのまま差し込める：

```python
from splendor.nn import NNEvaluator
from splendor.ismcts import choose_move
ev = NNEvaluator.from_checkpoint("checkpoints/latest.pt", device="mps")
move = choose_move(state, sims=200, evaluator=ev)   # IS-MCTS は無改造
```

> ボトルネックは Python の MCTS 木探索（CPU）側。GPU 稼働率をさらに上げたいときは
> `--games-per-iter` と `--width` を増やす。棋力優先なら `--sims` を増やす。

## 分散・データ駆動プラットフォーム（generate/train 分離・クラウド）

設計は `docs/training-platform-design.md`。生成と学習を**追記専用データストア**を介して疎結合にし、
常時稼働のCPU機（ミニPC）で生成、GPU機（ラップトップ）で学習、という協調を可能にする。

```bash
# データストアはローカルパス or gs://bucket/prefix（同じコマンドで両対応）
# 生成器（ステートレス・常時）: 現役モデルを拾い自己対戦→shard追記。無ければheuristicでbootstrap
python -m splendor.gen.generate --store gs://my-bucket/splendor --loop --games 16 --sims 64

# 学習器（GPU）: 直近データ(positions単位)を読み学習→model push→--promoteで現役更新
python -m splendor.train.store --store gs://my-bucket/splendor --steps 200 --warm-start --promote

# 状況確認（ヘッドレス/SSH向けCLIビュー。ローカルでも gs:// でも同じ）
python -m splendor.viz.status --store gs://my-bucket/splendor            # 一回表示
python -m splendor.viz.status --store gs://my-bucket/splendor --watch 10 # 10秒ごと更新
```

`status` は現役モデル・世代・データ量(positions)・生成器別・直近1hの生成レート・promote履歴を
torch不要で表示（シャードキーから安く集計し、詳細は直近 `--recent N` シャードだけ読む）。

`--store` は3コマンド共通で省略可。既定は `$SPLENDOR_STORE`、無ければコード内の
`DEFAULT_STORE`（上書き可能なハードコード）。`export SPLENDOR_STORE=…` か `--store` で上書き可。

**稼働状況（ライブ）**：generate/train_store は1サイクルごとに**生存情報(heartbeat)**を
ストアの `_live/<host>.json` に上書き（決定には一切使わない観測専用テレメトリ＝不変リネージュ
data/models/promote とは隔離）。`status` がTTL内のものを「● live」で表示する。ストアがローカル
なら生存情報もローカル、`gs://` なら**他環境（ミニPC等）の稼働も手元から見える**。
**終了時（Ctrl-C / systemd SIGTERM / 正常終了）は `state=stopped` を書き込む**ので、TTLを待たず
即「○ stopped」になる。
```
workers : 1 live / 1 known  (live = seen ≤180s)
  ● minipc   generate idle  gen0002  3s ago  16g/800pos 41s -> …
```

- **データ契約**: 棋譜 `seed+moves+result` ＋ 各手の密な方策ターゲット `visit_counts[45]+legal_mask[45]` ＋
  厚い provenance。特徴と z は学習側で**再生成**（`store.game_to_samples`）。
- **追記専用・状態はストア**: 現役モデルは promote イベントを `logical_generation` で畳み込んで確定（壁時計に依存しない）。
- **バックエンド差し替え**: `StorageBackend`（`LocalStore` / `GCSBackend`、`open_store(uri)` で分岐）。
  GCS はネイティブSDK（`pip install google-cloud-storage`、`gcloud auth application-default login`
  か サービスアカウントJSON）。`gs://` を使う時だけ依存が要る。

## ロードマップ

- v1（実装済み）: エンジン + ヒューリスティック IS-MCTS + 対局/感想戦 UI
- v2（実装済み）: self-play 学習（`nn.py`/`selfplay.py`/`train.py`）で `NNEvaluator` 用チェックポイント生成 + UI(`/training`) + arena
- v3（実装中）: 分散・データ駆動基盤（Phase 1 generate/train分離+追記ストア、Phase 2 GCSバックエンド）。次: プラグイン化(Generator/Learner/Sampler/Gating)、sampler allowlist/鮮度重み、gating統計
