# Splendor Trainer

ローカルで動く Splendor の AI 対戦 + **感想戦**（各手の勝率推移・最善手・悪手指摘）+ 局面解析ツール。
Splendor が上手くなるための学習環境です。

## 構成

```
splendor/
├─ backend/                 # Python: ゲームエンジン(SSOT) + IS-MCTS AI + FastAPI
│  ├─ splendor/             # エンジン本体（下記）
│  ├─ server.py             # REST /api + WebSocket /ws、静的フロントを配信
│  └─ tests/                # pytest（エンジン/記法/AI/API）
└─ frontend/                # バニラJS（ビルド不要）: 対局UI + 感想戦ビューア
```

### バックエンド（`backend/splendor/`）

| ファイル | 役割 |
|---|---|
| `cards.py` | 90枚の development card + 10 貴族（安定ID付き） |
| `rules.py` | 定数（トークン配分・勝利点15・上限10 など） |
| `rng.py` | seed から決定論シャッフル（再現可能） |
| `state.py` | `GameState` / `PlayerState`、視点ごとの隠れ情報マスク |
| `moves.py` | `Move`・合法手生成・適用（捨て札/貴族/ラウンド終了判定込み） |
| `notation.py` | テキスト記法 ⇄ Move、棋譜(JSON)の save/load/replay |
| `encoding.py` | 正準アクションインデックス + 特徴量ベクトル（将来のNN用） |
| `evaluator.py` | `Evaluator` interface + `HeuristicEvaluator`（学習不要） |
| `ismcts.py` | IS-MCTS（determinize + PUCT、評価器は差替可能） |
| `analyze.py` | 棋譜の全手解析（勝率前後・最善手・delta=悪手度） |
| `nn.py` | policy-value ネット(PyTorch) + `NNEvaluator`（学習済み重みを同 interface で供給） |
| `selfplay.py` | 並列・lockstep の自己対戦。多数局のリーフを1回のバッチ推論にまとめる(leaf-batching) |
| `train.py` | 自己対戦→学習ループ。`latest.pt`常時更新＋1時間ごとスナップショット＋`--resume` |

AI は **AlphaZero 形（policy-value 評価器 + IS-MCTS）**。v1 は学習不要の
`HeuristicEvaluator` で動作し、後から同 interface の `NNEvaluator`（学習済み重み）に
差し替えるだけで強化できる設計です。

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
押すだけ。サーバが `python -m splendor.train …` を子プロセスとして起動し、同じ画面で
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
python -m splendor.train --device mps --workers 8 \
    --games-per-iter 256 --sims 96 --eval-every 10 --hours 6

python -m splendor.train --resume            # latest.pt から再開（best/経過時間も復元）
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
python -m splendor.arena --report checkpoints/metrics.jsonl

# 任意の2チェックポイント / vs heuristic を直接対戦させて強さを比較
python -m splendor.arena --a checkpoints/hourly_005.pt --b heuristic --games 40
python -m splendor.arena --a checkpoints/latest.pt --b checkpoints/hourly_001.pt
```

ランダム初期ネットは heuristic に大きく負ける（勝率〜0%）。学習が進むと
この勝率が上がっていくのが「どれだけ強くなったか」の一次指標。

### 学習モニター（Web view・対局UIとは別画面）

学習プロセスとは別にサーバを立てると、`/training` で進捗をライブ表示できる
（`checkpoints/` の `status.json` / `metrics.jsonl` を読むだけなので、両者は疎結合）。

```bash
# ターミナル1: 学習（ファイルを書き出す）
python -m splendor.train --device mps --workers 8 --eval-every 10 --hours 6

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
python -m splendor.generate --store gs://my-bucket/splendor --loop --games 16 --sims 64

# 学習器（GPU）: 直近データ(positions単位)を読み学習→model push→--promoteで現役更新
python -m splendor.train_store --store gs://my-bucket/splendor --steps 200 --warm-start --promote
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
