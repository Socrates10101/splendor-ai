# CODEMAP — このコードに何があり、どう増えるか

> 迷ったらまずここを読む。**フォルダ＝工程**。新しいコードは必ずどれかの工程に属す。

## フォルダ構造（＝設計そのもの）

```
backend/splendor/
├─ engine/      不変エンジン（純stdlib・依存ゼロ・絶対に壊さない土台）
│   cards rules rng state moves notation encoding evaluator ismcts analyze
├─ nn.py        共有：policy-value ネット + NNEvaluator（torch）
├─ store.py     共有：追記専用データレイク + データ契約（LocalStore / GCSBackend）
│
├─ gen/         🟢 生成：自己対戦で対戦データを作る
│   generate.py  ステートレス生成器CLI（現役モデル→棋譜+π→ shards/ へ追記）
│
├─ train/       🟣 学習：データから新しいモデルを作る
│   store.py     ストア消費の学習器（直近データ→学習→models/へpublish+promote, --loop）
│                毎世代 manifest にロス等を記録 ＝ 学習ログ（後からUIで可視化可能）
│
├─ eval/        🔵 検証：モデルの強さを測る
│   arena.py     2評価器の head-to-head 対戦ライブラリ（勝率/Elo）
│   match.py     対戦CLI（current/genNNNN/heuristic/*.pt を選んで対戦・世代sweep）
│   worker.py    ★検証ワーカー（未評価モデルをheuristicと対戦→ eval/ に記録, --loop）
│
└─ viz/         📊 可視化：状況をCLIで見る
    status.py    稼働状況（現役モデル・ワーカー・生成ペース、色付き日本語）
    curve.py     学習曲線（manifestのloss + eval/の勝率をスパークライン）

backend/server.py + frontend/   🌐 インフラ：対局UI(/) + 感想戦（旧・学習操作UIは廃止）
scripts/run-generators.sh        🛠 N並列の生成器をまとめて起動
docs/                            設計書（training-platform-design.md）+ この地図
```

> 旧の単機・モノリシック学習（`gen/selfplay.py` + `train/local.py` + server の `/training`
> 操作UI）は**削除**した。残るのは分散1本。

## 学習ログ（後からUIでロス推移等を見るための土台）

学習の各種指標は**コードでなく store に残る**ので、後から Web UI でも CLI でも可視化できる：
- **損失の推移** = `models/<gen>/manifest.json`（`ploss/vloss/lr/train_positions/replay_ratio/parent/created_ts`、毎世代）
- **強さの推移** = `eval/<gen>.json`（検証ワーカーが記録した勝率）
- 今すぐ見るなら `python -m splendor.viz.curve`。将来の Web 学習UI も同じデータを読むだけ。

## データの流れ（3工程はストア越しに繋がる）

```
 🟢 gen.generate ──push 棋譜──▶ ┌──────── store（GCS or ローカル）────────┐
   （ミニPC・CPU・並列）          │ shards/   対戦データ                     │
                                 │ models/   学習済みモデル + manifest(loss) │
 🟣 train.store ──pull/push──▶   │ events/promote/  現役ポインタ            │
   （ラップトップ・GPU）          │ eval/     検証結果（勝率）                │
                                 │ _live/    各ワーカーの生存(heartbeat)     │
 🔵 eval.worker ──pull/push──▶   └──────────────┬──────────────────────────┘
   （任意のCPU）                                 │ read
                                  📊 viz.status / viz.curve（読むだけ）
```
- **state はコードでなく store にある**：現役モデル=promoteの畳み込み、進捗=manifest/eval。
- **エンジンと store の契約（データ形式）は不変**。可変なのは各工程の中身だけ。

## コマンド（フォルダ＝モジュールパス）

```bash
python -m splendor.gen.generate --loop     # 生成（scripts/run-generators.sh で並列）
python -m splendor.train.store  --loop     # 学習（現行）
python -m splendor.eval.worker  --loop     # 検証（eval/ に勝率ログ）
python -m splendor.eval.match --a current --b heuristic --games 80   # 単発の対戦確認
python -m splendor.viz.status   --watch 10 # 稼働状況
python -m splendor.viz.curve               # 学習曲線
uvicorn server:app --port 8000             # 対局UI(/) + 学習UI(/training)
```
`--store` は3工程共通で省略可（`$SPLENDOR_STORE` → `store.DEFAULT_STORE`）。

## どう増えるか（迷わない置き場所）

| やりたいこと | どこに足す |
|---|---|
| 新しい生成方策（best生成・混合・styled） | `gen/` ＋ `--model` セレクタ |
| 新しい学習法（gating昇格・蒸留・lineage） | `train/` |
| 新しい強さの測り方（並列大量対局・vs best） | `eval/` |
| 新しい見せ方（Web曲線・ダッシュボード） | `viz/` or `server.py`/`frontend/` |
| データ形式・保存先の変更 | `store.py`（契約は慎重に。version を上げる） |
| ゲーム規則・特徴量 | `engine/`（不変が原則。変えたら version を上げる） |

**鉄則**：エンジンと store契約は不変。増えるのは「工程の中身」だけ。だから散らからない。

## 不変 vs 可変（前回 net サイズがブレた反省）

- **不変（lineageに固定）**：net arch(width/blocks) ・各 *_version。継続学習では**モデルから自動継承**、run毎フラグでは変えない。
- **可変（run毎に選ぶ）**：どのモデルで生成/学習/検証するか（policy）、ケイデンス(steps/interval)。
- arch を変えるのは「新 lineage の明示開始」のときだけ。
