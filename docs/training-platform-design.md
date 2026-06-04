# Splendor 学習プラットフォーム 設計書（分散・データ駆動）

> 目的: 自己対戦データの生成と学習を疎結合にし、クラウドの追記専用データレイクを中心に、
> 複数環境（常時稼働のミニPC=CPU生成 / ラップトップ=GPU学習）で協調できる、
> データ駆動・ロジック差し替え自由な学習基盤を設計する。

---

## 0. 背景・現状

- 既存（すべてローカル単体）:
  - **不変コア**: 規則/状態/着手/RNG、`seed+moves` の再現可能な棋譜記法、45アクション空間＋126次元特徴、`Evaluator` interface、IS-MCTS、学習不要の `HeuristicEvaluator`(v1)。
  - **学習スタック**: `nn.py`(PolicyValueNet/NNEvaluator)、`selfplay.py`(並列leaf-batching自己対戦/B1点数決着/B9崩壊診断/温度)、`train.py`(リプレイバッファ・LRスケジュール・grad-clip・EMA best/top-K・hourly/latest・--hours/--resume/--init-from・workers並列)、`arena.py`、`server.py`+`/training`(起動/停止/監視/比較)。
- **計測事実**: CPU(自己対戦)が律速、GPUは余る。run1(256/4・定数lr)=vs heuristic 54%(途中崩壊→回復)、run2(事故的大ネット384/6＋盛りすぎ)=25%。
- **教訓**:
  1. **loss↓ ≠ 強い**（弱い自己対戦への過適合が起こる）。真の指標は arena 勝率のみ。
  2. **変数は一度に変えない**（run2の失敗）。
  3. **B1（上限切れ局を z=0 にせず点数決着）で value 崩壊を回避**、B9 で監視。
  4. **データを資産として扱う**べき（生成がCPU律速だから、生成物を貯めて再利用したい）。

---

## 1. 設計原則（憲法）

1. **Immutable append-only data lake** — クラウドは**追記のみ・書き換え/削除なし**（event-sourcing）。並列書き込みでも競合しない。
2. **Stateless triggered workers** — ワーカーは「発火時にクラウド状態を読む → 仕事する → push → 終了」。ローカルに永続状態を持たない。落ちても再発火で継続。
3. **State lives in the data layer** — 「現役モデル / データ量 / 進捗」の**唯一の真実はクラウドのカタログ＋レジストリ**。コードや特定マシンに状態を埋めない。これにより生成器が薄くなり、どの環境でも気軽に動く。
4. **Code via Git** — コードはGit配布。各ワーカーは `git pull` 後に実行。**commit SHA を provenance に刻む**（再現性）。
5. **Role at runtime / no master-slave** — 役割は実行時に**機材適性**で決める。少数ノードでは中央コーディネータ不要。協調はタスクの性質から導く（§7）。
6. **Invariant engine / variable strategy** — エンジンは不変、学習戦略はプラグインで可変（§10）。

---

## 2. 不変 vs 可変

| | 内容 | 扱い |
|---|---|---|
| **不変** | 規則/状態/着手/RNG、`seed+moves`記法、45アクション/126特徴エンコ、`Evaluator` interface、IS-MCTS、HeuristicEvaluator | **バージョン固定**し `engine_version` / `encoding_version` を provenance に記録 |
| **可変** | 生成方策、損失/LR/最適化、self-play/教師あり/蒸留、bootstrap、best選定、ネット構造 | **プラグイン**として config で選択 |

---

## 3. 全体アーキテクチャ（actor-learner ＋ クラウド・データレイク）

```
 ミニPC(常時・CPU)                    [クラウド：追記専用 SSOT]                ラップトップ(GPU)
 ┌───────────────┐   push shards   ┌──────────────────────────┐   pull recent  ┌───────────────┐
 │ generate loop │ ───────────────▶│ data/    : 棋譜+π+provenance │◀────────────── │ train loop    │
 │ 現役modelを取得 │                 │ models/  : checkpoint+manifest│ ──push model──▶│ 直近データで学習 │
 │ 自己対戦K局     │                 │ events/  : promote / claim    │                │ →model→promote │
 └───────────────┘                 │ catalog/ : 索引スナップショット │                └───────────────┘
        ▲                          └──────────────────────────┘                        │
        └──── 次バッチは "最新の promote(現役)" を読む ──────────────────────────────────┘
```

- **生成（actor）= ミニPC**: ステートレス。現役モデルで自己対戦を回し続け push。
- **学習（learner）= ラップトップ**: 直近データを pull → 学習 → model push → promote 追記。
- **非同期**: 互いに待たない。学習が publish した瞬間、生成が次バッチから拾う（フィードバックがクラウド越しに閉じる）。
- 将来は generator を増やすだけで水平スケール（stateless writer ＝ 競合しない）。

---

## 4. データ設計（スキーマ）

### 4.1 何を保存し、何を再生成するか
- **保存（一次データ）**:
  - **棋譜 (game record)**: `{game_id, seed, players, moves:[notation...], result:{winner, points[], plies}}` … 極小。エンジン不変なので **replay で全局面を再現可能**。
  - **各手の方策ターゲット π**: 探索の訪問分布（45アクション上）。**seed+moves からは復元不可**なので**保存必須**。
    **形式は密推奨**: `visit_counts[45]`（生の訪問数）＋ `legal_mask[45]`。
    ⚠ 疎保存は「未出現=0訪問」か「非合法」かが曖昧になり静かに汚染するため避ける（Codexレビュー指摘）。
  - **provenance（必須項目を厚く）**: `{generator_id, generator_kind, model_version, model_hash,
    mcts_config:{sims, c_puct, dirichlet_alpha, epsilon, temperature_schedule},
    action_space_version, policy_target_format_version, result_rule_version,
    encoding_version, engine_version, git_sha, worker_id, created_ts,
    player_to_move, ply, action_taken}`。
- **再生成（派生・保存不要）**:
  - **特徴量 features**: `encoding` から replay で生成（`encoding_version` を見て整合）。
  - **value ターゲット z**: `result` から手番視点で再計算（±1/0、**B1: 上限切れは点数決着**）。
    ⚠ ただし**生の `result` は厚めに残し**、z 生成ロジック（B1/同点/手番視点/人数前提）は
    `result_rule_version` で**バージョン管理**する。揺れると過去データの意味が静かに変わるため（レビュー指摘）。

> 利点: 一次データが極小（seed+moves+π）。エンコーディングや z 定義を変えても**作り直せる**（不変エンジンの恩恵）。

### 4.2 ストレージ・レイアウト（キーは一意・追記専用）
```
data/shards/<date>/<worker>_<runid>_<seq>_<gitsha>.parquet   # サンプル群（複数ゲーム）
models/<model_version>/weights.pt
models/<model_version>/manifest.json                          # {parent, config, git_sha, iter, created_ts, eval}
events/promote/<ts>_<model_version>.json                     # 現役ポインタ（最新を決定論で読む）
events/arena/<ts>_<a>_vs_<b>.json                            # 評価結果（追記）
events/claim/<ts>_<task>_<worker>.json                       # 任意：高価タスクの重複回避
catalog/manifest-<seq>.json                                  # 索引スナップショット（版を追記、最新を読む）
```
- **冪等キー**: `worker + runid + seq + gitsha` で再送・並走でも衝突しない。
- **現役モデル**: `current = events/promote を畳み込んで確定`。**書き換えずに追記でポインタ更新**。
  ⚠ **`ts`（壁時計）で順序付けない**（時計ずれで壊れる）。promote イベントに
  `{parent_model, candidate_model, arena_event_id, score, accepted_by, created_at, logical_generation}` を持たせ、
  読む側は **`logical_generation` 優先**で確定する（レビュー指摘）。
- **カタログ**: ログ全読みは重いので、**索引スナップショットを定期的に追記**（古い版は残す、最新を読む）。

---

## 5. StorageBackend interface（追記専用・差し替え可能）

```
put_object(key, bytes, if_absent=False) -> ok        # 一意キーへ追記。if_absent で put-if-absent(lease用)
get_object(key) -> bytes
list(prefix) -> [keys]
current_model() -> model_version | None              # events/promote の最新を決定論で
latest_manifest() -> catalog                         # catalog/manifest-* の最新
```
- 実装: **LocalFS**(Phase1) → **S3互換**(Phase2: AWS S3 / Cloudflare R2 / Backblaze B2 / 自前 MinIO)。
- S3 は new object に **read-after-write 一貫性**あり。**conditional write (put-if-absent / If-None-Match)** で専用ロックサービス無しに lease を実装可能（§7で必要時のみ）。

---

## 6. ワーカー（3種・すべてステートレス）

### 6.1 generate（CPU・ミニPC で常時）
```
git pull
net, ver = current_or_heuristic()        # 現役が無ければ HeuristicEvaluator にフォールバック(=bootstrapデータ)
while True:
    cur = backend.current_model()         # 安いポインタ確認（毎バッチ）
    if cur != ver: net, ver = download(cur)   # 版が変わった時だけ重みDL
    samples = selfplay(net, sims, K)      # K局（バッチ長 30秒〜2分）
    backend.put_object(shard_key(), encode(samples), provenance=...)
```
- 局の途中ではモデルを替えない（π/z整合）。古さは provenance に `model_version` として記録。

### 6.2 train（GPU・ラップトップ）
```
git pull
data = sampler.draw(backend.latest_manifest(), cfg)   # 直近window + 鮮度重み（+任意corpora）
new  = learner.train(load(current), data, cfg)        # GPU
backend.put_object(model_key(new), weights); put manifest
if gating.accept(new, reference):                      # arena 勝率で判定
    backend.put_object(promote_key(new), {...})        # 現役を更新（追記）
```

### 6.3 eval（CPU・任意）
任意 2 モデル（or heuristic）を arena 対戦し、結果を `events/arena/` に追記。gating／序列に使用。

---

## 7. タスク分類と協調（協調は WF から導く）

| タスク | コスト | 可換性 | 必要な協調 |
|---|---|---|---|
| 生成 | 安・大量 | **完全可換**（追記、重複も無害） | **不要（並列し放題）** |
| 学習 | 高(GPU) | 半可換（重複＝GPU無駄、結果は壊れない） | 弱い単一化 |
| 昇格(promote) | 安 | **非可換**（現役は1つ） | **追記＋決定論読み**で一貫性 |
| 評価(arena) | 中 | 可換（結果を追記） | 不要 |

- **並列性の源泉は協調でなく可換性**。生成は可換＋追記なので**ワーカーを増やすだけで並列協力**（主従不要）。
- **2ノード**: 生成=ミニPC（無協調）、学習＆昇格=ラップトップのみ（**単一writer＝役割慣習**）。→ **分散協調の機構は一切不要**。
- **将来の自律多ノード**: 学習の重複回避が要るときだけ **conditional-write の lease(claim イベント)** を限定導入。leader election（主従）は最終手段。
- **純不変との整合**: 可換タスクは純追記を維持。非可換（昇格）は**追記＋読み出し時の決定論解決**で不変を維持。lease は可変要素なので**必要時のみ**。

---

## 8. 生成のモデル更新ポリシー

- **確認はバッチ境界ごと**（局の途中では替えない）。
- **ポインタ確認は安い（小JSONのGET）→ 毎バッチ実施**。**重みDLは版変化時のみ**。
- バッチ長を **30秒〜2分**に設計（trainer の publish 間隔より細かくする必要はない）。
- 古さ（model_version の遅れ）は **provenance に記録 ＋ 学習側の鮮度重みサンプリングで吸収**（非同期 actor-learner の標準）。

---

## 9. 学習のデータ選択（sampler）

- **単位は positions / games**（“最新 N シャード”は**禁止**：シャードサイズ・worker速度・再送で偏る、レビュー指摘）。
- **混合比の初期値**: self-play 主学習 = **直近 window 70–90% ＋ archive/bootstrap 10–30%**。
- **鮮度重み**: `w = exp(-(current_generation - sample_generation) / tau)`、`tau = 5–20 世代`。
- **フィルタ＋allowlist**: provenance（generator_kind / strength / encoding_version）で選別し、
  **sampler 側に明示的な allowlist**（どの生成器/世代のデータを学習に入れてよいか）を持つ。
- **replay_ratio = 学習で見た positions ÷ 新規生成 positions を必ず可視化し上限を設ける**。
  GPUが余る環境では staleness より先に「**小さい直近データへの過学習**」が起きる（run2の教訓＋レビュー指摘）。

---

## 10. プラグイン（ロジック差し替え）

```
Generator.generate(net, cfg) -> samples          # selfplay-latest / best-gated / mixed / heuristic-bootstrap / styled
Learner.train(dataset, net, cfg) -> net          # selfplay-train / supervised-pretrain / distill
Sampler.draw(catalog, cfg) -> dataset            # recent-window / freshness-weighted / by-generator-mix
Gating.accept(candidate, reference) -> bool      # arena-winrate / EMA / margin
```
- **実験 = 宣言的 config**（YAML/JSON）で `{generator, sampler, learner, gating, schedule}` を選ぶ。
- 既存ロジックを「最初のプラグイン」として移植する。
- **境界を明確化（レビュー指摘）**: Generator=ゲーム生成と π 保存 / Sampler=データ選択と batch構成 /
  Learner=loss・optimizer・update / Gating=評価試合と accept 判定 / Evaluator=engineから見える推論interface。
  **Gating は Learner に含めない**（loss と強さを分離する＝run2 の教訓）。
  `Generator.generate(net,cfg)->samples` は粗いので、探索方策・対戦相手選択・温度schedule・保存形式を
  サブ設定として分離する。

---

## 11. ローカル捕捉 / 可観測性

- ダッシュボード（既存 `/training` を拡張）は**カタログを読む**: 生成レート（shards/hr）・データ量（生成器別/世代別/日付）・モデル系譜・現役・arena 勝率。
- **必要なモデル/データだけ pull**（read-through cache）。ローカルから学習・対局・検証。
- クラウドに保存しつつ手元から「いい感じに捕捉」を実現。

---

## 12. トレードオフ・既知の論点

- **staleness vs reuse**: 直近window＋蓄積コーパス、鮮度重みで両立。
- **loss ≠ 強さ**: 真の指標は arena 勝率（固定seed・十分局数・EMA）。best は EMA＋top-K＋arena 確定。
- **collapse**: B1（点数決着）＋ B9（capped/draw/entropy/`vloss→0` 監視）。
- **重複作業**: 生成は許容、学習は単一writer/慣習（or lease）。
- **object storage 一貫性**: new object は read-after-write OK。lease は conditional-write。
- **コスト**: managed S3互換（R2/B2は安）or MinIO 自前（無料）。
- **エンコーディング版管理**: `encoding_version` で古い π/特徴の整合を判断（不一致は再生成 or 除外）。

---

## 13. 移行フェーズ

- **Phase 1（最重要・前提）**: `train.py` を **`generate`/`train` に分離** ＋ **追記 StorageBackend(LocalFS)** ＋ **スキーマ確定**。→ ローカルで 2 プロセス協調を予行。
- **Phase 2**: StorageBackend を **S3互換/MinIO** に。→ ミニPC生成 ＋ ラップトップ学習 ＋ ダッシュボードのクラウド読み（クラウド保存＋ローカル捕捉が完成）。
- **Phase 3**: **プラグイン化**（Generator/Learner/Sampler/Gating）、実験=config。
- **Phase 4**: 生成器の**水平スケール**（stateless writer なので自然）。必要なら lease。
- **Phase 5**: クラスタ可観測ダッシュボード、半自動学習トリガ。

---

## 14. レビューしてほしい観点（open questions）

- (a) **データスキーマ**: 棋譜(seed+moves+result)＋per-move π＋provenance、特徴は再生成・z は再計算、という分割の妥当性。π の保存形式（疎/密）、`encoding_version` 運用の落とし穴。
- (b) **staleness/サンプリング**: 直近window＋鮮度重みの具体指針（混合比、世代減衰の付け方）。非同期 actor-learner として健全か。
- (c) **協調**: 2ノードは役割慣習で無協調、将来のみ conditional-write lease を段階導入、という方針の妥当性。純追記と lease の整合。
- (d) **プラグイン境界**: Generator/Learner/Sampler/Gating の切り方は過不足ないか。実験=config の粒度。
- (e) **見落とし・リスク**: 上記以外で、この構成の弱点・運用上の罠・先に潰すべき箇所。

---

## 15. Codexレビュー反映（要約と優先順位）

**総評**: 方向性は妥当。ただし「データレイク中心／追記専用／疎結合」を綺麗に言い切りすぎており、
実運用で壊れる箇所がある。最大リスクは *「きれいな append-only 基盤の上で、汚いデータを賢く再利用して
"弱くなる"」*。本文 §4・§7・§9・§10 に指摘を反映済み。

**先に固める順序（storage より先に）**:
1. **π の保存形式** = `visit_counts[45] + legal_mask[45]`（密）。
2. **provenance の必須項目**（model_hash / mcts_config / 各 *_version / ply・手番 など）。
3. **sampler の混合規則 ＋ allowlist**（positions単位・鮮度重み・replay_ratio上限）。
4. **gating の統計ルール**。
（StorageBackend / クラウド化はその後で十分。）

**追加で固める運用ルール（§12 を補強）**:
- **データ品質の隔離**: heuristic-bootstrap / 旧 self-play / 失敗run / 実験generator が同じ湖に入るので、
  provenance だけでなく **sampler 側 allowlist** で「学習に入れてよいデータ」を明示。
- **カタログ整合性ルール**: 壊れた manifest / 途中アップロード / 欠損 shard / 重複 game_id の扱いを定義
  （未定義だと学習が静かに汚れる）。
- **arena/gating の統計**: 「十分局数」を具体化 — **固定 seed・最低対局数・信頼区間・引き分け/点数決着の扱い・
  旧 best への回帰テスト**を決める。Gating は Learner から独立させる。
- **promote の決定論**: 壁時計 ts でなく `logical_generation` で確定（時計ずれ対策）。
