"use strict";
// Training monitor: polls /api/training/metrics (files written by the trainer
// process) and renders status + loss/winrate charts. Read-only; fully separate
// from the game UI at "/".

const POLL_MS = 3000;
const $ = (id) => document.getElementById(id);

function fmtDuration(sec) {
  if (sec == null || !isFinite(sec)) return "—";
  sec = Math.max(0, Math.floor(sec));
  const h = Math.floor(sec / 3600), m = Math.floor((sec % 3600) / 60), s = sec % 60;
  return h > 0 ? `${h}:${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`
               : `${m}:${String(s).padStart(2, "0")}`;
}
function fmtTime(epoch) {
  if (!epoch) return "—";
  const d = new Date(epoch * 1000);
  return d.toLocaleString("ja-JP", { month: "numeric", day: "numeric",
    hour: "2-digit", minute: "2-digit" });
}
function fmtBytes(n) {
  if (n == null) return "—";
  return n >= 1e6 ? (n / 1e6).toFixed(1) + " MB" : (n / 1e3).toFixed(0) + " KB";
}
function fmtNum(n) { return n == null ? "—" : n.toLocaleString("en-US"); }
function pct(x) { return x == null ? "—" : (x * 100).toFixed(1) + "%"; }

function setStatus(status) {
  const badge = $("status-badge"), text = $("status-text");
  let cls = "idle", label = "待機中";
  if (status) {
    if (status.state === "running") { cls = "running"; label = "稼働中"; }
    else { cls = "stopped"; label = "停止"; }
  }
  badge.className = "badge " + cls;
  text.textContent = label;
}

function renderStats(data) {
  const st = data.status || {};
  const now = st.now || (Date.now() / 1000);
  const elapsed = st.started_at ? now - st.started_at : null;
  let eta = null, endLabel = "—";
  if (st.target_end) {
    eta = st.target_end - now;
    endLabel = fmtTime(st.target_end);
  }
  const best = (st.best_winrate != null)
    ? `${pct(st.best_winrate)} <small>@${st.best_iter}</small>` : "—";

  const cells = [
    ["イテレーション", fmtNum(st.iteration ?? data.totalIters)],
    ["経過時間", fmtDuration(elapsed)],
    ["終了予定", st.target_end
      ? `${endLabel} <small>(残り ${fmtDuration(eta)})</small>` : "無制限"],
    ["生成速度", st.samples_per_s != null ? `${st.samples_per_s} <small>smp/s</small>` : "—"],
    ["サンプル累計", fmtNum(st.samples_seen)],
    ["バッファ", fmtNum(st.buffer)],
    ["最高勝率(best.pt)", best],
    ["未決着率(cap)", st.capped_rate != null ? pct(st.capped_rate) : "—"],
    ["policyエントロピー", st.policy_entropy != null ? st.policy_entropy.toFixed(2) : "—"],
    ["学習率(lr)", st.lr != null ? st.lr.toExponential(1) : "—"],
    ["環境", `${st.device || "—"} <small>×${st.workers ?? "?"}w</small>`],
  ];
  $("stats").innerHTML = cells.map(([k, v]) =>
    `<div class="stat"><div class="k">${k}</div><div class="v">${v}</div></div>`).join("");
}

// ---- minimal canvas line chart -------------------------------------------
function prepCanvas(canvas) {
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth, h = canvas.clientHeight;
  canvas.width = w * dpr; canvas.height = h * dpr;
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  return { ctx, w, h };
}

function drawChart(canvas, series, opts = {}) {
  const { ctx, w, h } = prepCanvas(canvas);
  const padL = 42, padR = 10, padT = 10, padB = 22;
  const x0 = padL, x1 = w - padR, y0 = h - padB, y1 = padT;
  const all = series.flatMap((s) => s.points);
  if (all.length < 2) {
    ctx.fillStyle = "#8d99ad"; ctx.font = "12px Inter";
    ctx.fillText("データ待ち…", x0, (y0 + y1) / 2);
    return;
  }
  const xs = all.map((p) => p.x), ys = all.map((p) => p.y);
  let xmin = Math.min(...xs), xmax = Math.max(...xs);
  let ymin = opts.ymin != null ? opts.ymin : Math.min(...ys);
  let ymax = opts.ymax != null ? opts.ymax : Math.max(...ys);
  if (xmax === xmin) xmax = xmin + 1;
  if (ymax === ymin) ymax = ymin + 1;
  const sx = (x) => x0 + (x - xmin) / (xmax - xmin) * (x1 - x0);
  const sy = (y) => y0 - (y - ymin) / (ymax - ymin) * (y0 - y1);

  // gridlines + y labels
  ctx.strokeStyle = "#2a3344"; ctx.fillStyle = "#8d99ad";
  ctx.font = "10px Inter"; ctx.lineWidth = 1;
  const ticks = opts.yticks || 4;
  for (let i = 0; i <= ticks; i++) {
    const yv = ymin + (ymax - ymin) * i / ticks;
    const py = sy(yv);
    ctx.globalAlpha = 0.5; ctx.beginPath();
    ctx.moveTo(x0, py); ctx.lineTo(x1, py); ctx.stroke();
    ctx.globalAlpha = 1;
    ctx.fillText(opts.yfmt ? opts.yfmt(yv) : yv.toFixed(2), 4, py + 3);
  }
  // reference line (e.g. 50% winrate)
  if (opts.ref != null && opts.ref >= ymin && opts.ref <= ymax) {
    ctx.strokeStyle = "#5b6b82"; ctx.setLineDash([4, 4]);
    const py = sy(opts.ref); ctx.beginPath();
    ctx.moveTo(x0, py); ctx.lineTo(x1, py); ctx.stroke();
    ctx.setLineDash([]);
  }
  // series
  for (const s of series) {
    if (s.points.length < 1) continue;
    ctx.strokeStyle = s.color; ctx.lineWidth = 2; ctx.beginPath();
    s.points.forEach((p, i) => {
      const px = sx(p.x), py = sy(p.y);
      i === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py);
    });
    ctx.stroke();
    if (s.dots) {
      ctx.fillStyle = s.color;
      s.points.forEach((p) => {
        ctx.beginPath(); ctx.arc(sx(p.x), sy(p.y), 2.5, 0, 7); ctx.fill();
      });
    }
  }
}

function renderCharts(metrics) {
  const ploss = metrics.filter((m) => m.ploss != null)
    .map((m) => ({ x: m.iter, y: m.ploss }));
  const vloss = metrics.filter((m) => m.vloss != null)
    .map((m) => ({ x: m.iter, y: m.vloss }));
  drawChart($("loss-chart"), [
    { points: ploss, color: "#5fa8ff" },
    { points: vloss, color: "#c7a049" },
  ], { ymin: 0, yfmt: (v) => v.toFixed(2) });

  const wr = metrics.filter((m) => m.eval_winrate != null)
    .map((m) => ({ x: m.iter, y: m.eval_winrate }));
  const wrEma = metrics.filter((m) => m.eval_winrate_ema != null)
    .map((m) => ({ x: m.iter, y: m.eval_winrate_ema }));
  drawChart($("winrate-chart"), [
    { points: wr, color: "#46c08a", dots: true },   // raw eval
    { points: wrEma, color: "#c7a049" },             // EMA (best.pt gating)
  ], { ymin: 0, ymax: 1, ref: 0.5, yfmt: (v) => (v * 100).toFixed(0) + "%" });

  // collapse diagnostics
  const capped = metrics.filter((m) => m.capped_rate != null)
    .map((m) => ({ x: m.iter, y: m.capped_rate }));
  const draws = metrics.filter((m) => m.draw_rate != null)
    .map((m) => ({ x: m.iter, y: m.draw_rate }));
  drawChart($("diag-rates"), [
    { points: capped, color: "#e2614e" },
    { points: draws, color: "#d9b44a" },
  ], { ymin: 0, ymax: 1, yfmt: (v) => (v * 100).toFixed(0) + "%" });

  const ent = metrics.filter((m) => m.policy_entropy != null)
    .map((m) => ({ x: m.iter, y: m.policy_entropy }));
  drawChart($("diag-entropy"), [
    { points: ent, color: "#5fa8ff" },
  ], { ymin: 0, yfmt: (v) => v.toFixed(2) });
}

const TAG_STYLE = {
  best: 'background:rgba(199,160,73,.18);color:#c7a049',
  peak: 'background:rgba(95,168,255,.16);color:#5fa8ff',
  latest: 'background:rgba(141,153,173,.16);color:#aab4c5',
  hourly: 'background:rgba(141,153,173,.10);color:#8d99ad',
  ckpt: 'background:rgba(141,153,173,.10);color:#8d99ad',
};
function tagChip(t) {
  return `<span style="${TAG_STYLE[t] || TAG_STYLE.ckpt};padding:1px 7px;`
    + `border-radius:999px;font-size:10px;font-weight:700">${t}</span>`;
}

function renderCkptBrowser(list) {
  const el = $("ckpt-browser");
  if (!el) return;
  if (!list || !list.length) { el.innerHTML = `<div class="hint">まだありません</div>`; return; }
  // group by run, preserving recency order of runs
  const groups = {};
  for (const c of list) { (groups[c.run] = groups[c.run] || []).push(c); }
  const runOrder = [...new Set(list.map((c) => c.run))];
  el.innerHTML = runOrder.map((run) => {
    const rows = groups[run]
      .sort((a, b) => (b.tag === "best") - (a.tag === "best") || b.mtime - a.mtime)
      .map((c) => `<tr>
        <td>${tagChip(c.tag)}</td>
        <td style="font-family:monospace">${c.name}</td>
        <td>${c.iteration != null ? "iter " + c.iteration : "—"}</td>
        <td>${c.winrate != null ? `<b style="color:#46c08a">${(c.winrate * 100).toFixed(0)}%</b>` : "—"}</td>
        <td>${fmtBytes(c.size)}</td>
        <td style="color:#8d99ad">${fmtTime(c.mtime)}</td>
      </tr>`).join("");
    return `<div style="margin-bottom:14px">
      <div style="font-weight:700;font-size:13px;margin-bottom:6px">📁 ${run}</div>
      <table><thead><tr><th></th><th>ファイル</th><th>iter</th><th>勝率</th>
        <th>サイズ</th><th>更新</th></tr></thead><tbody>${rows}</tbody></table>
    </div>`;
  }).join("");
}

// build a <select>'s options from the checkpoint list, preserving selection
function fillCkptSelect(sel, list, { withHeuristic = false, withNone = false } = {}) {
  if (!sel) return;
  const cur = sel.value;
  const opts = [];
  if (withNone) opts.push(`<option value="">（なし＝ゼロから学習）</option>`);
  if (withHeuristic) opts.push(`<option value="heuristic">heuristic（v1 AI）</option>`);
  for (const c of list || []) {
    const wr = c.winrate != null ? ` · ${(c.winrate * 100).toFixed(0)}%` : "";
    opts.push(`<option value="${c.path}">${c.label}${wr}</option>`);
  }
  sel.innerHTML = opts.join("");
  if (cur) sel.value = cur;  // keep the user's choice across refreshes
}

async function poll() {
  try {
    const res = await fetch("/api/training/metrics");
    if (!res.ok) throw new Error(res.status);
    const data = await res.json();
    const hasRun = data.status || (data.metrics && data.metrics.length);
    $("no-run").style.display = hasRun ? "none" : "block";
    $("dash").style.display = hasRun ? "block" : "none";
    setStatus(data.status);
    renderManaged(data.managed);
    if (hasRun) {
      renderStats(data);
      renderCharts(data.metrics || []);
      renderLog(data.log);
    }
    if (data.allCheckpoints) {
      renderCkptBrowser(data.allCheckpoints);
      fillCkptSelect($("f-init_from"), data.allCheckpoints, { withNone: true });
      fillCkptSelect($("ar-a"), data.allCheckpoints);
      fillCkptSelect($("ar-b"), data.allCheckpoints, { withHeuristic: true });
    }
  } catch (e) {
    setStatus(null);
  }
}

// ---- control panel: start / stop / presets -------------------------------
const FIELDS = ["hours", "workers", "games_per_iter", "sims", "eval_every",
  "eval_games", "train_steps", "lr", "temp", "entropy_coef", "width", "blocks",
  "device", "seed"];
let PRESETS = [];
let DEFAULTS = {};

function fieldEl(k) { return document.getElementById("f-" + k); }

function applyValues(vals) {
  for (const [k, v] of Object.entries(vals)) {
    const el = fieldEl(k);
    if (el) el.value = v;
  }
  updateCmdPreview();
}

function collectBody() {
  const body = { run: $("f-run").value || "run1", resume: $("f-resume").checked,
                 init_from: $("f-init_from").value || "" };
  for (const k of FIELDS) {
    const el = fieldEl(k);
    if (!el) continue;
    body[k] = el.type === "number" ? Number(el.value) : el.value;
  }
  return body;
}

function updateCmdPreview() {
  const b = collectBody();
  const parts = ["python -m splendor.train", `--ckpt-dir checkpoints_runs/${b.run}`];
  if (b.resume) parts.push("--resume");
  if (b.init_from) parts.push(`--init-from ${b.init_from}`);
  $("warm-note").style.display = b.init_from ? "block" : "none";
  const flagmap = { games_per_iter: "games-per-iter", eval_every: "eval-every",
    eval_games: "eval-games", train_steps: "train-steps", entropy_coef: "entropy-coef" };
  for (const k of FIELDS) {
    const flag = flagmap[k] || k;
    parts.push(`--${flag} ${b[k]}`);
  }
  $("cmd-preview").textContent = parts.join(" ");
}

async function initControls() {
  try {
    const opt = await (await fetch("/api/training/options")).json();
    PRESETS = opt.presets || [];
    const sel = $("f-preset");
    sel.innerHTML = `<option value="">— カスタム —</option>` +
      PRESETS.map((p) => `<option value="${p.id}">${p.label}</option>`).join("");
    DEFAULTS = opt.defaults || {};
    const cks = opt.checkpoints || [];
    fillCkptSelect($("f-init_from"), cks, { withNone: true });
    $("f-init_from").onchange = updateCmdPreview;
    fillCkptSelect($("ar-a"), cks);
    fillCkptSelect($("ar-b"), cks, { withHeuristic: true });
    $("ar-b").value = "heuristic";
    $("ar-run").onclick = startArena;
    pollArena();  // pick up any in-flight / last match
    sel.onchange = () => {
      const p = PRESETS.find((x) => x.id === sel.value);
      // reset every field to its default first, so switching presets never
      // leaves a stale value (e.g. a big net from a previous preset).
      if (p) applyValues({ ...DEFAULTS, ...p.values });
    };
    applyValues(DEFAULTS);
  } catch (e) { /* server may be momentarily unavailable */ }

  FIELDS.forEach((k) => { const el = fieldEl(k); if (el) el.oninput = updateCmdPreview; });
  $("f-run").oninput = updateCmdPreview;
  $("f-resume").onchange = updateCmdPreview;

  $("adv-toggle").onclick = () => {
    const a = $("adv");
    const show = a.style.display === "none";
    a.style.display = show ? "grid" : "none";
    $("adv-toggle").textContent = show ? "詳細設定 ▴" : "詳細設定 ▾";
  };

  $("btn-start").onclick = async () => {
    $("ctrl-msg").textContent = "起動中…";
    try {
      const r = await fetch("/api/training/start", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify(collectBody()),
      });
      const j = await r.json();
      if (!r.ok) throw new Error(j.detail || r.status);
      $("ctrl-msg").textContent = `開始しました（run=${j.run}, pid=${j.pid}）`;
    } catch (e) {
      $("ctrl-msg").textContent = "開始に失敗: " + e.message;
    }
    poll();
  };

  $("btn-stop").onclick = async () => {
    $("ctrl-msg").textContent = "停止要求を送信…";
    try {
      const r = await fetch("/api/training/stop", { method: "POST" });
      const j = await r.json();
      $("ctrl-msg").textContent = j.message || "停止しました";
    } catch (e) {
      $("ctrl-msg").textContent = "停止に失敗: " + e.message;
    }
    poll();
  };

  updateCmdPreview();
}

function renderManaged(managed) {
  const running = managed && managed.running;
  const badge = $("run-state"), text = $("run-state-text");
  badge.className = "badge " + (running ? "running" : "idle");
  text.textContent = running ? `稼働中 (pid ${managed.pid})` : "未稼働";
  $("btn-start").disabled = !!running;
  $("btn-stop").disabled = !running;
}

function renderLog(lines) {
  const el = $("log");
  if (!el) return;
  el.textContent = (lines && lines.length) ? lines.join("\n") : "(ログなし)";
  el.scrollTop = el.scrollHeight;
}

// ---- arena compare -------------------------------------------------------
let arenaTimer = null;
async function startArena() {
  const a = $("ar-a").value, b = $("ar-b").value || "heuristic";
  if (!a) { $("ar-msg").textContent = "A（候補）を選んでください"; return; }
  if (a === b) { $("ar-msg").textContent = "A と B が同じです"; return; }
  $("ar-msg").textContent = "対戦を開始…";
  $("ar-result").style.display = "none";
  try {
    const r = await fetch("/api/training/arena", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ a, b, games: Number($("ar-games").value),
        sims: Number($("ar-sims").value) }),
    });
    const j = await r.json();
    if (!r.ok) throw new Error(j.detail || r.status);
    pollArena();
  } catch (e) { $("ar-msg").textContent = "失敗: " + e.message; }
}

async function pollArena() {
  clearTimeout(arenaTimer);
  try {
    const s = await (await fetch("/api/training/arena")).json();
    if (s.state === "running") {
      const p = s.progress || {};
      $("ar-msg").textContent = `対戦中 ${p.done || 0}/${p.total || 0} …`;
      arenaTimer = setTimeout(pollArena, 1500);
    } else if (s.state === "done" && s.result) {
      const r = s.result, wr = (r.a_winrate * 100).toFixed(1);
      const verdict = r.a_winrate > 0.5 ? "→ A が強い" : r.a_winrate < 0.5 ? "→ B が強い" : "→ 互角";
      $("ar-msg").textContent = "完了 " + verdict;
      $("ar-result").style.display = "block";
      $("ar-result").textContent =
        `A = ${s.a}\nB = ${s.b}\n${r.games}局 / sims=${r.sims}\n`
        + `A ${r.a_wins}勝  B ${r.b_wins}勝  draw ${r.draws}\n`
        + `A 勝率 ${wr}%   Elo(A-B) ${r.elo_diff >= 0 ? "+" : ""}${Math.round(r.elo_diff)}`;
    } else if (s.state === "error") {
      $("ar-msg").textContent = "エラー: " + (s.result && s.result.error);
    }
  } catch (e) { $("ar-msg").textContent = "状態取得に失敗"; }
}

let timer = null;
function loop() { poll(); timer = setTimeout(loop, POLL_MS); }
document.addEventListener("visibilitychange", () => {
  if (document.hidden) { clearTimeout(timer); }
  else { loop(); }
});
window.addEventListener("resize", () => poll());
initControls();
loop();
