"use strict";

// ---------------------------------------------------------------------------
// Constants & helpers
// ---------------------------------------------------------------------------
const COLORS = ["w", "u", "g", "r", "k"];
const TOKENS = ["w", "u", "g", "r", "k", "y"];
const NAME = { w: "白", u: "青", g: "緑", r: "赤", k: "黒", y: "金" };
const gemClass = (c) => "gem-" + c;
const $ = (id) => document.getElementById(id);

async function api(path, body, method = "POST") {
  const opt = { method, headers: { "Content-Type": "application/json" } };
  if (method !== "GET" && body !== undefined) opt.body = JSON.stringify(body);
  const r = await fetch(path, opt);
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
  return r.json();
}

function primaryOf(notation) { return notation.split("/")[0]; }

// readable Japanese label for a move notation
function labelFor(note) {
  const [head, ...rest] = note.split("/");
  const [tag, val] = head.split(":");
  let s;
  if (tag === "T") {
    if (val.length === 2 && val[0] === val[1]) s = `取る: ${NAME[val[0]]}×2`;
    else s = "取る: " + [...val].map((c) => NAME[c]).join(" ");
  } else if (tag === "R") {
    s = val.endsWith("^") ? `予約: Lv${val[1]} 山札` : `予約: Lv${val[1]} #${val.split("#")[1]}`;
  } else if (tag === "P") {
    s = val.startsWith("H#") ? `購入: 予約#${val.split("#")[1]}` : `購入: Lv${val[1]} #${val.split("#")[1]}`;
  } else s = note;
  for (const seg of rest) {
    const [t, v] = seg.split(":");
    if (t === "D") s += ` (捨${[...v].map((c) => NAME[c]).join("")})`;
    if (t === "N") s += " ★貴族";
  }
  return s;
}

// parse a board location "L{t}#{s}" out of a notation, or null
function boardLoc(note) {
  const m = primaryOf(note).match(/^[PR]:L(\d)#(\d)$/);
  return m ? { tier: +m[1], slot: +m[2] } : null;
}

// ---------------------------------------------------------------------------
// App state
// ---------------------------------------------------------------------------
let game = null;       // { gameId, view, legalMoves }
let selectedGems = []; // token-take selection
let review = null;     // { record, ply, evalByPly }

// ---------------------------------------------------------------------------
// Card / token rendering
// ---------------------------------------------------------------------------
function cardEl(card, opts = {}) {
  const el = document.createElement("div");
  el.className = "card";
  if (!card) { el.classList.add("empty"); return el; }
  el.classList.add("tint-" + card.bonus);
  el.dataset.tier = card.tier;
  el.dataset.slot = card.slot;
  if (opts.affordable) el.classList.add("affordable");
  if (opts.best) el.classList.add("best");

  const top = document.createElement("div");
  top.className = "top";
  top.innerHTML = `<span class="pts">${card.points || ""}</span>` +
    `<span class="bonus ${gemClass(card.bonus)}"></span>`;
  const cost = document.createElement("div");
  cost.className = "cost";
  for (const c of COLORS) {
    if (card.cost[c]) {
      const pip = document.createElement("span");
      pip.className = `cost-pip ${gemClass(c)}`;
      pip.textContent = card.cost[c];
      cost.appendChild(pip);
    }
  }
  el.appendChild(top);
  el.appendChild(cost);
  return el;
}

function tokenChip(c) {
  return `<span class="chip ${gemClass(c)}">${NAME[c]}</span>`;
}

// ---------------------------------------------------------------------------
// PLAY view rendering
// ---------------------------------------------------------------------------
function renderPlay() {
  const view = game.view;
  // turn / winner
  const ti = $("turn-indicator");
  const banner = $("winner-banner");
  if (view.winner !== null) {
    const w = view.players[view.winner];
    banner.textContent = `🏆 ${w.name} の勝ち！ (${view.players.map(p => p.points).join(" - ")})`;
    banner.classList.remove("hidden");
    ti.textContent = "対局終了";
    $("start-review").classList.remove("hidden");
  } else {
    banner.classList.add("hidden");
    $("start-review").classList.add("hidden");
    const cur = view.players[view.toMove];
    ti.textContent = `手番: ${cur.name}${cur.type === "human" ? "（あなた）" : "（AI）"}  / ${view.ply}手目`;
  }

  // legal-move target sets
  const buyable = new Set(), reservable = new Set();
  for (const lm of game.legalMoves) {
    const loc = boardLoc(lm.notation);
    if (loc) (lm.notation[0] === "P" ? buyable : reservable).add(`${loc.tier}#${loc.slot}`);
  }

  // board
  const board = $("board");
  board.innerHTML = "";
  for (const tier of [3, 2, 1]) {
    const row = document.createElement("div");
    row.className = "tier-row";
    const label = document.createElement("div");
    label.className = "tier-label";
    label.textContent = "L" + tier;
    row.appendChild(label);

    const deck = document.createElement("div");
    deck.className = "deck";
    deck.innerHTML = `山札<b>${view.deckCounts["L" + tier]}</b>`;
    deck.title = `Lv${tier} 山札から予約`;
    deck.onclick = () => tryMovesForNotation(`R:L${tier}^`);
    row.appendChild(deck);

    for (const card of view.board["L" + tier]) {
      const key = card ? `${card.tier}#${card.slot}` : null;
      const el = cardEl(card, { affordable: card && buyable.has(key) });
      if (card) el.onclick = (e) => openCardMenu(e, card);
      row.appendChild(el);
    }
    board.appendChild(row);
  }

  // bank
  const bank = $("bank");
  bank.innerHTML = '<div class="bank-title">トークン</div>';
  for (const c of TOKENS) {
    const t = document.createElement("div");
    t.className = "token";
    if (selectedGems.includes(c) && c !== "y") t.classList.add("selected");
    if (c === "y") t.classList.add("disabled"); // gold not directly takeable
    t.innerHTML = `${tokenChip(c)}<span class="count">×${view.bank[c]}</span>`;
    if (c !== "y") t.onclick = () => toggleGem(c);
    bank.appendChild(t);
  }
  renderTakeTray();

  // nobles
  const nobles = $("nobles");
  nobles.innerHTML = '<div class="nobles-title">貴族</div>';
  for (const n of view.nobles) {
    const el = document.createElement("div");
    el.className = "noble";
    const req = Object.entries(n.requirement)
      .map(([c, v]) => `<span class="cost-pip ${gemClass(c)}">${v}</span>`).join("");
    el.innerHTML = `<span class="pts">${n.points}</span><div class="req">${req}</div>`;
    nobles.appendChild(el);
  }

  renderPlayers(view);
}

function renderTakeTray() {
  let tray = $("take-tray");
  if (!tray) {
    tray = document.createElement("div");
    tray.id = "take-tray";
    tray.className = "take-tray";
    $("bank").after(tray);
  }
  const sel = selectedGems.map(tokenChip).join("");
  tray.innerHTML = `<span>選択:</span><span class="sel">${sel || "—"}</span>`;
  const take = document.createElement("button");
  take.className = "primary";
  take.textContent = "取る";
  take.disabled = selectedGems.length === 0 || !humanTurn();
  take.onclick = confirmTake;
  const clr = document.createElement("button");
  clr.className = "ghost"; clr.textContent = "選択解除";
  clr.onclick = () => { selectedGems = []; renderPlay(); };
  tray.appendChild(take); tray.appendChild(clr);
}

function renderPlayers(view) {
  const wrap = $("players");
  wrap.innerHTML = "";
  view.players.forEach((p, idx) => {
    const el = document.createElement("div");
    el.className = "player" + (idx === view.toMove && view.winner === null ? " active" : "");
    const tally = COLORS.map((c) =>
      `<div class="col">${tokenChip(c)}<span>${p.bonuses[c]}<small style="color:var(--muted)">+${p.tokens[c]}</small></span></div>`
    ).join("");
    const gold = `<div class="col">${tokenChip("y")}<span>${p.tokens.y}</span></div>`;
    el.innerHTML = `<h3><span>${p.name}</span><span class="score">${p.points}点</span></h3>
      <div class="tally">${tally}${gold}</div>`;
    const rr = document.createElement("div");
    rr.className = "reserved-row";
    p.reserved.forEach((c, ri) => {
      if (c.hidden) {
        const h = document.createElement("div");
        h.className = "mini-hidden";
        h.textContent = `Lv${c.tier}?`;
        rr.appendChild(h);
      } else {
        const el2 = cardEl({ ...c, slot: -1 });
        el2.onclick = (e) => openReservedMenu(e, ri);
        rr.appendChild(el2);
      }
    });
    el.appendChild(rr);
    wrap.appendChild(el);
  });
}

// ---------------------------------------------------------------------------
// Interaction
// ---------------------------------------------------------------------------
function humanTurn() {
  return game && game.view.winner === null &&
    game.view.players[game.view.toMove].type === "human";
}

function toggleGem(c) {
  if (!humanTurn()) return;
  const i = selectedGems.indexOf(c);
  if (i >= 0) { selectedGems.splice(i, 1); }
  else {
    // allow a second identical pick (take2) or distinct picks (take3)
    const distinct = new Set(selectedGems);
    if (selectedGems.length >= 3) return;
    if (selectedGems.includes(c) && (selectedGems.length !== 1)) return;
    if (!selectedGems.includes(c) && distinct.size >= 3) return;
    if (selectedGems.length === 1 && selectedGems[0] !== c && selectedGems.includes(c)) return;
    selectedGems.push(c);
  }
  renderPlay();
}

function confirmTake() {
  let cand;
  if (selectedGems.length === 2 && selectedGems[0] === selectedGems[1]) {
    cand = `T:${selectedGems[0]}${selectedGems[0]}`;
  } else {
    const distinct = [...new Set(selectedGems)];
    cand = "T:" + COLORS.filter((c) => distinct.includes(c)).join("");
  }
  selectedGems = [];
  tryMovesForNotation(cand);
}

// Find legal moves matching a primary notation; apply or let the user choose.
function tryMovesForNotation(primary) {
  const matches = game.legalMoves.filter((lm) => primaryOf(lm.notation) === primary);
  if (matches.length === 0) { flash(`その手は指せません (${labelFor(primary)})`); renderPlay(); return; }
  if (matches.length === 1) { applyMove(matches[0].notation); return; }
  chooseMove(matches); // discard / noble variants
}

function openCardMenu(e, card) {
  if (!humanTurn()) return;
  e.stopPropagation();
  const loc = `${card.tier}#${card.slot}`;
  const matches = game.legalMoves.filter((lm) => {
    const l = boardLoc(lm.notation);
    return l && `${l.tier}#${l.slot}` === loc;
  });
  if (matches.length === 0) { flash("このカードへの合法手がありません"); return; }
  const buys = matches.filter((m) => m.notation[0] === "P");
  const reserves = matches.filter((m) => m.notation[0] === "R");
  openActionMenu(e, card, buys, reserves);
}

// Always-explicit chooser: 💰購入 / 📌確保 (unavailable side greyed out with a reason)
function openActionMenu(e, card, buys, reserves) {
  closeMenu();
  const menu = document.createElement("div");
  menu.id = "popmenu";
  menu.className = "action-menu";
  menu.style.position = "fixed";
  menu.style.zIndex = 50;
  menu.style.left = Math.min(e.clientX, window.innerWidth - 230) + "px";
  menu.style.top = Math.min(e.clientY + 8, window.innerHeight - 200) + "px";

  const head = document.createElement("div");
  head.className = "am-head";
  head.innerHTML = `<span class="bonus ${gemClass(card.bonus)}"></span>` +
    `<span>Lv${card.tier} カード</span>` +
    (card.points ? `<span class="am-pts">${card.points}点</span>` : "");
  menu.appendChild(head);

  menu.appendChild(actionButton("購入", "💰", "am-buy", buys, "トークンが足りません", menu));
  menu.appendChild(actionButton("確保", "📌", "am-reserve", reserves, "これ以上確保できません", menu));

  document.body.appendChild(menu);
  setTimeout(() => document.addEventListener("click", closeMenu, { once: true }), 0);
}

function actionButton(label, icon, cls, moves, disabledReason, menu) {
  const btn = document.createElement("button");
  btn.className = "am-act " + cls;
  const tail = moves.length > 1 ? `<small>${moves.length}通り ▸</small>` : "";
  btn.innerHTML = `<span class="am-ico">${icon}</span><span class="am-label">${label}</span>${tail}`;
  if (moves.length === 0) {
    btn.disabled = true;
    btn.title = disabledReason;
    btn.innerHTML += `<small class="am-reason">${disabledReason}</small>`;
  } else if (moves.length === 1) {
    btn.onclick = (ev) => { ev.stopPropagation(); closeMenu(); applyMove(moves[0].notation); };
  } else {
    // multiple variants (e.g. token discard / 貴族 choice) — drill in
    btn.onclick = (ev) => { ev.stopPropagation(); showVariants(menu, `${icon} ${label}`, moves); };
  }
  return btn;
}

function showVariants(menu, label, moves) {
  menu.innerHTML = "";
  const head = document.createElement("div");
  head.className = "am-head";
  head.textContent = `${label} — 選択`;
  menu.appendChild(head);
  for (const m of moves) {
    const b = document.createElement("button");
    b.className = "am-variant";
    b.textContent = labelFor(m.notation);
    b.onclick = (ev) => { ev.stopPropagation(); closeMenu(); applyMove(m.notation); };
    menu.appendChild(b);
  }
}

function openReservedMenu(e, ridx) {
  if (!humanTurn()) return;
  e.stopPropagation();
  const opts = game.legalMoves.filter((lm) => lm.notation.startsWith(`P:H#${ridx}`));
  if (opts.length === 0) { flash("この予約カードは購入できません"); return; }
  chooseMove(opts, e);
}

// popup chooser for ambiguous moves
function chooseMove(moves, e) {
  closeMenu();
  if (moves.length === 1) { applyMove(moves[0].notation); return; }
  const menu = document.createElement("div");
  menu.id = "popmenu";
  menu.className = "cardmenu";
  menu.style.position = "fixed";
  menu.style.flexWrap = "wrap";
  menu.style.background = "var(--panel-2)";
  menu.style.padding = "6px";
  menu.style.borderRadius = "8px";
  menu.style.zIndex = 50;
  menu.style.maxWidth = "260px";
  const x = e ? Math.min(e.clientX, window.innerWidth - 270) : 200;
  const y = e ? e.clientY + 8 : 200;
  menu.style.left = x + "px"; menu.style.top = y + "px";
  for (const m of moves) {
    const b = document.createElement("button");
    b.textContent = labelFor(m.notation);
    b.onclick = () => { closeMenu(); applyMove(m.notation); };
    menu.appendChild(b);
  }
  document.body.appendChild(menu);
  setTimeout(() => document.addEventListener("click", closeMenu, { once: true }), 0);
}
function closeMenu() { const m = $("popmenu"); if (m) m.remove(); }

async function applyMove(notation) {
  try {
    game = await api(`/api/games/${game.gameId}/moves`, { notation });
    selectedGems = [];
    renderPlay();
    maybeAiMove();
  } catch (err) { flash("エラー: " + err.message); }
}

async function maybeAiMove() {
  if (!game || game.view.winner !== null) return;
  if (game.view.players[game.view.toMove].type === "human") return;
  const think = $("thinking");
  think.classList.remove("hidden");
  think.textContent = "AI 思考中…";
  try {
    const sims = parseInt($("ai-sims").value, 10) || 600;
    const res = await api(`/api/games/${game.gameId}/ai-move`, { sims });
    game = res;
    const s = res.aiMove.summary;
    think.textContent = `AIの手: ${labelFor(res.aiMove.notation)}  (勝率 ${(s.winProb * 100).toFixed(0)}%, ${s.sims} sims)`;
    renderPlay();
    setTimeout(maybeAiMove, 300); // chain if still AI's turn (>2p) or stop
  } catch (err) { think.textContent = "AIエラー: " + err.message; }
}

function flash(msg) {
  const think = $("thinking");
  think.classList.remove("hidden");
  think.textContent = msg;
}

// ---------------------------------------------------------------------------
// Position analysis (live MCTS over WebSocket)
// ---------------------------------------------------------------------------
function analyzePosition() {
  if (!game) return;
  const think = $("thinking");
  think.classList.remove("hidden");
  think.textContent = "局面解析中…";
  const sims = (parseInt($("ai-sims").value, 10) || 600) * 3;
  const ws = new WebSocket(`ws://${location.host}/ws/position`);
  ws.onopen = () => ws.send(JSON.stringify({ gameId: game.gameId, sims }));
  ws.onmessage = (e) => {
    const m = JSON.parse(e.data);
    if (m.type === "error") { think.textContent = "解析エラー: " + m.message; ws.close(); return; }
    const s = m.summary;
    if (s && s.bestMove) {
      const tops = s.topMoves.map((t) => `${labelFor(t.notation)} ${(t.winProb * 100).toFixed(0)}%`).join(" / ");
      think.innerHTML = `最善手: <b>${labelFor(s.bestMove.notation)}</b> (勝率 ${(s.bestMove.winProb * 100).toFixed(0)}%) ` +
        `<span style="color:var(--muted)">[${s.sims} sims]</span><br><small>${tops}</small>`;
      highlightBest(s.bestMove.notation, $("board"));
    }
    if (m.type === "done") ws.close();
  };
  ws.onerror = () => { think.textContent = "WebSocket接続に失敗しました"; };
}

function highlightBest(notation, container) {
  container.querySelectorAll(".card.best").forEach((el) => el.classList.remove("best"));
  const loc = boardLoc(notation);
  if (!loc) return;
  const el = container.querySelector(`.card[data-tier="${loc.tier}"][data-slot="${loc.slot}"]`);
  if (el) el.classList.add("best");
}

// ---------------------------------------------------------------------------
// REVIEW (感想戦)
// ---------------------------------------------------------------------------
async function startReview() {
  const sims = Math.min(parseInt($("ai-sims").value, 10) || 600, 1000);
  flash(`感想戦を解析中… (${sims} sims/手)`);
  const record = await api(`/api/games/${game.gameId}/record`, undefined, "GET");

  // stream analysis progress
  const ws = new WebSocket(`ws://${location.host}/ws/analyze`);
  ws.onopen = () => ws.send(JSON.stringify({ gameId: game.gameId, sims }));
  ws.onmessage = async (e) => {
    const m = JSON.parse(e.data);
    if (m.type === "progress") {
      flash(`感想戦を解析中… ${m.done}/${record.moves.length} 手`);
    } else if (m.type === "done") {
      ws.close();
      review = { record: m.record, ply: 0 };
      switchTab("review");
      renderReview();
      selectPly(0);
    } else if (m.type === "error") {
      flash("解析エラー: " + m.message); ws.close();
    }
  };
}

function renderReview() {
  drawGraph();
  renderMoveList();
}

function drawGraph() {
  const svg = $("winprob-graph");
  const moves = review.record.moves;
  const n = moves.length;
  const W = 600, H = 160;
  const x = (i) => (n <= 1 ? 0 : (i / (n - 1)) * W);
  const y = (wp) => (1 - wp) * H;
  let line = "";
  const pts = [];
  const dots = [];
  moves.forEach((mv, i) => {
    const wp = mv.eval ? mv.eval.winProbP0 : 0.5;
    const px = x(i).toFixed(1), py = y(wp).toFixed(1);
    line += (i === 0 ? "M" : "L") + px + "," + py + " ";
    pts.push([px, py]);
    if (mv.eval && mv.eval.blunder) dots.push({ i, wp });
  });
  const area = pts.length
    ? `M${pts[0][0]},${H} L` + pts.map((p) => p.join(",")).join(" L") + ` L${pts[pts.length - 1][0]},${H} Z`
    : "";
  const blunderDots = dots.map((d) =>
    `<circle cx="${x(d.i).toFixed(1)}" cy="${y(d.wp).toFixed(1)}" r="4.5" fill="var(--bad)" filter="url(#glow)"></circle>`).join("");
  const grid = [0.25, 0.5, 0.75].map((g) => {
    const mid = g === 0.5;
    return `<line x1="0" y1="${(1 - g) * H}" x2="${W}" y2="${(1 - g) * H}" stroke="${mid ? "#4a5269" : "#2a3040"}" stroke-dasharray="${mid ? "0" : "3 5"}" stroke-width="${mid ? 1.2 : 1}"/>`;
  }).join("");
  svg.innerHTML =
    `<defs>
       <linearGradient id="wpfill" x1="0" y1="0" x2="0" y2="1">
         <stop offset="0%" stop-color="var(--accent)" stop-opacity="0.42"/>
         <stop offset="100%" stop-color="var(--accent)" stop-opacity="0.02"/>
       </linearGradient>
       <filter id="glow" x="-50%" y="-50%" width="200%" height="200%">
         <feGaussianBlur stdDeviation="2.5" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
       </filter>
     </defs>` +
    grid +
    (area ? `<path d="${area}" fill="url(#wpfill)"/>` : "") +
    `<path d="${line}" fill="none" stroke="var(--accent)" stroke-width="2.5" stroke-linejoin="round" stroke-linecap="round" filter="url(#glow)"/>` +
    blunderDots +
    `<line id="cursor-line" x1="0" y1="0" x2="0" y2="${H}" stroke="var(--gold)" stroke-width="1.5" stroke-dasharray="3 3"/>`;
  svg.onclick = (e) => {
    const r = svg.getBoundingClientRect();
    const i = Math.round(((e.clientX - r.left) / r.width) * (n - 1));
    selectPly(Math.max(0, Math.min(n - 1, i)));
  };
  $("graph-legend").innerHTML = `上=あなた有利 / 下=AI有利 ・ <span style="color:var(--bad)">●</span> 悪手(勝率-15%超)`;
}

function renderMoveList() {
  const list = $("move-list");
  list.innerHTML = "";
  review.record.moves.forEach((mv, i) => {
    const row = document.createElement("div");
    row.className = "mv";
    row.dataset.ply = i;
    const ev = mv.eval;
    const delta = ev ? ev.delta : 0;
    const dcls = delta < -0.001 ? "delta-bad" : (delta > 0.001 ? "delta-good" : "");
    const who = review.record.players.find((p) => p.id === mv.player);
    row.innerHTML = `<span class="ply">${i + 1}</span>` +
      `<span class="who">${who ? who.name.slice(0, 3) : mv.player}</span>` +
      `<span>${labelFor(mv.notation)}</span>` +
      (ev && ev.blunder ? `<span class="blunder">⚠</span>` : "") +
      `<span class="dlt ${dcls}">${ev ? (delta * 100).toFixed(0) + "%" : ""}</span>`;
    row.onclick = () => selectPly(i);
    list.appendChild(row);
  });
}

async function selectPly(i) {
  review.ply = i;
  document.querySelectorAll("#move-list .mv").forEach((el) =>
    el.classList.toggle("sel", +el.dataset.ply === i));
  // cursor line on graph
  const n = review.record.moves.length;
  const cx = n <= 1 ? 0 : (i / (n - 1)) * 600;
  const cl = $("winprob-graph").querySelector("#cursor-line");
  if (cl) { cl.setAttribute("x1", cx); cl.setAttribute("x2", cx); }

  // board state BEFORE this move (the decision point)
  const res = await api("/api/replay-view", { record: review.record, ply: i });
  renderReviewBoard(res.view, review.record.moves[i]);
  renderPlyEval(review.record.moves[i], i);
}

function renderReviewBoard(view, mv) {
  const board = $("review-board");
  board.innerHTML = `<div style="color:var(--muted);font-size:12px;margin-bottom:6px">
    ${mv ? `第${review.ply + 1}手の局面（手番: ${view.players[view.toMove].name}）` : "最終局面"}</div>`;
  for (const tier of [3, 2, 1]) {
    const row = document.createElement("div");
    row.className = "tier-row";
    const label = document.createElement("div");
    label.className = "tier-label"; label.textContent = "L" + tier;
    row.appendChild(label);
    for (const card of view.board["L" + tier]) row.appendChild(cardEl(card));
    board.appendChild(row);
  }
  // bank summary line
  const bankLine = document.createElement("div");
  bankLine.style.marginTop = "8px";
  bankLine.innerHTML = "場札: " + TOKENS.map((c) => `${tokenChip(c)}×${view.bank[c]}`).join(" ");
  board.appendChild(bankLine);
  if (mv && mv.eval && mv.eval.bestMove) highlightBest(mv.eval.bestMove.notation, board);
  // also mark the played card (blue)
  const loc = boardLoc(mv ? mv.notation : "");
  if (loc) {
    const el = board.querySelector(`.card[data-tier="${loc.tier}"][data-slot="${loc.slot}"]`);
    if (el) el.style.outline = el.classList.contains("best") ? "" : "2px solid var(--accent)";
  }
}

function renderPlyEval(mv, i) {
  const panel = $("ply-eval");
  if (!mv || !mv.eval) { panel.innerHTML = "<h3>最終局面</h3>"; return; }
  const ev = mv.eval;
  const who = review.record.players.find((p) => p.id === mv.player);
  const dcls = ev.delta < -0.001 ? "delta-bad" : "delta-good";
  const tops = ev.topMoves.map((t) => {
    const isBest = ev.bestMove && t.notation === ev.bestMove.notation;
    return `<div class="tm ${isBest ? "best" : ""}"><span>${labelFor(t.notation)}</span>` +
      `<span>${(t.winProb * 100).toFixed(0)}% <small style="color:var(--muted)">(${t.visits})</small></span></div>`;
  }).join("");
  panel.innerHTML = `
    <h3>第${i + 1}手 — ${who ? who.name : mv.player}</h3>
    <div class="row"><span>指した手</span><b>${labelFor(mv.notation)}</b></div>
    <div class="row"><span>局面勝率(手番側)</span><span class="big">${(ev.winProbBefore * 100).toFixed(0)}%</span></div>
    <div class="row"><span>この手の評価</span><span>${(ev.playedWinProb * 100).toFixed(0)}%</span></div>
    <div class="row"><span>最善との差</span><span class="${dcls}">${(ev.delta * 100).toFixed(0)}%${ev.blunder ? " ⚠悪手" : ""}</span></div>
    <div class="row"><span>最善手</span><b style="color:var(--warn)">${ev.bestMove ? labelFor(ev.bestMove.notation) : "—"}</b></div>
    <div class="topmoves"><div style="color:var(--muted);font-size:12px;margin-bottom:4px">候補手 (訪問数)</div>${tops}</div>`;
}

// ---------------------------------------------------------------------------
// Tabs & lifecycle
// ---------------------------------------------------------------------------
function switchTab(which) {
  $("tab-play").classList.toggle("active", which === "play");
  $("tab-review").classList.toggle("active", which === "review");
  $("play-view").classList.toggle("hidden", which !== "play");
  $("review-view").classList.toggle("hidden", which !== "review");
}

async function newGame() {
  const sims = parseInt($("ai-sims").value, 10) || 600;
  game = await api("/api/games", { aiSims: sims });
  selectedGems = [];
  review = null;
  $("tab-review").disabled = true;
  $("thinking").classList.add("hidden");
  switchTab("play");
  renderPlay();
  // if AI is the first player, let it move
  maybeAiMove();
}

$("new-game").onclick = newGame;
$("analyze-position").onclick = analyzePosition;
$("start-review").onclick = async () => {
  $("tab-review").disabled = false;
  await startReview();
};
$("tab-play").onclick = () => switchTab("play");
$("tab-review").onclick = () => { if (review) switchTab("review"); };

// auto-start a game on load
newGame().catch((e) => flash("起動エラー: " + e.message));
