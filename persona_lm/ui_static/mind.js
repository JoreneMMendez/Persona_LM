// PersonaLM UI, stage U2b: the memory graph (pan/zoom, click a node), the Mood page (mood map + history chart) and the Person card editor.
// All text from files goes on the page with textContent only. The graph and mood are read-only; the card changes one line at a time.
import { h, api, toast } from './app.js';

const NS = 'http://www.w3.org/2000/svg';
const sv = (tag, attrs, ...kids) => {
  const el = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs || {})) if (v != null && v !== false) el.setAttribute(k, v);
  for (const k of kids.flat()) if (k != null) el.append(k.nodeType ? k : document.createTextNode(String(k)));
  return el;
};
const clamp = (x, a, b) => Math.max(a, Math.min(b, x));

// ---------------------------------------------------------------- graph
function layout(nodes, edges, W, H) {
  const n = nodes.length, idx = new Map(nodes.map((d, i) => [d.id, i]));
  let seed = 12345; const rnd = () => ((seed = (seed * 1664525 + 1013904223) >>> 0) / 4294967296);   // same picture every time
  const P = nodes.map((_, i) => ({ x: W / 2 + Math.cos(i * 2.4) * (40 + 6 * i) * 0.6 + rnd() * 20, y: H / 2 + Math.sin(i * 2.4) * (40 + 6 * i) * 0.6 + rnd() * 20, vx: 0, vy: 0 }));
  const E = edges.map((e) => [idx.get(e.a), idx.get(e.b), e.count]).filter((e) => e[0] != null && e[1] != null);
  const k = Math.sqrt((W * H) / Math.max(n, 1)) * 0.55;
  for (let it = 0; it < 260; it++) {
    const cool = 1 - it / 260;
    for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) {
      let dx = P[i].x - P[j].x, dy = P[i].y - P[j].y, d2 = dx * dx + dy * dy + 0.01, d = Math.sqrt(d2), f = (k * k) / d2;
      P[i].vx += dx / d * f; P[i].vy += dy / d * f; P[j].vx -= dx / d * f; P[j].vy -= dy / d * f;
    }
    for (const [a, b, c] of E) {
      const dx = P[b].x - P[a].x, dy = P[b].y - P[a].y, d = Math.sqrt(dx * dx + dy * dy) + 0.01, f = (d - k) * 0.04 * Math.min(3, c);
      P[a].vx += dx / d * f; P[a].vy += dy / d * f; P[b].vx -= dx / d * f; P[b].vy -= dy / d * f;
    }
    for (const p of P) {
      p.vx += (W / 2 - p.x) * 0.01; p.vy += (H / 2 - p.y) * 0.01;
      const sp = Math.sqrt(p.vx * p.vx + p.vy * p.vy) || 1, cap = 30 * cool + 1;
      p.x += p.vx / sp * Math.min(sp, cap); p.y += p.vy / sp * Math.min(sp, cap); p.vx = p.vy = 0;
    }
  }
  return P;
}

export async function graphView(box) {
  box.replaceChildren(h('div', { class: 'skeleton', style: 'height:420px' }));
  let g;
  try { g = await api('/api/graph'); } catch (e) { box.replaceChildren(h('div', { class: 'callout bad', role: 'alert' }, h('p', null, e.message))); return; }
  if (!g.exists || !g.nodes.length) {
    box.replaceChildren(h('div', { class: 'empty' }, h('h2', null, 'Nothing to draw yet'), h('p', null, 'The graph appears once memories are linked to people or topics. Write [[Name]] in a memory to link it.'))); return;
  }
  const W = 1000, H = 640, P = layout(g.nodes, g.edges, W, H);
  const max = Math.max(...g.nodes.map((d) => d.score), 0.0001), top = new Set(g.nodes.slice(0, 40).map((d) => d.id));
  const view = { x: 0, y: 0, w: W, h: H };
  const svg = sv('svg', { class: 'graph-svg', viewBox: '0 0 ' + W + ' ' + H, role: 'group', 'aria-label': 'Memory graph. Use the node list for a text version.' });
  const gEdges = sv('g'), gNodes = sv('g'), byId = new Map();
  g.nodes.forEach((d, i) => byId.set(d.id, { d, p: P[i], r: 7 + 20 * Math.sqrt(d.score / max), el: null, lab: null, nbr: new Set() }));
  const lines = g.edges.map((e) => {
    const a = byId.get(e.a), b = byId.get(e.b); if (!a || !b) return null;
    a.nbr.add(e.b); b.nbr.add(e.a);
    const l = sv('line', { class: 'ge', x1: a.p.x, y1: a.p.y, x2: b.p.x, y2: b.p.y, 'stroke-width': 1 + Math.min(4, e.count) * 0.8 }); gEdges.append(l); return [e, l];
  }).filter(Boolean);
  let selected = null;
  const panel = h('div', { class: 'card graph-panel', 'aria-live': 'polite' }, h('p', { class: 'muted small' }, 'Click a circle (or pick one from the list) to see what the AI remembers about it.'));

  async function select(id) {
    selected = id;
    const n = byId.get(id);
    for (const [k, v] of byId) {
      v.el.classList.toggle('sel', k === id); v.el.classList.toggle('dim', k !== id && !n.nbr.has(k));
      v.lab.style.display = (k === id || n.nbr.has(k) || top.has(k)) ? '' : 'none';
    }
    for (const [e, l] of lines) { const on = e.a === id || e.b === id; l.classList.toggle('on', on); l.classList.toggle('dim', !on); }
    panel.replaceChildren(h('div', { class: 'row' }, h('h3', null, n.d.name), h('span', { class: 'badge' }, n.d.kind), h('span', { class: 'badge accent' }, n.d.memories + ' active memories')),
      h('div', { class: 'row small muted' }, n.nbr.size ? 'Linked with: ' : 'No links yet', ...[...n.nbr].map((k) => h('button', { class: 'chip chip-btn', type: 'button', onclick: () => select(k) }, byId.get(k).d.name))),
      h('div', { class: 'spinner', role: 'status', 'aria-label': 'Loading' }));
    try {
      const r = await api('/api/memory/list?' + new URLSearchParams({ q: n.d.name, status: 'active', sort: 'strength', limit: '30' }));
      if (selected !== id) return;
      const mine = r.items.filter((m) => m.nodes.some((x) => x.name === n.d.name && x.kind === n.d.kind));
      panel.lastChild.replaceWith(mine.length ? h('ul', { class: 'list' }, mine.map((m) => h('li', null, h('div', { class: 'grow' }, m.text), h('span', { class: 'path' }, m.id)))) : h('p', { class: 'muted small' }, 'No active memory text to show.'));
    } catch (e) { if (selected === id) panel.lastChild.replaceWith(h('p', { class: 'small', role: 'alert' }, e.message)); }
  }
  function clear() {
    selected = null;
    for (const v of byId.values()) { v.el.classList.remove('sel', 'dim'); v.lab.style.display = top.has(v.d.id) ? '' : 'none'; }
    for (const [, l] of lines) l.classList.remove('on', 'dim');
    panel.replaceChildren(h('p', { class: 'muted small' }, 'Click a circle (or pick one from the list) to see what the AI remembers about it.'));
  }
  for (const v of byId.values()) {
    const el = sv('g', { class: 'gn gn-' + v.d.kind, tabindex: '0', role: 'button', 'aria-label': v.d.name + ', ' + v.d.kind + ', ' + v.d.memories + ' memories' },
      sv('circle', { cx: v.p.x, cy: v.p.y, r: v.r }));
    const lab = sv('text', { x: v.p.x, y: v.p.y + v.r + 12, 'text-anchor': 'middle', class: 'gl' }, v.d.name);
    lab.style.display = top.has(v.d.id) ? '' : 'none';
    el.addEventListener('click', (e) => { e.stopPropagation(); select(v.d.id); });
    el.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); select(v.d.id); } });
    v.el = el; v.lab = lab; gNodes.append(el, lab);
  }
  svg.append(gEdges, gNodes);

  // pan (drag the background), zoom (wheel or buttons)
  const apply = () => svg.setAttribute('viewBox', [view.x, view.y, view.w, view.h].map((x) => x.toFixed(1)).join(' '));
  function zoom(f, cx, cy) {
    const nw = clamp(view.w * f, W / 8, W * 2), r = nw / view.w; cx = cx ?? view.x + view.w / 2; cy = cy ?? view.y + view.h / 2;
    view.x = cx - (cx - view.x) * r; view.y = cy - (cy - view.y) * r; view.w = nw; view.h = nw * H / W; apply();
  }
  const toSvg = (e) => { const b = svg.getBoundingClientRect(); return [view.x + (e.clientX - b.left) / b.width * view.w, view.y + (e.clientY - b.top) / b.height * view.h]; };
  svg.addEventListener('wheel', (e) => { e.preventDefault(); const [cx, cy] = toSvg(e); zoom(e.deltaY < 0 ? 0.85 : 1.18, cx, cy); }, { passive: false });
  let drag = null, moved = false;
  svg.addEventListener('pointerdown', (e) => { if (e.target.closest('.gn')) return; drag = { x: e.clientX, y: e.clientY, vx: view.x, vy: view.y }; moved = false; svg.setPointerCapture(e.pointerId); svg.classList.add('grabbing'); });
  svg.addEventListener('pointermove', (e) => {
    if (!drag) return; const b = svg.getBoundingClientRect(), dx = e.clientX - drag.x, dy = e.clientY - drag.y;
    if (Math.abs(dx) + Math.abs(dy) > 3) moved = true;
    view.x = drag.vx - dx / b.width * view.w; view.y = drag.vy - dy / b.height * view.h; apply();
  });
  const end = () => { drag = null; svg.classList.remove('grabbing'); };
  svg.addEventListener('pointerup', (e) => { const was = moved; end(); if (!was && !e.target.closest('.gn')) clear(); });
  svg.addEventListener('pointercancel', end);
  const reset = () => { Object.assign(view, { x: 0, y: 0, w: W, h: H }); apply(); };

  const kinds = [...new Set(g.nodes.map((d) => d.kind))];
  const list = h('details', { class: 'card' }, h('summary', null, 'Node list (text version)'),
    h('ul', { class: 'list' }, g.nodes.map((d) => h('li', null, h('button', { class: 'btn ghost', type: 'button', onclick: () => select(d.id) }, d.name), h('span', { class: 'badge' }, d.kind), h('span', { class: 'muted small' }, d.memories + ' memories')))));
  box.replaceChildren(h('div', { class: 'stack' },
    h('div', { class: 'row' }, h('span', { class: 'badge accent' }, g.shown + (g.total_nodes > g.shown ? ' of ' + g.total_nodes : '') + ' nodes'), h('span', { class: 'badge' }, g.edges.length + ' links'),
      ...kinds.map((k) => h('span', { class: 'legend gk-' + k }, k)), h('div', { class: 'spacer' }),
      h('button', { class: 'btn icon', type: 'button', 'aria-label': 'Zoom in', onclick: () => zoom(0.8) }, '+'), h('button', { class: 'btn icon', type: 'button', 'aria-label': 'Zoom out', onclick: () => zoom(1.25) }, '−'),
      h('button', { class: 'btn', type: 'button', onclick: () => { reset(); clear(); } }, 'Reset view')),
    h('p', { class: 'small muted' }, 'Bigger circle = stronger memories. A line means two nodes share a memory. Drag to move, scroll to zoom. Only active memories are drawn.'),
    h('div', { class: 'graph-wrap card' }, svg), panel, list));
}

// ---------------------------------------------------------------- mood
const fmtTime = (t) => { try { return new Date(t * 1000).toLocaleString(undefined, { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' }); } catch (e) { return ''; } };

function moodMap(m) {
  const S = 220, pad = 24, X = (v) => pad + (v + 1) / 2 * (S - 2 * pad), Y = (a) => S - pad - (a + 1) / 2 * (S - 2 * pad);
  const svg = sv('svg', { class: 'mood-map', viewBox: '0 0 ' + S + ' ' + S, role: 'img', 'aria-label': 'Mood map. Now: ' + m.label + ', valence ' + m.valence + ', arousal ' + m.arousal });
  svg.append(sv('rect', { x: pad, y: pad, width: S - 2 * pad, height: S - 2 * pad, class: 'mm-box' }), sv('line', { x1: X(0), y1: pad, x2: X(0), y2: S - pad, class: 'mm-axis' }), sv('line', { x1: pad, y1: Y(0), x2: S - pad, y2: Y(0), class: 'mm-axis' }),
    sv('text', { x: S / 2, y: S - 6, 'text-anchor': 'middle', class: 'mm-t' }, 'unpleasant ← valence → pleasant'), sv('text', { x: 10, y: S / 2, class: 'mm-t', transform: 'rotate(-90 10 ' + S / 2 + ')', 'text-anchor': 'middle' }, 'calm ← arousal → excited'));
  for (const [name, [v, a]] of Object.entries(m.labels)) svg.append(sv('circle', { cx: X(v), cy: Y(a), r: 3, class: 'mm-dot' }), sv('text', { x: X(v), y: Y(a) - 6, 'text-anchor': 'middle', class: 'mm-l' }, name));
  svg.append(sv('circle', { cx: X(m.baseline[0]), cy: Y(m.baseline[1]), r: 5, class: 'mm-base' }), sv('circle', { cx: X(m.valence), cy: Y(m.arousal), r: 7, class: 'mm-now' }));
  return svg;
}

function moodChart(m) {
  const hs = m.history, W = 560, H = 200, L = 34, R = 10, T = 12, B = 22;
  if (hs.length < 2) return h('p', { class: 'muted small' }, hs.length ? 'One message has moved the mood so far. A chart needs at least two.' : 'No messages have moved the mood yet.');
  const X = (i) => L + i / (hs.length - 1) * (W - L - R), Y = (v) => T + (1 - (v + 1) / 2) * (H - T - B);
  const val = hs.map((x) => (m.labels[x.label] || [0, 0])[0]), aro = hs.map((x) => (m.labels[x.label] || [0, 0])[1]);
  const path = (arr) => arr.map((v, i) => (i ? 'L' : 'M') + X(i).toFixed(1) + ' ' + Y(v).toFixed(1)).join(' ');
  const svg = sv('svg', { class: 'mood-chart', viewBox: '0 0 ' + W + ' ' + H, role: 'img', 'aria-label': 'Valence and arousal of the last ' + hs.length + ' feelings that moved the mood' });
  for (const v of [-1, 0, 1]) svg.append(sv('line', { x1: L, x2: W - R, y1: Y(v), y2: Y(v), class: v === 0 ? 'mm-axis' : 'mc-grid' }), sv('text', { x: L - 6, y: Y(v) + 4, 'text-anchor': 'end', class: 'mm-t' }, v > 0 ? '+1' : v < 0 ? '−1' : '0'));
  svg.append(sv('path', { d: path(val), class: 'mc-val', fill: 'none' }), sv('path', { d: path(aro), class: 'mc-aro', fill: 'none' }));
  hs.forEach((x, i) => svg.append(sv('circle', { cx: X(i), cy: Y(val[i]), r: 3.2, class: 'mc-vd' }, sv('title', null, x.label + ' · ' + fmtTime(x.time)))));
  return h('div', null, svg, h('div', { class: 'row small' }, h('span', { class: 'legend lg-val' }, 'valence (pleasant)'), h('span', { class: 'legend lg-aro' }, 'arousal (excited)'),
    h('span', { class: 'muted' }, 'oldest → newest, last ' + hs.length + ' messages that moved the mood')));
}

async function moodCard() {
  let m;
  try { m = await api('/api/mood'); } catch (e) { return h('div', { class: 'callout bad', role: 'alert' }, h('p', null, e.message)); }
  if (!m.exists) return h('div', { class: 'empty' }, h('h2', null, 'No mood yet'), h('p', null, 'mood.json appears when you chat with "Use a mood" ticked (Chat page, load screen), or from the terminal.'));
  return h('section', { class: 'card', 'aria-labelledby': 'h-mood' },
    h('div', { class: 'card-head' }, h('h2', { id: 'h-mood' }, 'Mood now'), h('span', { class: 'badge accent' }, m.label), h('span', { class: 'badge' }, Math.round(m.intensity * 100) + '% from its resting point'), h('span', { class: 'badge' }, m.turns + ' messages seen')),
    h('p', { class: 'small muted' }, 'Valence ' + m.valence.toFixed(2) + ', arousal ' + m.arousal.toFixed(2) + '. Only what you type moves the mood, and it drifts back to rest over hours. This view never changes it.'),
    h('div', { class: 'mood-grid' }, moodMap(m), h('div', null, h('h3', null, 'History'), moodChart(m),
      m.history.length ? h('ul', { class: 'list' }, m.history.slice(-6).reverse().map((x) => h('li', null, h('span', { class: 'badge' }, x.label), h('span', { class: 'muted small' }, 'sure ' + Math.round((x.confidence || 0) * 100) + '%'), h('span', { class: 'muted small' }, fmtTime(x.time))))) : null)));
}

// ---------------------------------------------------------------- person card
function cardSection(sec, reload) {
  const rows = sec.lines.map((ln, i) => {
    const old = ln.text, txt = h('div', { class: 'grow card-text' }, ln.text), tagSel = h('select', { 'aria-label': 'Line type for: ' + old.slice(0, 40) }, ['owner', 'auto', 'no'].map((t) => h('option', { value: t }, t)));
    tagSel.value = ln.tag;
    tagSel.addEventListener('change', async () => { try { await api('/api/person/edit', { section: sec.title, index: i, old, tag: tagSel.value }); toast('Line type changed', 'ok'); await reload(); } catch (e) { toast(e.message, 'bad'); tagSel.value = ln.tag; } });
    const li = h('li', null, txt, tagSel);
    const edit = h('button', { class: 'btn ghost', type: 'button', 'aria-label': 'Edit line: ' + old.slice(0, 40) }, 'Edit');
    const del = h('button', { class: 'btn ghost', type: 'button', 'aria-label': 'Remove line: ' + old.slice(0, 40) }, 'Remove');
    let armed = 0;
    del.addEventListener('click', async () => {
      if (!armed) { armed = setTimeout(() => { armed = 0; del.classList.remove('danger'); del.textContent = 'Remove'; }, 4000); del.classList.add('danger'); del.textContent = 'Really remove?'; return; }
      clearTimeout(armed);
      try { await api('/api/person/remove', { section: sec.title, index: i, old }); toast('Line removed', 'ok'); await reload(); } catch (e) { armed = 0; toast(e.message, 'bad'); }
    });
    edit.addEventListener('click', () => {
      const inp = h('input', { type: 'text', value: old, maxlength: '400', 'aria-label': 'Edit line' });
      const save = async () => { try { await api('/api/person/edit', { section: sec.title, index: i, old, text: inp.value }); toast('Saved', 'ok'); await reload(); } catch (e) { toast(e.message, 'bad'); } };
      inp.addEventListener('keydown', (e) => { if (e.key === 'Enter') save(); else if (e.key === 'Escape') reload(); });
      txt.replaceChildren(inp); inp.focus(); inp.select();
      edit.textContent = 'Save'; edit.onclick = save; del.textContent = 'Cancel'; del.onclick = () => reload();
    });
    li.append(h('div', { class: 'row', style: 'gap:2px;flex-wrap:nowrap' }, edit, del));
    return li;
  });
  const inp = h('input', { type: 'text', maxlength: '400', placeholder: 'Add a line to ' + sec.title, 'aria-label': 'New line for ' + sec.title });
  const add = async () => { if (!inp.value.trim()) return; try { await api('/api/person/add', { section: sec.title, text: inp.value }); toast('Added', 'ok'); await reload(); } catch (e) { toast(e.message, 'bad'); } };
  inp.addEventListener('keydown', (e) => { if (e.key === 'Enter') { e.preventDefault(); add(); } });
  return h('section', { class: 'card', 'aria-label': sec.title }, h('div', { class: 'card-head' }, h('h3', null, sec.title), sec.auto ? h('span', { class: 'badge info', title: 'A rebuild can replace lines marked auto' }, 'can be built from data') : null),
    rows.length ? h('ul', { class: 'list' }, rows) : h('p', { class: 'small muted' }, 'Empty.'), h('div', { class: 'row' }, h('div', { class: 'field grow' }, inp), h('button', { class: 'btn', type: 'button', onclick: add }, 'Add')));
}

async function personBlock(box) {
  const reload = async () => {
    let p;
    try { p = await api('/api/person'); } catch (e) { box.replaceChildren(h('div', { class: 'callout bad', role: 'alert' }, h('p', null, e.message))); return; }
    box.replaceChildren(h('div', { class: 'stack' },
      h('div', { class: 'card-head' }, h('h2', null, 'Person card'), p.exists ? null : h('span', { class: 'badge' }, 'not created yet')),
      h('p', { class: 'small muted' }, 'A short description of you that goes into the prompt. Lines you write are yours and never replaced. Lines marked auto were built from data and a rebuild replaces them; change the type to owner to keep one, or to no to stop it coming back. Building from data stays in the terminal: ', h('code', null, 'python -m persona_lm person build')),
      ...p.sections.filter((s) => s.title).map((s) => cardSection(s, reload))));
  };
  await reload();
}

export async function moodPage(main) {
  const moodBox = h('div'), cardBox = h('div');
  main.replaceChildren(h('div', { class: 'stack' }, h('p', { class: 'lead' }, 'How the AI feels right now, and the short card about you that it reads.'), moodBox, cardBox));
  moodBox.replaceChildren(h('div', { class: 'skeleton', style: 'height:200px' }));
  cardBox.replaceChildren(h('div', { class: 'skeleton', style: 'height:200px' }));
  moodBox.replaceChildren(await moodCard());
  await personBlock(cardBox);
}
