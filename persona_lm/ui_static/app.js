// PersonaLM UI (stages U0 to U2a): shell, theme, toasts, shortcuts, Settings/Privacy page, design kit.
// Everything is built with textContent / createElement (no innerHTML with data), and nothing is loaded from outside this server.

import { chatPage } from './chat.js';
import { memoryPage } from './memory.js';
import { moodPage } from './mind.js';
import { personasPage } from './personas.js';
import { dataPage } from './data.js';

const TOKEN = document.querySelector('meta[name="plm-token"]').content;

// ---------- tiny helpers
export function h(tag, attrs, ...kids) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === false || v == null) continue;
    if (k === 'class') el.className = v;
    else if (k === 'style') el.style.cssText = v;            // CSSOM, not setAttribute: the page's CSP forbids inline style attributes
    else if (k.startsWith('on') && typeof v === 'function') el.addEventListener(k.slice(2), v);
    else el.setAttribute(k, v === true ? '' : v);
  }
  for (const kid of kids.flat()) if (kid != null && kid !== false) el.append(kid.nodeType ? kid : document.createTextNode(String(kid)));
  return el;
}

const ICONS = { // static, trusted strings (24x24 stroke icons)
  chat: 'M21 12a8 8 0 0 1-11.6 7.1L4 20l1-4.6A8 8 0 1 1 21 12z',
  graph: 'M6 7a2 2 0 1 0 0-.01M18 5a2 2 0 1 0 0-.01M17 18a2 2 0 1 0 0-.01M7.5 8.5l8 -2.5M7 9l9 8M17.5 7l-.5 9',
  heart: 'M12 20s-7-4.4-7-10a4 4 0 0 1 7-2.6A4 4 0 0 1 19 10c0 5.6-7 10-7 10z',
  layers: 'M12 3l9 5-9 5-9-5 9-5zM3 13l9 5 9-5',
  database: 'M4 6c0-1.7 3.6-3 8-3s8 1.3 8 3-3.6 3-8 3-8-1.3-8-3zM4 6v12c0 1.7 3.6 3 8 3s8-1.3 8-3V6M4 12c0 1.7 3.6 3 8 3s8-1.3 8-3',
  chart: 'M4 20V4M4 20h16M8 16l4-5 3 3 5-7',
  shield: 'M12 3l8 3v6c0 4.5-3.2 8-8 9-4.8-1-8-4.5-8-9V6l8-3z',
  inbox: 'M3 13l3-8h12l3 8v6H3v-6zM3 13h5l1 3h6l1-3h5',
  compare: 'M9 4v16M15 4v16M4 8h5M15 16h5',
  box: 'M3 7l9-4 9 4v10l-9 4-9-4V7zM3 7l9 4 9-4M12 11v10',
  lock: 'M6 11V8a6 6 0 0 1 12 0v3M5 11h14v10H5V11z',
  menu: 'M4 7h16M4 12h16M4 17h16',
  sun: 'M12 16a4 4 0 1 0 0-8 4 4 0 0 0 0 8zM12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4',
  moon: 'M21 13a9 9 0 1 1-10-10 7 7 0 0 0 10 10z',
  auto: 'M12 21a9 9 0 1 0 0-18 9 9 0 0 0 0 18zM12 3v18M12 3a9 9 0 0 1 0 18',
  check: 'M5 12.5l4.5 4.5L19 7.5',
  info: 'M12 21a9 9 0 1 0 0-18 9 9 0 0 0 0 18zM12 11v5M12 8h.01',
};
export function icon(name) {
  const s = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  s.setAttribute('viewBox', '0 0 24 24'); s.setAttribute('fill', 'none'); s.setAttribute('stroke', 'currentColor');
  s.setAttribute('stroke-width', '1.8'); s.setAttribute('stroke-linecap', 'round'); s.setAttribute('stroke-linejoin', 'round');
  s.setAttribute('aria-hidden', 'true');
  const p = document.createElementNS('http://www.w3.org/2000/svg', 'path');
  p.setAttribute('d', ICONS[name] || ICONS.info); s.append(p);
  return s;
}

// ---------- API (every call carries the session token; nothing goes anywhere but this server)
export async function api(path, body) {
  const opt = { headers: { 'X-PLM-Token': TOKEN }, cache: 'no-store' };
  if (body !== undefined) { opt.method = 'POST'; opt.headers['Content-Type'] = 'application/json'; opt.body = JSON.stringify(body); }
  let res;
  try { res = await fetch(path, opt); } catch (e) { throw new Error('Cannot reach the PersonaLM server. Is the terminal window still running?'); }
  let data = {};
  try { data = await res.json(); } catch (e) { /* not JSON */ }
  if (!res.ok || data.ok === false) throw new Error(data.error || ('Request failed (' + res.status + ')'));
  return data;
}

// ---------- toasts
export function toast(msg, kind) {
  const box = document.getElementById('toasts');
  const t = h('div', { class: 'toast ' + (kind || '') }, msg);
  box.append(t);
  setTimeout(() => { t.classList.add('out'); setTimeout(() => t.remove(), 300); }, kind === 'bad' ? 6000 : 3200);
}

// ---------- theme: system -> light -> dark
const MODES = ['system', 'light', 'dark'];
function getMode() { try { return localStorage.getItem('plm-theme') || 'system'; } catch (e) { return 'system'; } }
function applyTheme(mode, announce) {
  const root = document.documentElement;
  if (mode === 'system') root.removeAttribute('data-theme'); else root.setAttribute('data-theme', mode);
  const btn = document.getElementById('theme-btn');
  btn.replaceChildren(icon(mode === 'dark' ? 'moon' : mode === 'light' ? 'sun' : 'auto'));
  btn.setAttribute('aria-label', 'Theme: ' + mode + '. Switch theme');
  try { localStorage.setItem('plm-theme', mode); } catch (e) { /* private mode: theme just is not remembered */ }
  if (announce) toast('Theme: ' + mode);
}
function cycleTheme() { applyTheme(MODES[(MODES.indexOf(getMode()) + 1) % MODES.length], true); }

// ---------- sections and pages
let SECTIONS = [];
const PLANNED = {
  chat: ['Talk to the AI with message bubbles, a token counter for the 512-token window, a settings drawer and a live mood badge.', 'python -m persona_lm chat --ckpt work\\run_chat2\\ckpt.npz --tokenizer work\\tok.json'],
  memory: ['A graph you can pan and zoom, a searchable list, add / edit / pin / forget, and a maintenance preview.', 'python -m persona_lm memory list'],
  mood: ['View and edit the mood and the Person card, with a mood history chart.', 'python -m persona_lm mood show'],
  prompt: ['See which memories and old-chat lines went into the prompt, and switch personas.', 'python -m persona_lm persona list'],
  data: ['Messenger scan and import wizard, the chat cleaner with a before/after report, duplicate report and privacy audit.', 'python clean_chat.py --input FILE'],
  training: ['The pipeline checklist, start and stop of training, a live loss chart and checkpoints.', 'python -m persona_lm status --dir work'],
  identity: ['Read-only core, versions, lineage check and probe scores. Protected changes stay typed-confirmation only.', 'python -m persona_lm identity show'],
  proposals: ['Proposals with a diff. Approving keeps the existing confirmation rules.', 'python -m persona_lm proposal list'],
  models: ['Pick a run or checkpoint and compare two checkpoints on the same prompts.', 'python -m persona_lm generate --ckpt CKPT --tokenizer TOK --prompt "hi"'],
  export: ['Export and verify a package, with the privacy warning and an explicit confirm.', 'python -m persona_lm export-package --out my_ai_package'],
};


function plannedPage(sec) {
  const [what, cmd] = PLANNED[sec.id] || ['', ''];
  return h('div', { class: 'stack' },
    h('div', { class: 'empty' },
      (() => { const i = icon(sec.icon); i.classList.add('ico'); return i; })(),
      h('h2', null, sec.label + ' arrives in stage ' + sec.stage),
      h('p', null, what),
      h('p', { class: 'small' }, 'Nothing here is a mock-up: this page stays empty until the working version is built. Until then, the terminal command is:'),
      h('p', null, h('code', null, cmd))));
}

async function settingsPage(main) {
  main.replaceChildren(h('div', { class: 'stack' }, h('div', { class: 'skeleton', style: 'height:120px' }), h('div', { class: 'skeleton', style: 'height:260px' })));
  let info, priv;
  try { [info, priv] = await Promise.all([api('/api/info'), api('/api/privacy')]); }
  catch (e) { main.replaceChildren(h('div', { class: 'callout bad', role: 'alert' }, h('p', null, e.message))); return; }

  const stored = h('ul', { class: 'list' }, priv.stored.map((it) => h('li', null,
    h('div', { class: 'grow' }, h('div', { class: 'name' }, it.label), h('div', { class: 'small muted' }, it.holds), h('div', { class: 'path' }, it.path + (it.kind === 'folder' ? '/' : ''))),
    h('div', { class: 'row', style: 'gap:6px;justify-content:flex-end' },
      h('span', { class: 'badge ' + (it.sensitivity === 'high' ? 'warn' : it.sensitivity === 'medium' ? 'info' : '') }, it.sensitivity === 'high' ? 'private' : it.sensitivity === 'medium' ? 'personal' : 'public'),
      h('span', { class: 'badge ' + (it.present ? 'ok' : '') }, it.present ? 'found' : 'not here')))));

  main.replaceChildren(h('div', { class: 'stack' },
    h('p', { class: 'lead' }, 'Everything this UI does happens on this computer. This page says what is stored where, and what never leaves.'),
    h('section', { class: 'card', 'aria-labelledby': 'h-never' },
      h('div', { class: 'card-head' }, h('h2', { id: 'h-never' }, 'What never leaves your computer')),
      h('ul', { class: 'checks' }, priv.never_leaves.map((t) => h('li', null, icon('check'), h('span', null, t))))),
    h('section', { class: 'card', 'aria-labelledby': 'h-stored' },
      h('div', { class: 'card-head' }, h('h2', { id: 'h-stored' }, 'What is stored where'), h('span', { class: 'muted small' }, 'project folder: ' + priv.root_name)),
      h('p', { class: 'small muted' }, 'The UI only checks whether these exist. ' + priv.private_folder + '.'),
      stored),
    h('section', { class: 'card', 'aria-labelledby': 'h-about' },
      h('div', { class: 'card-head' }, h('h2', { id: 'h-about' }, 'This session')),
      h('div', { class: 'row' },
        h('span', { class: 'badge accent' }, 'PersonaLM ' + info.version), h('span', { class: 'badge' }, 'UI stage ' + info.stage),
        h('span', { class: 'badge ok' }, '127.0.0.1 only'), h('span', { class: 'badge' }, 'token required')),
      h('p', { class: 'small muted', style: 'margin-top:12px' }, 'The page address holds a one-time key made when you started the UI. Closing the terminal ends the session and the key stops working.'),
      h('div', { class: 'row' },
        h('a', { class: 'btn', href: '#/design' }, 'Design kit'),
        h('button', { class: 'btn', onclick: cycleTheme }, 'Switch theme'),
        h('button', { class: 'btn', onclick: () => toast('Server answered: still connected', 'ok') }, 'Test toast')))));
}

function designPage(main) {
  const sw = ['--bg', '--surface', '--surface-2', '--line', '--ink', '--ink-3', '--accent', '--accent-soft', '--ok', '--warn', '--bad', '--info'];
  main.replaceChildren(h('div', { class: 'stack' },
    h('p', { class: 'lead' }, 'Components every later stage reuses. Switch the theme (key t) to check both.'),
    h('section', { class: 'card' }, h('h2', null, 'Colour tokens'), h('div', { class: 'kit-swatches', style: 'margin-top:12px' }, sw.map((v) => h('div', { class: 'swatch', style: 'background:var(' + v + ')' }, v.slice(2))))),
    h('section', { class: 'card' }, h('h2', null, 'Buttons and badges'), h('div', { class: 'row', style: 'margin-top:12px' },
      h('button', { class: 'btn primary' }, 'Primary'), h('button', { class: 'btn' }, 'Default'), h('button', { class: 'btn ghost' }, 'Ghost'), h('button', { class: 'btn danger' }, 'Danger'), h('button', { class: 'btn', disabled: true }, 'Disabled')),
      h('div', { class: 'row', style: 'margin-top:12px' }, h('span', { class: 'badge' }, 'neutral'), h('span', { class: 'badge accent' }, 'accent'), h('span', { class: 'badge ok' }, 'ok'), h('span', { class: 'badge warn' }, 'warning'), h('span', { class: 'badge bad' }, 'error'), h('span', { class: 'badge info' }, 'info'))),
    h('section', { class: 'card' }, h('h2', null, 'Form field'), h('div', { class: 'field', style: 'margin-top:12px;max-width:360px' },
      h('label', { for: 'kit-in' }, 'Temperature'), h('input', { id: 'kit-in', type: 'number', value: '0.8', step: '0.1' }), h('span', { class: 'hint' }, 'Lower is calmer, higher is more varied.'))),
    h('section', { class: 'card' }, h('h2', null, 'States'),
      h('div', { class: 'stack', style: 'margin-top:12px' },
        h('div', { class: 'callout' }, h('p', null, 'Info: something worth knowing.')), h('div', { class: 'callout warn' }, h('p', null, 'Warning: validation loss went up.')),
        h('div', { class: 'callout bad', role: 'alert' }, h('p', null, 'Error: the file could not be read.')),
        h('div', { class: 'row' }, h('span', { class: 'spinner', role: 'status', 'aria-label': 'Loading' }), h('span', { class: 'muted' }, 'Loading…'), h('div', { class: 'skeleton', style: 'width:160px;height:14px' })),
        h('div', { class: 'row' }, h('button', { class: 'btn', onclick: () => toast('Saved', 'ok') }, 'OK toast'), h('button', { class: 'btn', onclick: () => toast('Something failed', 'bad') }, 'Error toast'))))));
}

const PAGES = { chat: (main) => chatPage(main), memory: (main) => memoryPage(main), mood: (main) => moodPage(main), prompt: (main) => personasPage(main), data: (main) => dataPage(main), settings: (main) => settingsPage(main), design: (main) => designPage(main) };

// ---------- router + nav
function route() {
  const id = (location.hash.replace(/^#\/?/, '') || (SECTIONS.some((s) => s.id === 'chat') ? 'chat' : 'settings')).split('/')[0];
  const sec = SECTIONS.find((s) => s.id === id);
  const main = document.getElementById('main');
  const title = id === 'design' ? 'Design kit' : sec ? sec.label : 'Not found';
  document.getElementById('page-title').textContent = title;
  document.title = title + ' · PersonaLM';
  const activeId = id === 'design' ? 'settings' : id;
  document.querySelectorAll('#nav a').forEach((a) => { if (a.dataset.id === activeId) a.setAttribute('aria-current', 'page'); else a.removeAttribute('aria-current'); });
  main.style.animation = 'none'; void main.offsetWidth; main.style.animation = '';
  if (PAGES[id]) PAGES[id](main);
  else if (sec) main.replaceChildren(plannedPage(sec));
  else main.replaceChildren(h('div', { class: 'empty' }, h('h2', null, 'Page not found'), h('p', null, h('a', { href: '#/settings' }, 'Go to Settings'))));
  closeMenu();
  document.getElementById('page-title').focus({ preventScroll: true });
}

function buildNav() {
  const nav = document.getElementById('nav');
  nav.replaceChildren();
  SECTIONS.forEach((s, i) => {
    if (s.id === 'settings') nav.append(h('hr'));
    nav.append(h('a', { href: '#/' + s.id, 'data-id': s.id }, icon(s.icon), s.label, s.ready ? null : h('span', { class: 'tag' }, s.stage)));
  });
}

// ---------- menu (drawer on phones, collapsible on desktop)
const app = () => document.getElementById('app');
function openMenu() { app().classList.add('menu-open'); document.getElementById('scrim').hidden = false; document.getElementById('menu-btn').setAttribute('aria-expanded', 'true'); }
function closeMenu() { app().classList.remove('menu-open'); document.getElementById('scrim').hidden = true; document.getElementById('menu-btn').setAttribute('aria-expanded', 'false'); }
function toggleMenu() {
  if (window.matchMedia('(max-width: 860px)').matches) app().classList.contains('menu-open') ? closeMenu() : openMenu();
  else app().classList.toggle('collapsed');
}

// ---------- keyboard
let gPending = 0;
document.addEventListener('keydown', (e) => {
  const tag = (e.target.tagName || '').toLowerCase();
  if (e.metaKey || e.ctrlKey || e.altKey || tag === 'input' || tag === 'textarea' || tag === 'select' || e.target.isContentEditable) return;
  const dlg = document.getElementById('help-dialog');
  if (e.key === 'Escape') { closeMenu(); return; }
  if (dlg.open) return;
  if (gPending && Date.now() < gPending) {
    gPending = 0;
    const n = parseInt(e.key, 10);
    if (n >= 1 && n <= 9 && SECTIONS[n - 1]) { location.hash = '#/' + SECTIONS[n - 1].id; e.preventDefault(); }
    else if (e.key === 's') { location.hash = '#/settings'; e.preventDefault(); }
    return;
  }
  if (e.key === 't') cycleTheme();
  else if (e.key === 'g') gPending = Date.now() + 1200;
  else if (e.key === '[') toggleMenu();
  else if (e.key === '?') dlg.showModal();
});

// ---------- start
async function start() {
  const menuBtn = document.getElementById('menu-btn');
  menuBtn.replaceChildren(icon('menu'));
  menuBtn.addEventListener('click', toggleMenu);
  document.getElementById('scrim').addEventListener('click', closeMenu);
  document.getElementById('theme-btn').addEventListener('click', cycleTheme);
  const dlg = document.getElementById('help-dialog');
  document.getElementById('help-btn').addEventListener('click', () => dlg.showModal());
  document.getElementById('help-close').addEventListener('click', () => dlg.close());
  dlg.addEventListener('click', (e) => { if (e.target === dlg) dlg.close(); });
  applyTheme(getMode(), false);

  const dot = document.getElementById('conn-dot'), txt = document.getElementById('conn-text');
  try {
    const info = await api('/api/info');
    SECTIONS = info.sections;
    await api('/api/ping', {});
    dot.className = 'dot ok'; txt.textContent = 'Connected · v' + info.version;
  } catch (e) {
    dot.className = 'dot bad'; txt.textContent = 'Not connected';
    SECTIONS = [{ id: 'settings', label: 'Settings and privacy', stage: 'U0', icon: 'lock', ready: true }];
    toast(e.message, 'bad');
  }
  buildNav();
  window.addEventListener('hashchange', route);
  route();
}
start();
