// PersonaLM UI, stage U5a: the Training page, part 1. Read only: the pipeline plan (done / NEXT), the checkpoints, and a run's numbers and log.
// Starting and stopping training, and the live loss chart, come in U5b. All server text goes on the page with textContent only (through h()).
import { h, api, toast } from './app.js';

const fmt = (n) => (typeof n === 'number' ? n.toLocaleString('en-US') : '\u2014');
const f3 = (n) => (typeof n === 'number' ? n.toFixed(3) : '\u2014');

function copyBtn(text) {
  const b = h('button', { class: 'btn', type: 'button', 'aria-label': 'Copy the command' }, 'Copy');
  b.addEventListener('click', async () => {
    try { await navigator.clipboard.writeText(text); toast('Copied. Paste it in PowerShell.', 'ok'); }
    catch (e) { toast('Could not copy. Select the text and copy it by hand.', 'bad'); }
  });
  return b;
}

const errBox = (e) => h('div', { class: 'callout bad', role: 'alert' }, h('p', null, e.message || String(e)));

export async function trainingPage(main) {
  const root = h('div', { class: 'stack training' });
  main.replaceChildren(root);
  const state = { workdir: 'work', run: null, file: '' };
  const planBox = h('div', { class: 'card' });
  const ckptBox = h('div', { class: 'card' });
  const runBox = h('div', { class: 'card', id: 'run-view' });
  root.append(h('div', { class: 'card' }, h('h2', null, 'Training'),
    h('p', { class: 'muted' }, 'See what is done, what to do next, and how each run went. This page only reads. Training itself still runs in PowerShell with the commands shown here (starting it from this page comes in the next stage).')),
    planBox, ckptBox, runBox);

  // ---- 1. pipeline plan
  async function loadPlan() {
    planBox.replaceChildren(h('h2', null, 'Pipeline plan'), h('div', { class: 'skeleton', style: 'height:140px' }));
    let p;
    try { p = await api('/api/train/plan?workdir=' + encodeURIComponent(state.workdir)); }
    catch (e) { planBox.replaceChildren(h('h2', null, 'Pipeline plan'), errBox(e)); return; }
    const sel = h('select', { 'aria-label': 'Work folder' }, p.workdirs.map((w) => h('option', { value: w, selected: w === p.workdir }, w)));
    sel.addEventListener('change', () => { state.workdir = sel.value; loadPlan(); });
    const refresh = h('button', { class: 'btn', type: 'button' }, 'Refresh');
    refresh.addEventListener('click', () => { loadPlan(); loadCkpts(); });
    planBox.replaceChildren(h('div', { class: 'card-head' }, h('h2', null, 'Pipeline plan'), h('div', { class: 'row' }, sel, refresh)));
    if (p.error) planBox.append(h('div', { class: 'callout warn', role: 'alert' }, h('p', null, p.error)));
    if (!p.exists) {
      planBox.append(h('p', { class: 'muted' }, 'No project plan in ' + p.workdir + ' yet. Create one once, then this page tracks it:'),
        h('div', { class: 'dcmd' }, h('code', null, p.init_command), copyBtn(p.init_command)));
      return;
    }
    if (!p.steps) return;
    const pr = p.project;
    planBox.append(h('div', { class: 'row' },
      h('span', { class: 'badge info' }, 'model ' + pr.size), h('span', { class: 'badge' }, 'window ' + fmt(pr.ctx)), h('span', { class: 'badge' }, 'vocab ' + fmt(pr.vocab)),
      h('span', { class: 'badge ' + (pr.decisions_confirmed ? 'ok' : 'warn') }, pr.decisions_confirmed ? 'decisions confirmed' : 'decisions not confirmed yet')));
    if (!pr.decisions_confirmed) planBox.append(h('p', { class: 'small muted' }, 'Read the decisions with `pipeline plan`, then run `pipeline confirm` in the terminal. This page cannot confirm them for you.'));
    p.steps.forEach((s, i) => {
      const badge = s.state === 'done' ? h('span', { class: 'badge ok' }, 'done') : s.state === 'next' ? h('span', { class: 'badge accent' }, 'NEXT')
        : s.state === 'skipped' ? h('span', { class: 'badge warn' }, 'skipped') : null;
      const body = h('div', null, h('h3', null, s.title, ' ', badge));
      if (s.note) body.append(h('p', { class: 'small muted' }, s.note));
      if (s.command) body.append(h('div', { class: 'dcmd' }, h('code', null, s.command), copyBtn(s.command)));
      planBox.append(h('div', { class: 'dstep ' + (s.state === 'skipped' ? 'todo' : s.state) }, h('span', { class: 'dmark', 'aria-hidden': 'true' }, s.state === 'done' ? '\u2713' : String(i + 1)), body));
    });
    planBox.append(h('div', { class: 'dstep ' + (p.all_done ? 'next' : 'todo') }, h('span', { class: 'dmark', 'aria-hidden': 'true' }, String(p.steps.length + 1)),
      h('div', null, h('h3', null, 'Talk to it ', p.all_done ? h('span', { class: 'badge accent' }, 'NEXT') : null),
        h('p', { class: 'small muted' }, 'Any time after fine-tuning. The Chat page does the same with a mouse.'),
        h('div', { class: 'dcmd' }, h('code', null, p.chat_command), copyBtn(p.chat_command)))));
  }

  // ---- 2. checkpoints
  async function loadCkpts() {
    ckptBox.replaceChildren(h('h2', null, 'Checkpoints'), h('div', { class: 'skeleton', style: 'height:100px' }));
    let c;
    try { c = await api('/api/train/checkpoints'); }
    catch (e) { ckptBox.replaceChildren(h('h2', null, 'Checkpoints'), errBox(e)); return; }
    ckptBox.replaceChildren(h('div', { class: 'card-head' }, h('h2', null, 'Checkpoints'),
      h('span', { class: 'badge ' + (c.tokenizer_present ? 'ok' : 'warn') }, c.tokenizer_present ? 'frozen tokenizer ' + c.tokenizer_sha : 'work/tok.json not found')));
    if (!c.runs.length) { ckptBox.append(h('p', { class: 'muted' }, 'No run folder with a checkpoint yet in work, work2 or work3. Training creates one.')); return; }
    const rows = c.runs.map((r) => {
      const m = r.model || {};
      const val = r.val;
      const when = r.modified ? new Date(r.modified * 1000).toLocaleString() : '\u2014';
      const open = h('button', { class: 'btn', type: 'button', 'aria-label': 'Show numbers and log of ' + r.run }, 'Details');
      open.addEventListener('click', () => { state.run = r.run; state.file = ''; loadRun(); runBox.scrollIntoView({ behavior: 'smooth', block: 'start' }); });
      return h('tr', null,
        h('th', { scope: 'row' }, h('span', { class: 'mono' }, r.run)),
        h('td', null, r.step == null ? '\u2014' : fmt(r.step) + (r.steps_planned ? ' / ' + fmt(r.steps_planned) : '')),
        h('td', null, m.n_layers ? m.n_layers + 'L ' + m.d_model + 'd, window ' + fmt(m.ctx) : '\u2014'),
        h('td', null, val ? h('span', { class: 'badge ' + (val.rising ? 'warn' : 'ok') }, 'best ' + f3(val.best) + ' @' + fmt(val.best_step) + (val.rising ? ', now ' + f3(val.last) + ' (rising)' : '')) : h('span', { class: 'muted' }, 'no val numbers')),
        h('td', null, r.tokenizer_ok === true ? h('span', { class: 'badge ok' }, 'matches') : r.tokenizer_ok === false ? h('span', { class: 'badge bad' }, 'DIFFERENT') : h('span', { class: 'muted' }, '\u2014')),
        h('td', { class: 'small' }, (r.size_mb != null ? r.size_mb + ' MB' : '\u2014'), h('br'), h('span', { class: 'muted' }, when), r.has_main ? null : h('span', { class: 'badge warn' }, 'only .prev')),
        h('td', null, open));
    });
    ckptBox.append(h('div', { class: 'tbl-wrap' }, h('table', { class: 'tbl' },
      h('thead', null, h('tr', null, ['Run', 'Step', 'Model', 'Validation loss', 'Tokenizer', 'File', ''].map((x) => h('th', { scope: 'col' }, x)))),
      h('tbody', null, rows))), h('p', { class: 'small muted' }, c.note));
    if (c.runs.some((r) => r.val && r.val.rising)) ckptBox.append(h('div', { class: 'callout warn', role: 'status' }, h('p', null, 'A run with rising validation loss went past its best step: it is learning your chats by heart. The best step is usually not kept (only the newest two checkpoints are). Retrain with fewer steps, about 3 passes over the data.')));
  }

  // ---- 3. one run: numbers and log
  async function loadRun() {
    if (!state.run) { runBox.replaceChildren(h('h2', null, 'Run numbers and log'), h('p', { class: 'muted' }, 'Press Details on a run above.')); return; }
    runBox.replaceChildren(h('h2', null, 'Run: ' + state.run), h('div', { class: 'skeleton', style: 'height:160px' }));
    let hist, log;
    try {
      hist = await api('/api/train/history?run=' + encodeURIComponent(state.run));
      log = await api('/api/train/log?run=' + encodeURIComponent(state.run) + (state.file ? '&file=' + encodeURIComponent(state.file) : ''));
    } catch (e) { runBox.replaceChildren(h('h2', null, 'Run: ' + state.run), errBox(e)); return; }
    const last = hist.train.length ? hist.train[hist.train.length - 1] : null;
    runBox.replaceChildren(h('h2', null, 'Run: ' + state.run),
      h('div', { class: 'dstats' },
        h('div', null, h('span', { class: 'small muted' }, 'last step'), h('span', { class: 'num' }, fmt(hist.last_step))),
        h('div', null, h('span', { class: 'small muted' }, 'train loss now'), h('span', { class: 'num' }, f3(last && last.loss))),
        h('div', null, h('span', { class: 'small muted' }, 'best val loss'), h('span', { class: 'num' }, hist.best ? f3(hist.best.val_loss) + ' @' + fmt(hist.best.step) : '\u2014')),
        h('div', null, h('span', { class: 'small muted' }, 'log lines'), h('span', { class: 'num' }, fmt(hist.rows)))));
    if (!hist.has_metrics) runBox.append(h('p', { class: 'muted' }, 'This run has no metrics.jsonl yet (it is written while training).'));
    if (hist.val.length) {
      runBox.append(h('h3', null, 'Validation loss'), h('div', { class: 'tbl-wrap' }, h('table', { class: 'tbl' },
        h('thead', null, h('tr', null, h('th', { scope: 'col' }, 'Step'), h('th', { scope: 'col' }, 'Val loss'), h('th', { scope: 'col' }, ''))),
        h('tbody', null, hist.val.map((v) => h('tr', null, h('td', null, fmt(v.step)), h('td', null, f3(v.val_loss)), h('td', null, v.best ? h('span', { class: 'badge ok' }, 'best') : null)))))));
    }
    const pick = h('select', { 'aria-label': 'Log source' }, h('option', { value: '' }, 'Numbers from metrics.jsonl'), log.files.map((f) => h('option', { value: f, selected: f === log.file }, f)));
    pick.addEventListener('change', () => { state.file = pick.value; loadRun(); });
    runBox.append(h('div', { class: 'card-head', style: 'margin-top:12px' }, h('h3', null, 'Log (newest lines)'), pick),
      h('pre', { class: 'pp-text', tabindex: '0', 'aria-label': 'Log text' }, log.lines.length ? log.lines.join('\n') : 'Nothing logged yet.'),
      ...(log.cut ? [h('p', { class: 'small muted' }, 'Only the newest lines are shown.')] : []));
  }

  loadPlan(); loadCkpts(); loadRun();
}
