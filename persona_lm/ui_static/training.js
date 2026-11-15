// PersonaLM UI, stages U5a + U5b: the Training page. U5a: the pipeline plan (done / NEXT), the checkpoints, a run's numbers and log.
// U5b: start / stop (resume) of the NEXT pipeline step as a background process, the live loss chart, the rising-validation warning.
// The page sends only a step name; the server builds the command from the project plan. All server text goes on the page with textContent only (through h()).
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

const fmtTime = (sec) => {
  sec = Math.max(0, Math.round(sec || 0));
  const hh = Math.floor(sec / 3600), mm = Math.floor((sec % 3600) / 60), ss = sec % 60;
  return hh ? hh + ' h ' + String(mm).padStart(2, '0') + ' min' : mm ? mm + ' min ' + String(ss).padStart(2, '0') + ' s' : ss + ' s';
};
const LIVE = ['starting', 'running', 'stopping'];
const SVGNS = 'http://www.w3.org/2000/svg';
function sv(tag, attrs, text) {
  const el = document.createElementNS(SVGNS, tag);
  for (const [k, v] of Object.entries(attrs || {})) el.setAttribute(k, String(v));
  if (text != null) el.textContent = text;
  return el;
}

// Loss chart: thin line = training loss, line with dots = validation loss, ring = best validation step. Colours come from app.css (light and dark).
export function lossChart(hist) {
  const W = 640, H = 260, L = 46, R = 14, T = 12, B = 30;
  const tr = (hist.train || []).filter((p) => typeof p.loss === 'number'), va = hist.val || [];
  const wrap = h('div', { class: 'losschart' });
  if (!tr.length && !va.length) { wrap.append(h('p', { class: 'muted' }, 'No loss numbers yet. They appear after the first logged steps.')); return wrap; }
  const xs = tr.map((p) => p.step).concat(va.map((p) => p.step));
  const x0 = Math.min(...xs), x1 = Math.max(...xs, x0 + 1);
  const trl = tr.map((p) => p.loss).sort((a, b) => a - b);
  const p95 = trl.length ? trl[Math.min(trl.length - 1, Math.floor(trl.length * 0.95))] : 0;
  let lo = Math.min(...tr.map((p) => p.loss), ...va.map((p) => p.val_loss));
  let hi = Math.max(p95, ...va.map((p) => p.val_loss));
  if (!(hi > lo)) hi = lo + 1;
  const pad = (hi - lo) * 0.06; lo = Math.max(0, lo - pad); hi += pad;
  const X = (s) => L + ((s - x0) / (x1 - x0)) * (W - L - R);
  const Y = (v) => T + (1 - (Math.min(v, hi) - lo) / (hi - lo)) * (H - T - B);
  const svg = sv('svg', { viewBox: '0 0 ' + W + ' ' + H, class: 'lc-svg', role: 'img',
    'aria-label': 'Loss chart. ' + (tr.length ? 'Training loss ends at ' + f3(tr[tr.length - 1].loss) + '. ' : '') + (va.length ? 'Validation loss: best ' + f3(hist.best && hist.best.val_loss) + ' at step ' + fmt(hist.best && hist.best.step) + ', latest ' + f3(va[va.length - 1].val_loss) + '.' : 'No validation numbers yet.') });
  for (let i = 0; i <= 4; i++) {
    const v = lo + ((hi - lo) * i) / 4, y = Y(v);
    svg.append(sv('line', { x1: L, x2: W - R, y1: y, y2: y, class: 'lc-grid' }), sv('text', { x: L - 6, y: y + 4, class: 'lc-tick', 'text-anchor': 'end' }, v.toFixed(2)));
  }
  for (let i = 0; i <= 4; i++) {
    const s = x0 + ((x1 - x0) * i) / 4;
    svg.append(sv('text', { x: X(s), y: H - 10, class: 'lc-tick', 'text-anchor': i === 0 ? 'start' : i === 4 ? 'end' : 'middle' }, fmt(Math.round(s))));
  }
  if (tr.length) svg.append(sv('polyline', { points: tr.map((p) => X(p.step).toFixed(1) + ',' + Y(p.loss).toFixed(1)).join(' '), class: 'lc-train' }));
  if (va.length) {
    svg.append(sv('polyline', { points: va.map((p) => X(p.step).toFixed(1) + ',' + Y(p.val_loss).toFixed(1)).join(' '), class: 'lc-val' }));
    va.forEach((p) => svg.append(sv('circle', { cx: X(p.step), cy: Y(p.val_loss), r: 3, class: 'lc-dot' })));
    if (hist.best) svg.append(sv('circle', { cx: X(hist.best.step), cy: Y(hist.best.val_loss), r: 7, class: 'lc-best' }));
  }
  wrap.append(svg, h('div', { class: 'row small lc-legend' },
    h('span', null, h('i', { class: 'lc-key train', 'aria-hidden': 'true' }), ' training loss (thin line)'),
    h('span', null, h('i', { class: 'lc-key val', 'aria-hidden': 'true' }), ' validation loss (dots)'),
    h('span', null, h('i', { class: 'lc-key best', 'aria-hidden': 'true' }), ' best validation step (ring)')));
  return wrap;
}

const errBox = (e) => h('div', { class: 'callout bad', role: 'alert' }, h('p', null, e.message || String(e)));

export async function trainingPage(main) {
  const root = h('div', { class: 'stack training' });
  main.replaceChildren(root);
  const state = { workdir: 'work', run: null, file: '' };
  const planBox = h('div', { class: 'card' });
  const ckptBox = h('div', { class: 'card' });
  const runBox = h('div', { class: 'card', id: 'run-view' });
  const jobBox = h('div', { class: 'card', id: 'job-box' });
  root.append(h('div', { class: 'card' }, h('h2', null, 'Training'),
    h('p', { class: 'muted' }, 'See what is done, what to do next and how each run went. The NEXT step of the plan can be started and stopped from here; it runs in the background and keeps going if you close this page. The first step (the tokenizer) and any step that needs a file outside the project stay in PowerShell.')),
    planBox, jobBox, ckptBox, runBox);

  // ---- 1. pipeline plan
  async function loadPlan() {
    planBox.replaceChildren(h('h2', null, 'Pipeline plan'), h('div', { class: 'skeleton', style: 'height:140px' }));
    let p;
    try { p = await api('/api/train/plan?workdir=' + encodeURIComponent(state.workdir)); }
    catch (e) { planBox.replaceChildren(h('h2', null, 'Pipeline plan'), errBox(e)); return; }
    const sel = h('select', { 'aria-label': 'Work folder' }, p.workdirs.map((w) => h('option', { value: w, selected: w === p.workdir }, w)));
    sel.addEventListener('change', () => { state.workdir = sel.value; J.confirm = null; loadPlan(); loadJob(); });
    const refresh = h('button', { class: 'btn', type: 'button' }, 'Refresh');
    refresh.addEventListener('click', () => { loadPlan(); loadCkpts(); loadJob(); });
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
    else runBox.append(lossChart(hist), ...(hist.rising ? [riseWarning(hist)] : []));
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

  // ---- 4. run a step (U5b): controls, live loss chart, warning
  const J = { data: null, hist: null, confirm: null, force: false, timer: null, wasLive: false };
  async function loadJob(quiet) {
    clearTimeout(J.timer);
    if (!root.isConnected) return;
    if (!quiet && !J.data) jobBox.replaceChildren(h('h2', null, 'Run a step'), h('div', { class: 'skeleton', style: 'height:90px' }));
    try {
      J.data = await api('/api/train/job?workdir=' + encodeURIComponent(state.workdir));
      const job = J.data.job;
      J.hist = job && job.run ? await api('/api/train/history?run=' + encodeURIComponent(job.run)).catch(() => null) : null;
    } catch (e) { jobBox.replaceChildren(h('h2', null, 'Run a step'), errBox(e)); return; }
    const live = !!(J.data.job && LIVE.includes(J.data.job.state));
    if (J.wasLive && !live) {
      const st = J.data.job && J.data.job.state;
      toast(st === 'done' ? 'The step finished.' : st === 'stopped' ? 'Stopped. The checkpoint is saved.' : 'The step ended: ' + st, st === 'failed' || st === 'lost' ? 'bad' : 'ok');
      loadPlan(); loadCkpts();
    }
    J.wasLive = live;
    renderJob();
    if (live) J.timer = setTimeout(() => loadJob(true), 3000);
  }

  function riseWarning(hist) {
    return h('div', { class: 'callout warn', role: 'status' }, h('p', null,
      'Validation loss is rising: best ' + f3(hist.best.val_loss) + ' at step ' + fmt(hist.best.step) + ', now ' + f3(hist.last_val.val_loss) + ' (' + hist.evals_since_best + ' check' + (hist.evals_since_best === 1 ? '' : 's') + ' since the best). '
      + 'The model is starting to learn your chats by heart. Press "Stop and save" soon; only the newest two checkpoints are kept, so the best step is lost if training runs on.'));
  }

  async function act(path, body, okMsg) {
    try { await api(path, body); if (okMsg) toast(okMsg, 'ok'); }
    catch (e) { toast(e.message || String(e), 'bad'); }
    J.confirm = null; J.force = false;
    await loadJob();
  }

  function renderJob() {
    const d = J.data, job = d.job, live = !!(job && LIVE.includes(job.state));
    const kids = [h('div', { class: 'card-head' }, h('h2', null, 'Run a step'),
      h('span', { class: 'badge ' + (live ? 'accent' : 'muted') }, live ? 'a step is running' : 'nothing running'))];
    if (job) {
      const cls = { running: 'accent', starting: 'accent', stopping: 'warn', done: 'ok', stopped: 'info', failed: 'bad', lost: 'warn' }[job.state] || '';
      const words = { running: 'running', starting: 'starting', stopping: 'stopping (saving a checkpoint)', done: 'finished', stopped: 'stopped, checkpoint saved', failed: 'failed', lost: 'no sign of life' }[job.state] || job.state;
      kids.push(h('div', { class: 'row' }, h('strong', null, job.title || job.step), h('span', { class: 'badge ' + cls }, words),
        h('span', { class: 'small muted' }, (live ? 'running for ' : 'took ') + fmtTime(job.elapsed)), job.exit_code != null && job.exit_code !== 0 ? h('span', { class: 'badge bad' }, 'exit code ' + job.exit_code) : null));
      if (job.state === 'lost') kids.push(h('div', { class: 'callout warn', role: 'status' }, h('p', null, 'The step stopped sending its heartbeat. If the PC went to sleep it may continue; if it was restarted, the step is over. Nothing is lost beyond the last checkpoint: starting it again resumes from there.')));
      const hist = J.hist;
      if (job.planned_steps && hist && hist.last_step != null) {
        kids.push(h('div', { class: 'row' }, h('progress', { max: job.planned_steps, value: Math.min(hist.last_step, job.planned_steps), 'aria-label': 'Training progress', style: 'flex:1;min-width:160px' }),
          h('span', { class: 'small' }, 'step ' + fmt(hist.last_step) + ' of ' + fmt(job.planned_steps))));
      }
      if (hist && hist.has_metrics) kids.push(lossChart(hist), ...(hist.rising ? [riseWarning(hist)] : []));
      kids.push(h('pre', { class: 'pp-text', tabindex: '0', 'aria-label': 'Newest log lines', 'aria-live': 'off' }, job.log && job.log.length ? job.log.join('\n') : 'Waiting for the first lines...'));
    }
    if (live) {
      const stopBtn = h('button', { class: 'btn primary', type: 'button', disabled: job.state === 'stopping' }, job.state === 'stopping' ? 'Saving...' : 'Stop and save');
      stopBtn.addEventListener('click', () => act('/api/train/stop', {}, 'Stop asked. The trainer saves a checkpoint, then ends.'));
      const row = h('div', { class: 'row' }, stopBtn);
      if (!J.force) {
        const f = h('button', { class: 'btn danger', type: 'button' }, 'Force stop...');
        f.addEventListener('click', () => { J.force = true; renderJob(); });
        row.append(f);
      } else {
        const yes = h('button', { class: 'btn danger', type: 'button' }, 'Yes, end it now');
        yes.addEventListener('click', () => act('/api/train/stop', { force: true }, 'Ended. Progress since the last checkpoint is lost.'));
        const no = h('button', { class: 'btn', type: 'button' }, 'Cancel');
        no.addEventListener('click', () => { J.force = false; renderJob(); });
        row.append(h('span', { class: 'small' }, 'Ends the process without saving: you lose everything since the last checkpoint.'), yes, no);
      }
      kids.push(row, h('p', { class: 'small muted' }, d.note));
    } else {
      const p = d.plan;
      kids.push(h('h3', null, 'Start the next step (' + p.workdir + ')'));
      if (!p.exists || !p.steps.length) kids.push(h('p', { class: 'muted' }, p.why || 'No project plan in ' + p.workdir + ' yet (see the Pipeline plan above).'));
      else {
        for (const s of p.steps) {
          const resume = s.resume_step != null && s.can_start;
          const btn = h('button', { class: 'btn primary', type: 'button', disabled: !s.can_start, 'aria-label': (resume ? 'Resume ' : 'Start ') + s.title }, resume ? 'Resume from step ' + fmt(s.resume_step) : 'Start');
          btn.addEventListener('click', () => { J.confirm = s.name; renderJob(); });
          const badge = s.state === 'done' ? h('span', { class: 'badge ok' }, 'done') : s.state === 'next' ? h('span', { class: 'badge accent' }, 'NEXT') : s.state === 'skipped' ? h('span', { class: 'badge warn' }, 'skipped') : null;
          kids.push(h('div', { class: 'jobrow' }, h('div', null, h('strong', null, s.title), ' ', badge,
            s.can_start ? null : h('p', { class: 'small muted' }, s.reason)), btn));
          if (J.confirm === s.name) {
            const go = h('button', { class: 'btn primary', type: 'button' }, 'Yes, ' + (resume ? 'resume' : 'start') + ' now');
            go.addEventListener('click', () => act('/api/train/start', { workdir: p.workdir, step: s.name, confirm: true }, 'Started. It keeps running if you close this page.'));
            const no = h('button', { class: 'btn', type: 'button' }, 'Cancel');
            no.addEventListener('click', () => { J.confirm = null; renderJob(); });
            kids.push(h('div', { class: 'callout', role: 'group', 'aria-label': 'Confirm start' }, h('div', null,
              h('p', null, resume ? 'This continues the run from its saved checkpoint (step ' + fmt(s.resume_step) + '). It uses the GPU and may take hours.' : 'This starts the step below. Training uses the GPU and can take hours; you can stop it at any time and continue later.'),
              h('div', { class: 'dcmd' }, h('code', null, s.command), copyBtn(s.command)), h('div', { class: 'row' }, go, no))));
          }
        }
        if (!p.confirmed) kids.push(h('p', { class: 'small muted' }, 'Decisions are not confirmed yet: run `pipeline confirm` in PowerShell first.'));
      }
      kids.push(h('p', { class: 'small muted' }, 'Stopping later saves a checkpoint, like Ctrl+C. Starting the same step again resumes from it.'));
    }
    jobBox.replaceChildren(...kids);
    const lg = jobBox.querySelector('pre[aria-label="Newest log lines"]');
    if (lg) lg.scrollTop = lg.scrollHeight;
  }

  loadPlan(); loadJob(); loadCkpts(); loadRun();
}
