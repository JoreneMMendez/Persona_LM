// PersonaLM UI, stage U8: the "Language data" page. Pick English / Tagalog / Cebuano / Taglish, press Start: the server downloads the public text,
// runs the cleaning scripts and checks the files, so training can continue straight afterwards.
// The page sends only language ids, a mode and numbers; the server builds the command (see ui_corpus.py). All server text goes on the page with textContent only (through h()).
import { h, api, toast } from './app.js';

const LIVE = ['starting', 'running', 'stopping'];
const fmtMb = (n) => (typeof n === 'number' ? (n >= 1000 ? (n / 1000).toFixed(1) + ' GB' : Math.round(n) + ' MB') : '\u2014');
const fmtDate = (s) => (s ? new Date(s * 1000).toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric' }) : '');
const fmtTime = (sec) => {
  sec = Math.max(0, Math.round(sec || 0));
  const hh = Math.floor(sec / 3600), mm = Math.floor((sec % 3600) / 60), ss = sec % 60;
  return hh ? hh + ' h ' + String(mm).padStart(2, '0') + ' min' : mm ? mm + ' min ' + String(ss).padStart(2, '0') + ' s' : ss + ' s';
};
const errBox = (e) => h('div', { class: 'callout bad', role: 'alert' }, h('p', null, e.message || String(e)));

function copyBtn(text) {
  const b = h('button', { class: 'btn', type: 'button', 'aria-label': 'Copy the command' }, 'Copy');
  b.addEventListener('click', async () => {
    try { await navigator.clipboard.writeText(text); toast('Copied. Paste it in PowerShell.', 'ok'); }
    catch (e) { toast('Could not copy. Select the text and copy it by hand.', 'bad'); }
  });
  return b;
}

const STATE_BADGE = {
  ready: ['ok', 'cleaned and ready'],
  downloaded: ['warn', 'downloaded, not cleaned'],
  missing: ['', 'not downloaded'],
};

export async function corpusPage(main) {
  const root = h('div', { class: 'stack corpus' });
  main.replaceChildren(root);
  const S = { ov: null, sel: new Set(), mb: {}, mode: 'all', mix: false, confirm: false, ack: false, force: false, timer: null, wasLive: false, job: null };
  const pickBox = h('section', { class: 'card', 'aria-labelledby': 'h-pick' });
  const jobBox = h('section', { class: 'card', 'aria-labelledby': 'h-job' });
  const nextBox = h('section', { class: 'card', 'aria-labelledby': 'h-next' });
  root.append(
    h('p', { class: 'lead' }, 'Download public text in English, Tagalog, Cebuano and Taglish, clean it, and check it, in one go. Use this when training stops improving (validation loss rises) or you simply want more text. Only public text is downloaded; nothing of yours is read or uploaded.'),
    jobBox, pickBox, nextBox);

  // The page stops polling when you leave it (the route changes the DOM, so the root is no longer attached).
  const alive = () => root.isConnected;

  async function load(first) {
    if (first) pickBox.replaceChildren(h('h2', { id: 'h-pick' }, 'Choose languages'), h('div', { class: 'skeleton', style: 'height:200px' }));
    try { S.ov = await api('/api/corpus/overview'); } catch (e) { pickBox.replaceChildren(h('h2', { id: 'h-pick' }, 'Choose languages'), errBox(e)); return; }
    S.job = S.ov.job;
    if (first) for (const l of S.ov.languages) { if (!(l.id in S.mb)) S.mb[l.id] = l.default_mb; S.sel.add(l.id); }
    renderAll();
    schedule();
  }

  async function poll() {
    if (!alive()) return;
    try {
      const d = await api('/api/corpus/job');
      const was = S.job && LIVE.includes(S.job.state);
      S.job = d.job; S.ov.busy = d.busy; S.ov.training_running = d.training_running;
      const now = !!(d.job && LIVE.includes(d.job.state));
      if (was && !now) {
        const st = d.job && d.job.state;
        toast(st === 'done' ? 'Language data finished. Files are cleaned and checked.' : st === 'stopped' ? 'Stopped. Your earlier files were not changed.' : 'The run ended: ' + (st || 'unknown') + '. See the log.', st === 'done' ? 'ok' : st === 'failed' ? 'bad' : '');
        await load(false);
        return;
      }
      renderJob(); schedule();
    } catch (e) { schedule(8000); }
  }

  function schedule(ms) {
    clearTimeout(S.timer);
    if (!alive()) return;
    const live = !!(S.job && LIVE.includes(S.job.state));
    if (live) S.timer = setTimeout(poll, ms || 2500);
  }

  function renderAll() { renderPick(); renderJob(); renderNext(); }

  // ---------------------------------------------------------------- choose + start
  function renderPick() {
    const ov = S.ov, busy = ov.busy;
    const rows = ov.languages.map((l) => {
      const cb = h('input', { type: 'checkbox', id: 'lang-' + l.id, checked: S.sel.has(l.id), disabled: busy });
      cb.addEventListener('change', () => { cb.checked ? S.sel.add(l.id) : S.sel.delete(l.id); S.confirm = false; renderPick(); });
      const [cls, words] = STATE_BADGE[l.state] || ['', l.state];
      const files = [];
      if (l.raw && l.raw.exists) files.push('raw ' + fmtMb(l.raw.mb) + ' (' + fmtDate(l.raw.modified) + ')');
      if (l.clean.exists) files.push('clean ' + fmtMb(l.clean.mb) + ' (' + fmtDate(l.clean.modified) + ')');
      if (l.mix.exists) files.push('mix ' + fmtMb(l.mix.mb));
      const kids = [h('div', { class: 'row' }, h('label', { for: 'lang-' + l.id, class: 'name' }, l.label), h('span', { class: 'badge ' + cls }, words),
        l.partial ? h('span', { class: 'badge info' }, 'partial download kept') : null,
        h('span', { class: 'badge ' + (l.needs_ack ? 'warn' : '') }, 'licence: ' + l.license)),
      h('div', { class: 'small muted' }, l.source + '. ' + l.note),
      files.length ? h('div', { class: 'small' }, files.join('  \u00b7  ')) : null];
      if (S.sel.has(l.id) && l.capped && S.mode === 'all') {
        const inp = h('input', { type: 'number', id: 'mb-' + l.id, min: l.id === 'en' ? 10 : 0, max: 5000, step: 10, value: String(S.mb[l.id]), disabled: busy, 'aria-describedby': 'mbh-' + l.id });
        inp.addEventListener('input', () => { const n = parseInt(inp.value, 10); if (!Number.isNaN(n)) S.mb[l.id] = n; });
        kids.push(h('div', { class: 'field', style: 'max-width:240px;margin-top:8px' }, h('label', { for: 'mb-' + l.id }, 'Download up to (MB)'),
          inp, h('span', { class: 'hint', id: 'mbh-' + l.id }, l.id === 'en' ? 'FineWeb-Edu is big; this is how much you take.' : '0 means all of it.')));
      } else if (S.sel.has(l.id) && !l.capped && S.mode === 'all') {
        kids.push(h('p', { class: 'small muted' }, 'Downloads all of it (no size choice).'));
      }
      return h('li', null, cb, h('div', { class: 'grow' }, kids));
    });

    const modeSel = h('select', { id: 'mode', disabled: busy, 'aria-label': 'What to do' },
      h('option', { value: 'all', selected: S.mode === 'all' }, 'Download, then clean, then check'),
      h('option', { value: 'clean', selected: S.mode === 'clean' }, 'Only clean and check what is already downloaded'));
    modeSel.addEventListener('change', () => { S.mode = modeSel.value; S.confirm = false; renderPick(); });
    const mixCb = h('input', { type: 'checkbox', id: 'mix', checked: S.mix, disabled: busy });
    mixCb.addEventListener('change', () => { S.mix = mixCb.checked; S.confirm = false; renderPick(); });

    const kids = [h('div', { class: 'card-head' }, h('h2', { id: 'h-pick' }, 'Choose languages'),
      ov.free_mb != null ? h('span', { class: 'muted small' }, fmtMb(ov.free_mb) + ' free on this drive') : null),
    h('ul', { class: 'list' }, rows),
    h('div', { class: 'row', style: 'margin-top:12px' },
      h('div', { class: 'field grow' }, h('label', { for: 'mode' }, 'What to do'), modeSel),
      h('div', { class: 'row' }, mixCb, h('label', { for: 'mix' }, 'Also rebuild the training mix (public_mix/)')))];

    if (S.mode === 'all' && !ov.datasets_installed) {
      kids.push(h('div', { class: 'callout warn', role: 'alert' }, h('div', null,
        h('p', null, 'The downloader needs the `datasets` package, which is not installed. Run this once in PowerShell, then reload this page:'),
        h('div', { class: 'row' }, h('code', null, ov.install_command), copyBtn(ov.install_command)))));
    }
    if (ov.training_running) kids.push(h('div', { class: 'callout warn', role: 'status' }, h('p', null, 'A training job is running. Downloading and cleaning compete with it for the disk and CPU, and training may slow down. It is safer to wait until training has stopped.')));

    const picked = ov.languages.filter((l) => S.sel.has(l.id));
    const canStart = !busy && picked.length > 0 && (S.mode === 'clean' || ov.datasets_installed);
    const startBtn = h('button', { class: 'btn primary', type: 'button', disabled: !canStart }, 'Start');
    startBtn.addEventListener('click', () => {
      if (S.sel.has('en') && S.mode === 'all' && !(S.mb.en >= 10)) { toast('English needs at least 10 MB.', 'bad'); return; }
      S.confirm = true; S.ack = false; renderPick();
    });
    kids.push(h('div', { class: 'row', style: 'margin-top:12px' }, startBtn,
      busy ? h('span', { class: 'small muted' }, 'A job is running. Use the box above.') : picked.length ? null : h('span', { class: 'small muted' }, 'Tick at least one language.')));

    if (S.confirm && canStart) {
      const langs = picked.map((l) => l.id);
      const mb = {};
      if (S.mode === 'all') for (const l of picked) if (l.capped) mb[l.id] = S.mb[l.id];
      const cmd = ['python -m persona_lm corpus --lang ' + langs.join(',') + ' --mode ' + S.mode]
        .concat(Object.keys(mb).length ? ['--mb ' + Object.entries(mb).map(([k, v]) => k + '=' + v).join(' ')] : []).concat(S.mix ? ['--mix'] : []).join(' ');
      const needAck = picked.some((l) => l.needs_ack);
      const ack = needAck ? h('div', { class: 'row' }, h('input', { type: 'checkbox', id: 'ack', checked: S.ack }), h('label', { for: 'ack' }, 'I understand: Taglish posts come from real forum users and the licence is unknown. I will use them for my own private training only.')) : null;
      const go = h('button', { class: 'btn primary', type: 'button', disabled: needAck && !S.ack }, 'Yes, start now');
      if (ack) ack.querySelector('input').addEventListener('change', (e) => { S.ack = e.target.checked; go.disabled = !S.ack; });
      go.addEventListener('click', () => start(langs, mb));
      const no = h('button', { class: 'btn', type: 'button' }, 'Cancel');
      no.addEventListener('click', () => { S.confirm = false; renderPick(); });
      kids.push(h('div', { class: 'callout', role: 'group', 'aria-label': 'Confirm start' }, h('div', null,
        h('p', null, S.mode === 'all'
          ? 'This downloads public text from the internet (nothing of yours is sent), cleans it with the project\u2019s cleaning scripts and checks the files. It can take a long time and a lot of disk space. You can stop it at any time; your existing files stay as they are until a step finishes.'
          : 'This cleans and checks the text that is already in public_text/. No download.'),
        h('div', { class: 'dcmd' }, h('code', null, cmd), copyBtn(cmd)), ack, h('div', { class: 'row' }, go, no))));
    }
    pickBox.replaceChildren(...kids);
  }

  async function start(langs, mb) {
    try {
      const body = { languages: langs, mode: S.mode, mix: S.mix, mb, confirm: true };
      if (langs.includes('taglish')) body.ack_taglish = S.ack === true;
      const r = await api('/api/corpus/start', body);
      S.confirm = false; S.job = r.job; S.ov.busy = true;
      toast('Started. It keeps running if you close this page.', 'ok');
      renderAll(); schedule(800);
    } catch (e) { toast(e.message, 'bad'); }
  }

  async function stop(force) {
    try {
      const r = await api('/api/corpus/stop', force ? { force: true } : {});
      S.job = r.job; S.force = false;
      toast(force ? 'Ended now.' : 'Stopping after the current document.', force ? '' : 'ok');
      renderJob(); schedule(800);
      if (force) await load(false);
    } catch (e) { toast(e.message, 'bad'); }
  }

  // ---------------------------------------------------------------- the running / last job
  function renderJob() {
    const job = S.job;
    const head = h('div', { class: 'card-head' }, h('h2', { id: 'h-job' }, 'Progress'));
    if (!job) { jobBox.replaceChildren(head, h('p', { class: 'muted' }, 'Nothing has been run yet. Choose languages below and press Start.')); return; }
    const live = LIVE.includes(job.state);
    const cls = { running: 'accent', starting: 'accent', stopping: 'warn', done: 'ok', stopped: 'info', failed: 'bad', lost: 'warn' }[job.state] || '';
    const words = { running: 'running', starting: 'starting', stopping: 'stopping after this document', done: 'finished', stopped: 'stopped', failed: 'failed', lost: 'no sign of life' }[job.state] || job.state;
    const kids = [head, h('div', { class: 'row' },
      h('strong', null, (job.langs || []).join(', ') + (job.mode === 'clean' ? ' (clean only)' : '')), h('span', { class: 'badge ' + cls }, words),
      h('span', { class: 'small muted' }, (live ? 'running for ' : 'took ') + fmtTime(job.elapsed)),
      job.exit_code != null && job.exit_code !== 0 && job.state === 'failed' ? h('span', { class: 'badge bad' }, 'exit code ' + job.exit_code) : null)];
    const st = job.step || {};
    if (st.total) {
      const idx = job.state === 'done' ? st.total : Math.max(0, (st.index || 1) - (live ? 1 : 0));
      kids.push(h('div', { class: 'row' }, h('progress', { max: st.total, value: idx, 'aria-label': 'Steps finished', style: 'flex:1;min-width:160px' }),
        h('span', { class: 'small' }, 'step ' + Math.min(st.index || 0, st.total) + ' of ' + st.total + (st.title && live ? ': ' + st.title : ''))));
    }
    if (job.state === 'lost') kids.push(h('div', { class: 'callout warn', role: 'status' }, h('p', null, 'The job stopped sending its heartbeat. If the PC went to sleep it may continue; if it was restarted, the run is over. Files from finished steps are safe; start again to redo the rest.')));
    if (job.state === 'failed') kids.push(h('div', { class: 'callout bad', role: 'alert' }, h('p', null, 'The run failed. Read the last log lines below. Your earlier files were not replaced. If the network dropped, press Start again.')));
    kids.push(h('pre', { class: 'pp-text', tabindex: '0', 'aria-label': 'Newest log lines', 'aria-live': 'off' }, job.log && job.log.length ? job.log.join('\n') : 'Waiting for the first lines...'));
    if (live) {
      const stopBtn = h('button', { class: 'btn primary', type: 'button', disabled: job.state === 'stopping' }, job.state === 'stopping' ? 'Stopping...' : 'Stop');
      stopBtn.addEventListener('click', () => stop(false));
      const row = [stopBtn];
      if (!S.force) { const f = h('button', { class: 'btn ghost', type: 'button' }, 'Force stop...'); f.addEventListener('click', () => { S.force = true; renderJob(); }); row.push(f); }
      kids.push(h('div', { class: 'row' }, ...row));
      if (S.force) {
        const yes = h('button', { class: 'btn danger', type: 'button' }, 'End it now');
        yes.addEventListener('click', () => stop(true));
        const no = h('button', { class: 'btn', type: 'button' }, 'Keep running');
        no.addEventListener('click', () => { S.force = false; renderJob(); });
        kids.push(h('div', { class: 'callout warn', role: 'group' }, h('div', null, h('p', null, 'Force stop ends the process at once. The step in progress is lost (its half-written file is not used); finished steps are kept.'), h('div', { class: 'row' }, yes, no))));
      }
    }
    jobBox.replaceChildren(...kids);
    const lg = jobBox.querySelector('pre[aria-label="Newest log lines"]');
    if (lg) lg.scrollTop = lg.scrollHeight;
  }

  // ---------------------------------------------------------------- what to do after
  function renderNext() {
    const ov = S.ov;
    const kids = [h('div', { class: 'card-head' }, h('h2', { id: 'h-next' }, 'After it finishes: go to training'))];
    const ready = ov.languages.filter((l) => l.state === 'ready').map((l) => l.label);
    kids.push(h('p', { class: 'small muted' }, ready.length ? 'Cleaned and ready: ' + ready.join(', ') + '.' : 'Nothing is cleaned yet.'));
    if (!ov.projects.length) {
      kids.push(h('p', { class: 'small' }, 'No work folder has a project plan yet (see the Training page). Once one has, this box shows which public text it reads.'));
    }
    for (const p of ov.projects) {
      const rows = [h('div', { class: 'row' }, h('strong', null, p.workdir + '/'), p.tokens_done ? h('span', { class: 'badge info' }, 'public text already turned into tokens') : h('span', { class: 'badge' }, 'public text not tokenized yet'))];
      if (!p.has_pretrain_text) rows.push(h('p', { class: 'small muted' }, 'This project has no public pretraining text set.'));
      else {
        rows.push(h('p', { class: 'small' }, 'Reads: ', p.reads.map((r, i) => [i ? ', ' : '', h('code', null, r.path)])));
        const raw = p.reads.some((r) => r.kind === 'public_text');
        if (raw) rows.push(h('div', { class: 'callout warn', role: 'status' }, h('p', null, 'This project points at public_text/, the RAW download, not the cleaned files in public_clean/ or public_mix/. Training on it would skip the cleaning.')));
        else if (!p.reads.some((r) => r.kind === 'public_mix' || r.kind === 'public_clean')) rows.push(h('p', { class: 'small muted' }, 'It does not read public_clean/ or public_mix/, so new files here will not be used by it.'));
      }
      if (p.tokens_done) rows.push(h('p', { class: 'small' }, 'Its token files (data_pre) were made before. New text is only used after you tokenize again, into a new folder, with the same frozen tokenizer:'),
        h('div', { class: 'dcmd' }, h('code', null, ov.prepare_command), copyBtn(ov.prepare_command)),
        h('p', { class: 'small muted' }, 'Then point training at ' + ov.prepare_out + ' (--data-dir). Nothing here changes the tokenizer or your chat data.'));
      else rows.push(h('p', { class: 'small' }, 'Open the Training page: its step \u201cPrepare the public text\u201d turns the cleaned files into tokens.'));
      kids.push(...rows);
    }
    kids.push(h('div', { class: 'row' }, h('a', { class: 'btn', href: '#/training' }, 'Open the Training page')));
    nextBox.replaceChildren(...kids);
  }

  await load(true);
}
