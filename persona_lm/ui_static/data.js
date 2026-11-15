// PersonaLM UI, stage U4a: the Data page. Read only: it shows what exists, the privacy audit (COUNTS from stats.json) and the data-report verdict.
// The Messenger scan and import stay in the terminal (the raw export is outside the project and this UI never reads it); the wizard shows the exact commands.
// All text from the server goes on the page with textContent only (through h()). No message text or names ever come from these routes.
import { h, api, toast } from './app.js';

const fmt = (n) => (typeof n === 'number' ? n.toLocaleString('en-US') : '\u2014');

function copyBtn(text) {
  const b = h('button', { class: 'btn', type: 'button', 'aria-label': 'Copy the command' }, 'Copy');
  b.addEventListener('click', async () => {
    try { await navigator.clipboard.writeText(text); toast('Copied. Paste it in PowerShell.', 'ok'); }
    catch (e) { toast('Could not copy. Select the text and copy it by hand.', 'bad'); }
  });
  return b;
}

function stat(label, value) { return h('div', null, h('span', { class: 'small muted' }, label), h('span', { class: 'num' }, value)); }

export async function dataPage(main) {
  const root = h('div', { class: 'stack data' });
  main.replaceChildren(root);
  let size = null, folder = null;

  async function load() {
    root.replaceChildren(h('div', { class: 'skeleton', style: 'height:120px' }), h('div', { class: 'skeleton', style: 'height:240px' }));
    let ov;
    try { ov = await api('/api/data/overview'); }
    catch (e) { root.replaceChildren(h('div', { class: 'callout bad', role: 'alert' }, h('p', null, e.message))); return; }
    size = size || ov.default_size;
    const chat = ov.token_folders.filter((t) => t.exists && t.kind === 'chat');
    if (!folder || !ov.token_folders.some((t) => t.name === folder && t.exists)) folder = chat.length ? chat[chat.length - 1].name : (ov.token_folders.find((t) => t.exists) || {}).name || null;

    const head = h('div', { class: 'card' }, h('h2', null, 'Your chat data'),
      h('p', { class: 'muted' }, 'This page shows what each step produced and whether it is safe and big enough. It only reads counts. It never shows a message, a name or a number from your chats, and it never opens data_msgr/private/.'),
      h('div', { class: 'row' },
        h('span', { class: 'badge ' + (ov.export_remembered ? 'ok' : '') }, ov.export_remembered ? 'export folder remembered' : 'no export folder remembered yet'),
        ...ov.imports.map((i) => h('span', { class: 'badge ' + (i.exists ? 'ok' : '') }, i.name + (i.exists ? ' found' : ' not here'))),
        ...ov.token_folders.map((t) => h('span', { class: 'badge ' + (t.exists ? 'ok' : '') }, t.name + (t.exists ? ' (' + t.kind + ', ' + fmt(t.train_tokens) + ' tokens)' : ' not here')))));

    const wiz = h('div', { class: 'card' }, h('h2', null, 'Import wizard'),
      h('p', { class: 'muted small' }, 'Run each command in PowerShell from the project folder, then press Refresh. The step marked NEXT is the one to do now.'));
    ov.steps.forEach((s) => {
      const state = s.state;
      const body = h('div', null, h('h3', null, s.title, ' ', state === 'done' ? h('span', { class: 'badge ok' }, 'done') : state === 'next' ? h('span', { class: 'badge accent' }, 'NEXT') : null),
        h('p', { class: 'small muted' }, s.what));
      if (s.command) body.append(h('div', { class: 'dcmd' }, h('code', null, s.command), copyBtn(s.command)));
      if (state === 'done' && s.done_text) body.append(h('p', { class: 'small' }, s.done_text));
      wiz.append(h('div', { class: 'dstep ' + state }, h('span', { class: 'dmark', 'aria-hidden': 'true' }, state === 'done' ? '\u2713' : s.title.slice(0, 1)), body));
    });
    const refresh = h('button', { class: 'btn', type: 'button' }, 'Refresh');
    refresh.addEventListener('click', load);
    wiz.append(h('div', { class: 'row', style: 'margin-top:12px' }, refresh));

    const auditBox = h('div', { class: 'card', id: 'audit' }, h('h2', null, 'Privacy audit'), h('p', { class: 'muted', role: 'status' }, 'Loading\u2026'));
    const repBox = h('div', { class: 'card', id: 'report' }, h('h2', null, 'Data-report verdict'));
    root.replaceChildren(head, wiz, auditBox, repBox);
    paintAudit(auditBox);
    paintReport(repBox, ov);
  }

  async function paintAudit(box) {
    let a;
    try { a = await api('/api/data/audit'); } catch (e) { box.append(h('div', { class: 'callout bad', role: 'alert' }, h('p', null, e.message))); box.querySelector('[role=status]')?.remove(); return; }
    box.querySelector('[role=status]')?.remove();
    if (!a.exists) { box.append(h('div', { class: 'callout' }, h('p', null, 'No import yet (data_msgr/stats.json is missing). Do wizard steps 1 and 2 first.'))); return; }
    box.append(h('div', { class: 'callout ' + a.level, role: a.level === 'bad' ? 'alert' : null }, h('p', null, a.verdict)));
    box.append(h('div', { class: 'row', style: 'margin-top:8px' },
      h('span', { class: 'badge ' + (a.anonymized ? 'ok' : 'bad') }, a.anonymized ? 'anonymised' : 'NOT anonymised'),
      a.paranoid ? h('span', { class: 'badge info' }, 'paranoid mode') : null,
      h('span', { class: 'badge ' + (a.safe_to_upload ? 'ok' : 'warn') }, a.safe_to_upload ? 'passed the upload check' : 'keep on this computer'),
      h('span', { class: 'badge' }, a.learn_from_others ? 'learns from everyone\u2019s messages' : 'only your messages')));
    box.append(h('div', { class: 'dstats' }, stat('Chats', fmt(a.chats)), stat('Sessions', fmt(a.sessions)), stat('Your messages', fmt(a.my_messages)), stat('Period', a.period ? a.period.join(' to ') : '\u2014')));
    const table = (title, obj, empty) => {
      const keys = Object.keys(obj);
      return h('div', null, h('h3', null, title), keys.length
        ? h('div', { class: 'dkv' }, keys.map((k) => h('span', { class: 'badge' }, k + ': ' + fmt(obj[k]))))
        : h('p', { class: 'small muted' }, empty));
    };
    box.append(table('Possible leaks', a.leaks, 'None found.'), table('Suspicious (to look at)', a.suspicious, 'None.'),
      table('Masked by the anonymiser', a.masked, 'Nothing was masked.'), table('Message kinds dropped', a.dropped_message_kinds, 'None.'));
    if (a.size_verdict) box.append(h('p', { class: 'small muted' }, a.size_verdict));
    box.append(h('p', { class: 'small muted' }, a.note + ' To review the details, open data_msgr\\private\\audit.txt in a text editor yourself; the page will not show it.'));
  }

  function paintReport(box, ov) {
    const folders = ov.token_folders.filter((t) => t.exists);
    if (!folders.length) { box.append(h('div', { class: 'callout' }, h('p', null, 'No token folder yet. Do wizard step 4 (prepare) first.'))); return; }
    const fsel = h('select', { id: 'rf', 'aria-label': 'Token folder' }, folders.map((t) => h('option', { value: t.name }, t.name + ' (' + t.kind + ')')));
    fsel.value = folder;
    const ssel = h('select', { id: 'rs', 'aria-label': 'Model size' }, ov.sizes.map((x) => h('option', { value: x }, x)));
    ssel.value = size;
    const out = h('div', { 'aria-live': 'polite' });
    const go = async () => {
      folder = fsel.value; size = ssel.value;
      out.replaceChildren(h('p', { class: 'muted', role: 'status' }, 'Reading the token files\u2026'));
      try { paintVerdict(out, await api('/api/data/report?folder=' + encodeURIComponent(folder) + '&size=' + encodeURIComponent(size))); }
      catch (e) { out.replaceChildren(h('div', { class: 'callout bad', role: 'alert' }, h('p', null, e.message))); }
    };
    fsel.addEventListener('change', go); ssel.addEventListener('change', go);
    box.append(h('div', { class: 'row' }, h('div', { class: 'field' }, h('label', { for: 'rf' }, 'Token folder'), fsel), h('div', { class: 'field' }, h('label', { for: 'rs' }, 'Model size'), ssel)), out);
    go();
  }

  function paintVerdict(out, r) {
    const kids = [h('div', { class: 'callout ' + r.level }, h('p', null, r.verdict))];
    kids.push(h('div', { class: 'dstats' }, stat('Training tokens', fmt(r.train_tokens)), stat('Validation tokens', fmt(r.val_tokens)),
      stat(r.kind === 'chat' ? 'Examples' : 'Documents', fmt(r.examples)), r.loss_tokens != null ? stat('Tokens that count (loss)', fmt(r.loss_tokens)) : null,
      stat('Model', (r.params / 1e6).toFixed(1) + 'M, window ' + fmt(r.window)), stat('Tokens per parameter', String(r.tokens_per_param))));
    kids.push(h('div', { class: 'callout ' + (r.tokenizer.state === 'ok' ? 'ok' : r.tokenizer.state === 'bad' ? 'bad' : 'warn') }, h('p', null, 'Tokenizer: ' + r.tokenizer.text)));
    (r.warnings || []).forEach((w) => kids.push(h('div', { class: 'callout warn' }, h('p', null, w))));
    if (r.passes) kids.push(h('p', { class: 'small muted' }, 'About ' + fmt(r.passes.steps_for_3_passes) + ' steps (batch 1, accum 4) make 3 passes over this data. ' + r.passes.note));
    kids.push(h('p', { class: 'small muted' }, 'A rule of thumb, not a measurement. Same numbers as: python -m persona_lm data-report --data-dir ' + r.folder + ' --size ' + r.size));
    out.replaceChildren(...kids.filter(Boolean));
  }

  load();
}
