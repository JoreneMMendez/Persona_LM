// PersonaLM UI, stage U4a: the Data page. Read only: it shows what exists, the privacy audit (COUNTS from stats.json) and the data-report verdict.
// U4b adds the cleaner: preview (nothing written), the repeated-reply report, and one owner button that writes data_msgr_chats for the previewed plan.
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
    const cleanBox = h('div', { class: 'card', id: 'clean' });
    root.replaceChildren(head, wiz, auditBox, cleanBox, repBox);
    paintAudit(auditBox);
    paintClean(cleanBox, ov);
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

  function paintClean(box, ov) {
    box.append(h('h2', null, 'Clean the chats'),
      h('p', { class: 'muted small' }, 'Removes call and group notices, drops other people\u2019s lines as answers in group chats, and keeps each identical reply only a few times, so the model does not memorise stock lines. It writes one numbered file per chat into data_msgr_chats. Press Preview first: that changes nothing.'));
    const imp = ov.imports.find((i) => i.name === 'data_msgr');
    if (!imp || !imp.train_jsonl) { box.append(h('div', { class: 'callout' }, h('p', null, 'No import yet (data_msgr\\train.jsonl is missing). Do wizard steps 1 and 2 first.'))); return; }
    const mk = (id, label, v, tip) => {
      const input = h('input', { id, type: 'number', min: '0', max: '100', value: String(v), title: tip });
      return { input, wrap: h('div', { class: 'field', style: 'max-width:200px' }, h('label', { for: id }, label), input) };
    };
    const dShort = mk('cd1', 'Keep a short reply at most', 3, 'Replies of 25 characters or less. 0 = no limit.');
    const dLong = mk('cd2', 'Keep a longer reply at most', 5, '0 = no limit.');
    const others = h('input', { id: 'ck', type: 'checkbox' });
    const prev = h('button', { class: 'btn primary', type: 'button' }, 'Preview');
    const out = h('div', { 'aria-live': 'polite' });
    box.append(h('div', { class: 'row' }, dShort.wrap, dLong.wrap, h('label', { class: 'row', for: 'ck', style: 'gap:6px' }, others, 'Keep other people\u2019s lines in group chats')), h('div', { class: 'row', style: 'margin-top:8px' }, prev), out);
    const qs = () => 'max_dup=' + encodeURIComponent(dShort.input.value) + '&max_dup_long=' + encodeURIComponent(dLong.input.value) + '&keep_others=' + (others.checked ? '1' : '0');
    prev.addEventListener('click', async () => {
      prev.disabled = true;
      out.replaceChildren(h('p', { class: 'muted', role: 'status' }, 'Working on it (a few seconds)\u2026'));
      try { paintPlan(out, await api('/api/data/clean-preview?' + qs()), qs); }
      catch (e) { out.replaceChildren(h('div', { class: 'callout bad', role: 'alert' }, h('p', null, e.message))); }
      prev.disabled = false;
    });
  }

  const REMOVED = { system_msgs_removed: 'call / group notices removed', dropped_no_target: 'examples with nothing to learn from', dropped_other_group_speaker: 'other people\u2019s lines in group chats',
    dropped_repeated_short: 'repeated short replies', dropped_repeated_long: 'repeated longer replies' };

  function replyBlock(title, r) {
    const max = Math.max(1, ...r.buckets.map((b) => b.examples));
    return h('div', { class: 'card' }, h('h3', null, title),
      h('p', { class: 'small' }, fmt(r.total) + ' answers, ' + fmt(r.distinct) + ' different. The 10 most common ones make up ' + Math.round(r.top10_share * 100) + '% of them.'),
      ...r.buckets.map((b) => h('div', { class: 'row', style: 'gap:8px;flex-wrap:nowrap' },
        h('span', { class: 'small', style: 'width:110px;flex:none' }, b.label),
        h('span', { class: 'meter', style: 'flex:1;width:auto' }, h('span', { class: 'meter-fill', style: 'width:' + Math.round(100 * b.examples / max) + '%' })),
        h('span', { class: 'small muted', style: 'width:170px;flex:none;text-align:right' }, fmt(b.replies) + ' replies, ' + fmt(b.examples) + ' examples'))));
  }

  function paintPlan(out, p, qs) {
    const kids = [];
    const tr = p.splits.train, va = p.splits.val;
    kids.push(h('div', { class: 'dstats' }, stat('Training examples', fmt(tr.before) + ' \u2192 ' + fmt(tr.after)), stat('Validation examples', fmt(va.before) + ' \u2192 ' + fmt(va.after)),
      stat('Chats', fmt(p.chats.before) + ' \u2192 ' + fmt(p.chats.with_train_file) + ' files')));
    const rem = Object.keys(REMOVED).map((k) => [k, tr.removed[k], va.removed[k]]).filter((r) => r[1] || r[2]);
    kids.push(h('h3', null, 'What was removed'), rem.length
      ? h('div', { class: 'dkv' }, rem.map((r) => h('span', { class: 'badge' }, REMOVED[r[0]] + ': ' + fmt(r[1]) + ' train, ' + fmt(r[2]) + ' val')))
      : h('p', { class: 'small muted' }, 'Nothing needed removing.'));
    if (p.chats.val_only) kids.push(h('div', { class: 'callout warn' }, h('p', null, p.chats.val_only + ' chat(s) would have validation examples but no training file left, so their validation part is not written.')));
    kids.push(h('div', { class: 'grid' }, replyBlock('Repeated replies before', p.replies.before), replyBlock('Repeated replies after', p.replies.after)));

    const txt = h('div', { 'aria-live': 'polite' });
    const show = h('button', { class: 'btn', type: 'button' }, 'Show the most repeated replies');
    show.addEventListener('click', async () => {
      show.disabled = true;
      try {
        const r = await api('/api/data/repeats?text=1&' + qs());
        txt.replaceChildren(h('p', { class: 'small muted' }, r.note), r.top.length ? h('ul', { class: 'list' }, r.top.map((t) => h('li', null, h('div', { class: 'grow name' }, t.reply), h('span', { class: 'badge' }, t.before + ' \u2192 ' + t.after)))) : h('p', { class: 'small muted' }, 'No reply is repeated.'));
      } catch (e) { txt.replaceChildren(h('div', { class: 'callout bad', role: 'alert' }, h('p', null, e.message))); show.disabled = false; }
    });
    kids.push(h('div', { class: 'row' }, show), txt);

    const o = p.output;
    const apply = h('button', { class: 'btn primary', type: 'button' }, 'Write ' + o.folder);
    let armed = false;
    const result = h('div', { 'aria-live': 'polite' });
    apply.addEventListener('click', async () => {
      if (!o.usable) { toast(o.folder + ' cannot be used (it points outside the project).', 'bad'); return; }
      if (!armed) { armed = true; apply.textContent = 'Yes, write it now'; setTimeout(() => { if (armed) { armed = false; apply.textContent = 'Write ' + o.folder; } }, 6000); return; }
      armed = false; apply.disabled = true;
      try {
        const r = await api('/api/data/clean-apply', { plan_id: p.plan_id });
        toast('Written: ' + r.chat_files + ' chat files', 'ok');
        const cmd = r.next_command;
        const c = h('button', { class: 'btn', type: 'button' }, 'Copy'); c.addEventListener('click', async () => { try { await navigator.clipboard.writeText(cmd); toast('Copied', 'ok'); } catch (e) { toast('Could not copy', 'bad'); } });
        result.replaceChildren(h('div', { class: 'callout ok' }, h('p', null, 'Wrote ' + fmt(r.chat_files) + ' chat files (' + fmt(r.train_examples) + ' training and ' + fmt(r.val_examples) + ' validation examples) to ' + r.written + '.' + (r.moved_old_to ? ' The older files were moved to ' + r.moved_old_to + '.' : ''))),
          h('p', { class: 'small muted' }, 'Next, turn them into token files:'), h('div', { class: 'dcmd' }, h('code', null, cmd), c));
      } catch (e) { result.replaceChildren(h('div', { class: 'callout bad', role: 'alert' }, h('p', null, e.message))); apply.disabled = false; }
    });
    kids.push(h('h3', null, 'Write the files'),
      h('p', { class: 'small muted' }, o.exists && o.train_files ? o.folder + ' already holds ' + o.train_files + ' chat files. They are moved to _to_delete\\ (not deleted) first.' : o.folder + ' is empty or missing, so nothing is moved.'),
      h('div', { class: 'row' }, apply), result,
      h('p', { class: 'small muted' }, 'Same result as: ' + p.command));
    out.replaceChildren(...kids);
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
