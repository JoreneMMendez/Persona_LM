// PersonaLM UI, stage U2a: the Memory page. Searchable list, add / edit / pin / forget, and the maintenance plan with a preview.
// Memory texts are put on the page with textContent only (never as HTML). Every change is something YOU press; the AI cannot reach these buttons.
import { h, api, toast } from './app.js';

const PAGE = 100;
let epoch = 0;           // every visit gets a new number; an old visit stops itself when it sees a newer one

const dateOf = (iso) => { try { return new Date(iso).toLocaleDateString(undefined, { year: 'numeric', month: 'short', day: 'numeric' }); } catch (e) { return ''; } };
const pct = (x) => Math.round(Math.max(0, Math.min(1, x)) * 100);

export async function memoryPage(main) {
  const mine = ++epoch;
  const alive = () => mine === epoch;
  const f = { q: '', status: 'all', sort: 'recent' };
  let data = null, items = [], busy = false;

  // ---- skeleton
  const search = h('input', { type: 'search', id: 'mem-search', placeholder: 'Search words, names, emotions or an id like m0003', 'aria-label': 'Search memories', maxlength: '200', autocomplete: 'off' });
  const statusSel = h('select', { id: 'mem-status', 'aria-label': 'Show memories with status' },
    [['all', 'All statuses'], ['active', 'Active'], ['dormant', 'Dormant'], ['archived', 'Archived'], ['merged', 'Merged']].map(([v, t]) => h('option', { value: v }, t)));
  const sortSel = h('select', { id: 'mem-sort', 'aria-label': 'Sort by' },
    [['recent', 'Newest first'], ['oldest', 'Oldest first'], ['strength', 'Strongest first'], ['importance', 'Most important first']].map(([v, t]) => h('option', { value: v }, t)));
  const addBtn = h('button', { class: 'btn primary', type: 'button' }, 'Add memory');
  const planBtn = h('button', { class: 'btn', type: 'button' }, 'Maintenance…');
  const summary = h('p', { class: 'small muted', role: 'status' });
  const list = h('ul', { class: 'list mem-list', 'aria-label': 'Memories' });
  const moreBtn = h('button', { class: 'btn', type: 'button', hidden: true }, 'Show more');
  const err = h('div', { class: 'callout bad', role: 'alert', hidden: true });
  const planSlot = h('div');

  main.replaceChildren(h('div', { class: 'stack' },
    h('p', { class: 'lead' }, 'What the AI remembers, kept in memory.json on this computer. Only you can add, change or forget a memory here; nothing the AI says can.'),
    err,
    h('section', { class: 'card', 'aria-label': 'Find and manage memories' },
      h('div', { class: 'mem-bar' }, search, statusSel, sortSel),
      h('div', { class: 'row', style: 'margin-top:12px' }, addBtn, planBtn, h('div', { class: 'spacer' }), summary)),
    planSlot,
    h('section', { class: 'card mem-card' }, list, h('div', { class: 'row', style: 'justify-content:center;margin-top:12px' }, moreBtn))));
  list.append(h('li', null, h('div', { class: 'skeleton', style: 'height:56px;width:100%' })));

  function fail(e) { err.hidden = false; err.replaceChildren(h('p', null, e.message)); }

  // ---- loading
  async function load(append) {
    const offset = append ? items.length : 0;
    const p = new URLSearchParams({ q: f.q, status: f.status, sort: f.sort, limit: String(PAGE), offset: String(offset) });
    let d;
    try { d = await api('/api/memory/list?' + p); } catch (e) { if (alive()) fail(e); return; }
    if (!alive()) return;
    err.hidden = true;
    data = d;
    items = append ? items.concat(d.memories) : d.memories;
    paint();
  }

  function paint() {
    const c = data.counts;
    const parts = ['active', 'dormant', 'archived', 'merged'].filter((s) => c[s]).map((s) => c[s] + ' ' + s);
    summary.textContent = data.total ? ('Showing ' + items.length + ' of ' + data.matches + (data.matches !== data.total ? ' matching' : '') + ' · ' + data.total + ' in total' + (parts.length ? ' (' + parts.join(', ') + ')' : '')) : '';
    list.replaceChildren();
    if (!data.total) {
      list.append(h('li', { class: 'mem-empty' }, h('div', null,
        h('h2', null, data.exists ? 'Memory is empty' : 'No memory file yet'),
        h('p', { class: 'muted' }, data.exists ? 'Nothing is remembered yet.' : 'memory.json does not exist yet. It is created when you save your first memory.'),
        h('p', null, h('button', { class: 'btn primary', type: 'button', onclick: () => openForm(null) }, 'Add the first memory')))));
    } else if (!items.length) {
      list.append(h('li', { class: 'mem-empty' }, h('div', null, h('h2', null, 'Nothing matches'), h('p', { class: 'muted' }, 'Try fewer words, or set the status to “All statuses”.'))));
    } else {
      items.forEach((m) => list.append(row(m)));
    }
    moreBtn.hidden = items.length >= data.matches;
  }

  function row(m) {
    const meter = h('span', { class: 'meter mini', role: 'img', 'aria-label': 'Strength ' + pct(m.strength) + ' percent' }, h('span', { class: 'meter-fill', style: 'width:' + pct(m.strength) + '%' }));
    const pin = h('button', { class: 'btn ghost', type: 'button', 'aria-pressed': m.pinned ? 'true' : 'false', title: m.pinned ? 'Pinned: it never fades. Click to unpin.' : 'Pin: keep it from fading' }, m.pinned ? 'Unpin' : 'Pin');
    pin.addEventListener('click', () => change(pin, { id: m.id, pinned: !m.pinned }, m.pinned ? 'Unpinned ' + m.id : 'Pinned ' + m.id));
    return h('li', { class: 'mem' + (m.status !== 'active' ? ' off' : ''), 'data-id': m.id },
      h('div', { class: 'grow' },
        h('div', { class: 'mem-text' }, m.text),
        h('div', { class: 'row mem-meta' },
          h('span', { class: 'mono small muted' }, m.id),
          m.status !== 'active' ? h('span', { class: 'badge warn' }, m.status) : null,
          m.pinned ? h('span', { class: 'badge accent' }, 'pinned') : null,
          h('span', { class: 'badge' }, m.emotion),
          h('span', { class: 'small muted' }, 'importance ' + m.importance.toFixed(2)),
          h('span', { class: 'row', style: 'gap:6px' }, meter, h('span', { class: 'small muted' }, 'strength ' + m.strength.toFixed(2))),
          m.uses ? h('span', { class: 'small muted' }, 'used ' + m.uses + '×') : null,
          h('span', { class: 'small muted' }, dateOf(m.timestamp))),
        m.nodes.length ? h('div', { class: 'row mem-nodes', 'aria-label': 'Linked to' }, m.nodes.map((n) => h('span', { class: 'chip', title: n.kind }, n.name))) : null),
      h('div', { class: 'mem-actions' },
        pin,
        h('button', { class: 'btn ghost', type: 'button', onclick: () => openForm(m) }, 'Edit'),
        h('button', { class: 'btn ghost danger-text', type: 'button', onclick: () => openForget(m) }, 'Forget')));
  }

  async function change(btn, body, okMsg) {
    if (busy) return;
    busy = true; btn.disabled = true;
    try { await api('/api/memory/edit', body); toast(okMsg, 'ok'); await load(false); }
    catch (e) { toast(e.message, 'bad'); }
    finally { busy = false; btn.disabled = false; }
  }

  // ---- dialogs
  function dialog(title, body, actions) {
    const dlg = h('dialog', { class: 'dialog', 'aria-label': title }, h('div', { class: 'dialog-body' }, h('h2', null, title), body, h('div', { class: 'dialog-actions' }, actions)));
    dlg.addEventListener('close', () => dlg.remove());
    dlg.addEventListener('click', (e) => { if (e.target === dlg) dlg.close(); });
    document.body.append(dlg);
    dlg.showModal();
    return dlg;
  }

  function openForm(mem) {
    const max = (data && data.max_text) || 2000;
    const text = h('textarea', { id: 'mf-text', rows: '4', maxlength: String(max), required: true, 'aria-describedby': 'mf-hint' });
    const count = h('span', { class: 'small muted' });
    const emo = h('input', { type: 'text', id: 'mf-emo', list: 'mf-emos', maxlength: '24', autocomplete: 'off', 'aria-describedby': 'mf-emo-hint' });
    const emos = h('datalist', { id: 'mf-emos' }, [...new Set(['neutral', 'joy', 'sadness', 'anger', 'fear', 'stress', 'calm', 'love'].concat((data && data.emotions) || []))].map((e) => h('option', { value: e })));
    const imp = h('input', { type: 'range', id: 'mf-imp', min: '0', max: '1', step: '0.05' });
    const impOut = h('output', { for: 'mf-imp', class: 'mono' });
    const pin = h('input', { type: 'checkbox', id: 'mf-pin' });
    const msg = h('div', { class: 'callout bad', role: 'alert', hidden: true });
    text.value = mem ? mem.text : ''; emo.value = mem ? mem.emotion : 'neutral'; imp.value = mem ? mem.importance : 0.5; pin.checked = !!(mem && mem.pinned);
    const sync = () => { count.textContent = text.value.length + ' / ' + max; impOut.textContent = Number(imp.value).toFixed(2); };
    text.addEventListener('input', sync); imp.addEventListener('input', sync); sync();
    const save = h('button', { class: 'btn primary', type: 'button' }, mem ? 'Save changes' : 'Save memory');
    const cancel = h('button', { class: 'btn', type: 'button' }, 'Cancel');
    const body = h('div', { class: 'stack', style: 'margin-top:12px' },
      h('div', { class: 'field' }, h('label', { for: 'mf-text' }, 'What to remember'), text,
        h('span', { class: 'hint', id: 'mf-hint' }, 'One fact, in plain words. Write [[Maria]] or [[person:Maria]] to link a person, topic, event or feeling (kinds: ' + ((data && data.kinds) || []).join(', ') + ').'), count),
      h('div', { class: 'field' }, h('label', { for: 'mf-emo' }, 'Emotion tag'), emo, emos, h('span', { class: 'hint', id: 'mf-emo-hint' }, 'One lowercase word, like joy or stress.')),
      h('div', { class: 'field' }, h('label', { for: 'mf-imp' }, 'Importance'), h('div', { class: 'row' }, imp, impOut), h('span', { class: 'hint' }, '0.5 is normal, 0.9 is very important. Less important memories fade sooner.')),
      h('label', { class: 'check', for: 'mf-pin' }, pin, 'Pin (never fades)'),
      mem ? h('p', { class: 'small muted' }, 'Editing the text re-links the names in it. The previous version stays in memory.prev.json until the next save.') : null,
      msg);
    const dlg = dialog(mem ? 'Edit ' + mem.id : 'Add a memory', body, [cancel, save]);
    cancel.addEventListener('click', () => dlg.close());
    text.focus();
    save.addEventListener('click', async () => {
      if (busy) return;
      msg.hidden = true; busy = true; save.disabled = true;
      try {
        if (mem) {
          const b = { id: mem.id };
          if (text.value.trim() !== mem.text) b.text = text.value;
          if (emo.value.trim().toLowerCase() !== mem.emotion) b.emotion = emo.value;
          if (Number(imp.value) !== mem.importance) b.importance = Number(imp.value);
          if (pin.checked !== mem.pinned) b.pinned = pin.checked;
          if (Object.keys(b).length === 1) { dlg.close(); toast('Nothing changed'); return; }
          await api('/api/memory/edit', b); toast('Saved ' + mem.id, 'ok');
        } else {
          const d = await api('/api/memory/add', { text: text.value, emotion: emo.value || 'neutral', importance: Number(imp.value), pinned: pin.checked });
          toast('Remembered as ' + d.memory.id, 'ok');
        }
        dlg.close();
        if (alive()) { planSlot.replaceChildren(); await load(false); }
      } catch (e) { msg.hidden = false; msg.replaceChildren(h('p', null, e.message)); }
      finally { busy = false; save.disabled = false; }
    });
  }

  function openForget(mem) {
    const go = h('button', { class: 'btn danger', type: 'button' }, 'Forget for good');
    const cancel = h('button', { class: 'btn', type: 'button' }, 'Keep it');
    const msg = h('div', { class: 'callout bad', role: 'alert', hidden: true });
    const dlg = dialog('Forget ' + mem.id + '?', h('div', { class: 'stack', style: 'margin-top:12px' },
      h('blockquote', { class: 'mem-quote' }, mem.text),
      h('p', { class: 'small' }, 'This deletes the memory from memory.json and removes the old text from the backup file too. It cannot be undone. Summaries that were built from it keep their own text.'), msg), [cancel, go]);
    cancel.addEventListener('click', () => dlg.close());
    cancel.focus();
    go.addEventListener('click', async () => {
      if (busy) return;
      busy = true; go.disabled = true; msg.hidden = true;
      try {
        await api('/api/memory/forget', { id: mem.id, confirm: mem.id });
        toast('Forgot ' + mem.id, 'ok'); dlg.close();
        if (alive()) { planSlot.replaceChildren(); await load(false); }
      } catch (e) { msg.hidden = false; msg.replaceChildren(h('p', null, e.message)); }
      finally { busy = false; go.disabled = false; }
    });
  }

  // ---- maintenance: preview first, apply only that preview
  async function showPlan() {
    planBtn.disabled = true;
    planSlot.replaceChildren(h('section', { class: 'card' }, h('div', { class: 'row' }, h('span', { class: 'spinner', role: 'status', 'aria-label': 'Loading' }), h('span', { class: 'muted' }, 'Working out what maintenance would do…'))));
    let p;
    try { p = await api('/api/memory/plan'); } catch (e) { planSlot.replaceChildren(h('div', { class: 'callout bad', role: 'alert' }, h('p', null, e.message))); planBtn.disabled = false; return; }
    planBtn.disabled = false;
    if (!alive()) return;
    const brief = (m) => h('li', null, h('div', { class: 'grow' }, h('div', { class: 'mem-text' }, m.text), h('div', { class: 'small muted' }, m.id + ' · strength ' + m.strength.toFixed(2) + ' · importance ' + m.importance.toFixed(2))));
    const block = (title, hint, rows) => rows.length ? h('div', null, h('h3', null, title + ' (' + rows.length + ')'), h('p', { class: 'small muted' }, hint), h('ul', { class: 'list' }, rows)) : null;
    const apply = h('button', { class: 'btn primary', type: 'button', disabled: p.empty }, 'Apply this plan');
    const close = h('button', { class: 'btn', type: 'button' }, 'Close');
    close.addEventListener('click', () => planSlot.replaceChildren());
    apply.addEventListener('click', async () => {
      if (busy) return;
      busy = true; apply.disabled = true;
      try {
        const d = (await api('/api/memory/apply', { plan_id: p.plan_id })).done;
        toast('Done: ' + d.fade + ' faded, ' + d.revive + ' revived, ' + d.merged + ' merged, ' + d.summaries + ' summarised', 'ok');
        planSlot.replaceChildren(); await load(false);
      } catch (e) { toast(e.message, 'bad'); apply.disabled = false; }
      finally { busy = false; }
    });
    planSlot.replaceChildren(h('section', { class: 'card', 'aria-labelledby': 'plan-h' },
      h('div', { class: 'card-head' }, h('h2', { id: 'plan-h' }, 'Maintenance preview'), h('span', { class: 'small muted' }, 'Nothing has been changed yet.')),
      p.empty ? h('p', { class: 'muted' }, 'Nothing to do: no memory is faint enough to fade, no duplicates, and no topic has enough old memories to summarise.') : h('div', { class: 'stack' },
        block('Will fade to dormant', 'Weak and unpinned. Dormant memories stay in the file but are not recalled.', p.fade.map(brief)),
        block('Will become active again', 'Dormant memories that were used since.', p.revive.map(brief)),
        block('Duplicates to merge', 'The weaker copy is marked merged into the one that is kept.', p.duplicates.map((d) => h('li', null, h('div', { class: 'grow' },
          h('div', { class: 'small muted' }, 'keep ' + d.keep.id + ' · merge ' + d.drop.id + ' into it · ' + Math.round(d.similarity * 100) + '% the same words'),
          h('div', { class: 'mem-text' }, d.keep.text), h('div', { class: 'mem-text muted' }, d.drop.text))))),
        block('Summaries to write', 'Old memories about one topic become one short summary memory. The originals are archived, not deleted.', p.summaries.map((s) => h('li', null, h('div', { class: 'grow' },
          h('div', { class: 'small muted' }, s.node + ' · ' + s.count + ' memories: ' + s.ids.join(', ')), h('div', { class: 'mem-text' }, s.text)))))),
      h('div', { class: 'dialog-actions', style: 'margin-top:12px' }, close, apply)));
  }

  // ---- wiring
  let timer = 0;
  search.addEventListener('input', () => { clearTimeout(timer); timer = setTimeout(() => { f.q = search.value.trim(); load(false); }, 250); });
  statusSel.addEventListener('change', () => { f.status = statusSel.value; load(false); });
  sortSel.addEventListener('change', () => { f.sort = sortSel.value; load(false); });
  moreBtn.addEventListener('click', () => load(true));
  addBtn.addEventListener('click', () => openForm(null));
  planBtn.addEventListener('click', showPlan);
  document.addEventListener('keydown', (e) => {
    if (!alive() || !document.getElementById('mem-search') || e.key !== '/' || e.metaKey || e.ctrlKey || e.altKey) return;
    const t = (e.target.tagName || '').toLowerCase();
    if (t === 'input' || t === 'textarea' || t === 'select') return;
    e.preventDefault(); search.focus();
  });

  await load(false);
}
