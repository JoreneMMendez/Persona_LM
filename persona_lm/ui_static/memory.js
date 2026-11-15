// PersonaLM UI, stage U2a: the Memory page. Searchable list, add / edit / pin / forget, and the maintenance preview.
// Memory text is only ever put on the page with textContent (never as HTML). Nothing here is sent anywhere but this server.
import { h, api, toast } from './app.js';

const PAGE = 100;
const STATUS_LABEL = { all: 'All', active: 'Active', dormant: 'Dormant', archived: 'Archived', merged: 'Merged' };
const STATUS_HELP = {
  active: 'Used when the AI recalls things.',
  dormant: 'Faded away. Kept in the file, not recalled.',
  archived: 'Replaced by a summary memory.',
  merged: 'A duplicate, folded into another memory.',
};
const fmtDate = (iso) => { try { return new Date(iso).toLocaleDateString(undefined, { year: 'numeric', month: 'short', day: 'numeric' }); } catch (e) { return ''; } };

export async function memoryPage(main) {
  const state = { q: '', status: 'all', sort: 'newest', pinned: false, offset: 0 };
  let data = null, timer = 0, seq = 0, editing = null;

  // ---------- list
  const listBox = h('div', { class: 'mem-list', 'aria-live': 'polite' });
  const summary = h('div', { class: 'row mem-summary' });
  const moreBtn = h('button', { class: 'btn', type: 'button', onclick: () => load(true) }, 'Show more');
  const matchTxt = h('span', { class: 'muted small', role: 'status' });

  function chips(m) {
    return h('div', { class: 'row mem-meta' },
      h('span', { class: 'badge' + (m.status === 'active' ? '' : ' warn') }, m.status),
      m.pinned ? h('span', { class: 'badge accent' }, 'pinned') : null,
      h('span', { class: 'badge info' }, m.emotion),
      h('span', { class: 'meter mem-bar', title: 'Importance ' + m.importance.toFixed(2), role: 'img', 'aria-label': 'Importance ' + m.importance.toFixed(2) },
        h('span', { class: 'meter-fill', style: 'width:' + Math.round(m.importance * 100) + '%' })),
      h('span', { class: 'muted small', title: 'Importance fades with time unless pinned' }, 'strength ' + m.strength.toFixed(2)),
      m.uses ? h('span', { class: 'muted small' }, 'used ' + m.uses + '×') : null,
      h('span', { class: 'muted small' }, fmtDate(m.timestamp)),
      h('span', { class: 'path' }, m.id));
  }

  function card(m) {
    const armed = { t: 0 };
    const forgetBtn = h('button', { class: 'btn ghost', type: 'button', 'aria-label': 'Forget memory ' + m.id }, 'Forget');
    forgetBtn.addEventListener('click', async () => {
      if (!armed.t) {                                              // two steps: forgetting cannot be undone
        armed.t = setTimeout(() => { armed.t = 0; forgetBtn.classList.remove('danger'); forgetBtn.textContent = 'Forget'; }, 4000);
        forgetBtn.classList.add('danger'); forgetBtn.textContent = 'Really forget?'; return;
      }
      clearTimeout(armed.t); armed.t = 0;
      forgetBtn.disabled = true;
      try { await api('/api/memory/forget', { id: m.id, confirm: m.id }); toast('Forgot ' + m.id, 'ok'); await load(false, true); }
      catch (e) { forgetBtn.disabled = false; toast(e.message, 'bad'); }
    });
    const pinBtn = h('button', { class: 'btn ghost', type: 'button', 'aria-pressed': String(m.pinned), 'aria-label': (m.pinned ? 'Unpin ' : 'Pin ') + m.id,
      title: m.pinned ? 'Pinned memories never fade. Click to unpin.' : 'Pinned memories never fade.' }, m.pinned ? 'Unpin' : 'Pin');
    pinBtn.addEventListener('click', async () => {
      pinBtn.disabled = true;
      try { await api('/api/memory/edit', { id: m.id, pinned: !m.pinned }); await load(false, true); }
      catch (e) { pinBtn.disabled = false; toast(e.message, 'bad'); }
    });
    const links = m.summary_of.length ? h('p', { class: 'small muted' }, 'Summary of ' + m.summary_of.join(', ')) : m.merged_into ? h('p', { class: 'small muted' }, 'Merged into ' + m.merged_into) : null;
    return h('article', { class: 'card mem-item' + (m.status === 'active' ? '' : ' faded'), 'aria-label': 'Memory ' + m.id },
      h('p', { class: 'mem-text' }, m.text),
      m.nodes.length ? h('div', { class: 'row mem-nodes' }, m.nodes.map((n) => h('span', { class: 'chip', title: n.kind }, n.name))) : null,
      links, chips(m),
      h('div', { class: 'row mem-actions' }, h('button', { class: 'btn ghost', type: 'button', 'aria-label': 'Edit memory ' + m.id, onclick: () => openForm(m) }, 'Edit'), pinBtn, forgetBtn));
  }

  function paint() {
    const c = data.counts;
    summary.replaceChildren(
      h('span', { class: 'badge accent' }, data.total + (data.total === 1 ? ' memory' : ' memories')),
      ...['active', 'dormant', 'archived', 'merged'].filter((s) => c[s]).map((s) => h('span', { class: 'badge', title: STATUS_HELP[s] }, c[s] + ' ' + s)),
      h('span', { class: 'badge' }, data.pinned + ' pinned'), h('span', { class: 'badge' }, data.nodes + (data.nodes === 1 ? ' node' : ' nodes')));
    matchTxt.textContent = data.total ? (data.matched === data.total ? '' : data.matched + ' match') : '';
    if (!data.exists || data.total === 0) {
      listBox.replaceChildren(h('div', { class: 'empty' }, h('h2', null, data.exists ? 'No memories yet' : 'No memory file yet'),
        h('p', null, 'Memories are facts the AI can recall in a chat, kept in memory.json on this computer. Add the first one to start.'),
        h('p', null, h('button', { class: 'btn primary', type: 'button', onclick: () => openForm(null) }, 'Add a memory'))));
    } else if (!data.matched) {
      listBox.replaceChildren(h('div', { class: 'empty' }, h('h2', null, 'Nothing matches'), h('p', null, 'Try fewer words, or clear the filters.'),
        h('p', null, h('button', { class: 'btn', type: 'button', onclick: clearFilters }, 'Clear filters'))));
    } else listBox.replaceChildren(...data.items.map(card));
    moreBtn.hidden = !(data.offset + data.items.length < data.matched);
    maintBtn.disabled = !data.exists || !data.total;
  }

  async function load(more, keep) {
    const mine = ++seq;
    if (more && data) state.offset = data.items.length + data.offset; else state.offset = 0;
    const qs = new URLSearchParams({ q: state.q, status: state.status, sort: state.sort, limit: String(more || !keep || !data ? PAGE : Math.min(500, Math.max(PAGE, data.items.length))), offset: more && data ? String(state.offset) : '0' });
    if (state.pinned) qs.set('pinned', '1');
    try {
      const r = await api('/api/memory/list?' + qs.toString());
      if (mine !== seq) return;                                    // a newer search already replaced this one
      if (more && data) { r.items = data.items.concat(r.items); r.offset = 0; }
      data = r; paint();
    } catch (e) {
      if (mine !== seq) return;
      listBox.replaceChildren(h('div', { class: 'callout bad', role: 'alert' }, h('p', null, e.message)));
    }
  }

  function clearFilters() { state.q = ''; state.status = 'all'; state.pinned = false; search.value = ''; statusSel.value = 'all'; pinOnly.checked = false; load(false); }

  // ---------- toolbar
  const search = h('input', { type: 'search', id: 'mem-q', placeholder: 'Search words, names, ids…', 'aria-label': 'Search memories', maxlength: '200', autocomplete: 'off' });
  search.addEventListener('input', () => { clearTimeout(timer); timer = setTimeout(() => { state.q = search.value.trim(); load(false); }, 250); });
  const statusSel = h('select', { id: 'mem-status', 'aria-label': 'Show memories with status' }, Object.entries(STATUS_LABEL).map(([v, l]) => h('option', { value: v }, l)));
  statusSel.addEventListener('change', () => { state.status = statusSel.value; load(false); });
  const sortSel = h('select', { id: 'mem-sort', 'aria-label': 'Sort memories' },
    [['newest', 'Newest first'], ['oldest', 'Oldest first'], ['strength', 'Strongest first'], ['importance', 'Most important first']].map(([v, l]) => h('option', { value: v }, l)));
  sortSel.addEventListener('change', () => { state.sort = sortSel.value; load(false); });
  const pinOnly = h('input', { type: 'checkbox', id: 'mem-pin' });
  pinOnly.addEventListener('change', () => { state.pinned = pinOnly.checked; load(false); });

  const addBtn = h('button', { class: 'btn primary', type: 'button', onclick: () => openForm(null) }, 'Add memory');
  const maintBtn = h('button', { class: 'btn', type: 'button', onclick: openMaint, title: 'Preview a tidy-up: fade, merge duplicates, summarise' }, 'Tidy up…');

  // ---------- add / edit dialog
  const fText = h('textarea', { id: 'mem-f-text', rows: '4', maxlength: '2000', 'aria-describedby': 'mem-f-hint' });
  const fEmo = h('input', { type: 'text', id: 'mem-f-emo', maxlength: '24', placeholder: 'neutral', autocomplete: 'off' });
  const fImp = h('input', { type: 'range', id: 'mem-f-imp', min: '0', max: '1', step: '0.05', value: '0.5' });
  const fImpN = h('output', { for: 'mem-f-imp', class: 'small' }, '0.50');
  fImp.addEventListener('input', () => { fImpN.textContent = Number(fImp.value).toFixed(2); });
  const fPin = h('input', { type: 'checkbox', id: 'mem-f-pin' });
  const fErr = h('div', { class: 'callout bad', role: 'alert', hidden: true });
  const fTitle = h('h2', { id: 'mem-f-title' }, 'Add a memory');
  const fSave = h('button', { class: 'btn primary', type: 'submit' }, 'Save');
  const form = h('form', { class: 'dialog-body stack', novalidate: true }, fTitle,
    h('div', { class: 'field' }, h('label', { for: 'mem-f-text' }, 'What should the AI remember?'), fText,
      h('span', { class: 'hint', id: 'mem-f-hint' }, 'One fact. Write [[Name]] or [[person:Name]] to link it to a person or topic (kinds: person, topic, event, feeling).')),
    h('div', { class: 'field' }, h('label', { for: 'mem-f-emo' }, 'Feeling tag'), fEmo, h('span', { class: 'hint' }, 'One lowercase word, like joy, stress or neutral.')),
    h('div', { class: 'field' }, h('label', { for: 'mem-f-imp' }, 'Importance'), h('div', { class: 'row' }, fImp, fImpN), h('span', { class: 'hint' }, '0.5 is normal, 0.9 is very important.')),
    h('label', { class: 'check', for: 'mem-f-pin' }, fPin, h('span', null, 'Pin it (never fades)')), fErr,
    h('div', { class: 'dialog-actions' }, h('button', { class: 'btn', type: 'button', onclick: () => dlg.close() }, 'Cancel'), fSave));
  const dlg = h('dialog', { class: 'dialog', 'aria-labelledby': 'mem-f-title' }, form);
  dlg.addEventListener('click', (e) => { if (e.target === dlg) dlg.close(); });

  function openForm(m) {
    editing = m;
    fTitle.textContent = m ? 'Edit memory ' + m.id : 'Add a memory';
    fText.value = m ? m.text : ''; fEmo.value = m ? m.emotion : ''; fImp.value = m ? m.importance : '0.5'; fImpN.textContent = Number(fImp.value).toFixed(2); fPin.checked = m ? m.pinned : false;
    fErr.hidden = true; dlg.showModal(); fText.focus();
  }
  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    fSave.disabled = true; fErr.hidden = true;
    const body = { text: fText.value, emotion: fEmo.value.trim() || 'neutral', importance: Number(fImp.value), pinned: fPin.checked };
    try {
      if (editing) await api('/api/memory/edit', Object.assign({ id: editing.id }, body)); else await api('/api/memory/add', body);
      dlg.close(); toast(editing ? 'Saved ' + editing.id : 'Remembered', 'ok'); await load(false, true);
    } catch (err) { fErr.replaceChildren(h('p', null, err.message)); fErr.hidden = false; }
    fSave.disabled = false;
  });

  // ---------- maintenance dialog (a preview first; applying needs the preview's plan id)
  const mBody = h('div', { class: 'stack' });
  const mApply = h('button', { class: 'btn primary', type: 'button', disabled: true }, 'Apply this plan');
  const mDlg = h('dialog', { class: 'dialog wide', 'aria-labelledby': 'mem-m-title' }, h('div', { class: 'dialog-body stack' },
    h('h2', { id: 'mem-m-title' }, 'Tidy up memory'),
    h('p', { class: 'small muted' }, 'This is a preview. Nothing changes until you press Apply. Nothing is deleted: faded memories become dormant, duplicates are marked merged, old groups are archived behind a summary.'),
    mBody, h('div', { class: 'dialog-actions' }, h('button', { class: 'btn', type: 'button', onclick: () => mDlg.close() }, 'Close'), mApply)));
  mDlg.addEventListener('click', (e) => { if (e.target === mDlg) mDlg.close(); });

  const short = (t) => (t.length > 140 ? t.slice(0, 139) + '…' : t);
  function section(title, help, rows) {
    return rows.length ? h('section', { class: 'card' }, h('h3', null, title + ' (' + rows.length + ')'), h('p', { class: 'small muted' }, help), h('ul', { class: 'list' }, rows)) : null;
  }
  async function openMaint() {
    mApply.disabled = true; mApply.onclick = null;
    mBody.replaceChildren(h('div', { class: 'row' }, h('span', { class: 'spinner', role: 'status', 'aria-label': 'Loading' }), h('span', { class: 'muted' }, 'Working out a plan…')));
    mDlg.showModal();
    let p;
    try { p = await api('/api/memory/plan'); } catch (e) { mBody.replaceChildren(h('div', { class: 'callout bad', role: 'alert' }, h('p', null, e.message))); return; }
    if (p.empty) { mBody.replaceChildren(h('div', { class: 'callout ok' }, h('p', null, 'Nothing to tidy right now. Memory is in good shape.'))); return; }
    const li = (id, text, extra) => h('li', null, h('div', { class: 'grow' }, h('div', null, short(text)), extra ? h('div', { class: 'small muted' }, extra) : null), h('span', { class: 'path' }, id));
    mBody.replaceChildren(...[
      section('Fade to dormant', 'Weak and unused for a long time. Kept in the file, not recalled. Pin a memory first if it must stay.', p.fade.map((x) => li(x.id, x.text))),
      section('Bring back', 'Dormant, but used since. They become active again.', p.revive.map((x) => li(x.id, x.text))),
      section('Merge duplicates', 'Nearly the same words. The first is kept, the second is marked merged.', p.duplicates.map((x) => li(x.keep + ' ← ' + x.drop, x.keep_text, 'drops: ' + short(x.drop_text) + ' (' + Math.round(x.similarity * 100) + '% alike)'))),
      section('Summarise old groups', 'Several old memories about one topic become one summary memory; the originals are archived.', p.summaries.map((x) => li(x.ids.length + ' memories', x.text, 'about ' + x.node))),
    ].filter(Boolean));
    mApply.disabled = false;
    mApply.onclick = async () => {
      mApply.disabled = true;
      try { const r = await api('/api/memory/maintain', { plan_id: p.plan_id }); const d = r.done; mDlg.close();
        toast('Tidied: ' + d.fade + ' faded, ' + d.revive + ' revived, ' + d.merged + ' merged, ' + d.summaries + ' summaries', 'ok'); await load(false, true); }
      catch (e) { mApply.disabled = false; mBody.prepend(h('div', { class: 'callout bad', role: 'alert' }, h('p', null, e.message))); }
    };
  }

  // ---------- page
  main.replaceChildren(h('div', { class: 'stack' },
    h('p', { class: 'lead' }, 'What the AI remembers about you and the people in your life. Everything here is stored in memory.json on this computer, and only you can change it.'),
    h('div', { class: 'row mem-top' }, summary, h('div', { class: 'spacer' }), maintBtn, addBtn),
    h('div', { class: 'card mem-tools' },
      h('div', { class: 'field grow' }, h('label', { for: 'mem-q' }, 'Search'), search),
      h('div', { class: 'field' }, h('label', { for: 'mem-status' }, 'Status'), statusSel),
      h('div', { class: 'field' }, h('label', { for: 'mem-sort' }, 'Order'), sortSel),
      h('label', { class: 'check', for: 'mem-pin' }, pinOnly, h('span', null, 'Pinned only'))),
    matchTxt, listBox, h('div', { class: 'row', style: 'justify-content:center' }, moreBtn), dlg, mDlg));
  moreBtn.hidden = true;
  listBox.replaceChildren(h('div', { class: 'skeleton', style: 'height:96px' }), h('div', { class: 'skeleton', style: 'height:96px' }));
  await load(false);
}
