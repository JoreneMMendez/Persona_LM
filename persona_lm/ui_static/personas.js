// PersonaLM UI, stage U3b: the Personas page. A persona is a MASK: only prompt text changes, the AI is still itself underneath and its weights are the same.
// Wearing / taking off a mask is the owner's button here (it calls the same code as the typed /persona command). Adding, editing and removing masks stay in the terminal.
// All text from files goes on the page with textContent only (through h()).
import { h, api, toast } from './app.js';

export async function personasPage(main) {
  const root = h('div', { class: 'stack personas' });
  main.replaceChildren(root);
  let viewCore = null;                                       // null = the loaded identity core (switchable); else a core folder shown read only

  async function load() {
    root.replaceChildren(h('p', { class: 'muted', role: 'status' }, 'Loading\u2026'));
    let d;
    try { d = await api('/api/personas' + (viewCore ? '?core=' + encodeURIComponent(viewCore) : '')); }
    catch (e) { root.replaceChildren(h('div', { class: 'callout bad', role: 'alert' }, h('p', null, e.message))); return; }
    paint(d);
  }

  async function wear(name, btn) {
    btn.disabled = true;
    try {
      const r = await api('/api/personas/use', { name });
      toast(r.message, 'ok');
      paint(r);
    } catch (e) { btn.disabled = false; toast(e.message, 'bad'); }
  }

  function paint(d) {
    const out = [];
    out.push(h('div', { class: 'card' }, h('h2', null, 'Personas (masks)'),
      h('p', { class: 'muted' }, 'A persona is a mask: a short text added to what the model reads. The AI is still itself underneath, and its weights do not change. The conversation is kept when you switch; press Clear in Chat to start fresh.')));
    const cores = d.cores || [];
    if (cores.length) {
      const sel = h('select', { id: 'pc', 'aria-label': 'Identity core to show' });
      sel.append(h('option', { value: '' }, d.loaded && !viewCore ? 'The core loaded in Chat (' + d.core + ')' : 'The core loaded in Chat'));
      cores.forEach((c) => sel.append(h('option', { value: c.path }, c.path)));
      sel.value = viewCore || '';
      sel.addEventListener('change', () => { viewCore = sel.value || null; load(); });
      out.push(h('div', { class: 'field' }, h('label', { for: 'pc' }, 'Show the masks of'), sel));
    }
    if (!d.loaded) {
      out.push(h('div', { class: 'callout' }, h('p', null, d.has_model
        ? 'The model loaded in Chat is a plain checkpoint, so it has no masks. Masks belong to an identity core: load one in Chat (Identity core tab) to wear them.'
        : 'No identity core is loaded. Open Chat, pick the Identity core tab and load one to wear a mask. You can still look at a core\u2019s masks with the list above.')));
      out.push(h('a', { class: 'btn', href: '#/chat' }, 'Go to Chat'));
      root.replaceChildren(...out); return;
    }
    out.push(h('div', { class: 'row' }, h('span', { class: 'badge accent' }, d.core), h('span', { class: 'badge' }, d.ai_name),
      h('span', { class: 'badge ' + (d.active ? 'ok' : '') }, d.active ? 'wearing: ' + d.active : 'no mask on'),
      d.switchable ? null : h('span', { class: 'badge warn' }, 'read only (not the loaded core)')));
    if (d.switchable && d.active) {
      const off = h('button', { class: 'btn', type: 'button' }, 'Take the mask off');
      off.addEventListener('click', () => wear('none', off));
      out.push(h('div', { class: 'row' }, off));
    }
    if (!d.personas.length) {
      out.push(h('div', { class: 'callout' }, h('p', null, 'This core has no personas yet. Add one in the terminal:'), h('code', null, 'python -m persona_lm persona add "calm mentor" "You speak slowly and kindly."')));
    }
    d.personas.forEach((p) => {
      const btn = h('button', { class: 'btn' + (p.active ? '' : ' primary'), type: 'button' }, p.active ? 'Take off' : 'Wear this mask');
      btn.addEventListener('click', () => wear(p.active ? 'none' : p.name, btn));
      out.push(h('div', { class: 'card persona' + (p.active ? ' on' : '') },
        h('div', { class: 'row between' }, h('h3', null, p.name),
          h('div', { class: 'row' }, p.active ? h('span', { class: 'badge ok' }, 'on now') : null,
            p.examples ? h('span', { class: 'badge' }, p.examples + ' example' + (p.examples === 1 ? '' : 's')) : null,
            p.version ? h('span', { class: 'badge info' }, 'trained: ' + p.version) : null,
            p.stale ? h('span', { class: 'badge warn', title: 'The text changed after this version was trained, so the trained version learned the OLD text.' }, 'STALE') : null)),
        h('p', { class: 'persona-text' }, p.text),
        d.switchable ? h('div', { class: 'row' }, btn) : null));
    });
    if (d.block) out.push(h('div', { class: 'card' }, h('h3', null, 'What the model reads for the mask'),
      h('p', { class: 'small muted' }, 'This exact sentence goes into every prompt while the mask is on (the Prompt button in Chat shows it in place). If it is too long for the model\u2019s window, sending a message is refused and nothing is kept.'),
      h('pre', { class: 'pp-text' }, d.block)));
    out.push(h('div', { class: 'card' }, h('h3', null, 'Add, edit or remove a mask'),
      h('p', { class: 'small muted' }, 'These stay in the terminal (each change is logged in the core\u2019s lineage):'),
      h('pre', { class: 'pp-text' }, 'python -m persona_lm persona add "NAME" "text"\npython -m persona_lm persona edit "NAME" "new text"\npython -m persona_lm persona remove "NAME"')));
    root.replaceChildren(...out);
  }
  load();
}
