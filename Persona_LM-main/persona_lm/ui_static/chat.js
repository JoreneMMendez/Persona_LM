// PersonaLM UI, stages U1a + U1b: the Chat page. Pick a model, load it, talk to it; settings, save, clear, mood badge.
// Text from you and from the AI is only ever put on the page with textContent (never as HTML), and it is never a command:
// the server sends it to the model as a plain message.
import { h, api, toast, icon } from './app.js';

let epoch = 0;            // every visit to the page gets a new number; an old visit stops its own polling when it sees a newer one

const fmtDate = (t) => { try { return new Date(t * 1000).toLocaleDateString(undefined, { month: 'short', day: 'numeric' }); } catch (e) { return ''; } };
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

export async function chatPage(main) {
  const mine = ++epoch;
  main.replaceChildren(h('div', { class: 'stack' }, h('div', { class: 'skeleton', style: 'height:140px' }), h('div', { class: 'skeleton', style: 'height:260px' })));
  let st;
  try { st = await api('/api/chat/state'); }
  catch (e) { main.replaceChildren(h('div', { class: 'callout bad', role: 'alert' }, h('p', null, e.message))); return; }
  if (mine !== epoch) return;
  if (st.loaded) await showChat(main, mine, st); else await showLoader(main, mine, null);
}

// ------------------------------------------------------------------ choose and load a model
async function showLoader(main, mine, current) {
  let src;
  try { src = await api('/api/chat/sources'); }
  catch (e) { main.replaceChildren(h('div', { class: 'callout bad', role: 'alert' }, h('p', null, e.message))); return; }
  if (mine !== epoch) return;

  const ckSel = h('select', { id: 'ck', 'aria-describedby': 'ck-hint' });
  const tkSel = h('select', { id: 'tk' });
  const coreSel = h('select', { id: 'core' });
  const sys = h('textarea', { id: 'sys', rows: '2', maxlength: '600', placeholder: 'Optional. For example: You are Luna. Leave empty to chat without a system line.' });
  const ver = h('input', { id: 'ver', type: 'text', placeholder: 'only if the core has no current version, e.g. v1', maxlength: '40' });
  const moodBox = h('input', { type: 'checkbox', id: 'mood' });
  const err = h('div', { class: 'callout bad', role: 'alert', hidden: true });
  const loadBtn = h('button', { class: 'btn primary', type: 'button' }, 'Load model');
  let mode = src.checkpoints.length || !src.cores.length ? 'ckpt' : 'core';

  src.checkpoints.forEach((c) => ckSel.append(h('option', { value: c.path, 'data-tok': c.tokenizer || '' },
    c.path + '  ·  ' + c.size_mb + ' MB  ·  ' + fmtDate(c.modified) + (c.previous ? '  ·  previous save' : ''))));
  src.tokenizers.forEach((t) => tkSel.append(h('option', { value: t.path }, t.path)));
  src.cores.forEach((c) => coreSel.append(h('option', { value: c.path }, c.path)));
  const suggest = () => { const o = ckSel.selectedOptions[0]; if (o && o.dataset.tok) tkSel.value = o.dataset.tok; };
  ckSel.addEventListener('change', suggest); suggest();

  const ckBox = h('div', { class: 'stack' },
    h('div', { class: 'field' }, h('label', { for: 'ck' }, 'Checkpoint'), ckSel, h('span', { class: 'hint', id: 'ck-hint' }, 'Files named ckpt*.npz inside work, work2 and work3. The newest comes first.')),
    h('div', { class: 'field' }, h('label', { for: 'tk' }, 'Tokenizer'), tkSel, h('span', { class: 'hint' }, 'It must be the one the model was trained with (normally work/tok.json). It is filled in for you.')),
    h('div', { class: 'field' }, h('label', { for: 'sys' }, 'System line'), sys),
    h('div', { class: 'field' }, h('label', { class: 'check', for: 'mood' }, moodBox, ' Use a mood (mood.json)'),
      h('span', { class: 'hint' }, 'Off by default. When on, what you type moves the AI\u2019s mood, the mood is shown as a badge and saved in mood.json, and it also adds a mood line to what the model reads. The model was not fine-tuned on that line yet, so replies may change.')));
  const coreBox = h('div', { class: 'stack' },
    h('div', { class: 'field' }, h('label', { for: 'core' }, 'Identity core folder'), coreSel, h('span', { class: 'hint' }, 'Uses the core\u2019s current version, tokenizer, memory, mood and Person card. Nothing in it is changed by chatting.')),
    h('div', { class: 'field' }, h('label', { for: 'ver' }, 'Version (optional)'), ver));
  const tabBtn = (id, label) => h('button', { class: 'tab', role: 'tab', type: 'button', onclick: () => setMode(id) }, label);
  const tCk = tabBtn('ckpt', 'Checkpoint + tokenizer'), tCore = tabBtn('core', 'Identity core');
  const tabs = h('div', { class: 'tabs', role: 'tablist', 'aria-label': 'What to load' }, tCk, tCore);
  function setMode(m) {
    mode = m;
    tCk.setAttribute('aria-selected', String(m === 'ckpt')); tCore.setAttribute('aria-selected', String(m === 'core'));
    ckBox.hidden = m !== 'ckpt'; coreBox.hidden = m !== 'core';
  }

  const nothing = !src.checkpoints.length && !src.cores.length;
  loadBtn.addEventListener('click', async () => {
    err.hidden = true;
    const body = mode === 'core' ? { core: coreSel.value, version: ver.value.trim() || undefined } : { ckpt: ckSel.value, tokenizer: tkSel.value };
    if (sys.value.trim() && mode === 'ckpt') body.system = sys.value.trim();
    if (mode === 'ckpt' && moodBox.checked) body.mood = true;
    loadBtn.disabled = true; loadBtn.replaceChildren(h('span', { class: 'spinner', 'aria-hidden': 'true' }), ' Loading… (a few seconds)');
    try {
      const st = await api('/api/chat/load', body);
      if (mine !== epoch) return;
      toast('Model loaded', 'ok');
      await showChat(main, mine, st);
    } catch (e) {
      err.replaceChildren(h('p', null, e.message)); err.hidden = false;
      loadBtn.disabled = false; loadBtn.replaceChildren('Load model');
    }
  });

  const emptyIcon = icon('chat'); emptyIcon.classList.add('ico');
  main.replaceChildren(h('div', { class: 'stack' },
    h('p', { class: 'lead' }, 'Choose the model to talk to. Everything runs on this computer.'),
    nothing
      ? h('div', { class: 'empty' }, emptyIcon, h('h2', null, 'No checkpoints found'),
          h('p', null, 'The UI looks for ckpt*.npz files inside the work, work2 and work3 folders of the project. Start it from your project folder, or train a model first.'),
          h('p', null, h('code', null, 'python -m persona_lm ui --root C:\\Users\\user\\persona_lm_project')))
      : h('section', { class: 'card', 'aria-labelledby': 'h-load' },
          h('div', { class: 'card-head' }, h('h2', { id: 'h-load' }, 'Load a model'), current ? h('span', { class: 'muted small' }, 'Loading again starts a new conversation.') : null),
          h('div', { class: 'stack' }, src.cores.length && src.checkpoints.length ? tabs : null, ckBox, coreBox, err,
            h('div', { class: 'row' }, loadBtn, current ? h('button', { class: 'btn ghost', type: 'button', onclick: () => showChat(main, mine, current) }, 'Back to the chat') : null)))));
  if (!nothing) setMode(mode);
}

// ------------------------------------------------------------------ the conversation
async function showChat(main, mine, st) {
  const log = h('div', { class: 'chat-log', role: 'log', 'aria-live': 'polite', 'aria-label': 'Conversation', tabindex: '0' });
  const input = h('textarea', { id: 'msg', rows: '1', maxlength: '4000', placeholder: 'Write a message…  (Enter sends, Shift+Enter adds a line)', 'aria-label': 'Message' });
  const sendBtn = h('button', { class: 'btn primary', type: 'button', 'aria-label': 'Send message' }, 'Send');
  const meterFill = h('span', { class: 'meter-fill' });
  const meter = h('div', { class: 'meter', role: 'img' }, meterFill);
  const meterTxt = h('span', { class: 'small' });
  const draftTxt = h('span', { class: 'small muted' });
  const warn = h('div', { class: 'callout warn', role: 'status', hidden: true });
  let busy = false, tokens = st.tokens, draftTokens = 0, stopping = false, lastText = '', settings = st.settings, armed = 0;
  const moodBadge = h('span', { class: 'badge', hidden: !st.mood });
  function paintMood(m) {
    if (!m) { moodBadge.hidden = true; return; }
    moodBadge.hidden = false; moodBadge.textContent = 'feeling: ' + m.label;
    moodBadge.title = 'valence ' + m.valence.toFixed(1) + ' (unpleasant -1 to pleasant +1), arousal ' + m.arousal.toFixed(1) + ' (sleepy -1 to excited +1)';
    moodBadge.className = 'badge ' + (m.valence >= 0.25 ? 'ok' : m.valence <= -0.25 ? 'warn' : '');
  }
  paintMood(st.mood);

  const ai = st.name || 'AI';
  const bubble = (role, text) => h('div', { class: 'msg ' + (role === 'user' ? 'me' : 'ai') },
    h('div', { class: 'who' }, role === 'user' ? 'You' : ai), h('div', { class: 'bubble' }, text));
  const scrollDown = () => { log.scrollTop = log.scrollHeight; };

  function paintMeter() {
    const used = tokens.used + draftTokens, win = tokens.window;
    meterFill.style.width = Math.min(100, Math.round((used / win) * 100)) + '%';
    const over = used > tokens.prompt_budget;
    meter.classList.toggle('hot', over);
    meter.setAttribute('aria-label', 'Window use: ' + used + ' of ' + win + ' tokens');
    meterTxt.textContent = used + ' / ' + win + ' tokens';
    draftTxt.textContent = draftTokens ? ('this message: ' + draftTokens) : '';
    warn.hidden = !over;
    if (over) warn.replaceChildren(h('p', null, 'The conversation is longer than the AI can read at once (' + tokens.prompt_budget + ' tokens; the rest of the window is kept free for its reply). It will no longer see the oldest messages. They stay on this page.'));
  }

  st.history.forEach((m) => log.append(bubble(m.role, m.text)));
  if (!st.history.length) log.append(h('div', { class: 'chat-empty muted' }, 'Say something to start. The model is small, so short, plain sentences work best.'));

  let countTimer = 0;
  input.addEventListener('input', () => {
    input.style.height = 'auto'; input.style.height = Math.min(input.scrollHeight, 180) + 'px';
    clearTimeout(countTimer);
    if (!input.value.trim()) { draftTokens = 0; paintMeter(); return; }
    countTimer = setTimeout(async () => {
      try {
        const c = await api('/api/chat/count', { text: input.value });
        if (mine === epoch && !busy) { draftTokens = c.tokens; tokens = { ...tokens, used: c.used }; paintMeter(); }
      } catch (e) { /* the counter is only a hint */ }
    }, 250);
  });
  input.addEventListener('keydown', (e) => { if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) { e.preventDefault(); onSend(); } });

  function setBusy(b) {
    busy = b; input.disabled = b;
    [setBtn, saveBtn, clearBtn].forEach((x) => { x.disabled = b; });
    sendBtn.replaceChildren(b ? 'Stop' : 'Send');
    sendBtn.classList.toggle('primary', !b); sendBtn.classList.toggle('danger', b);
    sendBtn.setAttribute('aria-label', b ? 'Stop the reply' : 'Send message');
  }

  // Waits for one reply, showing it while it grows. `wrap` is the whole message, `bub` its bubble, `txt` the text node holder.
  async function follow(job, wrap, bub, txt) {
    let failed = false;
    while (mine === epoch) {
      let p;
      try { p = await api('/api/chat/poll?job=' + encodeURIComponent(job)); }
      catch (e) { toast(e.message, 'bad'); failed = true; break; }
      if (p.state === 'running') {
        if (p.partial) { bub.classList.remove('typing'); txt.textContent = p.partial; scrollDown(); }
        await sleep(160); continue;
      }
      bub.classList.remove('typing');
      if (p.tokens) tokens = p.tokens;
      if (p.mood !== undefined) paintMood(p.mood);
      if (p.state === 'error' || (p.state === 'stopped' && !p.reply)) {
        // the server put everything back as it was, so the page does too, and your message is given back to you
        const yours = wrap.previousSibling; wrap.remove(); yours?.remove(); input.value = lastText;
        if (p.state === 'error') { failed = true; toast(p.error || 'The AI could not answer.', 'bad'); }
      } else {
        txt.textContent = p.reply || p.partial || '';
        if (p.state === 'stopped') bub.append(h('div', { class: 'note' }, 'stopped'));
        (p.notes || []).forEach((n) => bub.append(h('div', { class: 'note' }, n)));
      }
      break;
    }
    stopping = false; draftTokens = 0;
    if (mine !== epoch) return;
    paintMeter(); setBusy(false); input.focus(); scrollDown();
    if (failed && !input.value) input.value = lastText;
  }

  async function onSend() {
    if (busy) {
      if (!stopping) { stopping = true; try { await api('/api/chat/stop', {}); } catch (e) { toast(e.message, 'bad'); } }
      return;
    }
    const text = input.value.trim();
    if (!text) return;
    lastText = text;
    log.querySelector('.chat-empty')?.remove();
    const mineMsg = bubble('user', text);
    const txt = h('span', null, h('span', { class: 'dots', role: 'status', 'aria-label': 'The AI is typing' }, h('i'), h('i'), h('i')));
    const bub = h('div', { class: 'bubble typing' }, txt);
    const wrap = h('div', { class: 'msg ai' }, h('div', { class: 'who' }, ai), bub);
    log.append(mineMsg, wrap); scrollDown();
    input.value = ''; input.style.height = 'auto'; draftTokens = 0; setBusy(true);
    try {
      const r = await api('/api/chat/send', { message: text });
      await follow(r.job, wrap, bub, txt);
    } catch (e) {
      toast(e.message, 'bad'); wrap.remove(); mineMsg.remove(); input.value = text; setBusy(false);
    }
  }
  sendBtn.addEventListener('click', onSend);

  // ---- settings drawer
  const tempN = h('input', { id: 's-temp', type: 'number', min: '0', max: '2', step: '0.05' });
  const tempR = h('input', { type: 'range', min: '0', max: '2', step: '0.05', 'aria-label': 'Temperature slider' });
  const topK = h('input', { id: 's-topk', type: 'number', min: '0', max: '1000', step: '1' });
  const maxN = h('input', { id: 's-max', type: 'number', min: '1', max: String(st.max_new_limit), step: '1' });
  tempR.addEventListener('input', () => { tempN.value = tempR.value; });
  tempN.addEventListener('input', () => { tempR.value = tempN.value; });
  const dlgErr = h('div', { class: 'callout bad', role: 'alert', hidden: true });
  const dlg = h('dialog', { class: 'dialog drawer', 'aria-labelledby': 'set-title' },
    h('div', { class: 'dialog-body stack' },
      h('h2', { id: 'set-title' }, 'Reply settings'),
      h('p', { class: 'small muted' }, 'They apply to the next reply, and stay until you load a model again.'),
      h('div', { class: 'field' }, h('label', { for: 's-temp' }, 'Temperature'), h('div', { class: 'row' }, tempR, tempN),
        h('span', { class: 'hint' }, 'Lower is steadier, higher is more varied. 0 always takes the most likely word. 0.3 to 0.5 suits full sentences.')),
      h('div', { class: 'field' }, h('label', { for: 's-topk' }, 'Top-k'), topK, h('span', { class: 'hint' }, 'Only the k most likely words are considered. 0 means no limit.')),
      h('div', { class: 'field' }, h('label', { for: 's-max' }, 'Longest reply (tokens)'), maxN,
        h('span', { class: 'hint' }, '1 to ' + st.max_new_limit + '. The rest of the window is kept free for the reply.')),
      dlgErr,
      h('div', { class: 'dialog-actions' },
        h('button', { class: 'btn', type: 'button', onclick: () => dlg.close() }, 'Cancel'),
        h('button', { class: 'btn primary', type: 'button', onclick: applySettings }, 'Apply'))));
  dlg.addEventListener('click', (e) => { if (e.target === dlg) dlg.close(); });
  function openSettings() {
    tempN.value = tempR.value = settings.temperature; topK.value = settings.top_k; maxN.value = settings.max_new;
    dlgErr.hidden = true; dlg.showModal(); tempN.focus();
  }
  async function applySettings() {
    const body = { temperature: Number(tempN.value), top_k: Number(topK.value), max_new: Number(maxN.value) };
    if ([body.temperature, body.top_k, body.max_new].some((v) => !Number.isFinite(v))) { dlgErr.replaceChildren(h('p', null, 'Fill in all three numbers.')); dlgErr.hidden = false; return; }
    try {
      const r = await api('/api/chat/settings', body);
      settings = r.settings; tokens = r.tokens; paintMeter(); dlg.close(); toast('Settings applied', 'ok');
    } catch (e) { dlgErr.replaceChildren(h('p', null, e.message)); dlgErr.hidden = false; }
  }

  // ---- save and clear
  async function onSave() {
    try { const r = await api('/api/chat/save', {}); toast('Saved to ' + r.saved, 'ok'); } catch (e) { toast(e.message, 'bad'); }
  }
  function disarm() { armed = 0; clearBtn.classList.remove('danger'); clearBtn.replaceChildren('Clear'); }
  async function onClear() {
    if (!armed) {                                                 // two steps: clearing cannot be undone
      armed = setTimeout(disarm, 4000); clearBtn.classList.add('danger'); clearBtn.replaceChildren('Really clear?'); return;
    }
    clearTimeout(armed); disarm();
    try {
      const r = await api('/api/chat/reset', {});
      log.replaceChildren(h('div', { class: 'chat-empty muted' }, 'New conversation. Say something to start.'));
      tokens = r.tokens; draftTokens = 0; input.value = ''; paintMeter(); paintMood(r.mood); warn.hidden = true; input.focus();
    } catch (e) { toast(e.message, 'bad'); }
  }
  const setBtn = h('button', { class: 'btn', type: 'button', onclick: openSettings }, 'Settings');
  const saveBtn = h('button', { class: 'btn', type: 'button', onclick: onSave, title: 'Save this conversation as a text file in saved_chats/' }, 'Save');
  const clearBtn = h('button', { class: 'btn', type: 'button', onclick: onClear, title: 'Start a new conversation (memory and mood files are kept)' }, 'Clear');

  const head = h('div', { class: 'chat-head' },
    h('div', { class: 'row' }, h('span', { class: 'badge accent' }, st.label), h('span', { class: 'badge' }, 'window ' + st.ctx), h('span', { class: 'badge' }, st.layers + ' layers'),
      st.name ? h('span', { class: 'badge' }, st.name) : null, moodBadge),
    h('div', { class: 'row' }, setBtn, saveBtn, clearBtn,
      h('button', { class: 'btn', type: 'button', onclick: () => { if (busy) { toast('Press Stop first.', 'bad'); return; } showLoader(main, mine, st); } }, 'Change model')));

  main.replaceChildren(h('div', { class: 'chat' }, head, log, warn, dlg,
    h('div', { class: 'composer card' }, input, h('div', { class: 'composer-foot' }, h('div', { class: 'row meter-row' }, meter, meterTxt, draftTxt), sendBtn))));
  paintMeter(); scrollDown();

  if (st.job) {                                                // the page was reloaded while a reply was being written: pick it up again
    const txt = h('span', null);
    const bub = h('div', { class: 'bubble typing' }, txt);
    const wrap = h('div', { class: 'msg ai' }, h('div', { class: 'who' }, ai), bub);
    log.append(wrap); setBusy(true); follow(st.job, wrap, bub, txt);
  } else input.focus({ preventScroll: true });
}
