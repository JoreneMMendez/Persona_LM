"""Stage 4 exports of the memory graph: an Obsidian vault (Markdown with [[wiki-links]]) and a one-file HTML graph you open in a browser.

Both contain PERSONAL TEXT. They are written on this computer only; nothing is uploaded. Both skip memories that were merged or
archived (they stay in memory.json), and mark dormant ones.
"""
import json
import os
import re


SHOWN = ("active", "dormant")
_BAD_FILE_CHARS = re.compile(r'[\\/:*?"<>|#^\[\]\x00-\x1f]')
_WINDOWS_RESERVED = {"con", "prn", "aux", "nul"} | {"com%d" % i for i in range(1, 10)} | {"lpt%d" % i for i in range(1, 10)}
FOLDERS = {"person": "People", "topic": "Topics", "event": "Events", "feeling": "Feelings"}


def safe_file_name(name, taken):
    """A file name Windows and Obsidian accept; `taken` (a set of lower-cased names) makes every name unique."""
    base = _BAD_FILE_CHARS.sub("-", name).strip(" .-") or "untitled"
    base = base[:80].rstrip(" .")
    if base.lower() in _WINDOWS_RESERVED:
        base += "_"
    cand, i = base, 2
    while cand.lower() in taken:
        cand = "%s (%d)" % (base, i)
        i += 1
    taken.add(cand.lower())
    return cand


def _link(file_name, shown):
    return "[[%s]]" % file_name if file_name == shown else "[[%s|%s]]" % (file_name, shown)


def _frontmatter(fields):
    lines = ["---"]
    for k, v in fields:
        lines.append("%s: %s" % (k, json.dumps(v, ensure_ascii=False)))
    lines.append("---")
    return "\n".join(lines)


def export_vault(store, out_dir, now=None):
    """Write the vault into `out_dir` (must not exist, or be empty: nothing is ever overwritten). Returns a small report dict."""
    if os.path.exists(out_dir) and (not os.path.isdir(out_dir) or os.listdir(out_dir)):
        raise ValueError("%s already exists and is not empty. Nothing was written. Choose a new folder name." % out_dir)
    now = store.clock() if now is None else now
    mems = [m for m in store.memories.values() if m["status"] in SHOWN]
    used_nodes = sorted({nid for m in mems for nid in m["nodes"]})
    scores = store.node_scores(now)
    taken = {"readme"}
    node_file = {nid: safe_file_name(store.nodes[nid]["name"], taken) for nid in used_nodes}
    mem_file = {m["id"]: safe_file_name(m["id"], taken) for m in mems}
    edges = store.edges(now)
    os.makedirs(out_dir, exist_ok=True)

    def write(rel, text):
        path = os.path.join(out_dir, *rel.split("/"))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)

    write("README.md", "# Memory vault\n\nExported from memory.json by persona_lm. Open this folder as a vault in Obsidian and use the graph view.\n\n"
                       "**This folder contains personal text. Do not upload it or put it in a public repository.**\n\n"
                       "It is a copy: edits here do not change memory.json. Memories marked `dormant` have faded; merged and archived memories are not exported.\n")
    for nid in used_nodes:
        node = store.nodes[nid]
        mine = sorted((m for m in mems if nid in m["nodes"]), key=lambda m: (m["timestamp"], m["id"]))
        linked = sorted(((e["count"], other) for (a, b), e in edges.items() if nid in (a, b) for other in ((b if a == nid else a),)
                         if other in node_file), key=lambda t: (-t[0], t[1]))
        body = [_frontmatter([("kind", node["kind"]), ("size", scores[nid]), ("memories", len(mine))]), "", "# " + node["name"], ""]
        if linked:
            body += ["## Linked to", ""] + ["- %s (%d shared memor%s)" % (_link(node_file[o], store.nodes[o]["name"]), c, "y" if c == 1 else "ies")
                                           for c, o in linked] + [""]
        body += ["## Memories", ""]
        for m in mine:
            body.append("- %s %s (%s)%s" % (m["timestamp"][:10], m["text"], _link(mem_file[m["id"]], m["id"]), " *(dormant)*" if m["status"] == "dormant" else ""))
        write("%s/%s.md" % (FOLDERS[node["kind"]], node_file[nid]), "\n".join(body) + "\n")
    for m in mems:
        text = m["text"]
        for nid in m["nodes"]:                      # turn the first mention of each node's name into a wiki-link
            if nid in node_file:
                name = store.nodes[nid]["name"]
                text = re.sub(r"(?<!\w)" + re.escape(name) + r"(?!\w)", lambda mo: _link(node_file[nid], mo.group(0)), text, count=1, flags=re.IGNORECASE)
        links = " ".join(_link(node_file[n], store.nodes[n]["name"]) for n in m["nodes"] if n in node_file)
        body = [_frontmatter([("timestamp", m["timestamp"]), ("source", m["source"]), ("emotion", m["emotion"]), ("importance", m["importance"]),
                              ("strength", round(store.strength(m, now), 3)), ("status", m["status"]), ("pinned", bool(m.get("pinned")))]),
                "", text, "", ("About: " + links) if links else ""]
        write("Memories/%s.md" % mem_file[m["id"]], "\n".join(body).rstrip() + "\n")
    return {"folder": out_dir, "memories": len(mems), "nodes": len(used_nodes)}


# ---------------------------------------------------------------------------------------------------------------------------------
GRAPH_LIMIT = 600           # more circles than this make the page slow and unreadable: show the biggest ones

LAYOUT_JS = r"""
/*LAYOUT_START*/
// One step of a force-directed layout. nodes: [{x,y,vx,vy,r}], edges: [{a,b,w}] (a,b are indexes), alpha: 1 -> 0 as it settles.
function stepLayout(nodes, edges, alpha, width, height) {
  var n = nodes.length, i, j, dx, dy, d2, d, f;
  for (i = 0; i < n; i++) {                         // every circle pushes the others away
    for (j = i + 1; j < n; j++) {
      dx = nodes[j].x - nodes[i].x; dy = nodes[j].y - nodes[i].y;
      d2 = dx * dx + dy * dy;
      if (d2 < 0.01) { dx = Math.random() - 0.5; dy = Math.random() - 0.5; d2 = dx * dx + dy * dy + 0.01; }
      d = Math.sqrt(d2);
      f = (900 + 6 * (nodes[i].r + nodes[j].r)) / d2 * alpha;
      nodes[i].vx -= dx / d * f; nodes[i].vy -= dy / d * f;
      nodes[j].vx += dx / d * f; nodes[j].vy += dy / d * f;
    }
  }
  for (i = 0; i < edges.length; i++) {              // a link pulls its two circles together (stronger link = shorter spring)
    var a = nodes[edges[i].a], b = nodes[edges[i].b];
    dx = b.x - a.x; dy = b.y - a.y; d = Math.sqrt(dx * dx + dy * dy) || 1;
    f = (d - (60 + a.r + b.r)) * 0.02 * (0.5 + edges[i].w) * alpha;
    a.vx += dx / d * f; a.vy += dy / d * f; b.vx -= dx / d * f; b.vy -= dy / d * f;
  }
  for (i = 0; i < n; i++) {                         // a gentle pull to the middle, friction, move
    nodes[i].vx += (width / 2 - nodes[i].x) * 0.004 * alpha; nodes[i].vy += (height / 2 - nodes[i].y) * 0.004 * alpha;
    if (!nodes[i].fixed) { nodes[i].vx *= 0.6; nodes[i].vy *= 0.6; nodes[i].x += nodes[i].vx; nodes[i].y += nodes[i].vy; }
    else { nodes[i].vx = 0; nodes[i].vy = 0; }
  }
}
/*LAYOUT_END*/
"""

PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Memory graph</title>
<style>
:root { --bg:#fbfbf9; --fg:#222; --muted:#666; --edge:#99a; --panel:#fff; --border:#ccc;
        --person:#2f7ed8; --topic:#2a9d6f; --event:#d98b1f; --feeling:#c4477a; }
@media (prefers-color-scheme: dark) { :root { --bg:#16181b; --fg:#e8e8e8; --muted:#9aa; --edge:#667; --panel:#20242a; --border:#444;
        --person:#5aa2ff; --topic:#4cc79a; --event:#f0b04e; --feeling:#ee79ab; } }
html,body { margin:0; height:100%; background:var(--bg); color:var(--fg); font:14px/1.4 system-ui, sans-serif; }
#wrap { position:relative; height:100%; }
canvas { display:block; width:100%; height:100%; cursor:grab; }
#panel { position:absolute; left:12px; top:12px; max-width:340px; background:var(--panel); border:1px solid var(--border); border-radius:8px; padding:10px 12px; }
#panel h1 { font-size:15px; margin:0 0 4px; } #panel p { margin:4px 0; color:var(--muted); font-size:12px; } #info { white-space:pre-wrap; margin-top:6px; }
.k { display:inline-block; width:10px; height:10px; border-radius:50%; margin:0 4px 0 8px; }
</style></head><body><div id="wrap"><canvas id="c"></canvas>
<div id="panel"><h1>Memory graph</h1>
<p><span class="k" style="background:var(--person);margin-left:0"></span>person<span class="k" style="background:var(--topic)"></span>topic<span class="k" style="background:var(--event)"></span>event<span class="k" style="background:var(--feeling)"></span>feeling</p>
<p>Bigger circle = more or stronger memories. Thicker line = more shared memories. Drag a circle to move it, drag the background to pan, scroll to zoom, hover to read.</p>
<p id="note"></p><div id="info">Hover over a circle.</div></div></div>
<script type="application/json" id="data">__DATA__</script>
<script>
__LAYOUT__
(function () {
  var data = JSON.parse(document.getElementById("data").textContent);
  var canvas = document.getElementById("c"), ctx = canvas.getContext("2d"), info = document.getElementById("info");
  document.getElementById("note").textContent = data.note || "";
  var W = 800, H = 600, dpr = window.devicePixelRatio || 1, view = { x: 0, y: 0, k: 1 };
  var maxScore = Math.max.apply(null, data.nodes.map(function (n) { return n.score; }).concat([0.0001]));
  var maxW = Math.max.apply(null, data.edges.map(function (e) { return e.w; }).concat([0.0001]));
  var nodes = data.nodes.map(function (n, i) {
    var a = i * 2.399963, rr = 40 * Math.sqrt(i + 1);
    return { id: n.id, label: n.label, kind: n.kind, texts: n.texts, extra: n.extra, score: n.score, r: 7 + 22 * Math.sqrt(n.score / maxScore),
             x: 400 + rr * Math.cos(a), y: 300 + rr * Math.sin(a), vx: 0, vy: 0, fixed: false };
  });
  var edges = data.edges.map(function (e) { return { a: e.a, b: e.b, w: e.w / maxW, count: e.count }; });
  var placed = false;
  function resize() { var r = canvas.getBoundingClientRect(); W = r.width; H = r.height; canvas.width = W * dpr; canvas.height = H * dpr;
    if (!placed) { view.x = W / 2 - 400; view.y = H / 2 - 300; placed = true; } draw(); }
  function css(v) { return getComputedStyle(document.documentElement).getPropertyValue(v).trim(); }
  function draw() {
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, W, H);
    ctx.translate(view.x, view.y); ctx.scale(view.k, view.k);
    ctx.strokeStyle = css("--edge");
    edges.forEach(function (e) { ctx.globalAlpha = 0.35 + 0.5 * e.w; ctx.lineWidth = 1 + 3 * Math.sqrt(e.w);
      ctx.beginPath(); ctx.moveTo(nodes[e.a].x, nodes[e.a].y); ctx.lineTo(nodes[e.b].x, nodes[e.b].y); ctx.stroke(); });
    ctx.globalAlpha = 1; ctx.font = "12px system-ui, sans-serif"; ctx.textAlign = "center";
    nodes.forEach(function (n) { ctx.fillStyle = css("--" + n.kind) || "#888"; ctx.beginPath(); ctx.arc(n.x, n.y, n.r, 0, 6.2832); ctx.fill();
      ctx.fillStyle = css("--fg"); ctx.fillText(n.label, n.x, n.y + n.r + 13); });
  }
  var alpha = 1, running = false;
  function tick() { if (alpha > 0.02) { stepLayout(nodes, edges, alpha, 800, 600); alpha *= 0.985; draw(); requestAnimationFrame(tick); } else { running = false; } }
  function start() { if (!running) { running = true; tick(); } }
  function toWorld(ev) { var r = canvas.getBoundingClientRect(); return { x: (ev.clientX - r.left - view.x) / view.k, y: (ev.clientY - r.top - view.y) / view.k }; }
  function pick(p) { for (var i = nodes.length - 1; i >= 0; i--) { var dx = p.x - nodes[i].x, dy = p.y - nodes[i].y; if (dx * dx + dy * dy <= nodes[i].r * nodes[i].r) return nodes[i]; } return null; }
  var drag = null, pan = null;
  canvas.addEventListener("mousedown", function (ev) { var p = toWorld(ev), n = pick(p);
    if (n) { drag = n; n.fixed = true; alpha = Math.max(alpha, 0.3); start(); } else { pan = { sx: ev.clientX, sy: ev.clientY, vx: view.x, vy: view.y }; } });
  window.addEventListener("mouseup", function () { if (drag) { drag.fixed = false; } drag = null; pan = null; });
  canvas.addEventListener("mousemove", function (ev) {
    var p = toWorld(ev);
    if (drag) { drag.x = p.x; drag.y = p.y; draw(); return; }
    if (pan) { view.x = pan.vx + ev.clientX - pan.sx; view.y = pan.vy + ev.clientY - pan.sy; draw(); return; }
    var n = pick(p);
    info.textContent = n ? n.label + " (" + n.kind + ")\n" + n.extra + "\n\n" + n.texts.map(function (t) { return "- " + t; }).join("\n") : "Hover over a circle.";
  });
  canvas.addEventListener("wheel", function (ev) { ev.preventDefault(); var f = ev.deltaY < 0 ? 1.1 : 1 / 1.1, r = canvas.getBoundingClientRect(),
    mx = ev.clientX - r.left, my = ev.clientY - r.top; view.x = mx - (mx - view.x) * f; view.y = my - (my - view.y) * f; view.k *= f; draw(); }, { passive: false });
  window.addEventListener("resize", resize);
  resize(); start();
  window.__graph = { nodes: nodes, edges: edges };
})();
</script></body></html>
"""


def graph_data(store, now=None, limit=GRAPH_LIMIT):
    now = store.clock() if now is None else now
    mems = [m for m in store.memories.values() if m["status"] in SHOWN]
    used = sorted({nid for m in mems for nid in m["nodes"]})
    scores = store.node_scores(now)
    ranked = sorted(used, key=lambda nid: (-scores[nid], nid))
    kept = ranked[:limit]
    index = {nid: i for i, nid in enumerate(kept)}
    nodes = []
    for nid in kept:
        mine = sorted((m for m in mems if nid in m["nodes"]), key=lambda m: (-store.strength(m, now), m["id"]))
        nodes.append({"id": nid, "label": store.nodes[nid]["name"], "kind": store.nodes[nid]["kind"], "score": scores[nid],
                      "extra": "%d memor%s" % (len(mine), "y" if len(mine) == 1 else "ies"),
                      "texts": [(m["text"] if len(m["text"]) <= 220 else m["text"][:219] + "…") for m in mine[:6]]})
    edges = [{"a": index[a], "b": index[b], "w": round(e["weight"], 4), "count": e["count"]}
             for (a, b), e in sorted(store.edges(now).items()) if a in index and b in index]
    note = "" if len(used) <= limit else "Showing the %d biggest of %d nodes." % (limit, len(used))
    return {"nodes": nodes, "edges": edges, "note": note}


def export_graph(store, out_path, now=None, limit=GRAPH_LIMIT):
    """One self-contained HTML file: no network, no outside scripts. Refuses to overwrite."""
    if os.path.exists(out_path):
        raise ValueError("%s already exists. Nothing was written. Choose another name." % out_path)
    data = graph_data(store, now, limit)
    blob = json.dumps(data, ensure_ascii=False).replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    html = PAGE.replace("__LAYOUT__", LAYOUT_JS).replace("__DATA__", blob)
    folder = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(folder, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(html)
    return {"file": out_path, "nodes": len(data["nodes"]), "edges": len(data["edges"])}
