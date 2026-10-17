"""`reflex ui`: a local web dashboard over runs/ — spec, metrics, label gallery, student-vs-teacher
eval frames, replay, stress. Pages are static HTML (runs/index.html, runs/<run>/view.html) that
read images relative to the run folder, so they also work when opened straight from disk."""

from __future__ import annotations

import csv
import functools
import json
import threading
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .utils import console, read_json, read_jsonl, runs_dir

STAGES = ["ingest", "spec", "spec_check", "label", "train", "eval", "stress", "export"]


def _results(run: Path) -> dict:
    return {s: read_json(run / "results" / f"{s}.json", default={}) for s in STAGES}


def _run_data(run: Path) -> dict:
    frames = []
    if (run / "frames.csv").exists():
        with open(run / "frames.csv") as f:
            frames = list(csv.DictReader(f))
    labels = {r["frame_id"]: r for r in read_jsonl(run / "labels_final.jsonl")}
    split = read_json(run / "split.json", default={})
    which = {i: k for k in ("train", "test", "gap") for i in split.get(k, [])}
    gallery = []
    for f in frames:
        r = labels.get(f["frame_id"])
        gallery.append({"id": f["frame_id"], "path": f["path"], "ep": f["episode"], "t": float(f["t_sec"]),
                        "label": r["top"] if r else None, "conf": round(r["conf"], 3) if r else None,
                        "source": r.get("source") if r else None, "split": which.get(f["frame_id"])})
    spec = read_json(run / "spec.json", default={})
    return {
        "name": run.name, "spec": spec, "results": _results(run),
        "report": read_json(run / "report.json", default={}),
        "train": read_json(run / "train.json", default={}),
        "labels": read_json(run / "label_summary.json", default={}),
        "stress": read_json(run / "stress_report.json", default={}),
        "predictions": read_jsonl(run / "predictions.jsonl"),
        "frames": gallery,
        "images": {k: (run / k).exists() for k in ["confusion.png", "disagreements.png", "stress.png",
                                                    "stress_examples.png", "spec_check.png"]},
        "fps": read_json(run / "results" / "ingest.json", default={}).get("fps") or 5,
    }


def build_run_page(run: Path) -> Path:
    data = json.dumps(_run_data(run)).replace("</", "<\\/")
    out = run / "view.html"
    out.write_text(RUN_HTML.replace("__DATA__", data).replace("__TITLE__", f"Reflex · {run.name}"))
    return out


def build_index(root: Path) -> Path:
    runs = []
    for d in sorted(p for p in root.iterdir() if p.is_dir() and (p / "frames.csv").exists()):
        build_run_page(d)
        res = _results(d)
        m = read_json(d / "report.json", default={}).get("metrics", {})
        spec = read_json(d / "spec.json", default={})
        runs.append({"name": d.name, "query": spec.get("query", ""), "stages": {s: res[s].get("status", "todo") for s in STAGES},
                     "agreement": m.get("agreement"), "speedup": m.get("speedup"), "p50_ms": m.get("p50_ms"),
                     "size_mb": m.get("size_mb"), "n_frames": res["ingest"].get("n_frames"),
                     "mtime": max((p.stat().st_mtime for p in d.iterdir()), default=0)})
    runs.sort(key=lambda r: -r["mtime"])
    out = root / "index.html"
    out.write_text(INDEX_HTML.replace("__DATA__", json.dumps(runs).replace("</", "<\\/")))
    return out


def serve(port: int = 8765, open_browser: bool = True, run: str | None = None) -> dict:
    root = runs_dir()
    if not root.exists():
        root.mkdir(parents=True)
    index = build_index(root)
    url = f"http://localhost:{port}/" + (f"{run}/view.html" if run else "index.html")
    handler = functools.partial(_QuietHandler, directory=str(root))
    httpd = ThreadingHTTPServer(("127.0.0.1", port), handler)
    console.print(f"[bold green]Reflex UI[/bold green] → {url}  [dim](Ctrl+C to stop; pages rebuild on reload of the index)[/dim]")
    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    return {"url": url, "index": str(index)}


class _QuietHandler(SimpleHTTPRequestHandler):
    def do_GET(self):  # rebuild pages on each index/view load so new results show up
        root = Path(self.directory)
        try:
            if self.path in ("/", "/index.html"):
                build_index(root)
            elif self.path.endswith("/view.html"):
                build_run_page(root / self.path.strip("/").split("/")[0])
        except Exception as e:  # noqa: BLE001 — never kill the server over a half-written run
            console.print(f"[yellow]ui rebuild failed: {e}[/yellow]")
        super().do_GET()

    def log_message(self, *a):
        pass


# ---------------------------------------------------------------- templates

CSS = """
:root{--bg:#f7f7f5;--panel:#fff;--ink:#1d1d1b;--muted:#6b6b66;--line:#e4e3de;--ok:#1f9d55;--bad:#d64545;--accent:#3b5bdb;--chip:#efeee9}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#141413;--panel:#1d1d1b;--ink:#ecebe6;--muted:#9b9a93;--line:#2e2d2a;--ok:#4cc38a;--bad:#ff6b6b;--accent:#7c96ff;--chip:#2a2926}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 -apple-system,BlinkMacSystemFont,Inter,Segoe UI,sans-serif}
header{padding:18px 24px;border-bottom:1px solid var(--line);display:flex;gap:16px;align-items:baseline;flex-wrap:wrap}
header h1{margin:0;font-size:20px}header a{color:var(--muted);text-decoration:none}
main{padding:20px 24px;max-width:1400px;margin:0 auto}
.muted{color:var(--muted)}.chip{display:inline-block;background:var(--chip);border-radius:999px;padding:2px 10px;margin:2px 4px 2px 0;font-size:12px}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:16px;margin-bottom:16px}
.tiles{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:12px;margin-bottom:16px}
.tile{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:12px 14px}.tile b{display:block;font-size:22px}.tile span{color:var(--muted);font-size:12px}
.tabs{display:flex;gap:4px;margin-bottom:16px;flex-wrap:wrap}.tabs button{border:1px solid var(--line);background:var(--panel);color:var(--ink);padding:6px 14px;border-radius:999px;cursor:pointer}
.tabs button.on{background:var(--ink);color:var(--bg)}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:10px}
.card{background:var(--panel);border:2px solid var(--line);border-radius:10px;overflow:hidden;font-size:12px}
.card img{width:100%;display:block;aspect-ratio:4/3;object-fit:cover;background:#000}.card div{padding:6px 8px}
.card.ok{border-color:var(--ok)}.card.bad{border-color:var(--bad)}
.filters{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:12px;align-items:center}
select,input{background:var(--panel);color:var(--ink);border:1px solid var(--line);border-radius:8px;padding:5px 8px}
img.fig{max-width:100%;border-radius:8px;border:1px solid var(--line);background:#fff}
table{border-collapse:collapse;width:100%}td,th{text-align:left;padding:6px 8px;border-bottom:1px solid var(--line)}
.bar{height:10px;background:var(--accent);border-radius:5px}.two{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:16px}
.stage{font-size:11px;padding:2px 8px;border-radius:999px;background:var(--chip);margin-right:4px}.stage.ok,.stage.ready{background:var(--ok);color:#fff}.stage.needs_input{background:#e8a33d;color:#fff}.stage.error{background:var(--bad);color:#fff}
#player img{max-width:100%;border-radius:8px}
@media (max-width:600px){header,main{padding:12px 16px}.grid{grid-template-columns:repeat(2,1fr)}}
"""

RUN_HTML = """<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title><style>""" + CSS + """</style></head><body>
<header><a href="../index.html">← all runs</a><h1 id="title"></h1><span class="muted" id="query"></span></header>
<main><div class="tabs" id="tabs"></div><div id="view"></div></main>
<script>
const D = __DATA__;
const $ = (h) => { const t = document.createElement('template'); t.innerHTML = h.trim(); return t.content.firstChild; };
const esc = (s) => String(s ?? '').replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const pct = (x) => x == null ? '—' : (100 * x).toFixed(1) + '%';
const num = (x, d = 1) => x == null ? '—' : Number(x).toFixed(d);
document.getElementById('title').textContent = 'Reflex · ' + D.name;
document.getElementById('query').textContent = D.spec.query ? '“' + D.spec.query + '”' : '';
const classes = D.spec.question ? (D.spec.question.type === 'predicate' ? ['no', 'yes'] : D.spec.question.choices.map(c => c.value)) : [];

function overview() {
  const m = (D.report.metrics || {}), s = D.spec;
  const stages = Object.entries(D.results).map(([k, v]) => `<span class="stage ${esc(v.status || 'todo')}">${k}: ${esc(v.status || 'todo')}</span>`).join('');
  const dist = D.labels.final_distribution || {};
  const tot = Object.values(dist).reduce((a, b) => a + b, 0) || 1;
  return `
  <div class="panel">${stages}</div>
  <div class="tiles">
    <div class="tile"><b>${pct(m.agreement)}</b><span>agreement with teacher</span></div>
    <div class="tile"><b>${num(m.macro_f1, 2)}</b><span>macro-F1</span></div>
    <div class="tile"><b>${num(m.p50_ms)} ms</b><span>student p50 (CPU)</span></div>
    <div class="tile"><b>${num(m.teacher_p50_ms, 0)} ms</b><span>teacher p50 (cloud)</span></div>
    <div class="tile"><b>${m.speedup ? m.speedup + '×' : '—'}</b><span>faster</span></div>
    <div class="tile"><b>${num(m.size_mb)} MB</b><span>model size</span></div>
    <div class="tile"><b>$${num(m.teacher_usd_per_1000, 2)}</b><span>teacher $/1k frames · student $0</span></div>
    <div class="tile"><b>${m.n_test ?? '—'}</b><span>held-out frames (${esc(m.split_method || '—')} split)</span></div>
  </div>
  <div class="two">
    <div class="panel"><h3>Spec</h3><p>${esc(s.task_summary || 'no spec yet')}</p>
      ${(s.question && s.question.choices || []).map(c => `<div><span class="chip">${esc(c.value)}</span><span class="muted">${esc(c.description)}</span></div>`).join('')}
      ${(s.visual_cues || []).length ? '<h4>Visual cues</h4><ul>' + s.visual_cues.map(c => `<li>${esc(c)}</li>`).join('') + '</ul>' : ''}
      ${(s.edge_cases || []).length ? '<h4>Edge cases</h4><ul>' + s.edge_cases.map(c => `<li>${esc(c)}</li>`).join('') + '</ul>' : ''}
    </div>
    <div class="panel"><h3>Labels</h3>
      <p class="muted">${D.labels.labelled ?? 0} labelled · ${D.labels.propagated ?? 0} propagated · ${D.labels.low_conf ?? 0} low-confidence · teacher p50 ${num(D.labels.teacher_p50_ms, 0)} ms</p>
      <table>${Object.entries(dist).map(([k, v]) => `<tr><td style="width:30%">${esc(k)}</td><td><div class="bar" style="width:${100 * v / tot}%"></div></td><td style="width:15%">${v}</td></tr>`).join('')}</table>
      ${D.train.backbone ? `<p class="muted">backbone ${esc(D.train.backbone)} · ${D.train.n_train} train / ${D.train.n_test} test / ${D.train.n_gap} gap frames · trained in ${D.train.total_s}s</p>` : ''}
    </div>
  </div>
  <div class="two">
    ${D.images['confusion.png'] ? '<div class="panel"><h3>Confusion (test split)</h3><img class="fig" src="confusion.png"></div>' : ''}
    ${D.report.per_class ? '<div class="panel"><h3>Per class</h3><table><tr><th>class</th><th>precision</th><th>recall</th><th>support</th></tr>' + Object.entries(D.report.per_class).map(([k, v]) => `<tr><td>${esc(k)}</td><td>${num(v.precision, 2)}</td><td>${num(v.recall, 2)}</td><td>${v.support}</td></tr>`).join('') + '</table></div>' : ''}
  </div>`;
}

function cards(items, render) {
  const grid = $('<div class="grid"></div>');
  items.slice(0, 600).forEach(x => grid.appendChild($(render(x))));
  return grid;
}

function filterBar(fields, onChange) {
  const bar = $('<div class="filters"></div>');
  fields.forEach(([key, label, opts]) => {
    const sel = $(`<select data-k="${key}"><option value="">${label}: all</option>${opts.map(o => `<option>${esc(o)}</option>`).join('')}</select>`);
    sel.onchange = onChange; bar.appendChild(sel);
  });
  bar.appendChild($('<span class="muted" id="count"></span>'));
  return bar;
}

function labelsTab(root) {
  const bar = filterBar([['label', 'class', classes], ['split', 'split', ['train', 'test', 'gap']], ['source', 'source', ['teacher', 'propagated']], ['conf', 'confidence', ['low', 'high']]], draw);
  root.append(bar); const holder = $('<div></div>'); root.append(holder);
  function draw() {
    const f = Object.fromEntries([...bar.querySelectorAll('select')].map(s => [s.dataset.k, s.value]));
    const minc = D.spec.min_conf ?? 0.6;
    const items = D.frames.filter(x => (!f.label || x.label === f.label) && (!f.split || x.split === f.split) && (!f.source || x.source === f.source)
      && (!f.conf || (f.conf === 'low' ? x.conf < minc : x.conf >= minc)));
    bar.querySelector('#count').textContent = items.length + ' frames' + (items.length > 600 ? ' (first 600 shown)' : '');
    holder.replaceChildren(cards(items, x => `<div class="card"><img loading="lazy" src="${esc(x.path)}"><div><b>${esc(x.label ?? 'unlabelled')}</b> ${x.conf != null ? num(x.conf, 2) : ''}
      <div class="muted">${esc(x.ep)} · ${num(x.t)}s · ${esc(x.split || '')} ${x.source === 'propagated' ? '· propagated' : ''}</div></div></div>`));
  }
  draw();
}

function evalTab(root) {
  if (!D.predictions.length) { root.append($('<p class="muted">No eval yet — run <code>reflex eval</code>.</p>')); return; }
  const P = D.predictions;
  const episodes = [...new Set(P.map(p => p.episode))];
  const player = $(`<div class="panel" id="player"><h3>Replay a held-out episode <select id="ep">${episodes.map(e => `<option>${esc(e)}</option>`).join('')}</select> <button id="play">⏸</button></h3><img id="pimg"><div id="pcap" class="muted"></div></div>`);
  root.append(player);
  let i = 0, playing = true;
  const frames = () => P.filter(p => p.episode === player.querySelector('#ep').value).sort((a, b) => a.t_sec - b.t_sec);
  player.querySelector('#play').onclick = (e) => { playing = !playing; e.target.textContent = playing ? '⏸' : '▶'; };
  player.querySelector('#ep').onchange = () => { i = 0; };
  window._replay = setInterval(() => {
    if (!playing) return; const fs = frames(); if (!fs.length) return; const p = fs[i++ % fs.length];
    player.querySelector('#pimg').src = 'eval_frames/' + p.frame_id + '.jpg';
    player.querySelector('#pcap').textContent = `${p.t_sec.toFixed(1)}s · Reflex ${p.student} (${p.ms.toFixed(1)} ms) · Teacher ${p.teacher}`;
  }, 1000 / Math.min(D.fps, 10));

  const bar = filterBar([['agree', 'agreement', ['disagreements', 'agreements']], ['teacher', 'teacher', classes], ['student', 'student', classes]], draw);
  root.append(bar); const holder = $('<div></div>'); root.append(holder);
  function draw() {
    const f = Object.fromEntries([...bar.querySelectorAll('select')].map(s => [s.dataset.k, s.value]));
    const items = P.filter(p => (!f.agree || (f.agree === 'disagreements') === (p.student !== p.teacher)) && (!f.teacher || p.teacher === f.teacher) && (!f.student || p.student === f.student))
      .sort((a, b) => (b.student !== b.teacher) - (a.student !== a.teacher) || b.student_conf * b.teacher_conf - a.student_conf * a.teacher_conf);
    const nd = P.filter(p => p.student !== p.teacher).length;
    bar.querySelector('#count').textContent = `${items.length} shown · ${nd}/${P.length} disagreements`;
    holder.replaceChildren(cards(items, p => `<div class="card ${p.student === p.teacher ? 'ok' : 'bad'}"><img loading="lazy" src="${esc(p.path)}"><div>
      <b>Reflex: ${esc(p.student)}</b> ${num(p.student_conf, 2)} · ${num(p.ms)} ms<br>Teacher: ${esc(p.teacher)} ${num(p.teacher_conf, 2)}
      <div class="muted">${esc(p.episode)} · ${num(p.t_sec)}s</div></div></div>`));
  }
  draw();
}

function stressTab(root) {
  const S = D.stress;
  if (!S.results) { root.append($('<p class="muted">No stress test yet — run <code>reflex stress --run ' + esc(D.name) + ' --default</code>.</p>')); return; }
  root.append($(`<div class="panel"><p>Clean agreement <b>${pct(S.clean_agreement)}</b> on ${S.n_frames} test frames · plan: ${esc(S.plan_source)}</p>
    <table><tr><th>perturbation</th><th>params</th><th>why</th><th>student</th><th>drop</th><th>teacher</th></tr>
    ${S.results.map(r => `<tr><td>${esc(r.kind)}</td><td class="muted">${esc(JSON.stringify(r.params))}</td><td>${esc(r.why)}</td><td>${pct(r.student_agreement)}</td><td>${pct(r.student_drop)}</td><td>${pct(r.teacher_agreement)}</td></tr>`).join('')}</table>
    ${(S.suggestions || []).length ? '<h4>Suggestions</h4><ul>' + S.suggestions.map(s => `<li>${esc(s)}</li>`).join('') + '</ul>' : ''}</div>`));
  root.append($(`<div class="two">${D.images['stress.png'] ? '<div class="panel"><img class="fig" src="stress.png"></div>' : ''}${D.images['stress_examples.png'] ? '<div class="panel"><img class="fig" src="stress_examples.png"></div>' : ''}</div>`));
}

const TABS = { Overview: (r) => r.append($('<div>' + overview() + '</div>')), Labels: labelsTab, Eval: evalTab, Stress: stressTab };
const tabs = document.getElementById('tabs'), view = document.getElementById('view');
function show(name) {
  location.hash = name; tabs.querySelectorAll('button').forEach(b => b.classList.toggle('on', b.textContent === name));
  clearInterval(window._replay);
  view.replaceChildren(); TABS[name](view);
}
Object.keys(TABS).forEach(n => { const b = $(`<button>${n}</button>`); b.onclick = () => show(n); tabs.append(b); });
show(TABS[location.hash.slice(1)] ? location.hash.slice(1) : 'Overview');
</script></body></html>"""

INDEX_HTML = """<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Reflex runs</title><style>""" + CSS + """</style></head><body>
<header><h1>Reflex runs</h1><span class="muted">Big models teach. Small models react.</span></header>
<main><div id="runs" class="two"></div></main>
<script>
const R = __DATA__;
const esc = (s) => String(s ?? '').replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const el = document.getElementById('runs');
el.innerHTML = R.length ? R.map(r => `<a href="${encodeURIComponent(r.name)}/view.html" style="color:inherit;text-decoration:none"><div class="panel">
  <h3 style="margin:0 0 4px">${esc(r.name)}</h3><div class="muted">${esc(r.query)}</div>
  <p><b>${r.agreement != null ? (100 * r.agreement).toFixed(1) + '%' : '—'}</b> agreement · <b>${r.speedup ?? '—'}×</b> faster · ${r.p50_ms ?? '—'} ms · ${r.size_mb ?? '—'} MB · ${r.n_frames ?? '?'} frames</p>
  <div>${Object.entries(r.stages).map(([k, v]) => `<span class="stage ${esc(v)}">${k}</span>`).join('')}</div></div></a>`).join('')
  : '<p class="muted">No runs yet — start with <code>reflex ingest &lt;data_dir&gt; --run &lt;name&gt;</code>.</p>';
</script></body></html>"""
