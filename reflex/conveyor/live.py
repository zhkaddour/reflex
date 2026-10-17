"""`reflex conveyor live`: identical belts side by side, one brain each, in the browser.

One renderer thread draws every belt at STREAM_FPS and shares the JPEGs with all viewers (MJPEG,
<img src="/stream/i">), so extra browser tabs cost almost nothing. Stats: /api/state. Controls:
POST /api/control. Startup runs a preflight (sprites, skill, key, CPU load) and warms every brain up
with one call so the first decision on stage doesn't pay connection setup."""

from __future__ import annotations

import json
import os
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2

from ..utils import console, price_usd, read_json
from .brains import Brain
from .engine import Runner
from .sim import PUSH_D, PX_PER_M, ZONE, Belt, Clock

STREAM_FPS = 20


def training_summary(run: Path) -> dict | None:
    """How the student was made, from the run's stage results (for the page footer)."""
    res = run / "results"
    lab, tr, ev = (read_json(res / f"{s}.json", default={}) for s in ("label", "train", "eval"))
    if not lab or not tr:
        return None
    n = lab.get("labelled", 0)
    return {"frames": n, "usd": round(price_usd(lab.get("input_tokens_per_frame", 0) * n,
                                                 lab.get("output_tokens_per_frame", 0) * n), 3),
            "train_min": round(tr.get("total_s", 0) / 60, 1), "agreement": ev.get("agreement"),
            "backbone": tr.get("backbone")}


class Live:
    def __init__(self, brains: list[Brain], sprites: dict, speed_mps: float = 0.3, seed: int = 7,
                 cam_fps: float = 30.0, run: Path | None = None):
        cv2.setNumThreads(2)  # keep OpenCV's pool from flooding the cores the brains need
        self.belts = [Belt(sprites, seed=seed, speed=speed_mps * PX_PER_M) for _ in brains]
        self.brains = brains
        self.training = training_summary(run) if run else None
        self.warm: dict[int, str] = {}
        self.jpegs: list[bytes] = [b""] * len(brains)
        self.seq = 0
        self.viewers = 0
        self.new_frame = threading.Condition()
        self.clock = Clock(self.belts)
        self.runners = [Runner(b, br, cam_fps) for b, br in zip(self.belts, brains)]
        self._stop = threading.Event()

    # ------------------------------------------------------------ startup

    def preflight(self):
        load = os.getloadavg()[0] / (os.cpu_count() or 1)
        if load > 1.5:
            console.print(f"[yellow]⚠ CPU load is {load:.1f}x the core count — close other heavy jobs "
                          f"(ingest/train) before the demo, latencies will be inflated[/yellow]")
        for i, (b, br) in enumerate(zip(self.belts, self.brains)):
            for _ in range(int(2.5 * 60)):        # fill the belt so the camera sees something real
                b.step(1 / 60)
            snap = b.capture(in_flight=False)
            t0 = time.perf_counter()
            try:
                br.predict(snap["frame"], snap["gt"])
                self.warm[i] = f"ok {(time.perf_counter() - t0) * 1000:.0f} ms"
                console.print(f"  [green]✓[/green] {br.title:16s} warm-up {self.warm[i][3:]}  [dim]{br.detail}[/dim]")
            except Exception as e:  # noqa: BLE001 — show it on the lane, keep the others
                self.warm[i] = f"error: {e}"
                console.print(f"  [red]✗[/red] {br.title:16s} {e}")
            b.reset()

    def start(self):
        self.preflight()
        self.clock.start()
        for r in self.runners:
            r.start()
        threading.Thread(target=self._render_loop, daemon=True).start()
        return self

    def _render_loop(self):
        while not self._stop.is_set():
            t0 = time.perf_counter()
            if self.viewers:
                jpegs = [cv2.imencode(".jpg", b.render(), [cv2.IMWRITE_JPEG_QUALITY, 78])[1].tobytes()
                         for b in self.belts]
                with self.new_frame:
                    self.jpegs, self.seq = jpegs, self.seq + 1
                    self.new_frame.notify_all()
            time.sleep(max(0.0, 1 / STREAM_FPS - (time.perf_counter() - t0)))

    # ------------------------------------------------------------ API

    def state(self) -> dict:
        speed = self.belts[0].speed
        lanes = []
        for i, (b, br, r) in enumerate(zip(self.belts, self.brains, self.runners)):
            last = r.last
            lanes.append({
                "name": br.name, "title": br.title, "detail": br.detail, "price_note": br.price_note,
                "where": br.where, "workers": br.workers, "votes": b.votes_needed(), "warm": self.warm.get(i, ""),
                "stats": b.stats.as_dict(b.t),
                "last": {"push": bool(last["push"]), "p": round(float(last["p"]), 3), "ms": round(last["ms"], 1),
                         "truth": last["gt"]["target_in_zone"]} if last else None})
        return {"speed_mps": round(speed / PX_PER_M, 3), "paused": self.clock.paused, "t": round(self.belts[0].t, 1),
                "deadline_s": round(PUSH_D / max(speed, 1e-6), 2),
                "look_s": round((ZONE[1] - ZONE[0]) / max(speed, 1e-6), 2),
                "pusher_m": PUSH_D / PX_PER_M, "cpu_load": round(os.getloadavg()[0] / (os.cpu_count() or 1), 2),
                "training": self.training, "lanes": lanes}

    def control(self, msg: dict):
        if "speed_mps" in msg:
            v = min(max(float(msg["speed_mps"]), 0.05), 4.0)
            for b in self.belts:
                b.speed = v * PX_PER_M
        if "paused" in msg:
            self.clock.paused = bool(msg["paused"])
            for r in self.runners:
                r.paused = self.clock.paused
        if msg.get("reset"):
            for b in self.belts:
                b.reset()

    def cam_jpeg(self, i: int) -> bytes | None:
        last = self.runners[i].last
        return cv2.imencode(".jpg", last["frame"], [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes() if last else None

    def stop(self):
        self._stop.set()
        for r in self.runners:
            r.stop()
        self.clock.stop()


def serve(live: Live, port: int = 8766, open_browser: bool = True):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, body: bytes, ctype: str, code: int = 200):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = self.path.split("?")[0]
            if path == "/":
                self._send(PAGE.encode(), "text/html; charset=utf-8")
            elif path == "/api/state":
                self._send(json.dumps(live.state()).encode(), "application/json")
            elif path.startswith("/cam/"):
                img = live.cam_jpeg(int(path.rsplit("/", 1)[1]))
                self._send(img, "image/jpeg") if img else self._send(b"", "text/plain", 204)
            elif path.startswith("/stream/"):
                self._stream(int(path.rsplit("/", 1)[1]))
            else:
                self._send(b"not found", "text/plain", 404)

        def _stream(self, i: int):
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            live.viewers += 1
            seen = -1
            try:
                while True:
                    with live.new_frame:
                        live.new_frame.wait_for(lambda: live.seq != seen, timeout=1.0)
                        seen, jpg = live.seq, live.jpegs[i]
                    if jpg:
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                         + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                live.viewers -= 1

        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0)
            live.control(json.loads(self.rfile.read(n) or b"{}"))
            self._send(json.dumps(live.state()).encode(), "application/json")

    console.print("[bold]preflight[/bold]")
    live.start()
    httpd = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    httpd.daemon_threads = True
    url = f"http://localhost:{port}/"
    console.print(f"[bold green]Reflex conveyor[/bold green] → {url}  [dim](Ctrl+C to stop · keys: space r 1-4 g f t)[/dim]")
    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        live.stop()


PAGE = r"""<!doctype html><html data-theme="dark"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Reflex Conveyor</title>
<style>
:root{--bg:#f4f3ef;--panel:#fff;--ink:#1b1b19;--muted:#6b6a64;--line:#e2e0da;--chip:#ecebe6;--ok:#16884a;--bad:#d23c3c;--warn:#b86e00;--accent:#3557d6;--track:#e2e0da}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#0f0f0e;--panel:#191917;--ink:#f0efe9;--muted:#9a998f;--line:#2b2a27;--chip:#24231f;--ok:#3ccf7f;--bad:#ff6363;--warn:#f0a73a;--accent:#7f98ff;--track:#2f2e2a}}
:root[data-theme=dark]{--bg:#0f0f0e;--panel:#191917;--ink:#f0efe9;--muted:#9a998f;--line:#2b2a27;--chip:#24231f;--ok:#3ccf7f;--bad:#ff6363;--warn:#f0a73a;--accent:#7f98ff;--track:#2f2e2a}
*{box-sizing:border-box}html,body{margin:0}body{background:var(--bg);color:var(--ink);font:15px/1.4 -apple-system,BlinkMacSystemFont,Inter,"Segoe UI",sans-serif;-webkit-font-smoothing:antialiased}
.num{font-variant-numeric:tabular-nums}
header{display:grid;grid-template-columns:minmax(0,1fr) auto;gap:24px;align-items:end;padding:14px 28px 12px;border-bottom:1px solid var(--line)}
.kicker{font-size:12px;letter-spacing:.14em;text-transform:uppercase;color:var(--muted);font-weight:600}
h1{margin:4px 0 6px;font-size:40px;line-height:1.05;letter-spacing:-.02em}
.lede{margin:0;color:var(--muted);font-size:16px;max-width:900px}
.speed{display:grid;gap:8px;justify-items:end}
.speed .big{font-size:44px;font-weight:700;line-height:1;letter-spacing:-.02em}.speed .big small{font-size:18px;color:var(--muted);font-weight:500}
.row{display:flex;gap:6px;flex-wrap:wrap;align-items:center;justify-content:flex-end}
button{border:1px solid var(--line);background:var(--panel);color:var(--ink);padding:7px 14px;border-radius:999px;cursor:pointer;font:inherit;font-size:14px}
button.on{background:var(--ink);color:var(--bg);border-color:var(--ink)}
input[type=range]{width:240px;accent-color:var(--accent)}
.budget{font-size:13px;color:var(--muted)}.budget b{color:var(--ink)}
.warnbar{margin-top:6px;padding:4px 10px;border-radius:8px;background:var(--chip);color:var(--warn);font-weight:600;font-size:13px}
main{padding:12px 28px;display:grid;gap:10px;--h:max(160px,calc((100vh - 190px) / var(--n,3) - 34px))}
.lane{display:grid;grid-template-columns:minmax(0,1fr) auto auto;gap:18px;align-items:center;background:var(--panel);border:1px solid var(--line);border-radius:16px;padding:12px 16px;border-left:6px solid var(--line)}
.lane.good{border-left-color:var(--ok)}.lane.mid{border-left-color:var(--warn)}.lane.bad{border-left-color:var(--bad)}
.tags{display:flex;gap:6px;flex-wrap:wrap}
.tag{font-size:11px;font-weight:700;letter-spacing:.08em;text-transform:uppercase;padding:3px 9px;border-radius:999px;background:var(--chip);color:var(--muted)}
.tag.off{background:var(--ink);color:var(--bg)}.tag.good{background:var(--ok);color:#fff}.tag.mid{background:var(--warn);color:#fff}.tag.bad{background:var(--bad);color:#fff}
.title{font-size:clamp(26px,2.5vw,42px);font-weight:750;letter-spacing:-.02em;line-height:1.05;margin:4px 0 2px}
.detail{color:var(--muted);font-size:13px}
.kpis{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px;margin-top:6px}
.kpi b{display:block;font-size:clamp(24px,2.3vw,40px);font-weight:700;line-height:1.05;letter-spacing:-.01em}.kpi span{font-size:12px;color:var(--muted)}
.kpi.good b{color:var(--ok)}.kpi.mid b{color:var(--warn)}.kpi.bad b{color:var(--bad)}
.bar{position:relative;height:8px;border-radius:4px;background:var(--track);margin:8px 0 2px;overflow:visible}
.bar i{position:absolute;left:0;top:0;bottom:0;border-radius:5px;background:var(--ok);max-width:100%}
.bar.over i{background:var(--bad)}.bar u{position:absolute;top:-4px;bottom:-4px;width:2px;background:var(--ink);opacity:.5}
.barlbl{font-size:12px;color:var(--muted)}
.minor{display:flex;gap:14px;flex-wrap:wrap;margin-top:6px;font-size:13px;color:var(--muted)}.minor b{color:var(--ink)}
.rel{margin-top:4px;font-size:15px;font-weight:650;color:var(--accent);min-height:20px}
.video{height:var(--h);width:auto;max-width:56vw;display:block;border-radius:10px;background:#222;aspect-ratio:1120/360;object-fit:contain}
.cam{width:calc(var(--h) * .82)}.cam img{width:100%;aspect-ratio:1;border-radius:10px;background:#222;display:block}
.cam .dec{font-size:16px;font-weight:700;margin-top:6px}.cam .dec.yes{color:var(--ok)}.cam .dec.wrong{color:var(--bad)}
.cam .cap{font-size:12px;color:var(--muted)}
.err{color:var(--bad);font-size:12px;margin-top:4px;word-break:break-word}.lane.down .video,.lane.down .cam img{filter:saturate(.35) brightness(.8)}
footer{padding:4px 28px 22px;color:var(--muted);font-size:13px;display:flex;gap:18px;flex-wrap:wrap}footer b{color:var(--ink)}
kbd{font:11px ui-monospace,monospace;border:1px solid var(--line);border-radius:4px;padding:0 4px}
@media (max-width:1300px){.cam{display:none}.lane{grid-template-columns:minmax(0,1fr) auto}}
@media (max-width:1050px){.lane{grid-template-columns:1fr}.video{width:100%;height:auto;max-width:none}}
@media (max-width:820px){header{grid-template-columns:1fr;padding:16px}.speed{justify-items:start}.row{justify-content:flex-start}h1{font-size:28px}
.lane{grid-template-columns:1fr;padding:14px}main{padding:12px 16px}footer,.warnbar{padding-left:16px;padding-right:16px;margin-left:16px;margin-right:16px}input[type=range]{width:100%}}
</style></head><body>
<header>
  <div><div class="kicker">Reflex · live sorting demo</div>
    <h1>Push every plastic bottle off the belt.</h1>
    <p class="lede" id="lede">Same belt, same objects, one brain per lane. The pusher sits <span id="pm">0.92</span> m after the camera: an answer that arrives late is a missed bottle.</p></div>
  <div class="speed">
    <div class="big num"><span id="spv">–</span> <small>m/s belt</small></div>
    <div class="row"><button data-v="0.2">0.2</button><button data-v="0.5">0.5</button><button data-v="1">1</button><button data-v="2">2 m/s</button>
      <button id="ramp" title="speed up 25% every 8 s">Ramp ▲</button><button id="pause">Pause</button><button id="reset">Reset</button></div>
    <input id="sp" type="range" min="-2.3" max="1.38" step="0.01" aria-label="belt speed">
    <div class="budget num">At this speed: answer within <b id="dl">–</b> · look at least every <b id="lk">–</b></div>
    <div id="load" class="warnbar" hidden></div>
  </div>
</header>
<main id="main"></main>
<footer><span id="train"></span><span>Prices: OpenAI list prices (Oct 2026): gpt-5.5 $5 / $30 per 1M in/out tokens; Decisions API $0.10 per 1M input, output free. $/hour = measured cost per call × current call rate.</span>
<span><kbd>space</kbd> pause <kbd>r</kbd> reset <kbd>1</kbd>–<kbd>4</kbd> speed <kbd>g</kbd> ramp <kbd>f</kbd> fullscreen <kbd>t</kbd> theme</span></footer>
<script>
const $=s=>document.querySelector(s),E=id=>document.getElementById(id);let S=null,built=false,ramp=null;
const post=m=>fetch('/api/control',{method:'POST',body:JSON.stringify(m)}).then(r=>r.json()).then(s=>{S=s;paint()});
const setSpeed=v=>post({speed_mps:v});
document.querySelectorAll('[data-v]').forEach(b=>b.onclick=()=>{stopRamp();setSpeed(+b.dataset.v)});
$('#sp').oninput=e=>{stopRamp();setSpeed(Math.exp(+e.target.value))};
$('#pause').onclick=()=>post({paused:!S.paused});$('#reset').onclick=()=>post({reset:true});
function stopRamp(){if(ramp){clearInterval(ramp);ramp=null;$('#ramp').classList.remove('on')}}
$('#ramp').onclick=()=>{if(ramp)return stopRamp();$('#ramp').classList.add('on');ramp=setInterval(()=>{const v=S.speed_mps*1.25;if(v>3.2)return stopRamp();setSpeed(v)},8000)};
document.addEventListener('keydown',e=>{if(e.target.tagName==='INPUT')return;const k=e.key;
 if(k===' '){e.preventDefault();$('#pause').click()}else if(k==='r')$('#reset').click();else if('1234'.includes(k)&&k.length===1){stopRamp();setSpeed([0.2,0.5,1,2][+k-1])}
 else if(k==='g')$('#ramp').click();else if(k==='f'){document.fullscreenElement?document.exitFullscreen():document.documentElement.requestFullscreen()}
 else if(k==='t'){const r=document.documentElement;r.dataset.theme=r.dataset.theme==='dark'?'light':'dark'}});
const fmtMs=ms=>ms==null?'–':ms<1000?Math.round(ms)+' ms':(ms/1000).toFixed(1)+' s';
const fmtUsd=v=>v==null?'–':v===0?'$0':v<0.1?'$'+v.toFixed(3):v<10?'$'+v.toFixed(2):'$'+v.toFixed(0);
const fmtX=x=>x>=100?Math.round(x).toLocaleString()+'×':x.toFixed(x>=10?0:1)+'×';
function build(){$('#main').style.setProperty('--n',S.lanes.length);$('#main').innerHTML=S.lanes.map((l,i)=>`<section class="lane" id="lane${i}">
 <div><div class="tags"><span class="tag ${l.where==='device'?'off':''}">${l.where==='device'?'offline':'cloud'}</span><span class="tag" id="st${i}">warming up</span></div>
  <div class="title">${l.title}</div><div class="detail">${l.detail}</div>
  <div class="kpis"><div class="kpi" id="k1${i}"><b class="num">–</b><span>bottles sorted</span></div>
   <div class="kpi" id="k2${i}"><b class="num">–</b><span>latency (median)</span></div>
   <div class="kpi" id="k3${i}"><b class="num">–</b><span>per hour, at this rate</span></div></div>
  <div class="bar" id="bar${i}"><i></i><u style="left:100%"></u></div><div class="barlbl num" id="bl${i}"></div>
  <div class="minor num" id="mn${i}"></div><div class="rel" id="rel${i}"></div><div class="err" id="err${i}"></div></div>
 <img class="video" src="/stream/${i}" alt="belt for ${l.title}">
 <div class="cam"><img id="cam${i}" alt="last frame ${l.title} answered"><div class="dec" id="dec${i}">waiting…</div><div class="cap">last frame this model answered</div></div>
</section>`).join('');built=true}
function kpi(id,v,cls){const e=E(id);e.firstElementChild.textContent=v;e.className='kpi '+(cls||'')}
function paint(){if(!S)return;if(!built)build();
 $('#spv').textContent=S.speed_mps.toFixed(2);$('#sp').value=Math.log(S.speed_mps);$('#pm').textContent=S.pusher_m.toFixed(2);
 $('#dl').textContent=S.deadline_s.toFixed(1)+' s';$('#lk').textContent=S.look_s.toFixed(1)+' s';$('#pause').textContent=S.paused?'Resume':'Pause';
 const L=$('#load');L.hidden=S.cpu_load<1.5;L.textContent=`⚠ CPU busy (load ${S.cpu_load.toFixed(1)}× cores): close other heavy jobs`;
 const T=S.training;$('#train').innerHTML=T?`<b>How the student was made:</b> ${T.frames.toLocaleString()} sim frames labelled by the Decisions API for <b>${fmtUsd(T.usd)}</b> · trained in <b>${T.train_min} min</b> on this laptop · <b>${Math.round((T.agreement||0)*100)}%</b> agreement with its teacher`:'';
 const p50s=S.lanes.map(l=>l.stats.p50_ms).filter(v=>v!=null),slow=Math.max(...p50s,0);
 const cost=S.lanes.map(l=>l.stats.usd_per_hour),dear=Math.max(...cost,0);
 const slowName=(S.lanes.find(l=>l.stats.p50_ms===slow)||{}).title, dearName=(S.lanes.find(l=>l.stats.usd_per_hour===dear)||{}).title;
 S.lanes.forEach((l,i)=>{const x=l.stats,dl=S.deadline_s*1000;
  const rc=x.recent_n>=3?x.recent_recall:null,tooSlow=x.p50_ms!=null&&x.p50_ms>dl;
  let st='warming up',cls='';if(x.down){st='no connection';cls='bad'}else if(l.warm.startsWith('error')&&!x.decisions){st='error';cls='bad'}
  else if(tooSlow){st='too slow for this speed';cls='bad'}else if(rc!=null){[st,cls]=rc>=.9?['keeping up','good']:rc>=.6?['slipping','mid']:['falling behind','bad']}
  E('st'+i).textContent=st;E('st'+i).className='tag '+cls;E('lane'+i).className='lane '+cls;
  const rec=x.recall;kpi('k1'+i,x.targets?`${x.tp}/${x.targets}`:'–',rec==null?'':rec>=.9?'good':rec>=.6?'mid':'bad');
  kpi('k2'+i,fmtMs(x.p50_ms),tooSlow?'bad':'');kpi('k3'+i,x.decisions?fmtUsd(x.usd_per_hour):'–');
  const f=x.p50_ms==null?0:x.p50_ms/dl;const b=E('bar'+i);b.className='bar'+(f>1?' over':'');b.firstElementChild.style.width=Math.min(100,f*100)+'%';
  E('bl'+i).textContent=x.p50_ms==null?'':`latency uses ${Math.round(f*100)}% of the ${S.deadline_s.toFixed(1)} s pusher deadline`;
  E('mn'+i).innerHTML=`<span><b>${x.fn}</b> missed</span><span><b>${x.fp}</b> wrong push${x.fp===1?'':'es'}</span><span><b>${x.late_yes}</b> late</span><span><b>${x.hz}</b> answers/s</span><span><b>${l.votes}</b> vote${l.votes>1?'s':''} to push</span>`;
  let rel='';if(x.p50_ms&&slow&&x.p50_ms<slow*0.5)rel+=`${fmtX(slow/x.p50_ms)} faster than ${slowName}`;
  if(dear>0&&x.decisions&&x.usd_per_hour<dear*0.5)rel+=(rel?' · ':'')+(x.usd_per_hour===0?`free vs ${fmtUsd(dear)}/h`:`${fmtX(dear/x.usd_per_hour)} cheaper`);
  E('rel'+i).textContent=rel;E('rel'+i).title=l.price_note;
  const d=l.last,dec=E('dec'+i);if(d){dec.textContent=`${d.push?'PUSH':'pass'} · ${fmtMs(d.ms)}`+(d.push!==d.truth?' · wrong':'');dec.className='dec '+(d.push!==d.truth?'wrong':d.push?'yes':'')}
  E('err'+i).textContent=x.down?`API unreachable: ${x.last_error.slice(0,90)}`:'';E('lane'+i).classList.toggle('down',x.down)})}
async function tick(){try{S=await (await fetch('/api/state')).json();paint()}catch(e){}}
setInterval(tick,400);tick();
setInterval(()=>document.querySelectorAll('.cam img').forEach((im,i)=>{if(!im.offsetParent)return;const n=new Image();n.onload=()=>{im.src=n.src};n.src='/cam/'+i+'?'+Date.now()}),600);
</script></body></html>"""
