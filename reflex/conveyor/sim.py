"""Synthetic recycling conveyor: belt, objects, camera, encoder-timed pusher, ground-truth scoring.

World units are px with PX_PER_M = 500 (1 px = 2 mm). The belt moves +x. A camera looks at an
inspection zone between two yellow lines; a pusher sits PUSH_D px downstream. When a brain says
"target in zone" for a frame captured at belt odometer `o`, the pusher is scheduled for odometer
`o + PUSH_D` — like a real sorting line timing its air jets off the belt encoder. Answers that come
back after that point are *late*: the object has already passed the pusher.

Everything is deterministic given the seed, and objects spawn by belt distance (not time), so belts
with the same seed see the same stream of objects whatever their speed."""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np

from .sprites import KINDS

PX_PER_M = 500
W, H = 1120, 360
BELT_Y0, BELT_Y1 = 60, 280
ZONE = (260, 420)                       # camera inspection zone (x range) between the yellow lines
CAM_BOX = (210, 40, 470, 300)           # camera crop x0, y0, x1, y1 (world px)
CAM_SCALE = 1.5                         # camera frame = 390 x 390 px
PUSH_D = 460                            # camera zone -> pusher distance
PUSH_MARGIN = 18
PUSH_SPAN = (ZONE[0] + PUSH_D - PUSH_MARGIN, ZONE[1] + PUSH_D + PUSH_MARGIN)
SPAWN_X, EXIT_X = -150, W + 70
STROKE_OUT, STROKE_BACK = 0.10, 0.22    # pusher paddle timing, s
MERGE_GAP = (ZONE[1] - ZONE[0]) / 2     # yes-captures closer than this are the same object
AMBIG = 30                              # ground truth: centre this close to a line = ambiguous
CLEAT = 60                              # belt cleat spacing (px) — makes motion visible


def mps(px_per_s: float) -> float:
    return px_per_s / PX_PER_M


@dataclass
class Obj:
    id: int
    kind: str
    target: bool
    x: float
    y: float
    rgba: np.ndarray                    # rotated sprite at scale 1, shadow baked in
    cache: dict = field(default_factory=dict)  # scale -> (premultiplied rgb, alpha)
    state: str = "belt"                 # belt | pushed | done
    vy: float = 0.0

    def layers(self, s: float):
        if s not in self.cache:
            img = self.rgba if s == 1 else cv2.resize(
                self.rgba, None, fx=s, fy=s, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
            a = img[..., 3:4].astype(np.float32) / 255
            self.cache[s] = (img[..., :3].astype(np.float32) * a, a)
        return self.cache[s]


@dataclass
class Stats:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0
    decisions: int = 0
    yes: int = 0
    late_yes: int = 0
    errors: int = 0
    usd: float = 0.0
    last_error: str = ""
    last_ok_t: float = -1.0              # sim time of the last good answer / last failed call
    last_err_t: float = -1.0
    lat_ms: deque = field(default_factory=lambda: deque(maxlen=300))
    stamps: deque = field(default_factory=lambda: deque(maxlen=300))  # sim time of each decision
    recent: deque = field(default_factory=lambda: deque(maxlen=8))    # last targets: pushed?

    def as_dict(self, sim_t: float) -> dict:
        lat = sorted(self.lat_ms)
        done = self.tp + self.fp + self.fn + self.tn
        recent = [t for t in self.stamps if t > sim_t - 5]
        hours = max(sim_t, 1e-6) / 3600
        return {"tp": self.tp, "fp": self.fp, "fn": self.fn, "tn": self.tn, "objects": done,
                "targets": self.tp + self.fn,
                "recall": round(self.tp / (self.tp + self.fn), 3) if self.tp + self.fn else None,
                "precision": round(self.tp / (self.tp + self.fp), 3) if self.tp + self.fp else None,
                "accuracy": round((self.tp + self.tn) / done, 3) if done else None,
                "decisions": self.decisions, "yes": self.yes, "late_yes": self.late_yes,
                "hz": round(len(recent) / min(5, max(sim_t, 1e-6)), 2),
                "p50_ms": round(lat[len(lat) // 2], 1) if lat else None,
                "p95_ms": round(lat[int(len(lat) * 0.95)], 1) if lat else None,
                "usd": round(self.usd, 4), "usd_per_hour": round(self.usd / hours, 2),
                "objects_per_min": round(done / max(sim_t, 1e-6) * 60, 1),
                "recent_recall": round(sum(self.recent) / len(self.recent), 3) if self.recent else None,
                "recent_n": len(self.recent),
                "usd_per_decision": round(self.usd / self.decisions, 6) if self.decisions else None,
                "errors": self.errors, "last_error": self.last_error, "down": self.down(sim_t)}

    def down(self, sim_t: float) -> bool:
        """The brain's backend is failing right now (e.g. no network): last call failed, recently."""
        return self.last_err_t > self.last_ok_t and sim_t - self.last_err_t < 6


def _rotate_with_shadow(spr: np.ndarray, angle: float, long_px: float, gain: float) -> np.ndarray:
    h, w = spr.shape[:2]
    s = long_px / max(h, w)
    spr = cv2.resize(spr, (max(2, round(w * s)), max(2, round(h * s))), interpolation=cv2.INTER_AREA)
    h, w = spr.shape[:2]
    pad = 10
    d = int(np.ceil(np.hypot(h, w))) + 2 * pad
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    m[:, 2] += (d - w) / 2, (d - h) / 2
    rot = cv2.warpAffine(spr, m, (d, d), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0, 0))
    a = rot[..., 3].astype(np.float32) / 255
    sh = np.roll(np.roll(cv2.GaussianBlur(a, (0, 0), 4), 5, 0), 4, 1) * 0.5
    a_tot = a + sh * (1 - a)
    rgb = rot[..., :3].astype(np.float32) * np.clip(gain, 0, 2)
    rgb = rgb * (a / np.maximum(a_tot, 1e-6))[..., None]   # shadow is black
    out = np.dstack([np.clip(rgb, 0, 255), a_tot * 255]).astype(np.uint8)
    ys, xs = np.where(out[..., 3] > 2)
    return out[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def _belt_texture(rng: np.random.Generator) -> np.ndarray:
    """Belt rows at scale 1: rubber with fine noise + cleats, width W + CLEAT (periodic)."""
    h, w = BELT_Y1 - BELT_Y0, W + CLEAT
    tex = np.full((h, w, 3), 52, np.float32) + rng.normal(0, 4, (h, w, 1))
    tex += np.linspace(-6, 6, h)[:, None, None]
    for x in range(0, w, CLEAT):
        tex[:, x:x + 3] += 22
        tex[:, x + 3:x + 5] -= 10
    return np.clip(tex, 0, 255).astype(np.uint8)


class Belt:
    """One conveyor. Thread-safe: the sim loop calls step(), brain workers call capture() and
    decide(), the web view calls render_display()."""

    def __init__(self, sprites: dict[str, list[np.ndarray]], seed: int = 0, speed: float = 150.0,
                 target: str = "plastic_bottle", p_target: float = 0.3, gap: tuple[float, float] = (170, 330),
                 light: dict | None = None):
        self.sprites, self.seed, self.target = sprites, seed, target
        self.p_target, self.gap = p_target, gap
        self.speed = speed                      # px/s
        self.min_votes = 1                      # yes-answers per object before the pusher fires (debounce)
        self.light = light or {}
        self.lock = threading.RLock()
        self.belt_tex = _belt_texture(np.random.default_rng(1234))
        self._tex_cache: dict = {}
        self._static_cache: dict = {}
        self.reset()

    # ------------------------------------------------------------ state

    def reset(self):
        with self.lock:
            self.epoch = getattr(self, "epoch", -1) + 1   # answers for frames from before a reset are dropped
            self.rng = np.random.default_rng(self.seed)
            self.noise_rng = np.random.default_rng(self.seed + 1)
            self.t = 0.0                        # sim time, s
            self.odo = 0.0                      # belt odometer, px
            self.next_spawn = 0.0
            self.objs: list[Obj] = []
            self.n_ids = 0
            self.schedule: list[dict] = []      # {"lo", "hi"} capture odometers of one detection run
            self.stroke_t = -1.0                # sim time the paddle fired
            self.events: list[dict] = []        # display flashes
            self.binned = 0
            self.stats = Stats()
            self.last_yes_t = -9.0
            self.in_flight = 0

    def _spawn(self):
        kinds = [k for k in self.sprites if k != self.target]
        is_t = self.target in self.sprites and self.rng.random() < self.p_target
        kind = self.target if is_t else kinds[self.rng.integers(len(kinds))]
        spr = self.sprites[kind][self.rng.integers(len(self.sprites[kind]))]
        long_px = KINDS.get(kind, (200,))[0] / 1000 * PX_PER_M * self.rng.uniform(0.85, 1.12)
        rgba = _rotate_with_shadow(spr, float(self.rng.uniform(0, 360)), long_px, float(self.rng.uniform(0.85, 1.1)))
        hh = rgba.shape[0] / 2
        y = self.rng.uniform(BELT_Y0 + min(hh, 100) * 0.8, BELT_Y1 - min(hh, 100) * 0.8)
        self.objs.append(Obj(self.n_ids, kind, kind == self.target, SPAWN_X + (self.odo - self.next_spawn), y, rgba))
        self.n_ids += 1
        self.next_spawn += self.rng.uniform(*self.gap)

    def _resolve(self, o: Obj, pushed: bool):
        st = self.stats
        if pushed:
            if o.target:
                st.tp += 1
                st.recent.append(True)
            else:
                st.fp += 1
                self._flash(o.x, BELT_Y1 + 30, "WRONG", (0, 140, 255))
        elif o.target:
            st.fn += 1
            st.recent.append(False)
            self._flash(W - 70, BELT_Y0 - 22, "MISSED", (60, 60, 230))
        else:
            st.tn += 1

    def _flash(self, x, y, text, color, dur=1.0):
        self.events.append({"x": x, "y": y, "text": text, "color": color, "until": self.t + dur})

    def step(self, dt: float):
        with self.lock:
            self.t += dt
            dx = self.speed * dt
            self.odo += dx
            while self.odo >= self.next_spawn:
                self._spawn()
            for o in self.objs:
                if o.state == "belt":
                    o.x += dx
                    if o.x > EXIT_X:
                        o.state = "done"
                        self._resolve(o, pushed=False)
                elif o.state == "pushed":
                    o.y += o.vy * dt
                    if o.y > H + 80:
                        o.state = "done"
            self.objs = [o for o in self.objs if o.state != "done"]
            # Encoder-timed pusher: fire when the middle of a detection run reaches the pusher.
            due = [s for s in self.schedule if self.odo >= (s["lo"] + s["hi"]) / 2 + PUSH_D]
            if due:
                self.schedule = [s for s in self.schedule if s not in due]
                if any(s["votes"] >= self.votes_needed() for s in due):
                    self._fire()
            self.events = [e for e in self.events if e["until"] > self.t]

    def votes_needed(self) -> int:
        """min_votes, relaxed when the brain answers too rarely to cast that many votes on one
        object: looks per object = decision rate x time a centre spends in the zone."""
        if self.min_votes <= 1:
            return 1
        recent = [t for t in self.stats.stamps if t > self.t - 3]
        hz = len(recent) / min(3, max(self.t, 1e-6))
        looks = hz * (ZONE[1] - ZONE[0]) / max(self.speed, 1e-6)
        return max(1, min(self.min_votes, int(looks // 3)))

    def _fire(self):
        self.stroke_t = self.t
        for o in self.objs:
            if o.state == "belt" and PUSH_SPAN[0] <= o.x <= PUSH_SPAN[1]:
                o.state, o.vy = "pushed", 1400.0
                self.binned += 1
                self._resolve(o, pushed=True)

    # ------------------------------------------------------------ camera + decisions

    def ground_truth(self) -> dict:
        with self.lock:
            on = [o for o in self.objs if o.state == "belt"]
            inz = [o for o in on if ZONE[0] <= o.x <= ZONE[1]]
            # Ambiguous: an object's centre is within AMBIG px of a yellow line (could go either way).
            amb = any(abs(o.x - z) < AMBIG for o in on for z in ZONE)
            return {"target_in_zone": any(o.target for o in inz), "kinds_in_zone": sorted(o.kind for o in inz),
                    "ambiguous": amb}

    def capture(self, in_flight: bool = True) -> dict:
        """Camera frame (BGR, 390x390) + the odometer, epoch and ground truth at capture."""
        with self.lock:
            snap = {"frame": self.render(CAM_BOX, CAM_SCALE, overlays=False), "odo": self.odo,
                    "epoch": self.epoch, "t": self.t, "gt": self.ground_truth()}
            self.in_flight += in_flight
        snap["frame"] = self._camera_fx(snap["frame"])
        return snap

    def _camera_fx(self, img: np.ndarray) -> np.ndarray:
        L = self.light
        f = img.astype(np.float32)
        k = int(round(self.speed * L.get("exposure", 1 / 250) * CAM_SCALE))
        if k >= 2:
            f = cv2.filter2D(f, -1, np.full((1, k), 1 / k, np.float32))
        f *= np.asarray(L.get("tint", (1.0, 1.0, 1.0)), np.float32) * L.get("gain", 1.0)
        h, w = f.shape[:2]
        yy, xx = np.mgrid[0:h, 0:w]
        r2 = ((xx - w / 2) ** 2 + (yy - h / 2) ** 2) / (w * w / 2)
        f *= (1 - L.get("vignette", 0.25) * r2)[..., None]
        f += self.noise_rng.normal(0, L.get("noise", 3.0), f.shape).astype(np.float32)
        return np.clip(f, 0, 255).astype(np.uint8)

    def decide(self, snap: dict, push: bool, ms: float, usd: float = 0.0, error: str | None = None):
        """A brain's answer for the frame captured in `snap` arrives now."""
        with self.lock:
            if snap["epoch"] != self.epoch:
                return
            odo = snap["odo"]
            self.in_flight = max(0, self.in_flight - 1)
            st = self.stats
            if error:
                st.errors += 1
                st.last_error = error[:200]
                st.last_err_t = self.t
                return
            st.decisions += 1
            st.last_ok_t = self.t
            st.lat_ms.append(ms)
            st.stamps.append(self.t)
            st.usd += usd
            if not push:
                return
            st.yes += 1
            self.last_yes_t = self.t
            for s in self.schedule:
                if s["lo"] - MERGE_GAP <= odo <= s["hi"] + MERGE_GAP:
                    s["lo"], s["hi"] = min(s["lo"], odo), max(s["hi"], odo)
                    s["votes"] += 1
                    return
            if self.odo > odo + PUSH_D + PUSH_MARGIN:
                st.late_yes += 1
                self._flash(ZONE[0], BELT_Y0 - 22, "LATE", (60, 60, 230), 0.6)
                return
            self.schedule.append({"lo": odo, "hi": odo, "votes": 1})

    # ------------------------------------------------------------ rendering

    def _static(self, box, s: float, overlays: bool) -> np.ndarray:
        key = (box, s, overlays)
        if key not in self._static_cache:
            img = np.full((H, W, 3), (38, 36, 34), np.uint8)
            cv2.rectangle(img, (0, BELT_Y0 - 10), (W, BELT_Y0), (150, 150, 155), -1)     # rails
            cv2.rectangle(img, (0, BELT_Y1), (W, BELT_Y1 + 10), (150, 150, 155), -1)
            for x in ZONE:                                                             # yellow lines
                cv2.rectangle(img, (x - 2, BELT_Y0 - 12), (x + 2, BELT_Y0), (0, 215, 255), -1)
                cv2.rectangle(img, (x - 2, BELT_Y1), (x + 2, BELT_Y1 + 12), (0, 215, 255), -1)
            if overlays:
                x0, x1 = PUSH_SPAN
                cv2.rectangle(img, (x0 - 6, 4), (x1 + 6, BELT_Y0 - 12), (90, 90, 95), -1)   # pusher housing
                cv2.rectangle(img, (x0, BELT_Y1 + 12), (x1, H - 4), (40, 80, 30), -1)       # bin
                cv2.putText(img, "PLASTIC", (x0 + 8, H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 230, 200), 1, cv2.LINE_AA)
                cv2.putText(img, "PUSHER", (x0 + 8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
                cv2.putText(img, "OTHER ->", (W - 90, H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)
            x0, y0, x1, y1 = box
            img = img[y0:y1, x0:x1]
            if s != 1:
                img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR)
            self._static_cache[key] = img
        return self._static_cache[key]

    def _tex(self, s: float) -> np.ndarray:
        if s not in self._tex_cache:
            t = self.belt_tex
            self._tex_cache[s] = t if s == 1 else cv2.resize(t, (round(t.shape[1] * s), round(t.shape[0] * s)))
        return self._tex_cache[s]

    def render(self, box=(0, 0, W, H), s: float = 1.0, overlays: bool = True) -> np.ndarray:
        with self.lock:
            x0, y0, x1, y1 = box
            img = self._static(box, s, overlays).copy()
            # Moving belt rows.
            tex = self._tex(s)
            per = round(CLEAT * s)
            off = per - int(round(self.odo * s)) % per
            by0, by1 = round((BELT_Y0 - y0) * s), round((BELT_Y1 - y0) * s)
            bx0 = round(x0 * s) + off
            strip = tex[:, bx0:bx0 + img.shape[1]]
            lo, hi = max(by0, 0), min(by1, img.shape[0])
            img[lo:hi, :strip.shape[1]] = strip[lo - by0:hi - by0]
            # Yellow lines across the belt (translucent, under the objects).
            for zx in ZONE:
                cx = round((zx - x0) * s)
                if 0 <= cx < img.shape[1]:
                    seg = img[lo:hi, max(cx - 1, 0):cx + 2].astype(np.float32)
                    img[lo:hi, max(cx - 1, 0):cx + 2] = (seg * 0.55 + np.array([0, 215, 255]) * 0.45).astype(np.uint8)
            for o in sorted(self.objs, key=lambda o: o.state == "pushed"):
                self._paste(img, o, x0, y0, s)
            if overlays:
                self._overlays(img)
            return img

    @staticmethod
    def _paste(img, o: Obj, x0, y0, s):
        P, A = o.layers(s)
        h, w = A.shape[:2]
        cx, cy = round((o.x - x0) * s - w / 2), round((o.y - y0) * s - h / 2)
        ix0, iy0 = max(cx, 0), max(cy, 0)
        ix1, iy1 = min(cx + w, img.shape[1]), min(cy + h, img.shape[0])
        if ix0 >= ix1 or iy0 >= iy1:
            return
        sp = (slice(iy0 - cy, iy1 - cy), slice(ix0 - cx, ix1 - cx))
        roi = img[iy0:iy1, ix0:ix1].astype(np.float32)
        img[iy0:iy1, ix0:ix1] = (roi * (1 - A[sp]) + P[sp]).astype(np.uint8)

    def _overlays(self, img):
        # Camera zone: lights up green on a "yes".
        col = (80, 220, 80) if self.t - self.last_yes_t < 0.25 else (210, 210, 210)
        cv2.rectangle(img, (ZONE[0], BELT_Y0), (ZONE[1], BELT_Y1), col, 2)
        cv2.putText(img, f"CAMERA  in flight: {self.in_flight}", (ZONE[0], BELT_Y0 - 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
        # Pusher paddle.
        dt = self.t - self.stroke_t
        ext = dt / STROKE_OUT if dt < STROKE_OUT else max(0.0, 1 - (dt - STROKE_OUT) / STROKE_BACK)
        py = round(BELT_Y0 + ext * (BELT_Y1 - BELT_Y0 - 14))
        x0, x1 = PUSH_SPAN
        cv2.rectangle(img, (x0 + 40, BELT_Y0 - 12), (x1 - 40, py), (120, 120, 125), -1)
        cv2.rectangle(img, (x0, py), (x1, py + 14), (0, 120, 230), -1)
        cv2.putText(img, f"{self.binned}", (x1 - 40, H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 255, 230), 1, cv2.LINE_AA)
        # Scheduled pushes: where on the belt each one will fire (moves with the belt = encoder tracking).
        mid = (PUSH_SPAN[0] + PUSH_SPAN[1]) / 2
        for sch in self.schedule:
            if sch["votes"] < self.votes_needed():
                continue
            x = round(mid - ((sch["lo"] + sch["hi"]) / 2 + PUSH_D - self.odo))
            cv2.fillPoly(img, [np.array([[x - 7, BELT_Y0 - 10], [x + 7, BELT_Y0 - 10], [x, BELT_Y0]], np.int32)], (0, 160, 255))
        if self.stats.down(self.t):
            x0, x1 = ZONE[0] - 40, PUSH_SPAN[1] + 30
            cv2.rectangle(img, (x0, 150), (x1, 196), (40, 40, 200), -1)
            cv2.putText(img, "NO CONNECTION - bottles pass unsorted", (x0 + 18, 181), cv2.FONT_HERSHEY_SIMPLEX,
                        0.85, (255, 255, 255), 2, cv2.LINE_AA)
        for e in self.events:
            cv2.putText(img, e["text"], (round(e["x"]), round(e["y"])), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        e["color"], 2, cv2.LINE_AA)


def random_light(rng: np.random.Generator) -> dict:
    """Per-episode camera conditions for training-data diversity."""
    return {"gain": float(rng.uniform(0.75, 1.2)), "noise": float(rng.uniform(1.5, 6)),
            "tint": tuple(float(v) for v in rng.uniform(0.9, 1.1, 3)), "vignette": float(rng.uniform(0.1, 0.45)),
            "exposure": float(rng.uniform(1 / 500, 1 / 150))}


class Clock:
    """Steps a set of belts in real time on a background thread (shared dt keeps them in lockstep)."""

    def __init__(self, belts: list[Belt], hz: float = 60.0):
        self.belts, self.hz = belts, hz
        self.paused = False
        self._stop = threading.Event()
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def start(self):
        self.thread.start()
        return self

    def stop(self):
        self._stop.set()

    def _loop(self):
        last = time.perf_counter()
        while not self._stop.is_set():
            time.sleep(1 / self.hz)
            now = time.perf_counter()
            dt, last = min(now - last, 0.1), now
            if not self.paused:
                for b in self.belts:
                    b.step(dt)
