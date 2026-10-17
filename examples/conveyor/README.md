# Conveyor demo: big models teach, small models react

A synthetic recycling line. Mixed waste rides a belt past a camera; a pusher downstream knocks
**plastic bottles** into a bin. Three brains run the same belt (same seed, same objects) at the
same time:

| brain | what | typical latency |
|---|---|---|
| `vlm` | general vision LLM, Responses API (`REFLEX_VLM_MODEL`, default `gpt-5.5`) | seconds |
| `decisions` | the Reflex teacher, Decisions API (`REFLEX_TEACHER_MODEL`) | ~0.3–1 s |
| `student` | the Reflex skill trained from the teacher's labels, ONNX on CPU | ~7 ms |
| `oracle@<ms>` | ground truth after a fixed delay: a perfect model with that latency (free) | as set |

## How timing works (why speed matters)

The camera sees an inspection zone between two yellow lines; the pusher is 0.92 m (460 px) further
down the belt. A "yes" for a frame captured at belt position `x` schedules the pusher to fire when
the belt has moved to `x + 0.92 m`. That is the belt-encoder timing real sorters use (think: a
trajectory planned against the encoder count, not the wall clock). Two limits follow:

- **deadline**: the answer must arrive before the object reaches the pusher → `latency < 0.92 m / speed`
- **coverage**: at least one frame must be answered while the bottle's centre is in the 0.32 m zone →
  `latency / requests_in_flight < 0.32 m / speed`

Running more requests at once (`vlm:4`) helps coverage but never the deadline. The student is
limited only by the 30 fps camera.

## Run it

```bash
reflex conveyor sprites                        # once: 35 object sprites via gpt-image-1.5 (~$0.5)
reflex conveyor live --brains vlm,decisions,oracle@0   # before training: the student isn't there yet
```

Train the student on footage from the sim:

```bash
reflex conveyor record data/conveyor           # 12 x 60 s episodes, random speed/lighting + ground_truth.json
reflex ingest data/conveyor --run conveyor --estimate 0
reflex spec --run conveyor --query "push plastic bottles off the recycling belt" --draft examples/conveyor/spec_draft.json
reflex label --run conveyor                    # ~1500 Decisions calls, ~$0.6
reflex train --run conveyor --backbone efficientnet_b0 --finetune   # ~5 min; see results below
reflex eval --run conveyor && reflex export --run conveyor --name conveyor
reflex conveyor score --run conveyor           # teacher AND student vs the simulator's ground truth
```

Then the showdown:

```bash
reflex conveyor live                           # http://localhost:8766: three belts, a speed slider, live stats
reflex conveyor bench --speeds 0.15,0.3,0.6,1.2,2.4   # → runs/conveyor/conveyor_bench.{json,png}
```

## Ground truth

The sim knows the truth, which normal Reflex runs don't have. `reflex conveyor score` reports the
teacher's and the student's accuracy against it, both on all frames and on *clear* frames (no object
centre within 6 cm of a yellow line). The first run of this demo found the teacher missing a third
of the bottles even though the student agreed with it 94% of the time: the spec's cues ("glass is
dark green/brown") were pushing dark plastic bottles to "no". Rewriting the cues (see
`spec_draft.json`) raised teacher recall on clear frames from 0.67 to 0.77, and that fix carries over
to the student.

## Results so far (M-series Mac, 1500 labelled frames)

Per-frame, against the simulator's ground truth on clear test frames:

| | precision | recall | latency |
|---|---|---|---|
| teacher (Decisions API) | 0.91 | 0.80 | ~350 ms |
| student, mobilenetv3_small (frozen) | 0.70 | 0.76 | ~6 ms |
| student, efficientnet_b0 (frozen) | 0.84 | 0.84 | ~24 ms |
| student, efficientnet_b0 `--finetune` | **0.96** | **0.88** | ~8 ms (CoreML) |

On the belt, from the real-time sweep (the mobilenet student, 1 vote): the VLM (gpt-5.5, ~2.6 s) keeps up only
at 0.15 m/s; the Decisions API sustains 1.2 m/s and collapses at 2.4 m/s; the student's accuracy
is the same at 0.15 and 2.4 m/s because it is bounded by the camera's 30 fps, not the network.
Fast brains see each object ~10 times, so one bad frame becomes a false push: the student
therefore needs 2 yes-votes per object (`student~k` to change it).

## Demo day checklist

- Close other heavy jobs (ingest, train, video calls): the page shows a "CPU busy" warning when load is high.
- `reflex conveyor live` runs a preflight: sprites, skill (should say *Apple Neural Engine / GPU*), API key,
  and one warm-up call per brain so the first answer on stage isn't slowed by connection setup.
- Present full screen (`f`). Keys: `space` pause, `r` reset, `1`–`4` speed 0.2 / 0.5 / 1 / 2 m/s, `g` ramp
  (+25% every 8 s until something breaks), `t` light/dark.
- Suggested script: start at 0.2 m/s (all keep up, but look at $/hour), press `2` (0.5 m/s: GPT-5.5 falls
  behind), press `4` (2 m/s: Decisions API slips, the student doesn't), then **turn wifi off**: both cloud lanes
  show NO CONNECTION and bottles pass unsorted, while the offline student keeps sorting. Turn wifi back on and
  the cloud lanes recover within seconds.
- Costs while running (Oct 2026 list prices): GPT-5.5 ≈ $15/h, Decisions API ≈ $0.50/h, student $0.

## Knobs

- `--brains vlm:4,decisions@gpt-6-sol,student~3`: `@model`, `:requests in flight`, `~yes-votes before pushing`
- `REFLEX_VLM_EFFORT` (default `low`), `REFLEX_VLM_PRICE_IN_PER_M` / `_OUT_PER_M` (for the $/hour tile)
- `reflex conveyor record --min-speed/--max-speed`: the speed range the student is trained on
- `--procedural`: simple shapes instead of sprites (no image API)
