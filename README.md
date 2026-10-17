# Reflex

**Big models teach. Small models react.**

Give Reflex a sentence and a few minutes of video. It hands back a **16 MB on-device vision model
that answers in ~8 ms, offline, for $0 per frame**, trained by a cloud model that takes ~350 ms
and costs money on every call.

> *"push plastic bottles off the recycling belt"* → 1,500 frames labelled by the OpenAI
> Decisions API for **$0.60** → a student that is **more accurate than its teacher** against
> ground truth, and **~40× faster**.

## Results

| task | student vs teacher | student latency | teacher latency | cost / 1k frames |
|---|---|---|---|---|
| Conveyor: "is this a plastic bottle?" (synthetic belt, 1,500 frames) | 96% agreement | **8.5 ms** | 350 ms | **$0** vs $0.40 |
| Real phone video: "is the MagSafe charger plugged in?" (864 frames) | 82% agreement | **7.5 ms** | 440 ms | **$0** vs $0.66 |

Student latency is the fine-tuned efficientnet_b0 on an M-series Mac (ONNX Runtime + CoreML), p50 from each skill's report.
The conveyor sim knows the truth, so we can score both models against it:

| on clear test frames vs ground truth | precision | recall |
|---|---|---|
| teacher (Decisions API) | 0.91 | 0.80 |
| **student (Reflex)** | **0.96** | **0.88** |

**Why speed matters**: on the live belt, a general VLM (~2.6 s per frame) only keeps up at
0.15 m/s and misses every bottle from 0.6 m/s. The Decisions API holds up to 1.2 m/s and
collapses at 2.4 m/s. The student is limited only by the 30 fps camera.
→ [examples/conveyor](examples/conveyor/README.md)

## How it works

```
videos + "vague query"
  ├─ ingest   video → frames + measured time/$ estimate
  ├─ spec     agent looks at sample frames → precise visual label spec (+ clarifying questions)
  ├─ label    Decisions API labels every frame (soft labels, cached, resumable)
  ├─ train    pretrained backbone + small head, minutes on a laptop; split by episode (no leakage)
  ├─ eval     student vs teacher: agreement, F1, latency, size, cost, worst disagreements
  ├─ stress   lighting / blur / noise / rotation / occlusion → who breaks first
  └─ export   skills/<name>/ → model.onnx + skill.json + predict.py + report.md
```

The whole lifecycle is a CLI built to be driven by a coding agent: just ask Claude Code or Codex
*"use reflex to train a skill that tells me if the charger is plugged in, from data/plug"*.
See [SKILL.md](SKILL.md) / [AGENTS.md](AGENTS.md). Exported skills with full reports:
[skills/conveyor](skills/conveyor/report.md), [skills/plug](skills/plug/report.md)
([stress test](skills/plug/stress.md)).

Prior work: Edge Impulse shipped GPT-4o-labelled training data for small models in 2024.
Reflex differs in that an agent runs the *whole* lifecycle (spec → labels → model → eval →
robustness) and reports honestly where the student and teacher disagree.

## Setup (macOS, Apple Silicon)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
cp .env.example .env   # add OPENAI_API_KEY
```

`REFLEX_LLM_MODEL` is optional. Without it Reflex runs in **agent mode**: the coding agent
calling it (any model) writes the spec and the stress plan. With it, `reflex` calls that
OpenAI model itself (headless).

## Quick start

```bash
reflex dummy --name rps --classes rock,paper,scissors,none   # placeholder skill for app integration

reflex ingest data/rps --run rps
reflex spec --run rps --query "is the hand showing rock, paper or scissors?"   # exit 2: agent writes draft
reflex spec --run rps --draft runs/rps/draft.json
reflex spec-check --run rps
reflex label --run rps --propagate-back 3
reflex train --run rps
reflex eval --run rps
reflex export --run rps
```

Browse every run (spec, metrics, labels, eval frames, disagreements, stress) with `reflex ui`.

Or just ask Claude Code: *"use reflex to train a rock-paper-scissors skill from data/rps"*.

## Using a skill

```python
from reflex.runtime import Skill, Teacher

skill = Skill("skills/rps")          # loads model.onnx + skill.json
out = skill.predict(frame_bgr)       # {"label": "rock", "probs": {...}, "ms": 3.2}

cloud = Teacher("skills/rps")        # same interface, calls the Decisions API live
out = cloud.predict(frame_bgr)       # same dict, "ms" ~ 300+
```

Every skill folder is self-contained: `python skills/rps/predict.py image.jpg` or
`python skills/rps/predict.py --webcam [--teacher]` (numpy + onnxruntime + opencv only).

## Testing without data or an API key

```bash
python examples/make_synthetic.py data/shapes
reflex ingest data/shapes --run shapes --estimate 0
python examples/make_synthetic.py data/shapes --fake-labels runs/shapes
reflex train --run shapes && reflex eval --run shapes && reflex export --run shapes
```

## Conveyor demo

A synthetic recycling line where a VLM, the Decisions API and a Reflex student race to push plastic
bottles off a belt in real time. The sim also produces the training footage (with ground truth).
See [examples/conveyor/README.md](examples/conveyor/README.md).

```bash
reflex conveyor live          # three identical belts side by side at http://localhost:8766
reflex conveyor bench         # max belt speed each brain sustains
```

## Commands

| Command | Does |
|---|---|
| `reflex ingest <dir> --run <r>` | frames + `frames.csv`, label 20 frames to estimate time/$ |
| `reflex spec --run <r> --query "..."` | spec request for the agent (exit 2), or headless LLM spec |
| `reflex spec --run <r> --draft d.json [--answers a.json]` | validate the agent's draft, handle questions |
| `reflex spec-check --run <r>` | label 24 frames → contact sheet + class distribution |
| `reflex label --run <r>` | label everything (concurrent, cached, resumable), `--propagate-back N` |
| `reflex models` | backbone menu |
| `reflex train --run <r> [--backbone ...] [--finetune]` | embeddings + soft-label head + ONNX |
| `reflex eval --run <r> [--human-check 30]` | student vs teacher report + demo eval frames/clip |
| `reflex stress --run <r> [--plan p.json \| --default] [--teacher-sample 20]` | robustness |
| `reflex export --run <r>` | `skills/<name>/` |
| `reflex run <dir> --name <n> --query "..."` | end to end |
| `reflex status --run <r>` | stage overview |
| `reflex conveyor sprites\|record\|live\|bench\|score` | conveyor demo (see examples/conveyor) |
| `reflex ui [--run <r>]` | local dashboard (http://localhost:8765): metrics, label gallery, student-vs-teacher eval frames with replay, stress |

Exit codes: `0` ok, `2` needs input, `1` error. Each stage writes `runs/<run>/results/<stage>.json`.
