# Reflex

**Big models teach. Small models react.**

Reflex turns a sentence and a few minutes of video into a tiny, fast perception model that
runs on the device. A big vision model (OpenAI **Decisions API**) is the *teacher*: the
calling agent turns your vague request into a precise visual label spec, the teacher labels
every frame, and Reflex trains a few-MB *student* (frozen pretrained backbone + small head
on soft labels) that runs in milliseconds on a laptop CPU, then reports exactly how it
compares with the teacher.

```
videos + "vague query"
  ├─ ingest   video → frames (fps sampling, resize) + measured time/cost estimate
  ├─ spec     agent (or LLM) looks at sample frames → spec.json + clarifying questions
  ├─ label    Decisions API labels every frame (soft labels, cached, resumable) [+ propagate-back]
  ├─ train    frozen timm backbone + head, seconds on a Mac; split by episode/time (no leakage)
  ├─ eval     student vs teacher: agreement, F1, latency, size, cost, disagreements, demo clip
  ├─ stress   realistic perturbations → who breaks first (optional)
  └─ export   skills/<name>/ → model.onnx + skill.json + predict.py + report
```

Prior work: Edge Impulse shipped GPT-4o-labelled training data for small models in 2024.
Reflex's difference is that an agent runs the *whole* lifecycle (spec → labels → model →
eval → robustness), packaged as a CLI any coding agent (Claude Code, Codex) can drive.
See [SKILL.md](SKILL.md) / [AGENTS.md](AGENTS.md).

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
| `reflex ui [--run <r>]` | local dashboard (http://localhost:8765): metrics, label gallery, student-vs-teacher eval frames with replay, stress |

Exit codes: `0` ok, `2` needs input, `1` error. Each stage writes `runs/<run>/results/<stage>.json`.
