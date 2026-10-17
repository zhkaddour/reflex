# AGENTS.md — Reflex


# Reflex — big models teach, small models react

Reflex turns a vague query + videos into `skills/<name>/` (≈2–25 MB ONNX model that runs in
milliseconds on CPU) and a report comparing it with the teacher.

**You (the agent) are the orchestrator.** Reflex never prompts interactively. When it needs a
decision it writes files and exits with **code 2**; you read the JSON result, do the step, re-run.

- Exit codes: `0` ok · `2` needs input (you or the user) · `1` error (read `message`).
- Every command writes `runs/<run>/<stage>.json` (`status`, numbers, file paths). Read that,
  not the console log.
- Needs `OPENAI_API_KEY` in `.env` (teacher). No other LLM key is needed in agent mode.

## Steps

```bash
reflex ingest data/rps --run rps            # frames + time/cost estimate (estimate.json)
reflex spec --run rps --query "is the hand showing rock, paper or scissors?"   # exit 2, kind=agent_draft
```

### Spec (agent mode)
`runs/<run>/spec.json` → `status: needs_input`, `kind: agent_draft`:
1. Read `runs/<run>/spec_request/README.md` (rules + JSON shape) and **look at every sample
   frame** in that folder.
2. Write a draft JSON (e.g. `runs/<run>/draft.json`): one primary question, crisp *visual*
   class definitions, a `none` class, `visual_cues`, `edge_cases`, `propagate_back`, `hflip_ok`,
   and up to 3 `questions` (mark `blocking` only if a wrong guess ruins the labels).
3. `reflex spec --run <run> --draft runs/<run>/draft.json`
4. Show the user the classes + `visual_cues`. If exit 2 with `kind: user_questions`: ask the
   user the `questions`, write `answers.json` as `{"q1": "answer", ...}`, update the draft if the
   answers change the classes, then `reflex spec --run <run> --draft <draft> --answers answers.json`
   (or just `--answers answers.json`). `--yes` accepts the defaults.

### Check, label, train, eval, export
```bash
reflex spec-check --run rps        # 24 frames → spec_check.png: LOOK at it; fix the spec if labels look wrong
reflex label --run rps             # all frames, resumable; --propagate-back N to anticipate actions
reflex train --run rps             # default mobilenetv3_small_100; `reflex models` for the menu
reflex eval --run rps              # report.md/json, confusion.png, disagreements.png, eval_frames/, eval_clip.mp4
reflex export --run rps            # → skills/rps/
```
Optional: `reflex stress --run rps` → exit 2 `kind: agent_plan`: read
`runs/<run>/stress_request/README.md`, write a plan JSON (5–6 realistic perturbations), then
`reflex stress --run rps --plan plan.json [--teacher-sample 20]`. `--default` uses a built-in plan.

One-shot: `reflex run data/rps --name rps --query "..." --draft draft.json` (still stops at spec
with exit 2 if questions are blocking; `--yes` accepts defaults). `reflex status --run rps`
lists finished stages.

## Reading the results (`runs/<run>/eval.json`)
- `agreement` / `macro_f1`: student vs teacher on held-out frames (split by episode or time,
  never random — so the numbers are honest).
- `p50_ms` vs `teacher_p50_ms`, `speedup`, `size_mb`, `teacher_usd_per_1000`.
- `per_class` recall and `disagreements` (look at `disagreements.png`).

## When results are poor
| Symptom | Suggest |
|---|---|
| spec-check labels look wrong / many low-confidence | rewrite class descriptions to be more visual; fewer/clearer classes |
| one class has low recall / few examples | record more footage of that class (check `train.json` `train_class_counts`) |
| agreement < ~85% with balanced data | `--backbone efficientnet_b0` or `vit_small_patch14_dinov2.lvd142m`; `--hidden 256`; `--finetune` |
| model reacts too late to an action | `reflex label --propagate-back 3` then retrain |
| stress shows big drops | capture footage matching the `suggestions` in stress.json |
| `split failed` | need more frames or more than one episode |

## Using a skill in an app
```python
from reflex.runtime import Skill, Teacher
skill = Skill("skills/rps"); out = skill.predict(frame_bgr)   # {"label","probs","ms"}
cloud = Teacher("skills/rps")                                  # same interface, live Decisions API
```
`skills/<name>/predict.py` is standalone (numpy + onnxruntime + opencv):
`python skills/rps/predict.py img.jpg` or `--webcam`.
