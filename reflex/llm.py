"""Optional headless orchestrator LLM (OpenAI Responses API, model = REFLEX_LLM_MODEL).

By default Reflex runs in *agent mode*: the calling coding agent (Claude Code, Codex, ...)
is the orchestrator and writes the spec / stress plan itself, so no extra LLM key or model
is needed. Set REFLEX_LLM_MODEL (or pass --llm) to let the engine call an LLM on its own."""

from __future__ import annotations

import base64
import json
import os
from importlib import resources
from pathlib import Path

from .utils import ReflexError


def prompt(name: str) -> str:
    return resources.files("reflex").joinpath(f"prompts/{name}.md").read_text()


def llm_model() -> str | None:
    return os.environ.get("REFLEX_LLM_MODEL") or None


def llm_json(instructions: str, text: str, images: list[Path] = ()) -> dict:
    from openai import OpenAI

    model = llm_model()
    if not model:
        raise ReflexError("headless mode needs REFLEX_LLM_MODEL in .env (or use agent mode: --draft/--plan)")
    content = [{"type": "input_text", "text": text}]
    for p in images:
        b64 = base64.b64encode(Path(p).read_bytes()).decode()
        content.append({"type": "input_image", "image_url": f"data:image/jpeg;base64,{b64}"})
    resp = OpenAI().responses.create(model=model, instructions=instructions,
                                     input=[{"role": "user", "content": content}],
                                     text={"format": {"type": "json_object"}})
    try:
        return json.loads(resp.output_text)
    except json.JSONDecodeError as e:
        raise ReflexError(f"LLM did not return JSON: {resp.output_text[:300]}") from e
