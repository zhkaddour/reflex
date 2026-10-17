"""Async Decisions API client: concurrency, retry with backoff, usage tracking, JSONL cache."""

from __future__ import annotations

import asyncio
import base64
import os
import random
import time
from pathlib import Path

import httpx
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeRemainingColumn

from .runtime import DECISIONS_URL, build_decisions_body, parse_decision, spec_classes
from .utils import ReflexError, append_jsonl, console, read_jsonl, spec_hash

RETRY_STATUS = {408, 409, 429, 500, 502, 503, 504}
MAX_TRIES = 6


class _Retryable(Exception):
    pass


class AsyncTeacher:
    """Labels JPEG files with one Decisions question. Prefers the openai SDK's
    `client.decisions` when present, otherwise POSTs to /v1/decisions with httpx."""

    def __init__(self, spec: dict, concurrency: int = 16, model: str | None = None):
        self.spec = spec
        self.classes = spec_classes(spec)
        self.model = model
        self.sem = asyncio.Semaphore(concurrency)
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ReflexError("OPENAI_API_KEY is not set (add it to .env)")
        self._sdk = None
        try:
            from openai import AsyncOpenAI

            client = AsyncOpenAI(api_key=key, max_retries=0, timeout=60)
            if hasattr(client, "decisions"):
                self._sdk = client
        except ImportError:
            pass
        self._http = httpx.AsyncClient(timeout=60, headers={"Authorization": f"Bearer {key}"})

    async def aclose(self):
        await self._http.aclose()
        if self._sdk is not None:
            await self._sdk.close()

    async def _call(self, body: dict):
        if self._sdk is not None:
            import openai

            try:
                return await self._sdk.decisions.create(**body)
            except (openai.RateLimitError, openai.APIConnectionError, openai.APITimeoutError) as e:
                raise _Retryable(str(e)) from e
            except openai.APIStatusError as e:
                if e.status_code in RETRY_STATUS:
                    raise _Retryable(str(e)) from e
                raise ReflexError(f"Decisions API {e.status_code}: {e.message}") from e
        try:
            r = await self._http.post(DECISIONS_URL, json=body)
        except (httpx.TimeoutException, httpx.TransportError) as e:
            raise _Retryable(str(e)) from e
        if r.status_code in RETRY_STATUS:
            raise _Retryable(f"{r.status_code}: {r.text[:200]}")
        if r.status_code >= 400:
            raise ReflexError(f"Decisions API {r.status_code}: {r.text[:800]}")
        return r.json()

    async def label_image(self, image_b64: str) -> dict:
        body = build_decisions_body(self.spec, image_b64, self.model)
        async with self.sem:
            for attempt in range(MAX_TRIES):
                t0 = time.perf_counter()
                try:
                    resp = await self._call(body)
                    break
                except _Retryable as e:
                    if attempt == MAX_TRIES - 1:
                        raise ReflexError(f"Decisions API kept failing: {e}") from e
                    await asyncio.sleep(min(30, 1.0 * 2 ** attempt) + random.random())
            latency_ms = (time.perf_counter() - t0) * 1000
        probs, usage = parse_decision(resp, self.spec)
        top = max(probs, key=probs.get)
        return {"probs": probs, "top": top, "conf": probs[top], "latency_ms": round(latency_ms, 1), **usage}


def _b64_file(path: Path) -> str:
    return base64.b64encode(Path(path).read_bytes()).decode()


async def _label_all(spec, items, concurrency, on_row, desc):
    teacher = AsyncTeacher(spec, concurrency=concurrency)
    rows, failures = [], []

    async def one(frame_id, path):
        try:
            row = {"frame_id": frame_id, **(await teacher.label_image(_b64_file(path)))}
        except ReflexError as e:
            failures.append((frame_id, str(e)))
            if len(failures) >= 3 and not rows:  # systematic error (bad key, bad schema...)
                raise
            return
        rows.append(row)
        on_row(row)
        progress.advance(task)

    with Progress(TextColumn(desc), BarColumn(), MofNCompleteColumn(), TimeRemainingColumn(),
                  console=console, transient=False) as progress:
        task = progress.add_task(desc, total=len(items))
        try:
            await asyncio.gather(*(one(fid, p) for fid, p in items))
        finally:
            await teacher.aclose()
    if failures:
        console.print(f"[yellow]{len(failures)} frames failed: {failures[0][1][:300]}[/yellow]")
        if not rows:
            raise ReflexError(f"all teacher calls failed: {failures[0][1]}")
    return rows, failures


def label_frames(spec: dict, frames: list[dict], run: Path, concurrency: int = 16,
                 cache: bool = True, desc: str = "labelling") -> tuple[list[dict], int]:
    """Label frames (dicts from frames.csv). With cache=True results go to run/labels.jsonl
    tagged with the spec hash, and frames already labelled under this spec are skipped.
    Returns (rows for the requested frames, n_failed)."""
    h = spec_hash(spec)
    cache_path = run / "labels.jsonl"
    done = {r["frame_id"]: r for r in read_jsonl(cache_path) if r.get("spec_hash") == h} if cache else {}
    todo = [(f["frame_id"], run / f["path"]) for f in frames if f["frame_id"] not in done]
    if done and todo:
        console.print(f"[dim]{len(frames) - len(todo)} frames already labelled (cache), {len(todo)} to go[/dim]")

    def on_row(row):
        row.update(source="teacher", spec_hash=h)
        if cache:
            append_jsonl(cache_path, [row])

    new_rows, failures = (asyncio.run(_label_all(spec, todo, concurrency, on_row, desc))
                          if todo else ([], []))
    by_id = {**done, **{r["frame_id"]: r for r in new_rows}}
    return [by_id[f["frame_id"]] for f in frames if f["frame_id"] in by_id], len(failures)
