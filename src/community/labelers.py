"""
Labelling backends. Each takes a batch of sampled records and returns label
dicts ({id, stance, topic, tickers}) for the ids it managed to label; ids it
skipped are simply retried on the next `label_gold` run.

- ClaudeLabeler: Anthropic API, structured outputs via `messages.parse`.
- OllamaLabeler: a local model (e.g. qwen3:8b) through Ollama's native
  /api/chat with a JSON-schema `format`; nothing leaves the machine.
"""
import json
import re
import urllib.error
import urllib.request
from typing import Optional

from pydantic import BaseModel

from .gold import PROMPTS, LabelBatch, _format_batch


def _slug(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", model).strip("-")


def source_name(model: str, prompt: str) -> str:
    """Label-file name; v1 keeps the bare model name so earlier files still match."""
    return _slug(model) if prompt == "v1" else f"{_slug(model)}-{prompt}"


def _keep_requested(labels: LabelBatch, records: list[dict]) -> list[dict]:
    wanted = {r["id"] for r in records}
    seen, out = set(), []
    for lbl in labels.labels:
        if lbl.id in wanted and lbl.id not in seen:
            seen.add(lbl.id)
            out.append(lbl.model_dump())
    return out


class ClaudeLabeler:
    def __init__(self, model: str = "claude-haiku-4-5", effort: Optional[str] = None, client=None,
                 prompt: str = "v2"):
        if client is None:
            import anthropic
            client = anthropic.Anthropic()
        self.client, self.model, self.effort, self.prompt = client, model, effort, prompt
        self.name = source_name(model, prompt)

    def ask(self, system: str, user: str, schema: type[BaseModel]) -> BaseModel:
        """One structured call; also used by the critic (critique.py) with its own prompt and schema."""
        # Haiku 4.5 rejects `effort`; only send it when asked (Opus / Sonnet)
        extra = {"output_config": {"effort": self.effort}} if self.effort else {}
        response = self.client.messages.parse(
            model=self.model,
            max_tokens=16000,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_format=schema,
            **extra,
        )
        if response.stop_reason == "refusal":
            raise RuntimeError(f"refused: {getattr(response.stop_details, 'category', None)}")
        return response.parsed_output

    def label(self, records: list[dict]) -> list[dict]:
        return _keep_requested(self.ask(PROMPTS[self.prompt], _format_batch(records), LabelBatch), records)


class OllamaLabeler:
    """
    `think=False` turns off Qwen3's reasoning trace: on CPU every thinking
    token costs ~0.2 s, and the JSON schema already constrains the answer.
    Pass think=None for models without a thinking mode (Ollama rejects the flag).
    """

    def __init__(self, model: str = "qwen3:8b", url: str = "http://localhost:11434",
                 num_ctx: int = 8192, timeout: float = 3600, think: Optional[bool] = False, prompt: str = "v2"):
        self.model, self.url, self.num_ctx, self.timeout, self.think = model, url.rstrip("/"), num_ctx, timeout, think
        self.prompt = prompt
        self.name = source_name(model, prompt)

    def _body(self, system: str, user: str, schema: type[BaseModel]) -> dict:
        body = {
            "model": self.model,
            "stream": False,
            "keep_alive": "30m",
            "format": schema.model_json_schema(),
            "options": {"temperature": 0, "num_ctx": self.num_ctx},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if self.think is not None:
            body["think"] = self.think
        return body

    def _payload(self, records: list[dict]) -> dict:
        return self._body(PROMPTS[self.prompt], _format_batch(records), LabelBatch)

    def label(self, records: list[dict]) -> list[dict]:
        return _keep_requested(self.ask(PROMPTS[self.prompt], _format_batch(records), LabelBatch), records)

    def ask(self, system: str, user: str, schema: type[BaseModel]) -> BaseModel:
        """One structured call; also used by the critic (critique.py) with its own prompt and schema."""
        req = urllib.request.Request(
            f"{self.url}/api/chat", data=json.dumps(self._body(system, user, schema)).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read())
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"Ollama {e.code}: {e.read().decode('utf-8', 'replace')[:200]}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"連不到 Ollama（{self.url}），請確認 Ollama 正在執行：{e.reason}") from e
        return schema.model_validate_json(data.get("message", {}).get("content", ""))


def make_labeler(backend: str, model: Optional[str] = None, effort: Optional[str] = None,
                 ollama_url: str = "http://localhost:11434", prompt: str = "v2"):
    if prompt not in PROMPTS:
        raise ValueError(f"unknown prompt version: {prompt}")
    if backend == "claude":
        return ClaudeLabeler(model or "claude-haiku-4-5", effort=effort, prompt=prompt)
    if backend == "ollama":
        model = model or "qwen3:8b"
        think = False if model.startswith(("qwen3", "deepseek-r1")) else None
        return OllamaLabeler(model, url=ollama_url, think=think, prompt=prompt)
    raise ValueError(f"unknown backend: {backend}")
