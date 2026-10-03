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

from .gold import SYSTEM_PROMPT, LabelBatch, _format_batch


def _slug(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", model).strip("-")


def _keep_requested(labels: LabelBatch, records: list[dict]) -> list[dict]:
    wanted = {r["id"] for r in records}
    seen, out = set(), []
    for lbl in labels.labels:
        if lbl.id in wanted and lbl.id not in seen:
            seen.add(lbl.id)
            out.append(lbl.model_dump())
    return out


class ClaudeLabeler:
    def __init__(self, model: str = "claude-haiku-4-5", effort: Optional[str] = None, client=None):
        if client is None:
            import anthropic
            client = anthropic.Anthropic()
        self.client, self.model, self.effort = client, model, effort
        self.name = _slug(model)

    def label(self, records: list[dict]) -> list[dict]:
        # Haiku 4.5 rejects `effort`; only send it when asked (Opus / Sonnet)
        extra = {"output_config": {"effort": self.effort}} if self.effort else {}
        response = self.client.messages.parse(
            model=self.model,
            max_tokens=16000,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": _format_batch(records)}],
            output_format=LabelBatch,
            **extra,
        )
        if response.stop_reason == "refusal":
            raise RuntimeError(f"refused: {getattr(response.stop_details, 'category', None)}")
        return _keep_requested(response.parsed_output, records)


class OllamaLabeler:
    """
    `think=False` turns off Qwen3's reasoning trace: on CPU every thinking
    token costs ~0.2 s, and the JSON schema already constrains the answer.
    Pass think=None for models without a thinking mode (Ollama rejects the flag).
    """

    def __init__(self, model: str = "qwen3:8b", url: str = "http://localhost:11434",
                 num_ctx: int = 8192, timeout: float = 3600, think: Optional[bool] = False):
        self.model, self.url, self.num_ctx, self.timeout, self.think = model, url.rstrip("/"), num_ctx, timeout, think
        self.name = _slug(model)

    def _payload(self, records: list[dict]) -> dict:
        body = {
            "model": self.model,
            "stream": False,
            "keep_alive": "30m",
            "format": LabelBatch.model_json_schema(),
            "options": {"temperature": 0, "num_ctx": self.num_ctx},
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": _format_batch(records)},
            ],
        }
        if self.think is not None:
            body["think"] = self.think
        return body

    def label(self, records: list[dict]) -> list[dict]:
        req = urllib.request.Request(
            f"{self.url}/api/chat", data=json.dumps(self._payload(records)).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read())
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"Ollama {e.code}: {e.read().decode('utf-8', 'replace')[:200]}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"連不到 Ollama（{self.url}），請確認 Ollama 正在執行：{e.reason}") from e
        content = data.get("message", {}).get("content", "")
        return _keep_requested(LabelBatch.model_validate_json(content), records)


def make_labeler(backend: str, model: Optional[str] = None, effort: Optional[str] = None,
                 ollama_url: str = "http://localhost:11434"):
    if backend == "claude":
        return ClaudeLabeler(model or "claude-haiku-4-5", effort=effort)
    if backend == "ollama":
        model = model or "qwen3:8b"
        think = False if model.startswith(("qwen3", "deepseek-r1")) else None
        return OllamaLabeler(model, url=ollama_url, think=think)
    raise ValueError(f"unknown backend: {backend}")
