"""
LINE Messaging API webhook receiver.

Run (env: LINE_CHANNEL_SECRET required, LINE_CHANNEL_ACCESS_TOKEN for names,
LINE_DB_PATH optional, default data/line.db):
    uvicorn src.line_webhook:create_app_from_env --factory --port 8000

Webhook URL to register in LINE Developers Console: https://<host>/callback
"""
import base64
import hashlib
import hmac
import json
import os
import sys
import urllib.error
import urllib.request
from typing import Optional

from fastapi import BackgroundTasks, FastAPI, Header, HTTPException, Request

from .line_adapter import apply_event
from .line_store import LineStore


def verify_signature(body: bytes, signature: str, channel_secret: str) -> bool:
    digest = hmac.new(channel_secret.encode("utf-8"), body, hashlib.sha256).digest()
    expected = base64.b64encode(digest).decode("ascii")
    return hmac.compare_digest(expected, signature or "")


class LineApi:
    """Minimal read-only client for group / member names."""

    BASE = "https://api.line.me/v2/bot"

    def __init__(self, access_token: str, timeout: float = 10):
        self.access_token = access_token
        self.timeout = timeout

    def _get(self, path: str) -> Optional[dict]:
        req = urllib.request.Request(
            self.BASE + path, headers={"Authorization": f"Bearer {self.access_token}"}
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read())
        except (urllib.error.URLError, TimeoutError, ValueError) as e:
            print(f"[LineApi] GET {path} 失敗: {e}", file=sys.stderr)
            return None

    def group_name(self, group_id: str) -> Optional[str]:
        data = self._get(f"/group/{group_id}/summary")
        return data.get("groupName") if data else None

    def member_name(self, group_id: str, user_id: str) -> Optional[str]:
        data = self._get(f"/group/{group_id}/member/{user_id}")
        return data.get("displayName") if data else None


def resolve_names(store: LineStore, api: LineApi) -> None:
    """Fill in missing group / member names; failures are retried on the next webhook."""
    for group_id in store.groups_missing_name():
        name = api.group_name(group_id)
        if name:
            store.upsert_group(group_id, name)
    for group_id, user_id in store.members_missing_name():
        name = api.member_name(group_id, user_id)
        if name:
            store.upsert_member(group_id, user_id, name)


def create_app(channel_secret: str, store: LineStore, api: Optional[LineApi] = None) -> FastAPI:
    if not channel_secret:
        raise ValueError("channel_secret is required to verify webhook signatures")

    app = FastAPI(title="LINE Triage Webhook")

    @app.post("/callback")
    async def callback(
        request: Request,
        background_tasks: BackgroundTasks,
        x_line_signature: str = Header(default=""),
    ):
        body = await request.body()
        if not verify_signature(body, x_line_signature, channel_secret):
            raise HTTPException(status_code=400, detail="invalid signature")

        payload = json.loads(body)
        for event in payload.get("events", []):
            try:
                apply_event(store, event)
            except Exception as e:  # one bad event must not drop the rest of the batch
                print(f"[webhook] 事件處理失敗 ({event.get('type')}): {e}", file=sys.stderr)

        # Name lookups hit the LINE API; do them after responding so LINE gets a fast 200
        if api is not None:
            background_tasks.add_task(resolve_names, store, api)
        return {"status": "ok"}

    @app.get("/healthz")
    def healthz():
        return {"status": "ok"}

    return app


def create_app_from_env() -> FastAPI:
    token = os.environ.get("LINE_CHANNEL_ACCESS_TOKEN")
    return create_app(
        channel_secret=os.environ.get("LINE_CHANNEL_SECRET", ""),
        store=LineStore(os.environ.get("LINE_DB_PATH", "data/line.db")),
        api=LineApi(token) if token else None,
    )
