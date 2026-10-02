from datetime import datetime

import pytest

from src.parser import Message


@pytest.fixture
def make_msg():
    """Build a Message with sensible defaults; msg_id auto-increments per test."""
    counter = iter(range(1, 10_000))

    def _make(sender, role, ts, text, **kw):
        return Message(
            group_id="g",
            msg_id=f"g_{next(counter):05d}",
            sender=sender,
            role=role,
            timestamp=datetime.fromisoformat(ts),
            text=text,
            **kw,
        )

    return _make
