from __future__ import annotations

import json
import logging
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any

correlation: ContextVar[dict[str, Any]] = ContextVar("correlation", default={})


class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        entry: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        ctx = correlation.get()
        if ctx:
            entry.update(ctx)

        data = getattr(record, "data", None)
        if isinstance(data, dict):
            entry.update(data)

        if record.exc_info and record.exc_info[0] is not None:
            entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(entry, default=str)


def configure_logging(level: int = logging.INFO) -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(JSONFormatter())

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()
    root.addHandler(handler)

    for name in ("uvicorn.access", "httpx", "httpcore"):
        logging.getLogger(name).setLevel(logging.WARNING)
