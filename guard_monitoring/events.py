from __future__ import annotations

import json
from pathlib import Path


class JsonlWriter:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.path.open("w", encoding="utf-8")

    def write(self, obj: dict):
        self.file.write(json.dumps(obj, ensure_ascii=False, default=_json_default) + "\n")
        self.file.flush()

    def close(self):
        if not self.file.closed:
            self.file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


def _json_default(value):
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "__dict__"):
        return value.__dict__
    return str(value)
