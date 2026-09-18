from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import Callable, TypeVar

from .contracts import ModuleResult, ModuleStatus, error_result, ok

logger = logging.getLogger(__name__)
T = TypeVar("T")


@dataclass
class _HealthEntry:
    status: str = ModuleStatus.UNKNOWN.value
    calls: int = 0
    successes: int = 0
    errors: int = 0
    unknowns: int = 0
    disabled: int = 0
    last_error: str | None = None
    last_detail: str | None = None


class ModuleHealthRegistry:
    """In-memory health counters used for debugging and UI reporting."""

    def __init__(self):
        self._entries: dict[str, _HealthEntry] = defaultdict(_HealthEntry)

    def record(self, name: str, result: ModuleResult) -> None:
        entry = self._entries[name]
        entry.calls += 1
        entry.status = result.status.value
        entry.last_detail = result.detail
        if result.status is ModuleStatus.OK:
            entry.successes += 1
            entry.last_error = None
        elif result.status is ModuleStatus.ERROR:
            entry.errors += 1
            entry.last_error = result.error
        elif result.status is ModuleStatus.UNKNOWN:
            entry.unknowns += 1
        elif result.status is ModuleStatus.DISABLED:
            entry.disabled += 1

    def snapshot(self) -> dict[str, dict]:
        return {name: asdict(entry) for name, entry in sorted(self._entries.items())}



def safe_run(
    module_name: str,
    fn: Callable[..., T],
    *args,
    health: ModuleHealthRegistry | None = None,
    **kwargs,
) -> ModuleResult[T]:
    """Run one module without letting its exception crash unrelated modules."""

    try:
        result = ok(fn(*args, **kwargs))
    except Exception as exc:  # isolation boundary: log full stack, return compact state
        logger.exception("Guard monitoring module failed: %s", module_name)
        result = error_result(exc)

    if health is not None:
        health.record(module_name, result)
    return result
