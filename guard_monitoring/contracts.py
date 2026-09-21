from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Generic, Optional, TypeVar

T = TypeVar("T")


class ModuleStatus(str, Enum):
    """Execution state for one monitoring module.

    OK        - module ran successfully for this frame.
    UNKNOWN   - module was not run because required input was unavailable.
    ERROR     - module attempted to run and raised an exception.
    DISABLED  - module is intentionally disabled by configuration.
    """

    OK = "ok"
    UNKNOWN = "unknown"
    ERROR = "error"
    DISABLED = "disabled"


@dataclass
class ModuleResult(Generic[T]):
    status: ModuleStatus
    value: Optional[T] = None
    error: Optional[str] = None
    detail: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.status is ModuleStatus.OK

    def to_dict(self) -> dict:
        data = asdict(self)
        data["status"] = self.status.value
        return data


def ok(value: T, detail: str | None = None) -> ModuleResult[T]:
    return ModuleResult(status=ModuleStatus.OK, value=value, detail=detail)


def unknown(detail: str) -> ModuleResult:
    return ModuleResult(status=ModuleStatus.UNKNOWN, value=None, detail=detail)


def disabled(detail: str = "disabled_by_config") -> ModuleResult:
    return ModuleResult(status=ModuleStatus.DISABLED, value=None, detail=detail)


def error_result(exc: Exception, detail: str | None = None) -> ModuleResult:
    return ModuleResult(
        status=ModuleStatus.ERROR,
        value=None,
        error=type(exc).__name__,
        detail=detail or str(exc),
    )
