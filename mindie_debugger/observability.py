"""Optional zero-dependency diagnostic bridge for `mindie-debugger`.

[Role]
  Records confirmed internal `mindie_debugger` exceptions and deterministic
  schema contract violations via `mindie_diagnostics` when installed locally,
  while staying a safe no-op inside remote NPU containers where
  `mindie_diagnostics` is absent.

[Invariants]
  * Imports no third-party modules at import time; `mindie_diagnostics` is
    resolved lazily inside `try ... except Exception:`.
  * Reports an unhandled exception only when its chain has no user-target or
    operational marker and its innermost traceback frame is owned by
    `mindie_debugger.*` (never user `--script` / `--callable` faults, missing
    `torch_npu`, or `OSError` / `TimeoutError`).

[Timeouts & Locks]
  Bounded local outbox write only; acquires no process locks.

[Side-Effect Policy]
  Writes sanitized diagnostic metadata to `mindie_diagnostics` if available;
  emits a single `mindie-diagnostic-incident: <id>` line on stderr only when an
  unhandled exception is about to propagate out of the CLI process.
"""
from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

_SOURCE_ROOT = Path(__file__).resolve().parent
_INTERNAL_TYPES = (RuntimeError, KeyError, AttributeError, IndexError, AssertionError, ZeroDivisionError)


class _DebuggerFailureMarker:
    __slots__ = ("result",)

    def __init__(self, result: Any) -> None:
        self.result = result


def mark_user_target_error(exc: BaseException) -> BaseException:
    try:
        exc._mindie_user_target_error = True  # type: ignore[attr-defined]
    except Exception:
        pass
    return exc


def call_user_target(fn: Any, *args: Any, **kwargs: Any) -> Any:
    try:
        return fn(*args, **kwargs)
    except BaseException as exc:
        mark_user_target_error(exc)
        raise


def is_owned_debugger_exception(exc: BaseException) -> bool:
    """True only when an internal exception terminates inside `mindie_debugger.*`."""
    seen: set[int] = set()
    pending: list[BaseException | None] = [exc]
    while pending and len(seen) < 32:
        cur = pending.pop()
        if cur is None or id(cur) in seen:
            continue
        seen.add(id(cur))
        if getattr(cur, "_mindie_user_target_error", False):
            return False
        if isinstance(cur, (OSError, TimeoutError, ValueError, ImportError, KeyboardInterrupt, SystemExit, GeneratorExit)):
            return False
        if getattr(cur, "__cause__", None) is not None:
            pending.append(cur.__cause__)
        if getattr(cur, "__context__", None) is not None and not getattr(cur, "__suppress_context__", False):
            pending.append(cur.__context__)
    if pending or type(exc) not in _INTERNAL_TYPES:
        return False
    tb = exc.__traceback__
    last_mod = ""
    steps = 0
    while tb is not None and steps < 4096:
        last_mod = str(tb.tb_frame.f_globals.get("__name__") or "")
        tb = tb.tb_next
        steps += 1
    if tb is not None:
        return False
    return last_mod == "mindie_debugger" or last_mod.startswith("mindie_debugger.")


def record_debugger_failure(
    operation: str,
    *,
    stage: str = "cli_dispatch",
    category: str = "internal_exception",
    exception: BaseException | None = None,
    reason_code: str | None = None,
    contract_name: str | None = None,
) -> dict[str, Any] | None:
    if exception is not None:
        seen: set[int] = set()
        pending: list[BaseException | None] = [exception]
        while pending and len(seen) < 32:
            cur = pending.pop(0)
            if cur is None or id(cur) in seen:
                continue
            seen.add(id(cur))
            marker = getattr(cur, "_mindie_debugger_diag", None)
            if type(marker) is _DebuggerFailureMarker:
                try:
                    exception._mindie_debugger_diag = marker  # type: ignore[attr-defined]
                except Exception:
                    pass
                return marker.result
            if getattr(cur, "__cause__", None) is not None:
                pending.append(cur.__cause__)
            if getattr(cur, "__context__", None) is not None and not getattr(cur, "__suppress_context__", False):
                pending.append(cur.__context__)
    try:
        from mindie_diagnostics.integration import record_failure
        base_kwargs: dict[str, Any] = {
            "component": "mindie-debugger",
            "operation": operation,
            "stage": stage,
            "category": category,
            "exception": exception,
        }
        v2_kwargs: dict[str, Any] = {"source_root": _SOURCE_ROOT}
        if reason_code is not None:
            v2_kwargs["reason_code"] = reason_code
        if contract_name is not None:
            v2_kwargs["contract_name"] = contract_name
        try:
            result = record_failure(**base_kwargs, **v2_kwargs)
        except TypeError:
            result = record_failure(**base_kwargs)
    except Exception:
        return None
    if exception is not None and isinstance(result, dict):
        try:
            exception._mindie_debugger_diag = _DebuggerFailureMarker(result)
        except Exception:
            pass
    return result if isinstance(result, dict) else None


def emit_incident_stderr(ref: dict[str, Any] | None) -> None:
    inc = ref.get("incident_id") if isinstance(ref, dict) else None
    if isinstance(inc, str) and len(inc) == 32 and all(c in "0123456789abcdef" for c in inc):
        sys.stderr.write(f"mindie-diagnostic-incident: {inc}\n")
        sys.stderr.flush()


def verify_triage_payload_contract(payload: Any, *, operation: str = "debugger.triage") -> bool:
    """Emit a contract_violation diagnostic and return False if `triage_npu_fault` breaks schema."""
    if (
        not isinstance(payload, dict)
        or not isinstance(payload.get("fault_domain"), str)
        or not isinstance(payload.get("error_codes"), list)
        or not isinstance(payload.get("decoded_codes"), list)
        or not isinstance(payload.get("recommended_actions"), list)
    ):
        record_debugger_failure(
            operation,
            stage="triage_contract",
            category="contract_violation",
            contract_name="triage_payload_v1",
            reason_code="invalid_triage_schema",
        )
        return False
    return True
