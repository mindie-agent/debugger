from __future__ import annotations

import io
import json
import sys
import types
from pathlib import Path

import pytest

from mindie_debugger import cli
from mindie_debugger.precision import call_state, npu_watch


class _NonTtyCp936Stdin:
    def __init__(self, raw: bytes) -> None:
        self.buffer = io.BytesIO(raw)

    def isatty(self) -> bool:
        return False

    def read(self) -> str:
        # Simulate Windows CP936 TextIOWrapper failing on UTF-8 bytes illegal in GBK.
        return self.buffer.getvalue().decode("cp936")


def test_triage_stdin_handles_cp936_utf8_bytes_and_none_stdin(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("MINDIE_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("MINDIE_DIAGNOSTICS_ROOT", str(tmp_path / "diag"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "home"))

    # UTF-8 em-dash + invalid GBK byte 0xFF inside CANN plog payload
    plog_bytes = (
        b"[ERROR] RUNTIME(1234,python3):2026-11-14-10:00:00.000.000 "
        b"AICore fault 507015 \xe2\x80\x94 kernel launch failed \xff\n"
    )
    monkeypatch.setattr(sys, "stdin", _NonTtyCp936Stdin(plog_bytes))
    rc = cli.main(["triage"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["scanned_files"] == ["<stdin>"]

    # Windowless pythonw.exe sets sys.stdin = None
    monkeypatch.setattr(sys, "stdin", None)
    monkeypatch.setenv("ASCEND_PROCESS_LOG_PATH", str(tmp_path / "missing-ascend-log"))
    rc_none = cli.main(["triage"])
    assert rc_none == 1
    err_payload = json.loads(capsys.readouterr().out)
    assert err_payload["reason"] == "no_log_input"


def test_timeout_error_not_swallowed_as_oserror(monkeypatch: pytest.MonkeyPatch) -> None:
    def raise_timeout(*_args, **_kwargs):
        raise TimeoutError("deadline_exceeded")

    fake_dist = types.SimpleNamespace(
        is_initialized=raise_timeout,
        get_rank=raise_timeout,
        get_world_size=raise_timeout,
    )
    fake_backend = types.SimpleNamespace(
        is_initialized=raise_timeout,
        current_stream=raise_timeout,
    )
    fake_torch = types.SimpleNamespace(distributed=fake_dist, npu=fake_backend)

    with pytest.raises(TimeoutError):
        call_state._distributed_initialized(fake_torch)
    with pytest.raises(TimeoutError):
        call_state._backend_initialized(fake_torch, "npu")
    with pytest.raises(TimeoutError):
        call_state._record_collective_group(fake_dist, "all_reduce", None, [])

    monkeypatch.setattr(npu_watch, "_require_torch", lambda: fake_torch)
    with pytest.raises(TimeoutError):
        npu_watch._current_stream(types.SimpleNamespace(type="npu"))
