"""CLI error-honesty regressions for mindie-debugger (#332 debugger checklist).

[Role]
  Pin that local argument errors, log I/O failures, and oversized traces stay
  visible: non-zero exit, explicit JSON error, never a fake remote-NPU success
  or a clean-log report.

[Invariants]
  * Path typos and unrelated ModuleNotFoundError are not classified as missing NPU.
  * Log stat/read/stdin OSError exits 1; TimeoutError is re-raised, never "".
  * compare-traces / watch reject JSON files above the 50 MiB cap.

[Timeouts & Locks]
  Pure in-process CLI calls; no sleeps, subprocesses, or locks.

[Side-Effect Policy]
  Writes only under pytest tmp_path; HOME / MINDIE_* isolated by conftest.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

from mindie_debugger import cli


def _stdout_json(capsys) -> dict:
    return json.loads(capsys.readouterr().out)


def test_npu_env_missing_matcher_rejects_local_errors():
    requests_exc = ModuleNotFoundError("No module named 'requests'", name="requests")
    path_exc = FileNotFoundError(2, "No such file", "/tmp/mindie/ascend/npu/vllm/cann.log")
    value_exc = ValueError("invalid mindie/ascend/npu json")
    torch_exc = ModuleNotFoundError("No module named 'torch_npu'", name="torch_npu")
    so_exc = OSError("libhccl.so: cannot open shared object file")

    assert cli._is_npu_env_missing_error(requests_exc) is False
    assert cli._is_npu_env_missing_error(path_exc) is False
    assert cli._is_npu_env_missing_error(value_exc) is False
    assert cli._is_npu_env_missing_error(torch_exc) is True
    assert cli._is_npu_env_missing_error(so_exc) is True


def test_precision_missing_local_module_is_not_remote_npu(capsys):
    rc = cli.main(["precision", "check-call", "--callable", "definitely_not_installed_mod:fn"])
    payload = _stdout_json(capsys)
    assert rc == 1
    assert payload["status"] == "error"
    assert payload["error_type"] == "ModuleNotFoundError"
    assert payload["status"] != "remote_npu_execution_required"


def test_precision_missing_torch_npu_exits_nonzero_with_optional_guide(monkeypatch, capsys):
    def boom(_spec):
        raise ModuleNotFoundError("No module named 'torch_npu'", name="torch_npu")

    monkeypatch.setattr(cli, "_import_callable", boom)
    rc = cli.main(["precision", "check-call", "--callable", "mod:fn"])
    payload = _stdout_json(capsys)
    assert rc == 1
    assert payload["status"] == "remote_npu_execution_required"
    assert "torch_npu" in payload["local_missing_environment"]
    summary = payload["remote_npu_execution_guide"]["summary"].lower()
    assert "scp" in summary or "ssh" in summary


def test_precision_missing_capsule_with_npu_tokens_in_path_is_local_error(tmp_path, capsys):
    missing = tmp_path / "npu_ascend_mindie_vllm.pt"
    rc = cli.main(["precision", "inspect-capsule", str(missing)])
    payload = _stdout_json(capsys)
    assert rc == 1
    assert payload["status"] != "remote_npu_execution_required"
    assert payload["status"] == "error"


def test_aclgraph_missing_script_and_typo_path_are_local_errors(tmp_path, capsys):
    rc = cli.main(["aclgraph", "guard"])
    payload = _stdout_json(capsys)
    assert rc == 1
    assert payload["status"] == "error"
    assert payload["reason"] == "script_or_callable_required"

    rc = cli.main(["aclgraph", "guard", "--script", str(tmp_path / "npu_missing.py")])
    payload = _stdout_json(capsys)
    assert rc == 1
    assert payload["status"] == "error"
    assert payload["reason"] == "script_path_not_found"


def test_triage_without_input_is_error_not_remote_success(capsys, monkeypatch):
    class EmptyStdin:
        def isatty(self):
            return False

        def read(self):
            return ""

    monkeypatch.delenv("ASCEND_PROCESS_LOG_PATH", raising=False)
    monkeypatch.setattr(sys, "stdin", EmptyStdin())
    rc = cli.main(["triage"])
    payload = _stdout_json(capsys)
    assert rc == 1
    assert payload["status"] == "error"
    assert payload["reason"] == "no_log_input"
    assert "remote_npu_execution_guide" not in payload


def test_readable_empty_log_is_honest_clean(tmp_path, capsys):
    log = tmp_path / "empty.log"
    log.write_text("no errors in this plog\n", encoding="utf-8")
    rc = cli.main(["triage", "--file", str(log)])
    payload = _stdout_json(capsys)
    assert rc == 0
    assert payload["fault_domain"] == "unknown"
    assert payload["scanned_file_count"] == 1


def test_read_bounded_log_text_does_not_hide_oserror_or_timeout(tmp_path, monkeypatch):
    log = tmp_path / "blocked.log"
    log.write_text("[ERROR] RUNTIME ErrCode=507015\n", encoding="utf-8")
    monkeypatch.setattr(Path, "stat", lambda self, *a, **k: (_ for _ in ()).throw(
        PermissionError(13, "Permission denied", os.fspath(self))
    ))
    with pytest.raises(PermissionError):
        cli._read_bounded_log_text(log)

    monkeypatch.setattr(Path, "stat", lambda self, *a, **k: (_ for _ in ()).throw(
        TimeoutError("stat timed out")
    ))
    with pytest.raises(TimeoutError):
        cli._read_bounded_log_text(log)


def test_triage_unreadable_file_is_not_reported_clean(tmp_path, monkeypatch, capsys):
    log = tmp_path / "blocked.log"
    log.write_text("[ERROR] RUNTIME ErrCode=507015\n", encoding="utf-8")
    blocked = os.path.abspath(str(log))
    real_stat = Path.stat

    def selective(self, *args, **kwargs):
        if os.path.abspath(os.fspath(self)) == blocked:
            raise PermissionError(13, "Permission denied", blocked)
        return real_stat(self, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", selective)
    rc = cli.main(["triage", "--file", str(log)])
    payload = _stdout_json(capsys)
    assert rc == 1
    assert payload["status"] == "error"
    assert payload["reason"] == "log_read_failed"
    assert payload["error_type"] == "PermissionError"
    assert payload.get("fault_domain") is None
    assert "scanned_files" not in payload


def test_triage_directory_unreadable_member_is_not_reported_clean(tmp_path, monkeypatch, capsys):
    root = tmp_path / "logs"
    root.mkdir()
    (root / "ok.log").write_text("[ERROR] RUNTIME ErrCode=507015\n", encoding="utf-8")
    bad = root / "blocked.log"
    bad.write_text("should not become an empty scan\n", encoding="utf-8")
    blocked = os.path.abspath(str(bad))
    real_stat = Path.stat

    def selective(self, *args, **kwargs):
        if os.path.abspath(os.fspath(self)) == blocked:
            raise PermissionError(13, "Permission denied", blocked)
        return real_stat(self, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", selective)
    rc = cli.main(["triage", "--root", str(root)])
    payload = _stdout_json(capsys)
    assert rc == 1
    assert payload["status"] == "error"
    assert payload["reason"] == "log_read_failed"
    assert payload.get("fault_domain") is None
    assert "scanned_files" not in payload


def test_triage_stdin_oserror_exits_1_and_timeout_is_reraised(monkeypatch, capsys):
    class BoomStdin:
        def isatty(self):
            return False

        def read(self):
            raise OSError(5, "Input/output error")

    monkeypatch.setattr(sys, "stdin", BoomStdin())
    rc = cli.main(["triage"])
    payload = _stdout_json(capsys)
    assert rc == 1
    assert payload["status"] == "error"
    assert payload["reason"] == "log_read_failed"

    class TimeoutStdin:
        def isatty(self):
            return False

        def read(self):
            raise TimeoutError("stdin timed out")

    monkeypatch.setattr(sys, "stdin", TimeoutStdin())
    with pytest.raises(TimeoutError):
        cli.main(["triage"])


def test_read_json_rejects_oversized_trace_and_allows_limit(tmp_path, monkeypatch):
    payload = '{"metrics":[]}'
    monkeypatch.setattr(cli, "_JSON_TRACE_MAX_BYTES", len(payload.encode()))
    at_limit = tmp_path / "ok.json"
    at_limit.write_text(payload, encoding="utf-8")
    assert cli._read_json(at_limit) == {"metrics": []}

    monkeypatch.setattr(cli, "_JSON_TRACE_MAX_BYTES", 32)
    big = tmp_path / "big.json"
    big.write_text(payload + (" " * 64), encoding="utf-8")
    with pytest.raises(ValueError, match="exceeding") as info:
        cli._read_json(big)
    message = str(info.value)
    assert str(big) in message
    assert str(big.stat().st_size) in message


def test_compare_traces_and_watch_reject_oversized_json(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "_JSON_TRACE_MAX_BYTES", 32)
    big = tmp_path / "big.json"
    small = tmp_path / "small.json"
    big.write_text('{"metrics":[]}' + (" " * 64), encoding="utf-8")
    small.write_text('{"metrics":[]}', encoding="utf-8")
    rc = cli.main(["precision", "compare-traces", str(big), str(small)])
    payload = _stdout_json(capsys)
    assert rc == 1
    assert payload["status"] == "error"
    assert payload["error_type"] == "ValueError"
    assert str(big) in payload["error"]

    rc = cli.main(["precision", "watch", str(big)])
    payload = _stdout_json(capsys)
    assert rc == 1
    assert payload["status"] == "error"
    assert payload["error_type"] == "ValueError"


def test_remote_guide_remains_explicit_success(capsys):
    rc = cli.main(["remote-guide"])
    payload = _stdout_json(capsys)
    assert rc == 0
    assert payload["status"] == "ok"
    summary = payload["remote_npu_execution_guide"]["summary"].lower()
    assert "scp" in summary or "ssh" in summary
