"""CLI error-honesty regressions for mindie-debugger (#332 debugger checklist).

[Role]
  Pin that local argument errors, log I/O failures, and oversized traces stay
  visible: non-zero exit, explicit JSON error, never a fake remote-NPU success
  or a clean-log report.

[Invariants]
  * Path typos and unrelated ModuleNotFoundError are not classified as missing NPU.
  * Log stat/read/stdin OSError exits 1; TimeoutError is re-raised, never "".
  * Directory walk yields file symlinks to regular logs; metadata I/O still fails.
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
    so_path_exc = FileNotFoundError(2, "No such file", "/tmp/notes/libhccl.so.bak")
    value_exc = ValueError("invalid mindie/ascend/npu json")
    torch_exc = ModuleNotFoundError("No module named 'torch_npu'", name="torch_npu")
    so_exc = ImportError("libhccl.so: cannot open shared object file")

    assert cli._is_npu_env_missing_error(requests_exc) is False
    assert cli._is_npu_env_missing_error(path_exc) is False
    assert cli._is_npu_env_missing_error(so_path_exc) is False
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
    missing = tmp_path / "libhccl.so"
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


def test_aclgraph_guard_and_inspect_target_exceptions_never_report_ok_or_clean(monkeypatch, capsys):
    class _NoopGuard:
        def __init__(self, *, strict=False):
            self.violations = []

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def report(self):
            return {
                "status": "clean" if not self.violations else "capture_hazard",
                "violation_count": len(self.violations),
                "first_violation": None,
                "violations": list(self.violations),
            }

    class _NoopInspector:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def report(self):
            return {
                "status": "ok",
                "capture_guard_violations": [],
                "address_violations": [],
                "layer_diffs": [],
                "first_divergent_layer": None,
            }

    import mindie_debugger.aclgraph as aclgraph_mod

    monkeypatch.setattr(aclgraph_mod, "ACLGraphCaptureGuard", _NoopGuard)
    monkeypatch.setattr(aclgraph_mod, "ACLGraphVllmInspector", _NoopInspector)

    # math:sqrt called with 0 args raises TypeError; JSON status must be "error", never "ok"/"clean".
    rc = cli.main(["aclgraph", "inspect", "--callable", "math:sqrt"])
    payload = _stdout_json(capsys)
    assert rc == 1
    assert payload["status"] == "error"
    assert "TypeError" in payload["exception"]

    rc = cli.main(["aclgraph", "guard", "--callable", "math:sqrt"])
    payload = _stdout_json(capsys)
    assert rc == 1
    assert payload["status"] == "error"
    assert "TypeError" in payload["exception"]

    # Clean zero-arg callable returns exit 0 with honest clean/ok status.
    rc = cli.main(["aclgraph", "guard", "--callable", "sys:getdefaultencoding"])
    payload = _stdout_json(capsys)
    assert rc == 0
    assert payload["status"] == "clean"


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


def test_triage_direntry_is_dir_oserror_is_not_empty_success(tmp_path, monkeypatch, capsys):
    root = tmp_path / "logs"
    nested = root / "rank0"
    nested.mkdir(parents=True)
    (nested / "plog-1.log").write_text("[ERROR] RUNTIME ErrCode=507015\n", encoding="utf-8")
    blocked = os.path.abspath(str(nested))
    real_scandir = os.scandir

    class EntryProxy:
        def __init__(self, entry):
            self._entry = entry
            self.path = entry.path
            self.name = entry.name

        def is_dir(self, *args, **kwargs):
            if os.path.abspath(self.path) == blocked:
                raise PermissionError(13, "Permission denied", self.path)
            return self._entry.is_dir(*args, **kwargs)

        def is_file(self, *args, **kwargs):
            return self._entry.is_file(*args, **kwargs)

        def is_symlink(self):
            return self._entry.is_symlink()

    class ScandirProxy:
        def __init__(self, inner):
            self._inner = inner

        def __enter__(self):
            self._inner.__enter__()
            return self

        def __exit__(self, *args):
            return self._inner.__exit__(*args)

        def __iter__(self):
            for entry in self._inner:
                yield EntryProxy(entry)

    monkeypatch.setattr(os, "scandir", lambda path: ScandirProxy(real_scandir(path)))
    rc = cli.main(["triage", "--root", str(root)])
    payload = _stdout_json(capsys)
    assert rc == 1
    assert payload["status"] == "error"
    assert payload["reason"] == "log_read_failed"
    assert payload["error_type"] == "PermissionError"
    assert payload.get("fault_domain") is None
    assert "scanned_files" not in payload


def _symlink_or_skip(link: Path, target: Path) -> None:
    try:
        link.symlink_to(target, target_is_directory=target.is_dir())
    except OSError as exc:
        pytest.skip(f"cannot create symlink: {exc}")


def test_triage_symlink_to_regular_log_is_not_empty_success(tmp_path, capsys):
    root = tmp_path / "logs"
    root.mkdir()
    real = tmp_path / "outside" / "device-0.log"
    real.parent.mkdir()
    real.write_text("[ERROR] RUNTIME ErrCode=507015\n", encoding="utf-8")
    link = root / "plog-1.log"
    _symlink_or_skip(link, real)
    found = list(cli._iter_regular_files(root))
    assert found == [link]
    rc = cli.main(["triage", "--root", str(root)])
    payload = _stdout_json(capsys)
    assert rc == 0
    assert payload.get("status") != "error"
    assert payload["scanned_file_count"] == 1
    assert str(link) in payload["scanned_files"][0]
    assert "507015" in payload["error_codes"]


def test_iter_regular_files_skips_directory_symlink_without_loop(tmp_path):
    root = tmp_path / "logs"
    nested = root / "rank0"
    nested.mkdir(parents=True)
    log = nested / "plog-1.log"
    log.write_text("[ERROR] RUNTIME ErrCode=507015\n", encoding="utf-8")
    _symlink_or_skip(root / "loop", root)
    found = list(cli._iter_regular_files(root))
    assert found == [log]


def test_triage_nested_unreadable_dir_is_not_empty_success(tmp_path, monkeypatch, capsys):
    root = tmp_path / "logs"
    nested = root / "rank0"
    nested.mkdir(parents=True)
    (nested / "plog-1.log").write_text("[ERROR] RUNTIME ErrCode=507015\n", encoding="utf-8")
    blocked = os.path.abspath(str(nested))
    real_scandir = os.scandir

    def selective(path):
        if os.path.abspath(path) == blocked:
            raise PermissionError(13, "Permission denied", str(path))
        return real_scandir(path)

    monkeypatch.setattr(os, "scandir", selective)
    rc = cli.main(["triage", "--root", str(root)])
    payload = _stdout_json(capsys)
    assert rc == 1
    assert payload["status"] == "error"
    assert payload["reason"] == "log_read_failed"
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
    with pytest.raises(ValueError, match="exceed") as info:
        cli._read_json(big)
    message = str(info.value)
    assert str(big) in message
    assert str(cli._JSON_TRACE_MAX_BYTES) in message


def test_read_json_enforces_limit_despite_stale_stat(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "_JSON_TRACE_MAX_BYTES", 32)
    path = tmp_path / "grow.json"
    path.write_text('{"metrics":[]}' + (" " * 64), encoding="utf-8")
    real_stat = Path.stat

    def tiny_stat(self, *args, **kwargs):
        if os.path.abspath(os.fspath(self)) == os.path.abspath(str(path)):
            return type("S", (), {"st_size": 8})()
        return real_stat(self, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", tiny_stat)
    with pytest.raises(ValueError, match="exceed"):
        cli._read_json(path)
    small = tmp_path / "small.json"
    small.write_text('{"metrics":[]}', encoding="utf-8")
    rc = cli.main(["precision", "compare-traces", str(path), str(small)])
    payload = _stdout_json(capsys)
    assert rc == 1
    assert payload["status"] == "error"
    assert payload["error_type"] == "ValueError"


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
