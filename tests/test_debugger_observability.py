"""Tier 0/1 tests for `mindie-debugger` Diagnostics 2.0 integration."""
from __future__ import annotations

import builtins
import importlib.util
import json
from pathlib import Path
import sys
import types
from typing import Any

import pytest

import mindie_debugger
from mindie_debugger import cli
from mindie_debugger.observability import (
    call_user_target,
    emit_incident_stderr,
    is_owned_debugger_exception,
    record_debugger_failure,
)


def _isolate_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    base = tmp_path.resolve()
    diag_root = base / "diag"
    cfg_file = base / "reporting.json"
    monkeypatch.setenv("HOME", str(base / "home"))
    monkeypatch.setenv("MINDIE_CONFIG_HOME", str(base / "config"))
    monkeypatch.setenv("MINDIE_DATA_HOME", str(base / "data"))
    monkeypatch.setenv("MINDIE_DIAGNOSTICS_ROOT", str(diag_root))
    monkeypatch.setenv("MINDIE_DIAGNOSTICS_CONFIG", str(cfg_file))
    (base / "home").mkdir(parents=True, exist_ok=True)
    diag_root.mkdir(parents=True, exist_ok=True)
    return diag_root, cfg_file


def _install_recorder_stub(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    recorded: list[dict[str, Any]] = []
    pkg = types.ModuleType("mindie_diagnostics")
    integ = types.ModuleType("mindie_diagnostics.integration")

    def fake_record_failure(**kwargs: Any) -> dict[str, Any]:
        recorded.append(dict(kwargs))
        return {"recorded": True, "logging_failed": False, "incident_id": "a" * 32}

    integ.record_failure = fake_record_failure  # type: ignore[attr-defined]
    pkg.integration = integ  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mindie_diagnostics", pkg)
    monkeypatch.setitem(sys.modules, "mindie_diagnostics.integration", integ)
    return recorded


def test_safe_noop_without_mindie_diagnostics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _isolate_env(tmp_path, monkeypatch)
    real_import = builtins.__import__

    def block_diag(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "mindie_diagnostics" or name.startswith("mindie_diagnostics."):
            raise ModuleNotFoundError("No module named 'mindie_diagnostics'")
        return real_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "mindie_diagnostics", raising=False)
    monkeypatch.delitem(sys.modules, "mindie_diagnostics.integration", raising=False)
    monkeypatch.setattr(builtins, "__import__", block_diag)

    import mindie_debugger.npu_fault_triage as nft

    monkeypatch.setattr(nft, "triage_npu_fault", lambda _text: {"broken": True})
    rc = cli.main(["triage", "--text", "sample"])
    assert rc == 1
    err_payload = json.loads(capsys.readouterr().out)
    assert err_payload["status"] == "error"
    assert err_payload["reason"] == "triage_contract_violation"


def test_normal_triage_emits_no_issue_and_broken_schema_emits_contract_violation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _isolate_env(tmp_path, monkeypatch)
    recorded = _install_recorder_stub(monkeypatch)

    rc = cli.main(["triage", "--text", "[ERROR] RUNTIME(100,200) 507015 aicore error"])
    assert rc == 0
    capsys.readouterr()
    assert recorded == []

    import mindie_debugger.npu_fault_triage as nft

    monkeypatch.setattr(nft, "triage_npu_fault", lambda _text: {"broken": True})
    rc2 = cli.main(["triage", "--text", "sample"])
    assert rc2 == 1
    err_payload = json.loads(capsys.readouterr().out)
    assert err_payload["status"] == "error"
    assert err_payload["reason"] == "triage_contract_violation"
    assert len(recorded) == 1
    ev = recorded[0]
    assert ev["component"] == "mindie-debugger"
    assert ev["operation"] == "debugger.triage"
    assert ev["stage"] == "triage_contract"
    assert ev["category"] == "contract_violation"
    assert ev["contract_name"] == "triage_payload_v1"
    assert ev["reason_code"] == "invalid_triage_schema"
    assert ev["source_root"] == Path(mindie_debugger.__file__).resolve().parent

    err = RuntimeError("internal bug")
    first = record_debugger_failure("debugger.triage", exception=err, reason_code="boom")
    second = record_debugger_failure("debugger.triage", exception=err, reason_code="boom")
    assert first == second
    assert len(recorded) == 2

    emit_incident_stderr(first)
    assert capsys.readouterr().err == f"mindie-diagnostic-incident: {'a' * 32}\n"


@pytest.mark.skipif(
    importlib.util.find_spec("mindie_diagnostics") is None,
    reason="mindie_diagnostics optional in standalone debugger CI",
)
def test_end_to_end_ingestion_when_diagnostics_installed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from mindie_diagnostics.ingestion import ingest
    from mindie_diagnostics.integration import configure_reporting
    from mindie_diagnostics.outbox import Outbox

    diag_root, cfg_file = _isolate_env(tmp_path, monkeypatch)
    configure_reporting(True, roots=[str(diag_root)], config=cfg_file)
    queue = Outbox(tmp_path.resolve() / "queue.db")

    import mindie_debugger.npu_fault_triage as nft

    monkeypatch.setattr(nft, "triage_npu_fault", lambda _text: {"broken": True})
    assert cli.main(["triage", "--text", "sample"]) == 1
    capsys.readouterr()
    assert ingest(diag_root, queue, config=cfg_file)["enqueued"] == 1
    ev = queue.claim()["payload"]["events"][-1]
    assert ev["component"] == "mindie-debugger"
    assert ev["attributes"]["category"] == "contract_violation"
    assert ev["attributes"]["contract_name"] == "triage_payload_v1"
    assert ev["attributes"]["reason_code"] == "invalid_triage_schema"
    assert all(
        not f["module"].startswith("mindie_debugger.mindie_debugger.")
        for f in ev["attributes"]["stack_frames"]
    )


def test_invalid_user_callable_is_caller_error_not_internal_defect(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _isolate_env(tmp_path, monkeypatch)
    recorded = _install_recorder_stub(monkeypatch)

    rc = cli.main(["precision", "check-call", "--callable", "math:nonexistent_user_callable"])
    assert rc == 1
    capsys.readouterr()
    assert recorded == []

    mod_path = tmp_path / "user_mod_target.py"
    mod_path.write_text(
        "class DummyModule:\n"
        "    def __call__(self):\n"
        "        raise RuntimeError('user forward failure')\n"
        "target = DummyModule()\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))
    imported = cli._import_callable("user_mod_target:target")
    assert type(imported).__name__ == "DummyModule"
    with pytest.raises(RuntimeError) as exc_info:
        call_user_target(imported)
    assert is_owned_debugger_exception(exc_info.value) is False

    cur: BaseException = PermissionError("root")
    for idx in range(40):
        nxt = RuntimeError(f"w{idx}")
        nxt.__cause__ = cur
        cur = nxt
    assert is_owned_debugger_exception(cur) is False
