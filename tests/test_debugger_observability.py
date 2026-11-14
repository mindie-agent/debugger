"""Tier 1 tests for `mindie-debugger` Diagnostics 2.0 integration."""
from __future__ import annotations

from pathlib import Path

import pytest

from mindie_debugger import cli
from mindie_diagnostics.ingestion import ingest
from mindie_diagnostics.integration import configure_reporting
from mindie_diagnostics.outbox import Outbox


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
    configure_reporting(True, roots=[str(diag_root)], config=cfg_file)
    return diag_root, cfg_file


def test_normal_triage_emits_no_issue_and_broken_schema_emits_contract_violation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    diag_root, cfg_file = _isolate_env(tmp_path, monkeypatch)
    queue = Outbox(tmp_path.resolve() / "queue.db")

    rc = cli.main(["triage", "--text", "[ERROR] RUNTIME(100,200) 507015 aicore error"])
    assert rc == 0
    capsys.readouterr()
    assert ingest(diag_root, queue, config=cfg_file)["enqueued"] == 0

    import json
    import mindie_debugger.npu_fault_triage as nft
    monkeypatch.setattr(nft, "triage_npu_fault", lambda _text: {"broken": True})
    rc2 = cli.main(["triage", "--text", "sample"])
    assert rc2 == 1
    err_payload = json.loads(capsys.readouterr().out)
    assert err_payload["status"] == "error"
    assert err_payload["reason"] == "triage_contract_violation"
    assert ingest(diag_root, queue, config=cfg_file)["enqueued"] == 1
    ev = queue.claim()["payload"]["events"][-1]
    assert ev["component"] == "mindie-debugger"
    assert ev["attributes"]["category"] == "contract_violation"
    assert ev["attributes"]["contract_name"] == "triage_payload_v1"
    assert ev["attributes"]["reason_code"] == "invalid_triage_schema"
    assert all(not f["module"].startswith("mindie_debugger.mindie_debugger.") for f in ev["attributes"]["stack_frames"])


def test_invalid_user_callable_is_caller_error_not_internal_defect(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    diag_root, cfg_file = _isolate_env(tmp_path, monkeypatch)
    queue = Outbox(tmp_path.resolve() / "queue.db")

    rc = cli.main(["precision", "check-call", "--callable", "math:nonexistent_user_callable"])
    assert rc == 1
    capsys.readouterr()
    assert ingest(diag_root, queue, config=cfg_file)["enqueued"] == 0

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
    from mindie_debugger.observability import call_user_target, is_owned_debugger_exception
    with pytest.raises(RuntimeError) as exc_info:
        call_user_target(imported)
    assert is_owned_debugger_exception(exc_info.value) is False

    cur: BaseException = PermissionError("root")
    for idx in range(40):
        nxt = RuntimeError(f"w{idx}")
        nxt.__cause__ = cur
        cur = nxt
    assert is_owned_debugger_exception(cur) is False
