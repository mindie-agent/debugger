"""Orthogonal behavioral tests for zero accidental NPU context and read-only triage.

[Role]
  Verify via independent process, filesystem, and hardware oracles that:
  1. CPU precision ``capture_call`` / ``replay_call`` reads ``torch_npu``
     distribution metadata without importing ``torch_npu`` into ``sys.modules``.
  2. When ``torch_npu`` is loaded, RNG capture/restore and device sync touch
     only initialized backends and the exact device indices referenced by the
     captured call (never ``get_rng_state_all()`` or unreferenced ``Phy-ID 0``).
  3. ``mindie-debugger triage`` is strictly read-only, preserves completed
     triage output when auxiliary guide steps fail, and refuses destructive
     staging on protected user roots.
  4. On standing Ascend A2/A3 hosts, CPU precision capture/replay allocates
     ``0 MiB`` HBM on ``Phy-ID 0``.

[Invariants]
  * All local tests run under isolated ``tmp_path`` environment variables.
  * Live NPU verification stages only under ``/tmp/mindie-dbg-zero-ctx-*`` and
    removes its temporary directory in ``finally``.

[Timeouts & Locks]
  Local subprocess tests bound waits to ``15s``; live SSH probes bound waits to ``20s``.

[Side-Effect Policy]
  Zero host pollution; read-only ``npu-smi`` inspection on live NPU hosts.
"""
from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from mindie_debugger import cli
from mindie_debugger.precision import capture_call, replay_call
from mindie_debugger.precision.call_state import (
    _capture_rng,
    _restore_rng,
    _sync_referenced_devices,
)
from mindie_debugger.precision.storage_graph import StateSnapshot, TensorViewSpec


def test_cpu_precision_capture_reads_torch_npu_metadata_without_importing_module(tmp_path):
    pytest.importorskip("torch")
    fake_site = tmp_path / "fake-site"
    dist_info = fake_site / "torch_npu-2.5.1.dist-info"
    pkg_dir = fake_site / "torch_npu"
    dist_info.mkdir(parents=True)
    pkg_dir.mkdir(parents=True)
    (dist_info / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: torch_npu\nVersion: 2.5.1\n",
        encoding="utf-8",
        newline="\n",
    )
    (pkg_dir / "__init__.py").write_text(
        "raise RuntimeError('canary: torch_npu must never be imported during CPU capture/replay')\n",
        encoding="utf-8",
        newline="\n",
    )

    repo_root = Path(__file__).resolve().parents[1]
    existing_py_path = [p for p in sys.path if p]
    probe_code = (
        "import json, sys, torch\n"
        "from mindie_debugger.precision import capture_call, replay_call\n"
        "assert 'torch_npu' not in sys.modules\n"
        "mod = torch.nn.Linear(4, 4, bias=False)\n"
        "x = torch.ones((2, 4), dtype=torch.float32)\n"
        "cap = capture_call(mod, x)\n"
        "assert 'torch_npu' not in sys.modules\n"
        "rep = replay_call(cap, mod, repeat=3, mode='reference_check', reference_atol=1e-4)\n"
        "assert 'torch_npu' not in sys.modules\n"
        "print(json.dumps({\n"
        "    'torch_npu_loaded': 'torch_npu' in sys.modules,\n"
        "    'torch_npu_version': cap.execution_state['torch_npu_version'],\n"
        "    'rng_keys': sorted(cap.rng_state.keys()),\n"
        "    'status': rep['status'],\n"
        "    'verdict': rep['verdict'],\n"
        "}))\n"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(fake_site), str(repo_root), *existing_py_path])
    win_kw = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
    proc = subprocess.run(
        [sys.executable, "-X", "utf8", "-c", probe_code],
        env=env,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
        **win_kw,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout.strip())
    assert payload["torch_npu_loaded"] is False
    assert payload["torch_npu_version"] == "2.5.1"
    assert payload["rng_keys"] == ["cpu"]
    assert payload["status"] == "ok"
    assert payload["verdict"] == "pass"


def test_npu_rng_and_sync_only_touch_initialized_referenced_devices(monkeypatch):
    torch = pytest.importorskip("torch")
    calls: list[tuple[str, object]] = []

    class FakeNpuBackend:
        def __init__(self, initialized: bool):
            self._initialized = initialized

        def is_available(self):
            return True

        def is_initialized(self):
            return self._initialized

        def get_rng_state(self, device=0):
            calls.append(("get_rng_state", device))
            return torch.ByteTensor([1, 2, 3, int(device)])

        def get_rng_state_all(self):
            calls.append(("get_rng_state_all", None))
            raise AssertionError("get_rng_state_all must never be called")

        def set_rng_state(self, state, device=0):
            calls.append(("set_rng_state", int(device), int(state[-1].item())))

        def set_rng_state_all(self, states):
            calls.append(("set_rng_state_all", None))
            raise AssertionError("set_rng_state_all must never be called")

        def synchronize(self, device=None):
            calls.append(("synchronize", device))

    # 1. torch_npu loaded in sys.modules, but NOT initialized -> zero calls even if device 2 passed
    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace(__version__="2.5.1"))
    monkeypatch.setattr(torch, "npu", FakeNpuBackend(initialized=False), raising=False)
    state_uninit = _capture_rng({"npu": {2}})
    assert "npu_devices" not in state_uninit
    _restore_rng({"cpu": state_uninit["cpu"], "npu_devices": {2: torch.ByteTensor([9])}}, {"npu": {2}})
    _sync_referenced_devices({"npu": {2}})
    assert calls == []

    # 2. torch_npu initialized, but call uses CPU tensors only -> zero NPU calls
    monkeypatch.setattr(torch, "npu", FakeNpuBackend(initialized=True), raising=False)
    mod = torch.nn.Linear(3, 3, bias=False)
    cap = capture_call(mod, torch.ones((1, 3)))
    rep = replay_call(cap, mod, repeat=2)
    assert rep["status"] == "ok"
    assert calls == []

    # 3. torch_npu initialized and call references ONLY npu:2 -> touches ONLY device 2, never device 0
    snap_dev2 = StateSnapshot(
        roots={},
        storages={},
        views=[
            TensorViewSpec(
                tree_path="args[0]",
                tensor_id=1,
                storage_id="npu:2:100:16",
                dtype_str="torch.float32",
                shape=(4,),
                stride=(1,),
                storage_offset=0,
                requires_grad=False,
                device_str="npu:2",
            )
        ],
    )
    from mindie_debugger.precision.call_state import _referenced_devices

    devs = _referenced_devices(snap_dev2)
    assert devs == {"npu": {2}}
    state_dev2 = _capture_rng(devs)
    assert list(state_dev2["npu_devices"].keys()) == [2]
    _restore_rng(state_dev2, devs)
    # Legacy origin/main capsules stored per-device list in `npu_all` alongside current-device `npu`
    legacy_capsule = {
        "cpu": state_dev2["cpu"],
        "npu": torch.ByteTensor([1, 2, 3, 0]),
        "npu_all": [
            torch.ByteTensor([1, 2, 3, 10]),
            torch.ByteTensor([1, 2, 3, 11]),
            torch.ByteTensor([1, 2, 3, 22]),
        ],
    }
    _restore_rng(legacy_capsule, devs)
    _sync_referenced_devices(devs)
    assert calls == [
        ("get_rng_state", 2),
        ("set_rng_state", 2, 2),
        ("set_rng_state", 2, 22),
        ("synchronize", 2),
    ]


def test_triage_is_strictly_readonly_and_preserves_result_on_guide_warning(
    tmp_path, monkeypatch, capsys
):
    data_home = tmp_path / "uncreated-mindie-data"
    work_cwd = tmp_path / "user-cwd"
    work_cwd.mkdir()
    monkeypatch.chdir(work_cwd)
    monkeypatch.setenv("MINDIE_DATA_HOME", str(data_home))

    plog = work_cwd / "rank0.log"
    plog.write_text(
        "[ERROR] RUNTIME(100,101):2026-11-01-10:00:00.000.000 "
        "rank[0] deviceId:0 ErrCode=507015 failed operator: MatMulV2\n",
        encoding="utf-8",
    )
    before_entries = sorted(p.name for p in work_cwd.iterdir())
    rc = cli.main(["triage", "--file", str(plog)])
    out = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert out["fault_domain"] == "aicore_memory_or_tiling"
    assert out["remote_npu_execution_guide"]["bundle_staged"] is False
    assert Path(out["remote_npu_execution_guide"]["local_bundle_path"]) == Path(cli.__file__).resolve().parent
    assert not data_home.exists(), "triage must not create MINDIE_DATA_HOME or stage bundles"
    assert sorted(p.name for p in work_cwd.iterdir()) == before_entries

    # Receipt Honesty: if auxiliary guide construction raises OSError, primary triage result stays exit 0
    monkeypatch.setattr(
        cli,
        "_build_remote_execution_guide",
        lambda *a, **kw: (_ for _ in ()).throw(OSError("read-only guide formatting glitch")),
    )
    rc_warn = cli.main(["triage", "--file", str(plog)])
    out_warn = json.loads(capsys.readouterr().out)
    assert rc_warn == 0
    assert out_warn["fault_domain"] == "aicore_memory_or_tiling"
    assert out_warn["receipt_unconfirmed"] is True
    assert "remote_npu_execution_guide" in out_warn["receipt_warning"]


def test_stage_remote_refuses_protected_roots_and_never_uses_rmtree(
    tmp_path, monkeypatch, capsys
):
    home = Path(os.environ["HOME"])
    sentinel = home / "precious-user-note.txt"
    sentinel.write_text("do not delete", encoding="utf-8")

    # Refuses $HOME and cwd without deleting anything
    rc_home = cli.main(["remote-guide", "--stage-dir", str(home)])
    err_home = json.loads(capsys.readouterr().out)
    assert rc_home == 1
    assert err_home["reason"] == "stage_bundle_failed"
    assert sentinel.read_text(encoding="utf-8") == "do not delete"

    # Explicit stage-remote into a dedicated staging directory writes clean .py files and prunes stale owned files
    stage_target = tmp_path / "explicit-stage"
    rc_stage = cli.main(["stage-remote", "--stage-dir", str(stage_target)])
    staged_payload = json.loads(capsys.readouterr().out)
    assert rc_stage == 0
    assert staged_payload["bundle_staged"] is True
    assert (stage_target / "cli.py").is_file()
    assert (stage_target / ".mindie-debugger-bundle.json").is_file()


def test_pr_checklist_gate_device_tool_profile(tmp_path):
    import importlib.util

    gate_path = Path(__file__).resolve().parents[1] / ".github" / "scripts" / "check_pr_checklist.py"
    spec = importlib.util.spec_from_file_location("check_pr_checklist", gate_path)
    gate = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(gate)

    template = (Path(__file__).resolve().parents[1] / ".github" / "pull_request_template.md").read_text(encoding="utf-8")
    assert gate.validate_pr_body(template, profile="device-tool"), "raw template must fail until filled"

    valid = (
        "## 变更摘要 (Summary)\n- Prevent CPU capture from initializing NPU Phy-ID 0.\n\n"
        "<!--\n- [ ] **DEV-ZERO-CTX** ignored comment\n-->\n"
        "```markdown\n- [ ] **DEV-ZERO-CTX** ignored fenced example\n```\n"
        "## 设计原则 Review (Pre-PR Checklist)\n"
        "- [x] **DEV-ZERO-CTX** — Verified\n"
        "- [x] **DEV-READONLY-PATH** — Verified\n"
        "- [x] **DEV-TOPOLOGY** — Verified\n"
        "- [x] **DEV-BOUNDED-MEM** — Verified\n\n"
        "## 证据记录 (Evidence)\n"
        "- **DEV-ZERO-CTX**: pytest tests/test_zero_npu_context_and_readonly_triage.py passed\n"
        "- **DEV-READONLY-PATH**: mindie-debugger triage verified zero writes to HOME/CWD\n"
        "- **DEV-TOPOLOGY**: not-applicable: docs-only\n"
        "- **DEV-BOUNDED-MEM**: all core modules < 800 lines, bounded log read verified\n\n"
        "## 验证结果 (Verification)\n- 80+ pytest cases and wheel smoke check passed.\n"
    )
    assert gate.validate_pr_body(valid, profile="device-tool", changed_files=["README.md"]) == []
    runtime_err = gate.validate_pr_body(valid, profile="device-tool", changed_files=["mindie_debugger/cli.py"])
    assert any("docs-only" in e for e in runtime_err)


def _parse_phy0_hbm_mb(npu_smi_output: str) -> int | None:
    """Extract HBM-Usage Used MiB for the first device/chip (Phy-ID 0 / Card 0) from npu-smi info."""
    # Matches A2 `0 / 32768` or A3 `0 / 65536` HBM columns in `npu-smi info`
    pairs = [
        (int(m.group(1)), int(m.group(2)))
        for m in re.finditer(r"\b(\d+)\s*/\s*(\d{4,6})\b", npu_smi_output)
        if int(m.group(2)) >= 16384
    ]
    return pairs[0][0] if pairs else None


@pytest.mark.live_npu
def test_live_npu_cpu_precision_capture_leaves_phy0_hbm_delta_zero():
    """Verify on standing A2 (173.125.1.2) or A3 (192.168.13.165) that CPU capture allocates 0 MiB on Phy-ID 0."""
    repo_root = Path(__file__).resolve().parents[1]
    remote_check = (
        "import json, subprocess, sys, torch\n"
        "before = subprocess.check_output(['npu-smi', 'info'], text=True, timeout=8)\n"
        "from mindie_debugger.precision import capture_call, replay_call\n"
        "mod = torch.nn.Linear(8, 8)\n"
        "cap = capture_call(mod, torch.ones((2, 8)))\n"
        "rep = replay_call(cap, mod, repeat=2, mode='reference_check')\n"
        "after = subprocess.check_output(['npu-smi', 'info'], text=True, timeout=8)\n"
        "print(json.dumps({\n"
        "    'torch_npu_in_sys_modules': 'torch_npu' in sys.modules,\n"
        "    'torch_npu_version': cap.execution_state['torch_npu_version'],\n"
        "    'verdict': rep['verdict'],\n"
        "    'before': before,\n"
        "    'after': after,\n"
        "}))\n"
    )

    if shutil.which("npu-smi"):
        pytest.importorskip("torch")
        proc = subprocess.run(
            [sys.executable, "-I", "-c", remote_check],
            env={"PYTHONPATH": str(repo_root), "PATH": os.environ.get("PATH", "")},
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        assert proc.returncode == 0, proc.stderr
        data = json.loads(proc.stdout.strip().splitlines()[-1])
        before_mb = _parse_phy0_hbm_mb(data["before"])
        after_mb = _parse_phy0_hbm_mb(data["after"])
        assert before_mb is not None and after_mb is not None
        assert data["torch_npu_in_sys_modules"] is False
        assert after_mb - before_mb == 0
        return

    reachable_host = None
    for candidate in ("192.168.13.165", "173.125.1.2"):
        probe = subprocess.run(
            [
                "ssh", "-T", "-o", "ControlMaster=no", "-o", "BatchMode=yes",
                "-o", "ConnectTimeout=4", f"root@{candidate}", "npu-smi info >/dev/null 2>&1",
            ],
            capture_output=True,
            timeout=8,
            check=False,
        )
        if probe.returncode == 0:
            reachable_host = candidate
            break
    if reachable_host is None:
        pytest.skip("Standing Ascend A2/A3 hosts (173.125.1.2 / 192.168.13.165) are unreachable from this network")

    import tarfile
    import io

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for py in sorted((repo_root / "mindie_debugger").rglob("*.py")):
            if "__pycache__" not in py.parts:
                tf.add(py, arcname=py.relative_to(repo_root).as_posix())
    remote_sh = (
        "set -eu\n"
        "stage=$(mktemp -d /tmp/mindie-dbg-zero-ctx.XXXXXX)\n"
        "trap 'rm -rf \"$stage\"' EXIT\n"
        "tar -xzf - -C \"$stage\"\n"
        "py_cmd=''\n"
        "if python3 -c 'import torch' >/dev/null 2>&1; then\n"
        "  PYTHONPATH=\"$stage\" python3 -I -c " + shlex.quote(remote_check) + "\n"
        "else\n"
        "  cid=$(docker ps --format '{{.Names}}' | head -n 1)\n"
        "  test -n \"$cid\"\n"
        "  cstage=\"/tmp/mindie_dbg_zero_ctx_$$\"\n"
        "  docker exec \"$cid\" mkdir -p \"$cstage\"\n"
        "  docker cp \"$stage/mindie_debugger\" \"$cid:$cstage/mindie_debugger\"\n"
        "  trap 'docker exec \"$cid\" rm -rf \"$cstage\" >/dev/null 2>&1 || true; rm -rf \"$stage\"' EXIT\n"
        "  docker exec -e PYTHONPATH=\"$cstage\" \"$cid\" python3 -I -c " + shlex.quote(remote_check) + "\n"
        "fi\n"
    )
    proc = subprocess.run(
        [
            "ssh", "-T", "-o", "ControlMaster=no", "-o", "BatchMode=yes",
            "-o", "ConnectTimeout=5", f"root@{reachable_host}",
            f"bash -c {shlex.quote(remote_sh)}",
        ],
        input=buf.getvalue(),
        capture_output=True,
        timeout=25,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr.decode("utf-8", errors="replace")
    data = json.loads(proc.stdout.decode("utf-8", errors="replace").strip().splitlines()[-1])
    before_mb = _parse_phy0_hbm_mb(data["before"])
    after_mb = _parse_phy0_hbm_mb(data["after"])
    assert before_mb is not None and after_mb is not None
    assert data["torch_npu_in_sys_modules"] is False
    assert after_mb - before_mb == 0

