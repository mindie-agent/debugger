# MindIE Debugger (`debugger`)

`mindie-debugger` owns Ascend NPU CANN fault triage (`npu_fault_triage`), ACLGraph capture/replay hazard detection (`aclgraph`), and the 3-primitive precision & memory-stomp localization toolkit (`mindie_debugger.precision`: `npu-watch`, `lockstep-diff`, `call-repro`).

- **Explicit Non-Ownership**: Internal MindIE Agent product telemetry and GitHub Issue outbox reporting belong to [`diagnostics`](../diagnostics); cluster device/port allocation belongs to [`coordinator`](../coordinator); fleet observation belongs to [`npu-top`](../npu-top).

---

## 1. Domain Invariants (`device-tool`)

1. **Zero Accidental NPU Context (`DEV-ZERO-CTX`)**:
   - Top-level package imports, `triage`, `decode-code`, and CPU tensor precision captures (`capture_call` / `replay_call` on CPU tensors) must **never** import `torch_npu` into `sys.modules` or allocate HBM on `Phy-ID 0` (`0 MiB` HBM delta).
   - Read `torch_npu` package versions via `importlib.metadata.version("torch_npu")`, never `__import__("torch_npu")`.
   - Query/restore `torch.npu.get_rng_state(dev)` and call `torch.npu.synchronize(dev)` **only** when `"torch_npu" in sys.modules`, `torch.npu.is_initialized()` is `True`, **and** `dev` is explicitly referenced by the captured tensors/generators. Never call `get_rng_state_all()` or `set_rng_state_all()`.
2. **Read-Only Triage & Receipt Honesty (`DEV-READONLY-PATH`)**:
   - `mindie-debugger triage` and `decode-code` are 100% read-only: guide references point directly to packaged assets without creating directories or running `shutil.rmtree`.
   - Explicit bundle staging (`stage-remote` / `remote-guide --stage-dir`) writes only clean `.py` files into an owned leaf directory and refuses protected roots (`/`, `$HOME`, `cwd`, package source).
   - Once primary triage succeeds, auxiliary guide or `/proc/<pid>` inspection errors keep exit code `0` and attach `receipt_warning` + `receipt_unconfirmed=True`.
3. **Topology & Storage-Alias Fidelity (`DEV-TOPOLOGY`)**:
   - Capsules preserve shared `UntypedStorage` aliases, views, strides, storage offsets, and explicit `torch.Generator` state across repeat trials; distributed topology mismatches fail visibly rather than fabricating single-rank success.
4. **Bounded Memory & Module Hygiene (`DEV-BOUNDED-MEM`)**:
   - Log reads (`4 MiB` head+tail window) and JSON trace comparisons (`50 MiB` cap) stay strictly bounded (`<= 512 MiB` RSS).
   - Every core module stays `< 800` lines with a top-30-line `[Role]` / `[Invariants]` / `[Timeouts & Locks]` / `[Side-Effect Policy]` header.

---

## 2. Module Map (`mindie_debugger/`)

- `cli.py`: `mindie-debugger` CLI dispatcher (`triage`, `decode-code`, `precision`, `aclgraph`, `remote-guide` / `stage-remote`).
- `npu_fault_triage.py`: Table-driven CANN/RTS/HCCL error-code decoder and multi-rank `plog` root-cause selector (prioritizing concrete AICore/driver/sync faults over cascading peer HCCL timeouts).
- `aclgraph.py`: `ACLGraphCaptureGuard`, `ACLGraphAddressTracker`, `ACLGraphDumper`, and `ACLGraphVllmInspector`.
- `observability.py`: Optional stdlib-first bridge to `mindie_diagnostics` for owned internal exceptions.
- `precision/`:
  - `storage_graph.py` & `tree_utils.py`: Shared `UntypedStorage` graph capture, exact view reconstruction, and in-place state restoration.
  - `call_state.py` & `call_repro.py`: `capture_call` / `replay_call` (`CallCapsule`), per-device RNG/generator restoration, collective trace verification, and `FakeQuantContract`.
  - `fp32_reference.py`, `lockstep_diff.py`, `metrics.py`, `jitter.py`: Automatic FP32 reference enclosure, `LockstepDiffer` with optional in-place healing, and ULP/SJR/cosine metrics.
  - `npu_watch.py` & `stimulus.py`: Zero-host-sync `WatchRingBuffer` crime-scene latch, stride-exact `MemoryWatchpoint`, and MoE safetensors slimming (`slim_moe_checkpoint`).

---

## 3. Verification Commands

```bash
# 1. Fast local unit + subprocess contract suite (Tier 0 + Tier 1, zero NPU required)
uv run --with torch pytest -q -m "not live_npu"

# 2. Standing Ascend A2 (173.125.1.2) / A3 (192.168.13.165) non-destructive Tier 2 suite
uv run --with torch pytest -q -m live_npu

# 3. Installed-wheel smoke check outside the source checkout + LF line-ending gate
python3 .github/scripts/check_lf_eol.py
python3 .github/scripts/check_pr_checklist.py --profile device-tool --body-file <pr_body.md>
```

---

## 4. Living Specifications

- Workspace architecture & guardrails: [`../mindie-agent/AGENTS.md`](../mindie-agent/AGENTS.md) and [`../mindie-agent/docs/historical-issues-and-guardrails.md`](../mindie-agent/docs/historical-issues-and-guardrails.md)
- Source rebuild & artifact recovery: [`../mindie-agent/docs/source-rebuild-contract.md`](../mindie-agent/docs/source-rebuild-contract.md)
