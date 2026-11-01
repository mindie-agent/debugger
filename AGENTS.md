# MindIE Debugger (`debugger`)

This repository owns the target Ascend NPU workload fault-triage (`npu_fault_triage`) and 3-primitive precision/memory-stomp localization toolkit (`mindie_debugger.precision`: `npu-watch`, `lockstep-diff`, `call-repro`).
MindIE Agent's own internal product telemetry and GitHub Issue reporting live separately in `diagnostics` (`mindie_diagnostics`).

Follow the workspace-wide rules in [`mindie-agent/AGENTS.md`](../mindie-agent/AGENTS.md):
- Keep every module `< 800` lines with a top-30-line `[Role]` / `[Invariants]` / `[Timeouts & Locks]` / `[Side-Effect Policy]` docstring header.
- Rely strictly on public PyTorch APIs (`storage.data_ptr()`, `untyped_storage()`, `register_forward_hook(..., with_kwargs=True)`, `TorchFunctionMode`); require zero custom hooks in `vllm-ascend` or `MindIE-SD`.
- Isolate `HOME`, `MINDIE_CONFIG_HOME`, `MINDIE_DATA_HOME`, and `MINDIE_DIAGNOSTICS_ROOT` in all tests.
