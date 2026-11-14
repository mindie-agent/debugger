# MindIE Debugger (`mindie-debugger`)

<p>
  <a href="https://github.com/mindie-agent/debugger/releases"><img src="https://img.shields.io/badge/Release-v0.1.2-blue" alt="Release v0.1.2"></a>
</p>

[中文](README.md)

`mindie-debugger` (Python package `mindie_debugger`) is MindIE Agent's deterministic Ascend NPU workload fault-triage and precision/memory-stomp localization component, strictly separated from [`diagnostics`](https://github.com/mindie-agent/diagnostics) (which handles MindIE Agent's own internal DFX logging and opt-in GitHub Issue reporting).

## Core Capabilities

1. **Universal 3-Primitive NPU Precision, State-Corruption & Memory-Stomp Toolkit (`mindie_debugger.precision`)**:
   - **`npu-watch` (`StorageGraph`, `MemoryWatchpoint`, `WatchRingBuffer`)**: Detects `untyped_storage` aliasing and byte-range overlaps across steps/ranks (`shares_storage` / `overlapping_views`), pinpoints the first submodule or call mutating protected state (`MemoryWatchpoint`), and provides `NPUGraph`-capture-safe, zero-sync device ring snapshots with a first-fault `NaN`/`Inf` latch (`WatchRingBuffer`).
   - **`call-repro` (`capture_call`, `replay_call`, `FakeQuantContract`)**: Captures pre/post `StorageGraph` snapshots for any operator or module call, replays against automatic FP32 reference promotion (`mode="reference_check"`), multi-trial determinism checks (`repeat=N, det_lock=True`), and `FakeQuantContract` oracles to separate `nondeterministic_jitter`, `quantization_inherent_loss`, and `real_numerical_bug`.
   - **`lockstep-diff` (`LockstepDiffer`, `slim_moe_checkpoint`)**: Pinpoints the first module boundary exceeding the ULP/jitter noise floor (`first_divergent_layer`) and shrinks large MoE checkpoints into deterministic single-node reproducers (`slim_moe_checkpoint`).
2. **Deterministic Ascend CANN Runtime Fault Triage (`mindie_debugger.npu_fault_triage`)**:
   - `triage_npu_fault` / `triage_ascend_plog` / `decode_cann_error_code`: Parses CANN `plog` and error logs to isolate the primary root cause (separating primary AICore `507015`/`EZ9999` or OOM `EL0004` faults from secondary HCCL `EI0006` cascade timeouts).
