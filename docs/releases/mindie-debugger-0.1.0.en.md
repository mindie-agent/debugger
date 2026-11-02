# mindie-debugger 0.1.0

Initial formal release extracting the Ascend NPU fault triage and three-primitive precision/memory-stomp debugging toolkit from `mindie-diagnostics` into the standalone zero-dependency `mindie-debugger` package, together with dedicated `ACLGraph` capture/replay debugging and remote NPU container staging:

- **Deterministic CANN `plog` Fault Triage (`triage_npu_fault` / `decode_cann_error_code`)**:
  - Table-driven CANN / RTS / HCCL / Driver error-code decoding and multi-rank `plog` root-cause selection (prioritizing primary AICore / HBM ECC / driver faults over cascading peer-rank HCCL wait timeouts).
  - Covers `vllm-ascend`, `xLLM`, and `MindIE-SD` model-level fault categories (`ACLGraph` stream-capture exhaustion, idle-DP collective hangs, `msmodelslim` quantized weight/scale mismatches, numerical `NaN`/`Inf` & determinism drift, Mooncake KV pool transfer faults, `xLLM` ATB/CANN ABI mismatch & multi-stream deadlocks, `MindIE-SD` RainFusion/FA sequence-divisibility tiling bounds & dual-stream `notify_wait` deadlocks) as well as custom-op ABI symbol shadowing (`161001`), multithreaded NPU context loss (`107003`), and `EngineCore` shared-memory frame corruption.
  - Supports automatic in-container `$ASCEND_PROCESS_LOG_PATH` / `~/ascend/log` discovery, `--pid` filtering, and live `/proc/<pid>/environ` + `/proc/<pid>/maps` environment drift auditing.
- **Three-Primitive NPU Precision, State-Corruption & Memory-Stomp Toolkit (`mindie_debugger.precision`)**:
  - `npu-watch` (`StorageGraph`, `MemoryWatchpoint`, `WatchRingBuffer`): tracks shared `UntypedStorage` view aliasing and post-op anomalies (`NaN`/`Inf`, silent zeroing, padded-tail pollution).
  - `lockstep-diff` (`LockstepDiffer`): lockstep differential execution with automatic input healing and `valid_length` active-token slicing.
  - `call-repro` (`capture_call`, `replay_call`): single-call capsule capture, deterministic replay, FP32 reference checking (`reference_check`), and fake-quant/jitter sensitivity probes.
- **`ACLGraph` / `NPUGraph` Capture & Replay Debugging (`mindie_debugger.aclgraph`)**:
  - `ACLGraphCaptureGuard`: catches forbidden host-device synchronizations (`.item()`, `.cpu()`, `bool(tensor)`, `.tolist()`) during graph capture.
  - `ACLGraphAddressTracker`: verifies replay input pointer stability (`rebound_pointer`), padded `slot_mapping` tail masking, and `block_table` / `seq_lens` KV capacity bounds (supporting Context Parallel local slot masks and Sliding Window null-block eviction).
  - `ACLGraphDumper` and `ACLGraphVllmInspector`: capture-safe device-indexed layer snapshots and Eager-mode layer-by-layer differential comparison.
- **Remote NPU Container Staging (`remote-guide`)**: automatically stages a clean pure-Python `mindie_debugger` bundle and emits step-by-step `remote-dev` push and execution commands when invoked on a host without `torch_npu` or local CANN `plog` directories.

The package `__version__` is synchronized to `0.1.0`.
