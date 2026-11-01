# MindIE Debugger (`mindie-debugger`)

Target NPU workload debugger for Ascend (`vllm-ascend`, `MindIE-SD`, `MindIE-LLM`, `SGLang`, `sgl-kernel-npu`, and CANN operators).

> **Domain Boundary**: `mindie-debugger` (`mindie_debugger`) exclusively targets **user NPU workloads** (CANN plog/AICore/HCCL fault triage, precision/memory-stomp debugging, and `ACLGraph` capture/replay debugging). MindIE Agent's own internal DFX logging, redaction, and GitHub Issue reporting live separately in `mindie-diagnostics` (`mindie_diagnostics`).

## Core Capabilities

1. **NPU Fault Log Triage (`mindie_debugger.npu_fault_triage`)**
   - Deterministic CANN plog / dmesg / torchrun / HCCL traceback causal-chain extraction (`triage_npu_fault_logs`).
   - CLI: `mindie-debugger triage --root <log_dir>`

2. **Three-Primitive NPU Precision & Memory-Stomp Toolkit (`mindie_debugger.precision`)**
   - **Shared Foundation (`StorageGraph`, `StateSnapshot`)**: Alias-aware storage topology graph (`storage.data_ptr()` + byte offset/span overlap detection).
   - **Primitive 1 (`MemoryWatchpoint`, `WatchRingBuffer`)**: Zero-sync on-device ring telemetry + pre/post crime-scene latch + storage-overlap write attribution (`mindie-debugger watch-summary`).
   - **Primitive 2 (`LockstepDiffer`, `resolve_healing_frontier`)**: Input-healed layerwise lockstep diffing that isolates genuine faulty layers from downstream cascading drift (`mindie-debugger heal-diff`).
   - **Primitive 3 (`capture_call`, `replay_call`, `FakeQuantContract`)**: Self-contained operator/function reproducer capsule with automatic FP32 reference check, Kahan/pairwise/per-head/per-group sensitivity probes, and 1-ULP boundary jitter (`mindie-debugger replay <capsule.pt>`).

3. **ACLGraph (`torch.npu.NPUGraph`) Debugging & Dumping (`mindie_debugger.aclgraph`)**
   - **`ACLGraphDumper`**: Zero-allocation, zero-sync in-graph D2D intermediate layer tensor dumper that records hooked layer outputs across `g.replay()` steps (`dumper.advance_step()`, `dumper.read_step(step_index)`) and diffs any replay step against Eager mode (`dumper.compare_with_eager(eager_fn, *args, step_index=k)`).
   - **`ACLGraphCaptureGuard`**: `TorchFunctionMode` guard that catches illegal host-sync operations (`.item()`, `.tolist()`, `bool(tensor)`, `.cpu()`, `.numpy()`) inside `torch.npu.graph(g)` capture with exact `file:line` stack frames.
   - **`ACLGraphAddressTracker`**: Static buffer pointer stability & `slot_mapping` padded tail (`-1` vs `>= 0`) validator across `capture` and `replay`.
