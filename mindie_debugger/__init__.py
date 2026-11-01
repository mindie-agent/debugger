"""Ascend NPU fault triage and 3-primitive precision/memory-stomp debugger.

[Role]
  Publish the deterministic CANN/NPU fault triage functions
  (``decode_cann_error_code``, ``triage_ascend_plog``, ``triage_npu_fault``)
  and the universal 3-primitive NPU precision, state-corruption & memory-stomp
  debugging toolkit (``StorageGraph``, ``MemoryWatchpoint``,
  ``WatchRingBuffer``, ``LockstepDiffer``, ``capture_call``, ``replay_call``).

[Invariants]
  * Importing this package performs no filesystem, network, subprocess, or
    NPU device initialization work; ``torch`` is loaded lazily on first tensor
    primitive call.
  * Fault triage returns structured data only and never executes shell commands.

[Timeouts & Locks]
  Import-time only; no locks or background threads are created.

[Side-Effect Policy]
  None at import time. Callers opt in to tensor snapshotting or hook attachment.
"""
from .npu_fault_triage import (
    decode_cann_error_code,
    triage_ascend_plog,
    triage_npu_fault,
)
from .precision import (
    CallCapsule,
    FakeQuantContract,
    LockstepDiffer,
    MemoryWatchpoint,
    StateSnapshot,
    StorageGraph,
    StorageNode,
    TensorViewSpec,
    WatchRingBuffer,
    build_deterministic_stimulus,
    capsule_summary,
    capture_call,
    capture_state_snapshot,
    fake_quantize,
    flatten_tree_with_paths,
    map_tree,
    reconstruct_from_snapshot,
    replay_call,
    restore_state_snapshot,
    slim_moe_checkpoint,
    snapshot_tensor_map,
)
from .aclgraph import (
    ACLGraphAddressTracker,
    ACLGraphCaptureGuard,
    ACLGraphDumper,
    ACLGraphVllmInspector,
)

__version__ = "0.1.0"
__all__ = [
    "ACLGraphAddressTracker",
    "ACLGraphCaptureGuard",
    "ACLGraphDumper",
    "ACLGraphVllmInspector",
    "CallCapsule",
    "FakeQuantContract",
    "LockstepDiffer",
    "MemoryWatchpoint",
    "StateSnapshot",
    "StorageGraph",
    "StorageNode",
    "TensorViewSpec",
    "WatchRingBuffer",
    "build_deterministic_stimulus",
    "capsule_summary",
    "capture_call",
    "capture_state_snapshot",
    "decode_cann_error_code",
    "fake_quantize",
    "flatten_tree_with_paths",
    "map_tree",
    "reconstruct_from_snapshot",
    "replay_call",
    "restore_state_snapshot",
    "slim_moe_checkpoint",
    "snapshot_tensor_map",
    "triage_ascend_plog",
    "triage_npu_fault",
]
