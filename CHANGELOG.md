# Changelog

## 0.1.0 (2026-11-02)

- Extract deterministic CANN `plog` / error-code fault triage (`triage_npu_fault`, `triage_ascend_plog`, `decode_cann_error_code`) and the 3-primitive NPU precision/memory-stomp debugging toolkit (`npu-watch`, `lockstep-diff`, `call-repro`) into standalone `mindie-debugger`, with first-class `vllm-ascend`, `xLLM`, and `MindIE-SD` framework fault classification.
- Add `mindie_debugger.aclgraph` (`ACLGraphCaptureGuard`, `ACLGraphAddressTracker`, `ACLGraphDumper`, `ACLGraphVllmInspector`) with KV `slot_mapping` / `block_table` bounds verification, Context Parallel / Sliding Window support, and Eager differential comparison.
- Support automatic in-container `$ASCEND_PROCESS_LOG_PATH` / `~/ascend/log` discovery, live `/proc/<pid>/environ` and `/proc/<pid>/maps` drift auditing, numeric-first metrics with `valid_length` active-token slicing, and `remote-guide` bundle staging for `remote-dev`.
