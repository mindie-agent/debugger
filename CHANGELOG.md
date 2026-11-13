# Changelog

## 0.1.1 (2026-11-13)

- **CLI Error Visibility & Native Escape Routes (`#2`, `#5`, `mindie-agent#332`)**: Stop masking local argument/path errors, unrelated `ModuleNotFoundError`s, and missing CANN `plog` roots as `exit 0` (`remote_npu_execution_required`); ensure `aclgraph guard` and `aclgraph inspect` report `status="error"` (never `"ok"` / `"clean"`) when target execution raises an exception before recording a graph violation; return exit code `1` on all failure paths, narrow NPU-runtime missing checks strictly to `torch_npu` / `libhccl.so` / `libascendcl.so` / `libtorch_npu.so`, and allow standard `scp`/`rsync`/SSH alongside optional `remote-dev` tools in remote execution guidance.
- **Log Read Honesty (`#2`, `mindie-agent#332`)**: Propagate `TimeoutError` and report explicit `log_read_failed` (`errno`, `path`, exit `1`) on file, directory-walk (`os.scandir`), or `stdin` `OSError` instead of silently substituting empty text (`""`) and reporting a clean scan; follow file symlinks while refusing directory-symlink recursion.
- **Bounded JSON Trace Reads (`#2`, `mindie-agent#332`)**: Enforce a strict `50 MiB` (`_JSON_TRACE_MAX_BYTES + 1` single-pass read) cap in `_read_json` for precision trace comparisons and watch summaries.
- **Protected Release Pipeline (`#3`, `#4`)**: Require protected `main` publication, real old-client upgrade verification, and explicit `MINDIE_RELEASE_DEPLOY_KEY` forwarding.

## 0.1.0 (2026-11-02)

- Extract deterministic CANN `plog` / error-code fault triage (`triage_npu_fault`, `triage_ascend_plog`, `decode_cann_error_code`) and the 3-primitive NPU precision/memory-stomp debugging toolkit (`npu-watch`, `lockstep-diff`, `call-repro`) into standalone `mindie-debugger`, with first-class `vllm-ascend`, `xLLM`, and `MindIE-SD` framework fault classification.
- Add `mindie_debugger.aclgraph` (`ACLGraphCaptureGuard`, `ACLGraphAddressTracker`, `ACLGraphDumper`, `ACLGraphVllmInspector`) with KV `slot_mapping` / `block_table` bounds verification, Context Parallel / Sliding Window support, and Eager differential comparison.
- Support automatic in-container `$ASCEND_PROCESS_LOG_PATH` / `~/ascend/log` discovery, live `/proc/<pid>/environ` and `/proc/<pid>/maps` drift auditing, numeric-first metrics with `valid_length` active-token slicing, and `remote-guide` bundle staging for `remote-dev`.
