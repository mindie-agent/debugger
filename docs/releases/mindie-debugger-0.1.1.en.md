# mindie-debugger 0.1.1

[中文](mindie-debugger-0.1.1.md)

This release resolves the cross-repo Design Principles audit items for `mindie-debugger` ([`mindie-agent#332`](https://github.com/mindie-agent/mindie-agent/issues/332))—restoring CLI error visibility, native SSH/CLI escape-route guidance, and bounded JSON trace loading—and adopts protected `main` release verification:

- **CLI Error Honesty & Native Escape Routes ([#2](https://github.com/mindie-agent/debugger/pull/2), [#5](https://github.com/mindie-agent/debugger/pull/5))**:
  - Stop masking missing local paths, directory `--script` arguments, unrelated `ModuleNotFoundError`s, or absent default CANN `plog` roots as `exit 0` (`status="remote_npu_execution_required"`); ensure `aclgraph guard` and `aclgraph inspect` set `status="error"` instead of leaving `"clean"` / `"ok"` when target execution raises an exception prior to any graph violation, and exit with code `1` on all failure paths.
  - Narrow `_is_npu_env_missing_error` strictly to missing Ascend runtime dependencies (`torch_npu`, `libhccl.so`, `libascendcl.so`, `libtorch_npu.so`), excluding `FileNotFoundError` / `PermissionError` filesystem errors, and document standard `scp` / `rsync` / SSH alongside optional `remote-dev` tools in remote execution guidance.
- **Explicit Log-Read & Directory-Walk Failures ([#2](https://github.com/mindie-agent/debugger/pull/2))**:
  - Remove silent empty-string (`""`) fallbacks on `OSError` in `_read_bounded_log_text` and `_triage_directory`; walk directories explicitly via `os.scandir` (following symlinks to regular log files without recursing directory symlinks), re-raise `TimeoutError`, and emit `log_read_failed` (`errno`, `path`, exit `1`) whenever file, directory, or `stdin` I/O fails.
- **Bounded Precision Trace JSON Loading ([#2](https://github.com/mindie-agent/debugger/pull/2))**:
  - Cap `_read_json` at `50 MiB` via a single-pass `_JSON_TRACE_MAX_BYTES + 1` read (closing the `stat` TOCTOU window) and raise an explicit `ValueError` naming the file path and byte count when exceeded.
- **Protected Release Pipeline ([#3](https://github.com/mindie-agent/debugger/pull/3), [#4](https://github.com/mindie-agent/debugger/pull/4))**:
  - Enforce protected-`main` publication policy checks, real old-client upgrade verification, and explicit `MINDIE_RELEASE_DEPLOY_KEY` secret forwarding.

The package `__version__` is synchronized to `0.1.1`. Actual test and benchmark results for this release are recorded in the bundled `benchmark-results.json`, `benchmark-results.md`, and CI runs.
