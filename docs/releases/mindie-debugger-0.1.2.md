# mindie-debugger 0.1.2：进程内结构化故障观测与溯源回执

[English](mindie-debugger-0.1.2.en.md)

补齐 `mindie-debugger` CLI 入口的进程内 Diagnostics 2.0 结构化异常观测与父网关溯源回执契约：

- **进程内未预期内部异常结构化记录（`cli.py`）**：对 CLI 分发与执行路径上的非预期内部异常（严格排除 `FileNotFoundError`、`PermissionError`、`TimeoutError`、`ValueError` 等预期输入/环境错误）调用 `mindie_diagnostics` 记录零自由文本故障事件，自动捕获精确源码栈帧与子命令阶段（`decode-code`、`triage`、`aclgraph`、`precision` 等）。
- **父网关溯源回执（`mindie-diagnostic-incident`）**：在记录成功后向 `stderr` 输出标准 `mindie-diagnostic-incident: <32-hex>` 标记，配合 `MINDIE_DIAGNOSTICS_INVOCATION_TOKEN` 供各 Harness 插件网关做本地同次调用溯源核验，零用户路径、命令参数或异常自由文本外泄。

包内 `__version__` 同步为 `0.1.2`。本版本的实际测试与基准结果以随包 `benchmark-results.json`、`benchmark-results.md` 及对应 CI 记录为准。
