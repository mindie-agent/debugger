# mindie-debugger 0.1.1

[English](mindie-debugger-0.1.1.en.md)

本版本围绕工程设计原则专项审计（[`mindie-agent#332`](https://github.com/mindie-agent/mindie-agent/issues/332)）补齐本地 CLI 错误可见性、逃生通道引导与精度 Trace 有界读取契约，并接入受保护主分支发布门禁：

- **CLI 错误诚实性与原生逃生通道（[#2](https://github.com/mindie-agent/debugger/pull/2)）**：
  - 消除将本地路径缺失、`--script` 目录误传、普通 `ModuleNotFoundError` 或缺失默认 `plog` 目录静默包装为 `exit 0`（`status="remote_npu_execution_required"`）的假成功回退；所有失败路径统一返回退出码 `1`。
  - `_is_npu_env_missing_error` 严格收敛至确切的昇腾运行时缺失（`torch_npu`、`libhccl.so`、`libascendcl.so`、`libtorch_npu.so`），排除 `FileNotFoundError` / `PermissionError` 等本地文件系统异常；远端执行提示同时保留原生 `scp` / `rsync` / SSH 与可选 `remote-dev` 工具通道，不限制 Agent 绕行排障。
- **日志扫描与文件读取故障显式可见（[#2](https://github.com/mindie-agent/debugger/pull/2)）**：
  - 移除 `_read_bounded_log_text` 与 `_triage_directory` 对 `OSError` 的空字符串 `""` 静默吞错；改用显式 `os.scandir` 遍历（保留指向常规日志文件的符号链接，不递归目录符号链接），前置 `except TimeoutError: raise`，遇到文件/目录/`stdin` 读取错误立即输出包含 `errno` 与 `path` 的 `log_read_failed` 并以退出码 `1` 退出。
- **精度 Trace 文件有界加载（[#2](https://github.com/mindie-agent/debugger/pull/2)）**：
  - `_read_json` 按 `50 MiB`（`_JSON_TRACE_MAX_BYTES + 1` 单次有界读取，闭合 `stat` TOCTOU 窗口）限制精度对比与 Watch 摘要输入体积，超限时抛出指明路径与字节数的显式 `ValueError`。
- **受保护发布流水线加固（[#3](https://github.com/mindie-agent/debugger/pull/3)、[#4](https://github.com/mindie-agent/debugger/pull/4)）**：
  - 接入受保护 `main` 分支发布策略校验、旧客户端真实升级验收与 `MINDIE_RELEASE_DEPLOY_KEY` 显式传递。

包内 `__version__` 同步为 `0.1.1`。本版本的实际测试与基准结果以随包 `benchmark-results.json`、`benchmark-results.md` 及对应 CI 记录为准。
