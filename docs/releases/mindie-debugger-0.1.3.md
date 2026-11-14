# mindie-debugger 0.1.3：Windows CP936 / pythonw.exe 标准流安全与超时异常透传

[English](mindie-debugger-0.1.3.en.md)

针对 Windows 中文 OEM 代码页（`CP936`）、无控制台宿主（`pythonw.exe`）及精度调试边界补齐跨平台确定性保护：

- **Windows `CP936` 管道与 `pythonw.exe` 空标准输入安全（`cli.py`）**：`mindie-debugger triage` 在读取管道输入前显式校验 `sys.stdin is not None`（兼容 Windows `pythonw.exe` 无标准流宿主），并优先从底层字节流按 `utf-8`（`errors="replace"`）解码，消除中文 Windows 默认 `CP936` 下对含非 GBK 字节 UTF-8 `plog` 的 `UnicodeDecodeError`。
- **精度与监控原语 `TimeoutError` 严格透传（`precision/call_state.py`、`lockstep_diff.py`、`npu_watch.py`）**：在所有捕获 `OSError` 的后端状态探测、流读取与分布式拓扑读取边界前显式优先重抛 `TimeoutError`（Python 3.11+ 中 `TimeoutError` 继承自 `OSError`），杜绝将调用超时误降级为后端不可用或静默默认值。

包内 `__version__` 同步为 `0.1.3`。本版本的实际测试与基准结果以随包 `benchmark-results.json`、`benchmark-results.md` 及对应 CI 记录为准。
