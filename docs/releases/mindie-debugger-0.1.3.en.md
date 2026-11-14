# mindie-debugger 0.1.3: Windows CP936 / pythonw.exe Stdio Safety & Timeout Propagation

[中文](mindie-debugger-0.1.3.md)

Hardens cross-platform CLI stream handling on Chinese OEM code-page (`CP936`) Windows and windowless `pythonw.exe` hosts, and enforces strict `TimeoutError` propagation across precision primitives:

- **Windows `CP936` Pipe & `pythonw.exe` `None`-stdin Guard (`cli.py`)**: `mindie-debugger triage` checks `sys.stdin is not None` before `.isatty()` and decodes piped `plog` streams from `sys.stdin.buffer` as UTF-8 (`errors="replace"`), preventing `AttributeError` under `pythonw.exe` and `UnicodeDecodeError` when UTF-8 logs are piped under `CP936` locales.
- **Strict `TimeoutError` Propagation (`precision/call_state.py`, `lockstep_diff.py`, `npu_watch.py`)**: Explicitly re-raises `TimeoutError` before every `except (... OSError):` boundary so Python 3.11+ `TimeoutError` subclasses are never masked as backend probe fallbacks.

In-package `__version__` is aligned to `0.1.3`. Actual test and benchmark results are recorded in the packaged `benchmark-results.json`, `benchmark-results.md`, and CI runs.
