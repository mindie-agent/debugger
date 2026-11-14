# mindie-debugger 0.1.2: In-Process Structured Fault Observability & Provenance Receipts

[中文](mindie-debugger-0.1.2.md)

Adds in-process Diagnostics 2.0 structured internal-exception recording and parent-gateway provenance receipts to the `mindie-debugger` CLI:

- **In-Process Internal-Exception Recording (`cli.py`)**: Records unexpected internal CLI dispatch failures through `mindie_diagnostics` with owned stack frames and subcommand stage attribution (`decode-code`, `triage`, `aclgraph`, `precision`, etc.), while strictly excluding expected operational/input errors (`FileNotFoundError`, `PermissionError`, `TimeoutError`, `ValueError`).
- **Provenance-Bound Incident Marker (`mindie-diagnostic-incident`)**: Emits `mindie-diagnostic-incident: <32-hex>` on `stderr` paired with local `MINDIE_DIAGNOSTICS_INVOCATION_TOKEN` verification so Harness plugin gateways link child crash receipts without exposing paths, arguments, or exception messages.

The package `__version__` is synchronized to `0.1.2`. Actual test and benchmark results for this release are recorded in the bundled `benchmark-results.json`, `benchmark-results.md`, and CI runs.
