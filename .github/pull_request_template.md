## 变更摘要 (Summary)

- Describe the root cause / trigger and the resulting behavioral change.

## 设计原则 Review (Pre-PR Checklist — `device-tool`)

- [ ] **DEV-ZERO-CTX** — Control paths and CPU captures never dynamically import `torch_npu`, never call `get_rng_state_all()`, and touch only initialized target device indices (`Phy-ID 0` HBM delta `== 0 MiB`).
- [ ] **DEV-READONLY-PATH** — `triage` / observation paths are strictly read-only (zero CWD/HOME cache pollution or `rmtree`), and auxiliary warnings preserve primary results (`isError=False` / exit `0`).
- [ ] **DEV-TOPOLOGY** — Physical/logical NPU topology, storage views/strides, and explicit per-device RNG/generator semantics remain exact without fabricated fallbacks.
- [ ] **DEV-BOUNDED-MEM** — Log/trace inputs and working sets remain strictly bounded (`<= 512 MiB`), core modules stay `< 800` lines with top-30-line invariant headers, and all files use Unix `LF`.

## 证据记录 (Evidence)

- **DEV-ZERO-CTX**: `<command / test and measured output, or not-applicable: docs-only>`
- **DEV-READONLY-PATH**: `<command / test and observed filesystem state, or not-applicable: docs-only>`
- **DEV-TOPOLOGY**: `<command / fixture or live A2/A3 result, or not-applicable: docs-only>`
- **DEV-BOUNDED-MEM**: `<command / memory & module bound check, or not-applicable: docs-only>`

## 验证结果 (Verification)

- Executed test suites, wheel smoke test, LF check, and any unexecuted live-hardware scope.
