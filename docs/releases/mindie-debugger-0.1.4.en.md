# mindie-debugger 0.1.4: Zero Accidental NPU Context, Read-Only Triage, and Receipt Honesty

[中文](mindie-debugger-0.1.4.md)

`mindie-debugger v0.1.4` hardens Ascend NPU fault triage and 3-primitive precision debugging under the `device-tool` clean-architecture contract:

- **Zero Accidental NPU Context (`DEV-ZERO-CTX`)**: Control paths and CPU tensor captures never import `torch_npu` or allocate HBM on `Phy-ID 0` (`0 MiB` HBM delta); replaces `get_rng_state_all()` / `set_rng_state_all()` with per-device RNG and stream synchronization gated strictly on initialized `torch_npu` and explicitly referenced device indices.
- **100% Read-Only Triage & Confined Leaf Staging (`DEV-READONLY-PATH`)**: `mindie-debugger triage` and `decode-code` reference packaged guide assets directly without creating directories or calling `rmtree`; `stage-remote` writes clean `.py` files only into owned leaf directories while rejecting `/`, `$HOME`, `cwd`, and package roots; auxiliary inspection warnings after successful triage preserve exit code `0` with `receipt_warning` + `receipt_unconfirmed=True`.
- **`device-tool` Modularity & Governance**: Splits large modules under `< 800` lines with top-30-line invariant headers and aligns PR validation to `--profile device-tool`.

Package `__version__` is updated to `0.1.4`.
