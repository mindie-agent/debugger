# mindie-debugger 0.1.4：零意外 NPU 上下文、纯只读分诊与回执诚实性加固

[English](mindie-debugger-0.1.4.en.md)

`mindie-debugger v0.1.4` 针对目标昇腾 NPU 负载分诊与三原语精度调试完成洁净架构收敛：

- **零意外 NPU 上下文污染（`DEV-ZERO-CTX`）**：控制路径与 CPU 张量胶囊捕获绝不动态导入 `torch_npu` 或在 `Phy-ID 0` 分配显存（`0 MiB` HBM 增量）；彻底移除 `get_rng_state_all()` / `set_rng_state_all()`，仅在 `torch_npu` 已加载且已初始化时按显式引用的设备索引查询/恢复随机数与同步状态。
- **100% 只读故障分诊与安全叶子目录暂存（`DEV-READONLY-PATH`）**：`mindie-debugger triage` 与 `decode-code` 直接引用随包内置指南资产，零目录创建、零 `rmtree`；`stage-remote` 仅向受控叶子目录写出纯 `.py` 复现脚本并拒绝根路径、`$HOME`、工作目录与源码树；主分诊成功后辅助环境读取异常保持退出码 `0` 并附带 `receipt_warning` + `receipt_unconfirmed=True`。
- **`device-tool` 模块化与治理门禁对齐**：核心大模块拆分为 `< 800` 行内聚子模块并补齐四段式契约头，启用 `--profile device-tool` PR 门禁。

包内 `__version__` 同步为 `0.1.4`。
