# mindie-debugger 0.1.0

首个正式发布版本，将原本内嵌于 `mindie-diagnostics` 的昇腾 NPU 故障分诊与三原语精度/内存踩踏定位工具包提取为独立的零依赖组件 `mindie-debugger`，并新增 `ACLGraph` 捕获/回放专项调试器与远端 NPU 容器自动发现能力：

- **确定性 CANN `plog` 故障分诊（`triage_npu_fault` / `decode_cann_error_code`）**：
  - 支持 CANN / RTS / HCCL / Driver 错误码查表解码与多 Rank `plog` 根因消歧（优先提取最早的 AICore / HBM ECC / 驱动故障，自动抑制后续由对端触发的级联 HCCL 超时报警）。
  - 覆盖 `vllm-ascend`、`xLLM` 与 `MindIE-SD` 模型级典型故障分类（`ACLGraph` 流捕获耗尽、Idle-DP 集合通信挂起、`msmodelslim` 量化权重/Scale 失配、数值 `NaN`/`Inf` 与确定性漂移、Mooncake KV 池传输错误、`xLLM` ATB/CANN 版本错配与多流死锁、`MindIE-SD` RainFusion/FA 序列长度整除越界与双流 `notify_wait` 死锁）以及自定义算子符号遮蔽（`161001`）、多线程 NPU 上下文丢失（`107003`）与 `EngineCore` 共享内存破坏。
  - 支持容器内 `$ASCEND_PROCESS_LOG_PATH` / `~/ascend/log` 自动发现、`--pid` 过滤以及在役进程 `/proc/<pid>/environ` 与 `/proc/<pid>/maps` 环境漂移审计。
- **三原语 NPU 精度、状态污染与内存踩踏定位（`mindie_debugger.precision`）**：
  - `npu-watch`（`StorageGraph`、`MemoryWatchpoint`、`WatchRingBuffer`）：追踪共享底层 `UntypedStorage` 的张量视图拓扑与写后异常（`NaN`/`Inf`、静默全零、尾部 Padding 污染）。
  - `lockstep-diff`（`LockstepDiffer`）：双执行路径同输入锁步对比与自动输入回填（Input Healing），支持 `valid_length` 有效 Token 切片与尾部脏槽位分离统计。
  - `call-repro`（`capture_call`、`replay_call`）：单次算子/模块调用胶囊化复现、FP32 黄金参考对比（`reference_check`）与量化/抖动敏感度探测。
- **`ACLGraph` / `NPUGraph` 捕获与回放专项调试（`mindie_debugger.aclgraph`）**：
  - `ACLGraphCaptureGuard`：拦截静态图捕获期间非法的 Host-Device 同步操作（`.item()`、`.cpu()`、`bool(tensor)`、`.tolist()`）。
  - `ACLGraphAddressTracker`：校验静态图回放期间输入张量指针稳定性（`rebound_pointer`）、尾部 Padding `slot_mapping` 掩码以及 `block_table` / `seq_lens` 物理块容量边界（兼容 Context Parallel 本地掩码与 Sliding Window 滑窗回收）。
  - `ACLGraphDumper` 与 `ACLGraphVllmInspector`：无同步设备端槽位快照与 Eager 模式逐层比对。
- **远端 NPU 容器一键分发引导（`remote-guide`）**：当本地宿主机缺少 `torch_npu` 或容器内 `plog` 目录时，自动生成纯净 Python 源码包并提供 `remote-dev` 推送与容器内执行指南。

包内 `__version__` 同步为 `0.1.0`。
