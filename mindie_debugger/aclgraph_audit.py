"""Static/AST contract auditor and runtime-log bridge for ACLGraph & NPU state bugs.

[Role]
  Provides zero-dependency AST contract verification (`audit_aclgraph_contracts`)
  for `vllm-ascend` / `MindIE-SD` `ACLGraph` (`NPUGraph`, `ACLGraphWrapper`,
  `UpdatableGraph`) capture/replay code, plus structured Python/runtime log
  triage (`augment_triage_with_runtime_clues`) when logs come from `pytest` or
  `EngineCore` rather than raw CANN `plog` files.

[Invariants]
  * Pure AST / text analysis (`torch` and `vllm` are not required to import).
  * Every contract violation is derived from live AST inspection of the target
    file(s), so applying the fix to the file transitions `status` from
    `"violation"` to `"ok"`.

[Timeouts & Locks]
  Synchronous, in-memory AST parsing of bounded `.py` source files; no locks.

[Side-Effect Policy]
  Read-only inspection of user-specified source or log paths.
"""
from __future__ import annotations

import ast
from pathlib import Path
from typing import Any


def _iter_py_files(target: Path) -> list[Path]:
    if target.is_file():
        return [target] if target.suffix == ".py" else []
    if target.is_dir():
        return sorted(p for p in target.rglob("*.py") if "__pycache__" not in p.parts)
    return []


def _unparse(node: ast.AST | None) -> str:
    if node is None:
        return ""
    try:
        return ast.unparse(node)
    except Exception:
        return ""


def _audit_file(path: Path) -> list[dict[str, Any]]:
    try:
        source = path.read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(source, filename=str(path))
    except Exception:
        return []

    findings: list[dict[str, Any]] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        fn_src = _unparse(node)

        # 1. causal_conv1d / graph-captured decode update missing out= or using negative null_block_id
        if node.name == "causal_conv1d":
            for sub in ast.walk(node):
                if isinstance(sub, ast.Call) and _unparse(sub.func).endswith("causal_conv1d_update"):
                    kw_names = {kw.arg for kw in sub.keywords if kw.arg}
                    if "out" not in kw_names:
                        findings.append({
                            "rule": "MISSING_OUT_BUFFER_IN_GRAPH_UPDATE",
                            "file": str(path),
                            "line": sub.lineno,
                            "function": node.name,
                            "detail": (
                                "causal_conv1d_update is called without `out=output`; skipped/padded rows "
                                "are not zero-preserved and out-of-place output allocation breaks NPUGraph replay."
                            ),
                        })
                if isinstance(sub, ast.Assign):
                    targets = [_unparse(t) for t in sub.targets]
                    val = _unparse(sub.value)
                    if "null_block_id" in targets and val == "PAD_SLOT_ID":
                        findings.append({
                            "rule": "NEGATIVE_NULL_BLOCK_ID_WITH_PACKED_STATE",
                            "file": str(path),
                            "line": sub.lineno,
                            "function": node.name,
                            "detail": (
                                "fla_npu causal_conv1d_update disables null-block filtering when null_block_id < 0 "
                                "(PAD_SLOT_ID = -1), causing -1 in kernel_indices to underflow kernel_state[-1] "
                                "(507011). Reserve an extra packed state row `kernel_state = torch.empty((requests + 1, "
                                "state_len, dim), ...)` and set `null_block_id = requests`."
                            ),
                        })

        # 2. _copy_conv_state_kernel storing -1 instead of REQUESTS for inactive packed rows
        if node.name == "_copy_conv_state_kernel":
            for sub in ast.walk(node):
                if isinstance(sub, ast.Call) and _unparse(sub.func).endswith("where"):
                    if len(sub.args) == 3 and _unparse(sub.args[2]) in ("-1", "-1.0"):
                        findings.append({
                            "rule": "NEGATIVE_NULL_BLOCK_ID_WITH_PACKED_STATE",
                            "file": str(path),
                            "line": sub.lineno,
                            "function": node.name,
                            "detail": (
                                "Inactive packed conv_state rows are mapped to -1 (`tl.where(active, request, -1)`), "
                                "which underflows `kernel_state[-1]` in AscendC causal_conv1d_update. Map inactive "
                                "rows to the reserved non-negative null slot `REQUESTS` (`tl.where(active, request, REQUESTS)`)."
                            ),
                        })

        # 3. AutoRegressiveAclGraphManager._updatable_graph_replay missing build_draft_attn_metadatas
        if node.name == "_updatable_graph_replay":
            if "build_fia_params" in fn_src and "build_draft_attn_metadatas" not in fn_src:
                findings.append({
                    "rule": "UPDATABLE_GRAPH_STALE_STEP_METADATA",
                    "file": str(path),
                    "line": node.lineno,
                    "function": node.name,
                    "detail": (
                        "_updatable_graph_replay calls build_fia_params / resolve_tasks without first calling "
                        "`self.speculator.build_draft_attn_metadatas(num_reqs, num_tokens + 1)` to refresh "
                        "per-step draft attention metadata before updating graph parameters."
                    ),
                })

        # 4. build_fia_params under PCP reading un-rebuilt input_block_tables[0] instead of slot_mappings
        if node.name == "build_fia_params":
            if "input_block_tables" in fn_src and "slot_mappings" not in fn_src and "draft_attn_metadatas" not in fn_src:
                findings.append({
                    "rule": "PCP_DRAFT_BLOCK_TABLE_NOT_REBUILT",
                    "file": str(path),
                    "line": node.lineno,
                    "function": node.name,
                    "detail": (
                        "build_fia_params reads `self.runner.block_tables.input_block_tables[0]` directly without "
                        "rebuilding `block_table` when `pcp_size > 1` (e.g. from `self.runner.block_tables.slot_mappings` "
                        "via `torch.index_select(..., dim=0, index=block_table_indices).view(num_reqs, max_blocks) // block_size`)."
                    ),
                })

        # 5. RoutedExpertsCapturer.capture writing TP-sharded topk_ids directly when dp_metadata is None
        if node.name == "capture" and "device_buffer" in fn_src and "topk_ids" in fn_src:
            has_dp_none_tp_gather = False
            for sub in ast.walk(node):
                if isinstance(sub, ast.If) and "dp_metadata is None" in _unparse(sub.test):
                    branch_src = "\n".join(_unparse(stmt) for stmt in sub.body)
                    if "all_gather" in branch_src or "_all_gather" in branch_src or "tp_size" in branch_src:
                        has_dp_none_tp_gather = True
            if not has_dp_none_tp_gather and "dp_metadata is None" in fn_src:
                findings.append({
                    "rule": "PARTIAL_BUFFER_WRITE_UNDER_TP_SHARDING",
                    "file": str(path),
                    "line": node.lineno,
                    "function": node.name,
                    "detail": (
                        "When `ctx.dp_metadata is None` (DP == 1) and `tp_size > 1` with ALLTOALL / MC2 / FUSED_MC2, "
                        "`topk_ids` is TP-sharded (`num_tokens // tp_size` rows). Writing `topk_ids` directly into "
                        "`self.device_buffer[:batch_size, layer_id, :]` leaves upper rows `[num_tokens // tp_size : num_tokens]` "
                        "unwritten (zeros or stale warmup batch data during ACLGraph replay). Gather `topk_ids` across "
                        "`get_tp_group().all_gather(topk_ids, 0)` and slice to `[:num_tokens]` before writing."
                    ),
                })

        # 6. AscendGatedDeltaNetAttention passing unsliced / num_decodes-sliced ssm_state_indices
        if "recurrent_gated_delta_rule" in fn_src:
            for sub in ast.walk(node):
                if isinstance(sub, ast.Call) and _unparse(sub.func).endswith("recurrent_gated_delta_rule"):
                    for kw in sub.keywords:
                        if kw.arg == "ssm_state_indices":
                            val = _unparse(kw.value)
                            if val == "non_spec_state_indices_tensor" or "num_decodes" in val:
                                findings.append({
                                    "rule": "PADDED_STATE_INDICES_LENGTH_MISMATCH",
                                    "file": str(path),
                                    "line": sub.lineno,
                                    "function": node.name,
                                    "detail": (
                                        f"`ssm_state_indices={val}` can exceed decode token count `T = num_decode_tokens` "
                                        "under FIA/ACLGraph request padding, triggering aclnnRecurrentGatedDeltaRule "
                                        "507015 tiling failure. Slice to `non_spec_state_indices_tensor[:num_decode_tokens]`."
                                    ),
                                })

    return findings


def audit_aclgraph_contracts(target: str | Path) -> dict[str, Any]:
    path = Path(target).resolve()
    files = _iter_py_files(path)
    violations: list[dict[str, Any]] = []
    for py_file in files:
        violations.extend(_audit_file(py_file))
    return {
        "target": str(path),
        "files_scanned": len(files),
        "contract_violation_count": len(violations),
        "contract_violations": violations,
    }


def augment_triage_with_runtime_clues(text: str, base_result: dict[str, Any]) -> dict[str, Any]:
    """Enrich CANN triage output when the log contains Python/vllm-ascend failure signatures."""
    clues: list[dict[str, Any]] = []
    actions: list[str] = []
    lower = text.lower()
    if "causal_conv" in lower or "conv_state" in lower or "pad_slot_id" in lower:
        clues.append({
            "category": "ACLGRAPH_PADDED_STATE_UNDERFLOW_OR_STOMP",
            "component": "vllm_ascend/models/glm5next/ops/causal_conv1d.py & vllm_ascend/ops/triton/kda/conv_state.py",
            "diagnosis": (
                "Decode batch padding (PAD_SLOT_ID = -1) under NPUGraph replay: fla_npu causal_conv1d_update "
                "disables null-block filtering when null_block_id < 0 (-1), causing -1 in packed indices to "
                "underflow kernel_state[-1] (507011 / state corruption), and missing `out=output` fails to "
                "preserve zeroed padded rows across graph replay."
            ),
        })
        actions.append(
            "Run `mindie-debugger aclgraph inspect --path vllm_ascend/models/glm5next/ops/causal_conv1d.py` "
            "and `mindie-debugger aclgraph inspect --path vllm_ascend/ops/triton/kda/conv_state.py`."
        )
    if "_updatable_graph_replay" in lower or "autoregressiveaclgraphmanager" in lower or "build_fia_params" in lower:
        clues.append({
            "category": "ACLGRAPH_UPDATABLE_GRAPH_STALE_METADATA",
            "component": "vllm_ascend/worker/v2/spec_decode/autoregressive/aclgraph.py & speculator.py",
            "diagnosis": (
                "AutoRegressiveAclGraphManager._updatable_graph_replay resolves FIA update tasks without calling "
                "`self.speculator.build_draft_attn_metadatas(num_reqs, num_tokens + 1)`, and `build_fia_params` "
                "reads `input_block_tables[0]` directly instead of rebuilding `block_table` from `slot_mappings` "
                "when `pcp_size > 1`."
            ),
        })
        actions.append(
            "Run `mindie-debugger aclgraph inspect --path vllm_ascend/worker/v2/spec_decode/autoregressive`."
        )
    if "routedexpertscapturer" in lower or "enable_return_routed_experts" in lower or "device_buffer" in lower:
        clues.append({
            "category": "ACLGRAPH_TP_SHARDED_PARTIAL_BUFFER_WRITE",
            "component": "vllm_ascend/patch/worker/patch_routed_experts_capture.py",
            "diagnosis": (
                "Under DP == 1 and TP > 1 with ALLTOALL / MC2 / FUSED_MC2, `topk_ids` in "
                "`RoutedExpertsCapturer.capture` is TP-sharded (`num_tokens // tp_size` rows), so writing it "
                "directly into `self.device_buffer` only updates the first `1 / tp_size` rows and replays stale "
                "warmup/zero rows in the upper batch half."
            ),
        })
        actions.append(
            "Run `mindie-debugger aclgraph inspect --path vllm_ascend/patch/worker/patch_routed_experts_capture.py`."
        )
    if clues:
        enriched = dict(base_result)
        enriched["runtime_fault_clues"] = clues
        enriched["recommended_next_steps"] = actions + list(base_result.get("recommended_next_steps", []))
        if base_result.get("fault_count", 0) == 0:
            enriched["summary"] = clues[0]["diagnosis"]
        return enriched
    return base_result
