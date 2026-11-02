"""
[Role]
Measure two deterministic CPU-only debugger workloads for formal releases:
Ascend CANN fault triage over a realistic HCCL-timeout plus HBM-ECC cascade and
vllm-ascend ACLGraph stream-capture plus TORCHAIR DFX root-cause classification.

[Invariants]
Business output is asserted before any latency is emitted. Triage selects the
HBM/driver root cause over cascading HCCL peer timeouts and accurately classifies
vllm-ascend ACLGraph stream-capture exhaustion (`207008`/`EL0009`/`EZ3002`) with
deterministic remediation guidance. The numbers are component microbenchmark
timings, not hardware or productivity claims.

[Timeouts & Locks]
Fixed bounded iteration counts; no network, subprocess, sleeps, or locks.

[Side-Effect Policy]
Inputs are in-memory constants. Filesystem writes occur only to caller-requested
paths (`MINDIE_BENCH_RESULT_PATH`, `--output-json`, `--output-markdown`).
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mindie_debugger import decode_cann_error_code, triage_npu_fault  # noqa: E402

TRIAGE_ITERATIONS = 500
ACLGRAPH_ITERATIONS = 500

_HCCL_HBM_PLOG = "\n".join([
    "[ERROR] HCCL(4211,4220):2026-09-22-14:55:01.000.000 "
    "[hccl_comm.cc:88] rank[1] HCCL_E_WAIT_TIMEOUT "
    "Notify wait timeout peer rank 0",
    "[ERROR] RUNTIME(4211,4219):2026-09-22-14:54:51.635.208 "
    "[device_error_core_proc.cc:1227] rank[0] deviceId:0 streamID:[90] "
    "taskID[686] PrintCoreInfo:An error occurs on the device, "
    "HBM ECC uncorrectable error detected, failed operator: MatMulV2, "
    "aic error mask: 0x800000",
    "[ERROR] DRIVER(4211,4221):2026-09-22-14:54:52.000.000 "
    "rank[0] deviceId:0 ErrCode=E89999, "
    "desc=[HBM ECC uncorrectable fault on device 0]",
    "[ERROR] HCCL(4211,4220):2026-09-22-14:55:02.000.000 "
    "[hccl_socket.cc:99] rank[1] ErrCode=EI0006 Getting socket times out",
])

_ACLGRAPH_PLOG = "\n".join([
    "[ERROR] RUNTIME(9001,9002):2026-09-22-18:00:00.000.000 "
    "rank[0] deviceId:0 streamID:[14] taskID[312] AclmdlRICaptureBegin failed: "
    "ErrCode=207008, EL0009 Insufficient_Stream_Resources during "
    "cudagraph_capture_sizes capture, failed operator: AddRmsNorm",
    "[ERROR] TORCHAIR(9001,9003):2026-09-22-18:00:00.100.000 "
    "rank[0] deviceId:0 torch.npugraph_ex BackendCompilerFailed: "
    "aclgraph stream capture fail, failed operator: AddRmsNorm, ErrCode=EZ3002",
    "[ERROR] HCCL(9001,9008):2026-09-22-18:00:12.000.000 "
    "rank[1] ErrCode=EI0002 Notify wait timeout peer rank 0",
])


def _triage_hccl_hbm() -> dict:
    return triage_npu_fault(_HCCL_HBM_PLOG)


def _verify_triage_hccl_hbm(result: dict) -> None:
    assert result["fault_domain"] == "device_oom_or_driver"
    assert result["error_codes"][0] == "E89999"
    assert "HBM ECC" in result["root_cause_line"]
    assert result["root_cause_rank"] == "0"
    assert result["device_id"] == "0"
    assert result["failed_operator"] == "MatMulV2"
    assert result["decoded_codes"][0]["subsystem"] == "driver_memory"
    assert result["cascading_timeout_detected"] is True
    assert result["cascading_rank"] == "1"
    assert len(result["cascading_timeout_lines"]) == 2
    assert "npu-smi" in result["recommended_tools"]
    assert "HCCL_ENTRY_LOG_ENABLE=1" in result["recommended_tools"]


def _triage_aclgraph() -> dict:
    decoded = decode_cann_error_code("507015")
    assert decoded["recognized"] is True and decoded["subsystem"] == "aicore_kernel"
    return triage_npu_fault(_ACLGRAPH_PLOG)


def _verify_triage_aclgraph(result: dict) -> None:
    assert result["fault_domain"] == "vllm_ascend_aclgraph_capture_failure"
    assert result["model_fault_category"] == "vllm_ascend_aclgraph_capture_failure"
    assert {"EL0009", "207008", "EZ3002"} <= set(result["error_codes"])
    assert result["graph_dfx_detected"] is True
    assert result["graph_dfx_stage"] == "compile_or_capture"
    assert result["failed_operator"] == "AddRmsNorm"
    assert result["cascading_timeout_detected"] is True
    assert result["cascading_rank"] == "1"
    assert any("FULL_DECODE_ONLY" in step for step in result["remediation_steps"])


def _measure(name: str, iterations: int, operation, verify) -> dict:
    verify(operation())
    started = time.perf_counter()
    for _ in range(iterations):
        result = operation()
    elapsed = time.perf_counter() - started
    verify(result)
    value = elapsed * 1_000_000 / iterations
    assert math.isfinite(value) and value > 0
    return {
        "name": name,
        "unit": "us/op",
        "value": value,
        "iterations": iterations,
    }


def _render_markdown(metrics: list[dict]) -> str:
    lines = [
        "# mindie-debugger Release Benchmark Results",
        "",
        "| Metric | Value | Unit | Iterations |",
        "| --- | ---: | --- | ---: |",
    ]
    for item in metrics:
        lines.append(
            f"| `{item['name']}` | {item['value']:.2f} | `{item['unit']}` | {item['iterations']} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run mindie-debugger release microbenchmarks.")
    parser.add_argument("cases", nargs="*", help="Benchmark case IDs to execute.")
    parser.add_argument("--output-json", help="Optional path to write JSON benchmark payload.")
    parser.add_argument("--output-markdown", help="Optional path to write Markdown benchmark summary.")
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    workloads = {
        "npu-fault-triage": (
            "npu_fault_triage_latency",
            TRIAGE_ITERATIONS,
            _triage_hccl_hbm,
            _verify_triage_hccl_hbm,
        ),
        "aclgraph-capture-triage": (
            "aclgraph_capture_triage_latency",
            ACLGRAPH_ITERATIONS,
            _triage_aclgraph,
            _verify_triage_aclgraph,
        ),
    }
    selected = list(args.cases) or list(workloads)
    unknown = [name for name in selected if name not in workloads]
    if unknown:
        raise SystemExit(
            "unknown benchmark case(s): " + ", ".join(unknown)
            + "; choose " + ", ".join(workloads)
        )

    metrics = [
        _measure(*workloads[name])
        for name in selected
    ]
    payload = {"status": "pass", "metrics": metrics}
    bench_env_path = os.environ.get("MINDIE_BENCH_RESULT_PATH")
    if bench_env_path:
        Path(bench_env_path).write_text(
            json.dumps(payload) + "\n",
            encoding="utf-8",
            newline="\n",
        )
    if args.output_json:
        out_json = Path(args.output_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(
            json.dumps(payload, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
    if args.output_markdown:
        out_md = Path(args.output_markdown)
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text(
            _render_markdown(metrics),
            encoding="utf-8",
            newline="\n",
        )
    if not bench_env_path and not args.output_json and not args.output_markdown:
        print(json.dumps(payload))


if __name__ == "__main__":
    main()
