"""CLI entry point (`mindie-debugger`) for CANN fault triage, ACLGraph, and NPU precision debugging.

[Role]
  Provides the `mindie-debugger` command-line interface for deterministic CANN
  `plog` / error-code triage (`triage`, `decode-code`), the 3-primitive
  precision/memory-stomp toolkit (`precision`), ACLGraph capture/replay
  debugging (`aclgraph`), and remote NPU container staging (`remote-guide`).

[Invariants]
  * All subcommands emit single-line ASCII JSON to stdout.
  * Triage and `remote-guide` subcommands are zero-NPU-dependency (`torch` is
    imported lazily only by precision/aclgraph runtime hooks).
  * When an NPU-dependent command is invoked on a host without `torch_npu` /
    `vllm` / NPU runtime, `mindie-debugger` stages a clean pure-Python bundle
    locally and emits a `remote_npu_execution_guide` for `remote-dev`.

[Timeouts & Locks]
  Single-invocation CLI; acquires no locks and spawns no background threads.

[Side-Effect Policy]
  Reads user-specified log/capsule/trace files and writes only to explicitly
  requested `--output` / `--save-output` / `--stage-dir` or bundle cache paths.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import runpy
import shlex
import shutil
import sys
from pathlib import Path
from typing import Any


def _stage_remote_bundle(stage_dir: str | Path | None = None) -> Path:
    """Copy clean `.py` sources of `mindie_debugger` (no `__pycache__`) to a local staging directory."""
    pkg_src = Path(__file__).resolve().parent
    if stage_dir is not None:
        bundle_root = Path(stage_dir).expanduser().resolve()
    else:
        data_home = Path(
            os.environ.get("MINDIE_DATA_HOME")
            or (Path.home() / ".local" / "share" / "mindie-agent")
        ).expanduser().resolve()
        bundle_root = data_home / "debugger-bundle" / "mindie_debugger"
    if bundle_root.exists():
        shutil.rmtree(bundle_root)
    bundle_root.mkdir(parents=True, exist_ok=True)
    for py_file in sorted(pkg_src.rglob("*.py")):
        if "__pycache__" in py_file.parts:
            continue
        rel = py_file.relative_to(pkg_src)
        dest = bundle_root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(py_file, dest)
    return bundle_root


def _build_remote_execution_guide(
    argv: list[str] | None = None,
    stage_dir: str | Path | None = None,
) -> dict[str, Any]:
    bundle_path = _stage_remote_bundle(stage_dir)
    quoted_args = " ".join(shlex.quote(a) for a in (argv or ["triage"]))
    return {
        "summary": (
            "CANN plogs ($ASCEND_PROCESS_LOG_PATH or ~/ascend/log/debug/plog), dynamic ACLGraph tools, and "
            "NPU precision tools reside/execute inside the target Ascend NPU container. Push the staged "
            "pure-Python `mindie_debugger` bundle to the remote NPU container with "
            "`mcp__remote-dev__remote_artifact_push` and execute it in-place with `mcp__remote-dev__remote_bash`."
        ),
        "local_bundle_path": str(bundle_path),
        "remote_stage_path": ".mindie_debugger/mindie_debugger",
        "step_1_push_bundle_to_npu_container": {
            "mcp_tool": "mcp__remote-dev__remote_artifact_push",
            "arguments": {
                "local_path": str(bundle_path),
                "remote_path": ".mindie_debugger/mindie_debugger",
            },
        },
        "step_2_execute_cli_in_npu_container": {
            "mcp_tool": "mcp__remote-dev__remote_bash",
            "command": f"PYTHONPATH=.mindie_debugger:$PYTHONPATH python3 -m mindie_debugger.cli {quoted_args}",
            "cli_examples": {
                "triage_cann_plog_auto": (
                    "PYTHONPATH=.mindie_debugger:$PYTHONPATH python3 -m mindie_debugger.cli triage"
                ),
                "triage_cann_plog_by_pid": (
                    "PYTHONPATH=.mindie_debugger:$PYTHONPATH python3 -m mindie_debugger.cli triage --pid <crashed_pid>"
                ),
                "triage_custom_log_dir": (
                    "PYTHONPATH=.mindie_debugger:$PYTHONPATH python3 -m mindie_debugger.cli triage --root <log_dir_or_file>"
                ),
                "aclgraph_guard": (
                    "PYTHONPATH=.mindie_debugger:$PYTHONPATH python3 -m mindie_debugger.cli aclgraph guard --script <repro_or_test.py>"
                ),
                "aclgraph_inspect": (
                    "PYTHONPATH=.mindie_debugger:$PYTHONPATH python3 -m mindie_debugger.cli aclgraph inspect "
                    "--script <repro_or_test.py> --compare-eager --dump-dir /tmp/aclgraph_dump"
                ),
                "precision_check_call": (
                    "PYTHONPATH=.mindie_debugger:$PYTHONPATH python3 -m mindie_debugger.cli precision check-call "
                    "--callable <module:qualname>"
                ),
                "precision_replay_call": (
                    "PYTHONPATH=.mindie_debugger:$PYTHONPATH python3 -m mindie_debugger.cli precision replay-call "
                    "--capsule <capsule.pt> --det-lock"
                ),
            },
        },
        "step_3_python_api_in_npu_container": {
            "ACLGraphCaptureGuard": (
                "from mindie_debugger.aclgraph import ACLGraphCaptureGuard\n"
                "with ACLGraphCaptureGuard(strict=False) as guard:\n"
                "    # run model warmup + graph capture\n"
                "print(guard.report())  # catches .item(), .cpu(), bool(tensor), .tolist() during capture"
            ),
            "ACLGraphAddressTracker": (
                "from mindie_debugger.aclgraph import ACLGraphAddressTracker\n"
                "tracker = ACLGraphAddressTracker()\n"
                "tracker.bind_capture_inputs(meta={'slot_mapping': slot_mapping, 'seq_lens': seq_lens})\n"
                "# during graph replay:\n"
                "rep = tracker.verify_replay_inputs(actual_tokens=actual, padded_tokens=padded, "
                "slot_mapping_tensor=slot_mapping, meta={'slot_mapping': slot_mapping, 'seq_lens': seq_lens})\n"
                "print(rep)  # detects rebound_pointer and unmasked_padded_slot_mapping"
            ),
            "ACLGraphDumper": (
                "from mindie_debugger.aclgraph import ACLGraphDumper\n"
                "with ACLGraphDumper(model, include=('*attn*', '*mlp*', '*conv*'), max_steps=4, device=device) as dumper:\n"
                "    dumper.reset_step(0)  # before capture / step 0\n"
                "    dumper.advance_step()  # before subsequent g.replay() steps\n"
                "    diff = dumper.compare_with_eager(eager_fn, static_input, step_index=1, atol=1e-3)\n"
                "    print(diff['first_divergent_layer'], diff['per_layer'])"
            ),
            "ACLGraphVllmInspector": (
                "from mindie_debugger.aclgraph import ACLGraphVllmInspector\n"
                "with ACLGraphVllmInspector(dump_dir='/tmp/acl_dump', compare_eager=True) as inspector:\n"
                "    # run vllm-ascend ACLGraphWrapper capture and replay\n"
                "print(inspector.report())"
            ),
            "precision_primitives": (
                "from mindie_debugger.precision import capture_call, replay_call\n"
                "cap = capture_call(fn, *args, **kwargs)\n"
                "print(replay_call(cap, fn, mode='reference_check', reference_atol=1e-3))"
            ),
        },
    }


def _add_precision_subparsers(pactions):
    inspect_capsule = pactions.add_parser(
        "inspect-capsule", aliases=["inspect_capsule"],
        help="print a saved call capsule summary")
    inspect_capsule.add_argument("capsule", nargs="?")
    inspect_capsule.add_argument("--capsule", "--capsule-path", "--path", dest="capsule_opt")
    replay = pactions.add_parser(
        "replay-call", aliases=["replay_call", "call-repro"],
        help="replay one captured call from its pre-call state")
    replay.add_argument("capsule", nargs="?")
    replay.add_argument("--capsule", "--capsule-path", dest="capsule_opt")
    replay.add_argument("--callable", help="explicit module:qualname callable")
    replay.add_argument("--repeat", type=int, default=1)
    replay.add_argument("--det-lock", action="store_true")
    replay.add_argument("--reference", help="explicit reference module:qualname")
    replay.add_argument("--reference-atol", type=float, default=1e-3)
    replay.add_argument("--reference-rtol", type=float, default=1e-3)
    replay.add_argument("--save-output")
    check_call = pactions.add_parser(
        "check-call", aliases=["check_call"],
        help="capture and run FP32 reference_check + sensitivity probes on a callable")
    check_call.add_argument("--callable", required=True, help="module:qualname returning (fn, args, kwargs) or zero-arg fn")
    check_call.add_argument("--reference-atol", type=float, default=1e-3)
    check_call.add_argument("--reference-rtol", type=float, default=1e-3)
    compare = pactions.add_parser(
        "compare-traces", aliases=["compare_traces", "lockstep-diff"],
        help="compare two lockstep JSON reports")
    compare.add_argument("baseline", nargs="?")
    compare.add_argument("target", nargs="?")
    compare.add_argument("--baseline", "--baseline-file", dest="baseline_opt")
    compare.add_argument("--target", "--target-file", dest="target_opt")
    compare.add_argument("--output")
    slim = pactions.add_parser(
        "slim-moe", aliases=["slim_moe"],
        help="slice routed experts in a safetensors MoE checkpoint")
    slim.add_argument("input_dir", nargs="?")
    slim.add_argument("output_dir", nargs="?")
    slim.add_argument("target", nargs="?", type=int)
    slim.add_argument("--input-dir", "--input", dest="input_dir_opt")
    slim.add_argument("--output-dir", "--output", dest="output_dir_opt")
    slim.add_argument("--target-routed-experts", dest="target_opt", type=int)
    watch = pactions.add_parser(
        "watch", help="inspect an exported npu-watch JSON record file")
    watch.add_argument("records", nargs="?")
    watch.add_argument("--records", "--file", dest="records_opt")
    watch.add_argument("--latch", action="store_true")


def _add_aclgraph_subparsers(aactions):
    guard = aactions.add_parser(
        "guard",
        help="run a script/callable under ACLGraphCaptureGuard to catch host-device syncs during graph capture")
    guard.add_argument("target", nargs="?", help="optional path to Python script")
    guard.add_argument("--script", "--path", "-p", dest="script", help="path to Python script")
    guard.add_argument("--callable", help="module:qualname callable to execute")
    guard.add_argument("--strict", action="store_true", help="raise immediately on first host-sync op")
    inspect_cmd = aactions.add_parser(
        "inspect", aliases=["run", "audit", "dump"],
        help="run a script/callable under ACLGraphVllmInspector (capture guard + address tracker + layer dumper)")
    inspect_cmd.add_argument("target", nargs="?", help="optional path to Python script")
    inspect_cmd.add_argument("--script", "--path", "-p", dest="script", help="path to Python script")
    inspect_cmd.add_argument("--callable", help="module:qualname callable to execute")
    inspect_cmd.add_argument("--dump-dir", help="optional directory to save per-layer ACLGraph dumps")
    inspect_cmd.add_argument("--include", default="*", help="comma-separated module glob patterns to dump")
    inspect_cmd.add_argument("--compare-eager", action="store_true", help="compare replayed ACLGraph layers against eager")
    inspect_cmd.add_argument("--atol", type=float, default=1e-3)
    inspect_cmd.add_argument("--rtol", type=float, default=1e-3)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        prog="mindie-debugger",
        description="Ascend NPU CANN fault triage, 3-primitive precision/memory-stomp, and ACLGraph debugger")
    sub = result.add_subparsers(dest="command", required=True)
    triage = sub.add_parser(
        "triage",
        help="triage Ascend CANN plog / stderr fault text or multi-rank log directory (auto-discovers $ASCEND_PROCESS_LOG_PATH or ~/ascend/log in NPU container)",
    )
    triage.add_argument("target", nargs="?", help="optional path to log file or CANN plog directory")
    triage.add_argument("--file", "-f", help="path to log file")
    triage.add_argument("--root", "-r", help="path to log directory containing *.log / plog* files (defaults to $ASCEND_PROCESS_LOG_PATH or ~/ascend/log)")
    triage.add_argument("--text", help="inline log / traceback text to triage")
    triage.add_argument("--pid", type=int, help="optional process ID filter for plog-<pid>_*.log files")
    triage.add_argument("--latest", type=int, default=32, help="max number of recent log files to scan in --root (default: 32)")
    triage.add_argument("--window-sec", type=float, default=600.0, help="time window in seconds around the newest fault log in --root (default: 600s)")
    decode = sub.add_parser("decode-code", help="decode a CANN / RTS / HCCL error code")
    decode.add_argument("code", help="error code (e.g. 507015, EZ9999, EI0006, 207008)")
    guide = sub.add_parser(
        "remote-guide", aliases=["guide", "stage-remote", "how-to-run"],
        help="stage the pure-Python mindie_debugger bundle and print remote-dev NPU container execution instructions")
    guide.add_argument("--stage-dir", help="optional local directory to write the clean mindie_debugger bundle")
    precision = sub.add_parser(
        "precision", help="storage-graph precision debugging: inspect capsule, replay, check-call, diff, slim MoE")
    pactions = precision.add_subparsers(dest="precision_action", required=True)
    _add_precision_subparsers(pactions)
    aclgraph = sub.add_parser(
        "aclgraph", help="ACLGraph (torch.npu.NPUGraph / vllm-ascend ACLGraphWrapper) capture guard, address tracker & layer dumper")
    aactions = aclgraph.add_subparsers(dest="aclgraph_action", required=True)
    _add_aclgraph_subparsers(aactions)
    return result


def _jsonable(value, depth=0):
    """Project runtime objects (tensors, exceptions) into JSON-safe values."""
    if depth > 8:
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item, depth + 1) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item, depth + 1) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "detach") and hasattr(value, "shape") and hasattr(value, "dtype"):
        return {"type": "tensor", "dtype": str(value.dtype),
                "shape": [int(item) for item in value.shape],
                "device": str(getattr(value, "device", "cpu"))}
    return str(value)


def _read_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _import_callable(spec):
    if not spec:
        return None
    import importlib
    cwd = os.getcwd()
    if cwd not in sys.path:
        sys.path.insert(0, cwd)
    module_name, separator, qualname = spec.partition(":")
    if not separator or not module_name:
        raise ValueError("callable must be module:qualname")
    target = importlib.import_module(module_name)
    for part in qualname.split("."):
        target = getattr(target, part)
    if not callable(target):
        raise ValueError(f"{spec} is not callable")
    return target


def _precision_compare(baseline_path, target_path):
    baseline = _read_json(baseline_path)
    target = _read_json(target_path)

    def index(payload):
        rows = payload.get("metrics") if isinstance(payload, dict) else payload
        if not isinstance(rows, list):
            return None
        indexed = {}
        for row in rows:
            if not isinstance(row, dict):
                continue
            key = (row.get("module_path", ""), row.get("invocation_ordinal", 0),
                   row.get("tree_path", row.get("path", "")), row.get("rank", 0))
            indexed[str(key)] = row
        return indexed

    left, right = index(baseline), index(target)
    if left is None or right is None:
        return {"status": "unsupported", "reason": "metric_rows_required"}
    baseline_status = baseline.get("status") if isinstance(baseline, dict) else None
    target_status = target.get("status") if isinstance(target, dict) else None
    if baseline_status not in (None, "ok") or target_status not in (None, "ok"):
        return {"status": "diverged", "reason": "trace_status_not_ok",
                "baseline_status": baseline_status, "target_status": target_status}
    if set(left) != set(right) or not left:
        return {"status": "diverged", "reason": "metric_keys_mismatch",
                "baseline_metrics": len(left), "target_metrics": len(right),
                "compared": len(set(left) & set(right))}
    numeric = ("max_abs_err", "rel_l2_err", "ulp_max", "sjr", "jitter_max",
               "nonfinite_mismatch_count", "sample_count")
    deltas = {}
    for key in sorted(set(left) & set(right)):
        for field in numeric:
            before, after = left[key].get(field), right[key].get(field)
            if isinstance(before, (int, float)) and isinstance(after, (int, float)):
                deltas.setdefault(field, []).append(float(after) - float(before))
    summary = {}
    for field, values in deltas.items():
        values = sorted(values)
        summary[field] = {
            "count": len(values),
            "max_delta": max(abs(item) for item in values),
            "median_delta": values[len(values) // 2],
        }
    return {
        "status": "ok",
        "compared": len(set(left) & set(right)),
        "baseline_metrics": len(left),
        "target_metrics": len(right),
        "deltas": summary,
    }


def _is_npu_env_missing_error(exc: BaseException) -> bool:
    if isinstance(exc, ModuleNotFoundError):
        return True
    msg = f"{type(exc).__name__}: {exc}".lower()
    return any(tok in msg for tok in ("torch_npu", "vllm", "npu", "ascend", "cann", "triton", "mindie"))


def _run_precision(args, raw_argv: list[str] | None = None):
    action = args.precision_action.replace("_", "-")
    if action == "call-repro":
        action = "replay-call"
    if action == "lockstep-diff":
        action = "compare-traces"
    try:
        if action == "inspect-capsule":
            from .precision import capsule_summary
            capsule = getattr(args, "capsule", None) or args.capsule_opt
            if not capsule:
                return {"status": "error", "reason": "capsule_required"}, 1
            return capsule_summary(capsule), 0
        if action == "replay-call":
            from .precision import replay_call
            capsule = getattr(args, "capsule", None) or args.capsule_opt
            if not capsule:
                return {"status": "error", "reason": "capsule_required"}, 1
            import torch
            from .precision.call_repro import _apply_det_lock
            _apply_det_lock(torch, args.det_lock)
            fn = _import_callable(args.callable)
            reference = _import_callable(args.reference)
            result = replay_call(
                capsule, fn, repeat=args.repeat, det_lock=args.det_lock,
                reference_fn=reference, reference_atol=args.reference_atol,
                reference_rtol=args.reference_rtol)
            if args.save_output and result.get("outputs") is not None:
                import torch
                torch.save(result["outputs"], args.save_output)
            return _jsonable(result), (0 if result.get("status") == "ok" else 1)
        if action == "check-call":
            from .precision import capture_call, replay_call
            target = _import_callable(args.callable)
            spec = target()
            if isinstance(spec, tuple) and len(spec) >= 2 and callable(spec[0]):
                fn = spec[0]
                call_args = spec[1] if isinstance(spec[1], (list, tuple)) else (spec[1],)
                call_kwargs = spec[2] if len(spec) >= 3 and isinstance(spec[2], dict) else {}
            else:
                fn, call_args, call_kwargs = target, (), {}
            cap = capture_call(fn, *call_args, **call_kwargs)
            result = replay_call(
                cap, fn, mode="reference_check",
                reference_atol=args.reference_atol, reference_rtol=args.reference_rtol)
            return _jsonable(result), (0 if result.get("status") == "ok" else 1)
    except Exception as exc:
        if _is_npu_env_missing_error(exc):
            return {
                "status": "remote_npu_execution_required",
                "local_missing_environment": f"{type(exc).__name__}: {exc}",
                "remote_npu_execution_guide": _build_remote_execution_guide(raw_argv),
            }, 0
        raise
    if action == "compare-traces":
        baseline = getattr(args, "baseline", None) or args.baseline_opt
        target = getattr(args, "target", None) or args.target_opt
        if not baseline or not target:
            return {"status": "error", "reason": "baseline_and_target_required"}, 1
        result = _precision_compare(baseline, target)
        if args.output:
            with open(args.output, "w", encoding="utf-8") as handle:
                json.dump(result, handle, ensure_ascii=True)
                handle.write("\n")
        return result, 0 if result.get("status") == "ok" else 1
    if action == "slim-moe":
        from .precision.stimulus import slim_moe_checkpoint
        input_dir = getattr(args, "input_dir", None) or args.input_dir_opt
        output_dir = getattr(args, "output_dir", None) or args.output_dir_opt
        target = getattr(args, "target", None) or args.target_opt
        if not input_dir or not output_dir or not target:
            return {"status": "error", "reason": "input_output_target_required"}, 1
        return _jsonable(slim_moe_checkpoint(input_dir, output_dir, target)), 0
    if action == "watch":
        records = getattr(args, "records", None) or args.records_opt
        if not records:
            return {"status": "unsupported", "reason": "records_file_required"}, 1
        payload = _read_json(records)
        if args.latch:
            return _jsonable(payload.get("latch", payload)), 0
        return _jsonable(payload), 0
    return {"status": "error", "reason": "unknown_precision_action"}, 2


_FAULT_HINT_RE = re.compile(
    r"\[ERROR\]|\bERROR\b|E[A-Z0-9][0-9][0-9A-Z]{3}|\b(?:107|161|207|507|561)\d{3}\b|"
    r"aicore\s+error|aic\s+error|HCCL_E_|RuntimeError|Traceback",
    re.IGNORECASE,
)


def _read_bounded_log_text(path: Path, max_bytes: int = 4 * 1024 * 1024) -> str:
    try:
        size = path.stat().st_size
    except OSError:
        return ""
    if size <= max_bytes:
        return path.read_text(encoding="utf-8", errors="replace")
    head_bytes = 256 * 1024
    tail_bytes = max_bytes - head_bytes
    with path.open("rb") as fh:
        head = fh.read(head_bytes).decode("utf-8", errors="replace")
        fh.seek(max(head_bytes, size - tail_bytes))
        tail = fh.read(tail_bytes).decode("utf-8", errors="replace")
    return head + "\n...[truncated middle of large plog]...\n" + tail


_CRITICAL_ASCEND_ENV_KEYS = (
    "ASCEND_TOOLKIT_HOME",
    "ATB_HOME_PATH",
    "HCCL_DETERMINISTIC",
    "LCCL_DETERMINISTIC",
    "VLLM_BATCH_INVARIANT",
    "ASCEND_LAUNCH_BLOCKING",
    "PYTORCH_NPU_ALLOC_CONF",
)


def _inspect_live_process_env(pid: int | None, proc_root: Path | None = None) -> dict[str, Any] | None:
    """Compare live /proc/<pid>/environ and /proc/<pid>/maps against the current shell environment."""
    if pid is None or int(pid) <= 0:
        return None
    root = proc_root if proc_root is not None else Path("/proc")
    proc_dir = root / str(int(pid))
    env_file = proc_dir / "environ"
    maps_file = proc_dir / "maps"
    if not env_file.is_file():
        return None
    try:
        with env_file.open("rb") as fh:
            raw_env = fh.read(256 * 1024)
    except TimeoutError:
        raise
    except OSError:
        return None
    proc_env: dict[str, str] = {}
    for item in raw_env.split(b"\x00"):
        if b"=" in item:
            k, _, v = item.decode("utf-8", errors="replace").partition("=")
            proc_env[k] = v
    drift: dict[str, dict[str, str | None]] = {}
    selected_env: dict[str, str] = {}
    for key in _CRITICAL_ASCEND_ENV_KEYS:
        p_val = proc_env.get(key)
        s_val = os.environ.get(key)
        if p_val is not None:
            selected_env[key] = p_val
        if p_val != s_val:
            drift[key] = {"process_value": p_val, "shell_value": s_val}
    cann_so_paths: list[str] = []
    if maps_file.is_file():
        try:
            with maps_file.open("rb") as fh:
                maps_text = fh.read(1024 * 1024).decode("utf-8", errors="replace")
            seen_so: set[str] = set()
            for line in maps_text.splitlines():
                parts = line.split(None, 5)
                if len(parts) < 6:
                    continue
                so_path = parts[5].strip()
                if ".so" in so_path and any(
                    tok in so_path.lower() for tok in ("ascend", "cann", "atb", "hccl", "_c_ascend")
                ):
                    if so_path not in seen_so:
                        seen_so.add(so_path)
                        cann_so_paths.append(so_path)
        except TimeoutError:
            raise
        except OSError:
            pass
    return {
        "pid": int(pid),
        "process_env": selected_env,
        "env_drift_against_shell": drift,
        "has_env_drift": bool(drift),
        "loaded_ascend_so_count": len(cann_so_paths),
        "loaded_ascend_so_sample": cann_so_paths[:16],
    }


def _default_cann_log_root() -> Path | None:
    env_path = os.environ.get("ASCEND_PROCESS_LOG_PATH", "").strip()
    if env_path:
        candidate = Path(env_path).expanduser()
        if candidate.exists():
            return candidate
    home_cann = Path.home() / "ascend" / "log"
    if home_cann.exists():
        return home_cann
    return None


def _triage_directory(
    root_dir: Path,
    *,
    pid: int | None = None,
    latest: int = 32,
    window_sec: float = 600.0,
) -> tuple[str, list[str]]:
    rank_pat = re.compile(r"(?:rank|device)[_-]?(\d+)", re.IGNORECASE)
    pid_token = str(pid) if pid is not None else None
    candidates: list[tuple[float, Path]] = []
    for path in root_dir.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in (".log", ".txt", ".out", ".err") and not path.name.startswith(("plog", "device-")):
            continue
        if pid_token is not None and f"plog-{pid_token}_" not in path.name and pid_token not in path.name:
            continue
        try:
            mtime = path.stat().st_mtime
        except OSError:
            mtime = 0.0
        candidates.append((mtime, path))
    if not candidates:
        return "", []
    candidates.sort(key=lambda item: (item[0], str(item[1])), reverse=True)
    anchor_mtime = candidates[0][0]
    for mtime, path in candidates[: max(latest * 2, 64)]:
        if _FAULT_HINT_RE.search(_read_bounded_log_text(path, max_bytes=512 * 1024)):
            anchor_mtime = mtime
            break
    if window_sec > 0 and len(candidates) > latest:
        windowed = [item for item in candidates if item[0] >= anchor_mtime - window_sec]
        selected = (windowed or candidates)[: max(1, latest)]
    else:
        selected = candidates[: max(1, latest)]
    selected_paths = [item[1] for item in sorted(selected, key=lambda item: str(item[1]))]
    chunks: list[str] = []
    for path in selected_paths:
        m = rank_pat.search(path.name) or rank_pat.search(str(path.parent))
        inferred_rank = m.group(1) if m else None
        for line in _read_bounded_log_text(path).splitlines():
            if inferred_rank is not None and "rank_id" not in line and "rank " not in line.lower():
                chunks.append(f"[rank_id: {inferred_rank}] {line}")
            else:
                chunks.append(line)
    return "\n".join(chunks), [str(p) for p in selected_paths]


def _run_aclgraph(args, raw_argv: list[str] | None = None):
    from .aclgraph import ACLGraphCaptureGuard, ACLGraphVllmInspector
    action = args.aclgraph_action.replace("_", "-")
    if action in ("run", "audit", "dump"):
        action = "inspect"
    cwd = os.getcwd()
    if cwd not in sys.path:
        sys.path.insert(0, cwd)
    target_arg = getattr(args, "script", None) or getattr(args, "target", None)
    if not target_arg and not getattr(args, "callable", None):
        return {
            "status": "remote_npu_execution_required",
            "reason": "Provide --script <repro.py> or --callable <mod:fn> inside the NPU container",
            "remote_npu_execution_guide": _build_remote_execution_guide(raw_argv),
        }, 0
    if target_arg and Path(target_arg).exists() and Path(target_arg).is_dir():
        return {
            "status": "remote_npu_execution_required",
            "reason": f"'{target_arg}' is a source directory; dynamic ACLGraph inspection requires running a script or pytest inside the NPU container.",
            "remote_npu_execution_guide": _build_remote_execution_guide(raw_argv),
        }, 0
    if action == "guard":
        err_info = None
        exc_obj = None
        with ACLGraphCaptureGuard(strict=bool(args.strict)) as guard:
            try:
                if target_arg:
                    runpy.run_path(str(Path(target_arg).resolve()), run_name="__main__")
                elif args.callable:
                    _import_callable(args.callable)()
            except Exception as exc:
                exc_obj = exc
                err_info = f"{type(exc).__name__}: {exc}"
        rep = guard.report()
        if exc_obj is not None and _is_npu_env_missing_error(exc_obj) and not rep.get("violations"):
            return {
                "status": "remote_npu_execution_required",
                "local_missing_environment": err_info,
                "remote_npu_execution_guide": _build_remote_execution_guide(raw_argv),
            }, 0
        if err_info:
            rep["exception"] = err_info
        code = 0 if (rep["status"] == "ok" and err_info is None) else 1
        return rep, code
    if action == "inspect":
        include = tuple(p.strip() for p in str(args.include).split(",") if p.strip()) or ("*",)
        err_info = None
        exc_obj = None
        with ACLGraphVllmInspector(
            dump_dir=getattr(args, "dump_dir", None),
            include=include,
            compare_eager=bool(args.compare_eager),
            atol=float(args.atol),
            rtol=float(args.rtol),
        ) as inspector:
            with ACLGraphCaptureGuard(strict=False) as guard:
                try:
                    if target_arg:
                        runpy.run_path(str(Path(target_arg).resolve()), run_name="__main__")
                    elif args.callable:
                        _import_callable(args.callable)()
                except Exception as exc:
                    exc_obj = exc
                    err_info = f"{type(exc).__name__}: {exc}"
        rep = inspector.report()
        for v in guard.report().get("violations", []):
            if v not in rep["capture_guard_violations"]:
                rep["capture_guard_violations"].append(v)
        if rep["capture_guard_violations"]:
            rep["status"] = "violation"
        if exc_obj is not None and _is_npu_env_missing_error(exc_obj) and rep["status"] == "ok":
            return {
                "status": "remote_npu_execution_required",
                "local_missing_environment": err_info,
                "remote_npu_execution_guide": _build_remote_execution_guide(raw_argv),
            }, 0
        if err_info:
            rep["exception"] = err_info
        code = 0 if (rep["status"] == "ok" and err_info is None) else 1
        return rep, code
    return {"status": "error", "reason": "unknown_aclgraph_action"}, 2


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(argv) if argv is not None else sys.argv[1:]
    args = parser().parse_args(raw_argv)
    if args.command in ("remote-guide", "guide", "stage-remote", "how-to-run"):
        guide = _build_remote_execution_guide( stage_dir=getattr(args, "stage_dir", None))
        print(json.dumps({"status": "ok", "remote_npu_execution_guide": guide}, ensure_ascii=True))
        return 0
    if args.command == "decode-code":
        from .npu_fault_triage import decode_cann_error_code
        print(json.dumps(decode_cann_error_code(args.code), ensure_ascii=True))
        return 0
    if args.command == "triage":
        from .npu_fault_triage import triage_npu_fault
        scanned_files: list[str] = []
        explicit_path = getattr(args, "root", None) or getattr(args, "file", None) or getattr(args, "target", None)
        if args.text is not None:
            text = args.text
        elif explicit_path:
            target_path = Path(explicit_path).expanduser()
            if not target_path.exists():
                print(json.dumps({
                    "status": "remote_npu_execution_required",
                    "reason": (
                        f"Log path '{target_path}' does not exist on the local host; CANN plogs "
                        "($ASCEND_PROCESS_LOG_PATH or ~/ascend/log/debug/plog) reside inside the remote NPU container."
                    ),
                    "remote_npu_execution_guide": _build_remote_execution_guide(raw_argv),
                }, ensure_ascii=True))
                return 0
            if target_path.is_dir():
                text, scanned_files = _triage_directory(
                    target_path, pid=args.pid, latest=args.latest, window_sec=args.window_sec
                )
            else:
                text, scanned_files = _read_bounded_log_text(target_path), [str(target_path)]
        else:
            stdin_text = ""
            if not sys.stdin.isatty() and args.pid is None:
                try:
                    stdin_text = sys.stdin.read()
                except TimeoutError:
                    raise
                except OSError:
                    stdin_text = ""
            if stdin_text.strip():
                text, scanned_files = stdin_text, ["<stdin>"]
            else:
                default_root = _default_cann_log_root()
                if default_root is not None:
                    text, scanned_files = _triage_directory(
                        default_root, pid=args.pid, latest=args.latest, window_sec=args.window_sec
                    )
                else:
                    print(json.dumps({
                        "status": "remote_npu_execution_required",
                        "reason": (
                            "No inline --text provided and default CANN plog directory "
                            "($ASCEND_PROCESS_LOG_PATH or ~/ascend/log) was not found on the local host. "
                            "Push mindie_debugger to the remote NPU container and run `triage` there, "
                            "or pass `--text '<error snippet>'`."
                        ),
                        "remote_npu_execution_guide": _build_remote_execution_guide(["triage"]),
                    }, ensure_ascii=True))
                    return 0
        payload = triage_npu_fault(text)
        if scanned_files:
            payload["scanned_file_count"] = len(scanned_files)
            payload["scanned_files"] = scanned_files[:16]
        live_env = _inspect_live_process_env(args.pid or payload.get("root_cause_pid"))
        if live_env is not None:
            payload["live_process_env_audit"] = live_env
        payload["remote_npu_execution_guide"] = _build_remote_execution_guide(raw_argv)
        print(json.dumps(payload, ensure_ascii=True))
        return 0
    if args.command == "aclgraph":
        try:
            result, code = _run_aclgraph(args, raw_argv)
        except Exception as exc:
            result = {"status": "error", "category": "aclgraph_operation_failed",
                      "error_type": type(exc).__name__, "error": str(exc)[:240]}
            code = 1
        print(json.dumps(_jsonable(result), ensure_ascii=True, default=str))
        return int(code)
    if args.command == "precision":
        try:
            result, code = _run_precision(args, raw_argv)
        except Exception as exc:
            result = {"status": "error", "category": "precision_operation_failed",
                      "error_type": type(exc).__name__, "error": str(exc)[:240]}
            code = 1
        print(json.dumps(_jsonable(result), ensure_ascii=True, default=str))
        return int(code)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
