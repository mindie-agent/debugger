"""CLI entry point (`mindie-debugger`) for CANN fault triage and NPU precision debugging.

[Role]
  Provides the `mindie-debugger` command-line interface for deterministic CANN
  `plog` / error-code triage (`triage`, `decode-code`) and the 3-primitive
  precision/memory-stomp toolkit (`precision`: `inspect-capsule`, `replay-call`
  / `call-repro`, `compare-traces` / `lockstep-diff`, `slim-moe`, `watch`).

[Invariants]
  * All subcommands emit single-line ASCII JSON to stdout and return 0 on
    success or non-zero on error/divergence.
  * Triage subcommands are zero-dependency (`torch` is imported lazily only by
    precision subcommands that load capsules or replay calls).

[Timeouts & Locks]
  Single-invocation CLI; acquires no locks and spawns no background threads.

[Side-Effect Policy]
  Reads user-specified log/capsule/trace files and writes only to explicitly
  requested `--output` / `--save-output` / `output_dir` paths.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import runpy
import sys
from pathlib import Path


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
        help="run a script/callable under ACLGraphCaptureGuard or audit source files for graph-capture contract bugs")
    guard.add_argument("target", nargs="?", help="optional path to Python script, source file, or directory")
    guard.add_argument("--script", "--path", "-p", dest="script", help="path to Python script, source file, or directory")
    guard.add_argument("--callable", help="module:qualname callable to execute")
    guard.add_argument("--strict", action="store_true", help="raise immediately on first host-sync op")
    inspect_cmd = aactions.add_parser(
        "inspect", aliases=["run", "audit"],
        help="inspect vllm-ascend / MindIE-SD source files or run a script under ACLGraphVllmInspector")
    inspect_cmd.add_argument("target", nargs="?", help="optional path to Python script, source file, or directory")
    inspect_cmd.add_argument("--script", "--path", "-p", dest="script", help="path to Python script, source file, or directory")
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
    triage = sub.add_parser("triage", help="triage Ascend NPU plog / stderr fault text or multi-rank log directory")
    triage.add_argument("--file", "-f", help="path to log file (reads stdin if omitted)")
    triage.add_argument("--root", "-r", help="path to log directory containing multi-rank *.log / plog* files")
    triage.add_argument("--text", help="inline log text to triage")
    decode = sub.add_parser("decode-code", help="decode a CANN / RTS / HCCL error code")
    decode.add_argument("code", help="error code (e.g. 507015, EZ9999, EI0006, 207008)")
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


def _run_precision(args):
    action = args.precision_action.replace("_", "-")
    if action == "call-repro":
        action = "replay-call"
    if action == "lockstep-diff":
        action = "compare-traces"
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


def _triage_directory(root_dir: Path) -> str:
    rank_pat = re.compile(r"rank[_-]?(\d+)", re.IGNORECASE)
    chunks: list[str] = []
    for path in sorted(root_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in (".log", ".txt", ".out", ".err") and not path.name.startswith("plog"):
            continue
        m = rank_pat.search(path.name) or rank_pat.search(str(path.parent))
        inferred_rank = m.group(1) if m else None
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if inferred_rank is not None and "rank_id" not in line and "rank " not in line.lower():
                chunks.append(f"[rank_id: {inferred_rank}] {line}")
            else:
                chunks.append(line)
    return "\n".join(chunks)


def _run_aclgraph(args):
    from .aclgraph import ACLGraphCaptureGuard, ACLGraphVllmInspector
    from .aclgraph_audit import audit_aclgraph_contracts
    action = args.aclgraph_action.replace("_", "-")
    if action in ("run", "audit"):
        action = "inspect"
    cwd = os.getcwd()
    if cwd not in sys.path:
        sys.path.insert(0, cwd)
    target_arg = getattr(args, "script", None) or getattr(args, "target", None)
    audit_rep = audit_aclgraph_contracts(target_arg) if target_arg and Path(target_arg).exists() else None
    is_dir = bool(target_arg and Path(target_arg).is_dir())
    if action == "guard":
        err_info = None
        with ACLGraphCaptureGuard(strict=bool(args.strict)) as guard:
            if not is_dir:
                try:
                    if target_arg:
                        runpy.run_path(str(Path(target_arg).resolve()), run_name="__main__")
                    elif args.callable:
                        _import_callable(args.callable)()
                    else:
                        return {"status": "error", "reason": "script_or_callable_required"}, 1
                except Exception as exc:
                    err_info = f"{type(exc).__name__}: {exc}"
        rep = guard.report()
        if audit_rep is not None:
            rep["contract_violations"] = audit_rep["contract_violations"]
            if audit_rep["contract_violations"]:
                rep["status"] = "violation"
        if err_info:
            if audit_rep is not None and err_info.startswith("ModuleNotFoundError:"):
                rep["runtime_skipped_reason"] = err_info
                err_info = None
            else:
                rep["exception"] = err_info
        code = 0 if (rep["status"] == "ok" and err_info is None) else 1
        return rep, code
    if action == "inspect":
        include = tuple(p.strip() for p in str(args.include).split(",") if p.strip()) or ("*",)
        err_info = None
        with ACLGraphVllmInspector(
            dump_dir=getattr(args, "dump_dir", None),
            include=include,
            compare_eager=bool(args.compare_eager),
            atol=float(args.atol),
            rtol=float(args.rtol),
        ) as inspector:
            with ACLGraphCaptureGuard(strict=False) as guard:
                if not is_dir:
                    try:
                        if target_arg:
                            runpy.run_path(str(Path(target_arg).resolve()), run_name="__main__")
                        elif args.callable:
                            _import_callable(args.callable)()
                        else:
                            return {"status": "error", "reason": "script_or_callable_required"}, 1
                    except Exception as exc:
                        err_info = f"{type(exc).__name__}: {exc}"
        rep = inspector.report()
        for v in guard.report().get("violations", []):
            if v not in rep["capture_guard_violations"]:
                rep["capture_guard_violations"].append(v)
        if audit_rep is not None:
            rep["contract_violations"] = audit_rep["contract_violations"]
            if audit_rep["contract_violations"]:
                rep["status"] = "violation"
        if rep["capture_guard_violations"]:
            rep["status"] = "violation"
        if err_info:
            if audit_rep is not None and err_info.startswith("ModuleNotFoundError:"):
                rep["runtime_skipped_reason"] = err_info
                err_info = None
            else:
                rep["exception"] = err_info
        code = 0 if (rep["status"] == "ok" and err_info is None) else 1
        return rep, code
    return {"status": "error", "reason": "unknown_aclgraph_action"}, 2


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "decode-code":
        from .npu_fault_triage import decode_cann_error_code
        print(json.dumps(decode_cann_error_code(args.code), ensure_ascii=True))
        return 0
    if args.command == "triage":
        from .aclgraph_audit import augment_triage_with_runtime_clues
        from .npu_fault_triage import triage_npu_fault
        if args.text is not None:
            text = args.text
        elif getattr(args, "root", None):
            text = _triage_directory(Path(args.root))
        elif args.file:
            text = Path(args.file).read_text(encoding="utf-8", errors="replace")
        else:
            text = sys.stdin.read()
        print(json.dumps(augment_triage_with_runtime_clues(text, triage_npu_fault(text)), ensure_ascii=True))
        return 0
    if args.command == "aclgraph":
        try:
            result, code = _run_aclgraph(args)
        except Exception as exc:
            result = {"status": "error", "category": "aclgraph_operation_failed",
                      "error_type": type(exc).__name__, "error": str(exc)[:240]}
            code = 1
        print(json.dumps(_jsonable(result), ensure_ascii=True, default=str))
        return int(code)
    if args.command == "precision":
        try:
            result, code = _run_precision(args)
        except Exception as exc:
            result = {"status": "error", "category": "precision_operation_failed",
                      "error_type": type(exc).__name__, "error": str(exc)[:240]}
            code = 1
        print(json.dumps(_jsonable(result), ensure_ascii=True, default=str))
        return int(code)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
