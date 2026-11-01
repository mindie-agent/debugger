"""ACLGraph / NPUGraph capture guard, static-pointer tracker, and in-graph dumper.

[Role]
  Provide deterministic debugging tools tailored for ``ACLGraph`` / ``NPUGraph``
  execution in ``vllm-ascend`` and ``MindIE-SD``:
  1. ``ACLGraphCaptureGuard``: catch forbidden host-device syncs (``.item()``,
     ``.cpu()``, ``bool(tensor)``, ``.tolist()``) inside graph capture regions.
  2. ``ACLGraphAddressTracker``: detect ``rebound_pointer`` bugs (where replay
     inputs allocate a new ``data_ptr()`` instead of ``.copy_()`` into the
     captured static buffer) and unmasked padded ``slot_mapping`` tails.
  3. ``ACLGraphDumper``: bake capture-safe, device-indexed D2D tensor copies and
     ``WatchRingBuffer`` probes into the captured graph so intermediate layer
     outputs during ``g.replay()`` can be dumped and diffed against Eager mode.

[Invariants]
  * ``ACLGraphDumper`` performs zero ``.item()``, zero host truth tests, and
    zero dynamic memory allocations inside the hooked forward pass after
    initial slot binding, so ``torch.npu.graph`` capture succeeds cleanly.
  * ``advance_step()`` updates the active replay slot via an in-place device
    kernel (``add_(1)``) without synchronizing the host.

[Timeouts & Locks]
  In-process hooks and ``TorchFunctionMode`` only; acquires no locks and spawns
  no threads or subprocesses.

[Side-Effect Policy]
  Allocates bounded static snapshot buffers on the target device when
  ``ACLGraphDumper`` is attached; removes all forward hooks on ``close()`` or
  context-manager exit.
"""
from __future__ import annotations

import fnmatch
import traceback
from typing import Any, Callable

from .precision.metrics import _tensor_metrics
from .precision.npu_watch import WatchRingBuffer
from .precision.storage_graph import _require_torch, _storage_identity
from .precision.tree_utils import flatten_tree_with_paths, path_to_string

_SYNC_FUNC_NAMES = frozenset({
    "item",
    "tolist",
    "__bool__",
    "__index__",
    "numpy",
    "nonzero",
})


def _first_user_frame() -> str:
    for frame in reversed(traceback.extract_stack()):
        if "mindie_debugger" not in frame.filename and "torch/overrides" not in frame.filename:
            return f"{frame.filename}:{frame.lineno} in {frame.name}"
    return "unknown:0"


def _first_tensor(value: Any):
    torch = _require_torch()
    if isinstance(value, torch.Tensor):
        return value
    for _, leaf in flatten_tree_with_paths(value):
        if isinstance(leaf, torch.Tensor):
            return leaf
    return None


class ACLGraphCaptureGuard:
    """Detect forbidden host-device synchronizations inside an ACLGraph capture region."""

    def __init__(self, *, strict: bool = False):
        self.strict = bool(strict)
        self.violations: list[dict[str, Any]] = []
        self._mode = None

    def _record(self, op_name: str, tensor: Any) -> None:
        entry = {
            "op": op_name,
            "tensor_shape": list(tensor.shape) if hasattr(tensor, "shape") else None,
            "tensor_dtype": str(tensor.dtype) if hasattr(tensor, "dtype") else None,
            "tensor_device": str(tensor.device) if hasattr(tensor, "device") else None,
            "call_site": _first_user_frame(),
        }
        self.violations.append(entry)
        if self.strict:
            raise RuntimeError(
                f"ACLGraphCaptureGuard: forbidden host sync '{op_name}' at {entry['call_site']}"
            )

    def __enter__(self) -> ACLGraphCaptureGuard:
        torch = _require_torch()
        guard = self

        class _GuardMode(torch.overrides.TorchFunctionMode):
            def __torch_function__(self, func, types, args=(), kwargs=None):
                kwargs = kwargs or {}
                name = getattr(func, "__name__", str(func))
                tensor = _first_tensor(args)
                if name in _SYNC_FUNC_NAMES and tensor is not None:
                    if name != "nonzero" or kwargs.get("as_tuple", False):
                        guard._record(name, tensor)
                elif name in ("cpu", "to") and tensor is not None:
                    target_dev = args[1] if len(args) > 1 else kwargs.get("device")
                    if name == "cpu" or (target_dev is not None and str(target_dev).startswith("cpu")):
                        if str(tensor.device) != "cpu":
                            guard._record(f"{name}(cpu)", tensor)
                return func(*args, **kwargs)

        self._mode = _GuardMode()
        self._mode.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if self._mode is not None:
            self._mode.__exit__(exc_type, exc, tb)
            self._mode = None
        return False

    def report(self) -> dict[str, Any]:
        return {
            "status": "clean" if not self.violations else "capture_hazard",
            "violation_count": len(self.violations),
            "first_violation": self.violations[0] if self.violations else None,
            "violations": list(self.violations),
        }


class ACLGraphAddressTracker:
    """Verify static buffer pointer stability and padded-tail masking across ACLGraph replay."""

    def __init__(self):
        self._bound: dict[str, dict[str, Any]] = {}

    def bind_capture_inputs(self, **roots: Any) -> dict[str, dict[str, Any]]:
        torch = _require_torch()
        self._bound.clear()
        for root_name, root_val in roots.items():
            for subpath, leaf in flatten_tree_with_paths(root_val):
                if not isinstance(leaf, torch.Tensor):
                    continue
                full_path = path_to_string((root_name,) + tuple(subpath))
                storage = leaf.untyped_storage()
                self._bound[full_path] = {
                    "data_ptr": int(leaf.data_ptr()),
                    "storage_id": _storage_identity(storage),
                    "storage_offset": int(leaf.storage_offset()),
                    "shape": tuple(int(s) for s in leaf.shape),
                    "stride": tuple(int(s) for s in leaf.stride()),
                    "dtype": str(leaf.dtype),
                }
        return dict(self._bound)

    def verify_replay_inputs(
        self,
        *,
        actual_tokens: int | None = None,
        padded_tokens: int | None = None,
        slot_mapping_tensor: Any = None,
        **roots: Any,
    ) -> dict[str, Any]:
        torch = _require_torch()
        violations: list[dict[str, Any]] = []
        for root_name, root_val in roots.items():
            for subpath, leaf in flatten_tree_with_paths(root_val):
                if not isinstance(leaf, torch.Tensor):
                    continue
                full_path = path_to_string((root_name,) + tuple(subpath))
                expected = self._bound.get(full_path)
                if expected is None:
                    continue
                observed_ptr = int(leaf.data_ptr())
                observed_shape = tuple(int(s) for s in leaf.shape)
                if observed_ptr != expected["data_ptr"]:
                    violations.append({
                        "kind": "rebound_pointer",
                        "tree_path": full_path,
                        "captured_data_ptr": expected["data_ptr"],
                        "replay_data_ptr": observed_ptr,
                        "hint": (
                            f"'{full_path}' was reassigned to a new tensor (data_ptr changed); "
                            "ACLGraph replay will silently read the stale capture-time buffer. "
                            "Use in-place .copy_() on the pre-allocated static tensor."
                        ),
                    })
                elif observed_shape != expected["shape"]:
                    violations.append({
                        "kind": "shape_mismatch_against_captured_bucket",
                        "tree_path": full_path,
                        "captured_shape": list(expected["shape"]),
                        "replay_shape": list(observed_shape),
                    })
        if (
            slot_mapping_tensor is not None
            and isinstance(slot_mapping_tensor, torch.Tensor)
            and actual_tokens is not None
            and padded_tokens is not None
            and int(padded_tokens) > int(actual_tokens)
        ):
            tail = slot_mapping_tensor.detach().reshape(-1)[int(actual_tokens):int(padded_tokens)]
            bad_slots = int((tail >= 0).sum().item())
            if bad_slots > 0:
                violations.append({
                    "kind": "unmasked_padded_slot_mapping",
                    "actual_tokens": int(actual_tokens),
                    "padded_tokens": int(padded_tokens),
                    "unmasked_slot_count": bad_slots,
                    "hint": (
                        "Padded tail slot_mapping[actual_tokens:padded_tokens] contains non-negative "
                        "slot IDs; fill padded slots with -1 so ACLGraph replay does not overwrite live KV/state."
                    ),
                })
        return {
            "status": "ok" if not violations else "violation",
            "bound_tensor_count": len(self._bound),
            "violation_count": len(violations),
            "first_violation": violations[0] if violations else None,
            "violations": violations,
        }


class ACLGraphDumper:
    """Capture-safe in-graph intermediate tensor dumper for ACLGraph / NPUGraph."""

    def __init__(
        self,
        model: Any,
        *,
        include: tuple[str, ...] | list[str] = ("*",),
        max_steps: int = 8,
        max_snapshot_numel: int = 16384,
        device: str | Any = "cpu",
    ):
        torch = _require_torch()
        self.model = model
        self.include = tuple(include)
        self.max_steps = max(int(max_steps), 1)
        self.max_snapshot_numel = max(int(max_snapshot_numel), 1)
        self.device = torch.device(device)
        self.step_counter = torch.zeros((), dtype=torch.int64, device=self.device)
        self._max_steps_tensor = torch.full(
            (), self.max_steps, dtype=torch.int64, device=self.device
        )
        self._step_idx_scratch = torch.zeros((), dtype=torch.int64, device=self.device)
        self._row_indices = torch.arange(
            self.max_steps, dtype=torch.int64, device=self.device
        ).unsqueeze(1)
        self._row_mask_scratch = torch.zeros(
            (self.max_steps, 1), dtype=torch.bool, device=self.device
        )
        self._slots: dict[str, Any] = {}
        self._scratch: dict[str, Any] = {}
        self._shapes: dict[str, tuple[int, ...]] = {}
        self._handles: list[Any] = []
        self._paused: bool = False
        self._attach()

    def _matches(self, name: str) -> bool:
        if not name:
            return False
        return any(fnmatch.fnmatch(name, pat) for pat in self.include)

    def _attach(self) -> None:
        for name, submodule in self.model.named_modules():
            if not self._matches(name):
                continue
            has_children = any(True for _ in submodule.children())
            if has_children and self.include == ("*",):
                continue
            if name not in self._slots:
                self._slots_init_for(name)
            handle = submodule.register_forward_hook(self._make_hook(name))
            self._handles.append(handle)

    def _slots_init_for(self, name: str) -> None:
        torch = _require_torch()
        self._slots[name] = torch.zeros(
            (self.max_steps, self.max_snapshot_numel),
            dtype=torch.float32,
            device=self.device,
        )
        self._scratch[name] = torch.zeros(
            (self.max_snapshot_numel,),
            dtype=torch.float32,
            device=self.device,
        )

    def _make_hook(self, name: str) -> Callable:
        torch = _require_torch()

        def _hook(_mod, _inputs, output):
            if self._paused:
                return
            tensor = _first_tensor(output)
            if tensor is None or (
                not tensor.is_floating_point()
                and tensor.dtype not in (torch.int32, torch.int64)
            ):
                return
            flat = tensor.detach().reshape(-1)
            take = min(int(flat.numel()), self.max_snapshot_numel)
            self._shapes.setdefault(name, tuple(int(s) for s in tensor.shape))
            slot_buf = self._slots[name]
            scratch = self._scratch[name]
            scratch.zero_()
            if take > 0:
                scratch[:take].copy_(flat[:take])
            torch.remainder(
                self.step_counter, self._max_steps_tensor, out=self._step_idx_scratch
            )
            torch.eq(
                self._row_indices, self._step_idx_scratch, out=self._row_mask_scratch
            )
            torch.where(
                self._row_mask_scratch, scratch.unsqueeze(0), slot_buf, out=slot_buf
            )

        return _hook

    def reset_step(self, step: int = 0) -> None:
        self.step_counter.fill_(int(step))

    def advance_step(self) -> None:
        """Advance replay step slot on-device without host synchronization."""
        self.step_counter.add_(1)

    def read_step(self, step_index: int = 0) -> dict[str, Any]:
        torch = _require_torch()
        idx = int(step_index) % self.max_steps
        tensors: dict[str, Any] = {}
        telemetry: dict[str, Any] = {}
        for name, slot_buf in self._slots.items():
            if name not in self._shapes:
                continue
            orig_shape = self._shapes[name]
            numel = 1
            for dim in orig_shape:
                numel *= dim
            take = min(numel, self.max_snapshot_numel)
            raw = slot_buf[idx, :take].detach().to("cpu").clone()
            if take == numel:
                tensors[name] = raw.reshape(orig_shape)
            else:
                tensors[name] = raw
            finite_mask = torch.isfinite(raw)
            finite_vals = raw[finite_mask]
            telemetry[name] = {
                "slot": idx,
                "valid_count": int(raw.numel()),
                "nan_count": int(torch.isnan(raw).sum().item()),
                "inf_count": int(torch.isinf(raw).sum().item()),
                "min": float(finite_vals.min().item()) if finite_vals.numel() > 0 else 0.0,
                "max": float(finite_vals.max().item()) if finite_vals.numel() > 0 else 0.0,
                "l2_norm": float(torch.linalg.vector_norm(finite_vals).item())
                if finite_vals.numel() > 0
                else 0.0,
                "abs_max": float(finite_vals.abs().max().item())
                if finite_vals.numel() > 0
                else 0.0,
            }
        return {"step_index": idx, "tensors": tensors, "telemetry": telemetry}

    def compare_with_eager(
        self,
        eager_fn: Callable,
        *args: Any,
        step_index: int = 0,
        atol: float = 1e-3,
        rtol: float = 1e-3,
        **kwargs: Any,
    ) -> dict[str, Any]:
        torch = _require_torch()
        graph_dump = self.read_step(step_index)["tensors"]
        eager_tensors: dict[str, Any] = {}
        temp_handles = []
        for name, submodule in self.model.named_modules():
            if name not in graph_dump:
                continue

            def _capture_eager(_m, _inp, out, _name=name):
                t = _first_tensor(out)
                if t is not None:
                    flat = t.detach().reshape(-1).to(device="cpu", dtype=torch.float32)
                    take = min(int(flat.numel()), self.max_snapshot_numel)
                    orig_shape = self._shapes[_name]
                    numel = 1
                    for dim in orig_shape:
                        numel *= dim
                    eager_tensors[_name] = (
                        flat[:take].reshape(orig_shape) if take == numel else flat[:take]
                    )

            temp_handles.append(submodule.register_forward_hook(_capture_eager))
        prev_paused = self._paused
        self._paused = True
        try:
            with torch.no_grad():
                eager_fn(*args, **kwargs)
        finally:
            for h in temp_handles:
                h.remove()
            self._paused = prev_paused

        per_layer = []
        first_divergent = None
        for name in self._slots:
            if name not in graph_dump or name not in eager_tensors:
                continue
            g_t = graph_dump[name]
            e_t = eager_tensors[name]
            metric = _tensor_metrics(name, e_t, g_t, [], None)
            close = torch.isclose(
                g_t.to(torch.float64), e_t.to(torch.float64),
                atol=float(atol), rtol=float(rtol), equal_nan=True,
            )
            mismatches = int((~close).sum().item())
            status = "ok" if (mismatches == 0 and metric.get("nonfinite_mismatch_count", 0) == 0) else "diverged"
            row = {
                "module_path": name,
                "status": status,
                "mismatch_count": mismatches,
                "max_abs_err": float(metric.get("max_abs_err", 0.0)),
                "rel_l2_err": float(metric.get("rel_l2_err", 0.0)),
                "ulp_max": float(metric.get("ulp_max", 0.0)),
                "nonfinite_mismatch_count": int(metric.get("nonfinite_mismatch_count", 0)),
            }
            per_layer.append(row)
            if status == "diverged" and first_divergent is None:
                first_divergent = name
        return {
            "status": "ok" if first_divergent is None else "diverged",
            "step_index": int(step_index),
            "first_divergent_layer": first_divergent,
            "per_layer": per_layer,
        }

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def __enter__(self) -> ACLGraphDumper:
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.close()
        return False


class ACLGraphVllmInspector:
    """Hooks vllm-ascend's ACLGraphWrapper to audit capture syncs, address stability, and layer dumps."""

    def __init__(
        self,
        *,
        include: tuple[str, ...] | list[str] = ("*",),
        max_steps: int = 8,
        strict_guard: bool = False,
        compare_eager: bool = False,
        atol: float = 1e-3,
        rtol: float = 1e-3,
    ):
        self.include = tuple(include)
        self.max_steps = int(max_steps)
        self.strict_guard = bool(strict_guard)
        self.compare_eager = bool(compare_eager)
        self.atol = float(atol)
        self.rtol = float(rtol)
        self.guard_violations: list[dict[str, Any]] = []
        self.address_reports: list[dict[str, Any]] = []
        self.layer_diffs: list[dict[str, Any]] = []
        self.dumpers: dict[int, ACLGraphDumper] = {}
        self.trackers: dict[Any, ACLGraphAddressTracker] = {}
        self._orig_call: Callable | None = None
        self._wrapper_cls: Any = None

    def attach(self) -> ACLGraphVllmInspector:
        try:
            from vllm_ascend.compilation.acl_graph import ACLGraphWrapper
        except ImportError:
            return self
        if self._orig_call is not None:
            return self
        self._wrapper_cls = ACLGraphWrapper
        self._orig_call = ACLGraphWrapper.__call__
        inspector = self
        orig_call = self._orig_call

        def _wrapped_call(wrapper_self, *args, **kwargs):
            torch = _require_torch()
            try:
                from vllm.forward_context import get_forward_context
                fwd_ctx = get_forward_context()
                batch_desc = getattr(fwd_ctx, "batch_descriptor", None)
                attn_meta = getattr(fwd_ctx, "attn_metadata", None)
            except Exception:
                fwd_ctx = None
                batch_desc = None
                attn_meta = None

            entries = getattr(wrapper_self, "concrete_aclgraph_entries", {})
            entry = entries.get(batch_desc) if batch_desc is not None else None
            is_capturing = entry is None or getattr(entry, "aclgraph", None) is None

            runnable = getattr(wrapper_self, "runnable", None)
            dumper = inspector.dumpers.get(id(wrapper_self))
            if dumper is None and isinstance(runnable, torch.nn.Module):
                dev = next((p.device for p in runnable.parameters()), torch.device("cpu"))
                dumper = ACLGraphDumper(
                    runnable,
                    include=inspector.include,
                    max_steps=inspector.max_steps,
                    device=dev,
                )
                inspector.dumpers[id(wrapper_self)] = dumper

            meta_dict: dict[str, Any] = {}
            for idx, arg in enumerate(args):
                if isinstance(arg, torch.Tensor):
                    meta_dict[f"arg_{idx}"] = arg
            for k, v in kwargs.items():
                if isinstance(v, torch.Tensor):
                    meta_dict[f"kw_{k}"] = v
            if isinstance(attn_meta, dict):
                for sub_meta in attn_meta.values():
                    for attr in ("slot_mapping", "seq_lens", "query_start_loc"):
                        val = getattr(sub_meta, attr, None)
                        if isinstance(val, torch.Tensor):
                            meta_dict.setdefault(attr, val)
            elif attn_meta is not None:
                for attr in ("slot_mapping", "seq_lens", "query_start_loc"):
                    val = getattr(attn_meta, attr, None)
                    if isinstance(val, torch.Tensor):
                        meta_dict[attr] = val

            tracker = inspector.trackers.setdefault(batch_desc, ACLGraphAddressTracker())
            if is_capturing:
                tracker.record_capture(meta=meta_dict)
                if dumper is not None:
                    dumper.reset_step(0)
                with ACLGraphCaptureGuard(strict=inspector.strict_guard) as guard:
                    out = orig_call(wrapper_self, *args, **kwargs)
                rep = guard.report()
                if rep["violations"]:
                    inspector.guard_violations.extend(rep["violations"])
                return out

            actual_tokens = getattr(attn_meta, "num_actual_tokens", None)
            padded_tokens = getattr(batch_desc, "num_tokens", None) if batch_desc is not None else None
            addr_rep = tracker.verify_replay(
                meta=meta_dict,
                actual_tokens=actual_tokens,
                padded_tokens=padded_tokens,
            )
            inspector.address_reports.append(addr_rep)
            step_idx = 0
            if dumper is not None:
                dumper.advance_step()
                step_idx = int(dumper.step_counter.detach().cpu().item()) % dumper.max_steps
            out = orig_call(wrapper_self, *args, **kwargs)
            if dumper is not None and inspector.compare_eager and callable(runnable):
                diff = dumper.compare_with_eager(
                    runnable,
                    *args,
                    step_index=step_idx,
                    atol=inspector.atol,
                    rtol=inspector.rtol,
                    **kwargs,
                )
                inspector.layer_diffs.append(diff)
            return out

        ACLGraphWrapper.__call__ = _wrapped_call
        return self

    def close(self) -> None:
        if self._wrapper_cls is not None and self._orig_call is not None:
            self._wrapper_cls.__call__ = self._orig_call
            self._orig_call = None
        for dumper in self.dumpers.values():
            dumper.close()
        self.dumpers.clear()

    def report(self) -> dict[str, Any]:
        addr_violations = [
            v for rep in self.address_reports for v in rep.get("violations", [])
        ]
        diverged_diffs = [d for d in self.layer_diffs if d.get("status") == "diverged"]
        has_issue = bool(self.guard_violations or addr_violations or diverged_diffs)
        return {
            "status": "violation" if has_issue else "ok",
            "capture_guard_violations": list(self.guard_violations),
            "address_violations": addr_violations,
            "layer_diffs": list(self.layer_diffs),
            "first_divergent_layer": (
                diverged_diffs[0].get("first_divergent_layer") if diverged_diffs else None
            ),
        }

    def __enter__(self) -> ACLGraphVllmInspector:
        return self.attach()

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.close()
        return False


__all__ = [
    "ACLGraphAddressTracker",
    "ACLGraphCaptureGuard",
    "ACLGraphDumper",
    "ACLGraphVllmInspector",
]
