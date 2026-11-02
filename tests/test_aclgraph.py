"""Tests for ACLGraph / NPUGraph capture guard, pointer tracker, and in-graph dumper.

[Role]
  Validate ``ACLGraphCaptureGuard``, ``ACLGraphAddressTracker``, and
  ``ACLGraphDumper`` across CPU and live Ascend NPU ``NPUGraph`` replay.

[Invariants]
  * ``ACLGraphCaptureGuard`` catches ``.item()``, ``bool(tensor)``, and
    ``.tolist()`` inside capture regions with exact call-site attribution.
  * ``ACLGraphAddressTracker`` catches rebound input pointers and unmasked
    padded ``slot_mapping`` tails (`>= 0` in `[actual_tokens:padded_tokens]`).
  * ``ACLGraphDumper`` captures multi-step intermediate layer outputs on-device
    and pinpoints ``first_divergent_layer`` against Eager execution.

[Timeouts & Locks]
  Completes in < 3s on CPU and < 15s on live NPU; acquires no locks.

[Side-Effect Policy]
  In-memory tensor operations only; removes all forward hooks on exit.
"""
from __future__ import annotations

import pytest

from mindie_debugger import (
    ACLGraphAddressTracker,
    ACLGraphCaptureGuard,
    ACLGraphDumper,
)

torch = pytest.importorskip("torch")


def test_aclgraph_capture_guard_catches_host_sync():
    x = torch.tensor([1.0, 2.0, 3.0])
    with ACLGraphCaptureGuard(strict=False) as guard:
        y = x * 2.0
        _ = y.sum().item()
        if y[0]:
            y = y + 1.0
    rep = guard.report()
    assert rep["status"] == "capture_hazard"
    assert rep["violation_count"] >= 2
    ops = [v["op"] for v in rep["violations"]]
    assert "item" in ops
    assert "__bool__" in ops


def test_aclgraph_address_tracker_catches_rebound_pointer_and_padded_slots():
    static_seq_lens = torch.tensor([10, 12, 0, 0], dtype=torch.int32)
    static_slots = torch.tensor([1, 2, -1, -1], dtype=torch.int64)
    tracker = ACLGraphAddressTracker()
    tracker.bind_capture_inputs(
        meta={"seq_lens": static_seq_lens, "slot_mapping": static_slots}
    )

    # Good replay: in-place copy_ preserves data_ptr and masks padded tail with -1
    static_seq_lens[:2].copy_(torch.tensor([11, 13], dtype=torch.int32))
    ok_rep = tracker.verify_replay_inputs(
        actual_tokens=2,
        padded_tokens=4,
        slot_mapping_tensor=static_slots,
        meta={"seq_lens": static_seq_lens, "slot_mapping": static_slots},
    )
    assert ok_rep["status"] == "ok"
    assert ok_rep["violation_count"] == 0

    # Buggy replay: reassigning seq_lens to a newly allocated tensor + unmasked tail slot
    rebound_seq_lens = torch.tensor([11, 13, 0, 0], dtype=torch.int32)
    static_slots[2] = 99  # stale non-negative slot in padded tail!
    bad_rep = tracker.verify_replay_inputs(
        actual_tokens=2,
        padded_tokens=4,
        slot_mapping_tensor=static_slots,
        meta={"seq_lens": rebound_seq_lens, "slot_mapping": static_slots},
    )
    assert bad_rep["status"] == "violation"
    kinds = {v["kind"] for v in bad_rep["violations"]}
    assert "rebound_pointer" in kinds
    assert "unmasked_padded_slot_mapping" in kinds

    # KV bounds & block_table capacity verification (generalizing verify_request_kv)
    active_bad_slots = torch.tensor([-1, 64, -1, -1], dtype=torch.int64)
    block_table = torch.tensor([[0, 8, -1], [1, -1, -1]], dtype=torch.int32)
    seq_lens = torch.tensor([20, 20], dtype=torch.int32)
    kv_rep = tracker.verify_replay(
        actual_tokens=2,
        padded_tokens=4,
        block_size=16,
        num_blocks=4,
        null_block_id=-1,
        meta={
            "slot_mapping": active_bad_slots,
            "block_table": block_table,
            "seq_lens": seq_lens,
        },
    )
    assert kv_rep["status"] == "violation"
    kv_kinds = {v["kind"] for v in kv_rep["violations"]}
    assert "active_slot_out_of_bounds" in kv_kinds
    assert "block_table_out_of_bounds" in kv_kinds
    assert "block_table_capacity_insufficient" in kv_kinds


class _ProjWithStaleScale(torch.nn.Module):
    def __init__(self, stale_scale_buf: torch.Tensor):
        super().__init__()
        self.linear = torch.nn.Linear(4, 4, bias=False)
        with torch.no_grad():
            self.linear.weight.copy_(torch.eye(4) * 2.0)
        self.stale_scale_buf = stale_scale_buf
        self.use_stale = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.linear(x)
        if self.use_stale:
            out = out + self.stale_scale_buf
        return out


class _TwoStageGraphModel(torch.nn.Module):
    def __init__(self, stale_scale_buf: torch.Tensor):
        super().__init__()
        self.norm = torch.nn.Linear(4, 4, bias=False)
        with torch.no_grad():
            self.norm.weight.copy_(torch.eye(4))
        self.proj = _ProjWithStaleScale(stale_scale_buf)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        return self.proj(h)


def test_aclgraph_dumper_pinpoints_divergent_layer_across_replay_steps():
    stale_buf = torch.zeros(4)
    model = _TwoStageGraphModel(stale_buf)
    static_x = torch.ones(2, 4)

    with ACLGraphDumper(model, include=("norm", "proj"), max_steps=4) as dumper:
        # Step 0 (healthy: stale_buf is 0.0)
        dumper.reset_step(0)
        _ = model(static_x)

        # Step 1 (simulate graph replay reading corrupted/stale buffer in proj)
        dumper.advance_step()
        static_x.fill_(3.0)
        stale_buf.fill_(0.5)
        _ = model(static_x)

        step0 = dumper.read_step(0)
        step1 = dumper.read_step(1)
        assert torch.allclose(step0["tensors"]["norm"], torch.ones(2, 4))
        assert torch.allclose(step1["tensors"]["norm"], torch.full((2, 4), 3.0))

        # Compare Step 1 graph dump against clean Eager execution
        def clean_eager(inp):
            prev = model.proj.use_stale
            model.proj.use_stale = False
            try:
                return model(inp)
            finally:
                model.proj.use_stale = prev

        diff = dumper.compare_with_eager(clean_eager, static_x, step_index=1, atol=1e-4)
        assert diff["status"] == "diverged"
        assert diff["first_divergent_layer"] == "proj"
        by_layer = {row["module_path"]: row for row in diff["per_layer"]}
        assert by_layer["norm"]["status"] == "ok"
        assert by_layer["proj"]["status"] == "diverged"
        assert by_layer["proj"]["max_abs_err"] == pytest.approx(0.5)


@pytest.mark.live_npu
def test_live_npu_aclgraph_dumper_captures_and_replays_without_host_sync():
    try:
        import torch_npu  # noqa: F401
    except ImportError:
        pytest.skip("torch_npu not installed")
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("Ascend NPU not available")

    device = torch.device("npu:0")
    stale_buf = torch.zeros(4, device=device)
    model = _TwoStageGraphModel(stale_buf).to(device).eval()
    static_x = torch.ones(2, 4, device=device)

    with ACLGraphDumper(model, include=("norm", "proj"), max_steps=4, device=device) as dumper:
        # Warmup outside capture
        dumper.reset_step(0)
        with torch.no_grad():
            _ = model(static_x)
        torch.npu.synchronize()

        # Capture real ACLGraph with ACLGraphDumper active AND ACLGraphCaptureGuard active
        g = torch.npu.NPUGraph()
        dumper.reset_step(0)
        with ACLGraphCaptureGuard(strict=True):
            with torch.npu.graph(g):
                static_out = model(static_x)
        torch.npu.synchronize()

        # Replay Step 0 (clean: stale_buf == 0)
        dumper.reset_step(0)
        static_x.fill_(2.0)
        stale_buf.zero_()
        g.replay()
        torch.npu.synchronize()

        # Replay Step 1 (stale buffer stomped with +0.75 inside captured proj)
        dumper.advance_step()
        static_x.fill_(3.0)
        stale_buf.fill_(0.75)
        g.replay()
        torch.npu.synchronize()
        assert torch.allclose(static_out.cpu(), torch.full((2, 4), 6.75))

        def clean_eager(inp):
            prev = model.proj.use_stale
            model.proj.use_stale = False
            try:
                return model(inp)
            finally:
                model.proj.use_stale = prev

        diff0 = dumper.compare_with_eager(clean_eager, torch.full((2, 4), 2.0, device=device), step_index=0, atol=1e-4)
        assert diff0["status"] == "ok"

        diff1 = dumper.compare_with_eager(clean_eager, static_x, step_index=1, atol=1e-4)
        assert diff1["status"] == "diverged"
        assert diff1["first_divergent_layer"] == "proj"
        by_layer = {row["module_path"]: row for row in diff1["per_layer"]}
        assert by_layer["norm"]["status"] == "ok"
        assert by_layer["proj"]["status"] == "diverged"
        assert by_layer["proj"]["max_abs_err"] == pytest.approx(0.75, abs=1e-4)
