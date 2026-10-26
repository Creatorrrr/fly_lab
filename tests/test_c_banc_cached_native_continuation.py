"""Generated CPU tests of continuation's new gate/zero-state boundary only."""

import copy
import json
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from tools import continue_c_banc_m3_native_timescale_body as runner


def zero():
    return {
        "body": {},
        "world": {},
        "cuts": {},
        "navigation": {},
        "parameters": {},
        "seed": 1,
        "control_tick": 0,
        "cpg_hash": "generated",
        "initial_position": [],
        "signed_forward_mm": 0.0,
        "network": {
            "rate": np.zeros(158706, np.float32),
            "adaptation": np.zeros(158706, np.float32),
            "drive": np.zeros(158706, np.float32),
            "output_mask": np.ones(158706, np.float32),
            "tick": 0,
            "device": "cuda",
        },
    }


def test_checkpoint_zero_requires_whole_exact_state():
    original = zero()
    runner.verify_zero(copy.deepcopy(original), original)
    for change in ("rate", "body", "clock", "device"):
        candidate = copy.deepcopy(original)
        if change == "rate":
            candidate["network"]["rate"][17] = 1
        elif change == "body":
            candidate["body"]["new"] = 1
        elif change == "clock":
            candidate["control_tick"] = 1
        else:
            candidate["network"]["device"] = "cpu"
        with pytest.raises((AssertionError, ValueError)):
            runner.verify_zero(candidate, original)


def test_rejected_freeze_does_not_enter_body_or_cuda(tmp_path):
    calls = []
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: calls.append("CUDA"))
    )
    with (
        patch.dict("sys.modules", {"torch": fake_torch}),
        patch.object(
            runner.body_protocol,
            "_native_body",
            side_effect=AssertionError("must not construct"),
        ),
        pytest.raises(ValueError, match="freeze"),
    ):
        runner.run(
            tmp_path / "unused", "bad", tmp_path / "review", "bad", tmp_path / "out"
        )
    assert calls == []
    result = json.loads((tmp_path / "out/result.json").read_text(encoding="utf-8"))
    assert result["body_started"] is False
    assert result["hypothesis_verdict"] == "invalid_not_tested"


def test_existing_output_is_never_reused(tmp_path):
    with pytest.raises(FileExistsError):
        runner.run(tmp_path / "unused", "bad", tmp_path / "unused", "bad", tmp_path)


@pytest.fixture
def generated_scope(monkeypatch):
    """Exercise real request/finalization boundaries without original data or CUDA."""
    calls = []
    freeze = {
        "minimum_free_disk_bytes": 0,
        "prepare_wall_limit_s": 20,
        "body_wall_limit_s": 180,
    }
    monkeypatch.setattr(runner, "verified_inputs", lambda *args: (freeze, {}))
    monkeypatch.setattr(
        runner,
        "load_cached_candidate",
        lambda *args: dict.fromkeys(
            (
                "graph",
                "cal",
                "observation",
                "startup",
                "neutral",
                "parameters",
                "anatomy",
            )
        ),
    )
    monkeypatch.setattr(
        runner.body_protocol.joint_fit,
        "one_cpu_thread",
        lambda: (
            SimpleNamespace(
                restore_original_limits=lambda: calls.append("threads restored")
            ),
            {},
        ),
    )

    def cleanup(entered, ledger):
        calls.append(("CUDA cleanup", entered))
        return []

    monkeypatch.setattr(runner.body_protocol, "cleanup_cuda_runtime", cleanup)
    return calls


def test_failed_entry_receipt_never_calls_body_or_cuda(
    tmp_path, monkeypatch, generated_scope
):
    original_write = runner.body_protocol.write_json

    def write(path, value):
        if (
            path.name == "advance-ledger.json"
            and value["status"] == "native_body_requested"
        ):
            raise OSError("request receipt failed")
        original_write(path, value)

    monkeypatch.setattr(runner.body_protocol, "write_json", write)
    monkeypatch.setattr(
        runner.body_protocol,
        "_native_body",
        lambda *args: pytest.fail("body entry after failed receipt"),
    )
    with pytest.raises(OSError, match="request receipt failed"):
        runner.run("fixture", "fixture", "fixture", "fixture", tmp_path / "out")
    assert ("CUDA cleanup", False) in generated_scope
    result = json.loads((tmp_path / "out/result.json").read_text(encoding="utf-8"))
    assert result["body_started"] is False


def test_entered_partial_construction_cleans_up_and_preserves_primary(
    tmp_path, monkeypatch, generated_scope
):
    def fail(*args):
        raise ValueError("constructor failure")

    monkeypatch.setattr(runner.body_protocol, "_native_body", fail)

    def cleanup(entered, ledger):
        generated_scope.append(("CUDA cleanup", entered))
        return [("generated cleanup failure", OSError("secondary error"))]

    monkeypatch.setattr(runner.body_protocol, "cleanup_cuda_runtime", cleanup)
    with pytest.raises(ValueError, match="constructor failure") as error:
        runner.run("fixture", "fixture", "fixture", "fixture", tmp_path / "out")
    assert ("CUDA cleanup", True) in generated_scope
    assert "threads restored" in generated_scope
    assert any("secondary error" in note for note in error.value.__notes__)
    result = json.loads((tmp_path / "out/result.json").read_text(encoding="utf-8"))
    assert result["primary_error"]["type"] == "ValueError"
    assert result["hypothesis_verdict"] == "invalid_not_tested"


@pytest.mark.parametrize(
    "failure", ["body_wall_limit_s", "maximum_raw_bytes", "resource_guard"]
)
def test_resource_failure_cannot_look_accepted(
    tmp_path, monkeypatch, generated_scope, failure
):
    body = {
        "validity": "valid_full_BANC_native_transfer_diagnostic",
        "hypothesis_verdict": "supported_development_diagnostic",
        "actual_neural_s": 10.1,
        "actual_physical_s": 10.1,
        "body_analysis_wall_s": 1,
        "stop_reason": "duration",
    }
    if failure == "body_wall_limit_s":
        body["body_analysis_wall_s"] = 181
    elif failure == "resource_guard":
        body["stop_reason"] = "resource_guard"
    else:
        monkeypatch.setattr(
            runner.body_protocol, "_raw_bytes", lambda out: (4 << 30) + 1
        )
    monkeypatch.setattr(runner.body_protocol, "_native_body", lambda *args: body)
    with pytest.raises((TimeoutError, RuntimeError)):
        runner.run("fixture", "fixture", "fixture", "fixture", tmp_path / "out")
    result = json.loads((tmp_path / "out/result.json").read_text(encoding="utf-8"))
    assert result["validity"] == "invalid_resource"
    assert result["hypothesis_verdict"] == "invalid_not_tested"
    assert result["resource_limit_met"] is False
    assert result["candidate_adopted"] is False
