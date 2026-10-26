"""Synthetic E183 contract checks. No original stores or native simulation."""

import json
import sys
from types import SimpleNamespace

import pytest

from tools import continue_c_banc_m3_native_timing as e183
from tools import optimize_c_banc_m3_native_timing as e181


def write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return e181._sha256(path)


def fake_gate():
    return e181.ValidatedGate(
        {"graph_path": "never-load", "graph_hash": "g"}, "e" * 64, "f" * 64, "lm"
    )


def fake_args(sign="plus", axis="lh"):
    return SimpleNamespace(
        axis=axis,
        child_sign=sign,
        freeze_sha256="a" * 64,
        source_review_sha256="b" * 64,
        child_request=None,
        child_request_sha256=None,
    )


def prepare_child(tmp_path, monkeypatch, sign):
    campaign = tmp_path / "campaign"
    root = campaign / "axis-lh"
    center = [0.0] * 6
    theta = list(e181.propose_axis(center, "lh", sign, 0.125))
    proposal = {
        "selected_theta": theta,
        "attempts": [{"step": 0.125, "status": "feasible", "theta": theta}],
    }
    intent_sha = write(
        root / "intent.json",
        {
            "schema": "flylab.m3.e183.axis-intent.v1",
            "axis": "lh",
            "freeze_sha256": "a" * 64,
            "center_theta": center,
            "previous_axis_sha256": None,
            "preflight": {sign: proposal},
        },
    )
    readiness_sha = write(
        root / f"{sign}-readiness.json",
        {
            "schema": "flylab.m3.e183.readiness.v1",
            "axis": "lh",
            "sign": sign,
            "native_permitted": True,
            "observations": [{"ready": True}, {"ready": True}],
        },
    )
    args = fake_args(sign)
    request = e183._make_request(
        args, fake_gate(), sign, center, proposal, intent_sha, readiness_sha, 1.0
    )
    args.child_request = root / f"{sign}-request.json"
    args.child_request_sha256 = write(args.child_request, request)
    monkeypatch.setattr(e183, "_axis_root", lambda axis: campaign / f"axis-{axis}")
    monkeypatch.setattr(e183, "_rel", lambda name: campaign)
    monkeypatch.setattr(e183, "_prior_axis", lambda axis, sha: (center, {}, None))
    monkeypatch.setattr(e181, "driver_ancestor", lambda request: True)
    monkeypatch.setattr(e181, "_directory_bytes", lambda root: 0)
    monkeypatch.setattr(e181, "_load_startup", lambda freeze: {})
    monkeypatch.setattr(
        e181.protocol.joint_fit,
        "one_cpu_thread",
        lambda: (SimpleNamespace(restore_original_limits=lambda: None), None),
    )
    calls = []

    def original_sentinel(path):
        calls.append(path)
        raise RuntimeError("ORIGINAL_SENTINEL")

    monkeypatch.setattr(e181.GraphStore, "load", original_sentinel)
    return args, request, calls


@pytest.mark.parametrize("sign", ["plus", "minus"])
def test_generated_child_request_reaches_original_boundary(tmp_path, monkeypatch, sign):
    args, request, calls = prepare_child(tmp_path, monkeypatch, sign)
    assert request["theta"][e181.LEGS.index("lh")] == (
        0.125 if sign == "plus" else -0.125
    )
    with pytest.raises(RuntimeError, match="ORIGINAL_SENTINEL"):
        e183._child(args, fake_gate())
    assert calls == [args.child_request.parent.parent]


@pytest.mark.parametrize("mutation", ["theta", "center", "intent", "readiness"])
def test_child_rejects_changed_request_before_original(tmp_path, monkeypatch, mutation):
    args, request, calls = prepare_child(tmp_path, monkeypatch, "plus")
    if mutation == "theta":
        request["theta"][2] = 0.0625
        args.child_request_sha256 = write(args.child_request, request)
    elif mutation == "center":
        request["center_theta"][2] = 0.03125
        args.child_request_sha256 = write(args.child_request, request)
    elif mutation == "intent":
        intent = json.loads((args.child_request.parent / "intent.json").read_text())
        intent["center_theta"][2] = 0.03125
        write(args.child_request.parent / "intent.json", intent)
    else:
        readiness = json.loads(
            (args.child_request.parent / "plus-readiness.json").read_text()
        )
        readiness["observations"][1]["ready"] = False
        write(args.child_request.parent / "plus-readiness.json", readiness)
    with pytest.raises(ValueError):
        e183._child(args, fake_gate())
    assert calls == []


def test_child_rejects_changed_axis_order_before_original(tmp_path, monkeypatch):
    args, _, calls = prepare_child(tmp_path, monkeypatch, "plus")
    monkeypatch.setattr(
        e183, "_prior_axis", lambda axis, sha: ([0.0] * 6, {}, "c" * 64)
    )
    with pytest.raises(ValueError, match="center/order"):
        e183._child(args, fake_gate())
    assert calls == []


def test_zero_native_startup_rejection_has_terminal_result(tmp_path, monkeypatch):
    campaign = tmp_path / "campaign"
    old = tmp_path / "old"
    baseline = {
        "validity": "valid_3s_diagnostic",
        "objective_mm_s": 0.2,
        "theta": [0.0] * 6,
    }
    write(old / "baseline" / "result.json", baseline)
    monkeypatch.setattr(
        e183, "_rel", lambda name: old if name == e183.e182.OUTPUT else campaign
    )
    monkeypatch.setattr(e183, "_axis_root", lambda axis: campaign / f"axis-{axis}")
    monkeypatch.setattr(
        e183, "_prior_axis", lambda axis, sha: ([0.0] * 6, baseline, None)
    )
    monkeypatch.setattr(e183, "_capacity", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        e183,
        "_stage_cost",
        lambda *args: {
            "native_upper_bound": 0,
            "native_accounted_s": 0.0,
            "stage_accounted_s": 0.1,
            "cumulative_with_inherited_s": e183.INHERITED_S + 0.1,
        },
    )
    monkeypatch.setattr(e181, "_directory_bytes", lambda root: 0)
    monkeypatch.setattr(
        e181,
        "_saved_startup_callback",
        lambda gate: (
            lambda theta: {
                "strict_E141_startup_screen": False,
            }
        ),
    )
    monkeypatch.setattr(
        e181.protocol.joint_fit,
        "one_cpu_thread",
        lambda: (SimpleNamespace(restore_original_limits=lambda: None), None),
    )
    monkeypatch.setattr(
        e183.e182, "_readiness", lambda: pytest.fail("native readiness reached")
    )
    monkeypatch.setattr(
        e183.e181, "supervise_command", lambda *a, **k: pytest.fail("native launched")
    )
    args = fake_args()
    result = e183._driver(args, fake_gate())
    assert result["technical_failure"] is False
    assert result["selection"]["selected_theta"] == [0.0] * 6
    assert result["accounting"]["native_upper_bound"] == 0
    assert (campaign / "axis-lh" / "result.json").is_file()


def test_spent_budget_blocks_next_child(tmp_path, monkeypatch):
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    monkeypatch.setattr(e183, "_rel", lambda name: campaign)
    monkeypatch.setattr(e181, "_directory_bytes", lambda root: 0)
    monkeypatch.setattr(
        e183.psutil, "disk_usage", lambda path: SimpleNamespace(free=30 << 30)
    )
    monkeypatch.setattr(
        e183,
        "_stage_cost",
        lambda *args: {
            "native_upper_bound": 0,
            "stage_accounted_s": 950,
            "cumulative_with_inherited_s": e183.INHERITED_S + 950,
        },
    )
    with pytest.raises(e181.ResourceGuard, match="cannot fit"):
        e183._capacity("lh", e183.time.perf_counter(), next_native=True)


def test_failed_native_time_accounted_once_with_reserve(tmp_path, monkeypatch):
    campaign = tmp_path / "campaign"
    root = campaign / "axis-lh"
    monkeypatch.setattr(e183, "_rel", lambda name: campaign)
    monkeypatch.setattr(e183, "_axis_root", lambda axis: campaign / f"axis-{axis}")
    monkeypatch.setattr(e183.time, "perf_counter", lambda: 80.0)
    write(root / "plus-request.json", {"driver_elapsed_s": 10.0})
    write(
        root / "plus-request-supervisor.json",
        {
            "started": True,
            "accounted_wall_s": 110.0,
            "child_wall_s": 60.0,
            "cost_exact": False,
            "status": "child_nonzero",
        },
    )
    cost = e183._stage_cost("lh", 0.0)
    assert cost["native_upper_bound"] == 1
    assert cost["native_actual_elapsed_s"] == 60.0
    assert cost["non_native_elapsed_s"] == 20.0
    assert cost["stage_accounted_s"] == 130.0


def test_measured_startup_rejection_is_scientific_ineligible_result():
    e183._classify_native_result(
        {
            "validity": "incomplete",
            "technical_failure": False,
            "startup_pass": False,
            "measured_health_stop": False,
            "objective_mm_s": None,
        }
    )
    with pytest.raises(ValueError, match="Unknown E183 incomplete"):
        e183._classify_native_result(
            {
                "validity": "incomplete",
                "technical_failure": False,
                "startup_pass": None,
                "measured_health_stop": False,
                "objective_mm_s": None,
            }
        )


def test_driver_gate_failure_receipt_blocks_later_native(tmp_path, monkeypatch):
    campaign = tmp_path / "campaign"
    monkeypatch.setattr(e183, "_rel", lambda name: campaign)
    calls = []

    def reject(_args):
        calls.append(1)
        raise ValueError("synthetic source gate rejected")

    monkeypatch.setattr(e183, "_gate", reject)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "runner",
            "--axis",
            "lh",
            "--freeze",
            "frozen.json",
            "--freeze-sha256",
            "a" * 64,
            "--source-review",
            "source.json",
            "--source-review-sha256",
            "b" * 64,
        ],
    )
    with pytest.raises(ValueError, match="synthetic source gate rejected"):
        e183.main()
    receipt = json.loads((campaign / "axis-lh-gate-failure.json").read_text())
    assert receipt["native_started"] is False
    assert receipt["freeze_sha256"] == "a" * 64
    assert receipt["elapsed_s"] >= 0
    with pytest.raises(e181.ResourceGuard, match="blocks retry"):
        e183.main()
    assert calls == [1]
    with pytest.raises(e181.ResourceGuard, match="blocks native"):
        e183._capacity("lh", e183.time.perf_counter(), next_native=True)


def test_previous_raw_hash_and_order_required(tmp_path, monkeypatch):
    campaign = tmp_path / "campaign"
    root = campaign / "axis-lh"
    monkeypatch.setattr(e183, "_axis_root", lambda axis: campaign / f"axis-{axis}")
    center = [0.0] * 6
    raw_sha = write(root / "plus" / "result.json", {"theta": center})
    write(root / "plus-request.json", {"attempted": True})
    result_sha = write(
        root / "result.json",
        {
            "axis": "lh",
            "technical_failure": False,
            "selection": {"selected_theta": center, "selected_direction": None},
            "selected_result": {"theta": center},
            "arm_result_sha256": {},
        },
    )
    review = {
        "schema": e183.AXIS_REVIEW_SCHEMA,
        "freeze_sha256": "a" * 64,
        "axis_result_sha256": result_sha,
        "selection_checked": True,
        "accepts_next_axis": True,
        "blocking_defects": [],
        "raw_sha256": {"plus": {"result.json": raw_sha}},
    }
    write(root / "parent-axis-review.json", review)
    assert e183._prior_axis("rf", "a" * 64)[0] == center
    write(root / "plus" / "result.json", {"theta": [0.1] * 6})
    with pytest.raises(ValueError, match="raw"):
        e183._prior_axis("rf", "a" * 64)
    write(root / "plus" / "result.json", {"theta": center})
    result = json.loads((root / "result.json").read_text())
    result["axis"] = "rm"
    write(root / "result.json", result)
    with pytest.raises(ValueError, match="previous E183 axis"):
        e183._prior_axis("rf", "a" * 64)
