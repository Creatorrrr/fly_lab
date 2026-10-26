"""Generated CPU checks for E241 time split, anatomy, and target alignment."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from tools import prepare_c_banc_m3_matched_input_basis as m


def test_training_geometry_does_not_use_later_stages_or_descriptors():
    stages = 16
    positive = np.ones((stages, 2), np.float32)
    negative = np.full((stages, 2), .5, np.float32)
    drive = np.linspace(-.4, .5, stages * 2, dtype=np.float32).reshape(stages, 2)
    descriptor = np.zeros((4, 2, 9), np.float64)
    descriptor[1, :, 0] = [1., 2.]
    descriptor[2, :, 0] = [4., 3.]
    descriptor[3, :, 0] = [9., 9.]
    membership = np.array([0, 1])
    pool = np.eye(2)
    cap = np.array([5., 6.])
    scale, prototype = m.training_geometry(positive, negative, drive,
        descriptor, membership, pool, cap, scale_stop=8,
        prototype_stop=3, count=2)
    changed = positive.copy(), negative.copy(), drive.copy(), descriptor.copy()
    changed[0][8:] = 1000
    changed[1][8:] = 1000
    changed[2][8:] = 1000
    changed[3][3:] = 1000
    later_scale, later_prototype = m.training_geometry(*changed,
        membership, pool, cap, scale_stop=8, prototype_stop=3, count=2)
    np.testing.assert_array_equal(scale, later_scale)
    for key in prototype:
        np.testing.assert_array_equal(prototype[key], later_prototype[key])
    assert np.max(prototype["anchor_rows"]) < 3


def test_actual_fixed_rates_and_initial_activation_define_new_intercept():
    H = np.zeros((3, 391), np.float64)
    H[1:, :343] = 2.
    observed = np.zeros((3, 391), np.float32)
    observed[:, 343:] = np.arange(48, dtype=np.float32)[None] / 10
    free, fixed = np.arange(343), np.arange(343, 391)
    pool = np.zeros((96, 391))
    pool[np.arange(391) % 96, np.arange(391)] = 1.
    moment = np.zeros((42, 96))
    moment[0, 0] = 1.
    rest = np.zeros(42)
    ltm = np.arange(6)
    groups = [{"leg": leg} for leg in m.LEGS] + [{"leg": "lf"}] * 90
    initial_activation = np.zeros(96)
    initial_activation[0] = .4
    held = np.ones(42) * .3
    values = m.matched_intercept(H, observed, free, fixed, pool,
        moment, rest, ltm, groups, initial_activation, held)
    np.testing.assert_array_equal(values["rate_intercept"][:, free], H[:, free])
    np.testing.assert_array_equal(values["rate_intercept"][:, fixed],
                                  observed[:, fixed].astype(np.float64))
    np.testing.assert_array_equal(values["activation_intercept"][0],
                                  initial_activation)
    np.testing.assert_array_equal(values["q_intercept"][0], held)
    assert values["q_intercept"][1, 0] != held[0]
    observed[2, fixed] = 7.
    changed = m.matched_intercept(H, observed, free, fixed, pool,
        moment, rest, ltm, groups, initial_activation, held)
    assert changed["rate_intercept"][2, fixed[0]] == 7.
    assert not np.array_equal(changed["activation_intercept"][2],
                              values["activation_intercept"][2])


def test_same_control_targets_and_cross_split_velocity_boundary():
    ticks = np.arange(2001, dtype=np.int64)
    times = ticks * .005
    target = np.repeat((ticks[1:] / 100)[:, None], 42, axis=1)
    applied = np.vstack((np.full((1, 42), .25), target)).astype(np.float64)
    index = m.target_alignment(ticks, times, applied, target)
    assert index["train_controls"][[0, -1]].tolist() == [1, 1200]
    assert index["validation_controls"][[0, -1]].tolist() == [1201, 2000]
    assert index["train_velocity_start"][0] == 0
    assert index["validation_velocity_start"][0] == 1200
    assert index["validation_velocity_end"][0] == 1201
    bad = applied.copy()
    bad[1201, 0] += 1.
    with pytest.raises(ValueError, match="alignment"):
        m.target_alignment(ticks, times, bad, target)
    with pytest.raises(ValueError, match="alignment"):
        m.target_alignment(ticks, times + 1e-5, applied, target)


def test_gate_rejects_unsafe_source_and_premature_original_path():
    with pytest.raises(ValueError, match="Unsafe"):
        m.safe("../outside")
    with pytest.raises(ValueError, match="Exact E241"):
        m.gate("wrong-freeze", "wrong-sha", "wrong-review", "wrong-sha",
               m.OUT)


def test_float64_inverse_retains_exact_float32_origin():
    source = np.linspace(1., 5., 391, dtype=np.float32)
    saved = source.astype(np.float64)
    np.testing.assert_array_equal(m.validate_inverse_provenance(saved), saved)
    with pytest.raises(ValueError, match="float32-derived"):
        m.validate_inverse_provenance(source)
    changed = saved.copy()
    changed[20] = np.nextafter(changed[20], np.inf)
    with pytest.raises(ValueError, match="float32-derived"):
        m.validate_inverse_provenance(changed)


def fake_context(*, requested, completed, output=None):
    return SimpleNamespace(calls_requested=requested, calls_completed=completed,
        transfer_count=3, upload_bytes=100, parameter_transfer_count=2,
        parameter_upload_bytes=20, planned_peak_bytes=200,
        peak_owned_pool_bytes=180, peak_device_used_bytes=185,
        baseline_device_used_bytes=5, launch_events=[], current_output=output)


def test_readback_failure_receipt_preserves_synchronized_ticks():
    ledger = {"history": {"requested": 1, "synchronized": 1},
              "basis": None, "cuda_requested": 1, "cuda_synchronized": 1,
              "conditional_ticks_requested": 40000,
              "conditional_ticks_synchronized": 40000}
    failed_readback = fake_context(requested=1, completed=1, output={"B": object()})
    m.refresh_cuda_totals(ledger, failed_readback, "basis")
    assert ledger["cuda_requested"] == 2
    assert ledger["cuda_synchronized"] == 2
    assert ledger["conditional_ticks_requested"] == 80000
    assert ledger["conditional_ticks_synchronized"] == 80000
    assert ledger["basis"]["launch_events"] == []


def test_late_publication_fails_closed_and_keeps_first_cause(tmp_path):
    def case(initial, broken):
        result = {"validity": m.VALID}
        ledger = {"errors": []}
        saved = {}
        calls = 0

        def preliminary():
            nonlocal calls
            calls += 1
            if broken == "preliminary":
                raise OSError("late preliminary write failure")

        def write_manifest():
            if broken == "manifest":
                raise OSError("late manifest write failure")
            return {"total_bytes": 20}

        writers = {"preliminary": preliminary,
                   "manifest": write_manifest,
                   "result": lambda: saved.update(result=dict(result)),
                   "ledger": lambda: saved.update(ledger=dict(ledger)),
                   "errors": lambda: saved.update(errors=list(ledger["errors"]))}
        primary = m.finish_receipts(tmp_path, result, ledger, initial,
                                    lambda: 10., writers=writers)
        assert primary is initial if initial else isinstance(primary, OSError)
        assert calls == 1
        assert result["validity"] == "incomplete_or_invalid"
        assert saved["result"]["validity"] == result["validity"]
        assert saved["ledger"]["errors"] == ledger["errors"]
        assert ledger["errors"][0]["stage"] == (
            "result-preliminary" if broken == "preliminary" else "output-manifest")

    case(None, "preliminary")
    case(None, "manifest")
    case(ArithmeticError("science first"), "preliminary")


def test_publication_time_overrun_invalidates_after_writes(tmp_path):
    result = {"validity": m.VALID}
    ledger = {"errors": []}
    saved = {}
    clock = iter((100., 111., 111.))
    writers = {"preliminary": lambda: None,
               "manifest": lambda: {"total_bytes": 20},
               "result": lambda: saved.update(result=dict(result)),
               "ledger": lambda: saved.update(ledger=dict(ledger)),
               "errors": lambda: saved.update(errors=list(ledger["errors"]))}
    primary = m.finish_receipts(tmp_path, result, ledger, None,
                                lambda: next(clock), writers=writers)
    assert isinstance(primary, TimeoutError)
    assert result["validity"] == "incomplete_or_invalid"
    assert saved["result"]["validity"] == result["validity"]
    assert saved["ledger"]["errors"] == ledger["errors"]


def test_generated_cleanup_preserves_primary_partial_raw_and_deadline(tmp_path):
    receipt = {"valid": True}
    gpu_error = ArithmeticError("CUDA readback first")
    context = fake_context(requested=1, completed=1, output={"B": object()})
    saved = {}
    released = []

    def save_receipt():
        saved.update(receipt.copy())

    primary = m.finish_generated_smoke(tmp_path, receipt, gpu_error, 0.,
        (("basis", context),), save_receipt=save_receipt,
        release=lambda value: released.append(value),
        read_partial=lambda value: ({"H": np.array([1.])},
            [{"array": "B", "type": "OSError", "message": "readback"}]),
        save_partial=lambda label, values: saved.update(partial=(label, values)),
        clock=lambda: 31.)
    assert primary is gpu_error
    assert released == [context]
    assert saved["partial"][0] == "basis"
    assert saved["basis"]["synchronized"] == 1
    assert receipt["valid"] is False
    assert any(item["stage"] == "generated-internal-time"
               for item in receipt["errors"])
    assert "Secondary E241 generated" in gpu_error.__notes__[0]


def test_late_manifest_failure_retains_current_file_inventory(tmp_path, monkeypatch):
    result = {"validity": m.VALID}
    ledger = {"errors": []}
    m.save_json(tmp_path / "ledger.json", ledger)
    original = m.manifest
    calls = 0

    def fail_once(out):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("synthetic first manifest failure")
        return original(out)

    monkeypatch.setattr(m, "manifest", fail_once)
    error = m.finish_receipts(tmp_path, result, ledger, None, lambda: 10.)
    assert isinstance(error, OSError)
    saved = json.loads((tmp_path / "result.json").read_text("utf-8"))
    assert saved["validity"] == "incomplete_or_invalid"
    inventory = json.loads((tmp_path / "output-manifest.json").read_text("utf-8"))
    assert set(inventory["files"]) == {"ledger.json", "publication-failure.json"}
    for name, entry in inventory["files"].items():
        assert entry["sha256"] == m.prior.sha256(tmp_path / name)
    assert json.loads((tmp_path / "ledger.json").read_text("utf-8"))["errors"] == ledger["errors"]
