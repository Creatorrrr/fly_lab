"""Generated E209 math, gate, and failure preservation; no original simulation."""

import json
from pathlib import Path

import numpy as np
import pytest

from tools import prepare_c_banc_m3_causal_neural_basis as runner
from tools.banc_m3_causal_neural_basis import (
    PersistentSignedInputs,
    compute_scales,
    normalized_basis,
    planned_gpu_bytes,
    reaction,
    run_basis_cuda,
    run_history_cuda,
    select_prototypes,
    simulate_basis_numpy,
    simulate_history_numpy,
)


def generated_inputs():
    e = np.zeros((16, 2), np.float32)
    i = np.zeros_like(e)
    d = np.zeros_like(e)
    e[0, 0] = 1.
    e[3::4, 0] = np.array([.2, .6, .1, .9], np.float32)
    i[1::4, 0] = .3
    d[2::4, 0] = -.2
    i[:, 1] = .7  # Pure inhibition cannot create rate from zero.
    return e, i, d


def test_history_stage_clock_and_discontinuous_negative_intermediate():
    e, i, d = generated_inputs()
    scale = compute_scales(e, i, d, block=3)
    history = simulate_history_numpy(e, i, d, scale, sample_every=1)
    assert history["h"].shape == (5, 2, 3)
    np.testing.assert_array_equal(history["h"][0], 0)
    np.testing.assert_allclose(history["descriptor"][0, 0, 0],
        np.log1p(1.) / scale[0, 0], rtol=0, atol=1e-15)
    np.testing.assert_array_equal(history["descriptor"][0, :, 3:], 0)
    x1 = np.log1p(1.) / scale[0, 0]
    k1 = x1 / .016
    k2 = -(.000125 * k1) / .016
    h3 = .000125 * k2
    assert h3 < 0
    assert reaction(np.zeros((1, 3)), np.array([[h3, 0., 0.]]))[0] == 0
    assert history["h"][1, 0, 0] >= 0


def test_farthest_prototype_first_row_ties_duplicates_and_partition():
    descriptor = np.zeros((4, 2, 6), np.float64)
    descriptor[1, 0, 0] = 1
    descriptor[2, 0, 0] = -1
    descriptor[3, 0, 0] = 1
    prototypes = select_prototypes(descriptor)
    assert prototypes["rows"][0, :3].tolist() == [0, 1, 2]
    assert prototypes["rows"][1].tolist() == [0] * 16
    assert prototypes["sigma2"][1] == 1
    weights = normalized_basis(descriptor[1], prototypes["centers"],
                               prototypes["sigma2"][:, None])
    np.testing.assert_allclose(np.sum(weights, axis=1), 1, rtol=0, atol=1e-15)
    assert np.all(weights > 0)
    assert reaction(np.zeros((2, 3)), np.zeros((2, 3))).tolist() == [0., 0.]


def test_generated_basis_probe_affine_and_nonzero_initial_rate():
    e, i, d = generated_inputs()
    scale = compute_scales(e, i, d)
    history = simulate_history_numpy(e, i, d, scale, sample_every=1)
    prototypes = select_prototypes(history["descriptor"])
    initial = np.array([.4, 1.2])
    tau = np.array([.02, .015])
    cap = np.array([2., 3.])
    basis = simulate_basis_numpy(e, i, d, scale, prototypes["centers"],
        prototypes["sigma2"], initial, tau, cap, sample_every=1)
    runner.validate_basis(history, basis, initial, cap)
    beta = (np.arange(16) + 1) / 17
    np.testing.assert_allclose(basis["probe_rate"],
        basis["H"] + cap[None] * np.sum(basis["B"] * beta, axis=2),
        rtol=1e-12, atol=1e-12)
    assert basis["probe_rate"][0].tolist() == initial.tolist()
    assert np.max(np.abs(basis["B"][:, 1])) == 0
    assert np.min(basis["B"]) >= 0


def test_exact_gpu_byte_plan_before_allocation():
    plan = planned_gpu_bytes((160000, 391))
    assert plan["channel_bytes"] == 3 * 160000 * 391 * 4
    assert plan["output_bytes"] == 2001 * 391 * 37 * 8
    assert plan["parameter_bytes"] == 391 * 106 * 8
    assert plan["peak_bytes"] == sum(plan[key] for key in
        ("channel_bytes", "output_bytes", "parameter_bytes"))
    assert plan["peak_bytes"] < runner.RESOURCE["gpu_buffer_bytes"]


def test_fake_first_failure_blocks_second_and_preserves_history_raw(tmp_path, monkeypatch):
    class Fake:
        n = 2
        ticks = 2

    calls = []

    def first(_context, _scale, *, sample_every):
        calls.append("history")
        return {"h": np.full((3, 2, 3), np.nan),
                "descriptor": np.zeros((3, 2, 6))}, .01

    def second(*_args, **_kwargs):
        calls.append("basis")
        raise AssertionError("second launch must be blocked")

    monkeypatch.setattr(runner, "run_history_cuda", first)
    monkeypatch.setattr(runner, "run_basis_cuda", second)
    with pytest.raises(ValueError, match="nonfinite"):
        runner.collect_conditional(tmp_path, Fake(), np.ones((2, 3)),
            np.zeros(2), np.ones(2), np.ones(2), sample_every=1)
    assert calls == ["history"]
    with np.load(tmp_path / "history.npz") as raw:
        assert np.isnan(raw["h"]).all()


def test_fake_basis_nonfinite_preserved_before_verdict(tmp_path, monkeypatch):
    class Fake:
        n = 2
        ticks = 2

    def first(_context, _scale, *, sample_every):
        return {"h": np.zeros((3, 2, 3)),
                "descriptor": np.zeros((3, 2, 6))}, .01

    def second(*_args, **_kwargs):
        basis = {"h": np.zeros((3, 2, 3)),
            "descriptor": np.zeros((3, 2, 6)),
            "B": np.full((3, 2, 16), np.nan),
            "H": np.zeros((3, 2)), "H_factor": np.ones((3, 2)),
            "probe_rate": np.zeros((3, 2))}
        return basis, .02

    monkeypatch.setattr(runner, "run_history_cuda", first)
    monkeypatch.setattr(runner, "run_basis_cuda", second)
    with pytest.raises(ValueError, match="finite"):
        runner.collect_conditional(tmp_path, Fake(), np.ones((2, 3)),
            np.zeros(2), np.ones(2), np.ones(2), sample_every=1)
    with np.load(tmp_path / "basis.npz") as raw:
        assert np.isnan(raw["B"]).all()


def test_deadline_after_basis_keeps_both_raw_sets(tmp_path, monkeypatch):
    class Fake:
        n = 2
        ticks = 2

    def first(_context, _scale, *, sample_every):
        return {"h": np.zeros((3, 2, 3)),
                "descriptor": np.zeros((3, 2, 6))}, .01

    def second(*_args, **_kwargs):
        return {"h": np.zeros((3, 2, 3)),
            "descriptor": np.zeros((3, 2, 6)),
            "B": np.zeros((3, 2, 16)),
            "H": np.zeros((3, 2)), "H_factor": np.ones((3, 2)),
            "probe_rate": np.zeros((3, 2))}, .02

    def deadline():
        raise TimeoutError("generated model deadline")

    monkeypatch.setattr(runner, "run_history_cuda", first)
    monkeypatch.setattr(runner, "run_basis_cuda", second)
    with pytest.raises(TimeoutError, match="deadline"):
        runner.collect_conditional(tmp_path, Fake(), np.ones((2, 3)),
            np.zeros(2), np.ones(2), np.ones(2),
            sample_every=1, after_basis=deadline)
    assert (tmp_path / "history.npz").is_file()
    assert (tmp_path / "basis.npz").is_file()


def test_partial_readback_keeps_accessible_siblings():
    class FakeCp:
        @staticmethod
        def asnumpy(value):
            if value == "bad":
                raise RuntimeError("generated readback failure")
            return np.array([value])

    context = object.__new__(PersistentSignedInputs)
    context.cp = FakeCp()
    context.current_output = {"first": "already", "bad": "bad", "last": "good"}
    context.partial_host = {"first": np.array([1])}
    arrays, errors = context.partial_readback()
    assert set(arrays) == {"first", "last"}
    assert arrays["last"].tolist() == ["good"]
    assert len(errors) == 1 and errors[0]["array"] == "bad"


def test_canonical_affine_raw_with_nonzero_original_state_and_fixed_mn(tmp_path):
    membership = np.empty(391, np.int64)
    membership[:48] = np.arange(48) % 6
    membership[48:] = 6 + np.arange(343) % 90
    pool = np.zeros((96, 391))
    pool[membership, np.arange(391)] = 1
    groups = [{"leg": runner.LEGS[m] if m < 6 else runner.LEGS[0]}
              for m in range(96)]
    start = {"adapter": {"calibration": {"pool": pool,
        "moment": np.ones((42, 96)) * .01, "rest": np.zeros(42),
        "ltm": np.arange(6)}, "parameters": {"membership": membership},
        "original": {"activation": np.linspace(.1, .2, 96)},
        "groups": groups},
        "body": {"action": {"joint_angles": np.ones(42) * .3}}}
    free, fixed = np.arange(48, 391), np.arange(48)
    basis = {"H": np.ones((3, 391)) * .1,
             "B": np.zeros((3, 391, 16))}
    basis["B"][1:, free, 0] = .2
    cap = np.ones(391) * 2
    original_rate = np.ones((3, 391)) * .3
    affine = runner.affine_readout(start, original_rate, basis, cap, free, fixed)
    original = runner.canonical_readout(start, original_rate)
    checks = runner.validate_affine_readout(start, affine, original, fixed)
    assert checks["fixed_ltm_tendon_exact"]
    assert np.count_nonzero(affine["D"][:, free]) > 0
    assert np.count_nonzero(affine["D"][:, fixed]) == 0
    assert affine["q_intercept"][0].tolist() == [0.3] * 42
    np.testing.assert_allclose(affine["q_probe_affine"],
        affine["q_probe_direct"], rtol=2e-12, atol=2e-12)
    bad = dict(affine)
    bad["q_probe_affine"] = affine["q_probe_affine"].copy()
    bad["q_probe_affine"][1, 0] += .01
    runner.save_arrays(tmp_path / "affine-readout.npz", bad)
    with pytest.raises(ValueError, match="affine command"):
        runner.validate_affine_readout(start, bad, original, fixed)
    with np.load(tmp_path / "affine-readout.npz") as saved:
        assert saved["q_probe_affine"][1, 0] == bad["q_probe_affine"][1, 0]


def test_gate_rejects_unapproved_sources_before_device(tmp_path):
    freeze = Path("verification/banc-natural-gait-m3-20260919/"
                  "e209-causal-neural-basis-freeze.json")
    review = tmp_path / "review.json"
    review.write_text(json.dumps({"schema": runner.REVIEW_SCHEMA,
        "approved": False, "blocking_defects": ["generated reject"],
        "freeze_sha256": runner.FREEZE_SHA, "source_sha256": {}}), encoding="utf-8")
    with pytest.raises(ValueError, match="source review"):
        runner.verify_gate(freeze, runner.FREEZE_SHA, review, runner.file_sha(review))
    with pytest.raises(ValueError, match="frozen SHA"):
        runner.verify_gate(freeze, "0" * 64, review, runner.file_sha(review))


def test_generated_cuda_two_conditional_launches_only():
    """Deferred one CUDA test: two kernels, 4 generated ticks each, body0."""
    import time

    import torch

    assert torch.cuda.is_available(), "Parent FIRST must precede generated CUDA"
    e, i, d = generated_inputs()
    scale = compute_scales(e, i, d)
    start = time.perf_counter()
    context = PersistentSignedInputs(e, i, d, sample_every=1)
    history, first_wall = run_history_cuda(context, scale, sample_every=1)
    prototypes = select_prototypes(history["descriptor"])
    initial = np.array([.4, 1.2])
    tau = np.array([.02, .015])
    cap = np.array([2., 3.])
    basis, second_wall = run_basis_cuda(context, scale, prototypes,
        initial, tau, cap, sample_every=1)
    reference_history = simulate_history_numpy(e, i, d, scale, sample_every=1)
    reference_basis = simulate_basis_numpy(e, i, d, scale,
        prototypes["centers"], prototypes["sigma2"], initial, tau, cap,
        sample_every=1)
    for key in history:
        np.testing.assert_allclose(history[key], reference_history[key], rtol=1e-12, atol=1e-12)
        np.testing.assert_array_equal(history[key], basis[key])
    for key in basis:
        np.testing.assert_allclose(basis[key], reference_basis[key], rtol=1e-11, atol=1e-11)
    runner.validate_basis(history, basis, initial, cap)
    assert context.calls_requested == context.calls_completed == 2
    print("E209_GENERATED_CUDA", {"launches": 2, "generated_ticks": 8,
        "body_steps": 0, "warmup_calls": 0, "first_wall_s": first_wall,
        "second_wall_s": second_wall, "total_wall_s": time.perf_counter() - start})
