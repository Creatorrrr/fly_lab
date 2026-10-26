"""E213 generated two-memory math, fail-closed mocks, and deferred CUDA."""

import json
from pathlib import Path

import numpy as np
import pytest

from tools import prepare_c_banc_m3_multiscale_neural_basis as runner
from tools.banc_m3_causal_neural_basis import (
    simulate_basis_numpy as old_basis_numpy,
)
from tools.banc_m3_causal_neural_basis import (
    simulate_history_numpy as old_history_numpy,
)
from tools.banc_m3_multiscale_neural_basis import (
    PersistentMultiscaleInputs,
    normalized_basis,
    planned_gpu_bytes,
    reaction,
    run_basis_cuda,
    run_history_cuda,
    select_prototypes_9d,
    simulate_basis_numpy,
    simulate_history_numpy,
    validate_inverse_slow,
)


def generated_inputs():
    e = np.zeros((16, 2), np.float32)
    i = np.zeros_like(e)
    d = np.zeros_like(e)
    e[0, 0] = 1.
    e[3::4, 0] = np.array([.2, .6, .1, .9], np.float32)
    i[1::4, 0] = .3
    d[2::4, 0] = -.2
    i[:, 1] = .7
    return e, i, d


def generated_parameters():
    scale = np.ones((2, 3), np.float64)
    inverse = np.array([3.5, 5.], np.float32).astype(np.float64)
    initial = np.array([.4, 1.2])
    tau = np.array([.02, .015])
    cap = np.array([2., 3.])
    return scale, inverse, initial, tau, cap


def test_history_fast_exact_and_slow_stage_reference():
    e, i, d = generated_inputs()
    scale, inverse, *_ = generated_parameters()
    multiscale = simulate_history_numpy(e, i, d, scale, inverse, sample_every=1)
    old = old_history_numpy(e, i, d, scale, sample_every=1)
    assert multiscale["descriptor"].shape == (5, 2, 9)
    np.testing.assert_array_equal(multiscale["h_fast"], old["h"])
    np.testing.assert_array_equal(multiscale["descriptor"][0, :, 3:], 0)
    np.testing.assert_array_equal(multiscale["descriptor"][:, :, :6],
                                  old["descriptor"])
    x1 = np.log1p(1.)
    k1 = x1 * inverse[0]
    k2 = -(k1 * .000125) * inverse[0]
    k3 = -(k2 * .000125) * inverse[0]
    k4 = -(k3 * .00025) * inverse[0] + np.log1p(float(e[3, 0])) * inverse[0]
    expected = .00025 / 6 * (k1 + 2*k2 + 2*k3 + k4)
    assert multiscale["h_slow"][1, 0, 0] == pytest.approx(expected, abs=1e-15)
    assert k2 < 0
    assert reaction(np.zeros((1, 3)), np.array([[k2, 0., 0.]]))[0] == 0
    with pytest.raises(ValueError, match="float32-cast"):
        validate_inverse_slow(np.array([3., 0.]), 2)


def test_nine_dimensional_fps_ties_degenerate_sigma_and_partition():
    descriptors = np.zeros((4, 2, 9))
    descriptors[1, 0, 8] = 1.
    descriptors[2, 0, 8] = -1.
    descriptors[3, 0, 8] = 1.
    proto = select_prototypes_9d(descriptors)
    assert proto["rows"][0, :3].tolist() == [0, 1, 2]
    assert proto["rows"][1].tolist() == [0] * 16
    assert proto["sigma2"][1] == 1.
    weights = normalized_basis(descriptors[1], proto["centers"],
                               proto["sigma2"][:, None])
    np.testing.assert_allclose(np.sum(weights, axis=1), 1., atol=1e-15, rtol=0)
    assert np.all(weights > 0)


def test_generated_basis_probe_envelope_and_fast_homogeneous_exact():
    e, i, d = generated_inputs()
    scale, inverse, initial, tau, cap = generated_parameters()
    history = simulate_history_numpy(e, i, d, scale, inverse, sample_every=1)
    prototypes = select_prototypes_9d(history["descriptor"])
    basis = simulate_basis_numpy(e, i, d, scale, inverse, prototypes["centers"],
        prototypes["sigma2"], initial, tau, cap, sample_every=1)
    runner.validate_history(history, scale, inverse, 2, 5)
    runner.validate_basis(history, basis, initial, cap)
    old_history = old_history_numpy(e, i, d, scale, sample_every=1)
    old_proto = select_prototypes_9d(
        np.concatenate((old_history["descriptor"],
                        np.zeros((5, 2, 3))), axis=2))
    old_basis = old_basis_numpy(e, i, d, scale, old_proto["centers"][:, :, :6],
        old_proto["sigma2"], initial, tau, cap, sample_every=1)
    np.testing.assert_array_equal(basis["h_fast"], old_basis["h"])
    np.testing.assert_array_equal(basis["H"], old_basis["H"])
    np.testing.assert_array_equal(basis["H_factor"], old_basis["H_factor"])
    np.testing.assert_allclose(np.sum(basis["B"], axis=2),
                               np.sum(old_basis["B"], axis=2),
                               atol=2e-10, rtol=2e-11)
    assert np.max(np.abs(basis["B"][:, 1])) == 0


def test_gpu_plan_and_two_launch_bound_without_device():
    plan = planned_gpu_bytes((160000, 391))
    assert plan["channel_bytes"] == 3 * 160000 * 391 * 4
    assert plan["output_bytes"] == 2001 * 391 * 49 * 8
    assert plan["parameter_bytes"] == 391 * 156 * 8
    assert plan["peak_bytes"] < runner.RESOURCE["gpu_bytes"]
    with pytest.raises(ValueError, match="complete"):
        planned_gpu_bytes((15, 2), sample_every=1)


def test_fake_first_nonfinite_blocks_second_and_preserves_raw(tmp_path, monkeypatch):
    class Fake:
        n = 2
        ticks = 2

    calls = []

    def first(_context, _scale, _inverse, *, sample_every):
        calls.append("history")
        return {"h_fast": np.full((3, 2, 3), np.nan),
                "h_slow": np.zeros((3, 2, 3)),
                "descriptor": np.zeros((3, 2, 9))}, .01

    def second(*_args, **_kwargs):
        calls.append("basis")
        raise AssertionError("basis must not launch after bad history")

    monkeypatch.setattr(runner, "run_history_cuda", first)
    monkeypatch.setattr(runner, "run_basis_cuda", second)
    scale, inverse, initial, tau, cap = generated_parameters()
    with pytest.raises(ValueError, match="nonfinite"):
        runner.collect_conditional(tmp_path, Fake(), scale, inverse,
            initial, tau, cap, sample_every=1)
    assert calls == ["history"]
    with np.load(tmp_path / "history.npz") as saved:
        assert np.isnan(saved["h_fast"]).all()


def test_fake_basis_nonfinite_and_post_basis_deadline_keep_both_raw(
        tmp_path, monkeypatch):
    class Fake:
        n = 2
        ticks = 2

    def first(_context, _scale, _inverse, *, sample_every):
        return {"h_fast": np.zeros((3, 2, 3)),
                "h_slow": np.zeros((3, 2, 3)),
                "descriptor": np.zeros((3, 2, 9))}, .01

    def second(*_args, **_kwargs):
        return {"h_fast": np.zeros((3, 2, 3)),
                "h_slow": np.zeros((3, 2, 3)),
                "descriptor": np.zeros((3, 2, 9)),
                "B": np.full((3, 2, 16), np.nan),
                "H": np.zeros((3, 2)), "H_factor": np.ones((3, 2)),
                "probe_rate": np.zeros((3, 2))}, .02

    monkeypatch.setattr(runner, "run_history_cuda", first)
    monkeypatch.setattr(runner, "run_basis_cuda", second)
    scale, inverse, initial, tau, cap = generated_parameters()
    with pytest.raises(TimeoutError, match="generated deadline"):
        runner.collect_conditional(tmp_path, Fake(), scale, inverse,
            initial, tau, cap, sample_every=1,
            after_basis=lambda: (_ for _ in ()).throw(
                TimeoutError("generated deadline")))
    assert (tmp_path / "history.npz").is_file()
    assert (tmp_path / "basis.npz").is_file()
    with np.load(tmp_path / "basis.npz") as saved:
        assert np.isnan(saved["B"]).all()


def test_partial_readback_preserves_accessible_siblings():
    class FakeCp:
        @staticmethod
        def asnumpy(value):
            if value == "bad":
                raise RuntimeError("generated readback failure")
            return np.array([value])

    context = object.__new__(PersistentMultiscaleInputs)
    context.cp = FakeCp()
    context.current_output = {"first": "already", "bad": "bad", "last": "good"}
    context.partial_host = {"first": np.array([1])}
    values, failures = context.partial_readback()
    assert set(values) == {"first", "last"}
    assert len(failures) == 1 and failures[0]["array"] == "bad"


def test_gate_rejects_unapproved_review_before_device(tmp_path):
    freeze = Path("verification/banc-natural-gait-m3-20260919/"
                  "e213-multiscale-neural-basis-freeze.json")
    review = tmp_path / "review.json"
    review.write_text(json.dumps({"schema": runner.REVIEW_SCHEMA,
        "approved": False, "blocking_defects": ["generated reject"],
        "freeze_sha256": runner.FREEZE_SHA, "source_sha256": {}}), encoding="utf-8")
    with pytest.raises(ValueError, match="source review"):
        runner.verify_gate(freeze, runner.FREEZE_SHA, review, runner.file_sha(review))
    with pytest.raises(ValueError, match="frozen SHA"):
        runner.verify_gate(freeze, "0" * 64, review, runner.file_sha(review))


def test_generated_cuda_two_launch_multiscale_only(tmp_path):
    """DEFERRED after parent review: 4 history + 4 basis ticks, zero body."""
    import time

    import torch

    assert torch.cuda.is_available(), "Parent FIRST must precede generated CUDA"
    e, i, d = generated_inputs()
    scale, inverse, initial, tau, cap = generated_parameters()
    started = time.perf_counter()
    context = PersistentMultiscaleInputs(e, i, d, sample_every=1)
    history, first_wall = run_history_cuda(context, scale, inverse, sample_every=1)
    np.savez(tmp_path / "generated-history-cuda-raw.npz", **history)
    prototypes = select_prototypes_9d(history["descriptor"])
    basis, second_wall = run_basis_cuda(context, scale, inverse, prototypes,
        initial, tau, cap, sample_every=1)
    np.savez(tmp_path / "generated-basis-cuda-raw.npz", **basis)
    np.savez(tmp_path / "generated-prototypes-raw.npz", **prototypes)
    reference_history = simulate_history_numpy(e, i, d, scale, inverse,
                                                sample_every=1)
    reference_basis = simulate_basis_numpy(e, i, d, scale, inverse,
        prototypes["centers"], prototypes["sigma2"], initial, tau, cap,
        sample_every=1)
    for key in history:
        np.testing.assert_allclose(history[key], reference_history[key],
                                   atol=1e-12, rtol=1e-12)
        np.testing.assert_array_equal(history[key], basis[key])
    for key in basis:
        np.testing.assert_allclose(basis[key], reference_basis[key],
                                   atol=1e-11, rtol=1e-11)
    runner.validate_basis(history, basis, initial, cap)
    assert context.calls_requested == context.calls_completed == 2
    print("E213_GENERATED_CUDA", {"launches": 2, "generated_ticks": 8,
        "body_steps": 0, "warmup_calls": 0, "first_wall_s": first_wall,
        "second_wall_s": second_wall, "total_wall_s": time.perf_counter() - started,
        "history_raw": str(tmp_path / "generated-history-cuda-raw.npz"),
        "basis_raw": str(tmp_path / "generated-basis-cuda-raw.npz")})
