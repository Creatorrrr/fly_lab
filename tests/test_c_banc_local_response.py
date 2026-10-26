"""Generated conditional MN response tests; original E150/E201 is never run."""

import json
import time
from pathlib import Path

import numpy as np
import pytest

from tools import verify_c_banc_m3_local_response as runner
from tools.banc_m3_local_response import simulate_cuda, simulate_numpy


def fixture_inputs(motors=3, steps=40):
    current = np.empty((4 * steps, motors), np.float32)
    for stage in range(4 * steps):
        current[stage] = np.asarray([-5 + stage % 3, 4 + .03 * stage,
                                     1.5 - .01 * stage][:motors], np.float32)
    p = {"initial_rate": np.array([3., 1., 2.][:motors], np.float32),
         "initial_adaptation": np.array([.4, .2, .6][:motors], np.float32),
         "tau": np.array([.02, .03, .04][:motors], np.float32),
         "threshold": np.array([1., .5, 1.][:motors], np.float32),
         "cap": np.array([100., 90., 120.][:motors], np.float32),
         "coefficient": np.array([.06, .08, .04][:motors], np.float32),
         "adaptation_gain": np.array([6., 7., 5.][:motors], np.float32),
         "inverse_adaptation_tau": np.array([3., 2., 4.][:motors], np.float32)}
    return current, p


def test_zero_nonzero_and_seven_local_finite_differences():
    current, p = fixture_inputs()
    zero = np.zeros((3, 7), np.float64)
    baseline = simulate_numpy(current, p, zero)
    assert baseline["rate"].shape == (3, 3)
    np.testing.assert_array_equal(baseline["rate_sensitivity"][0], 0)
    # Negative-current cell has a rectified zero target and decays.
    assert np.all(np.diff(baseline["rate"][:, 0]) < 0)
    w = zero.copy()
    w[1] = [.1, -.2, .05, .03, .08, -.04, .02]
    result = simulate_numpy(current, p, w)
    assert not np.array_equal(result["rate"][:, 1], baseline["rate"][:, 1])
    eps = 1e-5
    for j in range(7):
        plus, minus = w.copy(), w.copy()
        plus[1, j] += eps
        minus[1, j] -= eps
        a = simulate_numpy(current, p, plus)
        b = simulate_numpy(current, p, minus)
        np.testing.assert_allclose((a["rate"][:, 1] - b["rate"][:, 1]) / (2 * eps),
            result["rate_sensitivity"][:, 1, j], atol=2e-5, rtol=2e-4)
        np.testing.assert_allclose((a["adaptation"][:, 1] - b["adaptation"][:, 1]) / (2 * eps),
            result["adaptation_sensitivity"][:, 1, j], atol=2e-5, rtol=2e-4)
        np.testing.assert_array_equal(a["rate"][:, 0], result["rate"][:, 0])
        np.testing.assert_array_equal(a["rate"][:, 2], result["rate"][:, 2])


def test_cuda_numpy_forward_and_tangent_parity():
    import cupy as cp

    assert cp.cuda.runtime.getDeviceCount() >= 1
    current, p = fixture_inputs(steps=40)
    w = np.zeros((3, 7), np.float64)
    w[1] = [.1, -.2, .05, .03, .08, -.04, .02]
    setup_began = time.perf_counter()
    timing = {}
    cuda = simulate_cuda(current, p, w, timing=timing)
    cuda_wall = time.perf_counter() - setup_began
    ref_began = time.perf_counter()
    reference = simulate_numpy(current, p, w)
    reference_wall = time.perf_counter() - ref_began
    for key in reference:
        np.testing.assert_allclose(cuda[key], reference[key], atol=1e-9, rtol=1e-8)
    assert np.isfinite(cuda["rate_sensitivity"]).all()
    print("E202_GENERATED_COUNT", {"cuda_models": 0, "cuda_launches": 1,
        "cuda_generated_ticks": 40, "cuda_stages": 160,
        "cuda_warmup_ticks": timing["generated_warmup_ticks"],
        "cuda_compile_s": timing["compile_s"],
        "cuda_kernel_main_s": timing["kernel_main_s"],
        "cuda_compile_plus_main_wall_s": cuda_wall,
        "numpy_calls": 1, "numpy_generated_ticks": 40,
        "numpy_wall_s": reference_wall, "body_steps": 0})
    assert timing["conditional_cuda_launches"] == 1


def test_cuda_fortran_and_permuted_motor_columns():
    current, p = fixture_inputs(steps=20)
    w = np.zeros((3, 7), np.float64)
    w[:, 0] = [.1, -.2, .3]
    order = np.array([2, 0, 1])
    reordered_current = np.asfortranarray(current[:, order])
    reordered_weights = np.asfortranarray(w[order])
    reordered_p = {key: np.asfortranarray(value[order][None, :])[0]
                   for key, value in p.items()}
    assert reordered_current.flags.f_contiguous and not reordered_current.flags.c_contiguous
    assert reordered_weights.flags.f_contiguous and not reordered_weights.flags.c_contiguous
    cuda = simulate_cuda(reordered_current, reordered_p, reordered_weights)
    reference = simulate_numpy(reordered_current, reordered_p, reordered_weights)
    natural = simulate_numpy(current, p, w)
    for key in reference:
        np.testing.assert_allclose(cuda[key], reference[key], atol=1e-9, rtol=1e-8)
        np.testing.assert_allclose(cuda[key], natural[key][:, order], atol=1e-9, rtol=1e-8)


def test_rectifier_zero_boundary_has_zero_tangent():
    current = np.ones((4, 1), np.float32)
    p = {"initial_rate": np.zeros(1, np.float32),
         "initial_adaptation": np.zeros(1, np.float32),
         "tau": np.array([.02], np.float32),
         "threshold": np.ones(1, np.float32),
         "cap": np.array([100.], np.float32),
         "coefficient": np.array([.1], np.float32),
         "adaptation_gain": np.array([6.], np.float32),
         "inverse_adaptation_tau": np.array([3.], np.float32)}
    weights = np.zeros((1, 7), np.float64)
    for solver in (simulate_numpy, simulate_cuda):
        value = solver(current, p, weights, sample_every=1)
        np.testing.assert_array_equal(value["rate"], 0)
        np.testing.assert_array_equal(value["adaptation"], 0)
        np.testing.assert_array_equal(value["rate_sensitivity"], 0)
        np.testing.assert_array_equal(value["adaptation_sensitivity"], 0)


def test_cuda_readback_failure_retains_synchronized_phase_count(monkeypatch):
    import cupy as cp

    current, p = fixture_inputs(steps=20)
    timing = {}

    def fail_readback(_array):
        raise RuntimeError("generated readback failure")

    monkeypatch.setattr(cp, "asnumpy", fail_readback)
    with pytest.raises(RuntimeError, match="generated readback failure"):
        simulate_cuda(current, p, np.zeros((3, 7)), timing=timing)
    assert timing["cuda_launches_requested"] == 1
    assert timing["cuda_launches_completed"] == 1
    assert timing["conditional_ticks_completed"] == 20
    assert timing["stage_inputs_consumed"] == 80
    assert timing["kernel_main_s"] >= 0


def test_invalid_shapes_weights_and_stage_order():
    current, p = fixture_inputs()
    w = np.zeros((3, 7), np.float64)
    with pytest.raises(ValueError, match="seven weights"):
        simulate_numpy(current, p, np.full((3, 7), 1.01))
    with pytest.raises(ValueError, match="four-stage"):
        simulate_numpy(current[:-1], p, w)
    with pytest.raises(ValueError, match="four-stage"):
        simulate_numpy(np.empty((4, 0), np.float32), p, np.empty((0, 7)))
    with pytest.raises(ValueError, match="inverse adaptation"):
        simulate_numpy(current, {**p, "inverse_adaptation_tau": np.zeros(3)}, w)
    shifted = current.copy()
    shifted[[1, 2]] = shifted[[2, 1]]
    assert not np.array_equal(simulate_numpy(current, p, w)["rate"],
                              simulate_numpy(shifted, p, w)["rate"])


def test_snapshot_decoder_updates_fixed_activation_and_holds_row_zero():
    membership = np.arange(391) % 96
    pool = np.zeros((96, 391), np.float64)
    counts = np.bincount(membership, minlength=96)
    pool[membership, np.arange(391)] = 1 / counts[membership]
    groups = [{"leg": runner.LEGS[i % 6], "muscle": str(i)} for i in range(96)]
    start = {"adapter": {"calibration": {"pool": pool,
        "moment": np.zeros((42, 96)), "rest": np.zeros(42),
        "ltm": np.arange(6), "motor_ids": np.arange(391)},
        "parameters": {"membership": membership, "kappa": np.zeros(391),
        "slopes": np.zeros((96, 391, 3, 2)), "fixed_muscles": np.arange(96),
        "tau": np.array([.005, .01, .04])},
        "bank": np.zeros((391, 3)), "activation": np.zeros(96),
        "original": {"activation": np.zeros(96)}, "groups": groups},
        "body": {"action": {"joint_angles": np.full(42, 17.)}}}
    rates = np.ones((3, 391), np.float64)
    decoded = runner.decoder_from_snapshot(start, rates)
    assert decoded["bank"].shape == (3, 391, 3)
    np.testing.assert_array_equal(decoded["command"][0], 17.)
    np.testing.assert_array_equal(decoded["command"][1:], 0.)
    alpha = -np.expm1(-.005 / .01)
    np.testing.assert_allclose(decoded["activation"][1], alpha)
    np.testing.assert_allclose(decoded["activation"][2], alpha * (2 - alpha))
    assert decoded["tendon_names"].tolist() == [leg + "_tarsus" for leg in runner.LEGS]


def test_frozen_gate_rejects_review_before_original_read(tmp_path):
    freeze = Path("verification/banc-natural-gait-m3-20260919/"
                  "e202-local-response-freeze.json")
    review = tmp_path / "review.json"
    review.write_text(json.dumps({"schema": runner.REVIEW_SCHEMA,
        "approved": False, "blocking_defects": ["generated"],
        "freeze_sha256": runner.file_sha(freeze), "source_sha256": {}}),
        encoding="utf-8")
    with pytest.raises(ValueError, match="source review"):
        runner.verify_gate(freeze, runner.file_sha(freeze),
            review, runner.file_sha(review))
    with pytest.raises(ValueError, match="freeze SHA"):
        runner.verify_gate(freeze, "0" * 64, review, runner.file_sha(review))


def test_runner_fidelity_failure_keeps_raw_metrics_and_fails(tmp_path, monkeypatch):
    paths = {key: key for key in ("current", "parameters", "endpoints",
        "observation", "trace", "native", "start", "e201_manifest",
        "e201_result", "e201_review")}
    monkeypatch.setattr(runner, "verify_gate", lambda *args, **kwargs: (
        {"paths": paths, "input_sha256": {}, "prior_cumulative_s": 10.},
        {"source_sha256": {}}))
    monkeypatch.setattr(runner, "repository_path", lambda name, **kwargs: tmp_path / name)
    fake_parameters = {"motor_ids": np.arange(391, dtype=np.int64),
        **{key: np.ones(391, np.float32) for key in (
            "initial_rate", "initial_adaptation", "tau", "threshold", "cap",
            "coefficient", "adaptation_gain", "inverse_adaptation_tau")}}
    monkeypatch.setattr(runner, "load_original", lambda paths: (
        np.ones((4, 391), np.float32), fake_parameters, {}, {}, {}, {}))

    def fake_cuda(current, parameters, weights, *, timing):
        timing.update(compile_s=.01, kernel_main_s=.02,
            cuda_launches_requested=1, cuda_launches_completed=1,
            conditional_ticks_completed=1, stage_inputs_consumed=4)
        return {"rate": np.ones((2, 391)), "adaptation": np.zeros((2, 391)),
            "rate_sensitivity": np.zeros((2, 391, 7)),
            "adaptation_sensitivity": np.zeros((2, 391, 7))}

    monkeypatch.setattr(runner, "simulate_cuda", fake_cuda)
    monkeypatch.setattr(runner, "decoder_from_snapshot", lambda start, rates: {
        "bank": np.zeros((2, 391, 3)), "activation": np.zeros((2, 96)),
        "command": np.zeros((2, 42)), "tendon": np.zeros((2, 6)),
        "tendon_names": np.asarray([leg + "_tarsus" for leg in runner.LEGS])})
    monkeypatch.setattr(runner, "evaluate_raw", lambda *args: {
        "rate": {"rms": .6, "max_abs": 1., "pass": False},
        "activation": {"rms": 0., "max_abs": 0., "pass": True},
        "command": {"rms": 0., "max_abs": 0., "pass": True},
        "tendon": {"rms": 0., "max_abs": 0., "pass": True}})
    out = tmp_path / "output"
    with pytest.raises(ValueError, match="exceeded fidelity"):
        runner.run("unused", "unused", "unused", "unused", out)
    assert (out / "local-response.npz").is_file()
    assert (out / "decoded.npz").is_file()
    result = json.loads((out / "result.json").read_text(encoding="utf-8"))
    assert result["validity"] == "incomplete_or_invalid"
    assert result["metrics"]["rate"]["rms"] == .6
    ledger = json.loads((out / "ledger.json").read_text(encoding="utf-8"))
    assert ledger["conditional_calls_completed"] == 1
    assert ledger["cuda_phase_timing"]["conditional_ticks_completed"] == 1
