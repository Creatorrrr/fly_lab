"""Generated E203 persistent CUDA, decoder adjoint, and bounded-search tests."""

import time
from types import SimpleNamespace

import numpy as np
import pytest

from tools import fit_c_banc_m3_local_response as fit
from tools.banc_m3_local_response import simulate_cuda, simulate_numpy
from tools.fit_c_banc_m3_local_response import (
    DecoderAdjoint,
    PersistentLocalResponse,
    _ema,
    _ema_adjoint,
    direct_metrics,
    feasible,
    objective_and_command_gradient,
)


def generated_neural(motors=3, steps=20):
    current = np.full((4 * steps, motors), 3., np.float32)
    p = {"initial_rate": np.full(motors, 1., np.float32),
         "initial_adaptation": np.full(motors, .2, np.float32),
         "tau": np.full(motors, .03, np.float32),
         "threshold": np.full(motors, .5, np.float32),
         "cap": np.full(motors, 100., np.float32),
         "coefficient": np.full(motors, .08, np.float32),
         "adaptation_gain": np.full(motors, 6., np.float32),
         "inverse_adaptation_tau": np.full(motors, 3., np.float32)}
    return current, p


def generated_snapshot():
    membership = np.arange(391) % 96
    pool = np.zeros((96, 391), np.float64)
    counts = np.bincount(membership, minlength=96)
    pool[membership, np.arange(391)] = 1 / counts[membership]
    fixed = np.array([4, 15, 16, 19, 30, 31, 34, 35, 46, 47,
                      52, 63, 64, 67, 78, 82, 83, 94, 95])
    slopes = np.zeros((96, 391, 3, 2), np.float64)
    for motor, muscle in enumerate(membership):
        slopes[muscle, motor, :, 0] = .5 / counts[muscle]
        slopes[muscle, motor, :, 1] = 1.5 / counts[muscle]
    groups = [{"leg": ("lf", "lm", "lh", "rf", "rm", "rh")[i % 6],
               "muscle": str(i)} for i in range(96)]
    ltm = np.array([15, 30, 46, 63, 78, 94])
    for index, leg in zip(ltm, ("lf", "lm", "lh", "rf", "rm", "rh")):
        groups[int(index)]["leg"] = leg
    moment = np.arange(42 * 96, dtype=np.float64).reshape(42, 96) * 1e-5
    return {"adapter": {"calibration": {"pool": pool, "moment": moment,
        "rest": np.zeros(42), "ltm": ltm, "motor_ids": np.arange(391)},
        "parameters": {"membership": membership, "kappa": np.full(391, .13),
        "slopes": slopes, "fixed_muscles": fixed,
        "tau": np.array([.005, .01, .04])},
        "bank": np.zeros((391, 3)), "activation": np.zeros(96),
        "original": {"activation": np.zeros(96)}, "groups": groups},
        "body": {"action": {"joint_angles": np.full(42, 7.)}}}


def test_persistent_device_matches_e202_and_restarts_without_uploading_current():
    current, p = generated_neural()
    began = time.perf_counter()
    engine = PersistentLocalResponse(current, p)
    setup_s = time.perf_counter() - began
    zero = np.zeros((3, 7), np.float64)
    changed = zero.copy()
    changed[1, 0] = .2
    first = engine.evaluate(zero)
    middle = engine.evaluate(changed)
    last = engine.evaluate(zero)
    reference = simulate_cuda(current, p, zero)
    cpu = simulate_numpy(current, p, changed)
    for key in first:
        np.testing.assert_array_equal(first[key], last[key])
        np.testing.assert_array_equal(first[key], reference[key])
        np.testing.assert_allclose(middle[key], cpu[key], atol=1e-9, rtol=1e-8)
    assert engine.stats["current_uploads"] == 1
    assert engine.stats["parameter_uploads"] == 8
    assert engine.stats["weight_uploads"] == 3
    assert engine.stats["launch_requested"] == engine.stats["launch_synchronized"] == 3
    assert engine.stats["conditional_ticks_completed"] == 60
    assert engine.stats["generated_warmup_calls"] == 0
    print("E203_GENERATED_CUDA", {"persistent_evaluations": 3,
        "E202_reference_evaluations": 1, "neural_ticks": 80,
        "stages": 320, "warmup_ticks": 0, "models": 0,
        "device_setup_wall_s": setup_s,
        "kernel_main_wall_s": engine.stats["kernel_main_s"]})


def test_ema_and_decoder_adjoint_finite_difference():
    initial = np.array([.1, .3])
    values = np.array([[1., 2.], [3., 4.], [2., 1.]])
    alpha = .2
    y = _ema(values, alpha, initial)
    manual = np.empty_like(y)
    manual[0] = initial
    for t in range(1, 4):
        manual[t] = (1 - alpha) * manual[t - 1] + alpha * values[t - 1]
    np.testing.assert_allclose(y, manual, atol=1e-15)
    g = np.arange(8, dtype=np.float64).reshape(4, 2) * .1
    adj = _ema_adjoint(g, alpha)
    eps = 1e-6
    changed = values.copy()
    changed[1, 0] += eps
    numeric = (np.sum(_ema(changed, alpha, initial) * g) - np.sum(y * g)) / eps
    np.testing.assert_allclose(numeric, adj[2, 0], atol=1e-8)

    decoder = DecoderAdjoint(generated_snapshot())
    rates = np.full((21, 391), .9)
    rates[1:] += np.arange(20)[:, None] * .002
    decoded = decoder.forward(rates)
    assert decoded["bank"].shape == (21, 391, 3)
    np.testing.assert_array_equal(decoded["command"][0], 7.)
    gq = np.zeros((21, 42))
    gq[1:, :7] = .01
    grate = decoder.backward(gq, decoded)
    assert grate.shape == (21, 391)
    np.testing.assert_array_equal(grate[0], 0.)
    for t, m in ((1, 0), (2, 4), (5, 15), (8, 30),
                 (11, 72), (13, 163), (17, 255), (20, 390)):
        plus, minus = rates.copy(), rates.copy()
        plus[t, m] += eps
        minus[t, m] -= eps
        numeric = (np.sum(decoder.forward(plus)["command"] * gq)
                   - np.sum(decoder.forward(minus)["command"] * gq)) / (2 * eps)
        np.testing.assert_allclose(numeric, grate[t, m], atol=2e-5, rtol=2e-4)


def test_objective_gradient_and_selection_contract():
    decoder = DecoderAdjoint(generated_snapshot())
    rates = np.full((2001, 391), .9)
    decoded = decoder.forward(rates)
    indices = np.arange(100, 1924)
    target = {"indices": indices, "pair_start": indices[:-1],
        "pair_end": indices[1:],
        "target": decoded["command"][indices] + .001,
        "velocity": np.zeros((1823, 42)),
        "startup_command": decoded["command"][1:100] + .0002,
        "startup_tendon": decoded["tendon"][1:100].copy(),
        "baseline": {"command_rms": .01, "velocity_rms": .01,
            "leg_command_rms": [.01] * 6,
            "leg_velocity_rms": [.01] * 6}}
    free = np.zeros(2401)
    merit, gq, terms = objective_and_command_gradient(decoded, target, free)
    assert np.isfinite(merit) and len(terms) == 18
    metric = direct_metrics(decoded, target)
    assert feasible(metric, target["baseline"])
    eps = 1e-6
    plus, minus = rates.copy(), rates.copy()
    plus[120, 0] += eps
    minus[120, 0] -= eps
    a = objective_and_command_gradient(decoder.forward(plus), target, free)[0]
    b = objective_and_command_gradient(decoder.forward(minus), target, free)[0]
    numeric = (a - b) / (2 * eps)
    analytic = decoder.backward(gq, decoded)[120, 0]
    np.testing.assert_allclose(numeric, analytic, atol=2e-5, rtol=2e-4)


def test_active_guard_and_startup_sample_command_gradient():
    q = np.zeros((2001, 42), np.float64)
    indices = np.arange(100, 1924)
    target = {"indices": indices, "pair_start": indices[:-1],
        "pair_end": indices[1:], "target": np.full((1824, 42), .01),
        "velocity": np.full((1823, 42), .01),
        "startup_command": np.full((99, 42), .05),
        "startup_tendon": np.zeros((99, 6)),
        "baseline": {"command_rms": .005, "velocity_rms": .005,
            "leg_command_rms": [.005] * 6,
            "leg_velocity_rms": [.005] * 6}}
    free = np.zeros(2401)
    merit, gq, terms = objective_and_command_gradient(
        {"command": q}, target, free)
    assert np.isfinite(merit)
    assert terms["overall_velocity"]["penalty"] > 0
    assert terms["lf_command"]["penalty"] > 0
    assert terms["startup_rms"]["penalty"] > 0
    assert terms["startup_sample_penalty"] > 0
    eps = 1e-7
    for tick, joint in ((1, 0), (100, 0), (101, 8), (1923, 41)):
        plus, minus = q.copy(), q.copy()
        plus[tick, joint] += eps
        minus[tick, joint] -= eps
        a = objective_and_command_gradient({"command": plus}, target, free)[0]
        b = objective_and_command_gradient({"command": minus}, target, free)[0]
        np.testing.assert_allclose((a - b) / (2 * eps), gq[tick, joint],
                                   atol=2e-5, rtol=2e-4)


def test_e202_local_tangent_contracts_through_decoder():
    current, p = generated_neural(motors=391, steps=20)
    decoder = DecoderAdjoint(generated_snapshot())
    weights = np.zeros((391, 7))
    weights[0, 0] = .1
    result = simulate_numpy(current, p, weights, sample_every=1)
    decoded = decoder.forward(result["rate"])
    gq = np.zeros_like(decoded["command"])
    gq[1:, 0] = .01
    grate = decoder.backward(gq, decoded)
    analytic = np.einsum("tm,tmk->mk", grate,
                         result["rate_sensitivity"], optimize=True)[0, 0]
    eps = 1e-5
    plus, minus = weights.copy(), weights.copy()
    plus[0, 0] += eps
    minus[0, 0] -= eps
    a = decoder.forward(simulate_numpy(current, p, plus, sample_every=1)["rate"])
    b = decoder.forward(simulate_numpy(current, p, minus, sample_every=1)["rate"])
    numeric = np.sum((a["command"] - b["command"]) * gq) / (2 * eps)
    np.testing.assert_allclose(numeric, analytic, atol=2e-5, rtol=2e-4)


def test_runner_exact_cache_hard_cap_and_raw_preservation(tmp_path, monkeypatch):
    names = ("current", "parameters", "endpoints", "observation", "trace",
        "native", "start", "e201_manifest", "e201_result", "e201_review",
        "e202_response", "e202_decoded", "e202_result", "e202_final_review",
        "e202_manifest", "targets", "target_freeze", "mask")
    monkeypatch.setattr(fit, "verify_gate", lambda *args: (
        {"paths": {name: name for name in names}, "input_sha256": {},
         "prior_cumulative_s": 1.}, {"source_sha256": {}}))
    monkeypatch.setattr(fit, "repository_path", lambda name, **kwargs: tmp_path / name)
    start = generated_snapshot()
    # Assign the 48 fixed columns to six distinct frozen LTM muscles.
    membership = np.asarray(start["adapter"]["parameters"]["membership"]).copy()
    ltm = np.asarray(start["adapter"]["calibration"]["ltm"])
    membership[:] = 0
    membership[:48] = np.resize(ltm, 48)
    fixed = np.flatnonzero(np.isin(membership, ltm))
    free = np.setdiff1d(np.arange(391), fixed)
    assert len(fixed) == 48
    start["adapter"]["parameters"]["membership"] = membership
    roster = [f"generated:{i}" for i in range(391)]
    start["monotone_recruitment_patch"] = {"motor_roster": roster}
    with (tmp_path / "mask").open("wb") as stream:
        np.savez_compressed(stream, free_columns=free,
            fixed_columns=fixed, motor_ids=np.arange(391),
            membership=membership, ltm=ltm,
            trainable_mask=np.isin(np.arange(391), free),
            motor_roster=np.asarray(roster))
    rate = np.ones((21, 391), np.float64)
    response = {"rate": rate, "adaptation": np.zeros_like(rate),
        "rate_sensitivity": np.zeros((21, 391, 7)),
        "adaptation_sensitivity": np.zeros((21, 391, 7))}
    decoded = DecoderAdjoint(start).forward(rate)
    with (tmp_path / "e202_response").open("wb") as stream:
        np.savez_compressed(stream, **response)
    with (tmp_path / "e202_decoded").open("wb") as stream:
        np.savez_compressed(stream, **decoded)
    parameter = {"motor_ids": np.arange(391), "membership": membership,
        **{key: np.ones(391, np.float32) for key in (
            "initial_rate", "initial_adaptation", "tau", "threshold", "cap",
            "coefficient", "adaptation_gain", "inverse_adaptation_tau")}}
    monkeypatch.setattr(fit, "load_original", lambda paths: (
        np.zeros((80, 391), np.float32), parameter, {}, start, {}, {}))
    monkeypatch.setattr(fit, "load_targets", lambda *args: {"baseline": {"command_rms": 1.}})
    monkeypatch.setattr(fit, "objective_and_command_gradient", lambda d, t, w: (
        float(np.sum(w * w)), np.zeros_like(d["command"]), {"total": float(np.sum(w * w))}))
    monkeypatch.setattr(fit, "direct_metrics", lambda d, t: {"command_rms": 1.})
    monkeypatch.setattr(fit, "feasible", lambda metrics, baseline: False)

    class FakeDevice:
        def __init__(self, current, parameters):
            self.stats = {"launch_synchronized": 0, "launch_requested": 0,
                          "conditional_ticks_completed": 0}

        def evaluate(self, weights):
            self.stats["launch_requested"] += 1
            self.stats["launch_synchronized"] += 1
            self.stats["conditional_ticks_completed"] += 20
            return {key: value.copy() for key, value in response.items()}

    monkeypatch.setattr(fit, "PersistentLocalResponse", FakeDevice)

    def cached_optimizer(fun, zero, **kwargs):
        fun(zero)  # Must reuse preflight without a second CUDA evaluation.
        changed = zero.copy()
        changed[0] = .1
        fun(changed)
        return SimpleNamespace(success=True, status=0, message="generated",
                               nit=1, nfev=2)

    monkeypatch.setattr(fit, "minimize", cached_optimizer)
    out = tmp_path / "cache-output"
    fit.run("unused", "unused", "unused", "unused", out)
    ledger = __import__("json").loads((out / "ledger.json").read_text())
    assert ledger["cuda_evaluations"] == 2
    assert ledger["cache_hits"] == 1
    assert (out / "evaluation-000.npz").is_file()
    assert (out / "evaluation-001.npz").is_file()
    assert not (out / "evaluation-002.npz").exists()

    def failing_optimizer(fun, zero, **kwargs):
        raise RuntimeError("generated optimizer failure")

    monkeypatch.setattr(fit, "minimize", failing_optimizer)
    failed = tmp_path / "failed-output"
    with pytest.raises(RuntimeError, match="generated optimizer failure"):
        fit.run("unused", "unused", "unused", "unused", failed)
    assert (failed / "evaluation-000.npz").is_file()
    assert (failed / "evaluation-000.json").is_file()
    assert (failed / "result.json").is_file()
    result = __import__("json").loads((failed / "result.json").read_text())
    assert result["validity"] == "incomplete_or_invalid"
    assert (failed / "selected-state.npz").is_file()
    assert (failed / "selected-decoder.npz").is_file()
    failed_selection = __import__("json").loads((failed / "selection.json").read_text())
    assert failed_selection["status"] == "unverified_incomplete_raw"

    def over_cap_optimizer(fun, zero, **kwargs):
        for j in range(1, 61):
            changed = zero.copy()
            changed[0] = j / 100
            fun(changed)
        raise AssertionError("Hard CUDA cap was bypassed")

    monkeypatch.setattr(fit, "minimize", over_cap_optimizer)
    capped = tmp_path / "capped-output"
    fit.run("unused", "unused", "unused", "unused", capped)
    capped_ledger = __import__("json").loads((capped / "ledger.json").read_text())
    assert capped_ledger["cuda_evaluations"] == 60
    assert capped_ledger["evaluation_count"] == 60
    assert (capped / "evaluation-059.json").is_file()
    assert not (capped / "evaluation-060.json").exists()

    for broken_key in ("adaptation", "adaptation_sensitivity"):
        class BadDevice(FakeDevice):
            broken_field = broken_key

            def evaluate(self, weights):
                actual = super().evaluate(weights)
                if self.stats["launch_synchronized"] == 2:
                    actual[self.broken_field].flat[0] = np.nan
                return actual

        monkeypatch.setattr(fit, "PersistentLocalResponse", BadDevice)
        monkeypatch.setattr(fit, "minimize", cached_optimizer)
        invalid = tmp_path / f"invalid-{broken_key}"
        with pytest.raises(ValueError, match="invalid float64 finite response"):
            fit.run("unused", "unused", "unused", "unused", invalid)
        assert (invalid / "invalid-response-001.npz").is_file()
        with np.load(invalid / "invalid-response-001.npz") as raw:
            assert np.isnan(raw[broken_key].flat[0])
        assert (invalid / "selected-state.npz").is_file()
        assert (invalid / "selected-decoder.npz").is_file()
        invalid_result = __import__("json").loads((invalid / "result.json").read_text())
        assert invalid_result["validity"] == "incomplete_or_invalid"
        invalid_selection = __import__("json").loads((invalid / "selection.json").read_text())
        assert invalid_selection["status"] == "unverified_incomplete_raw"

    with (tmp_path / "e202_response").open("wb") as stream:
        divergent = {key: value.copy() for key, value in response.items()}
        divergent["rate"][0, 0] += 1
        np.savez_compressed(stream, **divergent)
    monkeypatch.setattr(fit, "PersistentLocalResponse", FakeDevice)
    fidelity = tmp_path / "fidelity-output"
    with pytest.raises(ValueError, match="bit-exact E202 response"):
        fit.run("unused", "unused", "unused", "unused", fidelity)
    assert (fidelity / "selected-state.npz").is_file()
    assert (fidelity / "selected-decoder.npz").is_file()
    fidelity_result = __import__("json").loads((fidelity / "result.json").read_text())
    assert fidelity_result["validity"] == "incomplete_or_invalid"
    fidelity_selection = __import__("json").loads((fidelity / "selection.json").read_text())
    assert fidelity_selection["status"] == "unverified_incomplete_raw"
    assert fidelity_selection["w0_reconstruction_verified"] is False


def test_all_response_fields_have_finite_float64_expected_shapes():
    rows, motors = 3, 2
    raw = {"rate": np.zeros((rows, motors), np.float64),
        "adaptation": np.zeros((rows, motors), np.float64),
        "rate_sensitivity": np.zeros((rows, motors, 7), np.float64),
        "adaptation_sensitivity": np.zeros((rows, motors, 7), np.float64)}
    fit.validate_response(raw, rows, motors)
    for key in raw:
        malformed = {name: value.copy() for name, value in raw.items()}
        malformed[key].flat[0] = np.nan
        with pytest.raises(ValueError, match=key):
            fit.validate_response(malformed, rows, motors)
        malformed[key] = malformed[key].astype(np.float32)
        with pytest.raises(ValueError, match=key):
            fit.validate_response(malformed, rows, motors)
        malformed[key] = malformed[key][:-1]
        with pytest.raises(ValueError, match=key):
            fit.validate_response(malformed, rows, motors)
