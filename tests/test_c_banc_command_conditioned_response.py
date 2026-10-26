"""Generated-only E205 command tangent, actual-limit loss, and runner checks."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from tools import fit_c_banc_m3_command_conditioned_response as fit
from tools.fit_c_banc_m3_local_response import DecoderAdjoint


def generated_snapshot():
    membership = np.arange(391) % 96
    counts = np.bincount(membership, minlength=96)
    pool = np.zeros((96, 391))
    pool[membership, np.arange(391)] = 1 / counts[membership]
    slopes = np.zeros((96, 391, 3, 2))
    slopes[membership, np.arange(391), :, 0] = .5 / counts[membership, None]
    slopes[membership, np.arange(391), :, 1] = 1.5 / counts[membership, None]
    ltm = np.array([15, 30, 46, 63, 78, 94])
    groups = [{"leg": "lf", "muscle": str(i)} for i in range(96)]
    for muscle, leg in zip(ltm, ("lf", "lm", "lh", "rf", "rm", "rh")):
        groups[int(muscle)]["leg"] = leg
    return {"adapter": {"calibration": {"pool": pool,
        "moment": np.arange(42 * 96).reshape(42, 96) * 1e-5,
        "rest": np.zeros(42), "ltm": ltm},
        "parameters": {"membership": membership,
        "kappa": np.full(391, .13), "slopes": slopes,
        "fixed_muscles": np.array([4, 15, 16, 19, 30, 31, 34, 35, 46,
            47, 52, 63, 64, 67, 78, 82, 83, 94, 95]),
        "tau": np.array([.005, .01, .04])},
        "bank": np.zeros((391, 3)), "activation": np.zeros(96),
        "original": {"activation": np.zeros(96)}, "groups": groups},
        "body": {"action": {"joint_angles": np.full(42, 7.)}}}


def test_command_tangent_diagonal_and_power_of_two_scale():
    start = generated_snapshot()
    decoder = DecoderAdjoint(start)
    rates = np.full((2001, 391), .9)
    decoded = decoder.forward(rates)
    sensitivity = np.zeros((2001, 391, 7))
    for motor in (0, 1, 4, 15, 30, 46, 63, 78, 94, 96):
        sensitivity[1:, motor, 0] = np.linspace(.1, .2, 2000)
    target = {"indices": np.arange(100, 1924),
        "baseline": {"command_rms": .05}}
    free = np.arange(343)
    result = fit.command_diagonal_scale(start,
        {"rate_sensitivity": sensitivity}, decoded, target, free)
    d, h, scale, exponent = (result[key] for key in ("D", "h", "scale", "exponent"))
    assert d.shape == (2001, 391, 7)
    assert h.shape == scale.shape == exponent.shape == (343, 7)
    np.testing.assert_array_equal(d[0], 0)
    assert np.all(h > 0) and np.all(exponent <= 0) and np.all(exponent >= -30)
    np.testing.assert_array_equal(scale, np.exp2(exponent))
    for motor in (0, 1, 4, 15, 30, 46, 96):
        perturbed = rates.copy()
        eps = 1e-5
        perturbed[:, motor] += eps * sensitivity[:, motor, 0]
        changed = decoder.forward(perturbed)
        group = start["adapter"]["parameters"]["membership"][motor]
        numeric = (changed["activation"][:, group] - decoded["activation"][:, group]) / eps
        np.testing.assert_allclose(numeric, d[:, motor, 0], atol=2e-5, rtol=2e-4)
        coefficient = np.sum((.02 * decoder.moment[:, group])**2)
        expected = (2 * coefficient * np.sum(d[target["indices"], motor, 0]**2)
                    / (1824 * 42 * (.8 * .05)**2) + 2e-4 / 2401)
        np.testing.assert_allclose(h[motor, 0], expected, atol=1e-15)
    u = np.array([.2, -.3, 1.])
    s = scale.reshape(-1)[:3]
    w = s * u
    np.testing.assert_array_equal((w / s) * s, w)
    np.testing.assert_array_equal((1 / s) * s, np.ones(3))
    grad_w = np.array([1.2, -.8, .4])
    grad_u = s * grad_w
    for k in range(3):
        eps = 1e-4
        plus, minus = u.copy(), u.copy()
        plus[k] += eps
        minus[k] -= eps
        loss = lambda v: np.sum(grad_w * (s * v))
        np.testing.assert_allclose((loss(plus) - loss(minus)) / (2 * eps),
                                   grad_u[k], atol=2e-5, rtol=2e-4)


def test_actual_limit_objective_active_guard_gradient():
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
    merit, gq, terms = fit.actual_limit_objective({"command": q}, target, free)
    assert np.isfinite(merit)
    assert terms["overall_velocity"]["penalty"] > 0
    assert terms["startup_sample_penalty"] > 0
    assert terms["startup_rms"]["limit"] == .018
    eps = 1e-7
    for tick, joint in ((1, 0), (100, 0), (101, 8), (1923, 41)):
        plus, minus = q.copy(), q.copy()
        plus[tick, joint] += eps
        minus[tick, joint] -= eps
        a = fit.actual_limit_objective({"command": plus}, target, free)[0]
        b = fit.actual_limit_objective({"command": minus}, target, free)[0]
        np.testing.assert_allclose((a - b) / (2 * eps), gq[tick, joint],
                                   atol=2e-5, rtol=2e-4)


def test_runner_conditioned_cache_cap_mismatch_and_failure_raw(tmp_path, monkeypatch):
    names = ("current", "parameters", "endpoints", "observation", "trace",
        "native", "start", "e201_manifest", "e201_result", "e201_review",
        "e202_response", "e202_decoded", "e202_result", "e202_final_review",
        "e202_manifest", "targets", "target_freeze", "mask",
        "e203_initial_raw", "e203_initial_note", "e203_result", "e203_ledger",
        "e203_manifest", "e203_final_review", "e203_conditioning",
        "e203_source_review", "e204_initial_raw", "e204_initial_note",
        "e204_response", "e204_decoded", "e204_result", "e204_ledger",
        "e204_manifest", "e204_final_review", "e204_direction",
        "e204_source_review")
    monkeypatch.setattr(fit, "verify_gate", lambda *args: (
        {"paths": {name: name for name in names}, "input_sha256": {},
         "prior_cumulative_s": 1.}, {"source_sha256": {}}))
    monkeypatch.setattr(fit, "repository_path", lambda name, **kwargs: tmp_path / name)
    start = generated_snapshot()
    membership = np.asarray(start["adapter"]["parameters"]["membership"]).copy()
    ltm = np.asarray(start["adapter"]["calibration"]["ltm"])
    membership[:] = 0
    membership[:48] = np.resize(ltm, 48)
    fixed = np.flatnonzero(np.isin(membership, ltm))
    free = np.setdiff1d(np.arange(391), fixed)
    start["adapter"]["parameters"]["membership"] = membership
    roster = [f"generated:{i}" for i in range(391)]
    start["monotone_recruitment_patch"] = {"motor_roster": roster}
    with (tmp_path / "mask").open("wb") as stream:
        np.savez_compressed(stream, free_columns=free, fixed_columns=fixed,
            motor_ids=np.arange(391), membership=membership, ltm=ltm,
            trainable_mask=np.isin(np.arange(391), free),
            motor_roster=np.asarray(roster))
    rate = np.ones((21, 391), np.float64)
    response = {"rate": rate, "adaptation": np.zeros_like(rate),
        "rate_sensitivity": np.zeros((21, 391, 7)),
        "adaptation_sensitivity": np.zeros((21, 391, 7))}
    response["rate_sensitivity"][1, free[0], 0] = 1.
    decoded = {"bank": np.zeros((21, 391, 3)),
        "activation": np.zeros((21, 96)), "command": np.zeros((21, 42)),
        "tendon": np.zeros((21, 6))}
    initial_weights = np.zeros((391, 7))
    initial_weights[free[0], 0] = .25
    initial_free = initial_weights[free].reshape(-1)
    with (tmp_path / "e204_response").open("wb") as stream:
        np.savez_compressed(stream, **response, weights=initial_weights)
    with (tmp_path / "e204_decoded").open("wb") as stream:
        np.savez_compressed(stream, **decoded)
    gradient0 = np.zeros(2401)
    gradient0[0] = 1 + 2e-4 * .25 / 2401
    with (tmp_path / "e204_initial_raw").open("wb") as stream:
        np.savez_compressed(stream, weights=initial_weights,
                            free_weights=initial_free, gradient=gradient0)
    (tmp_path / "e204_initial_note").write_text(json.dumps({"merit": .25**2}))
    parameter = {"motor_ids": np.arange(391), "membership": membership,
        **{key: np.ones(391, np.float32) for key in (
            "initial_rate", "initial_adaptation", "tau", "threshold", "cap",
            "coefficient", "adaptation_gain", "inverse_adaptation_tau")}}
    monkeypatch.setattr(fit, "load_original", lambda paths: (
        np.zeros((80, 391), np.float32), parameter, {}, start, {}, {}))
    monkeypatch.setattr(fit, "load_targets", lambda *args: {"baseline": {"command_rms": 1.}})

    class FakeDecoder:
        def __init__(self, source):
            pass

        def forward(self, rates):
            return {key: value.copy() for key, value in decoded.items()}

        def backward(self, command_gradient, output):
            result = np.zeros((21, 391))
            result[1, free[0]] = 1.
            return result

    monkeypatch.setattr(fit, "DecoderAdjoint", FakeDecoder)
    monkeypatch.setattr(fit, "objective_and_command_gradient", lambda d, t, w: (
        float(np.sum(w * w)), np.zeros((21, 42)), {"total": float(np.sum(w * w))}))
    monkeypatch.setattr(fit, "actual_limit_objective", lambda d, t, w: (
        1 + float(np.sum(w * w)), np.zeros((21, 42)),
        {"total": 1 + float(np.sum(w * w))}))
    monkeypatch.setattr(fit, "command_diagonal_scale", lambda *args: {
        "D": np.zeros((21, 391, 7)), "h": np.ones((343, 7)),
        "scale": np.full((343, 7), .5),
        "exponent": np.full((343, 7), -1), "free_columns": free})
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

    def cached(fun, initial, **kwargs):
        assert kwargs["options"]["maxls"] == 30
        assert kwargs["options"]["maxfun"] == 45
        assert kwargs["options"]["gtol"] == .5e-6
        assert kwargs["bounds"][0] == (-2., 2.)
        assert initial[0] == .5
        fun(initial)
        changed = initial.copy()
        changed[0] = .75
        fun(changed)
        return SimpleNamespace(success=True, status=0, message="generated", nit=1, nfev=2)

    monkeypatch.setattr(fit, "minimize", cached)
    output = tmp_path / "success"
    fit.run("unused", "unused", "unused", "unused", output)
    ledger = json.loads((output / "ledger.json").read_text())
    assert ledger["cuda_evaluations"] == 2 and ledger["cache_hits"] == 1
    assert ledger["initial_reconstruction_verified"] is True
    assert (output / "preconditioner.npz").is_file()
    with np.load(output / "evaluation-001.npz") as values:
        assert values["scaled_variables"][0] == .75
        assert values["free_weights"][0] == .375
        assert values["gradient"][0] == 1 + 2e-4 * .375 / 2401
        np.testing.assert_allclose(values["scaled_gradient"],
                                   .5 * values["gradient"], atol=0, rtol=0)
    with np.load(output / "selected-state.npz") as values:
        assert "scaled_variables" in values and "scale" in values

    def fail(fun, initial, **kwargs):
        raise RuntimeError("generated optimizer failure")

    monkeypatch.setattr(fit, "minimize", fail)
    failed = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="generated optimizer failure"):
        fit.run("unused", "unused", "unused", "unused", failed)
    assert (failed / "selected-state.npz").is_file()
    assert json.loads((failed / "result.json").read_text())["validity"] == "incomplete_or_invalid"

    def over_cap(fun, initial, **kwargs):
        for j in range(1, 46):
            changed = initial.copy()
            changed[0] = .5 + j / 100
            fun(changed)
        raise AssertionError("E205 cap bypassed")

    monkeypatch.setattr(fit, "minimize", over_cap)
    capped = tmp_path / "capped"
    fit.run("unused", "unused", "unused", "unused", capped)
    assert json.loads((capped / "ledger.json").read_text())["cuda_evaluations"] == 45
    assert (capped / "evaluation-044.npz").is_file()
    assert not (capped / "evaluation-045.npz").exists()

    class BadDevice(FakeDevice):
        def evaluate(self, weights):
            actual = super().evaluate(weights)
            if self.stats["launch_synchronized"] == 2:
                actual["adaptation_sensitivity"].flat[0] = np.nan
            return actual

    monkeypatch.setattr(fit, "PersistentLocalResponse", BadDevice)
    monkeypatch.setattr(fit, "minimize", cached)
    invalid = tmp_path / "invalid"
    with pytest.raises(ValueError, match="adaptation_sensitivity"):
        fit.run("unused", "unused", "unused", "unused", invalid)
    assert (invalid / "invalid-response-001.npz").is_file()
    assert (invalid / "selected-state.npz").is_file()
    assert json.loads((invalid / "result.json").read_text())["validity"] == "incomplete_or_invalid"

    with (tmp_path / "e204_response").open("wb") as stream:
        divergent = {key: value.copy() for key, value in response.items()}
        divergent["rate"][0, 0] += 1
        np.savez_compressed(stream, **divergent, weights=initial_weights)
    monkeypatch.setattr(fit, "PersistentLocalResponse", FakeDevice)
    mismatch = tmp_path / "mismatch"
    with pytest.raises(ValueError, match="bit-exact E204 response"):
        fit.run("unused", "unused", "unused", "unused", mismatch)
    assert (mismatch / "selected-state.npz").is_file()
    assert json.loads((mismatch / "selection.json").read_text())["status"] == (
        "unverified_incomplete_raw")
