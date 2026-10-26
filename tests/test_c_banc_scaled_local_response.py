"""Generated-only E204 coordinate, decoder-chain, and bounded-run checks."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from tools import fit_c_banc_m3_scaled_local_response as fit
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


def test_scaled_quadratic_chain_and_exact_bounds():
    assert fit.BOUND == 3209.6762087880074
    u = np.array([.2, -2., fit.BOUND])
    w = fit.scaled_to_native(u)
    np.testing.assert_allclose(w, fit.SCALE * u, atol=0, rtol=0)
    assert w[-1] == 1.
    target = np.array([.1, -.4, .3])
    gw = 2 * (w - target) + .2 * w
    gu = fit.native_to_scaled_gradient(gw)
    eps = 1e-4
    for k in range(3):
        plus, minus = u.copy(), u.copy()
        plus[k] += eps
        minus[k] -= eps
        def loss(v):
            x = fit.scaled_to_native(v)
            return np.sum((x - target)**2 + .1 * x**2)
        np.testing.assert_allclose((loss(plus) - loss(minus)) / (2 * eps),
                                   gu[k], atol=2e-5, rtol=2e-4)


def test_scaled_decoder_chain_finite_difference():
    decoder = DecoderAdjoint(generated_snapshot())
    base = np.full((21, 391), .9)
    tangent = np.zeros_like(base)
    tangent[1:, 0] = np.linspace(.2, .4, 20)
    tangent[1:, 48] = np.linspace(.1, .2, 20)
    u = np.array([.1, -.3])
    rates = base + fit.SCALE * (u[0] * tangent + u[1] * np.roll(tangent, 1, axis=0))
    decoded = decoder.forward(rates)
    gq = np.zeros_like(decoded["command"])
    gq[1:, :7] = .01
    gr = decoder.backward(gq, decoded)
    gw = np.array([np.sum(gr * tangent), np.sum(gr * np.roll(tangent, 1, axis=0))])
    gu = fit.native_to_scaled_gradient(gw)
    eps = 1e-3
    def loss(v):
        r = base + fit.SCALE * (v[0] * tangent + v[1] * np.roll(tangent, 1, axis=0))
        return np.sum(decoder.forward(r)["command"] * gq)
    for k in range(2):
        plus, minus = u.copy(), u.copy()
        plus[k] += eps
        minus[k] -= eps
        np.testing.assert_allclose((loss(plus) - loss(minus)) / (2 * eps),
                                   gu[k], atol=2e-5, rtol=2e-4)


def test_runner_scaled_cache_cap_and_failure_raw(tmp_path, monkeypatch):
    names = ("current", "parameters", "endpoints", "observation", "trace",
        "native", "start", "e201_manifest", "e201_result", "e201_review",
        "e202_response", "e202_decoded", "e202_result", "e202_final_review",
        "e202_manifest", "targets", "target_freeze", "mask",
        "e203_initial_raw", "e203_initial_note", "e203_result", "e203_ledger",
        "e203_manifest", "e203_final_review", "e203_conditioning",
        "e203_source_review")
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
    with (tmp_path / "e202_response").open("wb") as stream:
        np.savez_compressed(stream, **response)
    with (tmp_path / "e202_decoded").open("wb") as stream:
        np.savez_compressed(stream, **decoded)
    gradient0 = np.zeros(2401)
    gradient0[0] = fit.BOUND
    with (tmp_path / "e203_initial_raw").open("wb") as stream:
        np.savez_compressed(stream, free_weights=np.zeros(2401), gradient=gradient0)
    (tmp_path / "e203_initial_note").write_text(json.dumps({"merit": 0.}))
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
            result[1, free[0]] = fit.BOUND
            return result

    monkeypatch.setattr(fit, "DecoderAdjoint", FakeDecoder)
    monkeypatch.setattr(fit, "objective_and_command_gradient", lambda d, t, w: (
        float(np.sum(w * w)), np.zeros((21, 42)), {"total": float(np.sum(w * w))}))
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

    def cached(fun, zero, **kwargs):
        assert kwargs["options"]["maxls"] == 30
        assert kwargs["options"]["gtol"] == fit.SCALE * 1e-6
        assert kwargs["bounds"][0] == (-fit.BOUND, fit.BOUND)
        fun(zero)
        changed = zero.copy()
        changed[0] = 1.
        fun(changed)
        return SimpleNamespace(success=True, status=0, message="generated", nit=1, nfev=2)

    monkeypatch.setattr(fit, "minimize", cached)
    output = tmp_path / "success"
    fit.run("unused", "unused", "unused", "unused", output)
    ledger = json.loads((output / "ledger.json").read_text())
    assert ledger["cuda_evaluations"] == 2 and ledger["cache_hits"] == 1
    with np.load(output / "evaluation-001.npz") as values:
        assert values["scaled_variables"][0] == 1.
        assert values["free_weights"][0] == fit.SCALE
        assert values["gradient"][0] == fit.BOUND + 2e-4 * fit.SCALE / 2401
        np.testing.assert_allclose(values["scaled_gradient"],
                                   fit.SCALE * values["gradient"], atol=0, rtol=0)
    with np.load(output / "selected-state.npz") as values:
        assert "scaled_variables" in values

    def fail(fun, zero, **kwargs):
        raise RuntimeError("generated optimizer failure")

    monkeypatch.setattr(fit, "minimize", fail)
    failed = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="generated optimizer failure"):
        fit.run("unused", "unused", "unused", "unused", failed)
    assert (failed / "selected-state.npz").is_file()
    assert json.loads((failed / "result.json").read_text())["validity"] == "incomplete_or_invalid"

    def over_cap(fun, zero, **kwargs):
        for j in range(1, 61):
            changed = zero.copy()
            changed[0] = j / 100
            fun(changed)
        raise AssertionError("E204 cap bypassed")

    monkeypatch.setattr(fit, "minimize", over_cap)
    capped = tmp_path / "capped"
    fit.run("unused", "unused", "unused", "unused", capped)
    assert json.loads((capped / "ledger.json").read_text())["cuda_evaluations"] == 60
    assert (capped / "evaluation-059.npz").is_file()
    assert not (capped / "evaluation-060.npz").exists()

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
