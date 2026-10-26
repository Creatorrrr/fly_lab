"""Generated-only E170 Stage B checks; original NPZ reads are blocked."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import sparse

from tests import test_c_banc_closed_loop_recruitment as fixture
from tests import test_c_banc_onset_recruitment as onset_fixture
from tools import fit_c_banc_m3_history_recruitment as fit


@pytest.fixture(scope="module")
def generated():
    data, base, _ = fixture.generated(varying_bank=True)
    slopes = data["parameters"]["slopes"]
    rate = data["recruitment"]["motor_rate"]
    kappa = data["parameters"]["kappa"]
    _, history = fit.stage_a.history_state(rate, kappa)
    activation, command = fit.stage_a.evaluate_history(
        data["recruitment"]["bank"],
        data["recruitment"]["original_activation"],
        slopes,
        np.zeros_like(slopes),
        kappa,
        data["adapter"]["membership"],
        np.asarray(onset_fixture.FIXED, dtype=np.int64),
        data["adapter"]["moment"],
        data["adapter"]["rest"],
        history,
    )
    data["recruitment"]["activation"] = activation
    data["state"]["adapter"]["activation"] = activation[0].tolist()
    for row, value in zip(data["trace"]["samples"], command, strict=True):
        row["joint_targets_rad"] = value.tolist()
    data["startup"]["recorded_command"] = command
    data["startup"]["target_command"] = command[1:100] + 0.005
    base = onset_fixture.generated_full_freeze(data)
    freeze = json.loads(
        (
            fit.ROOT
            / "verification/banc-natural-gait-m3-20260919/e170-history-recruitment-freeze-v2.json"
        ).read_text(encoding="utf-8")
    )
    freeze["sample_contract"] = {**freeze["sample_contract"], **base["sample_contract"]}
    freeze["body_hash"] = base["body_hash"]
    return data, base, freeze


def prepare(generated):
    data, base, freeze = generated
    raw = {}
    fit.capture_loaded_input(data, raw)
    fit.validate_loaded_input(data, raw)
    checks = []
    saved = fit.e164.prepare_saved(data, base, raw, checks)
    fit.prepare_transfer(data, saved, freeze, raw, checks)
    fit.enforce(checks)
    return saved, raw


def forbid_original_npz(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("generated test attempted an original NPZ read")

    monkeypatch.setattr(np, "load", forbidden)


def test_generated_order_history_native_and_physical(generated, monkeypatch):
    forbid_original_npz(monkeypatch)
    data, _, freeze = generated
    saved, raw = prepare(generated)
    np.testing.assert_array_equal(
        raw["pair_motors"], np.repeat(freeze["sample_contract"]["variable_motors"], 3)
    )
    np.testing.assert_array_equal(raw["pair_filters"], np.tile(np.arange(3), 330))
    assert raw["history_F"].shape == (2001, 391)
    assert np.count_nonzero(raw["history_F"][0]) == 0
    baseline = raw["baseline_coeff"]
    assert np.count_nonzero(baseline[1980:]) == 0
    metrics, arrays = fit.candidate(saved, baseline)
    assert metrics == saved["baseline_metrics"]
    np.testing.assert_array_equal(arrays["candidate_activation"], saved["activation"])
    np.testing.assert_array_equal(
        arrays["candidate_command"][1:], saved["recorded_command"][1:]
    )
    native_a, native_q = fit.native_candidate(saved, baseline)
    np.testing.assert_array_equal(native_a, arrays["candidate_activation"])
    np.testing.assert_array_equal(native_q, arrays["candidate_command"])
    coeff = np.r_[baseline[:1980] * 0.99, np.full(1980, 1e-4)]
    _, varied = fit.candidate(saved, coeff)
    native_a, native_q = fit.native_candidate(saved, coeff)
    np.testing.assert_allclose(
        varied["candidate_activation"], native_a, atol=1e-12, rtol=0
    )
    np.testing.assert_allclose(
        varied["candidate_command"], native_q, atol=1e-12, rtol=0
    )
    assert all(fit.physical_gates(saved, coeff, freeze).values())
    bad = coeff.copy()
    bad[1980] = -1e-15
    assert not fit.physical_gates(saved, bad, freeze)["coefficient_exact_bounds"]
    endpoint = coeff.copy()
    endpoint[0], endpoint[1980] = 1.9, 0.2
    assert not fit.physical_gates(saved, endpoint, freeze)["coefficient_exact_bounds"]
    assert data["recruitment"]["bank"].shape == (2001, 391, 3)


def test_v2_tick_zero_is_preserved_and_tick_one_is_gated(generated, monkeypatch):
    forbid_original_npz(monkeypatch)
    data, base, freeze = generated
    raw = {}
    saved = fit.e164.prepare_saved(data, base, raw, [])
    saved["recorded_command"] = saved["recorded_command"].copy()
    saved["recorded_command"][0] += 0.326
    checks = []
    fit.prepare_transfer(data, saved, freeze, raw, checks)
    fit.enforce(checks)
    np.testing.assert_allclose(
        raw["baseline_tick_zero_command_discrepancy"], -0.326, atol=1e-12, rtol=0
    )
    saved["recorded_command"][1, 0] += 0.001
    checks = []
    fit.prepare_transfer(data, saved, freeze, {}, checks)
    with pytest.raises(ValueError, match="baseline differs"):
        fit.enforce(checks)


def test_resource_shape_is_captured_before_recurrence(generated, monkeypatch):
    forbid_original_npz(monkeypatch)
    data, base, freeze = generated
    saved = fit.e164.prepare_saved(data, base, {}, [])
    resource = {
        "resource": np.ones((1, 391)),
        "endpoint_rate": data["recruitment"]["motor_rate"],
        "bank": data["recruitment"]["bank"],
        "kappa": data["parameters"]["kappa"],
        "membership": data["adapter"]["membership"],
        "slopes": data["parameters"]["slopes"],
        "loaded_recruitment_control_tick": data["recruitment"]["control_tick"],
    }
    raw, progress = {}, {"resource_state_update_steps": 0}
    with pytest.raises(ValueError, match="resource shape/dtype"):
        fit.prepare_transfer(
            data, saved, freeze, raw, [], saved_resource=resource, progress=progress
        )
    assert raw["saved_resource_resource"].shape == (1, 391)
    assert progress["resource_state_update_steps"] == 0


@pytest.fixture(scope="module")
def problem(generated):
    saved, raw = prepare(generated)
    checks = []
    value = fit.build_problem(saved, generated[2], raw, checks)
    fit.enforce(checks)
    return saved, raw, value


def test_generated_cones_witness_and_true_physical_dual(
    generated, problem, monkeypatch
):
    forbid_original_npz(monkeypatch)
    saved, raw, conic = problem
    assert conic.A.shape[1] == 5941
    assert conic.cone_dims[0] == 24642
    assert len(conic.criteria) == 15
    assert (
        len(
            [
                key
                for key in raw
                if key.startswith("qr_") and key.endswith("_pointwise_residual")
            ]
        )
        == 18
    )
    assert np.min(raw["inner_witness_slack"][:24642]) >= -1e-12
    assert all(
        fit.physical_gates(saved, saved["baseline_coeff"], generated[2]).values()
    )
    x = raw["inner_witness"].copy()
    x[3960] += 0.01
    assert np.min((conic.b - conic.A @ x)[11882:15842]) < 0
    fake = SimpleNamespace(
        x=raw["inner_witness"],
        s=raw["inner_witness_slack"],
        z=np.zeros(conic.A.shape[0]),
        status="GeneratedFake",
        iterations=0,
        solve_time=0,
    )
    derived = fit.solver_derived(conic, fake, [], generated[2])
    changed = np.flatnonzero(derived["physical_budget_relaxed_b"] != conic.b)
    np.testing.assert_array_equal(
        changed, np.r_[np.arange(3960), np.arange(15842, 16326)]
    )
    np.testing.assert_array_equal(
        derived["physical_stationarity_lower"][:3960], np.zeros(3960)
    )
    assert np.isfinite(derived["relaxed_projected_margin_upper_bound"])


class FakeSolver:
    def __init__(self, P, q, A, b, cones, settings, x):
        self.A, self.b, self.settings, self.x = A, b, settings, x

    def set_termination_callback(self, callback):
        self.callback = callback

    def get_settings(self):
        return self.settings

    def solve(self):
        return SimpleNamespace(
            x=self.x,
            s=self.b - self.A @ self.x,
            z=np.zeros(len(self.b)),
            status="GeneratedFakeSolve",
            iterations=0,
            solve_time=0.0,
            obj_val=0.251,
            obj_val_dual=0.0,
            r_prim=0.0,
            r_dual=0.0,
        )


@pytest.mark.parametrize(
    ("target", "expected"),
    [
        ("solver-return-raw.npz", "solver_x"),
        ("candidate-raw.npz", "candidate_activation"),
    ],
)
def test_one_solve_first_raw_publication_failure_retains_arrays(
    tmp_path, generated, problem, monkeypatch, target, expected
):
    data, base, freeze = generated
    freeze = {
        **freeze,
        "resource_limits": {
            **freeze["resource_limits"],
            "free_bytes": 0,
            "internal_s": 120,
        },
    }
    resource_path = tmp_path / "generated-resource.npz"
    resource = {
        "resource": problem[0]["history_R"],
        "endpoint_rate": data["recruitment"]["motor_rate"],
        "bank": data["recruitment"]["bank"],
        "kappa": data["parameters"]["kappa"],
        "membership": data["adapter"]["membership"],
        "slopes": data["parameters"]["slopes"],
        "loaded_recruitment_control_tick": data["recruitment"]["control_tick"],
    }
    np.savez_compressed(resource_path, **resource)
    freeze["saved_resource_sha256"] = fit.sha256(resource_path)
    frozen_path = fit.frozen_path
    monkeypatch.setattr(
        fit,
        "frozen_path",
        lambda name: (
            resource_path if name == freeze["saved_resource"] else frozen_path(name)
        ),
    )
    monkeypatch.setattr(
        fit, "validate_freeze", lambda _path, _digest: (freeze, base, {})
    )
    cached_raw, cached_problem = problem[1], problem[2]

    def cached_build(_saved, _freeze, raw, _pending):
        raw.update(cached_raw)
        return cached_problem

    monkeypatch.setattr(fit, "build_problem", cached_build)
    gate = {
        "schema": fit.GATE_SCHEMA,
        "approved": True,
        "blocking_defects": [],
        "accepts_single_history_recruitment_fit": True,
        "original_processes_allowed": 1,
        "automatic_retries": 0,
        "freeze_sha256": fit.FREEZE_SHA,
        "design_sha256": fit.DESIGN_SHA,
        "source_sha256": {
            name: fit.sha256(fit.frozen_path(name)) for name in fit.OWNED
        },
    }
    gate_path = tmp_path / "source-review.json"
    gate_path.write_text(json.dumps(gate), encoding="utf-8")
    loaded = np.load

    def generated_only_load(path, *args, **kwargs):
        assert str(path).startswith(str(tmp_path)), "original NPZ read"
        return loaded(path, *args, **kwargs)

    monkeypatch.setattr(np, "load", generated_only_load)
    w0 = problem[0]["w0"]
    fake_x = np.r_[
        w0,
        np.full(1980, 1e-9),
        w0.reshape(990, 2).min(axis=1),
        (w0 + 1e-9).reshape(990, 2).max(axis=1),
        -0.251,
    ]

    def factory(P, q, A, b, cones, settings):
        return FakeSolver(P, q, A, b, cones, settings, fake_x)

    atomic = fit.atomic_npz
    attempts = []

    def fail_first(path, arrays):
        attempts.append(path.name)
        if path.name == target and attempts.count(target) == 1:
            raise OSError("generated first raw publication failure")
        return atomic(path, arrays)

    monkeypatch.setattr(fit, "atomic_npz", fail_first)
    out = tmp_path / "out"
    with pytest.raises(OSError, match="first raw publication failure"):
        fit.run(
            fit.ROOT
            / "verification/banc-natural-gait-m3-20260919/e170-history-recruitment-freeze-v2.json",
            fit.FREEZE_SHA,
            gate_path,
            fit.sha256(gate_path),
            out,
            input_loader=lambda _base: data,
            solver_factory=factory,
        )
    receipt = json.loads((out / "result.json").read_text(encoding="utf-8"))
    assert receipt["validity"] == "incomplete"
    assert receipt["counts"]["fit_calls"] == receipt["counts"]["solver_returns"] == 1
    assert attempts.count(target) == 1
    fallback = out / target.replace("-raw.npz", "-fallback.npz")
    assert fallback.exists()
    with np.load(fallback, allow_pickle=False) as saved:
        assert expected in saved


@pytest.mark.parametrize("failure_stage", ["native", "affine"])
def test_post_candidate_oracle_failure_retains_completed_candidate(
    tmp_path, generated, monkeypatch, failure_stage
):
    freeze = {
        **generated[2],
        "resource_limits": {**generated[2]["resource_limits"], "free_bytes": 0},
    }
    freeze["saved_resource_sha256"] = "generated-resource-marker"
    conic = SimpleNamespace(
        P=sparse.csc_matrix((5941, 5941)),
        q=np.r_[np.zeros(5940), -1.0],
        A=sparse.csc_matrix((1, 5941)),
        b=np.ones(1),
        cones=[fit.clarabel.NonnegativeConeT(1)],
    )

    class EmptyResource:
        files = ()

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    class TinySolver:
        def __init__(self, P, q, A, b, cones, settings):
            self.settings = settings

        def set_termination_callback(self, callback):
            self.callback = callback

        def get_settings(self):
            return self.settings

        def solve(self):
            return SimpleNamespace(
                x=np.zeros(5941),
                s=np.ones(1),
                z=np.zeros(1),
                status="GeneratedFake",
                iterations=0,
                solve_time=0.0,
                obj_val=0.0,
                obj_val_dual=0.0,
                r_prim=0.0,
                r_dual=0.0,
            )

    monkeypatch.setattr(fit, "validate_freeze", lambda *_args: (freeze, {}, {}))
    monkeypatch.setattr(fit, "validate_gate", lambda *_args: {})
    monkeypatch.setattr(fit, "snapshot", lambda *_args: {})
    monkeypatch.setattr(fit, "verify_snapshot", lambda *_args: None)
    actual_sha = fit.sha256
    resource_path = fit.frozen_path(freeze["saved_resource"])
    monkeypatch.setattr(
        fit,
        "sha256",
        lambda path: (
            "generated-resource-marker" if path == resource_path else actual_sha(path)
        ),
    )
    monkeypatch.setattr(fit, "capture_loaded_input", lambda *_args: None)
    monkeypatch.setattr(fit, "validate_loaded_input", lambda *_args: None)
    monkeypatch.setattr(
        fit.e164,
        "prepare_saved",
        lambda *_args: {"pair_start": np.array([1]), "pair_end": np.array([2])},
    )
    monkeypatch.setattr(fit, "prepare_transfer", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(fit, "build_problem", lambda *_args: conic)
    monkeypatch.setattr(
        fit, "solver_derived", lambda *_args: {"generated_derived": np.array([1.0])}
    )
    candidate_arrays = {
        "candidate_activation": np.full((3, 96), 0.17),
        "candidate_command": np.full((3, 42), 0.23),
        "candidate_velocity": np.zeros((1, 42)),
    }
    monkeypatch.setattr(fit, "candidate", lambda *_args: ({}, candidate_arrays))

    def fail(*_args):
        raise RuntimeError("generated post-candidate " + failure_stage + " failure")

    if failure_stage == "native":
        monkeypatch.setattr(fit, "native_candidate", fail)
    else:
        monkeypatch.setattr(
            fit,
            "native_candidate",
            lambda *_args: (np.full((3, 96), 0.17), np.full((3, 42), 0.23)),
        )
        monkeypatch.setattr(fit, "returned_affine_checks", fail)
    real_np_load = np.load
    monkeypatch.setattr(np, "load", lambda *_args, **_kwargs: EmptyResource())
    out = tmp_path / "out"
    with pytest.raises(RuntimeError, match="post-candidate"):
        fit.run(
            fit.ROOT
            / "verification/banc-natural-gait-m3-20260919/e170-history-recruitment-freeze-v2.json",
            fit.FREEZE_SHA,
            tmp_path / "gate.json",
            "generated",
            out,
            input_loader=lambda _base: {},
            solver_factory=TinySolver,
        )
    receipt = json.loads((out / "result.json").read_text(encoding="utf-8"))
    assert receipt["validity"] == "incomplete"
    assert receipt["counts"]["fit_calls"] == receipt["counts"]["solver_returns"] == 1
    assert receipt["primary_error"]["type"] == "RuntimeError"
    with real_np_load(out / "candidate-fallback.npz", allow_pickle=False) as raw:
        np.testing.assert_array_equal(
            raw["candidate_activation"], candidate_arrays["candidate_activation"]
        )
        np.testing.assert_array_equal(
            raw["candidate_command"], candidate_arrays["candidate_command"]
        )
        assert raw["returned_vector"].shape == (5941,)
        assert raw["candidate_alpha"].shape == raw["candidate_beta"].shape == (1980,)
