"""Synthetic checks for E197; no saved E150 numerical arrays are opened."""

from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import fit_c_banc_m3_prefilter_recruitment as e197


def test_prefilter_pulse_is_causal_and_hinge_precedes_ema() -> None:
    rate = np.array([[50.0], [0.0], [20.0], [0.0], [0.0]], dtype=np.float64)
    knot = np.array([10.0])
    tau = np.array([0.005, 0.01, 0.04])
    low, high = e197.prefilter_state(rate, knot, tau)
    alpha = -np.expm1(-0.005 / tau)
    assert np.array_equal(low[0], np.zeros((1, 3)))
    assert np.array_equal(high[0], np.zeros((1, 3)))
    np.testing.assert_allclose(low[1, 0], 0.0)
    np.testing.assert_allclose(high[1, 0], 0.0)
    np.testing.assert_allclose(low[2, 0], alpha * 10)
    np.testing.assert_allclose(high[2, 0], alpha * 10)
    np.testing.assert_allclose(low[3, 0], (1 - alpha) * low[2, 0])
    np.testing.assert_allclose(high[3, 0], (1 - alpha) * high[2, 0])
    linear = np.zeros((len(rate), 3))
    for tick in range(1, len(rate)):
        linear[tick] = linear[tick - 1] + alpha * (
            rate[tick, 0] - linear[tick - 1]
        )
    np.testing.assert_allclose(low[:, 0] + high[:, 0], linear, atol=1e-15)
    assert high[3, 0, 0] > 0
    assert max(linear[3, 0] - knot[0], 0) == 0
    prefix_low, prefix_high = e197.prefilter_state(rate[:4], knot, tau)
    np.testing.assert_array_equal(prefix_low, low[:4])
    np.testing.assert_array_equal(prefix_high, high[:4])


def test_prefilter_zero_kappa_and_variable_dt() -> None:
    rate = np.array([[9.0, 3.0], [4.0, 8.0], [4.0, 8.0]])
    knot = np.array([0.0, 5.0])
    tau = np.array([0.005, 0.01, 0.04])
    low, high = e197.prefilter_state(rate, knot, tau, dt=0.003)
    alpha = -np.expm1(-0.003 / tau)
    np.testing.assert_array_equal(low[:, 0], np.zeros((3, 3)))
    np.testing.assert_allclose(high[1, 0], alpha * 4)
    np.testing.assert_allclose(low[1, 1], alpha * 5)
    np.testing.assert_allclose(high[1, 1], alpha * 3)
    assert not np.array_equal(high[1], e197.prefilter_state(rate, knot, tau)[1][1])
    with pytest.raises(ValueError):
        e197.prefilter_state(rate, knot, tau, dt=0)


def _tiny_saved(
    tick_count: int = 102, *, noncontiguous: bool = False
) -> dict[str, object]:
    owner = np.empty(391, dtype=np.int64)
    motors = (
        np.sort(np.random.default_rng(197).choice(391, 330, replace=False))
        if noncontiguous else np.arange(330)
    )
    fixed_motors = np.setdiff1d(np.arange(391), motors)
    owner[motors] = np.arange(330) % 77
    owner[fixed_motors] = 77 + np.arange(61) % 19
    pair_motors = np.repeat(motors, 3)
    pair_filters = np.tile(np.arange(3), 330)
    pair_muscles = owner[pair_motors]
    rate = np.tile(np.linspace(0, 30, tick_count)[:, None], (1, 391))
    kappa = np.full(391, 10.0)
    low, high = e197.prefilter_state(rate, kappa)
    basis = np.empty((tick_count, 1980))
    basis[:, 0::2] = low[:, pair_motors, pair_filters]
    basis[:, 1::2] = high[:, pair_motors, pair_filters]
    slopes = np.zeros((96, 391, 3, 2))
    fixed = np.arange(77, 96, dtype=np.int64)
    original = np.zeros((tick_count, 96))
    original[1:, fixed] = 0.05
    group_legs = np.repeat(np.arange(6), [17, 15, 16, 17, 15, 16])
    moment = np.zeros((42, 96))
    for muscle in range(96):
        moment[7 * group_legs[muscle], muscle] = (muscle + 1) / 100
    target_indices = (
        np.arange(100, 1924) if tick_count == 2001 else np.arange(1, 101)
    )
    target_count = len(target_indices)
    return {
        "low": low, "high": high, "slopes": slopes, "membership": owner,
        "fixed_muscles": fixed, "original_activation": original,
        "activation": original.copy(), "branch_basis": basis,
        "pair_motors": pair_motors, "pair_filters": pair_filters,
        "pair_muscles": pair_muscles, "group_legs": group_legs,
        "moment": moment, "rest": np.linspace(-0.1, 0.1, 42),
        "recorded_command": np.full((tick_count, 42), 0.123),
        "target_indices": target_indices,
        "pair_start": target_indices[:-1], "pair_end": target_indices[1:],
        "target": np.zeros((target_count, 42)),
        "target_velocity": np.zeros((target_count - 1, 42)),
        "startup_target": np.zeros((99, 42)),
        "tendon": np.zeros((tick_count, 6)),
        "w0": np.zeros(1980),
        "canonical": np.zeros(1980),
        "baseline_metrics": {
            "overall_command_RMS": 0.001, "overall_velocity_RMS": 0.001,
            "by_leg": {
                name: {"command_RMS": 0.001, "velocity_RMS": 0.001}
                for name in e197.LEGS
            },
        },
        "startup_ltm_RMS": 0.0, "startup_ltm_max_abs": 0.0,
    }


def test_unanchored_native_matches_affine_and_preserves_tick_zero() -> None:
    saved = _tiny_saved()
    weights = np.linspace(0.0001, 0.0009, 1980)
    metrics, arrays = e197.candidate(saved, weights)
    native_a, native_q = e197.independent_candidate(saved, weights)
    np.testing.assert_allclose(native_a, arrays["candidate_activation"], atol=1e-13)
    np.testing.assert_allclose(native_q, arrays["candidate_command"], atol=1e-13)
    np.testing.assert_array_equal(native_q[0], saved["recorded_command"][0])
    expected = saved["rest"] + 0.02 * saved["moment"] @ native_a[2]
    np.testing.assert_allclose(native_q[2], expected, atol=1e-15)
    assert np.isfinite(metrics["overall_command_RMS"])
    assert not np.array_equal(native_q[2], saved["recorded_command"][2])
    np.testing.assert_array_equal(
        native_a[:, saved["fixed_muscles"]],
        saved["original_activation"][:, saved["fixed_muscles"]],
    )


def test_direct_affine_and_positive_physical_sums() -> None:
    saved = _tiny_saved()
    weights = np.full(1980, 0.1)
    _, arrays = e197.candidate(saved, weights)
    for kind, rows, expected in (
        ("command", np.arange(1, 101), arrays["candidate_command"][1:101]),
        ("velocity", np.arange(1, 100), arrays["candidate_velocity"]),
        ("startup", np.arange(1, 100), arrays["candidate_command"][1:100]),
    ):
        for leg in range(6):
            matrix, offset, active = e197.local_affine(saved, leg, rows, kind)
            joints = slice(7 * leg, 7 * (leg + 1))
            if kind == "velocity":
                direct = expected[:, joints]
            else:
                direct = expected[:, joints]
            np.testing.assert_allclose(
                (matrix @ weights[active] + offset).reshape(-1, 7),
                direct, atol=1e-13,
            )
    freeze = {"contract": {"motor_floor_fraction": 1e-6,
                           "tonic_lower_budget": 0.5,
                           "total_upper_budget": 2.0,
                           "physical_atol": 1e-12},
              "sample_contract": {"variable_muscles": list(range(77))}}
    physical = e197.physical_gates(saved, weights, freeze)
    assert all(physical.values())
    bad = weights.copy()
    bad[0] = -1e-4
    assert not e197.physical_gates(saved, bad, freeze)["coefficient_exact_bounds"]


def _full_fixture() -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    saved = _tiny_saved(2001, noncontiguous=True)
    owner = saved["membership"]
    motors = np.unique(saved["pair_motors"])
    counts = np.bincount(owner, minlength=96)
    pool = np.zeros((96, 391))
    pool[owner, np.arange(391)] = 1 / counts[owner]
    rate = np.tile(np.linspace(0, 30, 2001)[:, None], (1, 391))
    kappa = np.full(391, 10.0)
    low, high = e197.prefilter_state(rate, kappa)
    data = {
        "recruitment": {
            "motor_rate": rate, "bank": low + high,
            "original_activation": saved["original_activation"],
        },
        "parameters": {
            "kappa": kappa, "tau": e197.TAU.copy(), "slopes": saved["slopes"],
        },
        "adapter": {"membership": owner, "pool": pool},
    }
    root = Path(__file__).resolve().parents[1]
    freeze_path = root / "verification/banc-natural-gait-m3-20260919"
    freeze_path /= "e197-prefilter-recruitment-freeze.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    freeze["sample_contract"] = {
        "variable_motors": motors.tolist(),
        "variable_muscles": list(range(77)),
        "fixed_muscles": saved["fixed_muscles"].tolist(),
        "group_legs": saved["group_legs"].tolist(),
    }
    return data, saved, freeze


@pytest.fixture(scope="module")
def full_problem() -> tuple[dict[str, object], dict[str, object], dict[str, object],
                            dict[str, np.ndarray], object]:
    data, saved, freeze = _full_fixture()
    raw: dict[str, np.ndarray] = {}
    progress: dict[str, object] = {}
    e197.prepare_transfer(data, saved, freeze, raw, progress)
    problem = e197.build_problem(saved, freeze, raw)
    return data, saved, freeze, raw, problem


def test_full_generated_problem_rows_qr_and_canonical(full_problem: tuple) -> None:
    _, saved, freeze, raw, problem = full_problem
    assert len(np.unique(saved["pair_motors"])) == 330
    assert not np.array_equal(np.unique(saved["pair_motors"]), np.arange(330))
    assert problem.q.shape == (3961,)
    assert problem.A.shape[1] == 3961
    assert problem.cone_dims[0] == 20682
    assert len(problem.criteria) == len(problem.cones) - 1 == 15
    assert len([name for name in raw if name.startswith("qr_")
                and not name.endswith(("_active", "_direct_norms", "_compressed_norms"))]) == 18
    np.testing.assert_array_equal(raw["low_bank"][0], 0)
    np.testing.assert_array_equal(raw["high_bank"][0], 0)
    canonical = raw["canonical_equal_branch"].reshape(990, 2)
    np.testing.assert_array_equal(canonical[:, 0], canonical[:, 1])
    np.testing.assert_array_equal(canonical[saved["pair_filters"] != 1], 0)
    np.testing.assert_allclose(raw["canonical_linear_residual"], 0, atol=1e-12)
    assert not np.array_equal(raw["basis_difference"], 0)
    assert len(raw["equal_branch_diagnostic_vector"]) == 3961
    assert np.max(np.abs(raw["canonical_equal_branch_command"]
                         - raw["w0_prefilter_command"])) > 0
    x = np.r_[np.full(1980, 0.1), np.full(990, 0.1),
              np.full(990, 0.1), 0.0]
    checked = e197.returned_qr_diagnostics(saved, problem, x, freeze, raw)
    assert checked["returned_direct_norms"].shape == (33,)
    assert np.all(checked["returned_qr_absolute_residual"]
                  <= checked["returned_qr_tolerance"])
    true_b = problem.b.copy()
    true_b[:1980] += 1e-9
    true_b[3961:5941] += 1e-9
    true_b[11882:12366] += 1e-7 + 1e-12
    np.testing.assert_allclose(true_b[:1980], 0)
    np.testing.assert_allclose(true_b[3961:5941], 2)
    assert np.all(true_b[11882:12366] > problem.b[11882:12366])


def test_auxiliary_roundoff_does_not_change_runtime_gates() -> None:
    saved = _tiny_saved()
    freeze = {"contract": {"motor_floor_fraction": 1e-6,
                           "tonic_lower_budget": 0.5,
                           "total_upper_budget": 2.0,
                           "physical_atol": 1e-12,
                           "startup_limits": {"q_RMS": 0.018,
                                              "q_max_abs": 0.04,
                                              "LTM_RMS": 0.009,
                                              "LTM_max_abs": 0.025}},
              "sample_contract": {"variable_muscles": list(range(77))}}
    weights = np.full(1980, 0.1)
    _, arrays = e197.candidate(saved, weights)
    metrics = {"overall_command_RMS": 0.0, "overall_velocity_RMS": 0.0,
               "startup_q_RMS": 0.0, "startup_q_max_abs": 0.0,
               "by_leg": {name: {"command_RMS": 0.0, "velocity_RMS": 0.0}
                          for name in e197.LEGS}}
    x = np.r_[weights, np.full(990, 0.1), np.full(990, 0.1), 0.0]
    baseline = e197.candidate_gates(saved, freeze, x, metrics, arrays)
    perturbed = x.copy()
    perturbed[1980] += 1e-10
    perturbed[2970] -= 1e-10
    assert all(baseline.values())
    assert e197.candidate_gates(saved, freeze, perturbed, metrics, arrays) == baseline
    aux = e197.auxiliary_diagnostics(perturbed)
    assert aux["auxiliary_lower_slack"][0] < 0
    assert aux["auxiliary_upper_slack"][0] < 0


def _generated_runner(
    monkeypatch: pytest.MonkeyPatch, full_problem: tuple,
    tmp_path: Path, *, failure: str = "none",
) -> tuple[Path, dict[str, int]]:
    data, saved, original_freeze, built_raw, problem = full_problem
    freeze = deepcopy(original_freeze)
    if failure == "byte_cap":
        freeze["resource_limits"]["raw_bytes"] = 1
    if failure == "deadline":
        freeze["resource_limits"]["internal_s"] = 1e-9
    monkeypatch.setattr(e197, "validate_freeze", lambda *_: (freeze, {}, {}))
    monkeypatch.setattr(e197, "validate_gate", lambda *_: {})
    monkeypatch.setattr(
        e197.e168, "capture_loaded_input",
        lambda _data, raw: raw.update(synthetic_marker=np.asarray(197)),
    )
    monkeypatch.setattr(e197.e168, "validate_loaded_input", lambda *_: None)
    monkeypatch.setattr(e197.e164, "prepare_saved", lambda *_: dict(saved))

    def problem_factory(_saved: dict, _freeze: dict, raw: dict) -> object:
        raw.update(built_raw)
        if failure == "pre_solve":
            raise RuntimeError("synthetic pre-solve failure")
        return problem

    monkeypatch.setattr(e197, "build_problem", problem_factory)
    if failure == "qr_mismatch":
        actual_qr = e197.returned_qr_diagnostics

        def mismatched_qr(*args: object) -> dict[str, np.ndarray]:
            result = actual_qr(*args)
            result["returned_qr_absolute_residual"][0] = 1.0
            return result

        monkeypatch.setattr(e197, "returned_qr_diagnostics", mismatched_qr)
    calls = {"construct": 0, "solve": 0, "callback": 0}

    class FakeSolver:
        def __init__(self, _P: object, _q: object, _A: object, _b: object,
                     _cones: object, settings: object) -> None:
            calls["construct"] += 1
            self.settings = settings

        def set_termination_callback(self, _callback: object) -> None:
            calls["callback"] += 1

        def solve(self) -> SimpleNamespace:
            calls["solve"] += 1
            x = np.r_[np.full(1980, 0.1), np.full(990, 0.1),
                      np.full(990, 0.1), 0.0]
            if failure == "post_return":
                x[0] = np.nan
            return SimpleNamespace(
                x=x, s=problem.b - problem.A @ x,
                z=np.zeros_like(problem.b), status="Solved",
                iterations=1, solve_time=0.01, obj_val=0.0,
                obj_val_dual=0.0, r_prim=0.0, r_dual=0.0,
            )

        def get_settings(self) -> object:
            return self.settings

    root = Path(__file__).resolve().parents[1]
    freeze_path = root / "verification/banc-natural-gait-m3-20260919"
    freeze_path /= "e197-prefilter-recruitment-freeze.json"
    review = tmp_path / "synthetic-gate.json"
    review.write_text("{}", encoding="utf-8")
    out = tmp_path / ("generated-" + failure)
    kwargs = {
        "freeze_path": freeze_path,
        "freeze_sha256": e197.sha256(freeze_path),
        "source_review": review,
        "source_review_sha256": e197.sha256(review),
        "out_path": out,
        "input_loader": lambda _base: data,
        "solver_factory": FakeSolver,
    }
    if failure == "none":
        result = e197.run(**kwargs)
        assert result["validity"] == "valid"
    else:
        with pytest.raises((RuntimeError, ValueError)):
            e197.run(**kwargs)
    return out, calls


@pytest.mark.parametrize(
    ("failure", "expected_solve"),
    [("none", 1), ("pre_solve", 0), ("post_return", 1),
     ("byte_cap", 1), ("deadline", 0), ("qr_mismatch", 1)],
)
def test_generated_runner_one_solve_and_terminal_evidence(
    monkeypatch: pytest.MonkeyPatch, full_problem: tuple,
    tmp_path: Path, failure: str, expected_solve: int,
) -> None:
    out, calls = _generated_runner(monkeypatch, full_problem, tmp_path,
                                   failure=failure)
    assert calls["solve"] == expected_solve
    assert calls["construct"] == calls["callback"] == expected_solve
    result = json.loads((out / "result.json").read_text(encoding="utf-8"))
    ledger = json.loads((out / "ledger.json").read_text(encoding="utf-8"))
    assert (out / "output-manifest.json").exists()
    assert (out / "pre-solve-raw.npz").exists()
    assert result["candidate_adopted"] is False
    assert ledger["fit_calls"] == expected_solve
    if failure == "none":
        assert ledger["status"] == "complete"
        assert (out / "candidate-raw.npz").exists()
        with np.load(out / "candidate-raw.npz", allow_pickle=False) as values:
            assert values["returned_direct_norms"].shape == (33,)
            assert np.all(values["returned_qr_absolute_residual"]
                          <= values["returned_qr_tolerance"])
        with (
            np.load(out / "solver-derived-raw.npz", allow_pickle=False) as values,
            np.load(out / "pre-solve-raw.npz", allow_pickle=False) as before,
        ):
            true_b = values["true_b"]
            inner_b = before["problem_b"]
            np.testing.assert_allclose(true_b[:1980] - inner_b[:1980], 1e-9)
            np.testing.assert_allclose(
                true_b[3961:5941] - inner_b[3961:5941], 1e-9,
            )
            np.testing.assert_allclose(
                true_b[11882:12366] - inner_b[11882:12366], 1e-7 + 1e-12,
            )
    else:
        assert result["validity"] == "incomplete"
        assert ledger["status"] == "technical_failed"
        assert result["primary_error"]["type"] in ("RuntimeError", "ValueError")
    if failure == "post_return":
        assert (out / "solver-return-raw.npz").exists()
        with np.load(out / "solver-return-raw.npz", allow_pickle=False) as values:
            assert np.isnan(values["solver_x"][0])
    if failure == "byte_cap":
        assert (out / "candidate-raw.npz").exists()
        assert any(out.glob("result.json.invalid-stale-*"))
    if failure == "qr_mismatch":
        with np.load(out / "candidate-raw.npz", allow_pickle=False) as values:
            assert values["returned_qr_absolute_residual"][0] == 1.0
