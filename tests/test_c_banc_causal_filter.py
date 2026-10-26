import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import clarabel
import numpy as np
import pytest
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))
SPEC = importlib.util.spec_from_file_location(
    "fit_c_banc_m3_causal_filter", TOOLS / "fit_c_banc_m3_causal_filter.py"
)
assert SPEC is not None and SPEC.loader is not None
fit = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = fit
SPEC.loader.exec_module(fit)


def small_pool():
    membership = np.array([0, 0, 0, 1, 1, 2])
    pool = np.zeros((3, 6))
    pool[0, :3] = 1 / 3
    pool[1, 3:5] = 1 / 2
    pool[2, 5] = 1
    return pool, membership, np.array([1])


def test_causal_filter_has_future_prefix_invariance():
    rng = np.random.default_rng(146)
    rate = rng.normal(size=(31, 7))
    changed = rate.copy()
    changed[13:] = rng.normal(size=changed[13:].shape) * 100
    original_bank = fit.causal_filter_bank(rate)
    changed_bank = fit.causal_filter_bank(changed)
    np.testing.assert_array_equal(original_bank[:13], changed_bank[:13])
    assert not np.array_equal(original_bank[13:], changed_bank[13:])


def test_middle_branch_is_exact_10ms_zero_state_reference():
    rng = np.random.default_rng(10)
    rate = rng.uniform(size=(40, 5))
    reference = np.empty_like(rate)
    state = np.zeros(rate.shape[1])
    alpha = -np.expm1(-0.005 / 0.01)
    for index, row in enumerate(rate):
        state = state + alpha * (row - state)
        reference[index] = state
    bank = fit.causal_filter_bank(rate)
    np.testing.assert_array_equal(bank[:, :, fit.MIDDLE_BRANCH], reference)
    np.testing.assert_array_equal(bank[0, :, 0], rate[0] * -np.expm1(-1.0))


def test_anchor_is_last_motor_middle_and_lifted_baseline_is_guarded():
    pool, membership, ltm = small_pool()
    anchor = fit.build_causal_anchor_map(pool, membership, ltm)
    np.testing.assert_array_equal(anchor.variable_muscles, [0])
    np.testing.assert_array_equal(anchor.fixed_muscles, [1, 2])
    np.testing.assert_array_equal(anchor.anchors_flat, [3 * 2 + 1])
    assert np.array_equal(anchor.free_flat, np.sort(anchor.free_flat))
    lifted = fit.weights_from_free(pool, anchor, anchor.lifted_free)
    np.testing.assert_array_equal(lifted[1:], fit.canonical_weights(pool)[1:])
    np.testing.assert_allclose(lifted[0, :3, 0], (1 / 3) * 1e-6)
    np.testing.assert_allclose(lifted[0, :3, 1], (1 / 3) * (1 - 2e-6))
    np.testing.assert_allclose(lifted[0, :3, 2], (1 / 3) * 1e-6)
    assert (
        np.max(
            np.abs(
                anchor.constant
                + anchor.matrix @ anchor.canonical_free
                - fit.canonical_weights(pool)[
                    anchor.supported_muscles,
                    anchor.supported_motors,
                    anchor.supported_branches,
                ]
            )
        )
        <= 1e-15
    )


def test_mixed_filter_affine_command_and_velocity_equal_direct_evaluation():
    rng = np.random.default_rng(7)
    pool, membership, ltm = small_pool()
    anchor = fit.build_causal_anchor_map(pool, membership, ltm)
    bank = fit.causal_filter_bank(rng.uniform(size=(24, 6)))
    free = anchor.lifted_free.copy()
    free += np.linspace(-2e-8, 2e-8, free.size)
    weights = fit.weights_from_free(pool, anchor, free)
    canonical_activation = bank[:, :, 1] @ pool.T
    direct_activation = fit.activation_from_weights(
        bank, weights, canonical_activation, anchor.variable_muscles
    )
    design, offset = fit.activation_affine(bank, pool, anchor)
    affine_activation = np.einsum("tmf,f->tm", design, free) + offset
    np.testing.assert_allclose(direct_activation, affine_activation, atol=2e-15, rtol=0)
    moment = rng.normal(size=(4, 3))
    rest = rng.normal(size=4)
    direct_q = rest + fit.COMMAND_GAIN * direct_activation @ moment.T
    affine_q = rest + fit.COMMAND_GAIN * affine_activation @ moment.T
    np.testing.assert_allclose(direct_q, affine_q, atol=2e-15, rtol=0)
    np.testing.assert_allclose(
        np.diff(direct_q, axis=0) / fit.DT,
        np.diff(affine_q, axis=0) / fit.DT,
        atol=5e-13,
        rtol=0,
    )


def test_fixed_outputs_reuse_canonical_reduction_bit_exactly():
    rng = np.random.default_rng(9)
    pool, membership, ltm = small_pool()
    anchor = fit.build_causal_anchor_map(pool, membership, ltm)
    bank = fit.causal_filter_bank(rng.normal(size=(17, 6)))
    canonical_activation = bank[:, :, 1] @ pool.T
    weights = fit.weights_from_free(pool, anchor, anchor.lifted_free)
    actual = fit.activation_from_weights(
        bank, weights, canonical_activation, anchor.variable_muscles
    )
    np.testing.assert_array_equal(
        actual[:, anchor.fixed_muscles], canonical_activation[:, anchor.fixed_muscles]
    )
    np.testing.assert_array_equal(
        weights[anchor.fixed_muscles], fit.canonical_weights(pool)[anchor.fixed_muscles]
    )


def generated_uneven_problem():
    rng = np.random.default_rng(14602)
    membership = np.repeat(np.arange(96), 2)
    pool = np.zeros((96, membership.size))
    pool[membership, np.arange(membership.size)] = 0.5
    ltm = np.array([0, 17, 32, 48, 65, 80])
    bank = rng.uniform(0.0, 2.0, size=(2000, membership.size, 3))
    bank[:, :, 1] = fit.causal_filter_bank(rng.uniform(size=(2000, membership.size)))[
        :, :, 1
    ]
    muscle_legs = np.repeat(np.arange(6), [17, 15, 16, 17, 15, 16])
    anatomy = {
        "muscles": [
            {
                "leg": fit.LEGS[int(muscle_legs[muscle])],
                "columns": np.flatnonzero(membership == muscle).tolist(),
            }
            for muscle in range(96)
        ]
    }
    moment = np.zeros((42, 96))
    for muscle, leg_index in enumerate(muscle_legs):
        moment[7 * leg_index : 7 * (leg_index + 1), muscle] = rng.normal(
            scale=0.2, size=7
        )
    rest = rng.normal(scale=0.1, size=42)
    b = bank[:, :, 1]
    baseline_activation = b @ pool.T
    baseline_command = rest + fit.COMMAND_GAIN * baseline_activation @ moment.T
    target_indices = np.arange(174, 2001)
    pair_start = target_indices[:-1]
    pair_end = target_indices[1:]
    baseline_velocity = np.diff(baseline_command, axis=0)[pair_end - 2] / fit.DT
    startup_tendon = 0.2 * np.tanh(baseline_activation[:99, ltm] / 10)
    saved = {
        "pool": pool,
        "membership": membership,
        "ltm": ltm,
        "bank": bank,
        "b": b,
        "moment": moment,
        "rest": rest,
        "baseline": baseline_command,
        "recorded_rate": np.zeros_like(b),
        "target_indices": target_indices,
        "pair_start": pair_start,
        "pair_end": pair_end,
        "target": baseline_command[target_indices - 1] + 0.2,
        "target_velocity": baseline_velocity + 0.2,
        "startup_target": baseline_command[:99].copy(),
        "startup_tendon_target": startup_tendon,
    }
    freeze = {
        "branch_guard_fraction": 1e-6,
        "proposal_command_RMS_ratio": 0.8,
        "proposal_velocity_RMS_ratio": 1.0,
        "startup_limits": {
            "q_RMS": 0.018,
            "q_max_abs": 0.04,
            "LTM_RMS": 0.009,
            "LTM_max_abs": 0.025,
        },
        "margin_bounds": [-1.0, 1.0],
        "QR_absolute_atol": 1e-12,
        "QR_relative_atol": 1e-10,
        "baseline_feasibility_margin": -0.251,
    }
    return saved, {"anatomy": anatomy}, freeze


def test_full_uneven_anatomy_15_soc_affines_match_direct_physics():
    saved, data, freeze = generated_uneven_problem()
    problem, raw = fit.build_conic_problem(saved, data, freeze)
    assert len(problem.criteria) == 15
    assert problem.cone_kinds.count("SecondOrderConeT") == 15
    np.testing.assert_array_equal(
        np.bincount(
            [fit.LEGS.index(item["leg"]) for item in data["anatomy"]["muscles"]],
            minlength=6,
        ),
        [17, 15, 16, 17, 15, 16],
    )
    weights = fit.weights_from_free(
        saved["pool"], problem.anchor_map, problem.anchor_map.lifted_free
    )
    composite = fit.evaluate_causal_composite(saved, weights)
    arrays = composite["arrays"]
    command_error = arrays["candidate_selected_command_rad"] - saved["target"]
    velocity_error = (
        arrays["candidate_selected_velocity_rad_s"] - saved["target_velocity"]
    )
    startup_error = arrays["candidate_startup_command_rad"] - saved["startup_target"]
    expected_norms = {
        "overall_command": np.linalg.norm(command_error),
        "overall_velocity": np.linalg.norm(velocity_error),
        "overall_startup": np.linalg.norm(startup_error),
    }
    for index, leg in enumerate(fit.LEGS):
        expected_norms[f"{leg}_command"] = np.linalg.norm(
            command_error[:, 7 * index : 7 * (index + 1)]
        )
        expected_norms[f"{leg}_velocity"] = np.linalg.norm(
            velocity_error[:, 7 * index : 7 * (index + 1)]
        )
    for criterion in problem.criteria:
        affine_norm = np.linalg.norm(
            criterion.direct_matrix @ problem.anchor_map.lifted_free
            + criterion.direct_offset
        )
        assert affine_norm == pytest.approx(
            expected_norms[criterion.name], rel=2e-12, abs=2e-12
        )
    assert np.max(np.abs(raw["lifted_minus_canonical_selected_command_rad"])) > 0


def test_filter_failure_keeps_rates_bank_and_completed_update_counts(monkeypatch):
    rate = np.arange(2000 * 391, dtype=np.float64).reshape(2000, 391) / 1000
    saved = {"b": np.zeros((2000, 391))}
    monkeypatch.setattr(fit, "validate_saved_inputs", lambda *_: saved)
    partial, progress = {}, {}
    with pytest.raises(ValueError, match="10ms branch differs"):
        fit.validate_causal_saved_inputs(
            {"e143": {"recorded_motor_rate": rate}},
            {"control_dt_s": 0.005, "tau_s": [0.005, 0.01, 0.04]},
            partial=partial,
            progress=progress,
        )
    np.testing.assert_array_equal(partial["recorded_motor_rate"], rate)
    assert partial["filter_bank"].shape == (2000, 391, 3)
    assert np.isfinite(partial["filter_bank"]).all()
    assert progress["recorded_EMA_updates"] == 2000
    assert progress["recorded_EMA_scalar_updates"] == 2_346_000


def test_candidate_is_published_before_composite_failure(monkeypatch):
    pool, membership, ltm = small_pool()
    anchor = fit.build_causal_anchor_map(pool, membership, ltm)
    problem = SimpleNamespace(anchor_map=anchor)
    solution = SimpleNamespace(
        x=np.concatenate((anchor.lifted_free, [0.1])),
        s=np.empty(0),
        z=np.empty(0),
        status="Solved",
        iterations=1,
        solve_time=0.0,
        obj_val=-0.1,
        obj_val_dual=-0.1,
        r_prim=0.0,
        r_dual=0.0,
    )
    monkeypatch.setattr(
        fit,
        "solve_conic_problem",
        lambda *_args, **_kwargs: (solution, [], {"max_threads": 1}),
    )
    monkeypatch.setattr(fit, "solution_arrays", lambda *_: {"solver_x": solution.x})
    monkeypatch.setattr(
        fit,
        "evaluate_causal_composite",
        lambda *_: (_ for _ in ()).throw(RuntimeError("generated composite failure")),
    )
    partial = {}
    progress = {"timings_s": {}, "fit_calls": 1}
    saved = {"pool": pool}
    with pytest.raises(RuntimeError, match="generated composite failure"):
        fit.calculate(
            {},
            {"total_fit_limit_s": 30},
            progress,
            partial,
            prepared=(problem, {"pre": np.array([1])}, saved),
        )
    np.testing.assert_array_equal(
        partial["candidate_weights"],
        fit.weights_from_free(pool, anchor, anchor.lifted_free),
    )
    assert float(partial["returned_margin_t"]) == pytest.approx(0.1)


def reference_problem():
    soc_A, soc_b = fit.make_soc_rows(sparse.csr_matrix([[1.0]]), [0.0], 1.0, 1)
    nonnegative_A = sparse.csr_matrix(
        [[-1.0, 0.0], [1.0, 0.0], [0.0, -1.0], [0.0, 1.0]]
    )
    nonnegative_b = np.array([0.0, 1.0, 1.0, 1.0])
    A = sparse.vstack((nonnegative_A, soc_A), format="csc")
    b = np.concatenate((nonnegative_b, soc_b))
    return fit.conic.ConicProblem(
        sparse.csc_matrix((2, 2)),
        np.array([0.0, -1.0]),
        A,
        b,
        (clarabel.NonnegativeConeT(4), clarabel.SecondOrderConeT(2)),
        ("NonnegativeConeT", "SecondOrderConeT"),
        np.array([4, 2]),
        np.array([0.0, -1.0]),
        np.array([1.0, 1.0]),
        SimpleNamespace(),
        (),
        sparse.csr_matrix((0, 1)),
        np.empty(0),
    )


def solver_freeze():
    return {
        "solver_settings": {
            "max_threads": 1,
            "direct_solve_method": "qdldl",
            "max_iter": 100,
            "time_limit": 30.0,
            "verbose": False,
            "tol_gap_abs": 1e-10,
            "tol_gap_rel": 1e-10,
            "tol_feas": 1e-10,
            "tol_infeas_abs": 1e-10,
            "tol_infeas_rel": 1e-10,
        }
    }


def test_real_clarabel_small_problem_and_caller_owned_raw():
    problem = reference_problem()
    partial, progress = {}, {}
    solution, history, settings = fit.solve_conic_problem(
        problem, solver_freeze(), partial=partial, progress=progress
    )
    assert str(solution.status) == "Solved"
    np.testing.assert_allclose(solution.x, [0, 1], atol=2e-8)
    np.testing.assert_array_equal(partial["solver_x"], np.asarray(solution.x))
    np.testing.assert_array_equal(partial["solver_s"], np.asarray(solution.s))
    np.testing.assert_array_equal(partial["solver_z_raw"], np.asarray(solution.z))
    assert progress["fit_calls"] == 1
    assert history and settings["max_threads"] == 1


def test_safe_wrapper_keeps_bare_return_if_late_settings_retrieval_fails():
    problem = reference_problem()
    partial, progress = {}, {}

    class LateFailureSolver:
        def __init__(self, *_args):
            pass

        def set_termination_callback(self, callback):
            self.callback = callback

        def solve(self):
            point = np.array([0.0, 1.0])
            return SimpleNamespace(
                x=point,
                s=np.asarray(problem.b - problem.A @ point),
                z=np.zeros(problem.b.size),
                status="Solved",
                iterations=1,
                solve_time=0.0,
                obj_val=-1.0,
                obj_val_dual=-1.0,
                r_prim=0.0,
                r_dual=0.0,
            )

        def get_settings(self):
            raise RuntimeError("generated late settings failure")

    with pytest.raises(RuntimeError, match="generated late settings failure"):
        fit.solve_conic_problem(
            problem,
            {"solver_settings": {}},
            solver_factory=LateFailureSolver,
            partial=partial,
            progress=progress,
        )
    np.testing.assert_array_equal(partial["solver_x"], [0.0, 1.0])
    assert "solver_s" in partial and "solver_z_raw" in partial
    assert progress["fit_calls"] == 1
    assert progress["solver_state"]["phase"] == "settings_retrieval_error"


def minimal_freeze():
    return {
        "minimum_free_disk_bytes": 0,
        "internal_wall_limit_s": 60,
        "maximum_raw_bytes": 10_000_000,
    }


def test_run_saves_candidate_raw_before_assessment(monkeypatch, tmp_path):
    events = []
    monkeypatch.setattr(
        fit, "reviewed_inputs", lambda *_: (minimal_freeze(), {"source_sha256": {}})
    )
    monkeypatch.setattr(fit, "load_inputs", lambda *_: {})
    monkeypatch.setattr(
        fit, "validate_causal_saved_inputs", lambda *_args, **_kwargs: {}
    )
    monkeypatch.setattr(
        fit, "build_conic_problem", lambda *_: (object(), {"pre": np.array([1])})
    )
    monkeypatch.setattr(
        fit,
        "calculate",
        lambda *_args, **_kwargs: (
            {
                "candidate_weights": np.ones((1, 1, 1)),
                "dual_projected_margin_upper_bound": np.asarray(0.2),
            },
            {"solver": {}},
            {"saved": {}, "weights": np.ones((1, 1, 1)), "composite": {}},
            object(),
        ),
    )
    original_save = fit.save_npz_new

    def recording_save(path, arrays):
        events.append(("save", Path(path).name, sorted(arrays)))
        original_save(path, arrays)

    def assess(*_args):
        events.append(("assess",))
        return {"conditional_causal_filter_proposal": False}

    monkeypatch.setattr(fit, "save_npz_new", recording_save)
    monkeypatch.setattr(fit, "assess_physical_candidate", assess)
    result = fit.run(Path("f"), "f", Path("r"), "r", tmp_path / "out")
    assert result["status"] == "pass"
    assert [event[:2] for event in events] == [
        ("save", "conic-problem.npz"),
        ("save", "causal-filter-raw.npz"),
        ("assess",),
    ]


def test_primary_calculation_error_survives_late_resource_error(monkeypatch, tmp_path):
    monkeypatch.setattr(
        fit, "reviewed_inputs", lambda *_: (minimal_freeze(), {"source_sha256": {}})
    )
    monkeypatch.setattr(fit, "load_inputs", lambda *_: {})
    monkeypatch.setattr(
        fit, "validate_causal_saved_inputs", lambda *_args, **_kwargs: {}
    )
    monkeypatch.setattr(
        fit, "build_conic_problem", lambda *_: (object(), {"pre": np.array([1])})
    )

    def failed_calculation(_data, _freeze, _progress, partial, **_kwargs):
        partial["attempt"] = np.array([1])
        raise ValueError("primary generated calculation")

    monkeypatch.setattr(fit, "calculate", failed_calculation)
    monkeypatch.setattr(
        fit,
        "output_bytes",
        lambda *_: (_ for _ in ()).throw(OSError("late generated stat")),
    )
    with pytest.raises(ValueError, match="primary generated calculation") as caught:
        fit.run(Path("f"), "f", Path("r"), "r", tmp_path / "out")
    assert any(
        "late generated stat" in note for note in getattr(caught.value, "__notes__", [])
    )
    summary = json.loads(
        (tmp_path / "out" / "summary.json").read_text(encoding="utf-8")
    )
    assert summary["status"] == "failed"
    assert summary["failure"]["message"] == "primary generated calculation"
