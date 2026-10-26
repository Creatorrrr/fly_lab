"""Generated E212 anatomy/certificate/storage tests; BVLS call is deferred."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tools import analyze_c_banc_m3_anatomical_capacity as cap


def test_group_cap_sum_shared_muscle_and_exact_zero():
    D = np.zeros((3, 3, 16))
    D[0, 0, 0] = .1
    D[0, 1, 1] = .2
    D[1, 2, 0] = 1e-16  # weak positive must survive
    membership = np.array([0, 0, 1])
    U = cap.sum_group_caps(D, membership, np.array([0, 1, 2]),
                           np.array([0, 1]))
    np.testing.assert_array_equal(U[:, 0], [.1 + .2, 0., 0.])
    np.testing.assert_array_equal(U[:, 1], [0., 1e-16, 0.])
    D[0, 0, 0] = -1e-12
    with pytest.raises(ValueError, match="nonnegative"):
        cap.sum_group_caps(D, membership, np.arange(3), np.array([0, 1]))


def test_anatomical_ownership_rejects_cross_leg():
    groups = [{"leg": cap.LEGS[m // 16]} for m in range(96)]
    moment = np.zeros((42, 96))
    free_muscles = np.arange(0, 96, 16)
    columns = cap.anatomical_leg_columns(free_muscles, groups, moment,
                                         expected_muscles=6)
    assert [v.tolist() for v in columns] == [[i] for i in range(6)]
    moment[7, 0] = 1e-9
    with pytest.raises(ValueError, match="crosses"):
        cap.anatomical_leg_columns(free_muscles, groups, moment,
                                   expected_muscles=6)


def test_affine_minorant_is_global_box_lower_bound():
    C = np.zeros((7, 2))
    C[0, 0] = 1.
    C[1, 1] = 1.
    y = np.array([1.5, -.5, 0, 0, 0, 0, 0])
    record = cap.certificate(C, y, np.array([1., 0.]))
    assert record["f"] == pytest.approx(.5)
    assert record["minorant"] == pytest.approx(.5)
    assert record["lower"] == pytest.approx(.5)
    assert record["gap"] == pytest.approx(0)
    assert record["certified"]
    for a in np.linspace(0, 1, 11):
        for b in np.linspace(0, 1, 11):
            assert record["lower"] <= np.linalg.norm(
                C @ np.array([a, b]) - y)**2 + 1e-14
    with pytest.raises(ValueError, match="strict feasible"):
        cap.certificate(C, y, np.array([1. + 1e-14, 0.]))


def test_zero_cap_skips_mock_solver_and_weak_positive_calls_once():
    calls = []

    def fake_solver(C, y, **options):
        calls.append((C.copy(), y.copy(), options))
        return SimpleNamespace(x=np.array([.25]), status=1,
                               optimality=0., nit=2)

    C = np.zeros((7, 2))
    y = np.ones(7)
    zero = cap.solve_frame(C, y, active=np.array([False, False]),
                           solver=fake_solver)
    assert not zero["called"] and zero["status"] == -2
    assert zero["f"] == 7. and zero["lower"] == 7.
    C[0, 0] = 1e-16
    positive = cap.solve_frame(C, y, active=np.array([True, False]),
                               solver=fake_solver)
    assert positive["called"] and len(calls) == 1
    assert calls[0][0].shape == (7, 1)
    assert calls[0][2] == {"method": "bvls", "bounds": (0., 1.),
                           "tol": 1e-10, "max_iter": 100}
    assert positive["v"].tolist() == [.25, 0.]


def test_raw_return_before_box_validation_and_iteration_cap():
    seen = []

    def fake_solver(_C, _y, **_options):
        return SimpleNamespace(x=np.array([1. + 1e-9]), status=1,
                               optimality=0., nit=3)

    C = np.ones((7, 1))
    with pytest.raises(ValueError, match="strict feasible"):
        cap.solve_frame(C, np.zeros(7), solver=fake_solver,
                        raw_sink=lambda *args: seen.append(args[0].copy()))
    assert seen and seen[0][0] > 1.
    seen.clear()

    def capped(_C, _y, **_options):
        return SimpleNamespace(x=np.array([.5]), status=0,
                               optimality=1., nit=100)

    incomplete = cap.solve_frame(C, np.zeros(7), solver=capped,
                                 raw_sink=lambda *args: seen.append(args[0].copy()))
    assert incomplete["status"] == 0 and not incomplete["certified"]
    assert np.isfinite(incomplete["minorant"])
    assert len(seen) == 1


def test_partial_raw_files_have_nan_and_completed_authority(tmp_path):
    U = np.zeros((1923, 90))
    q0 = np.zeros((2001, 42))
    targets = np.zeros((1923, 42))
    arrays = cap.create_raw_store(tmp_path / "capacity-raw", U, q0, targets)
    assert np.array_equal(np.load(tmp_path / "capacity-raw" / "frame_indices.npy"),
                          np.arange(1, 1924))
    assert np.isnan(arrays["fraction"]).all()
    assert not arrays["completed"].any()
    record = cap.certificate(np.zeros((7, 1)), np.zeros(7), np.zeros(1))
    record.update(status=-2, optimality=0., nit=0, called=False,
                  active_count=0, wall_s=0.)
    cap.write_frame(arrays, 0, 0, np.array([0]), np.zeros(1), record)
    arrays["q_feasible"][0, :7] = 0.
    arrays["residual"][0, :7] = 0.
    arrays["completed"][0, 0] = True
    cap.flush_raw(arrays)
    assert arrays["completed"][0, 0] and not arrays["completed"][0, 1]
    assert np.isnan(arrays["f"][0, 1])
    with pytest.raises(ValueError, match="incomplete"):
        cap.summarize_complete(arrays, {}, {})
    del arrays
    assert not np.load(tmp_path / "capacity-raw" / "completed.npy")[0, 1]


def test_gate_rejects_unapproved_review_before_original_inputs(tmp_path):
    freeze = Path("verification/banc-natural-gait-m3-20260919/"
                  "e212-anatomical-capacity-freeze.json")
    review = tmp_path / "review.json"
    review.write_text(json.dumps({"schema": cap.REVIEW_SCHEMA,
        "approved": False, "blocking_defects": ["synthetic"],
        "freeze_sha256": cap.FREEZE_SHA, "source_sha256": {}}),
        encoding="utf-8")
    with pytest.raises(ValueError, match="frozen SHA"):
        cap.verify_gate(freeze, "0" * 64, review, cap.file_sha(review))
    with pytest.raises(ValueError, match="source review"):
        cap.verify_gate(freeze, cap.FREEZE_SHA, review, cap.file_sha(review))


def test_generated_bvls_one_analytic_box_projection(tmp_path):
    """DEFERRED until parent review: exactly one generated SciPy BVLS call."""
    C = np.zeros((7, 2))
    C[0, 0] = 1.
    C[1, 1] = 1.
    y = np.array([1.5, -.5, 0, 0, 0, 0, 0])

    def persist_before_check(v, solved, wall, active):
        np.savez(tmp_path / "generated-bvls-raw.npz",
                 C=C, y=y, v=v, status=int(solved.status),
                 success=bool(solved.success), message=str(solved.message),
                 optimality=float(solved.optimality), nit=int(solved.nit),
                 cost=float(solved.cost), fun=np.asarray(solved.fun),
                 active_mask=np.asarray(solved.active_mask),
                 unbounded_x=np.asarray(solved.unbounded_sol[0]),
                 unbounded_residuals=np.asarray(solved.unbounded_sol[1]),
                 unbounded_rank=np.asarray(solved.unbounded_sol[2]),
                 unbounded_singular_values=np.asarray(solved.unbounded_sol[3]),
                 wall_s=float(wall), active=active)

    raw = cap.solve_frame(C, y, raw_sink=persist_before_check)
    np.savez(tmp_path / "generated-bvls-derived.npz",
             f=raw["f"], minorant=raw["minorant"], lower=raw["lower"],
             upper=raw["upper"], gap=raw["gap"],
             residual=raw["residual"], gradient=raw["gradient"],
             certified=raw["certified"])
    print({"generated_bvls_calls": 1, "status": raw["status"],
           "v": raw["v"].tolist(), "f": raw["f"],
           "lower": raw["lower"], "gap": raw["gap"],
           "raw_file": str(tmp_path / "generated-bvls-raw.npz"),
           "derived_file": str(tmp_path / "generated-bvls-derived.npz")})
    assert (tmp_path / "generated-bvls-raw.npz").is_file()
    np.testing.assert_allclose(raw["v"], [1., 0.], atol=1e-12, rtol=0)
    assert raw["f"] == pytest.approx(.5, abs=1e-12)
    assert raw["lower"] == pytest.approx(.5, abs=1e-10)
    assert raw["certified"] and raw["gap"] <= 1e-8
