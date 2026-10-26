"""Small numerical science cases; none execute Original inputs, CUDA or bodies."""
import copy

import numpy as np
import pytest

from tools import verify_c_banc_m3_startup_ray as ray


def test_float32_actual_bound_is_separate_and_has_no_epsilon():
    allowed = np.asarray([1.0009, .9991], np.float32).astype(np.float64)
    assert ray.validate_actual_theta(allowed)
    # Requested .001 is mathematically bounded, but f32(1.001) exceeds the actual bound.
    rejected = np.asarray([1.001], np.float32).astype(np.float64)
    assert float(rejected[0] - 1) > .001
    with pytest.raises(ValueError, match="actual float32 delta"):
        ray.validate_actual_theta(rejected)
    with pytest.raises(ValueError):
        ray.validate_actual_theta(np.asarray([1.0009], np.float64))


def test_zero_and_nonfinite_gradient_do_not_create_candidates():
    assert ray.normalized_direction(np.zeros(3, np.float64)) is None
    with pytest.raises(ValueError):
        ray.normalized_direction(np.asarray([np.nan], np.float64))
    with pytest.raises(ValueError):
        ray.normalized_direction(np.asarray([1], np.int64))
    gradient = np.asarray([2., -4., 1.], np.float64)
    direction = ray.normalized_direction(gradient)
    assert np.array_equal(direction, np.asarray([-.5, 1., -.25], np.float64))


def tiny_csr(value=1.):
    return {"data": np.asarray([value, -2.], np.float32),
            "indices": np.asarray([0, 1], np.int32), "indptr": np.asarray([0, 1, 2], np.int32)}


def test_deployment_duplicate_and_noop_preserve_unselected_words():
    csr = tiny_csr()
    mask = {"positions": np.asarray([0], np.int64), "edge_parameters": np.asarray([0], np.int64)}
    gradient = np.asarray([-1.], np.float64)
    evidence, arrays = ray.proposal(gradient, ray.ALPHAS[0], csr, mask)
    assert evidence["eligible_representation"] and evidence["actual_max_delta"] <= .001
    assert arrays["selected_weights_candidate"].dtype == np.float32
    repeated, _ = ray.proposal(gradient, ray.ALPHAS[0], csr, mask, [evidence])
    assert repeated["reason"] == "rounded_duplicate"
    original = csr["data"].copy()
    changed = original.copy()
    changed[mask["positions"]] = arrays["selected_weights_candidate"]
    assert changed[1:].tobytes() == original[1:].tobytes()
    subnormal = np.nextafter(np.float32(0), np.float32(1))
    noop, _ = ray.proposal(gradient, ray.ALPHAS[0], tiny_csr(subnormal), mask)
    assert noop["reason"] == "deployed_noop"
    zero, arrays = ray.proposal(np.zeros(1, np.float64), ray.ALPHAS[0], csr, mask)
    assert zero["reason"] == "zero_R_gradient" and arrays is None
    with pytest.raises(ValueError):
        ray.proposal(gradient, .001, csr, mask)


def test_fixed_descending_first_eligible_stops_without_reselection():
    assert list(ray.ALPHAS) == sorted(ray.ALPHAS, reverse=True)
    rows = [{"alpha": ray.ALPHAS[0], "eligible": False}, {"alpha": ray.ALPHAS[1], "eligible": True}]
    assert ray.choose_first(rows) == 1
    with pytest.raises(ValueError, match="stop immediately"):
        ray.choose_first(rows + [{"alpha": ray.ALPHAS[2], "eligible": True}])
    with pytest.raises(ValueError):
        ray.choose_first(list(reversed(rows)))
    assert ray.choose_first([{"alpha": a, "eligible": False} for a in ray.ALPHAS]) is None
    assert ray.choose_first([]) is None
    gradient = np.asarray([1.], np.float64)
    counted = [dict(row, forward_evaluated=True) for row in rows]
    assert ray.validate_decision_roster(counted, 2, gradient) == 1
    with pytest.raises(ValueError, match="counted"):
        ray.validate_decision_roster(counted, 1, gradient)
    with pytest.raises(ValueError, match="omit"):
        ray.validate_decision_roster([], 0, gradient)


def receipt(evaluated):
    row = ray.expected_work(evaluated)
    row.update(confirm_requested=1, confirm_returned=1)
    row["confirmed"] = {k: row[k + "_returned"] for k in ("forward_controls", "forward_neural_ticks",
                       "recompute_controls", "recompute_neural_ticks", "backward_blocks", "backward_block_ticks")}
    return row


def test_requested_returned_receipts_count_all_actual_alpha_work():
    assert ray.expected_work(0)["forward_controls_returned"] == 115
    assert ray.expected_work(1)["forward_controls_returned"] == 345
    assert ray.expected_work(5)["forward_controls_returned"] == 1265
    assert ray.expected_work(5)["backward_blocks_returned"] == 16
    assert ray.validate_receipts({"actual-return": receipt(2)}, 2)
    for field, value in (("forward_controls_returned", 574), ("confirm_returned", 0),
                         ("backward_blocks_requested", True)):
        bad = copy.deepcopy(receipt(2));bad[field] = value
        with pytest.raises(ValueError):
            ray.validate_receipts({"partial-return": bad}, 2)
    bad = receipt(2);bad["confirmed"]["forward_neural_ticks"] -= 1
    with pytest.raises(ValueError):
        ray.validate_receipts({"unconfirmed": bad}, 2)


def test_startup_metric_excludes_boundary_zero():
    from tools import verify_c_banc_m3_common_descent_body as b
    command, tendon = np.zeros((100, 42), np.float64), np.zeros((100, 6), np.float64)
    command[0] = 1000.;tendon[0] = 1000.
    target = {"target_command": np.zeros((99, 42), np.float64), "target_tendon": np.zeros((99, 6), np.float64)}
    first = b.startup99(command, tendon, **target)
    assert first["passed"] and first["q_max"] == first["tendon_max"] == 0
    command[1, 0] = .05
    assert not b.startup99(command, tendon, **target)["passed"]
