"""E311 tiny literal algebra and fake publication; no actual saved values."""

import copy
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

from tools import analyze_c_banc_m3_decoder_obstruction as d


def sparse_arrays(matrix, prefix):
    matrix = matrix.tocsc()
    return {
        prefix + "_data": matrix.data.copy(),
        prefix + "_indices": matrix.indices.copy(),
        prefix + "_indptr": matrix.indptr.copy(),
        prefix + "_shape": np.array(matrix.shape, np.int64),
    }


@pytest.fixture
def case():
    # Two independent free motors in muscle0; singleton fixed muscle1.
    mapping = {
        "motor_ids": np.array([30, 10, 20], np.int64),
        "membership": np.array([0, 0, 1], np.int64),
        "pool": np.array([[0.5, 0.5, 0], [0, 0, 1]], np.float64),
        "low": np.array([[0, 0.5, 0], [0, 0.5, 0], [0, 1, 0]], np.float64),
        "high": np.array([[0, 0.5, 0], [0, 0.5, 0], [0, 1, 0]], np.float64),
        "kappa": np.ones(3, np.float64),
        "fixed": np.array([False, True]),
        "moment": np.array([[0, 1], [1, -1]], np.float64),
        "rest": np.zeros(2, np.float64),
        "ltm": np.array([1], np.int64),
        "leg_motor": np.zeros(3, np.int64),
        "leg_muscle": np.zeros(2, np.int64),
        "target_startup_command": np.zeros((99, 2), np.float64),
    }
    axes = {"q": np.zeros(2, np.int64), "ltm": np.zeros(1, np.int64)}
    startup = {
        "bank": np.ones((101, 3, 3), np.float64),
        "rate": np.zeros((101, 3), np.float64),
        "original_activation": np.tile(np.array([0, 3.0], np.float64), (101, 1)),
        "q": np.zeros((101, 2), np.float64),
        "ltm": np.zeros((101, 1), np.float64),
    }
    free, var, fixed = d.layout(mapping, axes)
    design, offset, _activation, q, _ = d.reconstruct(
        mapping,
        startup["bank"][1:100],
        startup["original_activation"][1:100],
        mapping["target_startup_command"],
    )
    startup["q"][1:100] = q
    A, b, dims, envelope = d.original_equations(design, offset, mapping, free, var)
    problem = {
        **sparse_arrays(A, "A"),
        **sparse_arrays(design, "startup_design"),
        "b": b,
        "startup_error_offset": offset.ravel(),
        "free_motor": free,
        "variable_muscle": var,
        "fixed_muscle": fixed,
        "cone_dimensions": np.array(dims, np.int64),
    }
    y = np.zeros(len(b), np.float64)
    y[envelope] = 1
    # fixed-only row q0 has error .06 > .03838 and exact zero A column.
    return mapping, startup, problem, y, axes


def test_fixed_only_row_box_exclusion_and_scope(case):
    summary, raw = d.compute(*case)
    assert summary["run_valid"] and summary["activation_box_exclusion_supported"]
    assert summary["old_ray"]["bounded_affine_infeasibility_supported"]
    assert summary["projected_ray"]["activation_box_excluded"]
    assert raw["anatomical_variable_support"].tolist() == [[False], [True]]
    row = summary["all_direct_violation_rows"][0]
    assert (
        row["control_tick"] == 1
        and row["joint"] == 0
        and row["fixed_muscles"][0]["motor_ids"] == [20]
    )
    assert row["anatomical_variable_support_zero"]
    assert summary["no_controller_generated"] and "E150/SSP" in summary["limits"]
    assert (
        summary["conservative_optimistic_rms_lower_bound"]
        <= summary["raw_optimistic_rms_lower_bound"]
    )


def test_initial_anchor_not_reconstructed(case):
    case[1]["q"][0] = 99
    summary, raw = d.compute(*case)
    assert summary["run_valid"] and np.array_equal(raw["initial_q_anchor"], [99, 99])
    assert raw["old_q_residual"].shape == (99, 2)


def test_dual_mass_ties_use_row_index_order(case):
    summary, raw = d.compute(*case)
    rows = [r["row_index"] for r in summary["top20_dual_mass_rows"]]
    assert rows == list(range(20))
    assert raw["dual_peak_mass"][0, 0] == 1


def test_reconstruction_mismatch_retains_residual_invalid(case):
    case[1]["q"][3, 1] += 0.01
    summary, raw = d.compute(*case)
    assert not summary["run_valid"] and summary["outcome"] == "invalid"
    assert summary["checks"]["old_q"]["max_abs"] > 0.009
    assert raw["old_q_residual"][2, 1] < -0.009


def test_actual_zero_bank_distinct_from_no_anatomical_support(case):
    mapping, startup, _, _, axes = case
    startup["bank"][:, :2] = 0
    free, var, _fixed = d.layout(mapping, axes)
    design, offset, _, q, _ = d.reconstruct(
        mapping,
        startup["bank"][1:100],
        startup["original_activation"][1:100],
        mapping["target_startup_command"],
    )
    startup["q"][1:100] = q
    A, b, _dims, envelope = d.original_equations(design, offset, mapping, free, var)
    p = {
        **case[2],
        **sparse_arrays(A, "A"),
        **sparse_arrays(design, "startup_design"),
        "b": b,
        "startup_error_offset": offset.ravel(),
    }
    y = np.zeros(len(b))
    y[envelope] = 1
    summary, raw = d.compute(mapping, startup, p, y, axes)
    assert summary["run_valid"] and not raw["instant_adjustable_support"].any()
    assert raw["anatomical_variable_support"][1, 0] and raw["U"][0, 0] == 0
    row = next(r for r in summary["all_direct_violation_rows"] if r["joint"] == 1)
    assert (
        not row["anatomical_variable_support_zero"]
        and row["instantaneous_adjustment_zero"]
    )


def test_signed_box_intervals_literal_corners():
    offset = np.array([[0.02, -0.1]], np.float64)
    moment = np.array([[2, -3], [-1, 4]], np.float64)
    upper = np.array([[0.5, 0.7]], np.float64)
    bounds, rms = d.activation_bounds(offset, moment, upper)
    corners = np.array(
        [
            offset[0] + 0.02 * moment @ np.array(x)
            for x in ((0, 0), (0.5, 0), (0, 0.7), (0.5, 0.7))
        ]
    )
    np.testing.assert_allclose(bounds["lower"][0], corners.min(0), rtol=0, atol=1e-17)
    np.testing.assert_allclose(bounds["upper"][0], corners.max(0), rtol=0, atol=1e-17)
    assert rms >= 0 and np.all(bounds["allowance"] > 0)


def test_relaxation_allows_independent_time_activation_and_contains_original(case):
    mapping, startup = case[:2]
    _, _, activation, _, _ = d.reconstruct(
        mapping,
        startup["bank"][1:100],
        startup["original_activation"][1:100],
        mapping["target_startup_command"],
    )
    upper = 2 * startup["bank"][1:100, :2].max(axis=(1, 2))
    assert np.all((activation[:, 0] >= 0) & (activation[:, 0] <= upper))
    # Separate time rows may use both extremes; there is no shared x constraint.
    bounds, _ = d.activation_bounds(
        np.zeros((2, 1)), np.ones((1, 1)), np.array([[0.0], [2.0]])
    )
    assert bounds["upper"].ravel().tolist() == [0, 0.04]


@pytest.mark.parametrize("valid", [False, True])
def test_projected_dual_soc_vector_minus_sign_and_box_support(valid):
    offset = np.array([[0.3]], np.float64)
    moment = np.array([[2, -3]], np.float64)
    upper = np.array([[0.1, 0.2]], np.float64)
    positive = np.array([[2.0]], np.float64)
    negative = np.array([[0.5]], np.float64)
    soc = np.array([[0.2]], np.float64)
    summary, raw = d.projected_ray(
        offset, moment, upper, positive, negative, 1.0, soc, valid
    )
    w = 2 - 0.5 - 0.2
    c = 0.02 * w * moment[0]
    expected = d.Q_PEAK * 2.5 + d.Q_RMS - w * 0.3 + np.sum(upper[0] * np.maximum(-c, 0))
    assert summary["max_possible_weighted_slack"] == pytest.approx(expected, abs=1e-16)
    assert raw["projected_w"][0, 0] == w and raw["activation_c"][0, 1] < 0
    assert summary["activation_box_excluded"] is valid
    assert summary["envelope_discarded"] and not summary["feasibility_proven"]


@pytest.mark.parametrize(
    "epsilon,excluded", [(1e-12, False), (1e-5, True), (-1e-3, False)]
)
def test_projected_upper_allowance_and_inconclusive(epsilon, excluded):
    summary, _ = d.projected_ray(
        np.array([[d.Q_PEAK + epsilon]]),
        np.zeros((1, 1)),
        np.zeros((1, 1)),
        np.ones((1, 1)),
        np.zeros((1, 1)),
        0.0,
        np.zeros((1, 1)),
        True,
    )
    assert summary["activation_box_excluded"] is excluded
    assert not summary["feasibility_proven"]


def test_old_ray_allowance_uses_absolute_A_before_cancellation():
    A = sparse.csc_matrix(np.array([[1.0], [-1.0], [0.0], [0.0]]))
    b = np.array([-0.1, -0.1, 0.0, 0.0])
    y = np.array([1.0, 1.0, 0.0, 0.0])
    ray, aty = d.old_ray(A, b, y, (2, 2))
    assert aty[0] == 0 and ray["max_possible_weighted_slack"] == -0.2
    assert ray["conservative_rounding_allowance"] == pytest.approx(
        1e-10 * (1 + 0.2 + 4)
    )


@pytest.mark.parametrize(
    "defect",
    [
        "negative_bank",
        "dtype",
        "membership",
        "fixed",
        "leg",
        "row",
        "cone",
        "nonfinite_y",
        "csc",
    ],
)
def test_bad_shapes_support_raw_and_cones_rejected(case, defect):
    mapping, startup, p, y, axes = case
    if defect == "negative_bank":
        startup["bank"][2, 0, 0] = -1
    if defect == "dtype":
        startup["rate"] = startup["rate"].astype(np.float32)
    if defect == "membership":
        mapping["membership"][0] = 1
    if defect == "fixed":
        mapping["fixed"][0] = True
    if defect == "leg":
        axes["q"][0] = 1
    if defect == "row":
        p["startup_error_offset"] = p["startup_error_offset"][:-1]
    if defect == "cone":
        p["cone_dimensions"][0] += 1
    if defect == "nonfinite_y":
        y[0] = np.inf
    if defect == "csc":
        p["A_indptr"][0] = 1
    with pytest.raises(ValueError):
        d.compute(*case)


def test_wrong_saved_ray_rejected(case):
    A = d.csc(case[2], "A")
    expected, _ = d.old_ray(A, case[2]["b"], case[3], tuple(case[2]["cone_dimensions"]))
    expected["b_dot_y"] += 1
    with pytest.raises(ValueError, match="arithmetic"):
        d.compute(*case, ray_audit=expected)


def test_negative_dual_never_supports_exclusion(case):
    case[3][0] = -1
    summary, _ = d.compute(*case)
    assert not summary["run_valid"] and not summary["projected_ray"]["dual_cone_valid"]
    assert not summary["activation_box_exclusion_supported"]


def test_input_immutability(case):
    previous = copy.deepcopy(case)
    d.compute(*case)
    for old, new in zip(previous, case, strict=True):
        if isinstance(old, dict):
            for key, v in old.items():
                assert np.array_equal(v, new[key])
        else:
            assert np.array_equal(old, new)


def test_selected_npz_never_accesses_teacher_or_P(tmp_path, monkeypatch):
    path = tmp_path / "input.npz"
    np.savez(path, safe=np.array([1.0]), train=np.ones(2), P=np.ones(3))
    calls = []
    original = np.lib.npyio.NpzFile.__getitem__

    def get(self, key):
        calls.append(key)
        return original(self, key)

    monkeypatch.setattr(np.lib.npyio.NpzFile, "__getitem__", get)
    assert d.selected_npz(path, ("safe",))["safe"][0] == 1
    assert calls == ["safe"]


def fake_run(tmp_path, case, **overrides):
    ray_path = tmp_path / d.INPUT_PATHS["ray_audit"]
    ray_path.parent.mkdir(parents=True)
    d.write_json(ray_path, {})
    freeze = {
        "approved_sources": {},
        "inherited_sources": {},
        "design": {"path": d.DESIGN, "sha256": d.DESIGN_SHA},
        "packages": d.packages(),
        "input_sha256": {},
    }
    A = d.csc(case[2], "A")
    ray, _ = d.old_ray(A, case[2]["b"], case[3], tuple(case[2]["cone_dimensions"]))
    d.write_json(ray_path, ray)
    kwargs = {
        "root": tmp_path,
        "gate_fn": lambda *args, **kwargs: freeze,
        "loader": lambda *args, **kwargs: case,
        "rss": lambda: 1,
    }
    kwargs.update(overrides)
    output = tmp_path / d.OUTPUT_PATH
    return lambda: d.run(
        tmp_path / "freeze", "0" * 64, tmp_path / "review", "0" * 64, output, **kwargs
    ), output


def test_fake_full_run_sealed_and_unpromoted(tmp_path, case):
    run, output = fake_run(tmp_path, case)
    result = run()
    assert (
        d.verify_publication(output)["run_valid"]
        and result["cost"]["compute_completed"] == 1
    )
    assert (
        not result["candidate_generated"]
        and not result["candidate_adopted"]
        and not result["F1_F8_complete"]
    )
    assert all(
        result["cost"][k] == 0
        for k in ("model_calls", "fit_calls", "solver_calls", "qr_calls")
    )


@pytest.mark.parametrize(
    "name",
    [
        "source-manifest.json",
        "input-receipt.json",
        "summary.json",
        "result.json",
        "result-sha256.json",
        "output-manifest.json",
    ],
)
def test_storage_primary_identity_and_invalidation(tmp_path, case, name):
    primary = OSError(name)

    def writer(path, value):
        if path.name == name:
            raise primary
        d.write_json(path, value)

    run, output = fake_run(tmp_path, case, writer=writer)
    with pytest.raises(OSError) as caught:
        run()
    assert caught.value is primary and primary.partial_result["run_valid"] is False
    assert primary.cost_receipt["gate_completed"] == 1
    assert primary.cost_receipt["compute_completed"] == int(
        name not in ("source-manifest.json", "input-receipt.json")
    )
    with pytest.raises(ValueError, match="invalid"):
        d.verify_publication(output)


def test_secondary_storage_and_locked_seal_cannot_restore_success(
    tmp_path, case, monkeypatch
):
    primary = OSError("late publication")
    secondary = OSError("blocked")

    def writer(path, value):
        if (
            path.name in ("failure.json", "result.json")
            and (path.parent / "invalidation.json").exists()
        ):
            raise secondary
        d.write_json(path, value)
        if path.name == "output-manifest.json":
            raise primary

    original = Path.unlink

    def unlink(path, *args, **kwargs):
        if path.name in ("result-sha256.json", "output-manifest.json"):
            raise secondary
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", unlink)
    run, output = fake_run(tmp_path, case, writer=writer)
    with pytest.raises(OSError) as caught:
        run()
    assert caught.value is primary and len(primary.__notes__) >= 4
    assert (output / "invalidation.json").exists() and primary.partial_raw[
        "U"
    ].shape == (99, 1)
    with pytest.raises(ValueError):
        d.verify_publication(output)


def test_load_failure_no_compute_and_original_exception(tmp_path, case):
    primary = RuntimeError("load")

    def loader(*args, **kwargs):
        raise primary

    run, _ = fake_run(tmp_path, case, loader=loader)
    with pytest.raises(RuntimeError) as caught:
        run()
    assert caught.value is primary and primary.cost_receipt["load_requested"] == 1
    assert (
        primary.cost_receipt["load_completed"]
        == primary.cost_receipt["compute_requested"]
        == 0
    )


def test_deadline_and_source_gate_no_retry(tmp_path, case):
    times = iter([0, 16, 16])
    run, _ = fake_run(tmp_path, case, wall_clock=lambda: next(times))
    with pytest.raises(ValueError) as caught:
        run()
    assert (
        caught.value.cost_receipt["load_requested"] == 0
        and caught.value.cost_receipt["retries"] == 0
    )


@pytest.mark.parametrize("raw_save_fails", [False, True])
def test_completed_compute_deadline_preserves_disk_raw_and_primary(
    tmp_path, case, monkeypatch, raw_save_fails
):
    clock = [0.0]
    secondary = OSError("partial raw locked")
    original = np.savez_compressed

    def save(path, **arrays):
        if Path(path).name == "partial-obstruction.npz" and raw_save_fails:
            raise secondary
        original(path, **arrays)

    monkeypatch.setattr(np, "savez_compressed", save)

    def compute(*args, **kwargs):
        result = d.compute(*args, **kwargs)
        clock[0] = 16.0
        return result

    run, output = fake_run(
        tmp_path, case, compute_fn=compute, wall_clock=lambda: clock[0]
    )
    with pytest.raises(ValueError, match="internal deadline") as caught:
        run()
    primary = caught.value
    assert primary.cost_receipt["compute_completed"] == 1
    assert primary.cost_receipt["analysis_completed"] == 0
    assert primary.partial_raw["U"].shape == (99, 1)
    assert not (output / "obstruction.npz").exists()
    if raw_save_fails:
        assert not (output / "partial-obstruction.npz").exists()
        assert any(repr(secondary) in note for note in primary.__notes__)
    else:
        with np.load(output / "partial-obstruction.npz", allow_pickle=False) as raw:
            np.testing.assert_array_equal(raw["U"], primary.partial_raw["U"])
        failure = d.read_json(output / "failure.json")
        assert failure["partial_raw_path"] == "partial-obstruction.npz"
        assert failure["partial_raw_semantically_valid"] is False
    with pytest.raises(ValueError, match="invalid"):
        d.verify_publication(output)


@pytest.mark.parametrize("end_time", [16.0, 19.0, 20.0])
def test_finalization_has_separate_five_second_reserve(tmp_path, case, end_time):
    clock = [0.0]

    def compute(*args, **kwargs):
        result = d.compute(*args, **kwargs)
        clock[0] = 14.0
        return result

    def writer(path, value):
        d.write_json(path, value)
        if path.name == "output-manifest.json":
            clock[0] = end_time

    run, output = fake_run(
        tmp_path,
        case,
        compute_fn=compute,
        writer=writer,
        wall_clock=lambda: clock[0],
    )
    if end_time <= 19:
        assert run()["run_valid"] and d.verify_publication(output)["run_valid"]
    else:
        with pytest.raises(ValueError, match="Finalization") as caught:
            run()
        assert caught.value.cost_receipt["compute_completed"] == 1
        with pytest.raises(ValueError, match="invalid"):
            d.verify_publication(output)


def test_gate_rejects_contract_or_hash_before_load(tmp_path):
    freeze = tmp_path / "freeze.json"
    review = tmp_path / "review.json"
    d.write_json(freeze, {})
    d.write_json(review, {})
    with pytest.raises(ValueError, match="SHA"):
        d.gate(freeze, "0" * 64, review, "0" * 64, root=tmp_path)
    with pytest.raises(ValueError, match="contract"):
        d.gate(freeze, d.sha256(freeze), review, d.sha256(review), root=tmp_path)


def test_public_contract_and_forbidden_imports():
    import ast

    assert d.INPUT_PATHS["e287_freeze"].endswith("e287-source-decoder-fit-freeze.json")
    assert (
        len(d.INPUT_PATHS) == 16
        and d.LIMITS["solver_calls"] == d.LIMITS["qr_calls"] == 0
    )
    assert 0 < d.rss_bytes() < d.LIMITS["host_bytes"]
    tree = ast.parse(Path(d.__file__).read_text())
    imports = [
        node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    ]
    imports += [
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    ]
    assert not any(
        v and v.startswith(("torch", "cupy", "flylab", "clarabel")) for v in imports
    )
