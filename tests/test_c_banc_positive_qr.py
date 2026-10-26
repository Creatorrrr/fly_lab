"""Generated only checks for the frozen E173 positive QR diagnosis."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "tools/diagnose_c_banc_m3_positive_qr.py"
)
SPEC = importlib.util.spec_from_file_location("e173_positive_qr", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
e173 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(e173)


def generated(family: str = "static", *, feasible: bool = True) -> tuple[dict, dict]:
    legs = ["lf", "lm", "lh", "rf", "rm", "rh"]
    ncoef = 1 if family == "static" else 2
    saved = {
        "w0": np.array([1.0]),
        "pair_muscles": np.zeros(990, dtype=np.int64),
        "pair_motors": np.zeros(990, dtype=np.int64),
    }
    for index, leg in enumerate(legs):
        active = (
            np.arange(ncoef, dtype=np.int64)
            if index == 0
            else np.array([], dtype=np.int64)
        )
        if index == 0:
            qr = (
                np.array([[1.0, -1.0]])
                if family == "static"
                else np.array([[1.0, 1.0, -1.0]])
            )
            if not feasible:
                qr[0, -1] = -10.0
        else:
            qr = np.zeros((1, 1))
        for kind in e173.KINDS:
            saved[f"qr_{kind}_{leg}"] = qr.copy()
            saved[f"qr_{kind}_{leg}_active"] = active.copy()
    # Positive B: the baseline has one residual unit in the feasible instance.
    saved["qr_command_lf"][0, -1] -= 1.0
    saved["qr_velocity_lf"][0, -1] -= 1.0
    saved["qr_startup_lf"][0, -1] -= 1.0
    freeze = {
        "sample_contract": {"legs": legs},
        "families": {family: {"coefficient_count": ncoef}},
        "contract": {
            "command_ratio": 0.8,
            "inner_guard": 1e-7,
            "coefficient_bound": 2.0,
            "inner_witness_margin": -0.251,
            "QR_absolute_atol": 1e-12,
            "QR_relative_atol": 1e-10,
            "resolution_factor": 128.0,
            "resolution_absolute_pad": 1e-12,
            "resolution_baseline_fraction": 0.001,
            "dual_projection_pad": 1e-12,
            "dual_upper_bound_pad": 1e-8,
        },
    }
    return saved, freeze


def test_sparse_scatter_soc_sign_rows_and_true_history_guards() -> None:
    saved, freeze = generated("history")
    p = e173.build_problem(saved, "history", freeze)
    assert p["M"].shape == (6, 2)
    np.testing.assert_array_equal(p["M"].toarray()[0], [1, 1])
    assert p["A"].shape == (15, 3)
    np.testing.assert_array_equal(p["A"].toarray()[:3], -np.eye(3))
    np.testing.assert_array_equal(p["A"].toarray()[3:6], np.eye(3))
    np.testing.assert_array_equal(p["A"].toarray()[6:7], [[-1, -1, 0]])
    np.testing.assert_array_equal(p["A"].toarray()[7:8], [[1, 1, 0]])
    np.testing.assert_array_equal(p["A"].toarray()[8], [0, 0, p["target"]])
    np.testing.assert_array_equal(p["A"].toarray()[9], [-1, -1, 0])
    assert p["b"][8] == pytest.approx(p["target"])
    assert p["b"][9] == pytest.approx(saved["qr_command_lf"][0, -1])
    np.testing.assert_allclose(p["true_b"][:2] - p["b"][:2], 1e-7)
    np.testing.assert_allclose(p["true_b"][3:5] - p["b"][3:5], 1e-7)
    np.testing.assert_allclose(p["true_b"][6:8] - p["b"][6:8], 1e-7)
    np.testing.assert_allclose(p["inner_lower"][:2], [1e-7] * 2)
    np.testing.assert_allclose(p["inner_upper"][:2], [2 - 1e-7] * 2)
    np.testing.assert_array_equal(p["lower"][:2], [0, 0])
    np.testing.assert_array_equal(p["upper"][:2], [2, 2])
    np.testing.assert_allclose(p["b"][:2], [-1e-7, -1e-7])
    np.testing.assert_allclose(p["b"][3:5], [2 - 1e-7] * 2)
    np.testing.assert_allclose(p["b"][6:8], [-1e-7, 2 - 1e-7])
    np.testing.assert_allclose(p["inner_lower"][-1], -1)
    np.testing.assert_allclose(p["inner_upper"][-1], 1)
    assert p["true_b"][2] == p["b"][2] == 1
    assert p["true_b"][5] == p["b"][5] == 1
    for name, c in {
        "baseline": p["baseline"],
        "sine": np.sin(np.arange(2) * 0.173),
    }.items():
        assert (
            e173.compare_local_global(saved, "history", p, c, freeze, name)[
                "max_abs_gap"
            ]
            == 0
        )


def test_known_feasible_small_cone_and_true_box() -> None:
    saved, freeze = generated("static")
    p = e173.build_problem(saved, "static", freeze)
    settings = e173.clarabel.DefaultSettings()
    settings.verbose = False
    solution = e173.clarabel.DefaultSolver(
        p["P"], p["q"], p["A"], p["b"], p["cones"], settings
    ).solve()
    assert str(solution.status) == "Solved"
    x = np.asarray(solution.x)
    assert 0 <= x[0] <= 2
    metrics, resolution = e173.qr_metrics(saved, "static", x[:-1], freeze)
    assert e173.returned_witness(p, x, metrics, resolution, "static", freeze)["valid"]


def test_true_projected_dual_restores_all_history_endpoint_guards() -> None:
    saved, freeze = generated("history")
    p = e173.build_problem(saved, "history", freeze)
    z = np.arange(p["A"].shape[0], dtype=float) / 5 - 1
    projected = e173.project_dual(z, p["linear_rows"], 1e-12)
    assert np.all(projected[: p["linear_rows"]] >= 0)
    assert projected[p["linear_rows"]] >= np.linalg.norm(
        projected[p["linear_rows"] + 1 :]
    )
    inner = e173.dual_bound(p, projected, False, 1e-8)
    true = e173.dual_bound(p, projected, True, 1e-8)
    expected = 1e-7 * sum(projected[i] for i in (0, 1, 3, 4, 6, 7))
    assert true["b_dot_z"] - inner["b_dot_z"] == pytest.approx(expected)
    np.testing.assert_allclose(true["stationarity"], inner["stationarity"])
    assert true["upper"] >= inner["upper"]


def test_inner_and_true_dual_use_separate_support_boxes() -> None:
    saved, freeze = generated("history")
    p = e173.build_problem(saved, "history", freeze)
    z = np.zeros(p["A"].shape[0])
    z[0] = 1.0
    inner = e173.dual_bound(p, z, False, 0.0)
    true = e173.dual_bound(p, z, True, 0.0)
    h = p["q"] + p["A"].T @ z
    expected_inner = np.minimum(h * p["inner_lower"], h * p["inner_upper"])
    expected_true = np.minimum(h * p["lower"], h * p["upper"])
    np.testing.assert_allclose(inner["box_terms"], expected_inner)
    np.testing.assert_allclose(true["box_terms"], expected_true)
    assert true["box_min_sum"] - inner["box_min_sum"] == pytest.approx(-1e-7)
    assert true["upper"] - inner["upper"] == pytest.approx(2e-7)


def test_per_leg_rms_uses_original_samples_across_different_qr_heights() -> None:
    static, freeze = generated("static")
    history, _ = generated("history")
    for kind in e173.KINDS:
        key = f"qr_{kind}_lf"
        history[key] = np.vstack((history[key], np.zeros_like(history[key])))
    static_metrics, _ = e173.qr_metrics(static, "static", np.array([1.0]), freeze)
    history_metrics, _ = e173.qr_metrics(
        history, "history", np.array([1.0, 0.0]), freeze
    )
    for kind, count in e173.LEG_DENOMINATORS.items():
        leg_static = static_metrics[kind]["legs"]["lf"]
        leg_history = history_metrics[kind]["legs"]["lf"]
        assert leg_static["baseline_rms"] == pytest.approx(
            leg_static["baseline_norm"] / np.sqrt(count)
        )
        assert leg_history["baseline_rms"] == pytest.approx(leg_static["baseline_rms"])
        assert leg_history["rms"] == pytest.approx(leg_static["rms"])


def test_full_solver_settings_serializes_noncallable_sequences() -> None:
    settings = e173.clarabel.DefaultSettings()
    recorded = e173.settings_record(settings)
    public = {
        key
        for key in dir(settings)
        if not key.startswith("_") and not callable(getattr(settings, key))
    }
    assert set(recorded) == public
    assert recorded["pardiso_iparm"] == list(settings.pardiso_iparm)
    assert len(recorded["pardiso_iparm"]) == 64
    assert recorded["time_limit"] == "inf"
    json.dumps(recorded, allow_nan=False)
    with pytest.raises(TypeError, match="unsupported solver setting"):
        e173.settings_record(SimpleNamespace(unsupported=object()))


def test_negative_dual_excludes_saved_family_small_cone() -> None:
    # M=0, nonzero offset: target is 80% of the fixed residual. A SOC dual
    # can certify t<0 for every true coefficient value.
    saved, freeze = generated("static")
    for kind in e173.KINDS:
        saved[f"qr_{kind}_lf"][0, 0] = 0
        saved[f"qr_{kind}_lf"][0, -1] = 1
    p = e173.build_problem(saved, "static", freeze)
    settings = e173.clarabel.DefaultSettings()
    settings.verbose = False
    solution = e173.clarabel.DefaultSolver(
        p["P"], p["q"], p["A"], p["b"], p["cones"], settings
    ).solve()
    assert str(solution.status) == "Solved"
    projected = e173.project_dual(np.asarray(solution.z), p["linear_rows"], 1e-12)
    assert e173.dual_bound(p, projected, True, 1e-8)["upper"] < 0


def test_signed_feasible_but_positive_excluded_asymmetric_case() -> None:
    saved, freeze = generated("static")
    for kind in e173.KINDS:
        saved[f"qr_{kind}_lf"][0] = [1.0, 10.0]
    p = e173.build_problem(saved, "static", freeze)
    assert p["baseline_norm"] == pytest.approx(11)
    assert p["target"] == pytest.approx(8.8)
    assert abs(-2 + 10) < p["target"]
    assert min(abs(c + 10) for c in (0, 2)) > p["target"]
    settings = e173.clarabel.DefaultSettings()
    settings.verbose = False
    solution = e173.clarabel.DefaultSolver(
        p["P"], p["q"], p["A"], p["b"], p["cones"], settings
    ).solve()
    projected = e173.project_dual(np.asarray(solution.z), p["linear_rows"], 1e-12)
    assert e173.dual_bound(p, projected, True, 1e-8)["upper"] < 0


@pytest.mark.parametrize("family,rows", [("static", 331), ("history", 661)])
def test_full_shape_noncontiguous_owner_scatter(family: str, rows: int) -> None:
    saved, freeze = generated(family)
    ncoef = 1980 if family == "static" else 3960
    freeze["families"][family]["coefficient_count"] = ncoef
    freeze["problem_shapes"] = {
        family: {
            "M_shape": [6 * rows, ncoef],
            "A_shape": [5949, 1981] if family == "static" else [15849, 3961],
            "variable_count": ncoef + 1,
            "linear_rows": 3962 if family == "static" else 11882,
            "soc_rows": 6 * rows + 1,
        }
    }
    saved["w0"] = np.full(1980, 0.5)
    for i, leg in enumerate(freeze["sample_contract"]["legs"]):
        active = np.array([2 + i, ncoef - 1 - i])
        qr = np.tile(np.array([0.25, -0.5, 0.75]), (rows, 1))
        for kind in e173.KINDS:
            saved[f"qr_{kind}_{leg}"] = qr.copy()
            saved[f"qr_{kind}_{leg}_active"] = active.copy()
    p = e173.build_problem(saved, family, freeze)
    assert p["M"].shape == (6 * rows, ncoef)
    assert p["M"][0, 2] == pytest.approx(0.25)
    assert p["M"][0, ncoef - 1] == pytest.approx(-0.5)
    assert p["M"][rows, 3] == pytest.approx(0.25)
    witness = e173.inner_witness(p, freeze)
    assert np.all(witness[:-1] >= p["inner_lower"][:-1])
    assert np.all(witness[:-1] <= p["inner_upper"][:-1])
    e173.compare_local_global(saved, family, p, witness[:-1], freeze, "inner")


def test_wrong_gate_fails_before_any_array_load(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    freeze = tmp_path / "freeze.json"
    gate = tmp_path / "gate.json"
    freeze.write_text(json.dumps({"schema": "wrong"}), encoding="utf-8")
    gate.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        np, "load", lambda *_args, **_kwargs: pytest.fail("original array opened")
    )
    with pytest.raises(ValueError, match="freeze schema"):
        e173.run(freeze, gate, tmp_path / "out", tmp_path)
    assert not (tmp_path / "out").exists()


def test_bare_return_caller_owned_and_first_save_fallback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    x = [1.0, 0.0]
    solution = SimpleNamespace(
        x=x, s=[2.0], z=[3.0], status="Solved", iterations=4, solve_time=0.01
    )
    raw = e173.bare_return(solution)
    x[0] = 999
    assert raw["x"][0] == 1
    original = np.savez_compressed
    calls = []

    def failing_once(path: Path, **values: np.ndarray) -> None:
        calls.append(path)
        if len(calls) == 1:
            raise OSError("first publication failed")
        original(path, **values)

    monkeypatch.setattr(np, "savez_compressed", failing_once)
    report = {"families": {}, "secondary_publication_errors": []}
    with pytest.raises(OSError, match="first publication failed"):
        e173.save_raw_first(tmp_path, "static", raw, report)
    assert len(calls) == 2
    with np.load(tmp_path / "static-solver-raw.fallback.npz") as archive:
        assert archive["x"][0] == 1
        assert str(archive["status"]) == "Solved"


def _run_generated(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, factory: object
) -> tuple[dict, Path]:
    static, static_freeze = generated("static")
    history, _ = generated("history")
    selected = {"static": static, "history": history}
    freeze = {
        **static_freeze,
        "families": {
            "static": {"coefficient_count": 1, "raw": "static"},
            "history": {"coefficient_count": 2, "raw": "history"},
        },
        "family_order": ["static", "history"],
        "allowed_keys": [],
        "archive_key_map": {},
        "input_sha256": {"static": "generated-static", "history": "generated-history"},
        "resource_limits": {
            "min_free_bytes": 0,
            "internal_wall_s": 1000,
            "max_output_bytes": 10**8,
            "family_fit_wall_s": 100,
            "total_fit_wall_s": 200,
        },
        "solver_settings": {"verbose": False},
        "allowed_solver_statuses": ["Solved", "AlmostSolved"],
        "design_sha256": "generated",
        "reused_helper_sha256": "generated",
        "reused_bounded_helper_sha256": "generated",
    }
    helper = SimpleNamespace(
        read_selected=lambda path, *_: selected[path.name],
        validate_saved=lambda *_: {"variable_muscles": [0]},
        coefficient_diagnostics=lambda *_: {},
    )
    monkeypatch.setattr(
        e173,
        "preflight",
        lambda *_: {
            "freeze": freeze,
            "helper": helper,
            "bounded": e173.bounded_module(),
            "threadpool_info": [],
        },
    )
    freeze_path, gate_path, out = (
        tmp_path / "freeze.json",
        tmp_path / "gate.json",
        tmp_path / "out",
    )
    freeze_path.write_text("{}", encoding="utf-8")
    gate_path.write_text("{}", encoding="utf-8")
    report = e173.run(freeze_path, gate_path, out, solver_factory=factory)
    return report, out


def test_at_most_two_fits_and_second_failure_keeps_first(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls = []

    def factory(*args: object) -> object:
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("second family unavailable")
        return e173.clarabel.DefaultSolver(*args)

    report, out = _run_generated(monkeypatch, tmp_path, factory)
    assert len(calls) == report["new_conic_fits"] == 2
    assert report["status"] == "invalid_not_tested"
    assert report["primary_error"]["message"] == "second family unavailable"
    assert report["families"]["static"]["status"] == "valid"
    assert (out / "static-solver-raw.npz").exists()
    assert (out / "static-derived.npz").exists()
    with np.load(out / "static-derived.npz") as derived:
        assert derived["returned_inner_linear_slack"].shape == (4,)
        assert derived["returned_true_linear_slack"].shape == (4,)
        assert np.isfinite(derived["returned_soc_slack"])
    assert "returned_slacks" in report["families"]["static"]


def test_get_settings_failure_keeps_solver_return(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls = []

    class Wrapper:
        def __init__(self, *args: object) -> None:
            self.solver = e173.clarabel.DefaultSolver(*args)

        def solve(self) -> object:
            calls.append(1)
            return self.solver.solve()

        def get_settings(self) -> None:
            raise RuntimeError("settings retrieval failed")

    report, out = _run_generated(monkeypatch, tmp_path, Wrapper)
    assert calls == [1]
    assert report["new_conic_fits"] == 1
    assert report["primary_error"]["message"] == "settings retrieval failed"
    with np.load(out / "static-solver-raw.npz") as archive:
        assert {"x", "s", "z", "status", "iterations", "solve_time"} <= set(
            archive.files
        )


def test_true_witness_rejects_endpoint_outside_box() -> None:
    saved, freeze = generated("history")
    p = e173.build_problem(saved, "history", freeze)
    metrics = {"command": {"rms": 0.0, "baseline_rms": 1.0}}
    bad = np.array([2.0, 1.0, 0.0])
    result = e173.returned_witness(p, bad, metrics, [0.0] * 6, "history", freeze)
    assert result["coefficient_box"]
    assert not result["history_endpoint_box"]
    assert not result["valid"]


def test_exact_negative_and_positive_upper_are_rejected() -> None:
    saved, freeze = generated("history")
    p = e173.build_problem(saved, "history", freeze)
    metrics = {"command": {"rms": 0.0, "baseline_rms": 1.0}}
    for x in (np.array([-1e-15, 0.0, 0.0]), np.array([0.5, -1e-15, 0.0])):
        result = e173.returned_witness(p, x, metrics, [0.0] * 6, "history", freeze)
        assert not result["coefficient_box"]
        assert not result["valid"]
    upper = e173.returned_witness(
        p, np.array([1.5, 0.6, 0.0]), metrics, [0.0] * 6, "history", freeze
    )
    assert upper["coefficient_box"]
    assert not upper["history_endpoint_box"]


def test_true_witness_keeps_inner_lower_coefficient_violation_separate() -> None:
    saved, freeze = generated("static")
    saved["qr_command_lf"][0] = [1.0, 0.0]
    p = e173.build_problem(saved, "static", freeze)
    x = np.array([0.0, 0.0])
    metrics, resolutions = e173.qr_metrics(saved, "static", x[:-1], freeze)
    assert e173.returned_witness(p, x, metrics, resolutions, "static", freeze)["valid"]
    diagnostics = e173.returned_slacks(p, x, "static")
    assert diagnostics["summary"]["inner"]["min_slack"] == pytest.approx(-1e-7)
    assert diagnostics["summary"]["inner"]["global_row"] == 0
    assert (
        diagnostics["summary"]["inner"]["blocks"]["negative_coefficient"]["global_row"]
        == 0
    )
    assert diagnostics["summary"]["true"]["min_slack"] == pytest.approx(0)
    assert diagnostics["summary"]["soc"]["slack"] > 0


def test_history_endpoint_slack_has_distinct_global_row_and_true_bound() -> None:
    saved, freeze = generated("history")
    p = e173.build_problem(saved, "history", freeze)
    x = np.array([1.5, 0.5, 0.0])
    metrics, resolutions = e173.qr_metrics(saved, "history", x[:-1], freeze)
    assert e173.returned_witness(p, x, metrics, resolutions, "history", freeze)["valid"]
    diagnostics = e173.returned_slacks(p, x, "history")
    endpoint = diagnostics["summary"]["inner"]["blocks"]["positive_endpoint"]
    assert endpoint["global_row"] == 7
    assert endpoint["min_slack"] == pytest.approx(-1e-7)
    assert diagnostics["summary"]["true"]["blocks"]["positive_endpoint"][
        "min_slack"
    ] == pytest.approx(0)
    assert diagnostics["summary"]["inner"]["blocks"]["negative_t"]["global_row"] == 2
    assert diagnostics["summary"]["inner"]["blocks"]["positive_t"]["global_row"] == 5
    invalid = np.array([1.5, 0.500001, 0.0])
    bad = e173.returned_slacks(p, invalid, "history")
    assert bad["summary"]["true"]["blocks"]["positive_endpoint"]["min_slack"] < 0
    bad_metrics, bad_resolution = e173.qr_metrics(
        saved, "history", invalid[:-1], freeze
    )
    assert not e173.returned_witness(
        p, invalid, bad_metrics, bad_resolution, "history", freeze
    )["valid"]
