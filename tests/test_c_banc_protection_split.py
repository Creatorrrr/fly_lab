"""Generated-only checks for E174; no original NPZ value is opened."""

from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from tools import diagnose_c_banc_m3_protection_split as e174

ROOT = Path(__file__).resolve().parents[1]
FROZEN = (
    ROOT
    / "verification/banc-natural-gait-m3-20260919/e174-protection-split-freeze.json"
)
GENERATOR = (
    ROOT / "verification/banc-natural-gait-m3-20260919/parent-e174-generated-fixture.py"
)


def generated(case: str = "feasible") -> tuple[dict, dict]:
    # Independent scalar row builder; it reads only the E174 JSON metadata.
    spec = importlib.util.spec_from_file_location("e174_scalar_fixture", GENERATOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    freeze = json.loads(FROZEN.read_text(encoding="utf-8"))
    saved, _ = module.build(freeze, case)
    return saved, freeze


def test_sparse_split_noncontiguous_columns_multi_soc_and_true_relaxation() -> None:
    saved, freeze = generated()
    e174.validate_saved(saved, freeze)
    startup = e174.build_problem(saved, "startup", freeze)
    muscle = e174.build_problem(saved, "muscle", freeze)
    assert startup["A"].shape == (16252, 1981)
    assert startup["linear_rows"] == 12278
    assert startup["soc_dims"] == [1987, 1987]
    assert muscle["A"].shape == (14353, 3961)
    assert muscle["linear_rows"] == 12366
    assert muscle["soc_dims"] == [1987]
    original = e174.sparse.csc_matrix(
        (
            saved["problem_A_data"],
            saved["problem_A_indices"],
            saved["problem_A_indptr"],
        ),
        shape=tuple(saved["problem_A_shape"]),
    )
    np.testing.assert_allclose(
        startup["A"][3962:3967].toarray(),
        original[12366:12371][:, np.r_[np.arange(1980), 3960]].toarray(),
    )
    np.testing.assert_allclose(
        muscle["A"][7922:7927].toarray(), original[7922:7927].toarray()
    )
    assert startup["A"][0, 0] == -1
    assert startup["A"][1980, 1980] == -1  # t lower row
    assert startup["A"][3961, 1980] == 1  # t upper row
    assert muscle["A"][0, 0] == -1
    assert muscle["A"][3960, 3960] == -1
    for problem in (startup, muscle):
        n = problem["q"].size
        np.testing.assert_allclose(problem["true_b"][:1980] - problem["b"][:1980], 1e-7)
        np.testing.assert_allclose(
            problem["true_b"][n : n + 1980] - problem["b"][n : n + 1980], 1e-7
        )
        assert problem["true_b"][1980] == problem["b"][1980]
    np.testing.assert_allclose(
        muscle["true_b"][11882:12366] - muscle["b"][11882:12366],
        1e-12,
        atol=1e-16,
    )
    np.testing.assert_array_equal(
        startup["true_b"][3962:12278], startup["b"][3962:12278]
    )


def test_anchor_physical_bounds_and_each_soc() -> None:
    saved, freeze = generated()
    for family in freeze["family_order"]:
        problem = e174.build_problem(saved, family, freeze)
        x = e174.inner_witness(saved, problem, family, freeze)
        assert np.all(x[:1980] >= 1e-7)
        assert np.all(x[:1980] <= 2 - 1e-7)
        assert x[-1] == -0.251
        physical = e174.physical_diagnostics(saved, x[:1980], freeze)
        assert physical["mn_floor_ok"]
        assert physical["muscle_min_ok"]
        assert physical["muscle_max_ok"]
        slacks = e174.returned_slacks(problem, x)
        assert slacks["summary"]["inner_min"] >= -1e-12
        assert all(item["slack"] >= -1e-12 for item in slacks["soc"])


def test_dual_projection_is_per_cone_and_support_uses_true_boxes() -> None:
    saved, freeze = generated()
    problem = e174.build_problem(saved, "startup", freeze)
    linear = problem["linear_rows"]
    z = np.zeros(problem["b"].shape)
    z[0] = 2
    z[linear : linear + 3] = [-2, 3, 0]
    start2 = linear + problem["soc_dims"][0]
    z[start2 : start2 + 3] = [1, 0.5, 0]
    projected = e174.project_dual(z, linear, problem["soc_dims"], 1e-12)
    for start, dim in (
        (linear, problem["soc_dims"][0]),
        (start2, problem["soc_dims"][1]),
    ):
        assert projected[start] >= np.linalg.norm(projected[start + 1 : start + dim])
    assert projected[linear] != projected[start2]
    inner = e174.dual_bound(problem, projected, False, 1e-8)
    true = e174.dual_bound(problem, projected, True, 1e-8)
    np.testing.assert_allclose(inner["stationarity"], true["stationarity"])
    np.testing.assert_allclose(
        inner["box_terms"],
        np.minimum(
            inner["stationarity"] * problem["inner_lower"],
            inner["stationarity"] * problem["inner_upper"],
        ),
    )
    np.testing.assert_allclose(
        true["box_terms"],
        np.minimum(
            true["stationarity"] * problem["lower"],
            true["stationarity"] * problem["upper"],
        ),
    )
    assert true["b_dot_z"] > inner["b_dot_z"]


def test_pointwise_and_physical_witness_are_distinct() -> None:
    saved, freeze = generated("startup_excluded")
    c = saved["w0"] * 0.7
    direct = e174.startup_pointwise(saved, c, freeze)
    assert direct.shape == (4158,)
    assert np.max(np.abs(direct)) > freeze["contract"]["startup_max_abs"]
    x = np.r_[c, 0.0]
    startup, diagnostics = e174.returned_witness(saved, x, "startup", freeze)
    assert diagnostics["qr"]["startup"]["direct_pointwise_max_abs"] == pytest.approx(
        np.max(np.abs(direct))
    )
    assert not startup["protection_ok"]
    muscle, _ = e174.returned_witness(
        saved, np.r_[c, np.zeros(1980), 0.0], "muscle", freeze
    )
    assert muscle["protection_ok"]


def _synthetic_gate_root(
    tmp_path: Path, saved: dict, freeze: dict
) -> tuple[Path, Path, Path]:
    root = tmp_path / "root"
    root.mkdir()
    for name in (e174.SOURCE, e174.TESTS):
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    freeze["design"] = "synthetic-design.md"
    (root / freeze["design"]).write_text("generated-only E174 test\n", encoding="utf-8")
    freeze["design_sha256"] = e174.sha256(root / freeze["design"])
    freeze["base_freeze"] = "synthetic-base-freeze.json"
    (root / freeze["base_freeze"]).write_text(
        json.dumps({"sample_contract": freeze["sample_contract"]}), encoding="utf-8"
    )
    freeze["base_freeze_sha256"] = e174.sha256(root / freeze["base_freeze"])
    freeze["reused_sources"] = {}
    for name in (
        "verification/banc-natural-gait-m3-20260919/e169-closed-loop-recruitment-freeze.json",
        "verification/banc-natural-gait-m3-20260919/e170-history-recruitment-freeze-v2.json",
    ):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "sample_contract": freeze["sample_contract"],
                    "input_sha256": freeze["shared_input_hashes"],
                }
            ),
            encoding="utf-8",
        )
    archive = root / freeze["raw"]
    archive.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        archive,
        **{
            freeze["archive_key_map"].get(name, name): saved[name]
            for name in freeze["allowed_qr_keys"] + freeze["allowed_problem_keys"]
        },
    )
    for name in freeze["input_sha256"]:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_bytes(b"generated synthetic pin\n")
    freeze["input_sha256"] = {
        name: e174.sha256(root / name) for name in freeze["input_sha256"]
    }
    freeze["resource_limits"]["min_free_bytes"] = 0
    freeze["resource_limits"]["internal_wall_s"] = 120
    freeze["resource_limits"]["max_output_bytes"] = 167772160
    freeze_path = root / "synthetic-freeze.json"
    freeze_path.write_text(json.dumps(freeze), encoding="utf-8")
    gate_path = root / "synthetic-source-gate.json"
    gate_path.write_text(
        json.dumps(
            {
                "schema": "flylab.m3.e174.parent-source-review.v1",
                "accepts_single_protection_split_diagnostic": True,
                "freeze_sha256": e174.sha256(freeze_path),
                "design_sha256": freeze["design_sha256"],
                "source_sha256": {
                    name: e174.sha256(root / name) for name in (e174.SOURCE, e174.TESTS)
                },
            }
        ),
        encoding="utf-8",
    )
    return root, freeze_path, gate_path


def test_generated_full_shape_preflight_loader_run_and_second_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    saved, freeze = generated()
    root, freeze_path, gate_path = _synthetic_gate_root(tmp_path, saved, freeze)
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        monkeypatch.setenv(name, "1")
    with threadpool_limits(limits=1):
        pre = e174.preflight(freeze_path, gate_path, root)
        selected = e174.load_saved(pre["freeze"], root)
        assert set(selected) == set(
            freeze["allowed_qr_keys"] + freeze["allowed_problem_keys"]
        )
        e174.validate_saved(selected, freeze)
        calls = []

        class FakeSolver:
            def __init__(self, _p, q, _a, b, _cones, settings):
                calls.append(q.size)
                if len(calls) == 2:
                    raise RuntimeError("synthetic second fit failed")
                self.q, self.b, self.settings = q, b, settings

            def solve(self):
                return SimpleNamespace(
                    x=np.r_[saved["w0"], -0.251],
                    s=np.ones_like(self.b),
                    z=np.zeros_like(self.b),
                    status="Solved",
                    iterations=3,
                    solve_time=0.01,
                )

            def get_settings(self):
                return self.settings

        out = tmp_path / "generated-out"
        report = e174.run(
            freeze_path, gate_path, out, root=root, solver_factory=FakeSolver
        )
    assert calls == [1981, 3961]
    assert report["new_conic_fits"] == 2
    assert report["status"] == "invalid_not_tested"
    assert report["families"]["startup"]["status"] == "valid"
    assert report["primary_error"]["message"] == "synthetic second fit failed"
    assert (out / "startup-solver-raw.npz").exists()
    assert (out / "startup-derived.npz").exists()
    assert (out / "output-manifest.json").exists()
    with np.load(out / "startup-derived.npz") as derived:
        assert derived["startup_pointwise"].shape == (4158,)
        assert derived["startup_baseline_pointwise"].shape == (4158,)
        assert {
            f"qr_{kind}_{leg}_{which}"
            for kind in e174.KINDS
            for leg in freeze["sample_contract"]["legs"]
            for which in ("returned", "baseline")
        } <= set(derived.files)


@pytest.mark.parametrize("failure", ["baseline", "anchor"])
def test_failing_preparation_retains_problem_and_computed_anchor_without_fit(
    failure: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    saved, freeze = generated()
    if failure == "baseline":
        freeze["contract"]["muscle_lower"] = 100.0
    else:
        freeze["contract"]["inner_witness_margin"] = 0.0
    root, freeze_path, gate_path = _synthetic_gate_root(tmp_path, saved, freeze)
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        monkeypatch.setenv(name, "1")
    with threadpool_limits(limits=1):
        report = e174.run(
            freeze_path,
            gate_path,
            tmp_path / "out",
            root=root,
            solver_factory=lambda *_: pytest.fail(
                "solver entered on failed preparation"
            ),
        )
    assert report["new_conic_fits"] == 0
    assert report["status"] == "invalid_not_tested"
    assert report["families"]["startup"]["status"] == "preparation_failed"
    assert report["primary_error"]["message"] == (
        "baseline physical" if failure == "baseline" else "anchor inner feasibility"
    )
    out = tmp_path / "out"
    assert (out / "startup-A.npz").exists()
    assert (out / "startup-problem.npz").exists()
    with np.load(out / "startup-preparation-raw.npz") as prep:
        assert prep["baseline_coefficients"].shape == (1980,)
        assert prep["anchor_a"].shape == (1980,)
        assert prep["anchor_coefficients"].shape == (1980,)
        assert prep["anchor_x"].shape == (1981,)
        assert prep["anchor_inner_linear_slack"].shape == (12278,)
        assert prep["anchor_soc_slack"].shape == (2,)


def _rotate_startup_qr(saved: dict, freeze: dict) -> None:
    key = "qr_startup_lf"
    qr = saved[key].copy()
    rotation = np.array([[2**-0.5, 2**-0.5], [-(2**-0.5), 2**-0.5]])
    qr[:2] = rotation @ qr[:2].copy()
    saved[key] = qr
    A = e174.sparse.csc_matrix(
        (
            saved["problem_A_data"],
            saved["problem_A_indices"],
            saved["problem_A_indptr"],
        ),
        shape=tuple(saved["problem_A_shape"]),
    ).tolil()
    dims = saved["problem_cone_dims"]
    start = 20682 + int(np.sum(dims[1:-1]))
    limit = freeze["contract"]["startup_RMS"] * np.sqrt(99 * 42)
    active = saved[key + "_active"]
    for row in (0, 1):
        A[start + 1 + row, :1980] = 0
        A[start + 1 + row, active] = -qr[row, :-1] / limit
        saved["problem_b"][start + 1 + row] = qr[row, -1] / limit
    A = A.tocsc()
    saved["problem_A_data"], saved["problem_A_indices"], saved["problem_A_indptr"] = (
        A.data,
        A.indices,
        A.indptr,
    )


def test_rotated_qr_keeps_direct_startup_vector_and_distinct_coordinate_max(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    saved, freeze = generated()
    _rotate_startup_qr(saved, freeze)
    e174.validate_saved(saved, freeze)
    root, freeze_path, gate_path = _synthetic_gate_root(tmp_path, saved, freeze)
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        monkeypatch.setenv(name, "1")
    coefficient = 0.7 * saved["w0"]

    class FakeSolver:
        def __init__(self, _p, q, _a, b, _cones, settings):
            self.q, self.b, self.settings = q, b, settings

        def solve(self):
            x = (
                np.r_[coefficient, -0.251]
                if self.q.size == 1981
                else np.r_[
                    coefficient,
                    coefficient.reshape(990, 2).min(axis=1),
                    coefficient.reshape(990, 2).max(axis=1),
                    -0.251,
                ]
            )
            return SimpleNamespace(
                x=x,
                s=np.ones_like(self.b),
                z=np.zeros_like(self.b),
                status="Solved",
                iterations=2,
                solve_time=0.01,
            )

        def get_settings(self):
            return self.settings

    out = tmp_path / "out"
    with threadpool_limits(limits=1):
        report = e174.run(
            freeze_path, gate_path, out, root=root, solver_factory=FakeSolver
        )
    assert report["status"] == "valid_diagnostic"
    with np.load(out / "startup-derived.npz") as derived:
        direct = derived["startup_pointwise"]
        qr = derived["qr_startup_lf_returned"]
        assert direct.shape == (4158,)
        assert qr.shape[0] < direct.shape[0]
        assert not np.isclose(np.max(np.abs(direct[:2])), np.max(np.abs(qr[:2])))
        assert np.linalg.norm(direct) == pytest.approx(
            report["families"]["startup"]["startup_direct_qr_norm_check"]["qr_norm"],
            abs=1e-12,
        )


def test_raw_publication_failure_and_late_resource_failure_cancel_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    saved, freeze = generated()
    root, freeze_path, gate_path = _synthetic_gate_root(tmp_path, saved, freeze)
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        monkeypatch.setenv(name, "1")

    class FakeSolver:
        def __init__(self, _p, q, _a, b, _cones, settings):
            self.q, self.b, self.settings = q, b, settings

        def solve(self):
            c = saved["w0"]
            x = (
                np.r_[c, -0.251]
                if self.q.size == 1981
                else np.r_[
                    c,
                    c.reshape(990, 2).min(axis=1),
                    c.reshape(990, 2).max(axis=1),
                    -0.251,
                ]
            )
            return SimpleNamespace(
                x=x,
                s=np.ones_like(self.b),
                z=np.zeros_like(self.b),
                status="Solved",
                iterations=2,
                solve_time=0.01,
            )

        def get_settings(self):
            return self.settings

    original_save = np.savez_compressed
    failed = []

    def fail_raw_once(path, **values):
        if str(path).endswith("startup-solver-raw.npz") and not failed:
            failed.append(True)
            raise OSError("synthetic raw publication failed")
        return original_save(path, **values)

    monkeypatch.setattr(np, "savez_compressed", fail_raw_once)
    with threadpool_limits(limits=1):
        first = e174.run(
            freeze_path,
            gate_path,
            tmp_path / "raw-failure",
            root=root,
            solver_factory=FakeSolver,
        )
    assert first["status"] == "invalid_not_tested"
    assert first["new_conic_fits"] == 1
    assert first["primary_error"]["message"] == "synthetic raw publication failed"
    assert (tmp_path / "raw-failure/startup-solver-raw.fallback.npz").exists()
    monkeypatch.setattr(np, "savez_compressed", original_save)
    original_check = e174.resource_check
    checks = []

    def fail_after_manifest(out, started, limits):
        checks.append(1)
        if len(checks) == 7:
            raise ValueError("synthetic late resource failure")
        return original_check(out, started, limits)

    monkeypatch.setattr(e174, "resource_check", fail_after_manifest)
    with threadpool_limits(limits=1):
        late = e174.run(
            freeze_path,
            gate_path,
            tmp_path / "late-failure",
            root=root,
            solver_factory=FakeSolver,
        )
    assert len(checks) == 7
    assert late["new_conic_fits"] == 2
    assert late["status"] == "invalid_not_tested"
    assert late["primary_error"]["message"] == "synthetic late resource failure"
    persisted = json.loads(
        (tmp_path / "late-failure/result.json").read_text(encoding="utf-8")
    )
    assert persisted["status"] == "invalid_not_tested"
    manifest = json.loads(
        (tmp_path / "late-failure/output-manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["files"]["result.json"] == e174.sha256(
        tmp_path / "late-failure/result.json"
    )


def test_bad_gate_blocks_any_array_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    freeze_path = tmp_path / "freeze.json"
    gate_path = tmp_path / "gate.json"
    freeze_path.write_text('{"schema":"wrong"}', encoding="utf-8")
    gate_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        np, "load", lambda *_args, **_kwargs: pytest.fail("array opened")
    )
    with pytest.raises(ValueError, match="freeze schema"):
        e174.run(freeze_path, gate_path, tmp_path / "out", tmp_path)
    assert not (tmp_path / "out").exists()
