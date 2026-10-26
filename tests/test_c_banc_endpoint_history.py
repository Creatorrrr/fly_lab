"""Generated E175 scalar checks. Original pre-solve NPZ values are never read."""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import sparse
from threadpoolctl import threadpool_limits

from tools import diagnose_c_banc_m3_endpoint_history as e175

ROOT = Path(__file__).resolve().parents[1]
FROZEN = (
    ROOT
    / "verification/banc-natural-gait-m3-20260919/e175-endpoint-history-freeze.json"
)
GENERATOR = (
    ROOT / "verification/banc-natural-gait-m3-20260919/parent-e175-generated-fixture.py"
)


def generated(case: str = "feasible") -> tuple[dict, dict, dict]:
    spec = importlib.util.spec_from_file_location("e175_scalar_fixture", GENERATOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    freeze = json.loads(FROZEN.read_text(encoding="utf-8"))
    saved, oracle = module.build(freeze, case)
    for name, value in saved.items():
        if name in freeze["selected_header_metadata"]:
            freeze["selected_header_metadata"][name] = {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "fortran_order": False,
            }
    return saved, freeze, oracle


def test_sparse_endpoint_problem_keeps_every_original_protection() -> None:
    saved, freeze, _ = generated()
    validation = e175.validate_saved(saved, freeze)
    problem = e175.build_problem(saved, freeze, validation)
    old = validation["A"]
    T = problem["T"]
    assert problem["A"].shape[1] == 5941
    assert problem["linear_rows"] == 28602
    assert len(problem["soc_dims"]) == 15
    assert sum(problem["soc_dims"]) + 28602 == problem["A"].shape[0]
    assert (problem["A"][19802:20286] - old[15842:16326] @ T).nnz == 0
    assert (problem["A"][20286:] - old[16326:] @ T).nnz == 0
    np.testing.assert_allclose(problem["true_b"][:3960] - problem["b"][:3960], 1e-7)
    np.testing.assert_allclose(
        problem["true_b"][5941:9901] - problem["b"][5941:9901], 1e-7
    )
    np.testing.assert_allclose(
        problem["true_b"][11882:19802] - problem["b"][11882:19802], 1e-8
    )
    np.testing.assert_allclose(
        problem["true_b"][19802:20286] - problem["b"][19802:20286], 1e-7 + 1e-12
    )
    np.testing.assert_array_equal(problem["true_b"][20286:], problem["b"][20286:])
    x = np.r_[np.ones(1980), np.full(1980, 0.25), np.zeros(1980), 0.0]
    np.testing.assert_allclose(
        T @ x, np.r_[np.ones(1980), np.full(1980, -0.75), np.zeros(1980), 0.0]
    )


def test_four_endpoints_control_envelope_and_physical_gates() -> None:
    saved, freeze, _ = generated()
    problem = e175.build_problem(saved, freeze)
    record = {}
    x = e175.inner_witness(saved, problem, freeze, record)
    assert record["slacks"]["summary"]["inner_min"] >= -1e-12
    assert len(record["slacks"]["soc"]) == 15
    assert all(v["slack"] >= -1e-12 for v in record["slacks"]["soc"])
    assert x[-1] == -0.5
    pair = 0
    altered = x.copy()
    altered[1980 + 2 * pair] = 0.0
    altered[1981 + 2 * pair] = 2.0
    four = e175.physical(saved, altered[:1980], altered[1980:3960], freeze)
    assert four["pair_min"][pair] == 0.0
    assert four["pair_max"][pair] == 2.0
    assert not (four["mn_floor_ok"] and four["muscle_min_ok"] and four["muscle_max_ok"])
    envelope = problem["A"][11882:19802] @ altered
    assert np.max(envelope - problem["b"][11882:19802]) > 0


@pytest.mark.parametrize("case", ["feasible", "startup_excluded"])
def test_returned_endpoint_witness_uses_direct_startup_and_all_limits(
    case: str,
) -> None:
    saved, freeze, oracle = generated(case)
    e175.validate_saved(saved, freeze)
    a, b = oracle["truth"][:1980], oracle["truth"][1980:]
    if case == "startup_excluded":
        a = b.copy()  # Satisfy the command target while violating startup.
    pair = np.stack((a.reshape(990, 2), b.reshape(990, 2)))
    x = np.r_[a, b, pair.min(axis=(0, 2)) - 1e-8, pair.max(axis=(0, 2)) + 1e-8, 0.0]
    result, diagnostics = e175.returned_witness(saved, x, freeze)
    assert result["valid"] == oracle["expected_witness"]
    direct = e175.startup_pointwise(saved, x[:3960], freeze)
    np.testing.assert_allclose(diagnostics["startup_direct"], direct)
    assert np.linalg.norm(direct) == pytest.approx(
        diagnostics["metrics"]["startup"]["rms"] * np.sqrt(4158), abs=1e-12
    )
    if case == "startup_excluded":
        assert result["startup_max_abs"] > freeze["contract"]["startup_max_abs"]


def test_bad_gate_blocks_all_array_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    freeze = tmp_path / "freeze.json"
    gate = tmp_path / "gate.json"
    freeze.write_text('{"schema":"wrong"}', encoding="utf-8")
    gate.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        np, "load", lambda *_args, **_kwargs: pytest.fail("array loaded")
    )
    with pytest.raises(ValueError, match="freeze schema"):
        e175.run(freeze, gate, tmp_path / "out", tmp_path)
    assert not (tmp_path / "out").exists()


def test_direct_cli_help_without_pythonpath(tmp_path: Path) -> None:
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    result = subprocess.run(
        [
            str(ROOT / ".venv/Scripts/python.exe"),
            "-X",
            "utf8",
            str(ROOT / e175.SOURCE),
            "--help",
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--source-gate" in result.stdout


def test_elementwise_physical_and_soc_validation_catches_small_corruption() -> None:
    saved, freeze, _ = generated()
    changed = {k: v.copy() for k, v in saved.items()}
    changed["problem_b"][15842] += 1e-10
    with pytest.raises(ValueError, match="original physical RHS"):
        e175.validate_saved(changed, freeze)
    changed = {k: v.copy() for k, v in saved.items()}
    A = sparse.csc_matrix(
        (
            changed["problem_A_data"],
            changed["problem_A_indices"],
            changed["problem_A_indptr"],
        ),
        shape=tuple(changed["problem_A_shape"]),
    ).tolil()
    A[24643, 10] += (
        2e-12  # A zero reference must get only the frozen absolute tolerance.
    )
    A = A.tocsc()
    (
        changed["problem_A_data"],
        changed["problem_A_indices"],
        changed["problem_A_indptr"],
    ) = A.data, A.indices, A.indptr
    freeze["selected_header_metadata"]["problem_A_data"]["shape"] = list(A.data.shape)
    freeze["selected_header_metadata"]["problem_A_indices"]["shape"] = list(
        A.indices.shape
    )
    with pytest.raises(ValueError, match="SOC QR: 0"):
        e175.validate_saved(changed, freeze)


def test_endpoint_witness_ignores_auxiliary_noise() -> None:
    saved, freeze, oracle = generated()
    a, b = oracle["truth"][:1980], oracle["truth"][1980:]
    x = np.r_[a, b, np.full(990, -1.0), np.full(990, 3.0), 0.0]
    result, _ = e175.returned_witness(saved, x, freeze)
    assert result["valid"]
    assert result["support"]
    assert not result["auxiliary_support"]


def _synthetic_gate_root(
    tmp_path: Path, case: str = "feasible"
) -> tuple[Path, Path, Path, dict, dict]:
    saved, freeze, oracle = generated(case)
    root = tmp_path / "root"
    root.mkdir()
    for name in (e175.SOURCE, e175.TESTS, *freeze["reused_sources"]):
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    design = root / freeze["design"]
    design.parent.mkdir(parents=True, exist_ok=True)
    design.write_text("generated E175 contract\n", encoding="utf-8")
    freeze["design_sha256"] = e175.sha256(design)
    base = root / freeze["base_freeze"]
    base.parent.mkdir(parents=True, exist_ok=True)
    base.write_text(
        json.dumps({"sample_contract": freeze["sample_contract"]}), encoding="utf-8"
    )
    freeze["base_freeze_sha256"] = e175.sha256(base)
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
    raw = root / freeze["raw"]
    raw.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        raw,
        **{
            freeze["archive_key_map"].get(k, k): saved[k]
            for k in freeze["allowed_qr_keys"] + freeze["allowed_problem_keys"]
        },
    )
    with e175.zipfile.ZipFile(raw) as archive:
        freeze["selected_uncompressed_bytes"] = sum(
            archive.getinfo(freeze["archive_key_map"].get(k, k) + ".npy").file_size
            for k in freeze["allowed_qr_keys"] + freeze["allowed_problem_keys"]
        )
    for name in freeze["input_sha256"]:
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.write_bytes(b"synthetic pin\n")
    freeze["input_sha256"] = {
        name: e175.sha256(root / name) for name in freeze["input_sha256"]
    }
    freeze["reused_sources"] = {
        name: e175.sha256(root / name) for name in freeze["reused_sources"]
    }
    freeze["resource_limits"]["min_free_bytes"] = 0
    freeze["resource_limits"]["internal_wall_s"] = 120.0
    freeze_path = root / "generated-freeze.json"
    freeze_path.write_text(json.dumps(freeze), encoding="utf-8")
    gate_path = root / "generated-gate.json"
    gate_path.write_text(
        json.dumps(
            {
                "schema": "flylab.m3.e175.parent-source-review.v1",
                "accepts_single_endpoint_history_diagnostic": True,
                "freeze_sha256": e175.sha256(freeze_path),
                "design_sha256": freeze["design_sha256"],
                "source_sha256": {
                    name: e175.sha256(root / name) for name in (e175.SOURCE, e175.TESTS)
                },
            }
        ),
        encoding="utf-8",
    )
    return root, freeze_path, gate_path, saved, oracle


def _solver(saved: dict, oracle: dict, status: str = "Solved") -> type:
    a, b = oracle["truth"][:1980], oracle["truth"][1980:]
    pair = np.stack((a.reshape(990, 2), b.reshape(990, 2)))
    x = np.r_[a, b, pair.min(axis=(0, 2)) - 1e-8, pair.max(axis=(0, 2)) + 1e-8, 0.0]

    class FakeSolver:
        calls = 0

        def __init__(self, _p, q, _a, rhs, cones, settings):
            type(self).calls += 1
            assert q.shape == (5941,)
            assert len(cones) == 16
            self.rhs, self.settings = rhs, settings

        def solve(self):
            return SimpleNamespace(
                x=x,
                s=np.ones_like(self.rhs),
                z=np.zeros_like(self.rhs),
                status=status,
                iterations=1,
                solve_time=0.01,
            )

        def get_settings(self):
            return self.settings

    return FakeSolver


def _cpu1(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        monkeypatch.setenv(key, "1")


def test_generated_replay_retains_raw_and_cone_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, freeze_path, gate_path, saved, oracle = _synthetic_gate_root(tmp_path)
    _cpu1(monkeypatch)
    solver = _solver(saved, oracle)
    out = tmp_path / "out"
    with threadpool_limits(limits=1):
        report = e175.run(freeze_path, gate_path, out, root, solver)
    assert solver.calls == 1
    assert report["status"] == "valid_diagnostic"
    assert report["conclusion"] == "supported_saved_endpoint_witness"
    assert (out / "endpoint-solver-raw.npz").exists()
    assert (out / "source-diagnose_c_banc_m3_protection_split.py").exists()
    with np.load(out / "endpoint-problem.npz") as problem:
        assert problem["cone_dims"].size == 16
        assert problem["cone_dims"][0] == 28602
    manifest = json.loads((out / "output-manifest.json").read_text(encoding="utf-8"))
    assert manifest["files"]["result.json"] == e175.sha256(out / "result.json")


def test_generated_anchor_rejection_preserves_preparation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, freeze_path, gate_path, saved, oracle = _synthetic_gate_root(tmp_path)
    _cpu1(monkeypatch)
    solver = _solver(saved, oracle)
    monkeypatch.setattr(e175, "anchor_valid", lambda *_: False)
    out = tmp_path / "out"
    with threadpool_limits(limits=1):
        report = e175.run(freeze_path, gate_path, out, root, solver)
    assert solver.calls == 0
    assert report["status"] == "invalid_not_tested"
    assert (out / "endpoint-preparation-raw.npz").exists()
    assert (out / "endpoint-problem.npz").exists()
    assert report["primary_error"]["message"] == "anchor inner feasibility"


@pytest.mark.parametrize(
    "failure",
    [
        "status",
        "raw_publication",
        "late_resource",
        "result_publication",
        "manifest_publication",
        "mismatch",
    ],
)
def test_generated_failure_retains_first_error_and_saved_raw(
    failure: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, freeze_path, gate_path, saved, oracle = _synthetic_gate_root(tmp_path)
    _cpu1(monkeypatch)
    if failure == "mismatch":
        with np.load(
            root / json.loads(freeze_path.read_text(encoding="utf-8"))["raw"],
            allow_pickle=False,
        ) as archive:
            changed = {
                name: archive[
                    json.loads(freeze_path.read_text(encoding="utf-8"))[
                        "archive_key_map"
                    ].get(name, name)
                ]
                for name in saved
            }
        data = changed["problem_A_data"].copy()
        for col in range(3960):
            lo, hi = changed["problem_A_indptr"][col : col + 2]
            rows = changed["problem_A_indices"][lo:hi]
            data[lo:hi][(rows >= 16326) & (rows < 24642)] *= 1.1
        changed["problem_A_data"] = data
        raw_path = root / json.loads(freeze_path.read_text(encoding="utf-8"))["raw"]
        freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
        np.savez_compressed(
            raw_path,
            **{
                freeze["archive_key_map"].get(k, k): changed[k]
                for k in freeze["allowed_qr_keys"] + freeze["allowed_problem_keys"]
            },
        )
        freeze["input_sha256"][freeze["raw"]] = e175.sha256(raw_path)
        freeze_path.write_text(json.dumps(freeze), encoding="utf-8")
        gate = json.loads(gate_path.read_text(encoding="utf-8"))
        gate["freeze_sha256"] = e175.sha256(freeze_path)
        gate_path.write_text(json.dumps(gate), encoding="utf-8")
        oracle["truth"] = np.r_[0.7 * saved["w0"], 0.7 * saved["w0"]]
    solver = _solver(
        saved, oracle, "MaxIterations" if failure == "status" else "Solved"
    )
    if failure == "raw_publication":
        original = np.savez_compressed
        tripped = []

        def fail_once(path, **arrays):
            if str(path).endswith("endpoint-solver-raw.npz") and not tripped:
                tripped.append(True)
                raise OSError("generated raw publication failure")
            return original(path, **arrays)

        monkeypatch.setattr(np, "savez_compressed", fail_once)
    if failure == "late_resource":
        original_check = e175._resource
        calls = []

        def fail_late(out, started, limits):
            calls.append(1)
            if len(calls) == 4:
                raise ValueError("generated late resource failure")
            return original_check(out, started, limits)

        monkeypatch.setattr(e175, "_resource", fail_late)
    if failure in ("result_publication", "manifest_publication"):
        original_write = e175.json_write
        tripped = []
        selected = (
            "result.json" if failure == "result_publication" else "output-manifest.json"
        )

        def fail_write_once(path, payload):
            if path.name == selected and not tripped:
                tripped.append(True)
                raise OSError(f"generated {selected} publication failure")
            return original_write(path, payload)

        monkeypatch.setattr(e175, "json_write", fail_write_once)
    out = tmp_path / "out"
    with threadpool_limits(limits=1):
        report = e175.run(freeze_path, gate_path, out, root, solver)
    assert solver.calls == 1
    assert report["status"] == "invalid_not_tested"
    assert report["new_conic_fits"] == 1
    assert (out / "endpoint-solver-raw.npz").exists() or (
        out / "endpoint-solver-raw.fallback.npz"
    ).exists()
    if failure == "raw_publication":
        assert report["primary_error"]["message"] == "generated raw publication failure"
        assert (out / "endpoint-solver-raw.fallback.npz").exists()
    elif failure == "status":
        assert report["primary_error"]["message"] == "solver status"
    elif failure == "mismatch":
        assert report["primary_error"]["message"] == "startup direct/QR norm mismatch"
        assert not report["witness"]["startup_norm_ok"]
        assert (out / "endpoint-derived.npz").exists()
    elif failure in ("result_publication", "manifest_publication"):
        selected = (
            "result.json" if failure == "result_publication" else "output-manifest.json"
        )
        assert (
            report["primary_error"]["message"]
            == f"generated {selected} publication failure"
        )
        if failure == "result_publication":
            assert (out / "result.fallback.json").exists()
    else:
        assert report["primary_error"]["message"] == "generated late resource failure"
    manifest = json.loads((out / "output-manifest.json").read_text(encoding="utf-8"))
    if (out / "result.json").exists():
        assert manifest["files"]["result.json"] == e175.sha256(out / "result.json")
    else:
        assert manifest["files"]["result.fallback.json"] == e175.sha256(
            out / "result.fallback.json"
        )
