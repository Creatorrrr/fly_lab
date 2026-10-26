"""Generated-only tests for the E171 saved QR diagnosis."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Self

import numpy as np
import pytest

MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "tools/diagnose_c_banc_m3_saved_qr_capacity.py"
)
SPEC = importlib.util.spec_from_file_location("e171_saved_qr_capacity", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
e171 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(e171)


def test_one_svd_full_rank_and_qr_rms(monkeypatch: pytest.MonkeyPatch) -> None:
    qr = np.array([[2.0, 0.0, -4.0], [0.0, 3.0, 3.0], [0.0, 0.0, 6.0]])
    original = np.linalg.svd
    calls = []

    def counted(
        *args: object, **kwargs: object
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(np.linalg, "svd", counted)
    raw = {}
    details = e171.solve_block(qr, np.zeros(2), raw)
    assert len(calls) == 1
    assert details["rank"] == 2
    np.testing.assert_allclose(raw["x"], [2.0, -1.0])
    assert details["actual_norm"] == pytest.approx(6.0)
    metric = e171.evaluate_qr(qr, raw["x"], np.zeros(2), 7)
    assert metric["rms"] == pytest.approx(6 / np.sqrt(7))
    assert "pointwise_max" not in metric


def test_rank_deficient_zero_and_cutoff_boundary() -> None:
    for matrix, offset, rank in (
        (np.array([[1.0, 1.0], [0.0, 0.0]]), np.array([2.0, 3.0]), 1),
        (np.zeros((2, 2)), np.array([2.0, 3.0]), 0),
        (np.diag([1.0, 1e-16]), np.array([2.0, 3.0]), 1),
    ):
        raw = {}
        details = e171.solve_block(np.column_stack((matrix, offset)), np.zeros(2), raw)
        assert details["rank"] == rank
        assert np.isfinite(raw["x"]).all()
        assert (
            np.linalg.norm(raw["actual_residual"] - raw["projected_residual"]) < 1e-12
        )
    qr = np.column_stack((np.diag([1.0, 1e-12]), [2.0, 3.0]))
    assert e171.solve_block(qr, np.zeros(2), {})["rank"] == 2


def test_history_signed_beta_uses_all_four_endpoints() -> None:
    alpha = np.ones((990, 2))
    beta = np.zeros((990, 2))
    alpha[0] = [0.7, 0.9]
    beta[0] = [-1.0, -0.2]
    coefficients = np.r_[alpha.ravel(), beta.ravel()]
    muscles = np.repeat(np.arange(330) % 77, 3).astype(np.int64)
    contract = {
        "motor_floor_fraction": 1e-6,
        "muscle_lower": 0.5,
        "muscle_upper": 2.0,
        "physical_atol": 1e-12,
    }
    result = e171.coefficient_diagnostics(
        coefficients,
        muscles,
        list(range(77)),
        "history",
        contract,
        np.repeat(np.arange(330) * 2, 3),
    )
    assert result["pair_min"][0] == pytest.approx(-0.3)
    assert result["pair_max"][0] == pytest.approx(0.9)
    assert result["branches"]["beta"]["below_zero_count"] == 2
    assert result["motors"][0]["minimum"] == pytest.approx(1.7)
    assert result["motors"][1]["motor_original"] == 2


def test_aggregate_uses_42_joint_denominator_and_resolution() -> None:
    legs = {}
    for leg in ("lf", "lm", "lh", "rf", "rm", "rh"):
        legs[leg] = {
            "metrics": {
                kind: {"norm": 7.0, "baseline_norm": 10.0} for kind in e171.KINDS
            }
        }
    result = e171.aggregate_family(legs, [0.0] * 6)
    assert result["metrics"]["command"]["rms"] == pytest.approx(
        np.sqrt(6 * 7**2 / (1824 * 42))
    )
    assert result["command_ratio"] == pytest.approx(0.7)
    assert result["attains_20_percent"] is True
    unresolved = e171.aggregate_family(legs, [1.0] * 6)
    assert unresolved["numerically_unresolved"] is True
    assert unresolved["attains_20_percent"] is False


def test_real_validator_generated_full_contract_and_rejections() -> None:
    fixed = [4, 15, 16, 19, 30, 31, 34, 35, 46, 47, 52, 63, 64, 67, 78, 82, 83, 94, 95]
    variable = [muscle for muscle in range(96) if muscle not in fixed]
    motors = np.array([i for i in range(391) if i % 7 != 0][:330], dtype=np.int64)
    groups = np.repeat(np.arange(6), [17, 15, 16, 17, 15, 16])
    owner = np.repeat(np.array([variable[i % 77] for i in range(330)]), 3)
    ticks = np.arange(100, 1924, dtype=np.int64)
    legs = ["lf", "lm", "lh", "rf", "rm", "rh"]
    freeze = {
        "sample_contract": {
            "legs": legs,
            "variable_motors": motors.tolist(),
            "variable_muscles": variable,
            "fixed_muscles": fixed,
            "group_legs": groups.tolist(),
            "group_counts": [17, 15, 16, 17, 15, 16],
            "target_count": 1824,
            "target_first_tick": 100,
            "target_last_tick": 1923,
            "startup_first_tick": 1,
            "startup_last_tick": 99,
            "target_indices_sha256": hashlib.sha256(ticks.tobytes()).hexdigest(),
        },
        "families": {
            "static": {"coefficient_count": 1980},
            "history": {"coefficient_count": 3960},
        },
    }
    families = {}
    for family in ("static", "history"):
        saved = {
            "pair_motors": np.repeat(motors, 3),
            "pair_filters": np.tile(np.arange(3, dtype=np.int64), 330),
            "pair_muscles": owner,
            "w0": np.ones(1980),
            "target_indices": ticks,
            "pair_start": ticks[:-1],
            "pair_end": ticks[1:],
        }
        for index, leg in enumerate(legs):
            a = np.flatnonzero(groups[np.repeat(owner, 2)] == index)
            active = a if family == "static" else np.r_[a, a + 1980]
            for kind in e171.KINDS:
                key = f"qr_{kind}_{leg}"
                saved[key] = np.zeros(
                    (min(e171.ROWS[kind], active.size + 1), active.size + 1)
                )
                saved[key + "_active"] = active
        families[family] = saved
    static, history = families["static"], families["history"]
    assert (
        e171.validate_saved(static, history, freeze)["legs"]
        == freeze["sample_contract"]["legs"]
    )
    assert not np.array_equal(np.unique(static["pair_motors"]), np.arange(330))
    changed = dict(static)
    changed["qr_command_lf_active"] = static["qr_command_lf_active"][::-1].copy()
    with pytest.raises(ValueError, match="active mapping"):
        e171.validate_saved(changed, history, freeze)
    changed = dict(static)
    changed["pair_muscles"] = static["pair_muscles"].copy()
    changed["pair_muscles"][0] = freeze["sample_contract"]["fixed_muscles"][0]
    with pytest.raises(ValueError, match="muscle"):
        e171.validate_saved(changed, history, freeze)
    changed = dict(static)
    changed["w0"] = static["w0"][:990]
    with pytest.raises(ValueError, match="w0: shape"):
        e171.validate_saved(changed, history, freeze)


def test_selected_load_never_accesses_forbidden_key(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class Archive:
        def __init__(self) -> None:
            self.accessed: list[str] = []

        def __enter__(self) -> Self:
            return self

        def __exit__(self, *_: object) -> None:
            return None

        def __getitem__(self, key: str) -> np.ndarray:
            self.accessed.append(key)
            if key == "forbidden_target":
                raise AssertionError("forbidden")
            return np.array([1.0])

    archive = Archive()
    monkeypatch.setattr(np, "load", lambda *_args, **_kwargs: archive)
    loaded = e171.read_selected(tmp_path / "synthetic.npz", ["qr_command_lf", "w0"])
    assert set(loaded) == {"qr_command_lf", "w0"}
    assert archive.accessed == ["qr_command_lf", "w0"]


def test_gate_failure_precedes_array_open(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    freeze = tmp_path / "freeze.json"
    gate = tmp_path / "gate.json"
    freeze.write_text(json.dumps({"schema": "wrong"}), encoding="utf-8")
    gate.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        np, "load", lambda *_args, **_kwargs: pytest.fail("array opened")
    )
    with pytest.raises(ValueError, match="freeze schema"):
        e171.run(freeze, gate, tmp_path / "out", tmp_path)
    assert not (tmp_path / "out").exists()


def test_raw_survives_post_return_numerical_failure() -> None:
    qr = np.array([[1.0, -2.0], [0.0, 3.0]])
    raw = {}
    # A NaN baseline fails only after SVD return and raw population.
    with pytest.raises(ValueError, match="command fit worse"):
        e171.solve_block(qr, np.array([np.nan]), raw)
    assert {"U", "s", "Vh", "x", "actual_residual", "projected_residual"} <= set(raw)


def test_infinite_full_condition_is_null() -> None:
    qr = np.column_stack((np.diag([1.0, np.nextafter(0.0, 1.0)]), [2.0, 3.0]))
    result = e171.solve_block(qr, np.zeros(2), {})
    assert result["full_condition"] is None
    assert result["rank"] == 1


def _generated_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> tuple[Path, Path, Path]:
    legs = ("lf", "lm", "lh", "rf", "rm", "rh")
    base = {
        "w0": np.zeros(6),
        "pair_muscles": np.repeat(np.arange(330) % 77, 3),
        "pair_motors": np.repeat(np.arange(330) * 2, 3),
    }
    for index, leg in enumerate(legs):
        for kind in e171.KINDS:
            key = f"qr_{kind}_{leg}"
            base[key] = np.array([[1.0, -1.0], [0.0, 0.0]])
            base[f"{key}_active"] = np.array([index], dtype=np.int64)
    static_path = tmp_path / "static.npz"
    history_path = tmp_path / "history.npz"
    np.savez(static_path, **base)
    np.savez(history_path, **base)
    freeze = {
        "family_order": ["static", "history"],
        "families": {
            "static": {"raw": str(static_path), "coefficient_count": 6},
            "history": {"raw": str(history_path), "coefficient_count": 6},
        },
        "allowed_keys": list(base),
        "archive_key_map": {},
        "resource_limits": {
            "internal_wall_s": 90,
            "max_output_bytes": 134217728,
            "min_free_bytes": 0,
        },
        "contract": {},
        "design_sha256": "generated",
    }
    monkeypatch.setattr(
        e171, "preflight", lambda *_args: {"freeze": freeze, "threadpool_info": []}
    )
    monkeypatch.setattr(
        e171,
        "validate_saved",
        lambda *_args: {"legs": legs, "variable_muscles": list(range(77))},
    )
    monkeypatch.setattr(
        e171, "coefficient_diagnostics", lambda *_args: {"generated": True}
    )
    freeze_path = tmp_path / "freeze.json"
    gate_path = tmp_path / "gate.json"
    freeze_path.write_text("{}", encoding="utf-8")
    gate_path.write_text("{}", encoding="utf-8")
    return freeze_path, gate_path, tmp_path / "out"


def test_generated_run_has_exactly_twelve_svd_calls(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    freeze_path, gate_path, out = _generated_run(monkeypatch, tmp_path)
    original = np.linalg.svd
    count = 0

    def counted(
        *args: object, **kwargs: object
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        nonlocal count
        count += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(np.linalg, "svd", counted)
    result = e171.run(freeze_path, gate_path, out)
    assert result["status"] == "valid_diagnostic"
    assert result["new_svd_fits"] == count == 12
    assert len(list(out.glob("*-svd-raw.npz"))) == 12
    assert result["startup_pointwise_max"] == "not_evaluated"


def test_first_raw_publication_failure_retains_exact_return(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    freeze_path, gate_path, out = _generated_run(monkeypatch, tmp_path)
    original = np.savez_compressed
    count = 0

    def fail_first(*args: object, **kwargs: object) -> None:
        nonlocal count
        count += 1
        if count == 1:
            raise OSError("first raw publication failed")
        original(*args, **kwargs)

    monkeypatch.setattr(np, "savez_compressed", fail_first)
    result = e171.run(freeze_path, gate_path, out)
    assert result["status"] == "invalid_not_tested"
    assert result["new_svd_fits"] == 1
    assert result["partial_blocks"][0]["returned_x_sha256"]
    with np.load(out / "static-lf-svd-raw.fallback.npz", allow_pickle=False) as raw:
        np.testing.assert_allclose(raw["x"], [1.0])
    assert json.loads((out / "result.json").read_text(encoding="utf-8"))[
        "partial_blocks"
    ]


def test_final_stage_resource_crossing_invalidates_result(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    freeze_path, gate_path, out = _generated_run(monkeypatch, tmp_path)
    original = e171.resource_check

    def cross_after_publication(
        path: Path, started: float, limits: dict[str, object]
    ) -> None:
        if (path / "result.json").exists():
            raise ValueError("internal wall limit")
        original(path, started, limits)

    monkeypatch.setattr(e171, "resource_check", cross_after_publication)
    result = e171.run(freeze_path, gate_path, out)
    assert result["new_svd_fits"] == 12
    assert result["status"] == "invalid_not_tested"
    assert (
        json.loads((out / "result.json").read_text(encoding="utf-8"))["status"]
        == "invalid_not_tested"
    )


def test_one_resolved_witness_survives_other_family_unresolved() -> None:
    assert (
        e171.classify_conclusion(
            [
                {"attains_20_percent": True, "numerically_unresolved": False},
                {"attains_20_percent": False, "numerically_unresolved": True},
            ]
        )
        == "supported_returned_saved_affine_witness"
    )


def test_result_publication_failure_retains_complete_report(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    freeze_path, gate_path, out = _generated_run(monkeypatch, tmp_path)
    original = e171.json_write
    failed = False

    def fail_result_once(path: Path, payload: dict[str, object]) -> None:
        nonlocal failed
        if path.name == "result.json" and not failed:
            failed = True
            raise OSError("result publication failed")
        original(path, payload)

    monkeypatch.setattr(e171, "json_write", fail_result_once)
    with pytest.raises(OSError, match="result publication failed"):
        e171.run(freeze_path, gate_path, out)
    fallback = json.loads((out / "result.fallback.json").read_text(encoding="utf-8"))
    assert fallback["status"] == "invalid_not_tested"
    assert fallback["new_svd_fits"] == 12
    assert len(fallback["families"]["static"]["legs"]) == 6
    assert len(fallback["families"]["history"]["legs"]) == 6


def test_per_leg_baseline_difference_fails_before_any_svd(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    freeze_path, gate_path, out = _generated_run(monkeypatch, tmp_path)
    history_path = tmp_path / "history.npz"
    with np.load(history_path, allow_pickle=False) as archive:
        saved = {key: archive[key] for key in archive.files}
    saved["qr_velocity_lf"] = saved["qr_velocity_lf"].copy()
    saved["qr_velocity_lf"][0, -1] = -2.0
    np.savez(history_path, **saved)
    monkeypatch.setattr(
        np.linalg, "svd", lambda *_args, **_kwargs: pytest.fail("SVD started")
    )
    result = e171.run(freeze_path, gate_path, out)
    assert result["status"] == "invalid_not_tested"
    assert result["new_svd_fits"] == 0
    assert "per-leg family baseline mismatch" in result["primary_error"]["message"]


def test_original_error_survives_later_resource_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    freeze_path, gate_path, out = _generated_run(monkeypatch, tmp_path)

    def fail_input(*_args: object) -> None:
        raise ValueError("first input failure")

    def fail_resource(*_args: object) -> None:
        raise ValueError("later final resource failure")

    monkeypatch.setattr(e171, "read_selected", fail_input)
    monkeypatch.setattr(e171, "resource_check", fail_resource)
    result = e171.run(freeze_path, gate_path, out)
    assert result["status"] == "invalid_not_tested"
    assert result["new_svd_fits"] == 0
    assert result["primary_error"]["message"] == "first input failure"
    assert (
        result["secondary_resource_errors"][0]["message"]
        == "later final resource failure"
    )


def test_persisted_velocity_aliases_and_missing_member_preflight(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "persisted.npz"
    starts = np.arange(100, 1923, dtype=np.int64)
    ends = starts + 1
    np.savez(path, affine_rows_velocity_start=starts, affine_rows_velocity_end=ends)
    loaded = e171.read_selected(path, ["pair_start", "pair_end"], e171.ARCHIVE_KEY_MAP)
    np.testing.assert_array_equal(loaded["pair_start"], starts)
    np.testing.assert_array_equal(loaded["pair_end"], ends)
    monkeypatch.setattr(
        np, "load", lambda *_args, **_kwargs: pytest.fail("array values opened")
    )
    e171.validate_archive_members(
        path, ["pair_start", "pair_end"], e171.ARCHIVE_KEY_MAP
    )
    with pytest.raises(KeyError, match="pair_start"):
        e171.validate_archive_members(path, ["pair_start", "pair_end"], {})
