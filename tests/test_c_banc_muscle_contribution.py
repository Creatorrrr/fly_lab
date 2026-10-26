"""Generated-only tests for the E179 saved-data analyzer."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tools import analyze_c_banc_m3_muscle_contribution as e179


def test_bank_uses_saved_adjacent_rows_and_endpoint_rate() -> None:
    rate = np.array([[0.0, 0.0], [2.0, 4.0], [5.0, 6.0]], dtype=np.float64)
    tau = np.array([0.005, 0.01, 0.04], dtype=np.float64)
    bank = np.zeros((3, 2, 3), dtype=np.float64)
    bank[0] = 0.3
    bank[1] = 0.7  # Deliberately distinct from a rolled-forward value.
    expected = np.stack(
        [
            bank[0] + -np.expm1(-0.005 / tau) * (rate[1, :, None] - bank[0]),
            bank[1] + -np.expm1(-0.005 / tau) * (rate[2, :, None] - bank[1]),
        ]
    )
    np.testing.assert_allclose(
        e179.bank_recurrence(rate, bank, tau), expected, rtol=0, atol=0
    )
    with pytest.raises(ValueError, match="motor rate"):
        e179.bank_recurrence(np.full_like(rate, np.nan), bank, tau)


def test_variable_signed_hinge_and_fixed_copy() -> None:
    bank = np.array(
        [
            [[1.0, 2.0, 3.0], [0.0, 0.0, 0.0], [4.0, 5.0, 6.0]],
            [[2.0, 3.0, 4.0], [8.0, 8.0, 8.0], [5.0, 6.0, 7.0]],
        ],
        dtype=np.float64,
    )
    original = np.array([[19.0, 23.0], [20.0, 29.0]], dtype=np.float64)
    membership = np.array([0, 1, 0], dtype=np.int64)
    fixed = np.array([1], dtype=np.int64)
    kappa = np.array([2.0, 1.0, 4.0], dtype=np.float64)
    low = np.zeros((2, 3, 3), dtype=np.float64)
    high = np.zeros_like(low)
    low[0, 0] = [1.0, 2.0, 3.0]
    low[0, 2] = [2.0, 3.0, 4.0]
    high[0, 0] = [0.5, 1.0, 1.5]
    high[0, 2] = [1.0, 1.5, 2.0]
    terms = e179.selected_contributions(
        bank, np.array([0, 2]), kappa, low[0, [0, 2]], high[0, [0, 2]]
    )
    assert np.any(terms["hinge"] < 0)
    expected_variable = np.sum(terms["low"], axis=(1, 2)) + np.sum(
        terms["hinge"], axis=(1, 2)
    )
    reconstructed = e179.reconstruct_activation(
        bank, original, membership, fixed, kappa, low, high
    )
    np.testing.assert_allclose(reconstructed[:, 0], expected_variable, rtol=0, atol=0)
    np.testing.assert_array_equal(reconstructed[:, 1], original[:, 1])
    assert e179.residual(reconstructed[:, 1], original[:, 1], "fixed")["max_abs"] == 0
    with pytest.raises(ValueError, match="residual"):
        e179.residual(np.array([1.0]), np.array([2.0]), "broken")


def test_native_event_exact_stability_and_unavailable() -> None:
    load = np.zeros(1000, dtype=np.float64)
    load[:300] = 0.03
    load[500:601] = 0.03  # 101 supported samples, then stable release.
    time = (np.arange(len(load)) + 1) * 0.0001
    events, stats = e179.native_events(load, time)
    assert [e["native_index"] for e in events] == [300, 601]
    assert [e["reason"] for e in events] == ["included", "included"]
    assert stats["included_count"] == 2
    assert stats["longest_nonsupport_samples"] == 399
    load[650] = 0.03  # Disrupt the post interval of the second transition.
    events, stats = e179.native_events(load, time)
    assert events[1]["reason"] == "unstable_post_nonsupport"
    assert stats["included_count"] == 1
    no_events, no_stats = e179.native_events(np.zeros_like(load), time)
    assert no_events == [] and no_stats["included_count"] == 0


def test_source_gate_rejects_before_any_original_loader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A malformed gate must fail at the SHA boundary, before source value loading.
    monkeypatch.setattr(
        e179, "_load_arm", lambda *_: pytest.fail("loaded original values")
    )
    freeze = tmp_path / "freeze.json"
    review = tmp_path / "review.json"
    freeze.write_text(json.dumps({"schema": e179.SCHEMA}), encoding="utf-8")
    review.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="freeze SHA"):
        e179.run(freeze, "0" * 64, review, "0" * 64, tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_publication_failure_removes_stage_and_no_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stage = tmp_path / ".e179-stage-generated"
    stage.mkdir()
    (stage / "result.json").write_text(
        '{"status":"valid_saved_data_diagnostic"}', encoding="utf-8"
    )
    output = tmp_path / "published"
    original_rename = Path.rename

    def fail_rename(path: Path, target: Path) -> Path:
        if path == stage:
            raise OSError("generated publication failure")
        return original_rename(path, target)

    monkeypatch.setattr(Path, "rename", fail_rename)
    with pytest.raises(OSError, match="generated publication failure"):
        e179._publish_stage(stage, output)
    assert not stage.exists() and not output.exists()
