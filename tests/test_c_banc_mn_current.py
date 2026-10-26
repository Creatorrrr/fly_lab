"""Synthetic E187 contracts. No original saved current or force is evaluated."""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from tools import analyze_c_banc_m3_mn_current as e187


def test_gate_rejects_unreviewed_source_before_numeric_load(tmp_path, monkeypatch):
    monkeypatch.setattr(e187, "ROOT", tmp_path)
    for relative in (e187.DESIGN, e187.SUPERVISOR):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("synthetic", encoding="utf-8")
    sources = {}
    for relative in (*e187.OWNED, *(f"generated/source-{i}.py" for i in range(11))):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("synthetic", encoding="utf-8")
        sources[relative] = e187.digest(path)
    inputs = {}
    for i in range(511):
        relative = f"generated/input-{i:03}.bin"
        path = tmp_path / relative
        path.write_bytes(bytes([i % 251]))
        inputs[relative] = e187.digest(path)
    for name, content in (
        (
            "e185-review.json",
            {
                "saved_raw_analysis_accepted": True,
                "child_observation_valid": True,
                "original_driver_accepted": False,
            },
        ),
        ("e186-review.json", {"run_validity": "valid", "candidate_adopted": False}),
    ):
        path = tmp_path / name
        path.write_text(json.dumps(content), encoding="utf-8")
        inputs[name] = e187.digest(path)
    # Keep the exact inventory cardinality; reviews occupy two of its entries.
    for key in ("generated/input-000.bin", "generated/input-001.bin"):
        inputs.pop(key)
    metadata = {
        "schema": "flylab.m3.e187.mn-current.metadata.v1",
        "design_path": e187.DESIGN,
        "design_sha256": e187.digest(tmp_path / e187.DESIGN),
        "owned_files": list(e187.OWNED),
        "original_numeric_analysis_runs": 0,
        "new_models": 0,
        "gpu_work": 0,
        "F6_exposed": 0,
        "candidate_adopted": False,
        "inputs": inputs,
        "headers": {},
        "reused_sources": {
            key: value for key, value in sources.items() if key not in e187.OWNED
        },
        "e185_review_path": "e185-review.json",
        "previous_review_path": "e186-review.json",
    }
    metadata_path = tmp_path / e187.METADATA
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    output = tmp_path / e187.OUTPUT
    freeze = {
        "schema": "flylab.m3.e187.mn-current.freeze.v1",
        "design_path": e187.DESIGN,
        "design_sha256": metadata["design_sha256"],
        "metadata_path": e187.METADATA,
        "metadata_sha256": e187.digest(metadata_path),
        "output_path": e187.OUTPUT,
        "parameters": e187.PARAMETERS,
        "parent_supervisor_sha256": e187.digest(tmp_path / e187.SUPERVISOR),
        "source_hashes": sources,
        "input_sha256": inputs,
    }
    freeze_path = tmp_path / "freeze.json"
    freeze_path.write_text(json.dumps(freeze), encoding="utf-8")
    review = {
        "schema": "flylab.m3.e187.mn-current.source-review.v1",
        "approved": True,
        "blocking_defects": [],
        "freeze_sha256": e187.digest(freeze_path),
        "design_sha256": freeze["design_sha256"],
        "metadata_sha256": freeze["metadata_sha256"],
        "source_hashes": sources,
    }
    review_path = tmp_path / "review.json"
    review_path.write_text(json.dumps(review), encoding="utf-8")
    monkeypatch.setattr(
        np, "load", lambda *a, **k: pytest.fail("array opened before gate")
    )
    assert e187.gate(freeze_path, review_path, output)[1] == e187.digest(freeze_path)
    source_path = tmp_path / e187.OWNED[0]
    source_path.write_text("mutated", encoding="utf-8")
    with pytest.raises(ValueError, match="source changed"):
        e187.gate(freeze_path, review_path, output)


def test_partial_incoming_preserves_missing_and_cuda_ftz():
    tiny = np.float32(1e-40)
    rate = np.array([[tiny, 2], [1, 3]], np.float32)
    ids = np.array([5, 2], np.int64)  # roster order is not graph order
    edge_ids = np.array([0, 1, 2], np.int64)
    pre = np.array([5, 9, 2], np.int64)
    weights = np.array([1e20, 4, -2], np.float32)
    mask = np.ones(10, np.float32)
    known, terms, cuda = e187.partial_terms(rate, ids, edge_ids, pre, weights, mask)
    assert known.tolist() == [True, False, True]
    assert np.isnan(terms[:, 1]).all() and np.isnan(cuda[:, 1]).all()
    assert terms[0, 0] > 0 and cuda[0, 0] == 0  # packed mask, CUDA FTZ multiply
    assert terms[0, 2] == -4 and terms[1, 2] == -6
    assert np.nan_to_num(terms[0]).sum() != terms[0, [0, 2]].sum() or np.isnan(
        terms[0, 1]
    )


def test_conditioned_zero_support_nan_and_nonfinite_weight_removal():
    values = np.array([[10, np.nan], [20, 40], [30, 60]], float)
    count = np.zeros((3, 6, 3), np.int64)
    count[:, 4, 0] = [2, 0, 1]
    count[:, 4, 2] = [0, 5, 0]
    magnitude = np.zeros((3, 6))
    magnitude[:, 4] = [4, 0, 2]
    mean, denom, mag_mean, mag_denom, time_mean, time_count = e187.conditioned(
        values, np.arange(3), count, magnitude
    )
    assert mean[0, 0] == pytest.approx((2 * 10 + 30) / 3)
    assert mean[0, 1] == 60 and denom[0, 1] == 1
    assert mean[2, 0] == 20 and mean[2, 1] == 40
    assert np.isnan(mean[1]).all() and np.all(denom[1] == 0)
    assert mag_mean[0] == pytest.approx((4 * 10 + 2 * 30) / 6)
    assert mag_mean[1] == 60 and mag_denom[1] == 2
    assert time_mean[0] == 20 and time_count[1] == 2


def test_fixed_52_windows_retain_all_11_events_and_zero_rm_support():
    windows = {
        "whole": np.arange(1, 601),
        "evaluation": np.arange(100, 600),
        "startup99": np.arange(1, 100),
        "selected8": np.concatenate([np.arange(i - 20, i + 21) for i in e187.SELECTED]),
    }
    for band, (a, b) in e187.BANDS.items():
        windows["selected8:" + band] = np.concatenate(
            [np.arange(i + a, i + b) for i in e187.SELECTED]
        )
    for event in e187.ALL_EVENTS:
        for band, (a, b) in e187.BANDS.items():
            windows[f"event{event}:{band}"] = np.arange(event + a, event + b)
    count = np.zeros((601, 6, 3), np.int64)
    signed = np.zeros((601, 6, 3))
    aligned = {
        "event_indices_all": np.array(e187.ALL_EVENTS),
        "event_indices_selected": np.array(e187.SELECTED),
        "window_labels": np.array(list(windows)),
        "native_count": count,
        "native_signed_force_sum_bw": signed,
        "native_negative_magnitude_sum_bw": np.zeros((601, 6)),
        "condition_native_count": np.zeros((52, 6, 3), np.int64),
        "condition_signed_force_sum_bw": np.zeros((52, 6, 3)),
    }
    result = e187.windows_from_e186(aligned)
    assert len(result) == 52
    assert "event148:post" in result and "event564:recover" in result
    assert result["selected8:post"].shape == (80,)
    count[131, 4, 0] = 1
    with pytest.raises(ValueError, match="condition count"):
        e187.windows_from_e186(aligned)


def test_full_boundary_coverage_uses_full_terms_after_three_seconds():
    ids = np.arange(6, dtype=np.int64)
    posts = np.array([10, 10, 11, 12, 13, 14], np.int64)
    known = np.array([True, False, True, True, True, True])
    boundaries = []
    for tick in (0, 200, 220, 2000, 600):
        records = []
        for post in (10, 11, 12, 13, 14):
            local_ids = ids[posts == post].tolist()
            if tick == 2000 and post == 10:
                terms = [1.0, -9.0]  # unobserved input is real, not zero
            elif tick == 0:
                terms = [0.0] * len(local_ids)  # no absolute denominator
            else:
                terms = [2.0] * len(local_ids)
            records.append(
                {
                    "post_id": post,
                    "edge_ids": local_ids,
                    "edge_terms": terms,
                    "absolute_input_sum": sum(map(abs, terms)),
                }
            )
        boundaries.append({"name": str(tick), "control_tick": tick, "posts": records})
    result = e187.boundary_coverage(boundaries, ids, posts, known)
    assert len(result) == 25
    late = next(x for x in result if x["control_tick"] == 2000 and x["post_id"] == 10)
    assert late["known_absolute_fraction"] == pytest.approx(0.1)
    assert late["known_positive_absolute_fraction"] == pytest.approx(0.1)
    assert late["known_negative_absolute_fraction"] == 0
    assert late["all_absolute_sum"] == 10
    zero = next(x for x in result if x["control_tick"] == 0 and x["post_id"] == 10)
    assert zero["known_absolute_fraction"] is None and zero["all_absolute_sum"] == 0
    boundaries[-1]["posts"][0]["edge_ids"] = [1, 0]
    with pytest.raises(ValueError, match="CSR edge order"):
        e187.boundary_coverage(boundaries, ids, posts, known)


def test_direct_script_cli_with_empty_pythonpath():
    root = Path(__file__).resolve().parents[1]
    script = root / e187.OWNED[0]
    process = subprocess.run(
        [sys.executable, "-X", "utf8", str(script), "--help"],
        cwd=root,
        env={**os.environ, "PYTHONPATH": ""},
        capture_output=True,
        text=True,
        check=False,
    )
    assert process.returncode == 0, process.stderr
    assert "--source-review" in process.stdout and "--freeze" in process.stdout


def test_reused_cuda_fixed_lane_reduction_differs_from_float64_sum():
    weights = np.ones(35, np.float32)
    weights[0] = np.float32(1e8)
    weights[1] = np.float32(-1e8)
    rates = np.ones(35, np.float32)
    masks = np.ones(35, np.float32)
    current = e187.warp_current(weights, rates, masks, mode="packed")
    assert current == np.float32(16)
    assert current != sum(float(w) for w in weights)


def test_exact_rejects_prefix_difference_and_output_collision(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="prefix values"):
        e187.exact(np.array([1.0, np.nan]), np.array([2.0, np.nan]), "prefix")
    with pytest.raises(ValueError, match="NaN mask"):
        e187.exact(np.array([np.nan]), np.array([1.0]), "prefix")
    monkeypatch.setattr(e187, "ROOT", tmp_path)
    output = tmp_path / e187.OUTPUT
    output.mkdir(parents=True)
    with pytest.raises(ValueError, match="output exists"):
        e187.execute(tmp_path / "none", tmp_path / "none", output)
    assert not (output / "terminal.json").exists()


def test_graph_provenance_rejects_changed_calibration_source():
    graph = {
        "indptr": np.array([0, 0, 0, 3], np.int32),
        "indices": np.array([0, 1, 0], np.int32),
        "counts": np.ones(3, np.int64),
        "weights": np.array([-0.275, 0.275, -0.275], np.float32),
    }
    actual = {
        "indptr": graph["indptr"].copy(),
        "indices": graph["indices"].copy(),
        "data": np.array([-0.06, 0.09, -1.92], np.float32),
    }
    cal = {
        "motor_ids": np.array([2], np.int64),
        "edge_ids": np.array([0, 1], np.int64),
        "source_values": np.array([-1.92, 0.03], np.float32),
        "values": np.array([-0.06, 0.09], np.float32),
        "individual_edge_ids": np.array([1], np.int64),
    }
    params = {
        "global_edge_ids": np.array([1], np.int64),
        "calibration_positions": np.array([1], np.int64),
        "native_fit_values32": np.array([0.09], np.float32),
    }
    flags = e187.graph_provenance(graph, actual, cal, params, {0}, 3, 3)
    assert flags[0]["navigation"] and flags[1]["individual_fit"]
    cal["source_values"][0] = -0.03
    with pytest.raises(ValueError, match="calibration source"):
        e187.graph_provenance(graph, actual, cal, params, {0}, 3, 3)
