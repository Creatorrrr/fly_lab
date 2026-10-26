"""Generated E178 contracts only; no E177 state or graph values are loaded."""

import hashlib
import json

import numpy as np
import pytest

from tools import analyze_c_banc_m3_saved_current as e178


def _hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_warp_current_preserves_float32_lane_and_reduction_order():
    weights = np.ones(35, np.float32)
    rates = np.ones(35, np.float32)
    masks = np.ones(35, np.float32)
    weights[0] = np.float32(1e8)
    weights[1] = np.float32(-1e8)
    weights[32] = np.float32(1)
    weights[33] = np.float32(1)
    # Small terms are lost in lane accumulation and shuffle, whereas a
    # sequential or float64 reduction gives a different answer.
    assert e178.warp_current(weights, rates, masks) == np.float32(16)
    assert e178.warp_current(
        np.array([], np.float32), np.array([], np.float32), np.array([], np.float32)
    ) == np.float32(0)
    with pytest.raises(ValueError):
        e178.warp_current(weights, rates[:-1], masks)
    with pytest.raises(ValueError):
        e178.warp_current(weights.astype(np.float64), rates, masks)


def test_warp_multiplies_mask_before_weight_and_preserves_negative_current():
    weights = np.array([-3, 2, 5], np.float32)
    rates = np.array([4, 7, 9], np.float32)
    masks = np.array([1, 0, 0], np.float32)
    assert e178.warp_current(weights, rates, masks) == np.float32(-12)


def test_cuda_ftz_and_torch_premask_are_distinct():
    tiny = np.float32(1e-40)
    weights = np.array([1], np.float32)
    rates = np.array([tiny], np.float32)
    masks = np.array([1e10], np.float32)
    assert e178.warp_current(weights, rates, masks, mode="packed") > 0
    assert e178.warp_current(weights, rates, masks, mode="reference") == 0
    assert e178._cuda_mul(np.float32(-tiny), np.float32(1e20)).view(
        np.uint32
    ) == np.float32(-0.0).view(np.uint32)
    assert e178._cuda_mul(np.float32(1e-20), np.float32(1e-20)) == 0
    with pytest.raises(ValueError, match="implementation"):
        e178.warp_current(weights, rates, masks, mode="unknown")


def test_edge_descriptive_term_is_retained_when_cuda_ftz_discards_it():
    piece = e178.decompose_row(
        np.array([0], np.int64),
        np.array([0], np.int32),
        np.array([1e20], np.float32),
        np.array([1e-40], np.float32),
        np.array([1], np.float32),
        np.array([1], np.int64),
        np.array([0.275], np.float32),
        {},
        mode="packed",
    )
    assert piece["edges"][0]["contribution"] > 0
    assert piece["edges"][0]["cuda_product_float32"] == 0
    assert piece["summary"]["ftz_effect_count"] == 1


def test_row_raw_sign_groups_overlap_and_masked_difference():
    piece = e178.decompose_row(
        np.array([10, 11, 12, 13, 14], np.int64),
        np.array([1, 2, 3, 4, 5], np.int32),
        np.array([-0.06, 0.03, -0.03, 0.09, 0.03], np.float32),
        np.array([2, 3, 4, 5, 8], np.float32),
        np.array([1, 1, 0, 1, 1], np.float32),
        np.array([2, 1, 1, 3, 1], np.int64),
        np.array([-0.55, 0.275, -0.275, 0.825, 0.275], np.float32),
        {
            10: {"feedback": True, "individual_fit": True},
            11: {"navigation": True, "other_calibration": True},
            12: {"individual_fit": True},
            13: {"other_calibration": True},
        },
    )
    edges = piece["edges"]
    assert [e["exclusive_group"] for e in edges] == list(e178.GROUPS)
    assert edges[0]["overlapping_flags"]["individual_fit"] is True
    assert edges[1]["overlapping_flags"]["other_calibration"] is True
    assert edges[2]["contribution"] == 0
    assert edges[2]["unmasked_contribution"] < 0
    assert piece["summary"]["negative_sum"] < 0
    assert piece["summary"]["positive_sum"] > 0
    assert piece["summary"]["double_sum"] == pytest.approx(
        piece["summary"]["positive_sum"] + piece["summary"]["negative_sum"]
    )
    assert piece["summary"]["top_negative"] == [10]
    assert piece["summary"]["top_positive"][0] == 13
    assert edges[0]["rate_base_weight"] == pytest.approx(-0.06)
    assert edges[0]["graph_weight_mV"] == pytest.approx(-0.55)


def test_row_invalid_input_and_exact_boundary_bits():
    with pytest.raises(ValueError, match="noncontiguous"):
        e178.decompose_row(
            np.array([1, 3]),
            np.array([0, 1]),
            np.ones(2, np.float32),
            np.ones(2, np.float32),
            np.ones(2, np.float32),
            np.ones(2, np.int64),
            np.ones(2, np.float32),
            {},
        )
    with pytest.raises(ValueError, match="nonfinite"):
        e178.decompose_row(
            np.array([1]),
            np.array([0]),
            np.array([np.nan], np.float32),
            np.ones(1, np.float32),
            np.ones(1, np.float32),
            np.ones(1, np.int64),
            np.ones(1, np.float32),
            {},
        )
    with pytest.raises(ValueError, match="bit mismatch"):
        e178.exact_bits(np.float32(0.0), np.float32(-0.0), "signed zero")


def test_source_gate_rejects_unreviewed_code_before_array_load(tmp_path, monkeypatch):
    for name in (*e178.OWNED, "docs/design.md", "inputs/a.npy"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"synthetic")
    freeze = {
        "schema": "flylab.m3.e178.saved-current.v1",
        "owned_files": list(e178.OWNED),
        "source_gate_schema": "flylab.m3.e178.parent-source-review.v1",
        "numeric_contract": {
            "schema": "flylab.m3.e178.cuda-current-arithmetic.v1",
            "csr_ftz": True,
            "fmad": False,
            "packed_mask_product": "torch_float32_preserve_subnormals",
            "reference_mask_product": "cuda_ftz_float32",
            "implementation_request": "packed",
            "csr_eligibility": "n<=2**18 and unique_weight_bits<=2**14 and nnz<2**31",
            "descriptive_contribution": "float64(weight)*float32(rate*mask)",
            "bit_exact_required": True,
        },
        "design_path": "docs/design.md",
        "design_sha256": _hash(tmp_path / "docs/design.md"),
        "input_sha256": {"inputs/a.npy": _hash(tmp_path / "inputs/a.npy")},
        "reused_source_sha256": {},
    }
    freeze_path = tmp_path / "freeze.json"
    freeze_path.write_text(json.dumps(freeze), encoding="utf-8")
    review = {
        "schema": freeze["source_gate_schema"],
        "accepts_saved_current": True,
        "blocking_defects": [],
        "freeze_sha256": _hash(freeze_path),
        "numeric_contract": freeze["numeric_contract"],
        "source_sha256": {name: _hash(tmp_path / name) for name in e178.OWNED},
    }
    review_path = tmp_path / "review.json"
    review_path.write_text(json.dumps(review), encoding="utf-8")
    monkeypatch.setattr(
        np, "load", lambda *args, **kwargs: pytest.fail("original array read")
    )
    assert (
        e178._review_and_hashes(
            tmp_path, freeze_path, _hash(freeze_path), review_path, _hash(review_path)
        )
        == freeze
    )
    review["blocking_defects"] = ["unresolved"]
    review_path.write_text(json.dumps(review), encoding="utf-8")
    with pytest.raises(ValueError, match="source gate"):
        e178._review_and_hashes(
            tmp_path, freeze_path, _hash(freeze_path), review_path, _hash(review_path)
        )
    review["blocking_defects"] = []
    review_path.write_text(json.dumps(review), encoding="utf-8")
    (tmp_path / "inputs/a.npy").write_bytes(b"changed")
    with pytest.raises(ValueError, match="input_sha256 mismatch"):
        e178._review_and_hashes(
            tmp_path, freeze_path, _hash(freeze_path), review_path, _hash(review_path)
        )


def test_npz_unavailable_motor_tau_is_allowed_but_used_parameters_are_finite(tmp_path):
    path = tmp_path / "cohort.npz"
    np.savez(
        path,
        observed_ids=np.array([7], np.int64),
        motor_log_tau=np.array([np.nan], np.float64),
        tau=np.array([0.02], np.float32),
    )
    header = {
        "observed_ids": {"shape": [1], "dtype": "<i8"},
        "motor_log_tau": {"shape": [1], "dtype": "<f8"},
        "tau": {"shape": [1], "dtype": "<f4"},
    }
    assert np.isnan(e178._load_npz(path, header)["motor_log_tau"][0])
    np.savez(
        path,
        observed_ids=np.array([7], np.int64),
        motor_log_tau=np.array([np.nan], np.float64),
        tau=np.array([np.nan], np.float32),
    )
    with pytest.raises(ValueError, match="nonfinite"):
        e178._load_npz(path, header)


def test_full_csr_provenance_applies_navigation_then_calibration_then_fit():
    graph = {
        "indptr": np.array([0, 0, 0, 5], np.int32),
        "indices": np.array([0, 1, 2, 0, 1], np.int32),
        "counts": np.ones(5, np.int64),
        "weights": np.array([-0.275, 0.275, 0.275, -0.275, 0.275], np.float32),
    }
    actual = {
        "indptr": graph["indptr"].copy(),
        "indices": graph["indices"].copy(),
        "data": np.array([-0.06, 0.09, 0.03, -1.92, 0.03], np.float32),
    }
    calibration = {
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
    freeze = {
        "node_count": 3,
        "edge_count": 5,
        "feedback_patch": {
            "edge_ids": [0, 1, 2, 3, 4],
            "source_values": [-0.06, 0.09, 0.03, -1.92, 0.03],
        },
    }
    provenance, counts = e178._weight_provenance(
        graph, actual, calibration, params, freeze, {0}
    )
    e178.exact_bits(provenance["expected_baseline"], actual["data"], "synthetic CSR")
    assert provenance["flags"][0] == {
        "navigation": True,
        "other_calibration": True,
        "feedback": True,
    }
    assert provenance["flags"][1] == {
        "other_calibration": True,
        "individual_fit": True,
        "feedback": True,
    }
    assert counts["navigation_edges"] == 2
    wrong = {**calibration, "source_values": np.array([-0.03, 0.03], np.float32)}
    with pytest.raises(ValueError, match="source values"):
        e178._weight_provenance(graph, actual, wrong, params, freeze, {0})
