"""Generated-only E256 six-edge, metric, and failure-publication checks."""

from __future__ import annotations

import copy
import time

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from tools import probe_c_banc_m3_lh_neural_response as io
from tools import probe_c_banc_m3_lh_timescale_partition as mod


def _fake_patch():
    ids = np.arange(25, dtype=np.int64)
    post = np.array([100] * 10 + [101] * 4 + [111970] * 6 + [103] * 5)
    counts = np.ones(25, np.int64)
    counts[15] = 3
    new = (counts.astype(np.float64) * .03).astype(np.float32)
    old = (new.astype(np.float64) * .001).astype(np.float32)
    patch = {"edge_ids": ids, "pre_ids": np.arange(25, dtype=np.int64),
             "post_ids": post, "counts": counts, "source_values": old,
             "target_values": new}
    intrinsic = np.r_[100, 101, 103, np.arange(200, 235)]
    original24 = np.arange(200, 224)
    roster = {"intrinsic_timescale_node_rows": intrinsic.tolist(),
              "core_adaptation_node_rows": [100, 101, 103, 111970],
              "original_24_node_rows": original24.tolist()}
    chosen = np.flatnonzero(post == 111970)
    prepared = {"original24_rows": original24.tolist(),
                "intrinsic_timescale38_rows": intrinsic.tolist(),
                "selected_post_rows": [111970],
                "selected_edges": {key: np.asarray(values)[chosen].tolist()
                                   for key, values in patch.items()},
                "preserved_edges": ids[post != 111970].tolist()}
    return patch, roster, prepared


def test_partition_is_existing_six_not_noncore_or_biological_label():
    patch, roster, prepared = _fake_patch()
    selected, preserved = mod.select_partition(patch, roster, prepared)
    assert selected["edge_ids"].tolist() == list(range(14, 20))
    assert len(preserved) == 19
    assert 111970 in roster["core_adaptation_node_rows"]
    tampered = copy.deepcopy(roster)
    tampered["intrinsic_timescale_node_rows"][-1] = 111970
    with pytest.raises(ValueError, match="membership"):
        mod.select_partition(patch, tampered, prepared)


def test_six_only_csr_preserves_other_19_and_topology():
    patch, roster, prepared = _fake_patch()
    selected, preserved = mod.select_partition(patch, roster, prepared)
    n = 111971
    # Reorder for an actual CSR row; selected IDs must then be14..19.
    indptr = np.zeros(n + 1, np.int32)
    indptr[101:] = 10
    indptr[102:] = 14
    indptr[104:] = 19
    indptr[111971:] = 25
    # The fixed preparation is ordered by postsynaptic row: the six selected
    # entries need to be the last row here, so keep the other five at row103.
    patch["post_ids"] = np.array([100] * 10 + [101] * 4 + [103] * 5
                                 + [111970] * 6)
    patch["edge_ids"] = np.arange(25, dtype=np.int64)
    prepared["selected_edges"] = {key: np.asarray(values)[19:].tolist()
                                  for key, values in patch.items()}
    prepared["preserved_edges"] = list(range(19))
    selected, preserved = mod.select_partition(patch, roster, prepared)
    original = csr_matrix((patch["source_values"], patch["pre_ids"], indptr),
                          shape=(n, n), dtype=np.float32)
    changed = mod.patch_six(original, selected, preserved)
    assert mod.e248.same_bits(changed.indptr, original.indptr)
    assert mod.e248.same_bits(changed.indices, original.indices)
    assert mod.e248.same_bits(changed.data[preserved], original.data[preserved])
    assert mod.e248.same_bits(changed.data[selected["edge_ids"]],
                              selected["target_values"])
    bad = selected.copy()
    bad["counts"] = selected["counts"].copy()
    bad["counts"][0] = 100
    with pytest.raises(ValueError):
        mod.patch_six(original, bad, preserved)


def test_q_rms_and_signed_parts_without_a_second_neural_solve():
    zeros = np.zeros((41, 24), np.float32)
    one = zeros.copy()
    one[1:, 0] = 20.
    five = zeros.copy()
    five[1:, 0] = 100.
    baseline = {a: zeros for a in mod.BASELINE}
    six = {a: one for a in mod.BASELINE}
    all25 = {a: five for a in mod.BASELINE}
    q, raw = mod.q_metrics(baseline, six, all25, baseline,
                           np.arange(24, dtype=np.int64), n=24)
    assert q["Q25_positive"] is True
    assert q["Q80_protection_pass"] is True
    assert q["Q6_over_Q25"] == pytest.approx(.2)
    assert raw["six_normalized"].shape == (2, 40, 24)
    assert q["six_positive_sum_normalized"] == pytest.approx(8.)
    negative = {a: -one for a in mod.BASELINE}
    q2, _ = mod.q_metrics(baseline, negative, all25, baseline,
                          np.arange(24, dtype=np.int64), n=24)
    assert q2["six_negative_sum_normalized"] == pytest.approx(-8.)
    assert q2["six_signed_sum_normalized"] == pytest.approx(-8.)
    unavailable, _ = mod.q_metrics(baseline, six, baseline, baseline,
                                   np.arange(24, dtype=np.int64), n=24)
    assert unavailable["Q25_positive"] is False
    assert unavailable["Q80_protection_pass"] is False


def test_fixed_response_and_interaction_criteria_with_q_protection():
    zeros = np.zeros((41, 56), np.float32)
    high = zeros.copy()
    high[1:] = .04
    cut_high = zeros.copy()
    cut_high[1:] = .03
    arms = {name: zeros.copy() for name in mod.CANDIDATE}
    arms["intact_high"] = high
    arms["bridge_cut_high"] = cut_high
    result = mod.e248.response_metrics(arms, np.arange(56))
    assert result["response_R"] == pytest.approx(.04 / 200.)
    assert result["interaction_D"] == pytest.approx(.01 / 200.)
    assert result["H248_supported"] is False  # D misses its absolute floor.
    arms["bridge_cut_high"] = zeros.copy()
    result = mod.e248.response_metrics(arms, np.arange(56))
    assert result["H248_supported"] is True


@pytest.mark.parametrize("fault", ["cleanup", "manifest"])
def test_fail_closed_nested_h_and_q_after_late_error(tmp_path, fault):
    (tmp_path / "raw.npy").write_bytes(b"science evidence")
    metrics = {"H256_supported": True,
               "candidate_response": {"H248_supported": True,
                                      "response_R": .001, "interaction_D": .002},
               "Q": {"Q80_protection_pass": True, "Q6": .001, "Q25": .01},
               "candidate_disk_future_exact": True,
               "candidate_sensory_cut_non_source_exact": True,
               "candidate_bridge_cut_nonbridge_exact": True}
    io.write_json(tmp_path / "metrics.json", copy.deepcopy(metrics))
    ledger = mod.e250.new_ledger()
    ledger["setup_capture_ticks_completed"] = 0
    result = {"validity": mod.VALIDITY, "H256_supported": True,
              "future_exact": True, "metrics": copy.deepcopy(metrics)}

    def persist(path, value):
        if fault == "manifest" and path.name == "output-manifest.json":
            raise OSError("late manifest")
        io.write_json(path, value)

    original = OSError("cleanup primary") if fault == "cleanup" else None
    error = mod.finish_receipts(
        tmp_path, {"resource_limits": {"internal_s": 34,
                                       "output_bytes": 100000}},
        ledger, result, [], original, time.perf_counter(), persist=persist)
    assert isinstance(error, OSError)
    assert str(error) == ("cleanup primary" if original else "late manifest")
    assert (tmp_path / "raw.npy").read_bytes() == b"science evidence"
    published = io.read_json(tmp_path / "result.json")
    assert published["H256_supported"] is False
    assert published["metrics"]["H256_supported"] is False
    assert published["metrics"]["candidate_response"]["H248_supported"] is False
    assert published["metrics"]["Q"]["Q80_protection_pass"] is False
    assert published["metrics"]["Q"]["Q6"] == .001
    assert io.read_json(tmp_path / "metrics.json")["H256_supported"] is False
    if fault == "manifest":
        assert not (tmp_path / "output-manifest.json").exists()


def test_gate_rejects_unapproved_hash_before_original_arrays(tmp_path):
    with pytest.raises(ValueError, match="SHA"):
        mod.gate(tmp_path / "missing", "bad", tmp_path / "review", "bad",
                 tmp_path / "output")


def test_execute_preserves_science_fault_through_cleanup_and_publication(
        tmp_path, monkeypatch):
    out = tmp_path / "out"

    def fail_before_model():
        (out / "partial.bin").write_bytes(b"partial science receipt")
        raise ValueError("first scientific fault")

    with monkeypatch.context() as patch:
        patch.setattr(mod.e248, "_configure_cpu", fail_before_model)
        patch.setattr(mod.gc, "collect", lambda: (_ for _ in ()).throw(
            OSError("secondary cleanup fault")))
        with pytest.raises(RuntimeError, match="first scientific fault"):
            mod.execute({"resource_limits": {"internal_s": 34,
                                            "output_bytes": 100000}}, out)

    assert (out / "partial.bin").read_bytes() == b"partial science receipt"
    result = io.read_json(out / "result.json")
    assert result["validity"] == "invalid_or_incomplete"
    assert result["H256_supported"] is False
    assert result["future_exact"] is False
    assert result["first_error"] == {"type": "ValueError",
                                     "message": "first scientific fault"}
    errors = io.read_json(out / "errors.json")["errors"]
    assert any(item["type"] == "ValueError" for item in errors)
    assert any(item.get("stage") == "cleanup" and item["type"] == "OSError"
               for item in errors)
    assert "partial.bin" in io.read_json(out / "output-manifest.json")["files"]
