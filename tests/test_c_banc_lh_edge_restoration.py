"""Generated-only checks for the E250 existing-edge contrast."""

from __future__ import annotations

import copy
import itertools
import json
import time

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from tools import probe_c_banc_m3_lh_edge_restoration as e250


def small_patch():
    pre = np.tile(np.arange(5, dtype=np.int64), 5)
    post = np.repeat(np.arange(5, 10, dtype=np.int64), 5)
    counts = np.arange(1, 26, dtype=np.int64)
    target = (counts.astype(np.float64) * .03).astype(np.float32)
    original = (target.astype(np.float64) * .001).astype(np.float32)
    rows = np.r_[post, 11]
    cols = np.r_[pre, 2]
    data = np.r_[original, np.float32(-2)]
    matrix = csr_matrix((data, (rows, cols)), shape=(12, 12), dtype=np.float32)
    matrix.sort_indices()
    ids = np.array([np.flatnonzero((np.searchsorted(matrix.indptr, np.arange(matrix.nnz),
        side="right") - 1 == row) & (matrix.indices == column))[0]
        for row, column in zip(post, pre)], np.int64)
    patch = {"edge_ids": ids, "pre_ids": pre, "post_ids": post,
             "source_values": original, "target_values": target, "counts": counts}
    return matrix, patch


def test_exact_existing_edge_patch_and_guards():
    source, patch = small_patch()
    changed = e250.patch_existing_edges(source, patch, np.arange(5), np.arange(5, 10))
    assert e250.prior.same_bits(changed.data[patch["edge_ids"]], patch["target_values"])
    assert changed.data[-1] == source.data[-1] == np.float32(-2)
    assert e250.prior.same_bits(changed.indices, source.indices)
    assert e250.prior.same_bits(changed.indptr, source.indptr)
    assert e250.prior.same_bits(source.data[patch["edge_ids"]], patch["source_values"])
    bad = copy.deepcopy(patch)
    bad["pre_ids"][0] = 4
    with pytest.raises(ValueError):
        e250.patch_existing_edges(source, bad, np.arange(5), np.arange(5, 10))
    bad = copy.deepcopy(patch)
    bad["target_values"][0] = 0
    with pytest.raises(ValueError):
        e250.patch_existing_edges(source, bad, np.arange(5), np.arange(5, 10))


def test_identity_transfer_changes_only_hash_chain():
    base = {"model_hash": "old", "tick": 40000,
            "rate": np.array([1., 2.], np.float32),
            "adaptation": np.array([.1, .2], np.float32),
            "drive": np.array([3., 4.], np.float32),
            "output_mask": np.array([1., 0.], np.float32),
            "core_adaptation": {"schema": "c", "ids": np.array([0], np.int64),
                                "gain": 0., "base_model_hash": "a"},
            "motor_adaptation": {"schema": "m", "ids": np.array([1], np.int64),
                                 "gains": np.array([1.], np.float32),
                                 "source_core_identity": "b"},
            "motor_timescale": {"schema": "t", "ids": np.array([1], np.int64),
                                "log_tau": np.array([0.], np.float64),
                                "source_motor_identity": "c"}}
    new = copy.deepcopy(base)
    new["model_hash"] = "new"
    for key, field in (("core_adaptation", "base_model_hash"),
                       ("motor_adaptation", "source_core_identity"),
                       ("motor_timescale", "source_motor_identity")):
        new[key][field] += "2"
    got, fields = e250.transplant_identity(base, new)
    assert got["model_hash"] == "new"
    assert fields == {"model_hash": ["model_hash"],
                      "core_adaptation": ["base_model_hash"],
                      "motor_adaptation": ["source_core_identity"],
                      "motor_timescale": ["source_motor_identity"]}
    assert e250.prior.same_bits(got["rate"], base["rate"])
    assert e250.prior.same_bits(got["output_mask"], base["output_mask"])
    bad = copy.deepcopy(new)
    bad["motor_timescale"]["log_tau"][0] = 1
    with pytest.raises((AssertionError, ValueError)):
        e250.transplant_identity(base, bad)


def test_cut_and_original_response_contract():
    baseline = np.zeros((41, 4), np.float32)
    source = np.array([0], np.int64)
    bridge = np.array([1], np.int64)
    mask = np.array([1, 0, 1, 1], np.float32)
    assert e250.prior.same_bits(e250.prior.select_mask(mask, source, bridge, "intact"), mask)
    assert e250.prior.same_bits(e250.prior.select_mask(mask, source, bridge, "sensory"),
                                 np.array([0, 0, 1, 1], np.float32))
    high = baseline.copy()
    high[1:, 0] = 2
    high[1:, 3] = 1
    rates = {name: baseline.copy() for name in e250.CANDIDATE}
    rates["intact_high"] = high
    rates["bridge_cut_high"][1:, 3] = .5
    assert e250.prior.response_metrics(rates, np.array([3]))["H248_supported"]
    cut = e250.prior.cut_validation(rates["sensory_cut_low"],
                                     rates["sensory_cut_high"], source)
    assert cut["non_source_bitexact"]


def test_packed_buffer_decode_without_cuda():
    source, patch = small_patch()
    weights = e250.patch_existing_edges(source, patch, np.arange(5), np.arange(5, 10))
    unique, inverse = np.unique(weights.data.view(np.uint32), return_inverse=True)
    entries = weights.indices.astype(np.uint32) | (inverse.astype(np.uint32) << 18)

    class FakeTensor:
        def __init__(self, value):
            self.value = value

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self.value

    class FakeNetwork:
        _rowptr = FakeTensor(weights.indptr.astype(np.int32))
        _columns = FakeTensor(entries.view(np.int32))
        _weight_values = FakeTensor(unique.view(np.float32))

    receipt = e250.packed_csr_receipt(FakeNetwork(), weights)
    assert receipt["edges"] == weights.nnz
    FakeNetwork._columns = FakeTensor((entries ^ np.uint32(1)).view(np.int32))
    with pytest.raises(AssertionError):
        e250.packed_csr_receipt(FakeNetwork(), weights)


def test_partial_arm_preserves_scoped_counts_and_raw(tmp_path, monkeypatch):
    monkeypatch.setattr(e250, "N", 4)
    monkeypatch.setattr(e250, "CONTROLS", 3)

    class Mask:
        def __init__(self, value):
            self.value = value

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self.value

    class FakeNetwork:
        def __init__(self, state):
            self.restore(state)

        def restore(self, state):
            self.state = copy.deepcopy(state)
            self.output_mask = Mask(self.state["output_mask"])

        def snapshot(self):
            return copy.deepcopy(self.state)

        def set_muted(self, ids):
            self.state["output_mask"][:] = 1
            self.state["output_mask"][ids] = 0

        def readout(self):
            return self.state["rate"].copy()

        def advance(self, drive, steps):
            if self.state["tick"] == 2 * e250.STEPS:
                raise RuntimeError("partial neural fault")
            self.state["tick"] += steps
            self.state["drive"] = drive.copy()
            self.state["rate"][0] += 1

    state = {"schema": "fake", "tick": 0, "model_hash": "fake",
             "rate": np.zeros(4, np.float32),
             "adaptation": np.zeros(4, np.float32),
             "drive": np.zeros(4, np.float32),
             "output_mask": np.ones(4, np.float32)}
    net, ledger = FakeNetwork(state), e250.new_ledger()
    with pytest.raises(RuntimeError, match="partial neural fault"):
        e250.run_arm(net, state, "intact_low", tmp_path / "baseline" / "intact_low",
                     ledger, np.array([0], np.int64), np.array([1], np.int64),
                     1., 2., time.perf_counter(), scope="baseline")
    entry = ledger["arms"]["baseline:intact_low"]
    assert entry["entered_controls"] == 3
    assert entry["completed_controls"] == 2
    assert entry["unknown_partial"]
    assert ledger["science_ticks_requested"] == 3 * e250.STEPS
    assert ledger["science_ticks_completed"] == 2 * e250.STEPS
    raw = np.load(tmp_path / "baseline" / "intact_low" / "rate.npy")
    assert np.isfinite(raw[:3]).all() and np.isnan(raw[3]).all()
    assert json.loads((tmp_path / "advance-ledger.json").read_text())[
        "arms"]["baseline:intact_low"]["completed_controls"] == 2


def test_gate_rejects_hash_before_any_array_load(tmp_path):
    with pytest.raises(ValueError, match="hash differs"):
        e250.gate(tmp_path / "missing-freeze", "wrong", tmp_path / "review", "wrong")


def test_gate_accepts_frozen_posix_output_on_windows(tmp_path, monkeypatch):
    design = "docs/BANC_M3_E250_LH_EDGE_RESTORATION_DESIGN_20260925.md"
    owned = ["tools/probe_c_banc_m3_lh_edge_restoration.py",
             "tests/test_c_banc_lh_edge_restoration.py"]
    freeze = {"schema": e250.FREEZE_SCHEMA, "design": design,
              "design_sha256": e250.prior.sha256(e250.ROOT / design),
              "owned_files": owned, "input_sha256": {},
              "reused_source_sha256": {}, "preservation_sha256": {},
              "packages": {}, "n": e250.N, "initial_tick": 40000,
              "declared_edges": 11584852,
              "baseline_arms": list(e250.BASELINE),
              "candidate_arms": list(e250.CANDIDATE),
              "reference_arm": e250.DISK,
              "output": "verification/banc-natural-gait-m3-20260919/e250-lh-edge-restoration",
              "resource_limits": {"neural_constructors": 3,
                                  "scientific_neural_ticks": 7200,
                                  "setup_neural_ticks_per_constructor": 60}}
    freeze_path, review_path = tmp_path / "freeze.json", tmp_path / "review.json"
    e250.prior.write_json(freeze_path, freeze)
    digest = e250.prior.sha256(freeze_path)
    monkeypatch.setattr(e250, "FREEZE_SHA", digest)
    e250.prior.write_json(review_path, {"schema": e250.REVIEW_SCHEMA,
        "approved": True, "blocking_defects": [], "freeze_sha256": digest,
        "source_sha256": {name: e250.prior.sha256(e250.ROOT / name)
                          for name in owned}})
    assert e250.gate(freeze_path, digest, review_path,
                     e250.prior.sha256(review_path)) == freeze


def test_publication_failure_revokes_support_and_keeps_first_error(tmp_path):
    ledger = e250.new_ledger()
    result = {"validity": e250.VALIDITY, "H250_supported": True,
              "future_exact": True, "finalization_pending": True}
    e250.prior.write_json(tmp_path / "metrics.json", {"H250_supported": True})
    injected = RuntimeError("science failure")
    attempts = {"manifest": 0}

    def persist(path, value):
        if path.name == "output-manifest.json":
            attempts["manifest"] += 1
            if attempts["manifest"] == 1:
                raise OSError("late manifest failure")
        e250.prior.write_json(path, value)

    freeze = {"resource_limits": {"internal_s": 100,
                                   "output_bytes": 1 << 20}}
    got = e250.finish_receipts(tmp_path, freeze, ledger, result, [], injected,
                               time.perf_counter(), persist=persist)
    assert got is injected
    assert not result["H250_supported"] and not result["future_exact"]
    assert not json.loads((tmp_path / "metrics.json").read_text())["H250_supported"]
    assert json.loads((tmp_path / "result.json").read_text())["validity"] != e250.VALIDITY
    assert not (tmp_path / "output-manifest.json").exists()


def test_late_deadline_and_result_retry_failure_remove_stale_success(tmp_path, monkeypatch):
    ledger = e250.new_ledger()
    result = {"validity": e250.VALIDITY, "H250_supported": True,
              "future_exact": True, "finalization_pending": True}
    e250.prior.write_json(tmp_path / "metrics.json", {"H250_supported": True})
    clock = itertools.chain([.1, .2], itertools.repeat(101.))
    monkeypatch.setattr(e250.time, "perf_counter", lambda: next(clock))
    writes = {"result": 0}

    def persist(path, value):
        if path.name == "result.json":
            writes["result"] += 1
            if writes["result"] == 2:
                raise OSError("invalid result rewrite failed")
        e250.prior.write_json(path, value)

    freeze = {"resource_limits": {"internal_s": 100, "output_bytes": 1 << 20}}
    first = e250.finish_receipts(tmp_path, freeze, ledger, result, [], None, 0.,
                                 persist=persist)
    assert isinstance(first, TimeoutError)
    assert writes["result"] == 2
    assert not (tmp_path / "result.json").exists()
    assert not (tmp_path / "output-manifest.json").exists()
    assert not json.loads((tmp_path / "metrics.json").read_text())["H250_supported"]
    errors = json.loads((tmp_path / "errors.json").read_text())["errors"]
    assert any("invalid result rewrite failed" in item["message"] for item in errors)


def test_persistent_publication_failure_leaves_no_stale_success(tmp_path):
    ledger = e250.new_ledger()
    result = {"validity": e250.VALIDITY, "H250_supported": True,
              "future_exact": True, "finalization_pending": True}
    e250.prior.write_json(tmp_path / "result.json", result)
    e250.prior.write_json(tmp_path / "output-manifest.json", {"stale": True})
    e250.prior.write_json(tmp_path / "errors.json", {"errors": []})
    e250.prior.write_json(tmp_path / "metrics.json", {"H250_supported": True})

    def persist(path, value):
        if path.name in {"result.json", "errors.json", "output-manifest.json"}:
            raise OSError("persistent storage failure")
        e250.prior.write_json(path, value)

    freeze = {"resource_limits": {"internal_s": 100, "output_bytes": 1 << 20}}
    primary = RuntimeError("original science failure")
    got = e250.finish_receipts(tmp_path, freeze, ledger, result, [], primary,
                               time.perf_counter(), persist=persist)
    assert got is primary
    assert not (tmp_path / "result.json").exists()
    assert not (tmp_path / "errors.json").exists()
    assert not (tmp_path / "output-manifest.json").exists()
    assert not json.loads((tmp_path / "metrics.json").read_text())["H250_supported"]
