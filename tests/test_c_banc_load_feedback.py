"""Generated CSR and tiny CPU network checks for the E176 wrapper only."""

from __future__ import annotations

import copy

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from flylab.c.storage import StateStore
from tools.banc_m3_load_feedback import (
    LoadFeedbackTrial,
    _checkpoint_digest,
    _exact,
    fixed_rm_load_patch,
    patch_feedback_weights,
)
from tools.banc_m3_monotone_recruitment import MonotoneRecruitmentTrial
from tools.banc_m3_motor_timescale import MotorTimescaleNetwork
from tools.probe_c_banc_m3_core_rhythm import source_matrix
from tools.verify_c_banc_observation import exact as native_exact


def generated_weights():
    return csr_matrix(
        np.array(
            [
                [0.0, 0.15, 0.0, -0.10, 0.0, 0.0],
                [0.08, 0.0, 0.06, 0.0, 0.0, 0.0],
                [0.0, -0.10, 0.0, 0.12, 0.0, 0.0],
                [0.0, 0.0, 0.09, 0.0, -0.08, 0.0],
                [0.0, 0.0, 0.0, 0.11, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.07, 0.0],
            ],
            dtype=np.float32,
        )
    )


def generated_network(weights=None):
    if weights is None:
        weights = generated_weights()
    return MotorTimescaleNetwork(
        weights,
        np.array([0.025, 0.03, 0.028, 0.024, 0.027, 0.026], np.float32),
        np.ones(6, np.float32),
        np.full(6, 1.5, np.float32),
        np.full(6, 40.0, np.float32),
        adaptation_gain=8.0,
        adaptation_tau_s=0.3,
        dt=0.00025,
        device="cpu",
        cuda_implementation="reference",
        capture_steps=20,
        core_ids=np.array([1, 3, 4], np.int64),
        core_gain=0.0,
        motor_ids=np.array([3, 4], np.int64),
        motor_gains=np.array([1.1, 2.3], np.float32),
        motor_log_tau=np.array([0.0, 0.25], np.float64),
    )


def generated_patch(weights, edge=0, target=0.18):
    row = np.searchsorted(weights.indptr, edge, side="right") - 1
    return {
        "schema": "flylab.m3-rm-load-feedback-edges.v1",
        "edge_ids": np.array([edge], np.int64),
        "pre_ids": np.array([weights.indices[edge]], np.int64),
        "post_ids": np.array([row], np.int64),
        "source_values": np.array([weights.data[edge]], np.float32),
        "target_values": np.array([target], np.float32),
    }


def test_fixed_patch_has_exact_five_rows_and_independent_arrays():
    expected = [
        (844054, 107685, 12036, 0.00003, 0.03),
        (2400285, 30981, 32830, 0.00006, 0.06),
        (7908675, 107685, 108052, 0.00003, 0.03),
        (8578714, 30981, 117483, 0.00006, 0.06),
        (11240394, 30981, 153939, 0.00003, 0.03),
    ]
    patch = fixed_rm_load_patch()
    assert patch["schema"] == "flylab.m3-rm-load-feedback-edges.v1"
    for i, (edge, pre, post, before, after) in enumerate(expected):
        assert int(patch["edge_ids"][i]) == edge
        assert int(patch["pre_ids"][i]) == pre
        assert int(patch["post_ids"][i]) == post
        assert patch["source_values"][i].tobytes() == np.float32(before).tobytes()
        assert patch["target_values"][i].tobytes() == np.float32(after).tobytes()
    assert all(
        patch[key].dtype == np.int64 for key in ("edge_ids", "pre_ids", "post_ids")
    )
    patch["target_values"][0] = 99
    assert fixed_rm_load_patch()["target_values"][0] == np.float32(0.03)


def test_patch_changes_only_selected_csr_bytes_and_never_aliases():
    source = generated_weights()
    original = source.copy()
    patch = generated_patch(source)
    result = patch_feedback_weights(source, patch)
    assert result.data[0] == np.float32(0.18)
    np.testing.assert_array_equal(
        result.data[1:].view(np.uint32), original.data[1:].view(np.uint32)
    )
    assert result.indices.tobytes() == original.indices.tobytes()
    assert result.indptr.tobytes() == original.indptr.tobytes()
    assert result.shape == original.shape
    np.testing.assert_array_equal(
        source.data.view(np.uint32), original.data.view(np.uint32)
    )
    for key in ("data", "indices", "indptr"):
        assert not np.shares_memory(getattr(result, key), getattr(source, key))
    patch["target_values"][0] = np.float32(0.3)
    source.data[0] = np.float32(0.4)
    assert result.data[0] == np.float32(0.18)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda p: p["edge_ids"].__setitem__(0, 999),
        lambda p: p["pre_ids"].__setitem__(0, 5),
        lambda p: p["post_ids"].__setitem__(0, 2),
        lambda p: p["source_values"].__setitem__(0, 0.16),
        lambda p: p["target_values"].__setitem__(0, -0.18),
        lambda p: p["target_values"].__setitem__(0, np.nan),
        lambda p: p["source_values"].__setitem__(0, 0),
        lambda p: p.__setitem__("edge_ids", p["edge_ids"].astype(np.int32)),
        lambda p: p.__setitem__("target_values", p["target_values"].astype(np.float64)),
        lambda p: p.__setitem__("pre_ids", np.array([0, 0], np.int64)),
        lambda p: p.__setitem__("extra", 1),
        lambda p: p.__setitem__("schema", np.array([1, 2])),
        lambda p: p.pop("schema"),
    ],
)
def test_patch_rejects_bad_schema_dtype_address_and_values(mutate):
    weights = generated_weights()
    patch = generated_patch(weights)
    mutate(patch)
    with pytest.raises(ValueError):
        patch_feedback_weights(weights, patch)


def test_patch_rejects_duplicate_edge_and_noncanonical_or_wrong_source():
    weights = generated_weights()
    patch = generated_patch(weights)
    for key in ("edge_ids", "pre_ids", "post_ids", "source_values", "target_values"):
        patch[key] = np.repeat(patch[key], 2)
    with pytest.raises(ValueError):
        patch_feedback_weights(weights, patch)
    with pytest.raises(ValueError):
        patch_feedback_weights(weights.astype(np.float64), generated_patch(weights))
    with pytest.raises(ValueError):
        patch_feedback_weights(np.eye(2, dtype=np.float32), generated_patch(weights))


class TinyBody:
    def __init__(self):
        self.value = np.array([2.5, 3.5], dtype=np.float64)
        self.close_calls = 0
        self.close_error = None

    def snapshot(self):
        return {"value": self.value.copy()}

    def restore(self, saved):
        self.value = saved["value"].copy()

    def close(self):
        self.close_calls += 1
        if self.close_error:
            raise self.close_error


def tiny_trial(network=None, body=None):
    base = object.__new__(MonotoneRecruitmentTrial)
    base.network = network if network is not None else generated_network()
    base.body = body if body is not None else TinyBody()
    base.adapter = {"sensory_filter": np.array([0.2, 0.4], np.float64)}
    base.graph = "generated-graph"
    base.wall_s = 0.0
    base.closed = False
    return base


@pytest.fixture
def tiny_checkpoint_path(monkeypatch):
    restored_bases = []

    def snapshot(self):
        if self.closed:
            raise ValueError("Closed synthetic trial")
        return {
            "schema": self.schema,
            "graph_hash": self.graph,
            "wall_s": self.wall_s,
            "network": self.network.snapshot(),
            "body": self.body.snapshot(),
            "adapter": copy.deepcopy(self.adapter),
            "storage_normalization": {"ordered": (1, 2), "scalar": np.float32(0.5)},
        }

    def from_checkpoint(cls, graph, saved):
        if graph != saved["graph_hash"] or saved["schema"] != cls.schema:
            raise ValueError("Wrong synthetic source")
        base = tiny_trial()
        base.network.restore(saved["network"])
        base.wall_s = saved["wall_s"]
        base.body.restore(saved["body"])
        base.adapter = copy.deepcopy(saved["adapter"])
        restored_bases.append(base)
        return base

    monkeypatch.setattr(MonotoneRecruitmentTrial, "snapshot", snapshot)
    monkeypatch.setattr(
        MonotoneRecruitmentTrial, "from_checkpoint", classmethod(from_checkpoint)
    )
    return restored_bases


def test_full_state_transfer_identity_noop_and_future(tiny_checkpoint_path):
    base = tiny_trial()
    state = base.network.snapshot()
    state["rate"] = np.array([4.0, 3.0, 2.0, 5.0, 1.0, 6.0], np.float32)
    state["adaptation"] = np.array([0.2, 0.1, 0.3, 0.4, 0.15, 0.25], np.float32)
    state["drive"] = np.array([2.8, 2.5, 2.7, 3.1, 2.9, 3.0], np.float32)
    state["output_mask"][0] = 0
    state["tick"] = 17
    base.network.restore(state)
    source_identity = base.network.identity
    patch = generated_patch(source_matrix(base.network), target=np.float32(0.15))
    trial = LoadFeedbackTrial(base, patch)
    assert base.closed and not trial.closed
    assert trial.network.identity == source_identity
    after = trial.network.snapshot()
    for key in ("rate", "adaptation", "drive", "output_mask"):
        assert after[key].tobytes() == state[key].tobytes()
    assert after["tick"] == state["tick"]
    assert (
        after["core_adaptation"]["base_model_hash"]
        == state["core_adaptation"]["base_model_hash"]
    )
    drive = state["drive"].copy()
    for _ in range(2):
        base.network.advance(drive, steps=20)
        trial.network.advance(drive, steps=20)
        for key in ("rate", "adaptation", "drive", "output_mask", "tick"):
            _exact(base.network.snapshot()[key], trial.network.snapshot()[key])
    trial.close()
    trial.close()
    assert base.body.close_calls == 1


def test_patch_rebuild_changes_all_network_identity_metadata(tiny_checkpoint_path):
    base = tiny_trial()
    before = base.network.snapshot()
    trial = LoadFeedbackTrial(base, generated_patch(source_matrix(base.network)))
    after = trial.network.snapshot()
    assert before["model_hash"] != after["model_hash"]
    assert (
        before["core_adaptation"]["base_model_hash"]
        != after["core_adaptation"]["base_model_hash"]
    )
    assert (
        before["motor_adaptation"]["source_core_identity"]
        != after["motor_adaptation"]["source_core_identity"]
    )
    assert (
        before["motor_timescale"]["source_motor_identity"]
        != after["motor_timescale"]["source_motor_identity"]
    )
    saved = trial.snapshot()
    for key in ("model_hash", "core_adaptation", "motor_adaptation", "motor_timescale"):
        _exact(
            saved["load_feedback_patch"]["source_network_metadata"][key], before[key]
        )
    trial.close()


def test_snapshot_restore_exact_future_and_alias_protection(
    tiny_checkpoint_path, tmp_path
):
    base = tiny_trial()
    original = base.network.snapshot()
    original["rate"] = np.array([4.0, 3.0, 2.0, 5.0, 1.0, 6.0], np.float32)
    original["adaptation"] = np.array([0.2, 0.1, 0.3, 0.4, 0.15, 0.25], np.float32)
    original["drive"] = np.array([2.8, 2.5, 2.7, 3.1, 2.9, 3.0], np.float32)
    original["output_mask"][0] = 0
    original["tick"] = 17
    base.network.restore(original)
    patch = generated_patch(source_matrix(base.network))
    trial = LoadFeedbackTrial(base, patch)
    saved = trial.snapshot()
    StateStore.save(tmp_path / "generated-feedback", saved)
    from_disk = StateStore.load(tmp_path / "generated-feedback")
    clone = LoadFeedbackTrial.from_checkpoint("generated-graph", from_disk)
    _exact(from_disk, clone.snapshot())
    saved["network"]["rate"][0] = 99
    saved["load_feedback_patch"]["edge_patch"]["target_values"][0] = 9
    assert clone.snapshot()["network"]["rate"][0] != 99
    assert (
        clone.snapshot()["load_feedback_patch"]["edge_patch"]["target_values"][0] != 9
    )
    drive = original["drive"]
    for _ in range(2):
        trial.network.advance(drive, steps=20)
        clone.network.advance(drive, steps=20)
        _exact(trial.network.snapshot(), clone.network.snapshot())
    trial.wall_s = 1.25
    clone.wall_s = 9.5
    future_a = trial.snapshot()
    future_b = clone.snapshot()
    assert (
        future_a["load_feedback_patch"]["snapshot_sha256"]
        == future_b["load_feedback_patch"]["snapshot_sha256"]
    )
    native_exact(future_a, future_b)
    StateStore.save(tmp_path / "future-with-runtime", future_b)
    disk_future = StateStore.load(tmp_path / "future-with-runtime")
    restored_future = LoadFeedbackTrial.from_checkpoint("generated-graph", disk_future)
    assert restored_future.snapshot()["wall_s"] == 9.5
    native_exact(future_a, restored_future.snapshot())
    trial.close()
    clone.close()
    restored_future.close()


@pytest.mark.parametrize("wall_s", [None, -0.1, float("inf"), 2, "2.0"])
def test_runtime_wall_s_requires_finite_nonnegative_float(tiny_checkpoint_path, wall_s):
    trial = LoadFeedbackTrial(tiny_trial(), generated_patch(generated_weights()))
    saved = trial.snapshot()
    saved["wall_s"] = wall_s
    with pytest.raises(ValueError, match="runtime wall_s"):
        LoadFeedbackTrial.from_checkpoint("generated-graph", saved)
    trial.close()


def test_runtime_wall_s_omission_is_rejected(tiny_checkpoint_path):
    trial = LoadFeedbackTrial(tiny_trial(), generated_patch(generated_weights()))
    saved = trial.snapshot()
    saved.pop("wall_s")
    with pytest.raises(ValueError, match="runtime wall_s"):
        LoadFeedbackTrial.from_checkpoint("generated-graph", saved)
    trial.close()


def test_restore_preserves_runtime_even_when_future_checksum_ignores_it(
    tiny_checkpoint_path, monkeypatch
):
    trial = LoadFeedbackTrial(tiny_trial(), generated_patch(generated_weights()))
    saved = trial.snapshot()
    original_restore = MonotoneRecruitmentTrial.from_checkpoint

    def incorrect_runtime(cls, graph, source):
        base = original_restore(graph, source)
        base.wall_s += 1.0
        return base

    monkeypatch.setattr(
        MonotoneRecruitmentTrial, "from_checkpoint", classmethod(incorrect_runtime)
    )
    with pytest.raises(ValueError, match="wall_s float bits differ"):
        LoadFeedbackTrial.from_checkpoint("generated-graph", saved)
    restored_base = tiny_checkpoint_path[-1]
    assert restored_base.closed and restored_base.body.close_calls == 1
    trial.close()


def test_nested_or_other_subclass_rejected_before_ownership(tiny_checkpoint_path):
    first = LoadFeedbackTrial(tiny_trial(), generated_patch(generated_weights()))
    before = first.snapshot()
    independent = LoadFeedbackTrial.from_checkpoint("generated-graph", before)
    with pytest.raises(TypeError, match="Exact original"):
        LoadFeedbackTrial(first, generated_patch(source_matrix(first.network)))
    assert not first.closed and first.body.close_calls == 0
    _exact(first.snapshot(), before)
    drive = np.full(6, 2.7, np.float32)
    first.network.advance(drive, steps=20)
    independent.network.advance(drive, steps=20)
    _exact(first.network.snapshot(), independent.network.snapshot())

    class Unsupported(MonotoneRecruitmentTrial):
        pass

    unsupported = object.__new__(Unsupported)
    unsupported.__dict__.update(tiny_trial().__dict__)
    with pytest.raises(TypeError, match="Exact original"):
        LoadFeedbackTrial(unsupported, generated_patch(generated_weights()))
    assert not unsupported.closed and unsupported.body.close_calls == 0
    first.close()
    independent.close()
    unsupported.close()


@pytest.mark.parametrize(
    "tamper",
    [
        lambda s: s["network"]["rate"].__setitem__(0, 4.0),
        lambda s: s["network"].__setitem__(
            "model_hash", s["network"]["model_hash"] + "x"
        ),
        lambda s: s["network"]["core_adaptation"].__setitem__("gain", 4.0),
        lambda s: s["load_feedback_patch"]["edge_patch"]["pre_ids"].__setitem__(0, 5),
        lambda s: s["load_feedback_patch"]["edge_patch"].__setitem__(
            "target_values",
            s["load_feedback_patch"]["edge_patch"]["target_values"].astype(np.float64),
        ),
        lambda s: s["network"].pop("drive"),
        lambda s: s.__setitem__("extra", 1),
    ],
)
def test_checkpoint_tamper_rejected(tiny_checkpoint_path, tamper):
    trial = LoadFeedbackTrial(tiny_trial(), generated_patch(generated_weights()))
    saved = trial.snapshot()
    tamper(saved)
    with pytest.raises(ValueError):
        LoadFeedbackTrial.from_checkpoint("generated-graph", saved)
    trial.close()


def test_owner_transfer_on_failure_preserves_original_and_cleanup_error(
    tiny_checkpoint_path,
):
    base = tiny_trial()
    base.body.close_error = RuntimeError("cleanup marker")
    wrong = generated_patch(source_matrix(base.network))
    wrong["source_values"][0] = np.float32(4.0)
    with pytest.raises(ValueError, match="source value") as failure:
        LoadFeedbackTrial(base, wrong)
    assert base.closed
    assert base.body.close_calls == 1
    assert any("cleanup marker" in note for note in failure.value.__notes__)


def test_closed_base_cannot_transfer_body(tiny_checkpoint_path):
    base = tiny_trial()
    base.closed = True
    with pytest.raises(ValueError, match="Open monotone"):
        LoadFeedbackTrial(base, generated_patch(generated_weights()))
    assert base.body.close_calls == 0


def test_restore_failure_after_owner_transfer_closes_body_once(tiny_checkpoint_path):
    trial = LoadFeedbackTrial(tiny_trial(), generated_patch(generated_weights()))
    saved = trial.snapshot()
    saved["network"]["model_hash"] = "bad candidate identity"
    saved["load_feedback_patch"]["snapshot_sha256"] = _checkpoint_digest(saved)
    with pytest.raises(ValueError, match="whole load-feedback checkpoint"):
        LoadFeedbackTrial.from_checkpoint("generated-graph", saved)
    restored_base = tiny_checkpoint_path[-1]
    assert restored_base.closed
    assert restored_base.body.close_calls == 1
    trial.close()
