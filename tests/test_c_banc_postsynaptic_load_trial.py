"""E313 generated fake whole ownership, nonzero restore, and metadata checks."""
import copy
import hashlib
import time
from types import SimpleNamespace

import numpy as np
import pytest

from tests import test_c_banc_ssp_recovery_trial as old
from tools import banc_m3_postsynaptic_load_trial as trial
from tools import banc_m3_ssp_recovery_synapses as ssp

BEGAN = time.perf_counter()
COUNTS = {"fake_networks": 0, "fake_ticks": 0, "whole_restores": 0, "original": 0, "CUDA": 0, "body": 0, "fit": 0}


def load_metadata(source, selection, threshold, tau, load_lambda):
    ids = np.unique(selection["post_ids"])
    return {"schema": trial.NETWORK_SCHEMA, "precision": "fake-post-load", "source_model_hash": source,
        "selection": copy.deepcopy(selection), "target_ids": ids, "threshold32": threshold[ids].copy(), "tau32": tau[ids].copy(), "load_lambda": float(load_lambda)}


def load_identity(metadata):
    return hashlib.sha256(trial.provenance_hash(metadata).encode()).hexdigest()


def checked_load_metadata(metadata):
    assert metadata["schema"] == trial.NETWORK_SCHEMA and metadata["precision"] == "fake-post-load"
    return copy.deepcopy(metadata)


class FakeLoad(old.FakeResource):
    def __init__(self, weights, tau, a, threshold, cap, **kwargs):
        COUNTS["fake_networks"] += 1
        super().__init__(weights, tau, a, threshold, cap, **kwargs)
        self.load_lambda = kwargs["load_lambda"]
        self.state["recurrent_resource"] = ssp.resource_metadata(self.source_timescale_identity, self._selection)
        self.state["postsynaptic_load"] = load_metadata(self.source_timescale_identity, self._selection, threshold, tau, self.load_lambda)
        self.state["postsynaptic_load_state"] = np.zeros(len(np.unique(self._selection["post_ids"])), np.float64)
        self.identity = load_identity(self.state["postsynaptic_load"])
        self.state["model_hash"] = self.identity
        self.immutable = trial.metadata_without_state(self.state)

    def import_base_snapshot(self, saved):
        trial.validate_neutral(saved)
        inherited = trial.metadata_without_state(self.state)
        inherited.pop("recurrent_resource"); inherited.pop("postsynaptic_load")
        inherited["model_hash"] = self.source_timescale_identity
        trial.require_equal(inherited, trial.metadata_without_state(saved), "Fake original metadata exact")
        for key in ("rate", "adaptation", "drive", "output_mask", "tick"):
            self.state[key] = copy.deepcopy(saved[key])

    def restore(self, saved):
        if self.fail_restore:
            raise RuntimeError("fake restore failed")
        trial.require_equal(trial.metadata_without_state(saved), self.immutable, "Fake immutable new metadata")
        self.state = copy.deepcopy(saved)

    def advance(self, drive, steps=1):
        COUNTS["fake_ticks"] += steps
        self.state["postsynaptic_load_state"] += steps * .001
        super().advance(np.asarray(drive, np.float32) * (1. - self.load_lambda * .1), steps)


API = SimpleNamespace(SCHEMA=trial.NETWORK_SCHEMA, PRECISION="fake-post-load", PostsynapticLoadNetwork=FakeLoad,
    checked_selection=old.checked_selection, load_metadata=load_metadata, checked_load_metadata=checked_load_metadata, load_identity=load_identity)


@pytest.fixture(autouse=True)
def generated_seams(monkeypatch):
    monkeypatch.setattr(trial, "MotorTimescaleNetwork", old.FakeNetwork)
    monkeypatch.setattr(trial, "network_api", lambda: API)
    monkeypatch.setattr(trial, "source_matrix", lambda network: network.matrix.copy())
    monkeypatch.setattr(ssp, "resource_metadata", old.resource_metadata)
    monkeypatch.setattr(trial.MonotoneRecruitmentTrial, "snapshot", old.old.fake_snapshot)
    monkeypatch.setattr(trial.MonotoneRecruitmentTrial, "from_checkpoint", classmethod(old.from_base))
    FakeLoad.fail = FakeLoad.mutate = FakeLoad.fail_restore = FakeLoad.fault = False


def wrap(base=None, value=1.):
    return trial.PostsynapticLoadTrial(base or old.base_session(), np.array([0, 1], np.int64),
        np.array([0, 1], np.int64), np.zeros(2, np.float64), np.full(2, .15, np.float64), value)


def test_new_layer_ownership_and_lambda_boundary_preserve_original_whole():
    states = []
    for value in (0., 1.):
        base = old.base_session()
        before, body = base.snapshot(), base.body
        session = wrap(base, value)
        assert base.closed and session.body is body and body.close_calls == 0
        trial.require_equal(trial.remove_resource_layer(session.snapshot()), before, "Original whole fields exact")
        assert np.all(session.network.snapshot()["postsynaptic_load_state"] == 0) and np.all(session.network.snapshot()["synaptic_resource"] == 1)
        states.append(session)
    assert states[0].network.identity != states[1].network.identity
    for session in states:
        session.network.advance(np.ones(4, np.float32), 20)
    assert not np.array_equal(states[0].network.state["rate"], states[1].network.state["rate"])


def test_actual_nonzero_fake_whole_disk_restore_and_next_future(tmp_path):
    from flylab.c.storage import StateStore
    session = wrap()
    session.network.advance(np.array([4., 2., 3., 1.], np.float32), 5)
    session.control_tick = 7
    session.body.state["clock"] = 7
    session.adapter.state["bank"] += .2
    session.sensory["filter"] += .1
    saved = session.snapshot()
    session.save(tmp_path / "saved")
    loaded = StateStore.load(tmp_path / "saved")
    trial.require_saved_whole_equal(loaded, saved, "Whole fake disk")
    restored = trial.PostsynapticLoadTrial.from_checkpoint(session.graph, loaded)
    COUNTS["whole_restores"] += 1
    trial.require_saved_whole_equal(restored.snapshot(), saved, "Nonzero h/R/bank/filter/body/wall")
    for owner in (session, restored):
        owner.network.advance(np.ones(4, np.float32), 3)
    trial.require_saved_whole_equal(restored.snapshot(), session.snapshot(), "Same fake future")


@pytest.mark.parametrize("field", ["old_schema", "lambda", "h_nan", "h_dtype", "h_high", "h_order", "R", "parameters", "identity"])
def test_corrupt_restore_is_rejected_before_new_body_and_keeps_live_whole(field):
    session = wrap()
    original = session.snapshot()
    saved = copy.deepcopy(original)
    if field == "old_schema":
        saved["schema"] = old.trial.TRIAL_SCHEMA
    elif field == "lambda":
        saved[trial.PATCH_KEY]["load_lambda"] = 0.
    elif field == "h_nan":
        saved["network"]["postsynaptic_load_state"][0] = np.nan
    elif field == "h_dtype":
        saved["network"]["postsynaptic_load_state"] = saved["network"]["postsynaptic_load_state"].astype(np.float32)
    elif field == "h_high":
        saved["network"]["postsynaptic_load_state"][0] = 1.1
    elif field == "h_order":
        saved["network"]["postsynaptic_load"]["target_ids"] = np.array([1, 0], np.int64)
    elif field == "R":
        saved["network"]["synaptic_resource"][0] = .9
    elif field == "parameters":
        saved[trial.PATCH_KEY]["original_threshold32"][0] = 3.
    else:
        saved["network"]["model_hash"] = "old"
    bodies = old.old.COUNTS["fake_bodies"]
    with pytest.raises((ValueError, AssertionError)):
        session.restore(saved)
    assert old.old.COUNTS["fake_bodies"] == bodies
    trial.require_equal(session.snapshot(), original, "Failed atomic whole restore unchanged")


def test_replacement_failure_keeps_old_and_commit_cleanup_failure_is_explicit():
    session = wrap()
    saved, old_body = session.snapshot(), session.body
    FakeLoad.fail = True
    with pytest.raises(RuntimeError):
        session.restore(saved)
    assert session.body is old_body and old_body.close_calls == 0
    FakeLoad.fail = False
    old_body.fail_close = True
    with pytest.raises(RuntimeError) as caught:
        session.restore(saved)
    assert "committed" in caught.value.__notes__[0] and session.body is not old_body and not session.closed
    trial.require_saved_whole_equal(session.snapshot(), saved, "Committed replacement state preserved")


@pytest.mark.parametrize("kind", ["constructor", "metadata", "nonzero", "active_u", "invalid_lambda"])
def test_initial_failure_preserves_caller_body(kind):
    base = old.base_session()
    before = base.snapshot()
    FakeLoad.fail = kind == "constructor"
    FakeLoad.mutate = kind == "metadata"
    if kind == "nonzero":
        base.network.state["tick"] = 1
    with pytest.raises((ValueError, RuntimeError)):
        if kind == "active_u":
            trial.PostsynapticLoadTrial(base, np.array([0, 1], np.int64), np.array([0, 1], np.int64), np.full(2, .05), np.full(2, .15), 1.)
        else:
            wrap(base, value=2. if kind == "invalid_lambda" else 1.)
    assert not base.closed and base.body.close_calls == 0
    trial.require_equal(base.snapshot()["body"], before["body"], "Caller body unchanged")


def test_generated_cost_receipt():
    import psutil
    memory = psutil.Process().memory_info()
    peak = getattr(memory, "peak_wset", memory.rss)
    print({"E313_wrapper_fake_counts": COUNTS, "wall_s": time.perf_counter() - BEGAN, "peak_rss_bytes": peak})
    assert peak < 2 * 1024 ** 3
