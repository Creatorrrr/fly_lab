"""Generated fake CPU sessions only: no native neural or body constructors."""

import copy
import time
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from tools import banc_m3_source_filter_trial as trial

COUNTS = {"base_fake_networks": 0, "source_fake_networks": 0, "fake_bodies": 0,
          "fake_advances": 0, "fake_ticks": 0, "fake_model_s": 0.0,
          "original_neural_constructors": 0, "actual_body_constructors": 0,
          "physics_steps": 0, "CUDA_calls": 0, "fit_calls": 0}
BEGAN = time.perf_counter()


class Tensor:
    def __init__(self, value):
        self.value = np.asarray(value).copy()

    def cpu(self):
        return self

    def numpy(self):
        return self.value


class FakeNetwork:
    def __init__(self):
        COUNTS["base_fake_networks"] += 1
        self.matrix = csr_matrix(np.array([[0, 0, 0], [2, 0, 0], [0, -3, 0]], np.float32))
        self.n, self.identity, self.device = 3, "original-native-identity", "cpu"
        self.cuda_implementation, self.interval_steps = "torch-cpu-csr", 20
        self.adaptation_gain, self.adaptation_tau_s, self.dt = 8., .3, .00025
        self.core_ids, self.core_gain = np.array([0], np.int64), 0.
        self.motor_ids = np.array([1, 2], np.int64)
        self.motor_gains = np.array([2, 3], np.float32)
        self.motor_log_tau = np.array([-1, 0], np.float64)
        for key in ("tau", "a", "threshold", "cap"):
            setattr(self, key, Tensor(np.ones(3, np.float32)))
        self.state = {"model_hash": self.identity, "schema": "native-timescale",
            "device": "cpu", "dt": self.dt, "motor_ids": self.motor_ids.copy(),
            "parameters": {"gain": self.motor_gains.copy(), "log_tau": self.motor_log_tau.copy()},
            **{key: np.zeros(3, np.float32) for key in ("rate", "adaptation", "drive")},
            "output_mask": np.ones(3, np.float32), "tick": 0}

    def snapshot(self):
        return copy.deepcopy(self.state)


class FakeFiltered:
    fail = False
    mutate_inherited = False

    def __init__(self, weights, tau, a, threshold, cap, **kwargs):
        COUNTS["source_fake_networks"] += 1
        if self.fail:
            raise RuntimeError("fake candidate constructor failure")
        self.kwargs = copy.deepcopy(kwargs)
        self.matrix = weights.copy()
        self._selection = trial.checked_selection(weights, kwargs["edge_ids"], kwargs["coefficients"], kwargs["free_motor_ids"])
        self.source_timescale_identity = "original-native-identity"
        metadata = {"schema": trial.SCHEMA, "source_model_hash": self.source_timescale_identity,
                    "precision": trial.PRECISION, "tau": trial.TAU.copy(), **copy.deepcopy(self._selection)}
        self.identity = trial.filter_identity(metadata)
        self.state = {"model_hash": self.identity, "schema": "native-timescale",
            "device": "cpu", "dt": kwargs["dt"], "motor_ids": kwargs["motor_ids"].copy(),
            "parameters": {"gain": kwargs["motor_gains"].copy(), "log_tau": kwargs["motor_log_tau"].copy()},
            **{key: np.zeros(3, np.float32) for key in ("rate", "adaptation", "drive")},
            "output_mask": np.ones(3, np.float32), "tick": 0, "source_filter": metadata,
            "source_history": np.zeros((len(self._selection["source_ids"]), 2), np.float64)}
        if self.mutate_inherited:
            self.state["dt"] = .0001
        self.immutable = trial.metadata_without_state(self.state)
        self.immutable.pop("source_history")

    @property
    def selection(self):
        return copy.deepcopy(self._selection)

    def snapshot(self):
        return copy.deepcopy(self.state)

    def restore(self, saved):
        metadata = trial.metadata_without_state(saved)
        metadata.pop("source_history")
        trial.require_equal(metadata, self.immutable, "Fake atomic source metadata")
        history = saved["source_history"]
        if history.shape != self.state["source_history"].shape or history.dtype != np.float64 or not np.isfinite(history).all():
            raise ValueError("Fake invalid source history")
        self.state = copy.deepcopy(saved)

    def advance(self, drive, steps=1):
        COUNTS["fake_advances"] += 1
        COUNTS["fake_ticks"] += steps
        COUNTS["fake_model_s"] += steps * self.state["dt"]
        for _ in range(steps):
            self.state["source_history"] += .01 * (self.state["rate"][self._selection["source_ids"], None]
                                                   - self.state["source_history"])
            self.state["rate"] += .01 * np.asarray(drive, np.float32)
            self.state["drive"] = np.asarray(drive, np.float32).copy()
            self.state["tick"] += 1


class FakeBody:
    def __init__(self):
        COUNTS["fake_bodies"] += 1
        self.close_calls = 0
        self.fail_close = False
        self.state = {"hash": "fake-body", "qpos": np.array([.2, -.1], np.float64), "clock": 0}

    def close(self):
        self.close_calls += 1
        if self.fail_close:
            raise RuntimeError("fake body cleanup failure")


def fake_snapshot(self):
    if self.closed:
        raise ValueError("Closed fake session")
    return {"schema": self.schema, "graph_hash": self.graph.hash,
            "network": self.network.snapshot(), "adapter": copy.deepcopy(self.adapter.state),
            "monotone_recruitment_patch": copy.deepcopy(self.monotone_recruitment_patch),
            "body": copy.deepcopy(self.body.state), "world": copy.deepcopy(self.world),
            "sensory": copy.deepcopy(self.sensory), "tendon_inputs": copy.deepcopy(self.tendon_inputs),
            "control_tick": self.control_tick, "seed": 19, "wall_s": .75}


def base_session():
    base = trial.MonotoneRecruitmentTrial.__new__(trial.MonotoneRecruitmentTrial)
    base.closed = False
    base.graph = SimpleNamespace(hash="generated-graph", nodes=[{"id": str(i)} for i in range(3)])
    base.body, base.network = FakeBody(), FakeNetwork()
    base.adapter = SimpleNamespace(identity="fixed-monotone-adapter",
        state={"hash": "fixed-monotone-adapter", "bank": np.zeros((2, 3)), "ltm": np.array([.1, .2])})
    base.monotone_recruitment_patch = {"schema": "fixed-monotone", "motor_ids": np.array([1, 2], np.int64),
                                     "slopes": np.ones((2, 2), np.float64)}
    base.world = {"odor": np.array([4.])}
    base.sensory = {"filter": np.array([3., 5.])}
    base.tendon_inputs = {"abdomen1": .2, "abdomen2": .2, "ltm": .1}
    base.control_tick = 0
    return base


def fake_from_checkpoint(cls, graph, saved):
    assert saved["schema"] == trial.MonotoneRecruitmentTrial.schema
    base = base_session()
    base.graph = graph
    base.network.state = copy.deepcopy(saved["network"])
    base.adapter.state = copy.deepcopy(saved["adapter"])
    base.body.state = copy.deepcopy(saved["body"])
    for key in ("world", "sensory", "tendon_inputs", "monotone_recruitment_patch", "control_tick"):
        setattr(base, key, copy.deepcopy(saved[key]))
    return base


@pytest.fixture(autouse=True)
def fakes(monkeypatch):
    monkeypatch.setattr(trial, "MotorTimescaleNetwork", FakeNetwork)
    monkeypatch.setattr(trial, "SourceFilteredSynapseNetwork", FakeFiltered)
    monkeypatch.setattr(trial, "source_matrix", lambda network: network.matrix.copy())
    monkeypatch.setattr(trial.MonotoneRecruitmentTrial, "snapshot", fake_snapshot)
    monkeypatch.setattr(trial.MonotoneRecruitmentTrial, "from_checkpoint", classmethod(fake_from_checkpoint))
    FakeFiltered.fail = FakeFiltered.mutate_inherited = False


def wrap(base=None, coefficients=None):
    return trial.SourceFilterTrial(base or base_session(), np.array([0, 1], np.int64),
        np.array([[.5, .25, .25], [1, 0, 0]], np.float64) if coefficients is None else coefficients,
        np.array([1, 2], np.int64))


def test_only_network_layer_changes_and_inputs_are_owned():
    base = base_session()
    before = base.snapshot()
    c = np.array([[.5, .25, .25], [1, 0, 0]], np.float64)
    body, adapter = base.body, base.adapter
    wrapped = wrap(base, c)
    assert base.closed and not wrapped.closed
    assert wrapped.body is body and wrapped.adapter is adapter
    source = trial.remove_source_layer(wrapped.snapshot())
    trial.require_equal(source, before, "Every original whole field")
    assert wrapped.network.identity != before["network"]["model_hash"]
    assert np.all(wrapped.network.state["source_history"] == 0)
    c[:] = 7
    base.monotone_recruitment_patch["slopes"][:] = 9
    assert np.all(wrapped.monotone_recruitment_patch["slopes"] == 1)
    assert wrapped.source_filter_trial_patch["selection"]["coefficients64"][0, 0] == .5
    snapshot = wrapped.snapshot()
    snapshot["source_filter_trial_patch"]["selection"]["coefficients64"][:] = 6
    assert wrapped.source_filter_trial_patch["selection"]["coefficients64"][0, 0] == .5
    assert body.close_calls == 0


@pytest.mark.parametrize("key", ["rate", "adaptation", "drive", "output_mask", "tick"])
def test_non_neutral_rejected_without_body_transfer(key):
    base = base_session()
    if key == "tick":
        base.network.state[key] = 1
    else:
        base.network.state[key][0] = 0 if key == "output_mask" else 1
    with pytest.raises(ValueError, match="neutral|Neutral"):
        wrap(base)
    assert not base.closed and base.body.close_calls == 0


@pytest.mark.parametrize("failure", ["construct", "metadata"])
def test_failed_initialization_preserves_caller_body(failure):
    base = base_session()
    FakeFiltered.fail = failure == "construct"
    FakeFiltered.mutate_inherited = failure == "metadata"
    with pytest.raises((ValueError, RuntimeError)):
        wrap(base)
    assert not base.closed and base.body.close_calls == 0
    assert type(base.network) is FakeNetwork


def test_disk_nonzero_history_and_next_future_match(tmp_path):
    from flylab.c.storage import StateStore

    wrapped = wrap()
    wrapped.network.advance(np.array([4, 2, 3], np.float32), 5)
    saved = wrapped.snapshot()
    assert np.any(saved["network"]["source_history"] != 0)
    wrapped.save(tmp_path / "checkpoint")
    loaded = StateStore.load(tmp_path / "checkpoint")
    trial.require_equal(loaded, saved, "Disk bytes")
    fresh = trial.SourceFilterTrial.from_checkpoint(wrapped.graph, loaded)
    trial.require_equal(fresh.snapshot(), saved, "Fresh whole history")
    drive = np.array([1, 3, 2], np.float32)
    wrapped.network.advance(drive, 3)
    fresh.network.advance(drive, 3)
    trial.require_equal(wrapped.snapshot(), fresh.snapshot(), "Nonzero history next future")


@pytest.mark.parametrize("field", ["schema", "selection", "history", "identity", "inherited", "monotone", "adapter"])
def test_invalid_checkpoint_rejected_before_body_construction(field):
    wrapped = wrap()
    saved = wrapped.snapshot()
    if field == "schema":
        saved["schema"] = "other"
    elif field == "selection":
        saved["network"]["source_filter"]["coefficients64"][0, 0] = .9
    elif field == "history":
        saved["network"]["source_history"][0, 0] = np.nan
    elif field == "identity":
        saved["network"]["model_hash"] = "other"
    elif field == "inherited":
        saved["network"]["dt"] = .1
    elif field == "monotone":
        saved["monotone_recruitment_patch"]["slopes"][0, 0] = 2
    else:
        saved["adapter"]["hash"] = "other"
    bodies = COUNTS["fake_bodies"]
    with pytest.raises(ValueError):
        trial.SourceFilterTrial.from_checkpoint(wrapped.graph, saved)
    assert COUNTS["fake_bodies"] == bodies


def test_restore_swaps_whole_body_and_failure_preserves_live_session():
    wrapped = wrap()
    saved = wrapped.snapshot()
    old_body = wrapped.body
    FakeFiltered.fail = True
    with pytest.raises(RuntimeError, match="candidate constructor failure"):
        wrapped.restore(saved)
    assert wrapped.body is old_body and old_body.close_calls == 0
    trial.require_equal(wrapped.snapshot(), saved, "Failed restore preserves old whole")
    FakeFiltered.fail = False
    wrapped.restore(saved)
    assert wrapped.body is not old_body and old_body.close_calls == 1
    trial.require_equal(wrapped.snapshot(), saved, "Successful whole restore")


def test_owned_failure_cleanup_retains_primary_error(monkeypatch):
    wrapped = wrap()
    saved = wrapped.snapshot()
    fresh_bodies = []
    def restore_base(cls, graph, source):
        base = fake_from_checkpoint(cls, graph, source)
        base.body.fail_close = True
        fresh_bodies.append(base.body)
        return base
    monkeypatch.setattr(trial.MonotoneRecruitmentTrial, "from_checkpoint", classmethod(restore_base))
    FakeFiltered.fail = True
    with pytest.raises(RuntimeError, match="candidate constructor failure") as caught:
        wrapped.restore(saved)
    assert "cleanup failure" in caught.value.__notes__[0]
    assert fresh_bodies[0].close_calls == 1
    assert wrapped.body.close_calls == 0 and not wrapped.closed


def test_successful_swap_cleanup_failure_reports_committed_state():
    wrapped = wrap()
    saved = wrapped.snapshot()
    old_body = wrapped.body
    old_body.fail_close = True
    with pytest.raises(RuntimeError, match="body cleanup failure") as caught:
        wrapped.restore(saved)
    assert "committed" in caught.value.__notes__[0]
    assert wrapped.body is not old_body and not wrapped.closed
    trial.require_equal(wrapped.snapshot(), saved, "New session committed")


def test_neutral_coefficients_follow_same_initialization_rule():
    wrapped = wrap(coefficients=np.tile(np.array([1, 0, 0], np.float64), (2, 1)))
    assert np.all(wrapped.snapshot()["network"]["source_history"] == 0)


def test_generated_cost_receipt():
    print("GENERATED_E271_COST", {**COUNTS, "wall_s": time.perf_counter() - BEGAN})
    assert COUNTS["original_neural_constructors"] == COUNTS["actual_body_constructors"] == 0
    assert COUNTS["physics_steps"] == COUNTS["CUDA_calls"] == COUNTS["fit_calls"] == 0
