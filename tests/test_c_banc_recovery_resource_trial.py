"""E300 generated fake CPU metadata/ownership sessions; no real neural/body."""

import copy
import hashlib
import importlib.util
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from tools import banc_m3_recovery_resource_trial as trial

spec = importlib.util.spec_from_file_location(
    "e271_generated_fixtures",
    Path(__file__).with_name("test_c_banc_source_filter_trial.py"),
)
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)
COUNTS = {
    "fake_candidate_constructors": 0,
    "fake_advances": 0,
    "fake_ticks": 0,
    "fake_resource_snapshots": 0,
    "pure_checks": 0,
}
BEGAN = time.perf_counter()
SCHEMA = "flylab.m3-recovery-resource-synapses.v1"
PRECISION = "generated-resource-contract"


def checked_selection(weights, edges, pop, u, tau, motors):
    COUNTS["pure_checks"] += 1
    edges = np.asarray(edges)
    pop = np.asarray(pop)
    pre = weights.indices[edges].astype(np.int64)
    post = np.searchsorted(weights.indptr[1:], edges, side="right").astype(np.int64)
    sources = np.unique(pre)
    selection = {
        "edge_ids": edges.copy(),
        "premotor_ids": np.asarray(pop).copy(),
        "motor_ids": np.asarray(motors).copy(),
        "pre_ids": pre.copy(),
        "post_ids": post,
        "source_ids": sources,
        "edge_source_slots": np.searchsorted(sources, pre).astype(np.int32),
        "weights32": weights.data[edges].copy(),
        "utilization": np.asarray(u).copy(),
        "recovery_tau": np.asarray(tau).copy(),
    }
    trial.validate_selection_metadata(selection, weights.shape[0], motors)
    return selection


def resource_metadata(identity, selection):
    return {
        "schema": SCHEMA,
        "source_model_hash": identity,
        "precision": PRECISION,
        **copy.deepcopy(selection),
    }


def resource_identity(metadata):
    digest = hashlib.sha256(
        (metadata["source_model_hash"] + SCHEMA + PRECISION).encode()
    )
    for key in sorted(metadata):
        v = metadata[key]
        if isinstance(v, np.ndarray):
            digest.update(
                key.encode()
                + v.dtype.str.encode()
                + repr(v.shape).encode()
                + v.tobytes()
            )
    return digest.hexdigest()


def checked_resource_metadata(metadata):
    assert set(metadata) == trial.SELECTION_KEYS | {
        "schema",
        "source_model_hash",
        "precision",
    }
    trial.require_equal(metadata["schema"], SCHEMA, "schema")
    trial.require_equal(metadata["precision"], PRECISION, "precision")
    return copy.deepcopy(metadata)


class FakeNetwork(old.FakeNetwork):
    def __init__(self):
        super().__init__()
        self.n = 4
        self.matrix = csr_matrix(
            np.array(
                [[0, -3, 0, 0], [2, 0, 0, 0], [0, 0, 0, 0], [0, 1, 0, 0]], np.float32
            )
        )
        self.motor_ids = np.array([3], np.int64)
        self.motor_gains = np.array([2], np.float32)
        self.motor_log_tau = np.array([-1], np.float64)
        for key in ("tau", "a", "threshold", "cap"):
            setattr(self, key, old.Tensor(np.ones(4, np.float32)))
        self.state.update(
            motor_ids=self.motor_ids.copy(),
            parameters={
                "gain": self.motor_gains.copy(),
                "log_tau": self.motor_log_tau.copy(),
            },
            **{key: np.zeros(4, np.float32) for key in ("rate", "adaptation", "drive")},
            output_mask=np.ones(4, np.float32),
        )


class FakeResource:
    fail = False
    mutate = False
    fail_restore = False
    fault = False

    def __init__(self, weights, tau, a, threshold, cap, **kwargs):
        COUNTS["fake_candidate_constructors"] += 1
        if self.fail:
            raise RuntimeError("fake candidate construction failed")
        self._selection = checked_selection(
            weights,
            kwargs["edge_ids"],
            kwargs["premotor_ids"],
            kwargs["utilization"],
            kwargs["recovery_tau"],
            kwargs["motor_ids"],
        )
        self.source_timescale_identity = "original-native-identity"
        self.kwargs = copy.deepcopy(kwargs)
        metadata = resource_metadata(self.source_timescale_identity, self._selection)
        self.identity = resource_identity(metadata)
        self.state = {
            "schema": "native-timescale",
            "model_hash": self.identity,
            "device": "cpu",
            "dt": kwargs["dt"],
            "motor_ids": kwargs["motor_ids"].copy(),
            "parameters": {
                "gain": kwargs["motor_gains"].copy(),
                "log_tau": kwargs["motor_log_tau"].copy(),
            },
            **{key: np.zeros(4, np.float32) for key in ("rate", "adaptation", "drive")},
            "output_mask": np.ones(4, np.float32),
            "tick": 0,
            "recurrent_resource": metadata,
            "synaptic_resource": np.ones(
                len(self._selection["source_ids"]), np.float64
            ),
        }
        if self.mutate:
            self.state["dt"] = 0.001
        self.immutable = trial.metadata_without_state(self.state)

    @property
    def selection(self):
        return copy.deepcopy(self._selection)

    def snapshot(self):
        if self.fault:
            error = RuntimeError("generated sticky resource fault")
            error.resource_fault_flags = 1
            raise error
        COUNTS["fake_resource_snapshots"] += 1
        return copy.deepcopy(self.state)

    def import_base_snapshot(self, saved):
        trial.validate_neutral(saved)
        if saved["model_hash"] != self.source_timescale_identity:
            raise ValueError("fake original identity changed")
        inherited = trial.metadata_without_state(self.state)
        inherited.pop("recurrent_resource")
        inherited["model_hash"] = self.source_timescale_identity
        trial.require_equal(
            inherited, trial.metadata_without_state(saved), "fake inherited metadata"
        )
        for key in ("rate", "adaptation", "drive", "output_mask", "tick"):
            self.state[key] = copy.deepcopy(saved[key])
        self.state["synaptic_resource"].fill(1.0)

    def restore(self, saved):
        if self.fail_restore:
            raise RuntimeError("fake restore failed")
        trial.require_equal(
            trial.metadata_without_state(saved), self.immutable, "fake atomic metadata"
        )
        value = saved["synaptic_resource"]
        if (
            value.dtype != np.float64
            or value.shape != self.state["synaptic_resource"].shape
            or not np.isfinite(value).all()
        ):
            raise ValueError("fake resource state")
        self.state = copy.deepcopy(saved)

    def advance(self, drive, steps=1):
        COUNTS["fake_advances"] += 1
        COUNTS["fake_ticks"] += steps
        for _ in range(steps):
            self.state["rate"] += np.asarray(drive, np.float32) * 0.01
            self.state["synaptic_resource"] *= (
                1.0
                - self._selection["utilization"]
                * 0.01
                * self.state["rate"][self._selection["source_ids"]]
            )
            self.state["drive"] = np.asarray(drive, np.float32).copy()
            self.state["tick"] += 1


API = SimpleNamespace(
    SCHEMA=SCHEMA,
    PRECISION=PRECISION,
    checked_selection=checked_selection,
    resource_metadata=resource_metadata,
    resource_identity=resource_identity,
    checked_resource_metadata=checked_resource_metadata,
    RecoveryResourceNetwork=FakeResource,
)


def base_session():
    base = old.base_session()
    base.network = FakeNetwork()
    base.graph = SimpleNamespace(
        hash="generated-graph", nodes=[{"id": str(i)} for i in range(4)]
    )
    base.monotone_recruitment_patch["motor_ids"] = base.network.motor_ids.copy()
    return base


def from_base(cls, graph, saved):
    base = base_session()
    base.graph = graph
    base.network.state = copy.deepcopy(saved["network"])
    base.adapter.state = copy.deepcopy(saved["adapter"])
    base.body.state = copy.deepcopy(saved["body"])
    for key in (
        "world",
        "sensory",
        "tendon_inputs",
        "monotone_recruitment_patch",
        "control_tick",
    ):
        setattr(base, key, copy.deepcopy(saved[key]))
    return base


@pytest.fixture(autouse=True)
def fakes(monkeypatch):
    monkeypatch.setattr(trial, "MotorTimescaleNetwork", FakeNetwork)
    monkeypatch.setattr(trial, "network_api", lambda: API)
    monkeypatch.setattr(trial, "source_matrix", lambda network: network.matrix.copy())
    monkeypatch.setattr(trial.MonotoneRecruitmentTrial, "snapshot", old.fake_snapshot)
    monkeypatch.setattr(
        trial.MonotoneRecruitmentTrial, "from_checkpoint", classmethod(from_base)
    )
    FakeResource.fail = FakeResource.mutate = FakeResource.fail_restore = (
        FakeResource.fault
    ) = False


def wrap(base=None, u=None):
    return trial.RecoveryResourceTrial(
        base or base_session(),
        np.array([0, 1], np.int64),
        np.array([0, 1], np.int64),
        np.array([0.05, 0.05], np.float64) if u is None else u,
        np.full(2, 0.15, np.float64),
    )


def test_only_resource_layer_changes_and_provenance_is_owned():
    base = base_session()
    before = base.snapshot()
    body, adapter = base.body, base.adapter
    u = np.array([0.05, 0.05], np.float64)
    wrapped = wrap(base, u)
    assert (
        base.closed
        and not wrapped.closed
        and wrapped.body is body
        and wrapped.adapter is adapter
    )
    trial.require_equal(
        trial.remove_resource_layer(wrapped.snapshot()),
        before,
        "every original whole field",
    )
    assert (
        np.all(wrapped.snapshot()["network"]["synaptic_resource"] == 1.0)
        and body.close_calls == 0
    )
    u[:] = 0.2
    base.monotone_recruitment_patch["slopes"][:] = 9
    assert np.all(
        wrapped.recovery_resource_trial_patch["selection"]["utilization"] == 0.05
    )
    assert np.all(wrapped.monotone_recruitment_patch["slopes"] == 1.0)
    saved = wrapped.snapshot()
    saved["recovery_resource_trial_patch"]["selection"]["weights32"][:] = 9
    assert wrapped.recovery_resource_trial_patch["selection"]["weights32"][0] == -3


@pytest.mark.parametrize("key", ["rate", "adaptation", "drive", "output_mask", "tick"])
def test_non_neutral_never_transfers_body(key):
    base = base_session()
    before = base.snapshot()
    if key == "tick":
        base.network.state[key] = 1
    else:
        base.network.state[key][0] = 0 if key == "output_mask" else 1
    with pytest.raises(ValueError, match="neutral|Neutral"):
        wrap(base)
    assert (
        not base.closed
        and base.body.close_calls == 0
        and type(base.network) is FakeNetwork
    )
    assert trial.equal(base.snapshot()["body"], before["body"])


@pytest.mark.parametrize("kind", ["construct", "metadata"])
def test_initialization_failure_preserves_caller(kind):
    base = base_session()
    before = base.snapshot()
    FakeResource.fail = kind == "construct"
    FakeResource.mutate = kind == "metadata"
    with pytest.raises((RuntimeError, ValueError)):
        wrap(base)
    assert not base.closed and base.body.close_calls == 0
    trial.require_equal(base.snapshot(), before, "failed init caller preserved")


def test_nonzero_disk_whole_restore_and_future_exact(tmp_path):
    from flylab.c.storage import StateStore

    wrapped = wrap()
    wrapped.network.advance(np.array([4, 2, 3, 1], np.float32), 5)
    saved = wrapped.snapshot()
    assert np.any(saved["network"]["synaptic_resource"] < 1.0)
    wrapped.save(tmp_path / "checkpoint")
    loaded = StateStore.load(tmp_path / "checkpoint")
    trial.require_equal(saved, loaded, "disk bytes")
    fresh = trial.RecoveryResourceTrial.from_checkpoint(wrapped.graph, loaded)
    trial.require_equal(fresh.snapshot(), saved, "nonzero whole state")
    for owner in (wrapped, fresh):
        owner.network.advance(np.ones(4, np.float32), 3)
    trial.require_equal(
        wrapped.snapshot(), fresh.snapshot(), "same resource next future"
    )


@pytest.mark.parametrize(
    "field",
    [
        "schema",
        "identity",
        "patch",
        "selection",
        "resource_nan",
        "resource_negative",
        "resource_high",
        "resource_dtype",
        "resource_shape",
        "tick0resource",
        "source_zero",
        "inherited",
        "monotone",
        "adapter",
        "premotor_motor",
        "source_slots",
    ],
)
def test_invalid_checkpoint_precedes_new_body(field):
    wrapped = wrap()
    saved = wrapped.snapshot()
    p = saved["recovery_resource_trial_patch"]
    net = saved["network"]
    if field == "schema":
        saved["schema"] = "other"
    elif field == "identity":
        net["model_hash"] = "other"
    elif field == "patch":
        p["patch_identity"] = "other"
    elif field == "selection":
        net["recurrent_resource"]["utilization"][0] = 0.2
    elif field == "resource_nan":
        net["synaptic_resource"][0] = np.nan
    elif field == "resource_negative":
        net["synaptic_resource"][0] = -0.1
    elif field == "resource_high":
        net["synaptic_resource"][0] = 1.01
    elif field == "resource_dtype":
        net["synaptic_resource"] = net["synaptic_resource"].astype(np.float32)
    elif field == "resource_shape":
        net["synaptic_resource"] = np.ones(3, np.float64)
    elif field == "tick0resource":
        net["synaptic_resource"][0] = 0.9
    elif field == "source_zero":
        net["tick"] = 1
        p["selection"]["utilization"][0] = 0.0
        net["recurrent_resource"]["utilization"][0] = 0.0
        net["synaptic_resource"][0] = 0.9
        net["model_hash"] = p["network_identity"] = resource_identity(
            net["recurrent_resource"]
        )
        p["patch_identity"] = trial.provenance_hash(
            {k: v for k, v in p.items() if k != "patch_identity"}
        )
    elif field == "inherited":
        net["dt"] = 0.01
    elif field == "monotone":
        saved["monotone_recruitment_patch"]["slopes"][0, 0] = 9
    elif field == "adapter":
        saved["adapter"]["hash"] = "other"
    else:
        p["selection"]["premotor_ids"] = (
            np.array([0, 3], np.int64)
            if field == "premotor_motor"
            else p["selection"]["premotor_ids"]
        )
        if field == "source_slots":
            p["selection"]["edge_source_slots"][0] = 9
        p["patch_identity"] = trial.provenance_hash(
            {k: v for k, v in p.items() if k != "patch_identity"}
        )
    before = old.COUNTS["fake_bodies"]
    with pytest.raises((ValueError, AssertionError)):
        trial.RecoveryResourceTrial.from_checkpoint(wrapped.graph, saved)
    assert old.COUNTS["fake_bodies"] == before


def test_restore_failed_replacement_keeps_old_and_success_closes_once():
    wrapped = wrap()
    saved = wrapped.snapshot()
    old_body = wrapped.body
    FakeResource.fail = True
    with pytest.raises(RuntimeError, match="construction failed"):
        wrapped.restore(saved)
    assert wrapped.body is old_body and old_body.close_calls == 0
    trial.require_equal(wrapped.snapshot(), saved, "failed restore live whole")
    FakeResource.fail = False
    wrapped.restore(saved)
    assert (
        wrapped.body is not old_body
        and old_body.close_calls == 1
        and not wrapped.closed
    )
    trial.require_equal(wrapped.snapshot(), saved, "successful replacement")


def test_failure_cleanup_keeps_primary(monkeypatch):
    wrapped = wrap()
    saved = wrapped.snapshot()
    new_bodies = []

    def restore(cls, graph, source):
        base = from_base(cls, graph, source)
        base.body.fail_close = True
        new_bodies.append(base.body)
        return base

    monkeypatch.setattr(
        trial.MonotoneRecruitmentTrial, "from_checkpoint", classmethod(restore)
    )
    FakeResource.fail = True
    with pytest.raises(RuntimeError, match="construction failed") as error:
        wrapped.restore(saved)
    assert (
        "cleanup failed" in error.value.__notes__[0] and new_bodies[0].close_calls == 1
    )
    assert wrapped.body.close_calls == 0 and not wrapped.closed


def test_successful_swap_cleanup_failure_reports_committed_state():
    wrapped = wrap()
    saved = wrapped.snapshot()
    old_body = wrapped.body
    old_body.fail_close = True
    with pytest.raises(RuntimeError, match="cleanup failure") as error:
        wrapped.restore(saved)
    assert (
        "committed" in error.value.__notes__[0]
        and wrapped.body is not old_body
        and not wrapped.closed
    )
    trial.require_equal(wrapped.snapshot(), saved, "committed new whole")


def test_neutral_u_zero_resource_one():
    wrapped = wrap(u=np.zeros(2, np.float64))
    assert np.all(wrapped.snapshot()["network"]["synaptic_resource"] == 1.0)


@pytest.mark.parametrize("wrong", ["closed", "nested"])
def test_reject_closed_or_layered_base_without_transfer(wrong):
    base = base_session()
    if wrong == "closed":
        base.closed = True
    else:
        base = wrap(base)
    body = base.body
    with pytest.raises(TypeError):
        wrap(base)
    assert body.close_calls == 0


def test_checkpoint_save_failure_preserves_live_owner(tmp_path, monkeypatch):
    from flylab.c.storage import StateStore

    wrapped = wrap()
    before = wrapped.snapshot()
    body = wrapped.body

    def fail(*args):
        raise OSError("generated disk failure")

    monkeypatch.setattr(StateStore, "save", fail)
    with pytest.raises(OSError, match="disk failure"):
        wrapped.save(tmp_path / "checkpoint")
    assert not wrapped.closed and wrapped.body is body and body.close_calls == 0
    trial.require_equal(wrapped.snapshot(), before, "save failure unchanged whole")


def test_nonzero_candidate_restore_failure_closes_only_new_owner(monkeypatch):
    wrapped = wrap()
    saved = wrapped.snapshot()
    old_body = wrapped.body
    new_bodies = []

    def restore_base(cls, graph, source):
        base = from_base(cls, graph, source)
        new_bodies.append(base.body)
        return base

    monkeypatch.setattr(
        trial.MonotoneRecruitmentTrial, "from_checkpoint", classmethod(restore_base)
    )
    FakeResource.fail_restore = True
    with pytest.raises(RuntimeError, match="restore failed"):
        wrapped.restore(saved)
    assert (
        new_bodies[0].close_calls == 1
        and old_body.close_calls == 0
        and wrapped.body is old_body
    )
    trial.require_equal(
        wrapped.snapshot(), saved, "failed nonzero neural restore leaves old whole"
    )


def test_nonzero_disk_tuple_parameters_and_future_match(tmp_path, monkeypatch):
    from flylab.c.storage import StateStore

    original_snapshot = old.fake_snapshot

    def snapshot(session):
        return {
            **original_snapshot(session),
            "parameters": {
                "joint_gain_multipliers": (1.0, 0.8, 1.2),
                "stored_numpy_scalar": np.float32(0.75),
            },
        }

    monkeypatch.setattr(trial.MonotoneRecruitmentTrial, "snapshot", snapshot)
    wrapped = wrap()
    wrapped.network.advance(np.array([4, 2, 1, 1], np.float32), 3)
    wrapped.save(tmp_path / "disk")
    loaded = StateStore.load(tmp_path / "disk")
    assert isinstance(loaded["parameters"]["joint_gain_multipliers"], list)
    assert type(loaded["parameters"]["stored_numpy_scalar"]) is float
    fresh = trial.RecoveryResourceTrial.from_checkpoint(wrapped.graph, loaded)
    report = trial.require_saved_whole_equal(fresh.snapshot(), loaded, "stored whole")
    assert report["numeric_tolerance"] == 0 and report["excluded_fields"] == []
    assert report["live_representation_changes"]
    for owner in (wrapped, fresh):
        owner.network.advance(np.array([2, 3, 1, 1], np.float32), 2)
    trial.require_equal(
        wrapped.snapshot(), fresh.snapshot(), "future exact including tuple parameters"
    )
    bad = copy.deepcopy(loaded)
    bad["parameters"]["joint_gain_multipliers"][1] += 1e-12
    with pytest.raises(ValueError, match="Whole original"):
        trial.RecoveryResourceTrial.from_checkpoint(wrapped.graph, bad)


@pytest.mark.parametrize(
    "change", ["array_value", "array_dtype", "wall_value", "field_missing"]
)
def test_storage_representation_never_relaxes_scientific_values(change):
    live = wrap().snapshot()
    live["parameters"] = {"tuple": (1.0, 2.0), "scalar": np.float64(0.5)}
    saved = trial.checkpoint_representation(copy.deepcopy(live), [])
    if change == "array_value":
        saved["body"]["qpos"][0] += 1e-12
    elif change == "array_dtype":
        saved["body"]["qpos"] = saved["body"]["qpos"].astype(np.float32)
    elif change == "wall_value":
        saved["wall_s"] += 1e-12
    else:
        saved.pop("sensory")
    with pytest.raises(ValueError):
        trial.require_saved_whole_equal(live, saved, "no relaxed values")


@pytest.mark.parametrize("value", [(1.0, 2.0), np.float64(0.5)])
def test_hash_covered_unstable_patch_rejected_without_transfer(value):
    base = base_session()
    base.monotone_recruitment_patch["unsupported_storage_type"] = value
    before = base.snapshot()
    with pytest.raises(ValueError, match="storage-stable"):
        wrap(base)
    assert not base.closed and base.body.close_calls == 0
    trial.require_equal(base.snapshot(), before, "unsupported patch caller unchanged")


def test_active_negative_rate_rejected_before_body_factory():
    wrapped = wrap(u=np.array([0.05, 0.0], np.float64))
    saved = wrapped.snapshot()
    saved["network"]["tick"] = 1
    saved["network"]["rate"][0] = np.float32(-1e-8)
    before = old.COUNTS["fake_bodies"]
    with pytest.raises(ValueError, match="active source rate"):
        trial.RecoveryResourceTrial.from_checkpoint(wrapped.graph, saved)
    assert old.COUNTS["fake_bodies"] == before


@pytest.mark.parametrize("index", [1, 2, 3])
def test_inactive_source_and_other_native_negative_tolerance_preserved(index):
    wrapped = wrap(u=np.array([0.05, 0.0], np.float64))
    saved = wrapped.snapshot()
    saved["network"]["tick"] = 1
    saved["network"]["rate"][index] = np.float32(-5e-5)
    restored = trial.RecoveryResourceTrial.from_checkpoint(wrapped.graph, saved)
    trial.require_equal(restored.snapshot(), saved, "inactive native rate exact")


@pytest.mark.parametrize("kind", ["whole", "metadata"])
def test_old_engine_schema_rejected_before_new_body(kind):
    wrapped = wrap()
    saved = wrapped.snapshot()
    if kind == "whole":
        saved["schema"] = "flylab.m3-recurrent-resource-trial.v1"
    else:
        saved["network"]["recurrent_resource"]["schema"] = (
            "flylab.m3-recurrent-resource-synapses.v1"
        )
    before = old.COUNTS["fake_bodies"]
    with pytest.raises(ValueError):
        trial.RecoveryResourceTrial.from_checkpoint(wrapped.graph, saved)
    assert old.COUNTS["fake_bodies"] == before


def test_fault_snapshot_and_save_cannot_publish_success(tmp_path):
    wrapped = wrap()
    saved = wrapped.snapshot()
    body = wrapped.body
    wrapped.network.fault = True
    with pytest.raises(RuntimeError, match="sticky") as error:
        wrapped.snapshot()
    assert error.value.resource_fault_flags == 1
    with pytest.raises(RuntimeError, match="sticky"):
        wrapped.save(tmp_path / "fault-checkpoint")
    assert not (tmp_path / "fault-checkpoint").exists() and body.close_calls == 0
    wrapped.restore(saved)
    assert wrapped.body is not body and body.close_calls == 1
    trial.require_equal(
        wrapped.snapshot(), saved, "whole replacement recovers valid stored state"
    )


def test_nonzero_disk_preserves_readout_rng_body_time_and_signed_zero(tmp_path):
    from flylab.c.storage import StateStore

    base = base_session()
    base.body.state.update(
        qvel=np.array([-0.0, 0.25], np.float64),
        contact=np.array([True, False]),
        rng={"word": np.array([17, 91], np.uint64)},
    )
    base.adapter.state["readout"] = {
        "motor": np.array([-0.0, 1.0], np.float64),
        "held": np.array([0.2]),
    }
    wrapped = wrap(base)
    wrapped.network.advance(np.array([4, 2, 1, 1], np.float32), 3)
    wrapped.network.state["rate"][0] = np.float32(-0.0)
    wrapped.network.state["drive"][1] = np.float32(-0.0)
    wrapped.network.state["output_mask"][1] = np.float32(-0.0)
    wrapped.control_tick = 1
    wrapped.body.state["clock"] = 0.005
    saved = wrapped.snapshot()
    wrapped.save(tmp_path / "whole")
    restored = trial.RecoveryResourceTrial.from_checkpoint(
        wrapped.graph, StateStore.load(tmp_path / "whole")
    )
    trial.require_equal(
        restored.snapshot(), saved, "all physical/controller/readout/zero bytes"
    )
    assert np.signbit(restored.snapshot()["network"]["rate"][0])
    assert np.signbit(restored.snapshot()["body"]["qvel"][0])


@pytest.mark.parametrize("tick", [1, True])
def test_initial_whole_tick_must_be_exact_zero_before_new_network(tick):
    base = base_session()
    base.control_tick = tick
    constructors = COUNTS["fake_candidate_constructors"]
    with pytest.raises(ValueError, match="whole control tick zero"):
        wrap(base)
    assert COUNTS["fake_candidate_constructors"] == constructors
    assert not base.closed and base.body.close_calls == 0


def test_generated_cost_receipt():
    assert time.perf_counter() - BEGAN < 90
    import psutil

    process = psutil.Process()
    peak = getattr(process.memory_info(), "peak_wset", process.memory_info().rss)
    assert peak < (1 << 30)
    print(
        "GENERATED_E300_COST",
        {
            "resource_fake_counts": COUNTS,
            "whole_fake_counts": old.COUNTS,
            "wall_s": time.perf_counter() - BEGAN,
            "tiny_solver": 0,
            "peak_process_bytes": peak,
            "fake_engine_contract_only": True,
        },
    )
