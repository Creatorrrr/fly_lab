"""E300 whole-session resource recurrence wrapper; no autonomous gait claim.

Initial construction transfers the caller's body only after all exact checks.
Whole restore creates a fresh network/body and commits it before old-body close;
a close failure reports the committed replacement. No neural-only restore path.
"""

from __future__ import annotations

import copy

import numpy as np

from tools.banc_m3_monotone_recruitment import MonotoneRecruitmentTrial
from tools.banc_m3_motor_timescale import MotorTimescaleNetwork
from tools.banc_m3_recurrent_resource_trial import require_saved_whole_equal
from tools.banc_m3_source_filter_trial import (
    cleanup_owned,
    equal,
    provenance_hash,
    require_equal,
    transferred_fields,
)
from tools.probe_c_banc_m3_core_rhythm import source_matrix
from tools.verify_c_banc_m3_source_native_transfer import checkpoint_representation

TRIAL_SCHEMA = "flylab.m3-recovery-resource-trial.v1"
PATCH_SCHEMA = "flylab.m3-recovery-resource-trial-patch.v1"
PATCH_KEY = "recovery_resource_trial_patch"
DYNAMIC = {"rate", "adaptation", "drive", "output_mask", "tick", "synaptic_resource"}
PATCH_KEYS = {
    "schema",
    "source_trial_schema",
    "source_network_identity",
    "network_identity",
    "source_adapter_identity",
    "graph_hash",
    "source_network_metadata",
    "selection",
    "source_monotone_patch",
    "patch_identity",
}
NETWORK_SCHEMA = "flylab.m3-recovery-resource-synapses.v1"
SELECTION_KEYS = {
    "edge_ids",
    "premotor_ids",
    "motor_ids",
    "pre_ids",
    "post_ids",
    "source_ids",
    "edge_source_slots",
    "weights32",
    "utilization",
    "recovery_tau",
}


def network_api():
    """Lazy import permits generated fake contract checks before E299 delivery."""
    from tools import banc_m3_recovery_resource_synapses

    return banc_m3_recovery_resource_synapses


def metadata_without_state(saved):
    return copy.deepcopy(
        {key: value for key, value in saved.items() if key not in DYNAMIC}
    )


def validate_neutral(saved):
    if (
        not isinstance(saved, dict)
        or type(saved.get("tick")) is not int
        or saved["tick"] != 0
    ):
        raise ValueError("New resource parameters require neutral tick zero")
    rate = saved.get("rate")
    if not isinstance(rate, np.ndarray) or rate.dtype != np.float32 or rate.ndim != 1:
        raise ValueError("Native neutral float32 rate required")
    for key in ("rate", "adaptation", "drive", "output_mask"):
        target = np.ones_like(rate) if key == "output_mask" else np.zeros_like(rate)
        require_equal(saved.get(key), target, "Neutral original " + key)
    if any(
        key in saved
        for key in (
            "recurrent_resource",
            "synaptic_resource",
            "source_filter",
            "source_history",
        )
    ):
        raise ValueError("Unmodified original timescale neural snapshot required")


def validate_selection_metadata(selection, n, motor_ids):
    """Own resource-specific structural validation before any fresh body exists."""
    if not isinstance(selection, dict) or set(selection) != SELECTION_KEYS:
        raise ValueError("Complete recurrent resource selection required")
    for key in ("edge_ids", "premotor_ids", "source_ids"):
        value = selection[key]
        if (
            not isinstance(value, np.ndarray)
            or value.dtype != np.int64
            or value.ndim != 1
            or not len(value)
            or np.any(value < 0)
            or np.any(value[1:] <= value[:-1])
        ):
            raise ValueError("Sorted unique resource selection IDs required")
    edges, pop, sources = (
        selection[key] for key in ("edge_ids", "premotor_ids", "source_ids")
    )
    motors = selection["motor_ids"]
    if (
        not isinstance(motors, np.ndarray)
        or motors.dtype != np.int64
        or motors.ndim != 1
        or not len(motors)
        or len(np.unique(motors)) != len(motors)
        or np.any(motors < 0)
        or np.any(motors >= n)
        or np.any(pop >= n)
        or np.intersect1d(pop, motors).size
    ):
        raise ValueError("Native disjoint premotor/motor rosters required")
    require_equal(motors, motor_ids, "Original native motor roster")
    for key in ("pre_ids", "post_ids"):
        value = selection[key]
        if (
            not isinstance(value, np.ndarray)
            or value.dtype != np.int64
            or value.shape != edges.shape
            or np.any(value < 0)
            or np.any(value >= n)
            or not np.isin(value, pop).all()
        ):
            raise ValueError("Selected endpoints must be existing premotor population")
    require_equal(
        sources,
        np.unique(selection["pre_ids"]),
        "Unique selected presynaptic resource pools",
    )
    require_equal(
        selection["edge_source_slots"],
        np.searchsorted(sources, selection["pre_ids"]).astype(np.int32),
        "Exact source resource slots",
    )
    weights = selection["weights32"]
    if (
        not isinstance(weights, np.ndarray)
        or weights.dtype != np.float32
        or weights.shape != edges.shape
        or not np.isfinite(weights).all()
        or np.any(weights == 0)
    ):
        raise ValueError("Native nonzero original selected weight bytes required")
    for key, low, high in (("utilization", 0.0, 0.5), ("recovery_tau", 0.02, 0.5)):
        value = selection[key]
        if (
            not isinstance(value, np.ndarray)
            or value.dtype != np.float64
            or value.shape != sources.shape
            or not np.isfinite(value).all()
            or np.any(value < low)
            or np.any(value > high)
        ):
            raise ValueError("Bounded native float64 resource " + key + " required")


def make_candidate(old, selection):
    if type(old) is not MotorTimescaleNetwork:
        raise TypeError("Exact original MotorTimescaleNetwork required")
    api = network_api()
    weights = source_matrix(old)
    native = api.checked_selection(
        weights,
        selection["edge_ids"],
        selection["premotor_ids"],
        selection["utilization"],
        selection["recovery_tau"],
        old.motor_ids,
    )
    require_equal(native, selection, "Original recurrent selection and parameter bytes")
    implementation = old.cuda_implementation
    if old.device == "cpu" and implementation == "torch-cpu-csr":
        implementation = "reference"
    return api.RecoveryResourceNetwork(
        weights,
        *(
            getattr(old, key).cpu().numpy().copy()
            for key in ("tau", "a", "threshold", "cap")
        ),
        adaptation_gain=old.adaptation_gain,
        adaptation_tau_s=old.adaptation_tau_s,
        dt=old.dt,
        device=old.device,
        cuda_implementation=implementation,
        capture_steps=old.interval_steps,
        core_ids=old.core_ids.copy(),
        core_gain=old.core_gain,
        motor_ids=old.motor_ids.copy(),
        motor_gains=old.motor_gains.copy(),
        motor_log_tau=old.motor_log_tau.copy(),
        edge_ids=native["edge_ids"],
        premotor_ids=native["premotor_ids"],
        utilization=native["utilization"],
        recovery_tau=native["recovery_tau"],
    )


def checked_checkpoint(saved):
    if not isinstance(saved, dict) or saved.get("schema") != TRIAL_SCHEMA:
        raise ValueError("Explicit recurrent resource whole checkpoint required")
    patch, network = saved.get(PATCH_KEY), saved.get("network")
    if (
        not isinstance(patch, dict)
        or set(patch) != PATCH_KEYS
        or patch.get("schema") != PATCH_SCHEMA
        or patch.get("source_trial_schema") != MonotoneRecruitmentTrial.schema
        or not isinstance(network, dict)
    ):
        raise ValueError("Complete recurrent resource trial provenance required")
    unhashed = copy.deepcopy(patch)
    checksum = unhashed.pop("patch_identity")
    if checksum != provenance_hash(unhashed):
        raise ValueError("Recurrent resource patch checksum changed")
    source_meta = patch["source_network_metadata"]
    if (
        not isinstance(source_meta, dict)
        or source_meta.get("model_hash") != patch["source_network_identity"]
        or saved.get("graph_hash") != patch["graph_hash"]
        or saved.get("adapter", {}).get("hash") != patch["source_adapter_identity"]
    ):
        raise ValueError("Original whole resource provenance changed")
    require_equal(
        saved.get("monotone_recruitment_patch"),
        patch["source_monotone_patch"],
        "Original whole monotone patch",
    )
    rate = network.get("rate")
    tick = network.get("tick")
    if (
        not isinstance(rate, np.ndarray)
        or rate.dtype != np.float32
        or rate.ndim != 1
        or type(tick) is not int
        or tick < 0
    ):
        raise ValueError("Native resource neural state and tick required")
    motors = patch["source_monotone_patch"].get("motor_ids")
    validate_selection_metadata(patch["selection"], len(rate), motors)
    api = network_api()
    metadata = network.get("recurrent_resource")
    if (
        getattr(api, "SCHEMA", None) != NETWORK_SCHEMA
        or not isinstance(metadata, dict)
        or metadata.get("schema") != NETWORK_SCHEMA
        or metadata.get("precision") != api.PRECISION
    ):
        raise ValueError(
            "Explicit recovery schema/precision required; old resource checkpoints are incompatible"
        )
    expected = api.resource_metadata(
        patch["source_network_identity"], patch["selection"]
    )
    require_equal(
        metadata, expected, "Resource metadata selection and exact parameters"
    )
    api.checked_resource_metadata(metadata)
    if (
        network.get("model_hash") != patch["network_identity"]
        or api.resource_identity(metadata) != patch["network_identity"]
    ):
        raise ValueError("Recurrent resource neural identity changed")
    inherited = metadata_without_state(network)
    inherited.pop("recurrent_resource")
    inherited["model_hash"] = patch["source_network_identity"]
    require_equal(inherited, source_meta, "All inherited neural metadata")
    for key in ("rate", "adaptation", "drive", "output_mask"):
        value = network.get(key)
        if (
            not isinstance(value, np.ndarray)
            or value.dtype != np.float32
            or value.shape != rate.shape
            or not np.isfinite(value).all()
        ):
            raise ValueError("Native finite inherited " + key + " required")
    if (
        np.any(rate < -1e-4)
        or np.any(network["adaptation"] < 0)
        or not np.isin(network["output_mask"], (0, 1)).all()
    ):
        raise ValueError("Invalid inherited neural values")
    resource = network.get("synaptic_resource")
    selection = patch["selection"]
    active_sources = selection["source_ids"][selection["utilization"] > 0]
    if np.any(rate[active_sources] < 0):
        raise ValueError(
            "Recovery active source rate must be nonnegative before whole body restoration"
        )
    if (
        not isinstance(resource, np.ndarray)
        or resource.dtype != np.float64
        or resource.shape != selection["source_ids"].shape
        or not np.isfinite(resource).all()
        or np.any(resource < 0)
        or np.any(resource > 1)
        or not equal(
            resource[selection["utilization"] == 0],
            np.ones(np.count_nonzero(selection["utilization"] == 0), np.float64),
        )
        or tick == 0
        and not equal(resource, np.ones_like(resource))
    ):
        raise ValueError(
            "Finite bounded resource; neutral source/tick must be exact one"
        )
    return copy.deepcopy(patch)


def remove_resource_layer(saved):
    patch = checked_checkpoint(saved)
    source = copy.deepcopy(saved)
    source.pop(PATCH_KEY)
    source["schema"] = MonotoneRecruitmentTrial.schema
    source["network"].pop("recurrent_resource")
    source["network"].pop("synaptic_resource")
    source["network"]["model_hash"] = patch["source_network_identity"]
    return source


class RecoveryResourceTrial(MonotoneRecruitmentTrial):
    schema = TRIAL_SCHEMA

    def __init__(self, base, edge_ids, premotor_ids, utilization, recovery_tau):
        if type(base) is not MonotoneRecruitmentTrial or base.closed:
            raise TypeError("Exact open original MonotoneRecruitmentTrial required")
        before = base.snapshot()
        if type(before.get("control_tick")) is not int or before["control_tick"] != 0:
            raise ValueError(
                "New recovery parameters require original whole control tick zero"
            )
        validate_neutral(before["network"])
        api = network_api()
        require_equal(
            base.monotone_recruitment_patch.get("motor_ids"),
            base.network.motor_ids,
            "Native original monotone motor roster",
        )
        selection = api.checked_selection(
            source_matrix(base.network),
            edge_ids,
            premotor_ids,
            utilization,
            recovery_tau,
            base.network.motor_ids,
        )
        validate_selection_metadata(
            selection, len(before["network"]["rate"]), base.network.motor_ids
        )
        candidate = make_candidate(base.network, selection)
        require_equal(
            candidate.selection, selection, "Constructed original recurrent selection"
        )
        candidate.import_base_snapshot(copy.deepcopy(before["network"]))
        initial = candidate.snapshot()
        inherited = metadata_without_state(initial)
        inherited.pop("recurrent_resource")
        inherited["model_hash"] = candidate.source_timescale_identity
        require_equal(
            inherited,
            metadata_without_state(before["network"]),
            "Exact inherited candidate neural metadata",
        )
        for key in ("rate", "adaptation", "drive", "output_mask", "tick"):
            require_equal(
                initial[key], before["network"][key], "Exact neutral imported " + key
            )
        require_equal(
            initial["synaptic_resource"],
            np.ones(len(selection["source_ids"]), np.float64),
            "Fresh resource one",
        )
        require_equal(
            base.snapshot(),
            before,
            "Caller whole session unchanged during construction",
        )
        patch = {
            "schema": PATCH_SCHEMA,
            "source_trial_schema": MonotoneRecruitmentTrial.schema,
            "source_network_identity": base.network.identity,
            "network_identity": candidate.identity,
            "source_adapter_identity": base.adapter.identity,
            "graph_hash": base.graph.hash,
            "source_network_metadata": metadata_without_state(before["network"]),
            "selection": candidate.selection,
            "source_monotone_patch": copy.deepcopy(base.monotone_recruitment_patch),
        }
        # The inherited hash remains type-sensitive. Hash-covered metadata
        # must already have StateStore-stable representation; unsupported
        # tuple/scalar metadata is rejected before caller ownership transfer.
        patch_changes = []
        require_equal(
            patch,
            checkpoint_representation(patch, patch_changes),
            "Hash-covered resource patch must be storage-stable",
        )
        patch["patch_identity"] = provenance_hash(patch)
        self.__dict__.update(transferred_fields(base))
        self.network = candidate
        setattr(self, PATCH_KEY, copy.deepcopy(patch))
        expected = copy.deepcopy(before)
        expected.update(
            schema=self.schema, network=initial, **{PATCH_KEY: copy.deepcopy(patch)}
        )
        require_equal(
            self.snapshot(), expected, "Only recurrent neural resource layer changed"
        )
        checked_checkpoint(self.snapshot())
        base.closed = True

    def snapshot(self):
        saved = MonotoneRecruitmentTrial.snapshot(self)
        saved["schema"] = self.schema
        saved[PATCH_KEY] = copy.deepcopy(getattr(self, PATCH_KEY))
        return saved

    def save(self, path):
        from flylab.c.storage import StateStore

        StateStore.save(path, self.snapshot())

    @classmethod
    def from_checkpoint(cls, graph, saved):
        patch = checked_checkpoint(saved)
        if graph.hash != patch["graph_hash"]:
            raise ValueError("Resource whole graph identity changed")
        source = remove_resource_layer(saved)
        base = MonotoneRecruitmentTrial.from_checkpoint(graph, source)
        try:
            require_saved_whole_equal(
                base.snapshot(), source, "Whole original monotone reconstruction"
            )
            if (
                base.network.identity != patch["source_network_identity"]
                or base.adapter.identity != patch["source_adapter_identity"]
            ):
                raise ValueError(
                    "Original reconstructed neural/adapter identity changed"
                )
            candidate = make_candidate(base.network, patch["selection"])
            if (
                candidate.source_timescale_identity != base.network.identity
                or candidate.identity != patch["network_identity"]
            ):
                raise ValueError("Reconstructed recurrent resource identity changed")
            candidate.restore(copy.deepcopy(saved["network"]))
            require_equal(
                candidate.snapshot(),
                saved["network"],
                "Nonzero saved resource state restoration",
            )
            trial = cls.__new__(cls)
            trial.__dict__.update(transferred_fields(base))
            trial.network = candidate
            setattr(trial, PATCH_KEY, copy.deepcopy(patch))
            require_saved_whole_equal(
                trial.snapshot(), saved, "Whole recurrent resource restoration"
            )
            base.closed = True
            return trial
        except Exception as error:
            cleanup_owned(base, error)
            raise

    def restore(self, saved):
        if self.closed:
            raise ValueError("Open recurrent resource trial required")
        patch = checked_checkpoint(saved)
        require_equal(
            patch,
            getattr(self, PATCH_KEY),
            "Same recurrent resource configuration for restore",
        )
        replacement = type(self).from_checkpoint(self.graph, saved)
        old_body = self.body
        self.__dict__.clear()
        self.__dict__.update(replacement.__dict__)
        replacement.closed = True
        try:
            old_body.close()
        except Exception as error:
            error.add_note(
                "RecoveryResourceTrial.restore committed the new whole session; old body cleanup failed"
            )
            raise
