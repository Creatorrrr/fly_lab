"""E313 whole-session postsynaptic load wrapper; no autonomous gait claim.

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

TRIAL_SCHEMA = "flylab.m3-postsynaptic-load-trial.v1"
PATCH_SCHEMA = "flylab.m3-postsynaptic-load-trial-patch.v1"
PATCH_KEY = "postsynaptic_load_trial_patch"
DYNAMIC = {"rate", "adaptation", "drive", "output_mask", "tick", "synaptic_resource", "postsynaptic_load_state"}
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
    "load_lambda",
    "load_metadata",
    "original_threshold32",
    "original_tau32",
}
NETWORK_SCHEMA = "flylab.m3-postsynaptic-load.v1"
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
    """Lazy import permits generated fake contract checks before E303 delivery."""
    from tools import banc_m3_postsynaptic_load

    return banc_m3_postsynaptic_load


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
            "postsynaptic_load",
            "postsynaptic_load_state",
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
    if np.any(selection["utilization"] != 0):
        raise ValueError("Postsynaptic load requires all source utilization zero")


def checked_lambda(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)) or value not in (0, 1):
        raise ValueError("Only fixed lambda0/lambda1 permitted")
    return float(value)


def make_candidate(old, selection, load_lambda):
    if type(old) is not MotorTimescaleNetwork:
        raise TypeError("Exact original MotorTimescaleNetwork required")
    load_lambda = checked_lambda(load_lambda)
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
    return api.PostsynapticLoadNetwork(
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
        load_lambda=load_lambda,
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
    metadata = network.get("postsynaptic_load")
    if (
        getattr(api, "SCHEMA", None) != NETWORK_SCHEMA
        or not isinstance(metadata, dict)
        or metadata.get("schema") != NETWORK_SCHEMA
        or metadata.get("precision") != api.PRECISION
    ):
        raise ValueError(
            "Explicit SSP schema/precision required; prior integrator checkpoints are incompatible"
        )
    for key in ("original_threshold32", "original_tau32"):
        value = patch[key]
        if not isinstance(value, np.ndarray) or value.dtype != np.float32 or value.shape != rate.shape or not np.isfinite(value).all() or np.any(value <= 0):
            raise ValueError("Original finite positive threshold/tau32 arrays required")
    load_lambda = checked_lambda(patch["load_lambda"])
    expected = api.load_metadata(patch["source_network_identity"], patch["selection"],
        patch["original_threshold32"], patch["original_tau32"], load_lambda)
    require_equal(
        metadata, expected, "Resource metadata selection and exact parameters"
    )
    api.checked_load_metadata(metadata)
    require_equal(metadata, patch["load_metadata"], "Whole patch load metadata")
    from tools import banc_m3_ssp_recovery_synapses as ssp
    require_equal(network.get("recurrent_resource"), ssp.resource_metadata(patch["source_network_identity"], patch["selection"]), "Underlying SSP u0 metadata")
    if (
        network.get("model_hash") != patch["network_identity"]
        or api.load_identity(metadata) != patch["network_identity"]
    ):
        raise ValueError("Recurrent resource neural identity changed")
    inherited = metadata_without_state(network)
    inherited.pop("recurrent_resource")
    inherited.pop("postsynaptic_load")
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
        np.any(rate < 0)
        or np.any(network["adaptation"] < 0)
        or not np.isin(network["output_mask"], (0, 1)).all()
    ):
        raise ValueError(
            "Nonnegative SSP neural rate/adaptation and native binary mask required"
        )
    resource = network.get("synaptic_resource")
    selection = patch["selection"]
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
    posts = np.unique(selection["post_ids"])
    h = network.get("postsynaptic_load_state")
    if (not isinstance(h, np.ndarray) or h.dtype != np.float64 or h.shape != posts.shape or not np.isfinite(h).all()
            or np.any(h < 0) or np.any(h > 1) or tick == 0 and not equal(h, np.zeros_like(h))):
        raise ValueError("Native bounded selected-post h and tick0 h0 required")
    return copy.deepcopy(patch)


def remove_resource_layer(saved):
    """Original-shaped whole state for body reconstruction, not RK4 future equality."""
    patch = checked_checkpoint(saved)
    source = copy.deepcopy(saved)
    source.pop(PATCH_KEY)
    source["schema"] = MonotoneRecruitmentTrial.schema
    source["network"].pop("recurrent_resource")
    source["network"].pop("synaptic_resource")
    source["network"].pop("postsynaptic_load")
    source["network"].pop("postsynaptic_load_state")
    source["network"]["model_hash"] = patch["source_network_identity"]
    return source


class PostsynapticLoadTrial(MonotoneRecruitmentTrial):
    schema = TRIAL_SCHEMA

    def __init__(self, base, edge_ids, premotor_ids, utilization, recovery_tau, load_lambda):
        load_lambda = checked_lambda(load_lambda)
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
        candidate = make_candidate(base.network, selection, load_lambda)
        require_equal(
            candidate.selection, selection, "Constructed original recurrent selection"
        )
        candidate.import_base_snapshot(copy.deepcopy(before["network"]))
        initial = candidate.snapshot()
        require_equal(initial["postsynaptic_load_state"], np.zeros(len(np.unique(selection["post_ids"])), np.float64), "Initial postsynaptic load zero")
        inherited = metadata_without_state(initial)
        inherited.pop("recurrent_resource")
        inherited.pop("postsynaptic_load")
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
            "load_lambda": float(load_lambda),
            "load_metadata": copy.deepcopy(initial["postsynaptic_load"]),
            "original_threshold32": base.network.threshold.cpu().numpy().copy(),
            "original_tau32": base.network.tau.cpu().numpy().copy(),
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
            for key, attr in (("original_threshold32", "threshold"), ("original_tau32", "tau")):
                require_equal(getattr(base.network, attr).cpu().numpy(), patch[key], "Original reconstructed " + key)
            candidate = make_candidate(base.network, patch["selection"], patch["load_lambda"])
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
                "PostsynapticLoadTrial.restore committed the new whole session; old body cleanup failed"
            )
            raise
