"""E271 experimental source filter above the unchanged whole monotone trial.

Construction transfers the caller's body only after successful validation.
Whole restore constructs a fresh body and network; callers must budget that
cost. A successful restore commits the new session before closing the old body.
If that final close fails, the new session remains committed and usable, and
the cleanup exception explicitly reports that fact. No neural-only shortcut is
provided for whole-session restore.
"""

from __future__ import annotations

import copy
import hashlib

import numpy as np

from tools.banc_m3_monotone_recruitment import MonotoneRecruitmentTrial
from tools.banc_m3_motor_timescale import MotorTimescaleNetwork
from tools.banc_m3_source_filtered_synapses import (
    PRECISION,
    SCHEMA,
    TAU,
    SourceFilteredSynapseNetwork,
    checked_selection,
)
from tools.probe_c_banc_m3_core_rhythm import source_matrix

TRIAL_SCHEMA = "flylab.m3-source-filter-trial.v1"
PATCH_SCHEMA = "flylab.m3-source-filter-trial-patch.v1"
DYNAMIC = {"rate", "adaptation", "drive", "output_mask", "tick"}
PATCH_KEYS = {"schema", "source_trial_schema", "source_network_identity", "network_identity",
              "source_adapter_identity", "graph_hash", "source_network_metadata", "selection",
              "source_monotone_patch", "patch_identity"}


def equal(left, right):
    """Strict bytes and types, including wall time and nested provenance."""
    if isinstance(right, dict):
        return isinstance(left, dict) and left.keys() == right.keys() and all(
            equal(left[key], value) for key, value in right.items())
    if isinstance(right, np.ndarray):
        return (isinstance(left, np.ndarray) and left.dtype == right.dtype
                and left.shape == right.shape and left.tobytes() == right.tobytes())
    if isinstance(right, (list, tuple)):
        return type(left) is type(right) and len(left) == len(right) and all(
            equal(a, b) for a, b in zip(left, right))
    if isinstance(right, (float, np.floating)):
        return type(left) is type(right) and np.asarray(left).tobytes() == np.asarray(right).tobytes()
    return type(left) is type(right) and left == right


def require_equal(actual, expected, label):
    if not equal(actual, expected):
        raise ValueError(label + " differs")


def provenance_hash(value):
    digest = hashlib.sha256()
    def visit(item):
        digest.update(type(item).__name__.encode() + b":")
        if isinstance(item, dict):
            for key in sorted(item):
                visit(key)
                visit(item[key])
        elif isinstance(item, np.ndarray):
            digest.update(item.dtype.str.encode() + repr(item.shape).encode() + item.tobytes())
        elif isinstance(item, (list, tuple)):
            for sub in item:
                visit(sub)
        else:
            digest.update(repr(item).encode())
        digest.update(b";")
    visit(value)
    return digest.hexdigest()


def filter_identity(metadata):
    """Reproduce E267's identity without constructing a neural model."""
    digest = hashlib.sha256((metadata["source_model_hash"] + SCHEMA + PRECISION).encode())
    for key in sorted(metadata):
        value = metadata[key]
        if isinstance(value, np.ndarray):
            digest.update(key.encode() + value.dtype.str.encode() + repr(value.shape).encode() + value.tobytes())
    return digest.hexdigest()


def metadata_without_state(saved):
    return copy.deepcopy({key: value for key, value in saved.items() if key not in DYNAMIC})


def validate_neutral(saved):
    """This wrapper permits new coefficients only at the exact neutral origin."""
    if not isinstance(saved, dict) or type(saved.get("tick")) is not int or saved["tick"] != 0:
        raise ValueError("New source candidate requires neutral tick zero")
    rate = saved.get("rate")
    if not isinstance(rate, np.ndarray) or rate.ndim != 1 or rate.dtype != np.float32:
        raise ValueError("Native float32 neutral rate required")
    for key in ("rate", "adaptation", "drive", "output_mask"):
        value = saved.get(key)
        target = np.ones_like(rate) if key == "output_mask" else np.zeros_like(rate)
        if not equal(value, target):
            raise ValueError("Neutral original " + key + " bytes required")
    if "source_filter" in saved or "source_history" in saved:
        raise ValueError("Unmodified source MotorTimescale snapshot required")


def initialize_neutral(candidate, saved):
    """Keep candidate metadata; validate the source before copying neutral arrays."""
    validate_neutral(saved)
    if saved.get("model_hash") != candidate.source_timescale_identity:
        raise ValueError("Original timescale identity changed")
    template = candidate.snapshot()
    inherited = metadata_without_state(template)
    inherited.pop("source_filter")
    inherited.pop("source_history", None)
    inherited["model_hash"] = candidate.source_timescale_identity
    require_equal(inherited, metadata_without_state(saved), "Inherited original neural metadata")
    for key in DYNAMIC:
        template[key] = copy.deepcopy(saved[key])
    template["source_history"] = np.zeros_like(template["source_history"], dtype=np.float64)
    candidate.restore(template)  # E267's atomic metadata/parameter/state validation.
    require_equal(candidate.snapshot(), template, "Neutral candidate initialization")


def make_candidate(old, selection):
    if type(old) is not MotorTimescaleNetwork:
        raise TypeError("Exact original MotorTimescaleNetwork required")
    weights = source_matrix(old)
    native = checked_selection(weights, selection["edge_ids"], selection["coefficients64"], selection["free_motor_ids"])
    require_equal(native, selection, "Native source edge selection")
    implementation = old.cuda_implementation
    if old.device == "cpu" and implementation == "torch-cpu-csr":
        implementation = "reference"
    return SourceFilteredSynapseNetwork(weights,
        *(getattr(old, key).cpu().numpy().copy() for key in ("tau", "a", "threshold", "cap")),
        adaptation_gain=old.adaptation_gain, adaptation_tau_s=old.adaptation_tau_s,
        dt=old.dt, device=old.device, cuda_implementation=implementation,
        capture_steps=old.interval_steps, core_ids=old.core_ids.copy(), core_gain=old.core_gain,
        motor_ids=old.motor_ids.copy(), motor_gains=old.motor_gains.copy(),
        motor_log_tau=old.motor_log_tau.copy(), edge_ids=native["edge_ids"],
        coefficients=native["coefficients64"], free_motor_ids=native["free_motor_ids"])


def validate_selection_metadata(selection, n, motor_ids):
    keys = {"edge_ids", "coefficients64", "coefficients32", "free_motor_ids", "pre_ids",
            "post_ids", "weights32", "source_ids", "edge_source_slots"}
    if not isinstance(selection, dict) or set(selection) != keys:
        raise ValueError("Complete E267 selection metadata required")
    ids, free, pre, post, sources = (selection[key] for key in
        ("edge_ids", "free_motor_ids", "pre_ids", "post_ids", "source_ids"))
    for values in (ids, free, sources):
        if (not isinstance(values, np.ndarray) or values.dtype != np.int64 or values.ndim != 1
                or not len(values) or np.any(values < 0) or np.any(values[1:] <= values[:-1])):
            raise ValueError("Sorted unique native selection identities required")
    for values in (pre, post):
        if (not isinstance(values, np.ndarray) or values.dtype != np.int64
                or values.shape != ids.shape or np.any(values < 0) or np.any(values >= n)):
            raise ValueError("In-range native selected endpoints required")
    c, c32, weights, slots = (selection[key] for key in
        ("coefficients64", "coefficients32", "weights32", "edge_source_slots"))
    if (not isinstance(c, np.ndarray) or c.dtype != np.float64 or c.shape != (len(ids), 3)
            or not np.isfinite(c).all() or np.any(c < 0)
            or np.any(c.sum(axis=1) < .125) or np.any(c.sum(axis=1) > 8)
            or not equal(c32, c.astype(np.float32))
            or np.any(c32.astype(np.float64).sum(axis=1) < .125)
            or np.any(c32.astype(np.float64).sum(axis=1) > 8)
            or not isinstance(weights, np.ndarray) or weights.dtype != np.float32
            or weights.shape != ids.shape or not np.isfinite(weights).all() or np.any(weights == 0)
            or not np.isin(post, free).all() or not np.isin(free, motor_ids).all()
            or not equal(sources, np.unique(pre))
            or not equal(slots, np.searchsorted(sources, pre).astype(np.int32))):
        raise ValueError("Source selection coefficients, endpoints or roster invalid")


def transferred_fields(base):
    """Transfer live resources, but own copied arrays and mutable provenance."""
    return {key: copy.deepcopy(value) if isinstance(value, (np.ndarray, dict, list, tuple)) else value
            for key, value in base.__dict__.items()}


def checked_checkpoint(saved):
    """Reject structural and provenance errors before constructing any body."""
    if not isinstance(saved, dict) or saved.get("schema") != TRIAL_SCHEMA:
        raise ValueError("Explicit source filter whole checkpoint required")
    patch, network = saved.get("source_filter_trial_patch"), saved.get("network")
    if (not isinstance(patch, dict) or set(patch) != PATCH_KEYS
            or patch["schema"] != PATCH_SCHEMA
            or patch["source_trial_schema"] != MonotoneRecruitmentTrial.schema
            or not isinstance(network, dict)):
        raise ValueError("Complete source filter trial provenance required")
    unhashed = copy.deepcopy(patch)
    checksum = unhashed.pop("patch_identity")
    if checksum != provenance_hash(unhashed):
        raise ValueError("Source filter trial patch hash mismatch")
    source_meta = patch["source_network_metadata"]
    if (not isinstance(source_meta, dict) or source_meta.get("model_hash") != patch["source_network_identity"]
            or saved.get("graph_hash") != patch["graph_hash"]
            or saved.get("adapter", {}).get("hash") != patch["source_adapter_identity"]):
        raise ValueError("Source whole-session identity changed")
    require_equal(saved.get("monotone_recruitment_patch"), patch["source_monotone_patch"], "Original monotone patch")
    selection = patch["selection"]
    metadata = network.get("source_filter")
    expected = {"schema": SCHEMA, "source_model_hash": patch["source_network_identity"],
                "precision": PRECISION, "tau": TAU.copy(), **copy.deepcopy(selection)}
    require_equal(metadata, expected, "Source coefficients and selection metadata")
    if (network.get("model_hash") != patch["network_identity"]
            or filter_identity(metadata) != patch["network_identity"]):
        raise ValueError("Source-filter model identity mismatch")
    inherited = metadata_without_state(network)
    inherited.pop("source_filter")
    inherited.pop("source_history")
    inherited["model_hash"] = patch["source_network_identity"]
    require_equal(inherited, source_meta, "Saved inherited neural metadata")
    sources, history = selection.get("source_ids"), network.get("source_history")
    if (not isinstance(sources, np.ndarray) or sources.dtype != np.int64 or sources.ndim != 1
            or not isinstance(history, np.ndarray) or history.dtype != np.float64
            or history.shape != (len(sources), 2) or not np.isfinite(history).all()):
        raise ValueError("Complete finite native source history required")
    rate = network.get("rate")
    if (not isinstance(rate, np.ndarray) or rate.ndim != 1 or rate.dtype != np.float32
            or type(network.get("tick")) is not int or network["tick"] < 0):
        raise ValueError("Complete native neural state required")
    motor_ids = patch["source_monotone_patch"].get("motor_ids")
    if not isinstance(motor_ids, np.ndarray) or motor_ids.dtype != np.int64:
        raise ValueError("Original monotone native motor roster required")
    validate_selection_metadata(selection, len(rate), motor_ids)
    for key in ("rate", "adaptation", "drive", "output_mask"):
        value = network.get(key)
        if (not isinstance(value, np.ndarray) or value.shape != rate.shape or value.dtype != np.float32
                or not np.isfinite(value).all()):
            raise ValueError("Invalid saved neural " + key)
    if (np.any(rate < -1e-4) or np.any(network["adaptation"] < 0)
            or not np.isin(network["output_mask"], (0, 1)).all()):
        raise ValueError("Invalid inherited neural values")
    return copy.deepcopy(patch)


def remove_source_layer(saved):
    """E150-shaped copy for its original whole from_checkpoint path only."""
    patch = checked_checkpoint(saved)
    source = copy.deepcopy(saved)
    source.pop("source_filter_trial_patch")
    source["schema"] = MonotoneRecruitmentTrial.schema
    source["network"].pop("source_filter")
    source["network"].pop("source_history")
    source["network"]["model_hash"] = patch["source_network_identity"]
    return source


def cleanup_owned(owner, error):
    try:
        if owner is not None and not owner.closed:
            owner.close()
    except Exception as secondary:  # noqa: BLE001
        error.add_note("Newly owned body cleanup failed: " + repr(secondary))


class SourceFilterTrial(MonotoneRecruitmentTrial):
    """Replace only the neural layer of one open original monotone session."""

    schema = TRIAL_SCHEMA

    def __init__(self, base, edge_ids, coefficients, free_motor_ids):
        if type(base) is not MonotoneRecruitmentTrial or base.closed:
            raise TypeError("Exact open original MonotoneRecruitmentTrial required")
        before = base.snapshot()
        validate_neutral(before["network"])
        selection = checked_selection(source_matrix(base.network), edge_ids, coefficients, free_motor_ids)
        require_equal(base.monotone_recruitment_patch.get("motor_ids"), base.network.motor_ids,
                      "Original monotone and native motor roster")
        candidate = make_candidate(base.network, selection)
        initialize_neutral(candidate, before["network"])
        patch = {"schema": PATCH_SCHEMA, "source_trial_schema": MonotoneRecruitmentTrial.schema,
                 "source_network_identity": base.network.identity, "network_identity": candidate.identity,
                 "source_adapter_identity": base.adapter.identity, "graph_hash": base.graph.hash,
                 "source_network_metadata": metadata_without_state(before["network"]),
                 "selection": candidate.selection,
                 "source_monotone_patch": copy.deepcopy(base.monotone_recruitment_patch)}
        patch["patch_identity"] = provenance_hash(patch)
        self.__dict__.update(transferred_fields(base))
        self.network = candidate
        self.source_filter_trial_patch = copy.deepcopy(patch)
        actual = self.snapshot()
        expected = copy.deepcopy(before)
        expected.update(schema=self.schema, network=candidate.snapshot(), source_filter_trial_patch=copy.deepcopy(patch))
        require_equal(actual, expected, "Only source neural layer changed")
        base.closed = True  # Body ownership transfers only after all checks succeed.

    def snapshot(self):
        saved = MonotoneRecruitmentTrial.snapshot(self)
        saved["schema"] = self.schema
        saved["source_filter_trial_patch"] = copy.deepcopy(self.source_filter_trial_patch)
        return saved

    def save(self, path):
        from flylab.c.storage import StateStore

        StateStore.save(path, self.snapshot())

    @classmethod
    def from_checkpoint(cls, graph, saved):
        patch = checked_checkpoint(saved)
        if graph.hash != patch["graph_hash"]:
            raise ValueError("Source trial graph identity mismatch")
        source = remove_source_layer(saved)
        base = MonotoneRecruitmentTrial.from_checkpoint(graph, source)
        trial = None
        try:
            require_equal(base.snapshot(), source, "Original whole monotone restoration")
            if base.network.identity != patch["source_network_identity"] or base.adapter.identity != patch["source_adapter_identity"]:
                raise ValueError("Reconstructed source neural/adapter identity changed")
            candidate = make_candidate(base.network, patch["selection"])
            if candidate.source_timescale_identity != base.network.identity or candidate.identity != patch["network_identity"]:
                raise ValueError("Reconstructed source filtered identity changed")
            candidate.restore(copy.deepcopy(saved["network"]))
            require_equal(candidate.snapshot(), saved["network"], "Whole saved source history restoration")
            trial = cls.__new__(cls)
            trial.__dict__.update(transferred_fields(base))
            trial.network = candidate
            trial.source_filter_trial_patch = copy.deepcopy(patch)
            require_equal(trial.snapshot(), saved, "Whole source filter checkpoint")
            base.closed = True
            return trial
        except Exception as error:
            # Until return, base owns the body; do not close both shared objects.
            cleanup_owned(base, error)
            raise

    def restore(self, saved):
        """Fresh whole reconstruction followed by one ownership swap."""
        if self.closed:
            raise ValueError("Open source filter trial required for restore")
        patch = checked_checkpoint(saved)
        require_equal(patch, self.source_filter_trial_patch, "Same trial configuration for restore")
        replacement = type(self).from_checkpoint(self.graph, saved)
        old_body = self.body
        self.__dict__.clear()
        self.__dict__.update(replacement.__dict__)
        replacement.closed = True
        try:
            old_body.close()
        except Exception as error:
            error.add_note("SourceFilterTrial.restore committed the new whole session; old body cleanup failed")
            raise
