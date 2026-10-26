"""E327 efficacy layer over the current SSPRK33/u0 whole trial.

The selected CSR values belong to a new network.  The source whole state is
recoverable without changing a captured pointer or claiming its model hash.
"""
from __future__ import annotations

import copy

import numpy as np
from scipy.sparse import csr_matrix

from tools import banc_m3_ssp_recovery_trial as source
from tools.banc_m3_source_filter_trial import (
    cleanup_owned,
    provenance_hash,
    require_equal,
    transferred_fields,
)
from tools.probe_c_banc_m3_core_rhythm import source_matrix
from tools.verify_c_banc_m3_source_native_transfer import checkpoint_representation

TRIAL_SCHEMA = "flylab.m3-ssp-efficacy-trial.v1"
PATCH_SCHEMA = "flylab.m3-ssp-efficacy-patch.v1"
PATCH_KEY = "ssp_efficacy_trial_patch"
DYNAMIC = source.DYNAMIC
PATCH_KEYS = {"schema", "source_trial_schema", "source_network_identity",
              "network_identity", "source_adapter_identity", "graph_hash",
              "source_network_metadata", "candidate_network_metadata",
              "positions", "edge_parameters", "theta64", "original32",
              "actual32", "source_patch_identity", "patch_identity"}


def patch_csr(weights: csr_matrix, positions: np.ndarray,
              edge_parameters: np.ndarray, theta: np.ndarray):
    """Return an owned exact-topology CSR and the f32 selected-value roster.

    Multiplication is f64(original f32) * f64(theta), rounded once to f32.
    Neutral theta is special-cased to preserve the original selected bytes.
    """
    if (not isinstance(weights, csr_matrix) or weights.dtype != np.float32
            or weights.shape[0] != weights.shape[1] or not weights.has_sorted_indices
            or not weights.has_canonical_format or not np.isfinite(weights.data).all()):
        raise ValueError("Canonical finite native float32 CSR required")
    weights.check_format(full_check=True)
    pos, groups, gain = positions, edge_parameters, theta
    if (not isinstance(pos, np.ndarray) or pos.dtype != np.int64 or pos.ndim != 1
            or not len(pos) or np.any(pos < 0) or np.any(pos >= weights.nnz)
            or len(np.unique(pos)) != len(pos)
            or not isinstance(groups, np.ndarray) or groups.dtype != np.int64
            or groups.shape != pos.shape or np.any(groups < 0)
            or not isinstance(gain, np.ndarray) or gain.dtype != np.float64
            or gain.ndim != 1 or not len(gain) or not np.isfinite(gain).all()
            or np.any(gain <= 0) or np.any(groups >= len(gain))
            or not np.array_equal(np.unique(groups), np.arange(len(gain)))
            or np.any(weights.data[pos] == 0)):
        raise ValueError("Exact selected existing edges, groups, positive f64 theta required")
    original = weights.data[pos].copy()
    actual = np.asarray(original.astype(np.float64) * gain[groups], np.float32)
    actual[gain[groups] == 1.] = original[gain[groups] == 1.]
    if (not np.isfinite(actual).all() or np.any(actual == 0)
            or np.any(np.signbit(actual) != np.signbit(original))):
        raise ValueError("Selected efficacy must retain finite nonzero edge signs")
    patched = weights.copy()
    patched.data[pos] = actual
    if (not np.array_equal(patched.indptr, weights.indptr)
            or not np.array_equal(patched.indices, weights.indices)):
        raise AssertionError("CSR topology changed")
    unselected = np.ones(weights.nnz, bool)
    unselected[pos] = False
    if patched.data[unselected].tobytes() != weights.data[unselected].tobytes():
        raise AssertionError("Unselected CSR changed")
    return patched, original, actual


def make_candidate(old, positions, edge_parameters, theta):
    """Build the same SSP/u0 class with a new CSR and all original settings."""
    api = source.network_api()
    if type(old) is not api.SSPRecoveryResourceNetwork:
        raise TypeError("Exact current SSPRecoveryResourceNetwork required")
    matrix, original, actual = patch_csr(source_matrix(old), positions,
                                         edge_parameters, theta)
    selection = old.selection
    checked = api.checked_selection(matrix, selection["edge_ids"],
                                    selection["premotor_ids"],
                                    selection["utilization"],
                                    selection["recovery_tau"], old.motor_ids)
    implementation = old.cuda_implementation
    if old.device == "cpu" and implementation == "torch-cpu-csr":
        implementation = "reference"
    candidate = api.SSPRecoveryResourceNetwork(
        matrix,
        *(getattr(old, key).detach().cpu().numpy().copy()
          for key in ("tau", "a", "threshold", "cap")),
        adaptation_gain=old.adaptation_gain,
        adaptation_tau_s=old.adaptation_tau_s,
        dt=old.dt, device=old.device,
        cuda_implementation=implementation,
        capture_steps=old.interval_steps,
        core_ids=old.core_ids.copy(), core_gain=old.core_gain,
        motor_ids=old.motor_ids.copy(), motor_gains=old.motor_gains.copy(),
        motor_log_tau=old.motor_log_tau.copy(),
        edge_ids=checked["edge_ids"], premotor_ids=checked["premotor_ids"],
        utilization=checked["utilization"], recovery_tau=checked["recovery_tau"])
    require_equal(candidate.selection, checked, "Rebuilt patched resource selection")
    old_state = old.snapshot()
    state = candidate.snapshot()
    for key in DYNAMIC:
        state[key] = copy.deepcopy(old_state[key])
    candidate.restore(state)
    observed = candidate.snapshot()
    for key in DYNAMIC:
        require_equal(observed[key], old_state[key], "Transferred dynamic " + key)
    if candidate.identity == old.identity and not np.array_equal(actual, original):
        raise ValueError("Patched network retained original identity")
    return candidate, original, actual


def metadata_without_state(saved):
    return source.metadata_without_state(saved)


def _source_representation(saved, patch):
    """Inverse only the efficacy layer, retaining every whole dynamic value."""
    old = copy.deepcopy(saved)
    old.pop(PATCH_KEY)
    old["schema"] = source.TRIAL_SCHEMA
    old["network"] = {**copy.deepcopy(patch["source_network_metadata"]),
                      **{key: copy.deepcopy(saved["network"][key]) for key in DYNAMIC}}
    return old


def checked_checkpoint(saved):
    if not isinstance(saved, dict) or saved.get("schema") != TRIAL_SCHEMA:
        raise ValueError("Explicit SSP efficacy whole checkpoint required")
    patch, network = saved.get(PATCH_KEY), saved.get("network")
    if (not isinstance(patch, dict) or set(patch) != PATCH_KEYS
            or patch.get("schema") != PATCH_SCHEMA
            or patch.get("source_trial_schema") != source.TRIAL_SCHEMA
            or not isinstance(network, dict)):
        raise ValueError("Complete SSP efficacy provenance required")
    unhashed = copy.deepcopy(patch)
    checksum = unhashed.pop("patch_identity")
    if checksum != provenance_hash(unhashed):
        raise ValueError("SSP efficacy patch checksum changed")
    if (saved.get("graph_hash") != patch["graph_hash"]
            or saved.get("adapter", {}).get("hash") != patch["source_adapter_identity"]
            or saved.get(source.PATCH_KEY, {}).get("patch_identity")
               != patch["source_patch_identity"]
            or network.get("model_hash") != patch["network_identity"]):
        raise ValueError("Whole source/candidate identity changed")
    require_equal(metadata_without_state(network),
                  patch["candidate_network_metadata"],
                  "Candidate static neural metadata")
    old = _source_representation(saved, patch)
    source.checked_checkpoint(old)
    if old["network"]["model_hash"] != patch["source_network_identity"]:
        raise ValueError("Source SSP model identity changed")
    if (not isinstance(patch["original32"], np.ndarray)
            or patch["original32"].dtype != np.float32
            or not isinstance(patch["actual32"], np.ndarray)
            or patch["actual32"].dtype != np.float32
            or patch["original32"].shape != patch["positions"].shape
            or patch["actual32"].shape != patch["positions"].shape):
        raise ValueError("Native selected weight roster required")
    if (not np.array_equal(patch["actual32"],
            np.asarray(patch["original32"].astype(np.float64)
                       * patch["theta64"][patch["edge_parameters"]], np.float32))
            or np.any(np.signbit(patch["actual32"])
                      != np.signbit(patch["original32"]))):
        raise ValueError("Selected efficacy values changed")
    return copy.deepcopy(patch)


class SSPEfficacyTrial(source.SSPRecoveryResourceTrial):
    schema = TRIAL_SCHEMA

    def __init__(self, base, positions, edge_parameters, theta):
        if type(base) is not source.SSPRecoveryResourceTrial or base.closed:
            raise TypeError("Exact open current SSP whole trial required")
        before = base.snapshot()
        source.checked_checkpoint(before)
        candidate, original, actual = make_candidate(base.network, positions,
                                                       edge_parameters, theta)
        after = candidate.snapshot()
        patch = {
            "schema": PATCH_SCHEMA,
            "source_trial_schema": source.TRIAL_SCHEMA,
            "source_network_identity": base.network.identity,
            "network_identity": candidate.identity,
            "source_adapter_identity": base.adapter.identity,
            "graph_hash": base.graph.hash,
            "source_network_metadata": metadata_without_state(before["network"]),
            "candidate_network_metadata": metadata_without_state(after),
            "positions": positions.copy(),
            "edge_parameters": edge_parameters.copy(),
            "theta64": theta.copy(),
            "original32": original,
            "actual32": actual,
            "source_patch_identity": before[source.PATCH_KEY]["patch_identity"],
        }
        changes = []
        require_equal(patch, checkpoint_representation(patch, changes),
                      "Storage-stable efficacy provenance")
        patch["patch_identity"] = provenance_hash(patch)
        require_equal(base.snapshot(), before, "Caller whole source unchanged")
        self.__dict__.update(transferred_fields(base))
        self.network = candidate
        setattr(self, PATCH_KEY, copy.deepcopy(patch))
        expected = copy.deepcopy(before)
        expected.update(schema=self.schema, network=after,
                        **{PATCH_KEY: copy.deepcopy(patch)})
        require_equal(self.snapshot(), expected, "Only efficacy layer changed")
        checked_checkpoint(self.snapshot())
        base.closed = True

    def snapshot(self):
        saved = source.SSPRecoveryResourceTrial.snapshot(self)
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
            raise ValueError("Whole graph identity changed")
        old = _source_representation(saved, patch)
        base = source.SSPRecoveryResourceTrial.from_checkpoint(graph, old)
        try:
            source.require_saved_whole_equal(base.snapshot(), old,
                                             "Whole source SSP reconstruction")
            if (base.network.identity != patch["source_network_identity"]
                    or base.adapter.identity != patch["source_adapter_identity"]):
                raise ValueError("Reconstructed source identity changed")
            candidate, original, actual = make_candidate(
                base.network, patch["positions"], patch["edge_parameters"], patch["theta64"])
            require_equal(original, patch["original32"], "Rebuilt original edge values")
            require_equal(actual, patch["actual32"], "Rebuilt efficacy edge values")
            if candidate.identity != patch["network_identity"]:
                raise ValueError("Rebuilt candidate model identity changed")
            require_equal(candidate.snapshot(), saved["network"],
                          "Rebuilt candidate neural state")
            trial = cls.__new__(cls)
            trial.__dict__.update(transferred_fields(base))
            trial.network = candidate
            setattr(trial, PATCH_KEY, copy.deepcopy(patch))
            source.require_saved_whole_equal(trial.snapshot(), saved,
                                             "Whole efficacy reconstruction")
            base.closed = True
            return trial
        except Exception as error:
            cleanup_owned(base, error)
            raise

    def restore(self, saved):
        if self.closed:
            raise ValueError("Open efficacy trial required")
        patch = checked_checkpoint(saved)
        require_equal(patch, getattr(self, PATCH_KEY), "Same efficacy configuration")
        replacement = type(self).from_checkpoint(self.graph, saved)
        old_body = self.body
        self.__dict__.clear()
        self.__dict__.update(replacement.__dict__)
        replacement.closed = True
        try:
            old_body.close()
        except Exception as error:
            error.add_note("SSPEfficacyTrial.restore committed replacement; old-body close failed")
            raise
