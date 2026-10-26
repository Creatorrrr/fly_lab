"""E224 two-arm canonical body wrapper with a separate neural-current effector."""

# Owned-session cleanup must preserve the primary construction error.
# ruff: noqa: BLE001

from __future__ import annotations

import copy
import hashlib

import numpy as np

from tools.banc_m3_motor_timescale import MotorTimescaleTrial
from tools.banc_m3_native_grouped_trial import (
    CanonicalRate64Adapter,
    _array_bytes,
    checked_grouped_parameters,
    unwrap_e150_checkpoint,
)
from tools.banc_m3_neural_current_effector import NeuralCurrentEffectorNetwork
from tools.probe_c_banc_m3_core_rhythm import source_matrix
from tools.verify_c_banc_observation import exact

SCHEMA = "flylab.m3-neural-current-body.v1"
PATCH_SCHEMA = "flylab.m3-neural-current-body-patch.v1"
BRIDGE_SCHEMA = "flylab.m3-neural-current-effector-bridge.v1"
MODES = ("legacy_canonical", "effector_canonical")


class EffectorCanonicalBridge:
    """Only decoder input changes; encode and neuronal readout stay original."""

    def __init__(self, canonical, network):
        if not isinstance(canonical, CanonicalRate64Adapter):
            raise TypeError("Canonical float64 anatomical adapter required")
        if not isinstance(network, NeuralCurrentEffectorNetwork):
            raise TypeError("Separate E223 effector network required")
        if not np.array_equal(canonical.motor_ids, network.native_motor_ids):
            raise ValueError("Canonical and effector motor order changed")
        self.canonical = canonical
        self.network = network
        self.identity = hashlib.sha256((BRIDGE_SCHEMA + canonical.identity
                                        + network.identity).encode("ascii")).hexdigest()
        self.last_decoder_input32 = np.zeros(len(network.native_motor_ids), np.float32)

    def __getattr__(self, key):
        return getattr(self.canonical, key)

    def decode(self, rates, *, motor_cut=False, dt=.005):
        original = np.asarray(rates)
        if (original.shape != (391,) or original.dtype != np.float32
                or not np.isfinite(original).all()):
            raise ValueError("Original float32 neuronal motor readout required")
        effector = self.network.effector_readout()
        self.last_decoder_input32 = effector.copy()
        return self.canonical.decode(effector, motor_cut=motor_cut, dt=dt)

    def snapshot(self):
        return {"schema": BRIDGE_SCHEMA, "hash": self.identity,
                "canonical_identity": self.canonical.identity,
                "network_identity": self.network.identity,
                "canonical": self.canonical.snapshot(),
                "last_decoder_input32": self.last_decoder_input32.copy()}

    def restore(self, saved):
        if (not isinstance(saved, dict) or set(saved) != {
                "schema", "hash", "canonical_identity", "network_identity",
                "canonical", "last_decoder_input32"}
                or saved["schema"] != BRIDGE_SCHEMA
                or saved["hash"] != self.identity
                or saved["canonical_identity"] != self.canonical.identity
                or saved["network_identity"] != self.network.identity):
            raise ValueError("Complete neural-current bridge identity required")
        last = saved["last_decoder_input32"]
        if (not isinstance(last, np.ndarray) or last.dtype != np.float32
                or last.shape != (391,) or not np.isfinite(last).all()
                or np.any(last < 0)):
            raise ValueError("Finite decoder input checkpoint required")
        self.canonical.restore(saved["canonical"])
        self.last_decoder_input32 = last.copy()


def unwrap_neural_current_checkpoint(saved):
    """Return an original timescale state; never serialize a network pointer."""
    if not isinstance(saved, dict) or saved.get("schema") != SCHEMA:
        raise ValueError("Explicit E224 trial checkpoint required")
    patch = saved.get("neural_current_trial_patch")
    if (not isinstance(patch, dict) or patch.get("schema") != PATCH_SCHEMA
            or patch.get("mode") not in MODES):
        raise ValueError("Complete E224 trial mode/provenance required")
    source = copy.deepcopy(saved)
    source.pop("neural_current_trial_patch")
    source["schema"] = MotorTimescaleTrial.schema
    network, adapter = source.get("network"), source.get("adapter")
    if not isinstance(network, dict) or not isinstance(adapter, dict):
        raise TypeError("E224 network and adapter checkpoints required")
    effector_patch = None
    if patch["mode"] == "effector_canonical":
        effector_patch = network.pop("neural_current_effector", None)
        if not isinstance(effector_patch, dict):
            raise ValueError("Separate E223 effector checkpoint required")
        if adapter.get("schema") != BRIDGE_SCHEMA:
            raise ValueError("E224 effector bridge checkpoint required")
        source["adapter"] = adapter["canonical"]
        network["model_hash"] = patch["source_timescale_identity"]
    elif "neural_current_effector" in network or adapter.get("schema") == BRIDGE_SCHEMA:
        raise ValueError("Legacy arm cannot reinterpret an effector checkpoint")
    return source, patch, effector_patch


class NeuralCurrentTrial(MotorTimescaleTrial):
    """Own one exact E150 body and select one preregistered motor decoder."""

    schema = SCHEMA

    def __init__(self, base, parameters, prototypes, beta, mode, *,
                 effector_network_snapshot=None, bridge_snapshot=None):
        if type(base) is not MotorTimescaleTrial or mode not in MODES:
            raise TypeError("Exact E150 timescale trial and fixed arm mode required")
        self.__dict__.update(base.__dict__)
        try:
            p, c, b = checked_grouped_parameters(base, parameters, prototypes, beta)
            old = base.network
            source = old.snapshot()
            if (self.control_tick == 0 and old.tick == 0
                    and (np.any(source["rate"] != 0)
                         or np.any(source["adaptation"] != 0)
                         or np.any(source["drive"] != 0)
                         or np.any(base.adapter.activation != 0))):
                raise ValueError("E224 start requires zero original neuron/muscle state")
            candidate = None
            if mode == "effector_canonical":
                free, fixed = p["free_columns"], p["fixed_columns"]
                candidate = NeuralCurrentEffectorNetwork(source_matrix(old),
                    *(getattr(old, key).cpu().numpy().copy()
                      for key in ("tau", "a", "threshold", "cap")),
                    adaptation_gain=old.adaptation_gain,
                    adaptation_tau_s=old.adaptation_tau_s, dt=old.dt,
                    device=old.device, cuda_implementation=old.cuda_implementation,
                    capture_steps=old.interval_steps, core_ids=old.core_ids,
                    core_gain=old.core_gain, motor_ids=old.motor_ids,
                    motor_gains=old.motor_gains, motor_log_tau=old.motor_log_tau,
                    free_motor_ids=old.motor_ids[free],
                    fixed_motor_ids=old.motor_ids[fixed],
                    membership=p["membership"][free], scale=p["scale"][free],
                    centers=c["centers"][free], sigma2=c["sigma2"][free], beta=b)
                if candidate.source_timescale_identity != old.identity:
                    raise ValueError("E223 original timescale source identity changed")
                if effector_network_snapshot is None:
                    if self.control_tick != 0 or old.tick != 0:
                        raise ValueError("New effector bridge requires neutral E150 start")
                    patched = copy.deepcopy(source)
                    patched["model_hash"] = candidate.identity
                    patched["neural_current_effector"] = candidate.snapshot()[
                        "neural_current_effector"]
                else:
                    patched = copy.deepcopy(effector_network_snapshot)
                    unwrapped = copy.deepcopy(patched)
                    native = unwrapped.pop("neural_current_effector")
                    unwrapped["model_hash"] = old.identity
                    exact(source, unwrapped, "E224 exact unwrapped neuron checkpoint")
                    if native.get("source_timescale_identity") != old.identity:
                        raise ValueError("E224 original neuron provenance changed")
                candidate.restore(patched)
            canonical = CanonicalRate64Adapter(base.adapter.original,
                base.adapter.calibration, base.adapter.groups)
            canonical.restore(base.adapter.snapshot())
            if canonical.identity != base.adapter.identity:
                raise ValueError("E224 canonical anatomy identity changed")
            if mode == "effector_canonical":
                bridge = EffectorCanonicalBridge(canonical, candidate)
                if bridge_snapshot is not None:
                    bridge.restore(bridge_snapshot)
                self.network = candidate
                self.adapter = bridge
            else:
                if effector_network_snapshot is not None or bridge_snapshot is not None:
                    raise ValueError("Legacy arm must not carry effector state")
                self.adapter = canonical
            self.tendon_inputs.update(self.adapter.tendon_inputs)
            self.neural_current_trial_patch = {
                "schema": PATCH_SCHEMA, "mode": mode,
                "source_timescale_identity": old.identity,
                "source_adapter_identity": base.adapter.identity,
                "network_identity": self.network.identity,
                "adapter_identity": self.adapter.identity,
                "graph_hash": self.graph.hash,
                "parameter_sha256": {key: _array_bytes(value)
                                     for key, value in p.items()},
                "prototype_sha256": {key: _array_bytes(value)
                                     for key, value in c.items()},
                "beta_sha256": _array_bytes(b),
                "parameters": copy.deepcopy(p),
                "prototypes": copy.deepcopy(c), "beta": b.copy()}
        except BaseException as error:
            base.closed = True
            try:
                self.close()
            except BaseException as secondary:
                error.add_note("E224 body cleanup failed: " + repr(secondary))
            raise

    @property
    def mode(self):
        return self.neural_current_trial_patch["mode"]

    def snapshot(self):
        self.tendon_inputs.update(self.adapter.tendon_inputs)
        return {**super().snapshot(),
                "neural_current_trial_patch": copy.deepcopy(
                    self.neural_current_trial_patch)}

    @classmethod
    def from_e150_checkpoint(cls, graph, saved, parameters, prototypes, beta,
                             mode):
        source = unwrap_e150_checkpoint(saved)
        base = MotorTimescaleTrial.from_checkpoint(graph, source)
        trial = None
        try:
            exact(source, base.snapshot(), "E224 unwrapped E150 neutral state")
            trial = cls(base, parameters, prototypes, beta, mode)
            return trial
        except BaseException as error:
            owner = trial if trial is not None else base
            try:
                if not getattr(owner, "closed", False):
                    owner.close()
            except BaseException as secondary:
                error.add_note("E224 source cleanup failed: " + repr(secondary))
            raise

    @classmethod
    def from_checkpoint(cls, graph, saved):
        source, patch, effector = unwrap_neural_current_checkpoint(saved)
        base = MotorTimescaleTrial.from_checkpoint(graph, source)
        trial = None
        try:
            if (base.network.identity != patch["source_timescale_identity"]
                    or base.adapter.identity != patch["source_adapter_identity"]
                    or graph.hash != patch["graph_hash"]):
                raise ValueError("E224 source graph/network/adapter changed")
            p, c, b = patch["parameters"], patch["prototypes"], patch["beta"]
            if (patch["parameter_sha256"] != {key: _array_bytes(value)
                    for key, value in p.items()}
                    or patch["prototype_sha256"] != {key: _array_bytes(value)
                    for key, value in c.items()}
                    or patch["beta_sha256"] != _array_bytes(b)):
                raise ValueError("E224 grouped anatomy/prototype bytes changed")
            trial = cls(base, p, c, b, patch["mode"],
                effector_network_snapshot=(saved["network"] if effector is not None
                                           else None),
                bridge_snapshot=(saved["adapter"] if effector is not None else None))
            if (trial.network.identity != patch["network_identity"]
                    or trial.adapter.identity != patch["adapter_identity"]):
                raise ValueError("E224 trial mode identity changed")
            exact(saved, trial.snapshot(), "E224 whole two-arm checkpoint")
            return trial
        except BaseException as error:
            owner = trial if trial is not None else base
            try:
                if not getattr(owner, "closed", False):
                    owner.close()
            except BaseException as secondary:
                error.add_note("E224 checkpoint cleanup failed: " + repr(secondary))
            raise


__all__ = ("EffectorCanonicalBridge", "NeuralCurrentTrial",
           "unwrap_neural_current_checkpoint")
