"""E222 trial layer: E221 recurrent network with original canonical anatomy."""

# Broad catches retain the primary model error across owned-body cleanup.
# ruff: noqa: BLE001

from __future__ import annotations

import copy
import hashlib

import numpy as np

from tools.banc_m3_monotone_recruitment import MonotoneRecruitmentTrial
from tools.banc_m3_motor_timescale import MotorTimescaleNetwork, MotorTimescaleTrial
from tools.banc_m3_native_grouped_response import NativeGroupedResponseNetwork
from tools.banc_m3_rhythmic_recruitment import AnatomicalRecruitmentAdapter
from tools.probe_c_banc_m3_core_rhythm import source_matrix
from tools.verify_c_banc_observation import exact

SCHEMA = "flylab.m3-native-grouped-body.v1"
PATCH_SCHEMA = "flylab.m3-native-grouped-body-patch.v1"


def _array_bytes(value):
    value = np.asarray(value)
    digest = hashlib.sha256()
    digest.update(value.dtype.str.encode())
    digest.update(repr(value.shape).encode())
    digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


def unwrap_e150_checkpoint(saved):
    """Recover the exact E150 canonical lower layer without starting a model."""
    if saved.get("schema") != MonotoneRecruitmentTrial.schema:
        raise ValueError("Exact E150 monotone start checkpoint required")
    patch = saved.get("monotone_recruitment_patch")
    adapter = saved.get("adapter")
    if (not isinstance(patch, dict) or not isinstance(adapter, dict)
            or not isinstance(adapter.get("original"), dict)
            or patch.get("source_adapter_identity") != adapter["original"].get("hash")
            or patch.get("source_motor_timescale_patch") is None):
        raise ValueError("Complete E150 monotone/canonical provenance required")
    exact(saved.get("motor_timescale_patch"),
          patch["source_motor_timescale_patch"],
          "E222 original timescale patch under monotone layer")
    source = copy.deepcopy(saved)
    source.pop("monotone_recruitment_patch")
    source["schema"] = MotorTimescaleTrial.schema
    source["adapter"] = copy.deepcopy(adapter["original"])
    return source


def unwrap_native_checkpoint(saved):
    """Expose the old full neural/adapter state and keep native state separate."""
    if saved.get("schema") != SCHEMA:
        raise ValueError("Explicit E222 native trial checkpoint required")
    patch = saved.get("native_grouped_patch")
    if not isinstance(patch, dict) or patch.get("schema") != PATCH_SCHEMA:
        raise ValueError("Complete E222 native trial patch required")
    source = copy.deepcopy(saved)
    source.pop("native_grouped_patch")
    source["schema"] = MotorTimescaleTrial.schema
    network = source.get("network")
    if not isinstance(network, dict) or "native_grouped_response" not in network:
        raise ValueError("E222 native neural checkpoint state missing")
    native = network.pop("native_grouped_response")
    network["model_hash"] = patch["source_timescale_identity"]
    return source, patch, native


def checked_grouped_parameters(base, parameters, prototypes, beta):
    """Freeze the real 391-MN/96-muscle anatomical ownership and E217 geometry."""
    if type(base.network) is not MotorTimescaleNetwork:
        raise TypeError("Exact E150 motor-timescale neural source required")
    if type(base.adapter) is not AnatomicalRecruitmentAdapter:
        raise TypeError("Exact E150 anatomical adapter required")
    net, adapter = base.network, base.adapter
    d = adapter.calibration
    keys = ("motor_ids", "membership", "free_columns", "fixed_columns", "scale",
            "tau", "cap", "inverse_adaptation_tau", "pool", "moment", "rest", "ltm")
    if any(key not in parameters for key in keys) or any(
            key not in prototypes for key in ("centers", "sigma2", "membership")):
        raise ValueError("Complete E217 actual grouped geometry required")
    p = {key: np.array(parameters[key], copy=True) for key in keys}
    c = {key: np.array(prototypes[key], copy=True)
         for key in ("centers", "sigma2", "membership")}
    b = np.array(beta, copy=True)
    free, fixed = p["free_columns"], p["fixed_columns"]
    if (p["motor_ids"].shape != (391,) or p["motor_ids"].dtype != np.int64
            or not np.array_equal(p["motor_ids"], net.motor_ids)
            or p["membership"].shape != (391,) or p["membership"].dtype != np.int64
            or free.dtype != np.int64 or fixed.dtype != np.int64
            or free.shape != (343,) or fixed.shape != (48,)
            or not np.array_equal(np.sort(np.r_[free, fixed]), np.arange(391))
            or not np.array_equal(free, np.sort(free))
            or not np.array_equal(fixed, np.sort(fixed))
            or p["scale"].shape != (391, 3) or p["scale"].dtype != np.float64
            or c["centers"].shape != (391, 64, 9)
            or c["centers"].dtype != np.float64
            or c["sigma2"].shape != (391,) or c["sigma2"].dtype != np.float64
            or b.shape != (96, 64) or b.dtype != np.float64):
        raise ValueError("Exact original 391/343/48/96/64 grouped shape required")
    if (any(not np.isfinite(value).all() for value in (*p.values(), *c.values(), b))
            or np.any(p["scale"] <= 0) or np.any(c["sigma2"] <= 0)
            or np.any((b < 0) | (b > 1))
            or not np.array_equal(p["membership"], c["membership"])):
        raise ValueError("Finite bounded E217/E218 grouped parameters required")
    if (not np.array_equal(p["pool"], d["pool"])
            or not np.array_equal(p["moment"], d["moment"])
            or not np.array_equal(p["rest"], d["rest"])
            or not np.array_equal(p["ltm"], d["ltm"])
            or not np.array_equal(np.argmax(d["pool"], axis=0), p["membership"])
            or not np.array_equal(np.sum(d["pool"] > 0, axis=0), np.ones(391))
            or not np.array_equal(np.flatnonzero(
                np.bincount(p["membership"][free], minlength=96) == 0),
                np.sort(d["ltm"]))
            or np.any(b[np.asarray(d["ltm"], dtype=np.int64)] != 0)):
        raise ValueError("Original anatomical owner/fixed-only rows changed")
    for key, actual in (("tau", net.tau.detach().cpu().numpy()[net.motor_ids]),
                        ("cap", net.cap.detach().cpu().numpy()[net.motor_ids]),
                        ("inverse_adaptation_tau", net._full_inverse32[net.motor_ids])):
        if not np.array_equal(p[key], actual.astype(p[key].dtype)):
            raise ValueError("Actual E150 motor physiology differs: " + key)
    return p, c, b


class CanonicalRate64Adapter(AnatomicalRecruitmentAdapter):
    """Only promote the live float32 motor readout before canonical decoding."""

    def decode(self, rates, *, motor_cut=False, dt=.005):
        raw = np.asarray(rates)
        if raw.shape != (391,) or raw.dtype != np.float32:
            raise ValueError("Original native float32 motor readout required")
        return super().decode(raw.astype(np.float64), motor_cut=motor_cut, dt=dt)


class NativeGroupedTrial(MotorTimescaleTrial):
    """Own one body and replace only E150 monotone decoding plus neural response."""

    schema = SCHEMA

    def __init__(self, base, parameters, prototypes, beta, *,
                 native_network_snapshot=None):
        if type(base) is not MotorTimescaleTrial:
            raise TypeError("Restored exact E150 timescale trial required")
        self.__dict__.update(base.__dict__)
        candidate = None
        try:
            p, c, b = checked_grouped_parameters(base, parameters, prototypes, beta)
            source = base.network.snapshot()
            old = base.network
            free, fixed = p["free_columns"], p["fixed_columns"]
            candidate = NativeGroupedResponseNetwork(source_matrix(old),
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
                membership=p["membership"][free],
                scale=p["scale"][free], centers=c["centers"][free],
                sigma2=c["sigma2"][free], beta=b)
            if candidate.source_timescale_identity != old.identity:
                raise ValueError("Native response source identity changed")
            if native_network_snapshot is None:
                if (base.control_tick != 0 or old.tick != 0
                        or np.any(source["rate"] != 0)
                        or np.any(source["adaptation"] != 0)
                        or np.any(source["drive"] != 0)
                        or np.any(base.adapter.activation != 0)):
                    raise ValueError("Native switch requires exact neutral E150 start")
                patched = copy.deepcopy(source)
                patched["model_hash"] = candidate.identity
                patched["native_grouped_response"] = candidate.snapshot()[
                    "native_grouped_response"]
            else:
                patched = copy.deepcopy(native_network_snapshot)
                unwrapped = copy.deepcopy(patched)
                native = unwrapped.pop("native_grouped_response")
                unwrapped["model_hash"] = old.identity
                exact(source, unwrapped, "E222 exact unwrapped neural checkpoint")
                if native.get("source_timescale_identity") != old.identity:
                    raise ValueError("E222 source timescale provenance changed")
            candidate.restore(patched)
            canonical = CanonicalRate64Adapter(base.adapter.original,
                base.adapter.calibration, base.adapter.groups)
            canonical.restore(base.adapter.snapshot())
            if canonical.identity != base.adapter.identity:
                raise ValueError("Canonical adapter identity changed")
            self.network = candidate
            self.adapter = canonical
            self.tendon_inputs.update(canonical.tendon_inputs)
            self.native_grouped_patch = {
                "schema": PATCH_SCHEMA,
                "source_timescale_identity": old.identity,
                "canonical_adapter_identity": canonical.identity,
                "graph_hash": self.graph.hash,
                "parameter_sha256": {key: _array_bytes(value)
                                     for key, value in p.items()},
                "prototype_sha256": {key: _array_bytes(value)
                                     for key, value in c.items()},
                "beta_sha256": _array_bytes(b),
                "parameters": copy.deepcopy(p),
                "prototypes": copy.deepcopy(c),
                "beta": b.copy()}
        except BaseException as error:
            base.closed = True
            try:
                self.close()
            except BaseException as secondary:
                error.add_note("E222 body cleanup failed: " + repr(secondary))
            raise

    def snapshot(self):
        self.tendon_inputs.update(self.adapter.tendon_inputs)
        return {**super().snapshot(),
                "native_grouped_patch": copy.deepcopy(self.native_grouped_patch)}

    @classmethod
    def from_e150_checkpoint(cls, graph, saved, parameters, prototypes, beta):
        source = unwrap_e150_checkpoint(saved)
        base = MotorTimescaleTrial.from_checkpoint(graph, source)
        trial = None
        try:
            exact(source, base.snapshot(), "E222 unwrapped E150 source")
            trial = cls(base, parameters, prototypes, beta)
            return trial
        except BaseException as error:
            owner = trial if trial is not None else base
            try:
                if not getattr(owner, "closed", False):
                    owner.close()
            except BaseException as secondary:
                error.add_note("E222 source close failed: " + repr(secondary))
            raise

    @classmethod
    def from_checkpoint(cls, graph, saved):
        source, patch, native = unwrap_native_checkpoint(saved)
        base = MotorTimescaleTrial.from_checkpoint(graph, source)
        trial = None
        try:
            if (base.network.identity != patch["source_timescale_identity"]
                    or base.adapter.identity != patch["canonical_adapter_identity"]
                    or graph.hash != patch["graph_hash"]):
                raise ValueError("E222 source graph/network/adapter changed")
            p, c, b = (patch["parameters"], patch["prototypes"], patch["beta"])
            if (patch["parameter_sha256"] != {key: _array_bytes(value)
                    for key, value in p.items()}
                    or patch["prototype_sha256"] != {key: _array_bytes(value)
                    for key, value in c.items()}
                    or patch["beta_sha256"] != _array_bytes(b)):
                raise ValueError("E222 grouped parameter bytes changed")
            trial = cls(base, p, c, b, native_network_snapshot=saved["network"])
            exact(native, trial.snapshot()["network"]["native_grouped_response"],
                  "E222 exact native state")
            exact(saved, trial.snapshot(), "E222 whole native checkpoint")
            return trial
        except BaseException as error:
            owner = trial if trial is not None else base
            try:
                if not getattr(owner, "closed", False):
                    owner.close()
            except BaseException as secondary:
                error.add_note("E222 clone close failed: " + repr(secondary))
            raise
