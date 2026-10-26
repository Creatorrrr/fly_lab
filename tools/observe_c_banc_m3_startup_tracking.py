"""Fixed Original startup tracking capture. Numerical imports follow Admission.

Source proposal only: this file does not grant, reserve, account, or adopt a gait.
"""
# ruff: noqa: BLE001 -- retain primary and detach/publication errors on every exit.
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import time
from pathlib import Path

ENTRY = time.perf_counter()
ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "flylab.m3.startup-tracking"
STARTS = (100, 164, 228, 292)
DRIVE_ENDS = tuple(c + j for c in STARTS for j in range(1, 5))
LEGS = ("lf", "lm", "lh", "rf", "rm", "rh")
RESULT_RESERVE_BYTES = 3 << 20


def leg_joint_indices(adapter_joints, joint_names, np):
    """The deployed adapter's physical indices, checked against named DOF suffixes."""
    indices = np.asarray(adapter_joints)
    require(indices.shape == (6, 7) and indices.dtype.kind in "iu"
            and len(np.unique(indices)) == 42 and np.all((indices >= 0) & (indices < 42)), "actual 6x7 unique adapter joint map")
    require(len(joint_names) == 42 and len(set(joint_names)) == 42, "42 unique physical joint names")
    for leg, row in zip(LEGS, indices, strict=True):
        suffixes = (f"/c_thorax-{leg}_coxa-yaw", f"/c_thorax-{leg}_coxa-pitch", f"/c_thorax-{leg}_coxa-roll",
                    f"/{leg}_coxa-{leg}_trochanterfemur-pitch", f"/{leg}_coxa-{leg}_trochanterfemur-roll",
                    f"/{leg}_trochanterfemur-{leg}_tibia-pitch", f"/{leg}_tibia-{leg}_tarsus1-pitch")
        require(all(joint_names[int(i)].endswith(suffix) for i, suffix in zip(row, suffixes, strict=True)), "actual namespaced leg/DOF binding " + leg)
    return {leg: row.astype(np.int64).tolist() for leg, row in zip(LEGS, indices, strict=True)}


def require(ok, label):
    if not ok:
        raise ValueError(label)


class MethodBinding:
    """Restore both value and inherited/instance ownership, including exceptions."""
    def __init__(self, obj, name):
        self.obj, self.name = obj, name
        self.owned = name in vars(obj)
        self.value = vars(obj).get(name)
        self.call = getattr(obj, name)

    def restore(self):
        if self.owned:
            setattr(self.obj, self.name, self.value)
        elif self.name in vars(self.obj):
            delattr(self.obj, self.name)
        require((self.name in vars(self.obj)) == self.owned, "original method ownership restored")


def copied(value, shape, dtype, np, label):
    # Tensor.cpu is an observation copy, never a neural readout/advance.
    host = value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
    require(host.shape == shape and host.dtype == np.dtype(dtype) and np.isfinite(host).all(),
            "actual finite shape/dtype: " + label)
    return host


def derived_effective(raw, metadata, np):
    """Configured ctrl clamp only; no claim of activation/torque/achievable q."""
    if metadata["control_clamp"]["status"] != "VERIFIED_COMPILED_FLAG":
        return None
    limited = np.asarray(metadata["actuator_ctrllimited"], bool)
    ranges = np.asarray(metadata["actuator_ctrlrange_rad"], np.float64)
    require(raw.shape[-1] == len(limited) and ranges.shape == (len(limited), 2), "clamp order")
    effective = raw.copy()
    if metadata["control_clamp"]["enabled"]:
        effective[..., limited] = np.clip(effective[..., limited], ranges[limited, 0], ranges[limited, 1])
    return effective


def interpolation_contract(continuous, motor_cut, tau):
    interpolate = not motor_cut if continuous else True
    return {"interpolate": interpolate, "interpolate_argument": "explicit not motor_cut" if continuous else "omitted: body API default True",
            "interpolation": "body.step_joint_targets exponential native ctrl trajectory" if tau and interpolate else "held raw ctrl"}


def compiled_metadata(session, np):
    body, network, adapter = session.body, session.network, session.adapter
    names = list(body.joint_names)
    leg_indices = leg_joint_indices(adapter.joints, names, np)
    joint, qpos, act = (np.asarray(getattr(body, name)) for name in ("joint_ids", "qpos_ids", "act_ids"))
    require(len(names) == 42 and len(set(names)) == 42 and all(v.shape == (42,) and v.dtype.kind in "iu" for v in (joint, qpos, act)),
            "ordered compiled 42 joint/qpos/actuator mapping")
    require(np.array_equal(body.m.actuator_trnid[act, 0], joint)
            and np.array_equal(body.m.jnt_qposadr[joint], qpos), "compiled actuator/joint/qpos binding")
    ranges = np.asarray(body.m.actuator_ctrlrange[act], np.float64)
    limited = np.asarray(body.m.actuator_ctrllimited[act], bool)
    require(ranges.shape == (42, 2) and np.isfinite(ranges).all() and np.all(ranges[limited, 0] <= ranges[limited, 1]), "compiled finite ctrl limits")
    disable = int(body.m.opt.disableflags)
    transmission = np.asarray(body.m.actuator_trntype[act])
    gear = np.asarray(body.m.actuator_gear[act], np.float64)
    joint_kind = np.asarray(body.m.jnt_type[joint])
    joint_enum = getattr(getattr(body.mj, "mjtTrn", None), "mjTRN_JOINT", None)
    hinge_enum = getattr(getattr(body.mj, "mjtJoint", None), "mjJNT_HINGE", None)
    require(gear.shape == (42, 6) and np.isfinite(gear).all(), "compiled transmission gear")
    mapping_verified = bool(joint_enum is not None and hinge_enum is not None
                            and np.all(transmission == int(joint_enum)) and np.all(joint_kind == int(hinge_enum))
                            and np.all(gear[:, 0] == 1) and np.all(gear[:, 1:] == 0))
    clamp = {"status": "UNVERIFIED", "enabled": None, "disable_bit": None}
    bits = getattr(body.mj, "mjtDisableBit", None)
    enum = getattr(bits, "mjDSBL_CLAMPCTRL", None) if bits is not None else None
    if enum is not None:
        bit = int(enum)
        require(bit > 0 and bit & (bit - 1) == 0 and disable >= 0, "compiled clamp enum/flags")
        clamp = {"status": "VERIFIED_COMPILED_FLAG", "enabled": not bool(disable & bit), "disable_bit": bit}
    continuous = bool(session.p.continuous_joint_activation)
    tau = float(getattr(body, "joint_target_activation_tau_s", 0.0))
    require(math.isfinite(tau) and tau >= 0, "finite actual activation tau")
    interpolation = interpolation_contract(continuous, bool(session.cuts["motor"]), tau)
    metadata = {
        "joint_names": names, "leg_order": list(LEGS), "leg_joint_indices": leg_indices,
        "joint_ids": joint.tolist(), "qpos_ids": qpos.tolist(), "actuator_ids": act.tolist(),
        "actuator_names": [body.mj.mj_id2name(body.m, body.mj.mjtObj.mjOBJ_ACTUATOR, int(i)) for i in act],
        "joint_range_rad": body.m.jnt_range[joint].tolist(), "joint_limited": body.m.jnt_limited[joint].astype(bool).tolist(),
        "actuator_ctrlrange_rad": ranges.tolist(), "actuator_ctrllimited": limited.tolist(),
        "disableflags": disable, "enableflags": int(body.m.opt.enableflags), "control_clamp": clamp,
        "physics_dt_s": float(body.m.opt.timestep), "control_dt_s": .005, "native_steps_per_control": 50,
        "continuous_joint_activation": continuous, "motor_cut": bool(session.cuts["motor"]),
        "activation_tau_s": tau, **interpolation,
        "effective_interpretation": "derived configured ctrl clamp; activation and servo dynamics are separate",
        "network_class": type(network).__module__ + "." + type(network).__qualname__,
        "adapter_class": type(adapter).__module__ + "." + type(adapter).__qualname__,
        "network_identity": network.identity, "adapter_identity": adapter.identity,
        "joint_types": joint_kind.tolist(), "actuator_gear": gear.tolist(),
        "ctrl_to_q_mapping": {"status": "VERIFIED_UNIT_GEAR_JOINT_HINGE" if mapping_verified else "UNVERIFIED",
                              "joint_transmission_enum": int(joint_enum) if joint_enum is not None else None,
                              "hinge_enum": int(hinge_enum) if hinge_enum is not None else None,
                              "interpretation": "unit gear hinge transmission; servo dynamics remain separate"},
        "actual_q_unit": "rad", "request_unit": "rad", "native_ctrl_unit": "rad" if mapping_verified else "UNVERIFIED actuator input units",
        "native_contact_timing": "post integration; paired with pre_q by explicit native interval",
        "control_row_timing": "boundary t after interval (t-1,t]; boundary0 anchor excluded from applied intervals",
        "command_to_foot": "UNAVAILABLE: no independent full-context kinematic reconstruction",
    }
    for name in ("actuator_trntype", "actuator_dyntype", "actuator_gaintype", "actuator_biastype", "actuator_dynprm", "actuator_gainprm", "actuator_biasprm"):
        value = np.asarray(getattr(body.m, name)[act])
        require(np.isfinite(value).all(), "finite compiled actuator dynamics")
        metadata[name] = value.tolist()
    require(abs(metadata["physics_dt_s"] - .0001) <= 1e-15, "fixed actual native dt")
    return metadata


class TrackingObserver:
    """Read/copy only, installed outside existing LiveObserver/contact wrappers."""
    def __init__(self, session, np, metadata, check, *, n=158706, resource_n=7136, synthetic=False):
        require((n, resource_n) == (158706, 7136) or synthetic is True, "production fixed dimensions")
        self.s, self.np, self.metadata, self.check = session, np, metadata, check
        self.n, self.resource_n, self.synthetic = n, resource_n, synthetic
        self.started = self.returned = self.neural_calls = 0
        self.undo = []
        self.a = {
            "boundary_control": np.zeros(301, np.int64), "boundary_network": np.zeros(301, np.int64),
            "boundary_time_s": np.zeros(301, np.float64), "boundary_valid": np.zeros(301, bool),
            "requested_target_rad": np.zeros((301, 42), np.float64), "actual_q_rad": np.zeros((301, 42), np.float64),
            "native_control_end": np.zeros(15000, np.int64), "native_substep": np.zeros(15000, np.int64),
            "native_pre_time_s": np.zeros(15000, np.float64), "native_post_time_s": np.zeros(15000, np.float64),
            "native_raw_ctrl_rad": np.zeros((15000, 42), np.float64),
            "native_effective_ctrl_rad": np.zeros((15000, 42), np.float64),
            "native_pre_q_rad": np.zeros((15000, 42), np.float64), "native_post_q_rad": np.zeros((15000, 42), np.float64),
            "native_started": np.zeros(15000, bool), "native_returned": np.zeros(15000, bool),
            "native_post_observed": np.zeros(15000, bool), "native_effective_verified": np.zeros(15000, bool),
        }
        self.context = {
            "pre_control": np.asarray(STARTS, np.int64), "pre_network": np.zeros(4, np.int64),
            "pre_time_s": np.zeros(4, np.float64), "pre_valid": np.zeros(4, bool),
            "rate": np.zeros((4, n), np.float32), "adaptation": np.zeros((4, n), np.float32),
            "bank": np.zeros((4, 391, 3), np.float64), "original_activation": np.zeros((4, 96), np.float64),
            "resource": np.zeros((4, resource_n), np.float64), "u_zero": np.zeros(4, bool), "resource_one": np.zeros(4, bool),
            "drive_control_end": np.asarray(DRIVE_ENDS, np.int64), "drive_network_pre": np.zeros(16, np.int64),
            "drive_time_pre_s": np.zeros(16, np.float64), "drive_valid": np.zeros(16, bool),
            "drive": np.zeros((16, n), np.float32),
        }
        network, adapter = session.network, session.adapter
        leaves = (("source_ids", network._resource_sources, (resource_n,), "int64"),
                  ("utilization", network._resource_u, (resource_n,), "float64"),
                  ("recovery_tau_s", network._resource_tau, (resource_n,), "float64"),
                  ("motor_ids", adapter.motor_ids, (391,), "int64"),
                  ("output_mask", network.output_mask, (n,), "float32"))
        for name, value, shape, dtype in leaves:
            self.context[name] = copied(value, shape, dtype, np, name).copy()
        require(np.all(self.context["utilization"] == 0) and np.all(self.context["recovery_tau_s"] > 0), "actual neutral utilization and tau")
        require(len(np.unique(self.context["source_ids"])) == resource_n
                and np.all((self.context["source_ids"] >= 0) & (self.context["source_ids"] < n))
                and len(np.unique(self.context["motor_ids"])) == 391
                and np.all((self.context["motor_ids"] >= 0) & (self.context["motor_ids"] < n))
                and np.isin(self.context["output_mask"], (0, 1)).all(), "actual ordered source/motor/output mask")

    def boundary(self, row, boundary):
        np, s = self.np, self.s
        c = int(s.control_tick)
        require(0 <= c <= 300 and not self.a["boundary_valid"][c], "one control boundary")
        require(int(s.network.tick) == c * 20 and abs(s.body.physics_time() - c * .005) <= 2e-8, "actual boundary clocks")
        requested = copied(np.asarray(row["joint_targets_rad"], np.float64), (42,), "float64", np, "requested target")
        actual = copied(np.asarray(row["joint_angles_rad"], np.float64), (42,), "float64", np, "actual measured q")
        require(np.array_equal(requested, boundary["command"]), "full row request equals existing raw command")
        self.a["requested_target_rad"][c] = requested
        self.a["actual_q_rad"][c] = actual
        self.a["boundary_control"][c], self.a["boundary_network"][c] = c, int(s.network.tick)
        self.a["boundary_time_s"][c] = s.body.physics_time()
        self.a["boundary_valid"][c] = True

    def install(self):
        s, np = self.s, self.np
        neural, physics = MethodBinding(s.network, "advance_readout"), MethodBinding(s.body, "_step_physics")

        def advance(drive, ids, steps):
            self.check("science")
            c = int(s.control_tick)
            require(0 <= c < 300 and steps == 20 and int(s.network.tick) == 20 * c
                    and abs(s.body.physics_time() - .005 * c) <= 2e-8, "neural entry c generates c+1")
            require(np.array_equal(np.asarray(ids), self.context["motor_ids"]), "actual ordered readout motor IDs")
            if c in STARTS:
                j = STARTS.index(c)
                require(not self.context["pre_valid"][j], "one actual pre-state per window")
                for name, value, shape, dtype in (
                    ("rate", s.network.rate, (self.n,), "float32"),
                    ("adaptation", s.network.adaptation, (self.n,), "float32"),
                    ("bank", s.adapter.bank, (391, 3), "float64"),
                    ("original_activation", s.adapter.original.activation, (96,), "float64"),
                    ("resource", s.network.resource, (self.resource_n,), "float64"),
                ):
                    np.copyto(self.context[name][j], copied(value, shape, dtype, np, name))
                u = copied(s.network._resource_u, (self.resource_n,), "float64", np, "actual u")
                require(np.array_equal(u, self.context["utilization"]), "utilization unchanged")
                self.context["u_zero"][j] = bool(np.all(u == 0))
                self.context["resource_one"][j] = bool(np.all(self.context["resource"][j] == 1))
                require(self.context["u_zero"][j] and self.context["resource_one"][j], "actual u0/resource1 pre-state proof")
                self.context["pre_network"][j], self.context["pre_time_s"][j] = int(s.network.tick), s.body.physics_time()
                self.context["pre_valid"][j] = True
            end = c + 1
            if end in DRIVE_ENDS:
                j = DRIVE_ENDS.index(end)
                require(not self.context["drive_valid"][j], "one actual finalized drive")
                np.copyto(self.context["drive"][j], copied(drive, (self.n,), "float32", np, "actual finalized drive"))
                self.context["drive_network_pre"][j], self.context["drive_time_pre_s"][j] = int(s.network.tick), s.body.physics_time()
                self.context["drive_valid"][j] = True
            self.neural_calls += 1
            return neural.call(drive, ids, steps)  # exactly one call, original args/return

        def step(*args, **kwargs):
            self.check("science")
            slot = self.started
            require(slot < 15000, "bounded actual native capture")
            c = int(s.control_tick)
            sub = slot - 50 * c
            require(0 <= c < 300 and 0 <= sub < 50, "native interval/substep continuity")
            pre = s.body.physics_time()
            require(abs(pre - (50 * c + sub) * .0001) <= 2e-8, "native pre clock")
            raw = copied(s.body.d.ctrl[s.body.act_ids], (42,), "float64", np, "actual pre-step raw ctrl")
            q = copied(s.body.d.qpos[s.body.qpos_ids], (42,), "float64", np, "actual pre-step q")
            self.a["native_control_end"][slot], self.a["native_substep"][slot] = c + 1, sub
            self.a["native_pre_time_s"][slot] = pre
            self.a["native_raw_ctrl_rad"][slot], self.a["native_pre_q_rad"][slot] = raw, q
            effective = derived_effective(raw, self.metadata, np)
            if effective is not None:
                self.a["native_effective_ctrl_rad"][slot] = effective
                self.a["native_effective_verified"][slot] = True
            self.a["native_started"][slot] = True
            self.started += 1
            primary = None
            try:
                answer = physics.call(*args, **kwargs)  # existing LiveObserver/contact path
                self.a["native_returned"][slot] = True
                self.returned += 1
                return answer
            except Exception as error:
                primary = error
                raise
            finally:
                try:
                    post = s.body.physics_time()
                    self.a["native_post_time_s"][slot] = post
                    np.copyto(self.a["native_post_q_rad"][slot], copied(s.body.d.qpos[s.body.qpos_ids], (42,), "float64", np, "actual post-step q"))
                    self.a["native_post_observed"][slot] = True
                    if self.a["native_returned"][slot]:
                        require(abs(post - pre - .0001) <= 2e-8, "native post integration clock")
                except Exception as error:
                    if primary is not None:
                        primary.add_note("tracking post observation: " + str(error))
                    else:
                        raise

        try:
            s.network.advance_readout = advance
            self.undo.append(neural)
            s.body._step_physics = step
            self.undo.append(physics)
        except Exception:
            self.close()
            raise
        return self

    def close(self):
        errors = []
        for binding in reversed(self.undo):
            try:
                binding.restore()
            except Exception as error:
                errors.append(str(error))
        self.undo = []
        require(not errors, "tracking detach: " + repr(errors))

    def payload(self):
        # Missing cells are zero+invalid, never uninitialized memory or invented samples.
        count = int(self.a["boundary_valid"].sum())
        control = {k: v[:count].copy() for k, v in self.a.items() if k.startswith("boundary_") or k in ("requested_target_rad", "actual_q_rad")}
        native = {k: v[:self.started].copy() for k, v in self.a.items() if k.startswith("native_")}
        return {**control, **native}

    def summary(self):
        return {"synthetic": self.synthetic, "neural_calls_requested": self.neural_calls,
                "native_started": self.started, "native_returned": self.returned,
                "boundary_rows": int(self.a["boundary_valid"].sum()),
                "pre_complete_bitmap": self.context["pre_valid"].tolist(), "drive_complete_bitmap": self.context["drive_valid"].tolist(),
                "capture_complete": bool(self.a["boundary_valid"].all() and self.a["native_returned"].all()
                                         and self.a["native_post_observed"].all() and self.context["pre_valid"].all()
                                         and self.context["drive_valid"].all() and self.neural_calls == 300),
                "extra_neural_or_physics_calls": 0, "offline_replay": 0, "gradient": 0,
                "host_copy_is_unproven_nonperturbing": True}


def continuous_arm(data, arm, identity, record, timer, publisher, watch, *, tracker_factory=None, artifact_hook=None):
    """Small copied J skeleton; one owned actual instance 0→99→300."""
    import numpy as np

    from tools import verify_c_banc_m3_candidate_body_handoff as body
    from tools import verify_c_banc_m3_common_descent_body as b
    from tools import verify_c_banc_m3_startup_baseline as neutral

    require(arm in ("neutral", "candidate"), "fixed initializer")
    session = live = tracker = factory = None
    original_bindings, rows, controls, samples = [], [], [], []
    counters, construction, artifacts = b.new_counters(), {}, {}
    result = {"schema": SCHEMA + ".body-record.v1", "arm": arm, "candidate": identity,
              "run_valid": False, "capture_complete": False, "classification": "technical_invalid",
              "first_error": None, "secondary_errors": [], "candidate_adopted": False, "F1_F8": "INCOMPLETE",
              "biology": "UNPROVEN", "M2_UI": "PRESERVED", "startup": None, "metrics": None, "gates": None}
    stage, reason, metadata, tracking_metadata = "construct", "duration", None, None
    try:
        factory = b.OriginalFactory()
        timer.fence("pre_array")
        session = factory.construct(data, construction, lambda: timer.check("science"))
        require(not getattr(session, "closed", False), "one live owned instance")
        if arm == "neutral":
            data.pop("selected_weights", None)
            proof = neutral.initial_neutral(session, data, timer, b=b, factory=factory)
        else:
            proof = body.initial_body(session, data, identity, record, timer, b=b, factory=factory)
        result["transfer"] = proof
        artifacts["transfer"] = publisher.json("transfer.json", proof)
        metadata, tracking_metadata = b.trace_metadata(session), compiled_metadata(session, np)
        row, boundary = b.capture(session)
        tracking_metadata["initial_native_contact_load_bw"] = row["feet"]["normal_load_bw"]
        rows.append(row); controls.append(boundary); counters["completed_rows"] = 1
        # Save pre-Live ownership too: inherited method restoration is otherwise lost.
        original_bindings = [MethodBinding(session.network, "advance_readout"), MethodBinding(session.body, "_step_physics")]
        live = b.LiveObserver(session, counters, samples, timely=lambda: timer.check("science"),
                              install_contact=factory.install_contact, synchronize=factory.synchronize).install()
        tracker = (tracker_factory or TrackingObserver)(session, np, tracking_metadata, timer.check)
        tracker.boundary(row, boundary)
        tracker.install()
        objects = tuple(id(x) for x in (session, session.network, session.body, session.adapter, live, tracker))
        for control in range(1, 301):
            stage = "control-" + str(control)
            timer.check("science")
            counters["controls_requested"] += 1
            try:
                session.advance(1, capture=False)
            except Exception:
                counters["advance_unreturned_unknown"] = True
                raise
            counters["controls_host_returned"] += 1
            factory.synchronize()
            counters["physics_device_confirmed"] = counters["physics_host_returned"]
            row, boundary = b.capture(session)
            tracker.boundary(row, boundary)
            rows.append(row); controls.append(boundary)
            require(objects == tuple(id(x) for x in (session, session.network, session.body, session.adapter, live, tracker)), "same actual instances")
            require(session.control_tick == control and session.network.tick == control * 20
                    and len(samples) == control * 50 and tracker.started == tracker.returned == control * 50
                    and abs(session.body.physics_time() - control * .005) <= 2e-8
                    and session._cpg_digest() == proof["cpg_hash"], "actual closed-loop continuity")
            counters["controls_completed"] += 1
            counters["completed_rows"], counters["observer_samples"] = len(controls), len(samples)
            body.monitor_check(watch.snapshot(), timer.limits)
            if control == 99:
                body.verify_work(counters, construction, 99)
                prefix = b.native_arrays(samples)
                require(b.validate_native(prefix) == 4950 and np.all(prefix["upright"] >= .9)
                        and not prefix["nonfoot_contact"].any() and not any(r["fault"] for r in rows), "healthy known prefix")
                result["startup"] = b.startup99(np.stack([x["command"] for x in controls]),
                                                np.stack([x["tendon"] for x in controls]), **data["startup"])
                publisher.json("prefix99.json", {"rows": len(controls), "samples": len(samples), "counters": counters, "startup": result["startup"]})
                if not result["startup"]["passed"]:
                    reason = "startup_rejected"
                    break
        body.verify_work(counters, construction, counters["controls_completed"], full=reason == "duration")
        require(not counters.get("advance_unreturned_unknown", False), "known returned work")
        result.update(run_valid=True, reason=reason, classification="scientific_rejected")
    except Exception as error:
        body.fail(result, error, stage)
    finally:
        if factory and session:
            try:
                factory.synchronize()
                counters["physics_device_confirmed"] = counters["physics_host_returned"]
            except Exception as error:
                body.fail(result, error, "final-device-confirmation")
        # Reverse exact install order even on partial health-stop / failed install.
        for label, callback in (("tracking-detach", tracker.close if tracker else None),
                                ("live-detach", live.close if live else None)):
            if callback:
                try:
                    callback()
                except Exception as error:
                    body.fail(result, error, label)
        for binding in reversed(original_bindings):
            try:
                binding.restore()
            except Exception as error:
                body.fail(result, error, "original-method-ownership")
        body.finish_owned(None, session, result, timer)
        result["all_owned_ended"] = bool(session is not None and getattr(session, "closed", False))
    result.update(counters=counters, construction=construction)
    if tracker is not None:
        result["tracking"] = tracker.summary()
        result["context_array_sha256"] = {key: hashlib.sha256(value.tobytes(order="C")).hexdigest() for key, value in tracker.context.items()}
        result["capture_complete"] = bool(result["run_valid"] and result["all_owned_ended"] and tracker.summary()["capture_complete"])
    try:
        if controls:
            raw, native = b.control_arrays(controls, session, data), b.native_arrays(samples)
            artifacts["control"] = publisher.arrays("control.npz", raw)
            artifacts["native"] = publisher.arrays("native.npz", native)
            artifacts["trace"] = publisher.json("trace.json", b.lean_trace(rows, metadata))
            if tracker is not None:
                artifacts["tracking"] = publisher.arrays("tracking.npz", tracker.payload())
                artifacts["context"] = publisher.arrays("context.npz", tracker.context)
                result["tracking_metadata"] = tracking_metadata
            if artifact_hook is not None:
                artifact_hook(tracker, result, artifacts, publisher)
            if result["run_valid"] and reason == "duration":
                result["metrics"] = b.evaluate(publisher.folder, raw, native, data, result["startup"])
                result["gates"], _passed = body.screen(result["metrics"], result["startup"])
                result["classification"] = "diagnostic_only_rejected_candidate"
            require(session._cpg_digest() == result["transfer"]["cpg_hash"], "CPG unchanged after close")
            for item in data["ledger"]:
                require(b.e.digest(timer.root / item["path"], timer.check) == item["expected_sha256"], "Original input posthash")
            artifacts["ledger"] = publisher.json("selected-ledger.json", body.normalize(data["ledger"]))
    except Exception as error:
        result["capture_complete"] = False
        body.fail(result, error, "partial-or-full-publication")
    if watch:
        try:
            result["sampler"] = watch.close()
            body.monitor_check(result["sampler"], timer.limits, closed=True)
        except Exception as error:
            result["capture_complete"] = False
            body.fail(result, error, "sampler-close")
    result["output_usage_before_body_report"] = publisher.usage()
    artifacts["body_report"] = publisher.json("body-report.json", body.normalize(result))
    manifest = {"schema": SCHEMA + ".capture-manifest.v1", "role": timer.role, "request": timer.request_spec,
                "roster": timer.request["roster"], "historical_manifest": timer.manifest_spec,
                "artifacts": artifacts, "run_valid": result["run_valid"], "capture_complete": result["capture_complete"],
                "candidate_adopted": False, "F1_F8": "INCOMPLETE", "output_usage_before_manifest": publisher.usage()}
    artifacts["manifest"] = publisher.manifest("tracking-manifest.json", manifest)
    result["artifacts"] = artifacts
    return result


def bounded_publisher(admission):
    from tools import verify_c_banc_m3_startup_ray as ray

    class Publisher(ray.Publisher):
        def __init__(self, admitted):
            require(admitted.role in ("tracking_neutral", "tracking_candidate", "tracking_diagnosis", "geometry_capture", "geometry_analyze"), "finite observation bounded publisher")
            self.result_cap = admitted.limits["output_cap_bytes"] - RESULT_RESERVE_BYTES
            self.serialization_peak_bytes = self.serialization_total_bytes = 0
            super().__init__(admitted)

        def usage(self):
            return {"durable_result_bytes_so_far": self.total, "result_cap_bytes": self.result_cap,
                    "reserved_supervision_bytes": RESULT_RESERVE_BYTES,
                    "serialization_payload_peak_bytes": self.serialization_peak_bytes,
                    "serialization_buffer_plus_getvalue_upper_peak_bytes": 2 * self.serialization_peak_bytes,
                    "serialization_payload_total_bytes": self.serialization_total_bytes,
                    "host_cap_bytes": admission.limits["host_bytes"], "logical_combined_cap_bytes": admission.limits["output_cap_bytes"]}

        def write(self, name, raw):
            require(self.total + len(raw) <= self.result_cap, "result cap with three MiB supervision reserve")
            return super().write(name, raw)

        def arrays(self, name, arrays):
            import numpy as np
            estimate = sum(v.nbytes for v in arrays.values()) + 1024 * len(arrays)
            require(estimate + self.total <= self.result_cap and 2 * estimate <= admission.limits["host_bytes"], "temporary serialization and reserved durable cap before allocation")
            buffer = io.BytesIO()
            np.savez(buffer, **arrays)
            require(buffer.tell() <= estimate, "bounded serialization bytes")
            self.serialization_peak_bytes = max(self.serialization_peak_bytes, buffer.tell())
            self.serialization_total_bytes += buffer.tell()
            return self.write(name, buffer.getvalue())

        def manifest(self, name, value):
            from tools import verify_c_banc_m3_candidate_body_handoff as body
            # Fixed-point byte count, without a self hash or a second publication.
            value["durable_result_bytes_including_manifest"] = self.total
            for _ in range(8):
                raw = json.dumps(body.normalize(value), ensure_ascii=True, allow_nan=False, sort_keys=True).encode("utf8")
                expected = self.total + len(raw)
                if value["durable_result_bytes_including_manifest"] == expected:
                    return self.write(name, raw)
                value["durable_result_bytes_including_manifest"] = expected
            raise ValueError("stable finite manifest byte count")

    return Publisher(admission)


def worker(admission):
    require(admission.role in ("tracking_neutral", "tracking_candidate") and admission.stage == "pre_array", "admitted actual arm")
    admission.fence("pre_array")
    from tools import verify_c_banc_m3_candidate_body_handoff as body
    from tools import verify_c_banc_m3_common_descent_body as b
    from tools import verify_c_banc_m3_reference_parameter_direction as ref
    from tools import verify_c_banc_m3_startup_ray as ray
    publisher = bounded_publisher(admission)
    watch = body.gpu_watch({"limits": admission.limits})  # precedes first array/body load
    try:
        selected = ray.json_load(ray.leaf(admission.roster["selected_candidate"], admission.root))
        audit = ray.json_load(ray.leaf(admission.roster["candidate_audit"], admission.root))
        fixed = admission.roster["fixed_candidate"]
        require(audit["raw_valid"] is True and audit["selected"] == selected["index"] == fixed["index"]
                and selected["alpha"] == fixed["alpha"] == .0001, "fixed historically rejected audited candidate")
        ray.exact_value(audit["selected_candidate"], selected, "historical audited identity")
        ray.exact_value(selected["manifest"], admission.manifest_spec, "historical J manifest preserved")
        ray.exact_value(selected["grant"], admission.request["grant"], "same J grant")
        ray.exact_value(selected["arrays"], admission.roster["candidate_arrays"], "fixed candidate archive")
        arrays = ray.saved_arrays(selected["arrays"], admission)
        require(ray.array_digest(arrays["theta_actual"]) == selected["theta_sha256"] == fixed["theta_sha256"]
                and ray.array_digest(arrays["selected_weights_candidate"]) == selected["selected_sha256"] == fixed["selected_sha256"], "actual fixed weights/theta")
        arm = "neutral" if admission.role == "tracking_neutral" else "candidate"
        data = ray.arm_data(arrays, admission.inputs["body_inventory"]["inventory"], admission, arm, b=b)
        if arm == "neutral":
            admission.fence("pre_array")
            data["reference"] = ref.native().load_selected(admission.root, data["inventory"]["selected_archives"]["reference"], data["ledger"], timely=admission.check, generated=False)
        result = continuous_arm(data, arm, selected, {"palette": selected["palette"]}, admission, publisher, watch)
        require(result["run_valid"] is True, "technical-invalid actual capture cannot exit success")
        return result
    except Exception as error:
        try:
            publisher.json("failure.json", {"schema": SCHEMA + ".partial-failure.v1", "error_type": type(error).__name__,
                                           "error": str(error), "run_valid": False, "capture_complete": False, "candidate_adopted": False})
        except Exception as secondary:
            error.add_note("failure publication: " + str(secondary))
        raise
    finally:
        if watch.snapshot()["state"] != "CLOSED":
            closed = watch.close()
            body.monitor_check(closed, admission.limits, closed=True)
            publisher.json("sampler.json", closed)


def main(argv=None):
    from tools.banc_m3_startup_tracking_admission import Admission
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    parser.add_argument("--pre-spawn", action="store_true")
    args = parser.parse_args(argv)
    entry = ENTRY
    if not args.pre_spawn:
        entry = float(json.loads(os.environ["FLYLAB_DIRECT_SCIENCE_OWNER"])["entry_counter_s"])
    admission = Admission(args.request, stage="pre_spawn" if args.pre_spawn else "pre_array", entry=entry)
    if args.pre_spawn:
        print(json.dumps({"pre_spawn": True, "role": admission.role}, sort_keys=True))
        return 0
    result = worker(admission)
    print(json.dumps({"role": admission.role, "run_valid": result["run_valid"], "capture_complete": result["capture_complete"], "candidate_adopted": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
