"""Neutral local geometry capture; numerical/body imports follow Admission."""
# ruff: noqa: BLE001 -- retain primary/partial capture and cleanup failures.
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time

ENTRY = time.perf_counter()
SCHEMA = "flylab.m3.local-foot-geometry"
CONTROLS = (100, 101, 164, 165, 228, 229, 292, 293)
ANCHORS = (100, 164, 228, 292)
LEGS = ("lf", "lm", "lh", "rf", "rm", "rh")
FK_FIELDS = ("qpos0", "body_pos", "body_quat", "body_parentid", "body_mocapid", "body_jntadr", "body_jntnum", "body_dofadr", "body_dofnum",
             "jnt_pos", "jnt_axis", "jnt_type", "jnt_bodyid", "jnt_qposadr", "jnt_dofadr", "dof_bodyid", "dof_jntid", "dof_parentid")
DIMENSIONS = ("nq", "nv", "njnt", "nbody", "nmocap")
DYNAMIC_FIELDS = ("actuator_gainprm", "actuator_biasprm", "actuator_dynprm")
CONTROL_FIELDS = ("actuator_ctrllimited", "actuator_ctrlrange", "actuator_actlimited", "actuator_actrange",
                  "actuator_trntype", "actuator_trnid", "actuator_gear")


def require(ok, label):
    if not ok:
        raise ValueError(label)


def array_identity(value):
    return {"shape": list(value.shape), "dtype": value.dtype.str, "nbytes": int(value.nbytes),
            "sha256": hashlib.sha256(value.tobytes(order="C")).hexdigest()}


def fk_identity(model, np):
    return {"dimensions": {k: int(getattr(model, k)) for k in DIMENSIONS},
            "fields": {k: array_identity(np.asarray(getattr(model, k))) for k in FK_FIELDS}}


def control_identity(model, active_ids, np):
    """Small compiled clamp/transmission identity; dynamic gain/bias are separate."""
    ids = np.asarray(active_ids, np.int64)
    require(ids.shape == (42,) and len(np.unique(ids)) == 42 and np.all((ids >= 0) & (ids < model.nu)), "active42 actuator identity")
    return {"active_actuator_ids": ids.tolist(), "nu": int(model.nu), "disableflags": int(model.opt.disableflags),
            "enableflags": int(model.opt.enableflags),
            "fields": {key: array_identity(np.asarray(getattr(model, key)[ids])) for key in CONTROL_FIELDS}}


def model_dimensions(model):
    values = {key: int(getattr(model, key)) for key in (*DIMENSIONS, "nu")}
    require(0 < values["nq"] <= 4096 and 0 < values["nv"] <= 4096 and 0 < values["njnt"] <= 4096
            and 0 < values["nbody"] <= 20000 and 0 <= values["nmocap"] <= 1024 and 0 < values["nu"] <= 4096, "finite compiled model dimensions")
    return values


def pose_fields(dim):
    nq, nv, nm = dim["nq"], dim["nv"], dim["nmocap"]
    return {"control": ((8,), "int64"), "network": ((8,), "int64"), "model_time_absolute_s": ((8,), "float64"),
            "physics_time_normalized_s": ((8,), "float64"), "valid": ((8,), "bool"),
            "qpos": ((8, nq), "float64"), "qvel": ((8, nv), "float64"), "mocap_pos": ((8, nm, 3), "float64"),
            "mocap_quat": ((8, nm, 4), "float64"), "tip_world_mm": ((8, 6, 3), "float64"), "thorax_points_mm": ((8, 4, 3), "float64"),
            "normal_load_bw": ((8, 6), "float64"), "recorded_load_bearing": ((8, 6), "bool"),
            "anchor_control": ((4,), "int64"), "jacobian_valid": ((4,), "bool"), "jacobian_full_nv": ((4, 6, 3, nv), "float64"),
            "interval_raw_ctrl_rad": ((4, 50, 42), "float64"), "interval_valid": ((4,), "bool")}


def empty_model_work():
    return {"size_requested": 0, "size_returned": 0, "actual_size_type": None, "actual_size_bytes": None,
            "buffer_requested": 0, "buffer_returned": 0, "save_requested": 0, "save_returned": 0,
            "copy_requested": 0, "copy_returned": 0, "publication_requested": 0, "publication_returned": 0}


def save_model_once(mj, model, np, publisher, payload_bytes, limits, *, work=None):
    work = empty_model_work() if work is None else work
    require(work["size_requested"] == 0, "one model-size preflight")
    work["size_requested"] += 1
    size = mj.mj_sizeModel(model)
    work["size_returned"] += 1
    work["actual_size_type"] = type(size).__module__ + "." + type(size).__qualname__
    typed_size = type(size) is int and size.bit_length() <= 64
    work["actual_size_bytes"] = size if typed_size else None
    work["model_cap_bytes"], work["geometry_payload_bytes"] = 96 << 20, payload_bytes
    work["preflight"] = publisher.json("model-size-preflight.json", dict(work))
    require(typed_size and 0 < size <= (96 << 20),
            "typed bounded mj_sizeModel before allocation: type=" + work["actual_size_type"]
            + "; size=" + (str(size) if typed_size else "<not a 64-bit integer>"))
    # Future unchanged common archives/trace reserve plus actual geometry/MJB.
    require(publisher.total + size + payload_bytes + (48 << 20) <= publisher.result_cap
            and 2 * size + 2 * payload_bytes <= limits["host_bytes"], "model buffer/copy/durable/common reserve within caps")
    work["buffer_requested"] += 1
    buffer = np.empty(size, np.uint8)
    work["buffer_returned"] += 1
    work["save_requested"] += 1
    mj.mj_saveModel(model, filename=None, buffer=buffer)
    work["save_returned"] += 1
    work["copy_requested"] += 1
    raw = buffer.tobytes()
    work["copy_returned"] += 1
    require(len(raw) == size, "exact one model serialization")
    work["publication_requested"] += 1
    spec = publisher.write("model.mjb", raw)
    work["publication_returned"] += 1
    work.update(mj_sizeModel_bytes=size, buffer_plus_copy_peak_bytes=2 * size,
                durable_model_bytes=spec["bytes"], common_result_reserve_bytes=48 << 20)
    return spec, work


def geometry_factory(publisher, timer, *, serialization=None):
    from tools.observe_c_banc_m3_startup_tracking import MethodBinding, TrackingObserver
    serialization = empty_model_work() if serialization is None else serialization

    class GeometryTracker(TrackingObserver):
        def __init__(self, session, np, metadata, check):
            super().__init__(session, np, metadata, check)
            body, model = session.body, session.body.m
            dim = model_dimensions(model)
            self.g = {k: np.zeros(shape, dtype) for k, (shape, dtype) in pose_fields(dim).items()}
            self.forward_requested = self.forward_returned = self.jac_requested = self.jac_returned = 0
            self.forward_binding = None
            self.last_forward_time = None
            self.fk = fk_identity(model, np)
            self.control_id = control_identity(model, metadata["actuator_ids"], np)
            self.control_boundary_verified, self.control_end_verified = {}, False
            t0 = float(body.t0)
            require(np.isfinite(t0) and abs(float(body.d.time) - body.physics_time() - t0) <= 2e-8, "actual absolute/normalized t0 relation")
            self.geometry_metadata = {"schema": SCHEMA + ".metadata.v1", "dimensions": dim, "fk_identity": self.fk,
                "control_identity": self.control_id, "control_boundary_verified": self.control_boundary_verified,
                "control_end_verified": False, "physics_t0_absolute_s": t0,
                "clock_channels": {"model_time_absolute_s": "actual data.time; restored unchanged", "physics_time_normalized_s": "actual body.physics_time() = data.time - t0"},
                "mujoco_version": body.mj.__version__, "integrator": int(model.opt.integrator), "timestep": float(model.opt.timestep),
                "active_joint_ids": list(metadata["joint_ids"]), "active_qpos_ids": list(metadata["qpos_ids"]),
                "active_dof_ids": model.jnt_dofadr[body.joint_ids].astype(np.int64).tolist(), "active_joint_names": metadata["joint_names"],
                "tip_ids": np.asarray(body._tip_ids, np.int64).tolist(), "thorax_id": int(body.thorax),
                "tip_names": [body.mj.mj_id2name(model, body.mj.mjtObj.mjOBJ_BODY, int(i)) for i in body._tip_ids],
                "thorax_name": body.mj.mj_id2name(model, body.mj.mjtObj.mjOBJ_BODY, int(body.thorax)),
                "control_mapping": metadata, "boundary_source": "after existing body._forward_physics return; same post-forward full state",
                "tip_definition": "tarsus5 body origins; not contact pads", "length_unit": "mm", "joint_unit": "rad",
                "AP_axis": "start thorax anterior axis", "vertical_axis": "world +Z", "dynamic_model_hashes": {}}
            require(all(name.endswith('/' + leg + '_tarsus5') for leg, name in zip(LEGS, self.geometry_metadata["tip_names"], strict=True))
                    and self.geometry_metadata["thorax_name"].endswith('/c_thorax')
                    and np.all(model.jnt_type[body.joint_ids] == int(body.mj.mjtJoint.mjJNT_HINGE))
                    and len(set(self.geometry_metadata["active_dof_ids"])) == 42, "compiled ordered tips/thorax/hinge mapping")
            self.model_spec, self.serialization = save_model_once(body.mj, model, np, publisher, sum(v.nbytes for v in self.g.values()), timer.limits,
                                                                 work=serialization)
            self.jp, self.jr = np.zeros((3, dim["nv"]), np.float64), np.zeros((3, dim["nv"]), np.float64)

        def install(self):
            super().install()
            self.forward_binding = MethodBinding(self.s.body, "_forward_physics")
            original = self.forward_binding.call

            def forward(*args, **kwargs):
                self.check("science")
                self.forward_requested += 1
                value = original(*args, **kwargs)
                self.forward_returned += 1
                self.last_forward_time = float(self.s.body.d.time)
                return value

            try:
                self.s.body._forward_physics = forward
            except Exception:
                self.close()
                raise
            return self

        def boundary(self, row, boundary):
            super().boundary(row, boundary)
            c, np, body = int(self.s.control_tick), self.np, self.s.body
            if c not in CONTROLS:
                return
            require(self.forward_requested == self.forward_returned == c and self.last_forward_time == float(body.d.time), "exact post-existing-forward boundary")
            require(fk_identity(body.m, np) == self.fk, "FK fields remain byte invariant")
            require(control_identity(body.m, self.metadata["actuator_ids"], np) == self.control_id, "boundary compiled control mapping invariant")
            self.control_boundary_verified[str(c)] = True
            i, d = CONTROLS.index(c), body.d
            require(not self.g["valid"][i], "one geometry observation per boundary")
            for key in ("qpos", "qvel", "mocap_pos", "mocap_quat"):
                value = np.asarray(getattr(d, key))
                require(value.shape == self.g[key][i].shape and value.dtype == np.float64 and np.isfinite(value).all(), "full actual pose " + key)
                np.copyto(self.g[key][i], value)
            absolute, normalized = float(d.time), float(body.physics_time())
            require(abs(normalized - .005 * c) <= 2e-8
                    and abs(absolute - normalized - self.geometry_metadata["physics_t0_absolute_s"]) <= 2e-8, "observed normalized control clock and fixed t0 relation")
            self.g["control"][i], self.g["network"][i] = c, int(self.s.network.tick)
            self.g["model_time_absolute_s"][i], self.g["physics_time_normalized_s"][i] = absolute, normalized
            self.g["tip_world_mm"][i] = d.xpos[body._tip_ids]
            origin, axes = d.xpos[body.thorax], d.xmat[body.thorax].reshape(3, 3)
            self.g["thorax_points_mm"][i, 0] = origin
            self.g["thorax_points_mm"][i, 1:] = origin + axes.T
            self.g["normal_load_bw"][i] = row["feet"]["normal_load_bw"]
            self.g["recorded_load_bearing"][i] = row["feet"]["load_bearing"]
            self.geometry_metadata["dynamic_model_hashes"][str(c)] = {key: array_identity(np.asarray(getattr(body.m, key))) for key in DYNAMIC_FIELDS}
            self.g["valid"][i] = True
            if c in ANCHORS:
                j = ANCHORS.index(c)
                self.g["anchor_control"][j] = c
                for leg, tip in enumerate(body._tip_ids):
                    self.check("science")
                    self.jac_requested += 1
                    body.mj.mj_jacBody(body.m, d, self.jp, self.jr, int(tip))
                    self.jac_returned += 1
                    self.g["jacobian_full_nv"][j, leg] = self.jp
                self.g["jacobian_valid"][j] = True
            if c - 1 in ANCHORS:
                j = ANCHORS.index(c - 1)
                slots = np.flatnonzero((self.a["native_control_end"] == c) & self.a["native_returned"])
                require(len(slots) == 50 and self.metadata["activation_tau_s"] == 0
                        and self.metadata["interpolation"] == "held raw ctrl", "neutral actual tau0 held native interval")
                raw = self.a["native_raw_ctrl_rad"][slots]
                require(np.array_equal(raw, np.broadcast_to(raw[0], raw.shape)), "all50 actual interval control values held")
                self.g["interval_raw_ctrl_rad"][j] = raw
                self.g["interval_valid"][j] = True

        def close(self):
            first = None
            try:
                if self.forward_binding is not None:
                    self.forward_binding.restore()
            except Exception as error:
                first = error
            try:
                super().close()
                require(fk_identity(self.s.body.m, self.np) == self.fk, "final FK byte invariance")
                require(control_identity(self.s.body.m, self.metadata["actuator_ids"], self.np) == self.control_id, "final compiled control mapping invariant")
                self.control_end_verified = self.geometry_metadata["control_end_verified"] = True
            except Exception as error:
                if first is not None:
                    first.add_note(str(error))
                else:
                    first = error
            if first is not None:
                raise first

        def geometry_summary(self):
            return {"boundary_valid": self.g["valid"].tolist(), "jacobian_valid": self.g["jacobian_valid"].tolist(),
                    "interval_valid": self.g["interval_valid"].tolist(), "existing_forward_requested": self.forward_requested,
                    "existing_forward_returned": self.forward_returned, "jacobian_requested": self.jac_requested, "jacobian_returned": self.jac_returned,
                    "extra_live_forward": 0, "extra_live_neural": 0, "extra_live_physics": 0, "serialization": self.serialization,
                    "control_boundary_verified": dict(self.control_boundary_verified), "control_end_verified": self.control_end_verified,
                    "complete": bool(self.g["valid"].all() and self.g["jacobian_valid"].all() and self.g["interval_valid"].all()
                                     and self.control_end_verified and len(self.control_boundary_verified) == 8
                                     and self.jac_requested == self.jac_returned == 24 and self.forward_requested == self.forward_returned == 300)}

    return GeometryTracker


def geometry_artifacts(tracker, result, artifacts, publisher, *, serialization=None):
    if tracker is None:
        result["geometry"] = {"complete": False, "status": "UNAVAILABLE_TRACKER_NOT_RETURNED", "serialization": serialization}
        result["capture_complete"] = False
        artifacts["geometry_work"] = publisher.json("model-serialization-work.json", serialization)
        return
    artifacts["geometry_work"] = publisher.json("model-serialization-work.json", tracker.serialization)
    artifacts["model"] = tracker.model_spec
    artifacts["geometry"] = publisher.arrays("geometry.npz", tracker.g)
    artifacts["geometry_metadata"] = publisher.json("geometry-metadata.json", tracker.geometry_metadata)
    result["geometry"] = tracker.geometry_summary()
    result["capture_complete"] = bool(result["capture_complete"] and result["geometry"]["complete"])


def worker(admission):
    require(admission.role == "geometry_capture" and admission.stage == "pre_array", "one unchanged neutral geometry role")
    admission.fence("pre_array")
    from tools import observe_c_banc_m3_startup_tracking as tracking
    from tools import verify_c_banc_m3_candidate_body_handoff as body
    from tools import verify_c_banc_m3_common_descent_body as b
    from tools import verify_c_banc_m3_reference_parameter_direction as ref
    from tools import verify_c_banc_m3_startup_ray as ray

    publisher, watch = tracking.bounded_publisher(admission), body.gpu_watch({"limits": admission.limits})
    primary_error = None
    serialization = empty_model_work()  # owner survives a GeometryTracker constructor failure
    try:
        selected = ray.json_load(ray.leaf(admission.roster["selected_candidate"], admission.root))
        audit = ray.json_load(ray.leaf(admission.roster["candidate_audit"], admission.root))
        fixed = admission.roster["candidate_identity"]
        require(selected["alpha"] == fixed["alpha"] == .0001 and selected["index"] == fixed["index"] == 4 and audit["raw_valid"] is True, "unchanged rejected candidate provenance")
        ray.exact_value(audit["selected_candidate"], selected, "same audited identity")
        arrays = ray.saved_arrays(admission.roster["candidate_arrays"], admission)
        data = ray.arm_data(arrays, admission.inputs["body_inventory"]["inventory"], admission, "neutral", b=b)
        data["reference"] = ref.native().load_selected(admission.root, data["inventory"]["selected_archives"]["reference"], data["ledger"], timely=admission.check, generated=False)
        result = tracking.continuous_arm(data, "neutral", selected, {"palette": selected["palette"]}, admission, publisher, watch,
                                         tracker_factory=geometry_factory(publisher, admission, serialization=serialization),
                                         artifact_hook=lambda *args: geometry_artifacts(*args, serialization=serialization))
        require(result["run_valid"] and result["capture_complete"], "complete technically valid neutral geometry capture")
        return result
    except Exception as error:
        primary_error = error
        raise
    finally:
        try:
            if watch.snapshot()["state"] != "CLOSED":
                publisher.json("sampler.json", watch.close())
        except Exception as error:
            if primary_error is None:
                raise
            primary_error.add_note("sampler cleanup: " + type(error).__name__ + ": " + str(error))


def main(argv=None):
    from tools.banc_m3_local_foot_geometry_admission import Admission
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    args = parser.parse_args(argv)
    entry = float(json.loads(os.environ["FLYLAB_DIRECT_SCIENCE_OWNER"])["entry_counter_s"])
    result = worker(Admission(args.request, entry=entry))
    print(json.dumps({"run_valid": result["run_valid"], "capture_complete": result["capture_complete"], "candidate_adopted": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
