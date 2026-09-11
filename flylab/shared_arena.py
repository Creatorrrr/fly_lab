"""Several independently controlled flies in one CPU MuJoCo world."""

import hashlib
import math

import numpy as np

from . import CONTROL_DT, PHYSICS_DT
from .body_options import BodyOptions, make_flybody, make_world
from .common import number


class SharedFlyArena:
    def __init__(self, count=2, *, seed=42, spacing_mm=6.0, collisions=True):
        import mujoco as mj
        from flygym.anatomy import BodySegment, JointDOF, RotationAxis
        from flygym.compose import ActuatorType
        from flygym.flybody import FlyBodyContactBodiesPreset
        from flygym.simulation import Simulation
        from flygym.utils.math import Rotation3D
        from flygym_demo.complex_terrain import (
            FlyBodyPreprogrammedSteps,
            HybridControllerObservation,
            HybridTurningController,
            apply_locomotion_action,
        )

        if type(count) is not int or not 2 <= count <= 4:
            raise ValueError("Shared arena supports two to four flies")
        if type(seed) is not int or not 0 <= seed <= 2**32 - 4:
            raise ValueError("Invalid shared-arena seed")
        number(spacing_mm, "spacing_mm", 3.0, 20.0)
        if type(collisions) is not bool:
            raise ValueError("collisions must be boolean")
        self.spec = {
            "count": count,
            "seed": seed,
            "spacing_mm": spacing_mm,
            "collisions": collisions,
        }
        self.mj, self.ActuatorType = mj, ActuatorType
        self.observe_control, self.apply = (
            HybridControllerObservation.from_sim,
            apply_locomotion_action,
        )
        self.world = make_world(BodyOptions(model="flybody"))
        self.flies, self.controllers = {}, {}
        for i in range(count):
            name = f"fly_{i + 1}"
            fly = make_flybody(name)
            # Each fly collides with other flies, never itself through these masks.
            for segment, geoms in fly.bodyseg_to_mjcfgeom.items():
                include = (
                    segment.is_leg()
                    or segment.link in ("thorax", "head")
                    or segment.link.startswith("abdomen")
                )
                for geom in geoms:
                    geom.contype = (1 << i) if collisions and include else 0
                    geom.conaffinity = (
                        (((1 << count) - 1) ^ (1 << i)) if collisions and include else 0
                    )
            angle = 2 * math.pi * i / count
            pos = (
                spacing_mm / 2 * math.cos(angle),
                spacing_mm / 2 * math.sin(angle),
                0.8,
            )
            heading = angle + math.pi
            self.world.add_fly(
                fly,
                pos,
                Rotation3D(
                    "quat", (math.cos(heading / 2), 0, 0, math.sin(heading / 2))
                ),
                bodysegs_with_ground_contact=FlyBodyContactBodiesPreset.TIBIA_TARSUS_ONLY,
                add_ground_contact_sensors=False,
            )
            self.flies[name] = fly
            order = fly.get_actuated_jointdofs_order(ActuatorType.POSITION)
            order = [
                JointDOF(
                    BodySegment(d.parent.name),
                    BodySegment(d.child.name),
                    RotationAxis(d.axis.value),
                )
                for d in order
            ]
            controller = HybridTurningController(
                timestep=PHYSICS_DT,
                preprogrammed_steps=FlyBodyPreprogrammedSteps(),
                output_dof_order=order,
            )
            controller.reset(seed=seed + i)
            controller.enable_adhesion = False
            self.controllers[name] = controller
        self.world.mjcf_root.compiler.fusestatic = False
        self.sim = Simulation(self.world, timestep=PHYSICS_DT)
        self.sim.reset()
        self.m, self.d = self.sim.mj_model, self.sim.mj_data
        self.names = list(self.flies)
        self.neutral_controls = {
            n: self.d.ctrl[
                self.sim._intern_actuatorids_by_type_by_fly[ActuatorType.POSITION][n]
            ].copy()
            for n in self.names
        }
        self.drives = {n: np.zeros(2) for n in self.names}
        self.body_ids = {n: self.sim._internal_bodyids_by_fly[n] for n in self.names}
        self.thorax_ids = {
            n: mj.mj_name2id(self.m, mj.mjtObj.mjOBJ_BODY, n + "/c_thorax")
            for n in self.names
        }
        self.owners = np.full(self.m.nbody, -1, int)
        for i, n in enumerate(self.names):
            self.owners[self.body_ids[n]] = i
        self.initialized_at = 0.0
        self.fault = None
        self.renderer = None
        for _ in range(2000):
            mj.mj_step(self.m, self.d)
        self._check(2000 * PHYSICS_DT)
        mj.mj_forward(self.m, self.d)
        self.initialized_at = float(self.d.time)
        self.model_hash = hashlib.sha256(
            self.world.mjcf_root.to_xml().encode()
        ).hexdigest()

    def _check(self, expected):
        if (
            not np.isfinite(self.d.qpos).all()
            or not np.isfinite(self.d.qvel).all()
            or any(self.d.warning.number)
            or abs(float(self.d.time) - expected) > 1e-8
        ):
            self.fault = "Invalid shared physics state or automatic time reset"
            raise RuntimeError(self.fault)

    def set_drives(self, drives):
        if not isinstance(drives, dict) or not drives or set(drives) - set(self.names):
            raise ValueError("Named fly drives required")
        candidate = {}
        for name, value in drives.items():
            v = np.asarray(value, dtype=float)
            if v.shape != (2,) or not np.isfinite(v).all() or np.any(np.abs(v) > 1.3):
                raise ValueError("Each fly needs two finite drives in -1.3..1.3")
            candidate[name] = v.copy()
        self.drives.update(candidate)

    def advance(self, steps=1):
        if type(steps) is not int or not 1 <= steps <= 200:
            raise ValueError("Use 1..200 control steps")
        if self.fault:
            raise RuntimeError(self.fault)
        for _ in range(steps):
            expected = float(self.d.time) + CONTROL_DT
            for _ in range(round(CONTROL_DT / PHYSICS_DT)):
                for name, controller in self.controllers.items():
                    if np.max(np.abs(self.drives[name])) < 1e-7:
                        # Zero input holds the official neutral pose.
                        self.sim.set_actuator_inputs(
                            name,
                            self.ActuatorType.POSITION,
                            self.neutral_controls[name],
                        )
                        continue
                    action = controller.step(
                        self.drives[name], self.observe_control(self.sim, name)
                    )
                    self.apply(self.sim, name, action)
                self.sim.step()
            self._check(expected)
        self.mj.mj_forward(self.m, self.d)
        return self.observation()

    def observation(self):
        contact = self.d.contact[: self.d.ncon]
        cross = []
        for c in contact:
            a, b = (
                int(self.owners[self.m.geom_bodyid[c.geom1]]),
                int(self.owners[self.m.geom_bodyid[c.geom2]]),
            )
            if a >= 0 and b >= 0 and a != b and not c.exclude:
                cross.append([self.names[a], self.names[b], float(c.dist)])
        return {
            "schema": "flylab.shared-arena.v1",
            "time_s": float(self.d.time) - self.initialized_at,
            "shared_world": True,
            "physical_worlds": 1,
            "controller": "independent FlyGym hybrid controllers",
            "neural_connected": False,
            "collisions_enabled": self.spec["collisions"],
            "fault": self.fault,
            "model_hash": self.model_hash,
            "interfly_contacts": cross,
            "flies": {
                n: {
                    "position_mm": self.d.xpos[self.thorax_ids[n]].tolist(),
                    "drive": self.drives[n].tolist(),
                    "joint_angles_rad": self.sim.get_joint_angles(n).tolist(),
                }
                for n in self.names
            },
        }

    def snapshot(self):
        typ = self.mj.mjtState.mjSTATE_INTEGRATION
        state = np.empty(self.mj.mj_stateSize(self.m, typ))
        self.mj.mj_getState(self.m, self.d, state, typ)
        controllers = {}
        for n, c in self.controllers.items():
            rng = c.cpg_network.random_state.get_state()
            controllers[n] = {
                "cpg": {
                    k: np.asarray(getattr(c.cpg_network, k)).tolist()
                    for k in (
                        "curr_phases",
                        "curr_magnitudes",
                        "intrinsic_freqs",
                        "intrinsic_amps",
                    )
                },
                "rng": [
                    rng[0],
                    rng[1].tolist(),
                    int(rng[2]),
                    int(rng[3]),
                    float(rng[4]),
                ],
                "reflex": {
                    k: np.asarray(getattr(c, k)).tolist()
                    for k in (
                        "retraction_correction",
                        "stumbling_correction",
                        "retraction_persistence_counter",
                    )
                },
            }
        return {
            "schema": "flylab.shared-checkpoint.v1",
            "model_hash": self.model_hash,
            "spec": self.spec.copy(),
            "state": state.tolist(),
            "controllers": controllers,
            "drives": {n: v.tolist() for n, v in self.drives.items()},
            "initialized_at": self.initialized_at,
            "fault": self.fault,
        }

    def restore(self, saved):
        if (
            not isinstance(saved, dict)
            or saved.get("schema") != "flylab.shared-checkpoint.v1"
            or saved.get("model_hash") != self.model_hash
            or saved.get("spec") != self.spec
        ):
            raise ValueError("Shared arena checkpoint model mismatch")
        if set(saved.get("controllers", {})) != set(self.names) or set(
            saved.get("drives", {})
        ) != set(self.names):
            raise ValueError("Saved fly identities differ")
        if saved.get("fault") is not None and not isinstance(saved["fault"], str):
            raise ValueError("Invalid saved fault")
        # Validate using independent controller copies before mutating live state.
        import copy

        state = np.asarray(saved["state"], dtype=float)
        typ = self.mj.mjtState.mjSTATE_INTEGRATION
        if (
            state.shape != (self.mj.mj_stateSize(self.m, typ),)
            or not np.isfinite(state).all()
        ):
            raise ValueError("Invalid shared integration state")
        candidate = copy.deepcopy(self.controllers)
        for n, c in candidate.items():
            s = saved["controllers"][n]
            if (
                set(s) != {"cpg", "reflex", "rng"}
                or set(s["cpg"])
                != {
                    "curr_phases",
                    "curr_magnitudes",
                    "intrinsic_freqs",
                    "intrinsic_amps",
                }
                or set(s["reflex"])
                != {
                    "retraction_correction",
                    "stumbling_correction",
                    "retraction_persistence_counter",
                }
            ):
                raise ValueError("Incomplete shared controller state")
            for obj, fields in ((c.cpg_network, s["cpg"]), (c, s["reflex"])):
                for k, v in fields.items():
                    dst = getattr(obj, k)
                    v = np.asarray(v)
                    if v.shape != np.shape(dst) or not np.isfinite(v).all():
                        raise ValueError("Invalid controller state")
                    dst[:] = v
            r = s["rng"]
            c.cpg_network.random_state.set_state(
                (r[0], np.asarray(r[1], np.uint32), int(r[2]), int(r[3]), float(r[4]))
            )
        drives = {n: np.asarray(saved["drives"][n], float) for n in self.names}
        if any(
            v.shape != (2,) or not np.isfinite(v).all() or (np.abs(v) > 1.3).any()
            for v in drives.values()
        ):
            raise ValueError("Invalid saved drives")
        start = number(saved["initialized_at"], "initialized_at", 0, 1e8)
        self.mj.mj_setState(self.m, self.d, state, typ)
        self.mj.mj_forward(self.m, self.d)
        self.controllers, self.drives = candidate, drives
        self.initialized_at, self.fault = start, saved["fault"]

    def preview(self):
        if self.renderer is None:
            self.renderer = self.mj.Renderer(self.m, height=360, width=640)
        cam = self.mj.MjvCamera()
        cam.type = self.mj.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = np.mean(
            [self.d.xpos[self.thorax_ids[n]] for n in self.names], axis=0
        )
        cam.distance = max(12.0, self.spec["spacing_mm"] * 2)
        cam.azimuth = 100
        cam.elevation = -55
        self.renderer.update_scene(self.d, cam)
        return self.renderer.render().copy()

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
