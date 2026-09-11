"""Official FlyGym GPUSimulation adapter for matched held-input comparisons."""

import numpy as np

from . import CONTROL_DT, PHYSICS_DT


class OfficialGPUComparison:
    def __init__(self, bodies):
        import mujoco as mj
        import mujoco_warp as mjw
        import warp as wp
        from flygym.warp import GPUSimulation

        if not bodies or any(b.model_hash != bodies[0].model_hash for b in bodies):
            raise ValueError("Identical body models are required")
        first = bodies[0]
        if first.physics_profile.noslip_iterations or first.physics_profile.multiccd:
            raise ValueError("Use matched noslip=0, multiccd=False body options")
        if (
            first.body_options.actuation != "legs"
            or first.body_options.tendons != "none"
            or first.body_options.attachment != "free"
        ):
            raise ValueError(
                "Official GPU comparison currently supports free leg-servo bodies"
            )
        self.wp, self.mjw, self.bodies = wp, mjw, list(bodies)
        self.sim = GPUSimulation(
            first.native_world,
            len(bodies),
            timestep=PHYSICS_DT,
            max_contacts=first.physics_profile.max_contacts,
            max_constraints=first.physics_profile.max_constraints,
        )
        # FlyLab applies environment edits after compilation. Keep official
        # stepping/query methods, but upload the exact effective reference model.
        if (self.sim.mj_model.nq, self.sim.mj_model.nv, self.sim.mj_model.nu) != (
            first.m.nq,
            first.m.nv,
            first.m.nu,
        ):
            raise ValueError("Official model topology differs")
        np.testing.assert_array_equal(self.sim.mj_model.names, first.m.names)
        self.sim.mj_model = first.m
        self.sim.mj_data = mj.MjData(first.m)
        self.sim.mjw_model = mjw.put_model(first.m)
        self.sim.mjw_data = mjw.put_data(
            first.m,
            first.d,
            nworld=len(bodies),
            njmax=first.physics_profile.max_constraints,
            nconmax=first.physics_profile.max_contacts,
        )
        self.reset_from_bodies()
        self.graph = None

    def reset_from_bodies(self):
        for key in (
            "qpos",
            "qvel",
            "act",
            "ctrl",
            "qacc_warmstart",
            "qfrc_applied",
            "xfrc_applied",
            "mocap_pos",
            "mocap_quat",
            "eq_active",
            "history",
        ):
            data = np.asarray([getattr(b.d, key) for b in self.bodies])
            if data.size:
                getattr(self.sim.mjw_data, key).assign(data)
        self.sim.mjw_data.time.assign(
            np.asarray([b.d.time for b in self.bodies], np.float32)
        )
        self.mjw.forward(self.sim.mjw_model, self.sim.mjw_data)
        self.wp.synchronize()

    def advance(self, targets, adhesion):
        b = self.bodies[0]
        targets = np.asarray(targets, float)
        adhesion = np.asarray(adhesion, bool)
        if (
            targets.shape != (len(self.bodies), 42)
            or not np.isfinite(targets).all()
            or adhesion.shape != (len(self.bodies), 6)
        ):
            raise ValueError("Expected batched 42-joint and six-foot controls")
        self.sim.set_actuator_inputs(b.fly.name, b.ActuatorType.POSITION, targets)
        # The same actuator mapping used by the existing batch controller.
        ctrl = self.sim.mjw_data.ctrl.numpy()
        ids = b.sim._intern_adhesionactuatorids_by_fly[b.fly.name]
        ctrl[:, ids] = adhesion.astype(np.float32)
        self.sim.mjw_data.ctrl.assign(ctrl)
        if self.graph is None:
            # Compile before capture; the warm-up state is restored immediately.
            saved = self.snapshot()
            self.sim.step()
            self.wp.synchronize()
            self.restore(saved)
            with self.wp.ScopedCapture() as captured:
                for _ in range(round(CONTROL_DT / PHYSICS_DT)):
                    self.sim.step()
                self.mjw.forward(self.sim.mjw_model, self.sim.mjw_data)
            self.graph = captured.graph
        self.wp.capture_launch(self.graph)

    def snapshot(self):
        return {
            key: getattr(self.sim.mjw_data, key).numpy().copy()
            for key in (
                "qpos",
                "qvel",
                "act",
                "ctrl",
                "qacc_warmstart",
                "qfrc_applied",
                "xfrc_applied",
                "mocap_pos",
                "mocap_quat",
                "eq_active",
                "history",
                "time",
            )
        }

    def restore(self, saved):
        for key, value in saved.items():
            dest = getattr(self.sim.mjw_data, key)
            if value.shape != dest.numpy().shape or not np.isfinite(value).all():
                raise ValueError("Invalid GPU comparison state: " + key)
        for key, value in saved.items():
            if value.size:
                getattr(self.sim.mjw_data, key).assign(value)
        self.mjw.forward(self.sim.mjw_model, self.sim.mjw_data)
        self.wp.synchronize()

    def audit(self):
        d = self.sim.mjw_data
        self.wp.synchronize()
        if (
            not np.isfinite(d.qpos.numpy()).all()
            or not np.isfinite(d.qvel.numpy()).all()
        ):
            raise RuntimeError("Nonfinite official GPU state")
        if (
            int(d.nacon.numpy()[0]) > d.naconmax
            or int(d.ncollision.numpy()[0]) > d.naconmax
            or int(d.nefc.numpy().max()) > d.njmax
        ):
            raise RuntimeError("Official GPU contact/constraint capacity exceeded")

    def close(self):
        self.wp.synchronize()
        self.graph = None
        self.sim = None
