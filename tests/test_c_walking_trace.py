"""Observer contracts plus real, minimal MuJoCo contact mechanics (not BANC gait)."""

import importlib.util
import unittest
from types import SimpleNamespace

import numpy as np

from flylab.c.walking_trace import WalkingTrace, contact_diagnostics, summarize_trace


def row(tick):
    return {
        "control_tick": tick,
        "time_s": tick * 0.005,
        "body": {"position": [tick * 0.01, 1, 0]},
        "upright": 1.0,
        "joint_tracking_error_rad": [0.0] * 42,
        "feet": {"load_bearing": [False] * 6, "slip_mm_s": [None] * 6},
    }


class TraceContracts(unittest.TestCase):
    def test_bounds_clocks_copying_and_paging(self):
        trace = WalkingTrace(3)
        for i in range(5):
            source = row(i)
            trace.append(source)
            source["body"]["position"][0] = 999
        page = trace.page(after=0, limit=2)
        self.assertEqual([s["control_tick"] for s in page["samples"]], [2, 3])
        self.assertTrue(page["gap_before_page"])
        self.assertTrue(page["has_more"])
        self.assertEqual(page["evicted_samples"], 2)
        page["samples"][0]["body"]["position"][0] = 777
        self.assertEqual(trace.page()["samples"][0]["body"]["position"][0], 0.02)
        self.assertEqual(trace.page(after=3)["samples"][0]["control_tick"], 4)
        with self.assertRaises(ValueError):
            trace.append(row(4))
        invalid = row(5)
        invalid["time_s"] += 1
        with self.assertRaises(ValueError):
            trace.append(invalid)
        for options in ({"limit": 201}, {"limit": True}, {"after": -2}):
            with self.assertRaises(ValueError):
                trace.page(**options)

    def test_epochs_and_missing_contact_measurements(self):
        self.assertNotEqual(WalkingTrace().epoch, WalkingTrace().epoch)
        rows = [row(i) for i in [0, 1, 4]]
        result = summarize_trace(rows)
        self.assertEqual(result["missing_control_intervals"], 2)
        self.assertEqual(result["gait_verdict"], "NOT_ASSESSED")
        self.assertIsNone(result["legs"][0]["loaded_slip_mean_mm_s"])
        self.assertAlmostEqual(result["observed_path_mm"], 0.01)
        rows[1]["feet"]["load_bearing"][0] = True
        rows[1]["feet"]["slip_mm_s"][0] = 2.0
        result = summarize_trace(rows)
        self.assertEqual(result["legs"][0]["stance_transitions"], 1)
        self.assertEqual(result["legs"][0]["loaded_slip_mean_mm_s"], 2.0)


@unittest.skipUnless(importlib.util.find_spec("mujoco"), "Optional MuJoCo runtime")
class PhysicalContactDiagnostics(unittest.TestCase):
    def model(self, moving_floor=False):
        import mujoco as mj

        floor = (
            '<body name="ground" pos="0 0 -.1"><freejoint/>'
            '<geom name="floor" type="box" size="5 5 .1" mass="100"/></body>'
            if moving_floor
            else '<geom name="floor" type="plane" size="5 5 .1"/>'
        )
        xml = (
            '<mujoco><option timestep=".0001"/><worldbody>'
            + floor
            + (
                '<body name="foot" pos="0 0 .45"><freejoint/>'
                '<geom name="pad" type="sphere" size=".5" mass="1"/>'
                "</body></worldbody></mujoco>"
            )
        )
        m = mj.MjModel.from_xml_string(xml)
        d = mj.MjData(m)
        floor_id = mj.mj_name2id(m, mj.mjtObj.mjOBJ_GEOM, "floor")
        pad_id = mj.mj_name2id(m, mj.mjtObj.mjOBJ_GEOM, "pad")
        d.qvel[-6] = 2.0
        if moving_floor:
            d.qvel[0] = 2.0
        mj.mj_forward(m, d)
        mapping = np.full(m.ngeom, -1, dtype=int)
        mapping[pad_id] = 0
        body = SimpleNamespace(
            m=m,
            d=d,
            mj=mj,
            weight0=9.81,
            _force_maps={"feet": (mapping, 30)},
            support_geom_ids={floor_id},
        )
        return body

    def state(self, body):
        mj = body.mj
        kind = mj.mjtState.mjSTATE_INTEGRATION
        result = np.empty(mj.mj_stateSize(body.m, kind))
        mj.mj_getState(body.m, body.d, result, kind)
        return result

    def test_actual_contact_velocity_and_no_state_mutation(self):
        body = self.model()
        before = self.state(body)
        result = contact_diagnostics(body)
        self.assertTrue(result["load_bearing"][0])
        self.assertAlmostEqual(result["slip_mm_s"][0], 2.0, places=12)
        self.assertIsNone(result["slip_mm_s"][1])
        np.testing.assert_array_equal(
            before.view(np.uint64), self.state(body).view(np.uint64)
        )
        # Same next physics step with and without the extra observer calls.
        control = self.model()
        for _ in range(20):
            contact_diagnostics(body)
            body.mj.mj_step(body.m, body.d)
            control.mj.mj_step(control.m, control.d)
            np.testing.assert_array_equal(
                self.state(body).view(np.uint64), self.state(control).view(np.uint64)
            )

    def test_real_flybody_capture_preserves_future(self):
        if not importlib.util.find_spec("flygym"):
            self.skipTest("Optional FlyGym")
        from flylab.body import FlyGymBody
        from flylab.body_options import BodyOptions
        from flylab.c.walking_trace import capture_control
        from flylab.engine import config_values
        from flylab.sensors import default_world

        options = BodyOptions(
            model="flybody",
            actuation="whole_body",
            servo_profile="tracking_all",
            tendons="all",
        )
        body = FlyGymBody(
            42, default_world(), config_values(None), body_options=options
        )
        self.addCleanup(body.close)
        control = FlyGymBody(
            42, default_world(), config_values(None), body_options=options
        )
        self.addCleanup(control.close)
        tendon = {name: 0.2 for name in body.tendon_control.names}
        session = SimpleNamespace(
            body=body,
            control_tick=0,
            tendon_inputs=tendon,
            rate=np.zeros(3),
            fault=None,
        )
        baseline_cpg = body.cpg_state()
        for tick in range(11):
            session.control_tick = tick
            before = self.state(body)
            sample = capture_control(session)
            self.assertEqual(sample["control_tick"], tick)
            self.assertEqual(
                set(sample["body"]["legs"]), {"lf", "lm", "lh", "rf", "rm", "rh"}
            )
            np.testing.assert_array_equal(
                before.view(np.uint64), self.state(body).view(np.uint64)
            )
            np.testing.assert_array_equal(
                self.state(body).view(np.uint64), self.state(control).view(np.uint64)
            )
            if tick < 10:
                for value in (body, control):
                    value.step_joint_targets(
                        value.neutral, np.zeros(6, dtype=bool), tendon_inputs=tendon
                    )
        self.assertEqual(body.cpg_state(), baseline_cpg)

    def test_moving_surface_uses_relative_not_absolute_speed(self):
        body = self.model(moving_floor=True)
        result = contact_diagnostics(body)
        self.assertTrue(result["load_bearing"][0])
        self.assertAlmostEqual(result["slip_mm_s"][0], 0.0, places=12)


if __name__ == "__main__":
    unittest.main()
