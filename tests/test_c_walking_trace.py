"""Observer contracts plus real, minimal MuJoCo contact mechanics (not BANC gait)."""

import importlib.util
import unittest
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from flylab.c.walking_trace import (
    WalkingTrace,
    capture_control,
    capture_initial_state,
    capture_neural_request,
    command_interval,
    contact_diagnostics,
    summarize_trace,
)


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

    def test_export_anchor_is_never_reused_after_eviction_or_a_gap(self):
        initial = {"control_tick": 10, "time_s": .05, "body": {"state": [1, 2, 3]},
                   "body_hash": "body", "units": {"time": "model seconds"}}
        def command_row(tick):
            return {**row(tick), "command_interval": command_interval(tick),
                    "joint_targets_rad": [0.0] * 42, "adhesion": [False] * 6,
                    "tendon_inputs": {}, "neural_request": {"end_tick": tick}}
        trace = WalkingTrace(3);trace.start(initial)
        trace.append(command_row(10));trace.append(command_row(11))
        result = trace.export_metadata(include_initial=True)
        self.assertTrue(result["command_replay"]["eligible"])
        self.assertFalse(result["command_replay"]["verified"])
        self.assertEqual(result["initial_physics"]["control_tick"], 10)
        result["initial_physics"]["body"]["state"][0] = 999
        self.assertEqual(trace.initial["body"]["state"][0], 1)
        self.assertNotIn("initial_physics", trace.export_metadata())
        trace.append(command_row(12));trace.append(command_row(13))
        result = trace.export_metadata(include_initial=True)
        self.assertFalse(result["command_replay"]["eligible"])
        self.assertIn("INITIAL_SAMPLE_EVICTED_OR_UNAVAILABLE", result["command_replay"]["reasons"])
        self.assertIsNone(result["initial_physics"])
        self.assertEqual(result["trace_contract"]["body_hash"], "body")
        gapped = WalkingTrace();gapped.start(initial)
        gapped.append(command_row(10));gapped.append(command_row(12))
        result = gapped.export_metadata(include_initial=True, trace_error="diagnostic failure")
        self.assertFalse(result["command_replay"]["eligible"])
        self.assertIn("MISSING_CONTROL_INTERVALS", result["command_replay"]["reasons"])
        self.assertIn("TRACE_CAPTURE_ERROR", result["command_replay"]["reasons"])
        with self.assertRaises(ValueError):
            gapped.start(initial)

    def test_command_interval_and_initial_clock_are_explicit(self):
        self.assertIsNone(command_interval(0))
        self.assertEqual(command_interval(8), {"start_tick": 7, "end_tick": 8, "start_s": .035, "end_s": .04})
        trace = WalkingTrace();trace.start({"control_tick": 8, "time_s": .04})
        with self.assertRaises(ValueError):
            trace.append(row(9))
        trace.append(row(8));trace.append(row(9))
        self.assertIn("INCOMPLETE_COMMAND_RECORDS", trace.export_metadata()["command_replay"]["reasons"])


class TraceRequestContracts(unittest.TestCase):
    def session(self):
        @dataclass
        class Config:
            neural_dt_s: float = .00025
        body = SimpleNamespace(
            d=SimpleNamespace(qpos=np.zeros(42), qvel=np.zeros(42), ctrl=np.zeros(42), xpos=np.zeros((1, 3))),
            body_ids=np.array([0]), body_indices={"c_thorax": 0}, qpos_ids=np.arange(42),
            last_action=SimpleNamespace(joint_angles=np.zeros(42), adhesion_onoff=np.zeros(6, bool)),
            pose=lambda: (np.zeros(3), np.eye(3), np.zeros(6)),
            snapshot=lambda: {"state": [1.0, 2.0], "push_left": .003, "push_force": [1., 2., 3.]},
            physics_time=lambda: .01, physics_identity=lambda: {"timestep": .0001},
            model_hash="body", joint_names=[str(i) for i in range(42)],
            joint_ids=np.arange(42), act_ids=np.arange(42),
            m=SimpleNamespace(opt=SimpleNamespace(timestep=.0001),
                jnt_range=np.tile([-.5, .3], (42, 1)), jnt_limited=np.ones(42),
                actuator_ctrlrange=np.tile([-.5, .3], (42, 1)), actuator_ctrllimited=np.ones(42)),
            push_left=.003, push_force=np.array([1., 2., 3.]), fault=None,
        )
        return SimpleNamespace(
            body=body, control_tick=2, trace=WalkingTrace(), trace_error=None,
            graph=SimpleNamespace(hash="graph", nodes=[
                {"id": "left", "cell_type": "DNg100", "soma_side": "left", "output_side": "left"},
                {"id": "right", "cell_type": "DNg100", "soma_side": "right", "output_side": "right"},
            ]), descending=np.array([0, 1]), network=SimpleNamespace(identity="network"),
            adapter=SimpleNamespace(identity="adapter"), p=Config(), options=Config(), world={},
            navigation=None, cuts={"motor": False, "descending": False, "circuit": False, "sensory": False},
            tendon_inputs={}, rate=np.zeros(2), fault=None,
        )

    def test_capture_reads_existing_navigation_and_host_request_without_encoding(self):
        from flylab.c.target_sense import TargetSensor
        session = self.session()
        session.navigation = SimpleNamespace(
            sensor=TargetSensor([6, -3]), cut=False,
            encoder=SimpleNamespace(turn=1, braking=False, ports=np.array([2, 3])),
        )
        drive = np.array([5000., 4999., -5000., 5000.], np.float32)
        original = drive.copy()
        request = capture_neural_request(session, drive, np.zeros(3), np.eye(3))
        self.assertEqual(request["descending_drive"], [5000., 4999.])
        self.assertEqual(request["steering_port_drive"], [-5000., 5000.])
        self.assertEqual(request["navigation_at_start"]["turn_request"], 1)
        self.assertAlmostEqual(request["navigation_at_start"]["distance_mm"], np.sqrt(45))
        self.assertEqual(request["external_force_at_start"]["push_left_s"], .003)
        np.testing.assert_array_equal(drive, original)
        self.assertEqual(session.navigation.encoder.turn, 1)
        session._trace_neural_request = request;session.control_tick = 3
        feet={"normal_load_bw": [0.] * 6}
        with patch("flylab.c.walking_trace.contact_diagnostics", return_value=feet):
            sample = capture_control(session)
            self.assertEqual(sample["command_interval"]["start_tick"], 2)
            self.assertEqual(sample["neural_request"]["start_tick"], 2)
            self.assertEqual(sample["navigation"]["target_xz_mm"], [6., -3.])
            session.control_tick = 5
            self.assertIsNone(capture_control(session)["neural_request"])

    def test_first_record_snapshots_cpu_physics_once_and_never_reads_network_state(self):
        from flylab.c.banc_walking import BancWalkingSession
        session = self.session();calls=[]
        def snapshot():
            calls.append(session.control_tick)
            return {"state": [1., 2.], "push_left": .003, "push_force": [1., 2., 3.]}
        session.body.snapshot=snapshot
        with patch("flylab.c.walking_trace.contact_diagnostics", return_value={"normal_load_bw": [0.] * 6}):
            BancWalkingSession._record_control(session)
            self.assertIsNone(session.trace_error)
            session.control_tick=3
            BancWalkingSession._record_control(session)
        self.assertEqual(calls, [2])
        self.assertEqual(session.trace.initial["descending_ports"][0]["soma_side"], "left")
        self.assertEqual(session.trace.initial["body"]["push_left"], .003)
        self.assertEqual(session.trace.initial["joint_commands"]["joint_range_rad"][0], [-.5, .3])
        session.body.physics_time=lambda: float("nan")
        with self.assertRaises(ValueError):
            capture_initial_state(session)

    def test_observer_failure_stays_separate_from_model_fault(self):
        from flylab.c.banc_walking import BancWalkingSession
        session = self.session()
        with patch("flylab.c.banc_walking.capture_neural_request", side_effect=ValueError("observer only")):
            BancWalkingSession._record_neural_request(session, np.zeros(2), np.zeros(3), np.eye(3))
        self.assertEqual(session.trace_error, "observer only")
        self.assertIsNone(session.fault);self.assertIsNone(session.body.fault)


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
