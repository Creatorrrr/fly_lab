"""Independent signals guard against misleading biological gait measurements."""

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.analyze_c_banc_biological_gait import (
    circular_summary,
    geometric_knee,
    load_model,
    load_reference,
    phase_and_events,
    phase_summary,
    position_forward_speed,
    validate_clock,
)


def oscillation():
    t = np.arange(0, 3, 0.005)
    phases = np.array([0, np.pi, 0, np.pi, 0, np.pi])
    ap = 0.2 * np.cos(2 * np.pi * 5 * t[:, None] + phases)
    return {
        "t": t,
        "dt": 0.005,
        "ap": ap,
        "moving": np.ones(len(t), dtype=bool),
        "straight": np.ones(len(t), dtype=bool),
        "speed": np.full(len(t), 5.0),
    }


class BiologicalGaitAnalysisTests(unittest.TestCase):
    def test_navigation_free_trace_and_explicit_braking_window(self):
        rows = []
        for i in range(141):
            point = np.array([i * 0.005, 0.0, 0.0])
            chain = (
                np.array([[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 2, 0], [2, 2, 0]])
                + point
            )
            rows.append(
                {
                    "time_s": i * 0.005,
                    "control_tick": i,
                    "navigation": None,
                    "body": {
                        "basis": np.eye(3).tolist(),
                        "position": point.tolist(),
                        "legs": {
                            leg: chain.tolist()
                            for leg in ("lf", "lm", "lh", "rf", "rm", "rh")
                        },
                        "yaw": 0.0,
                    },
                    "feet": {"load_bearing": [True] * 6},
                    "upright": 1.0,
                    "fault": None,
                }
            )
        payload = {
            "samples": rows,
            "parameters": {},
            **{
                key: "test"
                for key in ("body_hash", "model_hash", "adapter_hash", "graph_hash")
            },
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "teacher.json"
            path.write_text(json.dumps(payload))
            data = load_model(path)
            self.assertAlmostEqual(data["end_s"], 0.7)
            self.assertEqual(np.count_nonzero(data["window"]), 40)
            rows[120]["navigation"] = {"brake_request": True}
            path.write_text(json.dumps(payload))
            braking = load_model(path)
            self.assertAlmostEqual(braking["end_s"], 0.6)
            self.assertEqual(np.count_nonzero(braking["window"]), 20)

    def test_position_speed_uses_physical_forward_axis(self):
        t = np.arange(200) * 0.005
        positions = np.column_stack((t * 3, t * 0, t * 4))
        forward = np.tile([0.6, 0, 0.8], (len(t), 1))
        np.testing.assert_allclose(
            position_forward_speed(positions, forward, 0.005), 5.0, atol=1e-12
        )

    def test_csv_anatomical_names_override_column_order(self):
        rows = []
        expected = {"LF": 10, "LM": 20, "LH": 30, "RF": 40, "RM": 50, "RH": 60}
        for frame in range(4):
            row = {"frame_idx": frame, "video_id": 0, "Th_x": 0, "Th_y": 0, "Th_z": 0}
            for leg in ("LF", "LM", "LH", "RH", "RM", "RF"):
                joints = {
                    "ThC": [0, 0, 0],
                    "CTr": [0, 1, 0],
                    "FTi": [1, 1, 0],
                    "TiTa": [1, 2, 0],
                    "Claw": [expected[leg], 2, 0],
                }
                for joint, point in joints.items():
                    for axis, value in zip("xyz", point):
                        row[f"{leg}-{joint}_{axis}"] = value
            rows.append(row)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "reference.csv"
            with path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=rows[0])
                writer.writeheader()
                writer.writerows(rows)
            data = load_reference(path)
        np.testing.assert_allclose(
            data["ap"][0] * data["lengths"], [10, 20, 30, 40, 50, 60]
        )
        np.testing.assert_allclose(data["knee"], 90)

    def test_antiphase_and_period(self):
        data = oscillation()
        phi, events, eligible, _ = phase_and_events(data)
        stats = phase_summary(phi, eligible, data["dt"])
        for pair in ("lf-rf", "lm-rm", "lh-rh"):
            self.assertAlmostEqual(stats[pair]["mean_cycles"], 0.5, places=10)
            self.assertAlmostEqual(stats[pair]["resultant_length"], 1.0, places=10)
        for cycles in events:
            self.assertGreater(len(cycles), 10)
            self.assertTrue(np.allclose([c["period_s"] for c in cycles], 0.2))
            self.assertTrue(
                np.allclose([c["retraction_fraction"] for c in cycles], 0.5)
            )

    def test_inphase_is_not_antiphase(self):
        data = oscillation()
        data["ap"][:, 3] = data["ap"][:, 0]
        phi, _, eligible, _ = phase_and_events(data)
        stats = phase_summary(phi, eligible, data["dt"])["lf-rf"]
        self.assertAlmostEqual(stats["mean_antiphase_distance_cycles"], 0.5, places=10)

    def test_constant_leg_has_no_phase_or_cycles(self):
        data = oscillation()
        data["ap"][:, 0] = 0.123
        phi, events, _, _ = phase_and_events(data)
        self.assertTrue(np.isnan(phi[:, 0]).all())
        self.assertEqual(events[0], [])

    def test_no_events_bridge_excluded_bout(self):
        data = oscillation()
        data["moving"][180:240] = False
        phi, events, _, bouts = phase_and_events(data)
        self.assertEqual(len(bouts), 2)
        self.assertTrue(np.isnan(phi[180:240]).all())
        for cycles in events:
            self.assertTrue(
                all(c["next_aep_s"] < 0.9 or c["aep_s"] >= 1.2 for c in cycles)
            )

    def test_clock_gap_rejected(self):
        with self.assertRaisesRegex(ValueError, "Noncontiguous"):
            validate_clock([0, 0.005, 0.015, 0.020])

    def test_circular_average_wrap_and_geometry_rotation(self):
        s = circular_summary([0.99, 0.01])
        self.assertLess(min(s["mean_cycles"], 1 - s["mean_cycles"]), 1e-10)
        p = np.zeros((2, 6, 5, 3))
        p[:, :, 1] = [1, 0, 0]
        p[:, :, 3] = [0, 1, 0]
        angle = geometric_knee(p)
        rotation = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
        self.assertTrue(np.allclose(angle, 90))
        self.assertTrue(np.allclose(angle, geometric_knee(p @ rotation + 100)))


if __name__ == "__main__":
    unittest.main()
