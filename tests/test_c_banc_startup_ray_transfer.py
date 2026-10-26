"""Generated finite CPU fixtures only. R1 Source proposal; tests NOT_RUN.

ROOT adopts the reader under tools/analyze_c_banc_m3_startup_ray_transfer.py
before running this proposed unittest module. No saved Original inputs are used.
"""
import importlib
import io
import tempfile
import unittest
import zipfile
from pathlib import Path


class StartupRayTransferMathTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import numpy as np
        cls.np = np
        cls.reader = importlib.import_module("tools.analyze_c_banc_m3_startup_ray_transfer")

    def test_R_exact_label_windows_and_fixed_denominator(self):
        np, reader = self.np, self.reader
        command = np.repeat(np.arange(301, dtype=np.float64)[:, None], 42, axis=1)
        observed = reader.r_windows(command, np)
        np.testing.assert_array_equal(observed[:, :, 0], [[101, 102, 103, 104], [165, 166, 167, 168],
                                                        [229, 230, 231, 232], [293, 294, 295, 296]])
        normalization = np.arange(1, 7, dtype=np.float64)
        target = observed.copy()
        perturbed = observed.copy()
        perturbed[:, :, :7] += 2
        # Only first leg differs: four controls * seven coordinates * 2^2 / 672.
        np.testing.assert_allclose(reader.r_score(perturbed, target, normalization, np), np.repeat(1 / 6, 4), atol=1e-14)
        np.testing.assert_array_equal(reader.r_score(target, target, normalization, np), np.zeros(4))

    def test_interval_contact_requires_both_endpoints_and_excludes_transition(self):
        np, reader = self.np, self.reader
        ap = np.repeat(np.arange(5, dtype=np.float64)[:, None], 6, axis=1)
        contact = np.repeat(np.array([False, True, True, False, False])[:, None], 6, axis=1)
        result = reader.interval_ap(ap, contact, np.ones(5, dtype=bool), np)["lf"]
        self.assertEqual(result["loaded"]["intervals"], 1)
        self.assertEqual(result["unloaded"]["intervals"], 1)
        self.assertEqual(result["transition"]["intervals"], 2)
        self.assertEqual(result["loaded"]["sum_leg_lengths"], 1)
        self.assertEqual(result["transition"]["sum_leg_lengths"], 2)
        inside = np.array([False, True, True, True, False])
        self.assertEqual(reader.interval_ap(ap, contact, inside, np)["lf"]["unloaded"]["intervals"], 0)

    def test_native_slip_weighting_invalid_and_unavailable_are_not_zero(self):
        np, reader = self.np, self.reader
        native = {"time_s": np.array([.49, .5, .6, 1.5]),
                  "normal_load_bw": np.zeros((4, 6)), "slip_mm_s": np.full((4, 6), np.nan),
                  "slip_valid": np.zeros((4, 6), dtype=bool)}
        native["normal_load_bw"][1:3, 0] = [.02, .06]
        native["slip_mm_s"][1:3, 0] = [1, 3]
        native["slip_valid"][1:3, 0] = True
        result = reader.slip_summary(native, np)
        self.assertAlmostEqual(result["all"]["value_mm_s"], 2.5)
        self.assertEqual(result["all"]["supported_count"], 2)
        self.assertAlmostEqual(result["all"]["load_weight_denominator_bw"], .08)
        self.assertIsNone(result["per_leg"]["lm"]["value_mm_s"])
        native["slip_valid"][2, 0] = False
        invalid = reader.slip_summary(native, np)["all"]
        self.assertFalse(invalid["available"])
        self.assertIsNone(invalid["value_mm_s"])
        self.assertEqual(invalid["invalid_supported_count"], 1)
        self.assertAlmostEqual(invalid["valid_load_weight_denominator_bw"], .02)

    def test_temporal_canonical200_and_strict199_have_distinct_successor(self):
        np, reader = self.np, self.reader
        command = np.repeat(np.arange(301, dtype=np.float64)[:, None], 42, axis=1)
        previous, successor = reader.pair_slices(command)
        self.assertEqual(previous.shape, (200, 42))
        self.assertEqual((previous[0, 0], previous[-1, 0], successor[-1, 0]), (100, 299, 300))
        previous, successor = reader.pair_slices(command, True)
        self.assertEqual(previous.shape, (199, 42))
        self.assertEqual((previous[-1, 0], successor[-1, 0]), (298, 299))
        np.testing.assert_array_equal(successor - previous, np.ones((199, 42)))

    def test_geometry_projects_recorded_body_up_and_retains_world_mm(self):
        np, reader = self.np, self.reader
        # Body-forward world-z; body-up world-y; lateral world-x, right-handed.
        basis = [[0., 1., 0.], [0., 0., 1.], [1., 0., 0.]]
        points = [[0., 2., float(i)] for i in range(8)]
        rows = [{"body": {"basis": basis, "position": [0., 1., 0.],
                 "legs": {leg: points for leg in reader.LEGS}}} for _ in range(301)]
        data = {"window": np.arange(301) >= 100, "lengths": np.repeat(7., 6),
                "ap": np.ones((301, 6)), "knee": np.repeat(90., 301 * 6).reshape(301, 6),
                "contact": np.zeros((301, 6), dtype=bool)}
        data["window"][-1] = False
        result = reader.geometry_summary({"samples": rows}, data, np)["per_leg"]["lf"]
        self.assertAlmostEqual(result["height_body_up_leg_lengths"]["median"], 1 / 7)
        self.assertEqual(result["height_world_y_mm"]["median"], 2)
        self.assertEqual(result["knee_interior_deg"]["median"], 90)
        self.assertEqual(result["knee_p95_minus_p05_deg"], 0)

    def test_selected_header_rejects_large_declared_shape_before_array_load(self):
        np, reader = self.np, self.reader
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            header = io.BytesIO()
            np.lib.format.write_array_header_1_0(header, {"descr": "<f8", "fortran_order": False, "shape": (10 ** 9,)})
            with zipfile.ZipFile(root / "bad.npz", "w") as archive:
                archive.writestr("command.npy", header.getvalue())
            local = reader.Reader.__new__(reader.Reader)
            local.root, local.check = root, lambda label: None
            local.specs, local.members, local.expanded = {"fixture": {"path": "bad.npz"}}, {}, 0
            with self.assertRaisesRegex(ValueError, "header/payload"):
                local.numeric("fixture", {"command": ((301, 42), "float64")}, np)


if __name__ == "__main__":
    unittest.main()
