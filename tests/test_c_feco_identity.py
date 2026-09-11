import copy
import unittest

import numpy as np

from tools.probe_c_feco_identity import SCHEMA, build_candidate, remap_inputs


class FeCOIdentityTests(unittest.TestCase):
    def test_annotation_row_number_never_substitutes_for_a_root_match(self):
        port = {
            "graph_hash": "test",
            "external_claw_root_ids": ["100"],
            "root_alignment": [
                {"root_888": "100", "cell_type": "SNpp50", "fanc_match": "17"}
            ],
        }
        review = [{"query_id": "100", "match_id": "17", "valid": "t"}]
        directory = [{"filename": "1_root_id_100_hit_id_900_tail.png"}]
        annotations = [
            {"id": "17", "pt_root_id": "999", "valid": "t", "cell_type": "claw_flx"},
            {"id": "3", "pt_root_id": "900", "valid": "t", "cell_type": "claw_ext"},
        ]
        candidate = build_candidate(port, review, directory, annotations)
        self.assertFalse(candidate["biological_validation"])
        self.assertEqual(candidate["assignments"][0]["lee_annotation_row_id"], "3")
        self.assertEqual(
            candidate["assignments"][0]["candidate_feature"], "claw_negative"
        )
        for bad in (
            [annotations[0]],
            [annotations[1], {**annotations[1], "cell_type": "claw_flx"}],
            [{**annotations[1], "valid": "f"}],
        ):
            self.assertEqual(
                build_candidate(port, review, directory, bad)["assignments"], []
            )
        mirrored = [{"filename": "1_root_id_100_hit_id_m900_tail.png"}]
        self.assertEqual(
            build_candidate(port, review, mirrored, annotations)["assignments"], []
        )
        conflict = review + [{"query_id": "100", "match_id": "18", "valid": "t"}]
        self.assertEqual(
            build_candidate(port, conflict, directory, annotations)["assignments"], []
        )

    def test_remap_changes_only_declared_cells_and_preserves_saved_timing(self):
        candidate = {
            "schema": SCHEMA,
            "biological_validation": False,
            "port_count": 3,
            "previous_features": {
                "10": "claw_positive",
                "20": "claw_negative",
                "30": "claw_positive",
            },
            "assignments": [
                {
                    "banc_root": "10",
                    "previous_feature": "claw_positive",
                    "candidate_feature": "claw_negative",
                    "lee_subtype": "claw_ext",
                }
            ],
        }
        # Deliberately shuffled graph/input order, with an unrelated input.
        roots = ["20", "40", "10", "30"]
        indices = np.array([1, 3, 0, 2])
        values = np.array([[7, 1, 3, 1], [8, 2, 4, 2], [0, 0, 0, 0]], dtype=np.float32)
        original = values.copy()
        actual = remap_inputs(roots, indices, values, candidate)
        np.testing.assert_array_equal(actual[:, 3], values[:, 2])
        np.testing.assert_array_equal(actual[:, :3], values[:, :3])
        np.testing.assert_array_equal(values, original)
        np.testing.assert_array_equal(actual[-1], 0)
        for broken in ("duplicate", "wrong_previous", "wrong_subtype", "missing"):
            bad = copy.deepcopy(candidate)
            if broken == "duplicate":
                bad["assignments"].append(bad["assignments"][0].copy())
            elif broken == "wrong_previous":
                bad["assignments"][0]["previous_feature"] = "claw_negative"
            elif broken == "wrong_subtype":
                bad["assignments"][0]["lee_subtype"] = "claw_flx"
            else:
                bad["previous_features"]["50"] = "claw_positive"
                bad["port_count"] += 1
            with self.assertRaises(ValueError):
                remap_inputs(roots, indices, values, bad)
        values[0, 3] = 2
        with self.assertRaises(ValueError):
            remap_inputs(roots, indices, values, candidate)


if __name__ == "__main__":
    unittest.main()
