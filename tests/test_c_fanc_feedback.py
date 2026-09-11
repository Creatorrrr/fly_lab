import unittest

import numpy as np

from tools.probe_c_fanc_feedback import extend_claw_graph


class FancFeedbackTests(unittest.TestCase):
    def test_exact_contact_identity_preserves_feedback_and_outside_edges(self):
        core = np.array([[0.0, 2.0], [-3.0, 0.0]])

        def edge(identity, pre, post):
            return {
                "id": identity,
                "valid": "t",
                "pre_pt_root_id": pre,
                "post_pt_root_id": post,
            }

        rows = [
            edge("1", "30", "10"),
            edge("1", "30", "10"),
            edge("2", "30", "10"),
            edge("3", "20", "30"),
            edge("4", "30", "90"),
            edge("5", "80", "30"),
        ]
        matrix, roots, retained, outside = extend_claw_graph(
            core, ["10", "20"], [1, -1], ["30"], rows
        )
        np.testing.assert_array_equal(matrix[:2, :2], core)
        self.assertEqual(roots, ["10", "20", "30"])
        self.assertEqual(matrix[2, 0], 2)
        self.assertEqual(matrix[1, 2], -1)
        self.assertEqual(len(retained), 3)
        self.assertEqual({r["synapse_id"] for r in outside}, {"4", "5"})
        self.assertEqual(np.count_nonzero(matrix), 4)
        with self.assertRaises(ValueError):
            extend_claw_graph(
                core, ["10", "20"], [1, -1], ["30"], rows + [edge("1", "30", "20")]
            )
        with self.assertRaises(ValueError):
            extend_claw_graph(core, ["10", "20"], [1, 1], ["30"], rows)
        with self.assertRaises(ValueError):
            extend_claw_graph(core, ["10", "20"], [1, -1], ["20"], rows)


if __name__ == "__main__":
    unittest.main()
