"""DNg100 soma side must not silently become its VNC motor target side."""

import copy
import unittest

from tools.probe_c_banc_descending import descending_targets


class DescendingTargetTests(unittest.TestCase):
    def setUp(self):
        self.rows = [
            {
                "pt_root_id": "720575941626500746",
                "cell_type": "DNg100",
                "side": "left",
                "manc_121_match_id": "id:10339",
            },
            {
                "pt_root_id": "720575941500851362",
                "cell_type": "DNg100",
                "side": "right",
                "manc_121_match_id": "id:10093",
            },
        ]

    def test_target_side_is_independent_of_soma_and_array_order(self):
        targets = descending_targets(self.rows)
        self.assertEqual(targets["left"]["index"], 1)
        self.assertEqual(targets["left"]["source_side"], "right")
        self.assertEqual(targets["right"]["index"], 0)
        self.assertEqual(targets["right"]["manc_match"], "id:10339")
        self.assertEqual(descending_targets(self.rows[::-1])["left"]["index"], 0)

    def test_conflicting_source_identity_is_rejected(self):
        for key, value in (
            ("side", "left"),
            ("cell_type", "DNa02"),
            ("manc_121_match_id", "id:10339"),
        ):
            rows = copy.deepcopy(self.rows)
            rows[1][key] = value
            with (
                self.subTest(key=key),
                self.assertRaisesRegex(ValueError, "identity mismatch"),
            ):
                descending_targets(rows)

    def test_missing_or_duplicate_exact_id_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Missing"):
            descending_targets(self.rows[:1])
        with self.assertRaisesRegex(ValueError, "Ambiguous"):
            descending_targets(self.rows + self.rows[:1])
