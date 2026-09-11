import copy
import csv
import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np

from flylab.c.graph import GraphStore
from flylab.c.research_annotations import FIELDS, rate_weights


class ResearchAnnotationContract(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.source = Path(self.temp.name) / "reference.csv"
        nodes = []
        for i, nt in enumerate(("ACH", "GABA", "", "ACH", "ACH"), start=1):
            nodes.append(
                {
                    "id": f"flywire:banc:888:{i}",
                    "root_id": str(i),
                    "nt_type": nt,
                    "cell_type": f"cell{i}",
                    "soma_side": "left",
                    "super_class": "motor",
                    "source_annotations": {
                        "peripheral_target_type": "tibia_flexor_muscle",
                        "neurotransmitter_verified": "gaba" if i == 2 else None,
                    },
                }
            )
        # Explicit outgoing AND incoming contacts distinguish column/row edits.
        self.graph = GraphStore.from_edges(
            nodes,
            [0, 1, 2, 3, 4, 4],
            [1, 2, 3, 4, 0, 2],
            [2, 3, 4, 5, 6, 7],
            metadata={"dataset_id": "flywire_banc", "snapshot_id": "888"},
            unknown_policy="mask_zero",
        )
        self.rows = [
            dict(
                zip(
                    FIELDS,
                    (
                        str(i),
                        f"cell{i}" if i != 4 else "conflicting_cell",
                        "left",
                        "motor",
                        "tibia_flexor_muscle",
                        "acetylcholine" if i == 2 else "glutamate",
                    ),
                )
            )
            for i in range(1, 5)
        ]

    def write_source(self):
        with self.source.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, FIELDS)
            writer.writeheader()
            writer.writerows(self.rows)
        return hashlib.sha256(self.source.read_bytes()).hexdigest()

    def test_candidate_changes_only_eligible_outgoing_columns(self):
        checksum = self.write_source()
        g = self.graph
        before_nodes = copy.deepcopy(g.nodes)
        before_weights = g.weights.copy()
        baseline, no_audit = rate_weights(g)
        candidate, audit = rate_weights(
            g, reference=self.source, expected_sha256=checksum
        )
        expected = baseline.toarray()
        expected[1, 0] = -0.06  # unverified ACH -> GLUT
        expected[3, 2] = -0.12  # previously masked, contact remains present
        np.testing.assert_array_equal(candidate.toarray(), expected)
        self.assertIsNone(no_audit)
        self.assertEqual(audit["candidate_neuron_count"], 2)
        self.assertEqual(audit["changed_pair_count"], 2)
        self.assertEqual(audit["newly_unmasked_pair_count"], 1)
        self.assertEqual(
            [row["status"] for row in audit["sign_differences"]],
            [
                "candidate",
                "excluded_existing_curation",
                "candidate",
                "excluded_anatomical_conflict",
            ],
        )
        np.testing.assert_array_equal(candidate.indptr, baseline.indptr)
        np.testing.assert_array_equal(candidate.indices, baseline.indices)
        np.testing.assert_array_equal(g.weights, before_weights)
        self.assertEqual(g.nodes, before_nodes)
        self.assertFalse(g.weights.flags.writeable)
        self.assertFalse(audit["biological_validation"])

    def test_bad_sources_are_rejected_without_graph_mutation(self):
        g = self.graph
        before = g.weights.copy()
        checksum = self.write_source()
        for options in (
            {"reference": self.source},
            {"expected_sha256": checksum},
            {"reference": self.source, "expected_sha256": "0" * 64},
        ):
            with self.assertRaises(ValueError):
                rate_weights(g, **options)
        self.rows.append(self.rows[0])
        checksum = self.write_source()
        with self.assertRaisesRegex(ValueError, "duplicate"):
            rate_weights(g, reference=self.source, expected_sha256=checksum)
        np.testing.assert_array_equal(g.weights, before)

    def test_mixed_reference_nt_and_other_specimens_are_not_inferred(self):
        self.rows[0]["neurotransmitter_verified"] = "acetylcholine,glutamate"
        checksum = self.write_source()
        _, audit = rate_weights(
            self.graph, reference=self.source, expected_sha256=checksum
        )
        self.assertEqual(audit["candidate_neuron_count"], 1)
        self.graph.manifest["snapshot_id"] = "626"
        with self.assertRaisesRegex(ValueError, "v888"):
            rate_weights(self.graph, reference=self.source, expected_sha256=checksum)


if __name__ == "__main__":
    unittest.main()
