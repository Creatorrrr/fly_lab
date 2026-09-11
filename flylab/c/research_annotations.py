"""Audited BANC NT sensitivity candidates, without modifying GraphStore.

A second source's ``verified`` label is evidence to compare, not authority to
overwrite existing curation. Only an unverified, anatomically concordant row
can change the sign of its outgoing contacts in this explicit rate candidate.
The common GLUT-inhibitory assumption is retained; this is not an NMJ model.
"""

import csv
import hashlib
import io
import re
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

from .graph import SIGN, normalized_nt
from .integrity import digest

FIELDS = (
    "pt_root_id",
    "cell_type",
    "side",
    "super_class",
    "peripheral_target_type",
    "neurotransmitter_verified",
)
POLICY = "banc888-unverified-concordant-nt-sign-v1"


def validate_reference(reference, expected_sha256):
    """Reject incomplete or malformed CLI/programmatic configuration early."""
    if reference is None and expected_sha256 is None:
        return
    if (
        not isinstance(reference, (str, Path))
        or not str(reference)
        or not isinstance(expected_sha256, str)
        or not re.fullmatch(r"[a-f0-9]{64}", expected_sha256)
    ):
        raise ValueError(
            "Annotation reference and explicit SHA256 are required together"
        )


def _reference_rows(reference, expected_sha256):
    payload = Path(reference).read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ValueError("Annotation source SHA256 mismatch")
    reader = csv.DictReader(io.StringIO(payload.decode("utf-8-sig"), newline=""))
    header = reader.fieldnames or []
    if len(set(header)) != len(header) or not set(FIELDS) <= set(header):
        raise ValueError("Annotation source columns missing or duplicated")
    rows = {}
    for row in reader:
        root = row["pt_root_id"]
        if (
            None in row
            or any(row[field] is None for field in FIELDS)
            or not isinstance(root, str)
            or not re.fullmatch(r"[1-9][0-9]{0,19}", root)
            or root in rows
        ):
            raise ValueError("Invalid or duplicate annotation root ID/row")
        rows[root] = row
    if not rows:
        raise ValueError("Empty annotation source")
    return rows


def rate_weights(graph, *, reference=None, expected_sha256=None):
    """Return the full contact-based rate matrix and optional provenance audit.

    Default arithmetic matches E6/E11 exactly. No roster, adjacency, anatomical
    count, original weight, or node annotation is changed, including for cells
    excluded from the candidate. IDs remain decimal strings throughout.
    """
    validate_reference(reference, expected_sha256)
    data = np.asarray(np.sign(graph.weights) * graph.counts * 0.03, np.float32)
    audit = None
    if reference is not None:
        if (graph.manifest.get("dataset_id"), graph.manifest.get("snapshot_id")) != (
            "flywire_banc",
            "888",
        ):
            raise ValueError("Annotation candidate requires BANC v888")
        rows = _reference_rows(reference, expected_sha256)
        matched, differences, proposals = 0, [], {}
        for index, node in enumerate(graph.nodes):
            row = rows.get(node["root_id"])
            if row is None:
                continue
            matched += 1
            proposed = normalized_nt(row["neurotransmitter_verified"])
            current = normalized_nt(node.get("nt_type"))
            if proposed not in SIGN or SIGN[proposed] == SIGN.get(current, 0):
                continue
            annotations = node.get("source_annotations", {})
            current_anatomy = {
                "cell_type": node.get("cell_type") or "",
                "side": node.get("soma_side") or "",
                "super_class": node.get("super_class") or "",
                "peripheral_target_type": annotations.get("peripheral_target_type")
                or "",
            }
            conflicts = {
                field: {"graph": value, "reference": row[field]}
                for field, value in current_anatomy.items()
                if value != row[field]
            }
            verified = annotations.get("neurotransmitter_verified")
            status = (
                "excluded_existing_curation"
                if verified
                else "excluded_anatomical_conflict"
                if conflicts
                else "candidate"
            )
            differences.append(
                {
                    "index": index,
                    "id": node["id"],
                    "cell_type": node.get("cell_type"),
                    "side": node.get("soma_side"),
                    "graph_nt": current,
                    "graph_verified_nt": verified,
                    "reference_verified_nt": row["neurotransmitter_verified"],
                    "reference_manc_match": row.get("manc_121_match_id"),
                    "anatomical_conflicts": conflicts,
                    "status": status,
                }
            )
            if status == "candidate":
                proposals[index] = SIGN[proposed]
        # W[post,pre]: only PRESYNAPTIC columns are affected. Retain explicit
        # zero entries and every original contact, including formerly masked NTs.
        candidate_signs = np.zeros(graph.n, np.int8)
        eligible = np.zeros(graph.n, bool)
        for index, sign in proposals.items():
            candidate_signs[index] = sign
            eligible[index] = True
        affected = eligible[graph.indices]
        baseline = data[affected].copy()
        data[affected] = (
            candidate_signs[graph.indices[affected]] * graph.counts[affected] * 0.03
        )
        audit = {
            "schema": "flylab.rate-annotation-candidate.v1",
            "policy": POLICY,
            "graph_hash": graph.hash,
            "reference_sha256": expected_sha256,
            "reference_rows": len(rows),
            "matched_rows": matched,
            "unmatched_reference_rows": len(rows) - matched,
            "node_count": graph.n,
            "pair_edge_count": len(data),
            "anatomical_synapse_count": int(graph.counts.sum()),
            "candidate_neuron_count": len(proposals),
            "candidate_outgoing_pair_count": int(affected.sum()),
            "changed_pair_count": int(np.count_nonzero(data[affected] != baseline)),
            "newly_unmasked_pair_count": int(np.count_nonzero(baseline == 0)),
            "sign_differences": differences,
            "rate_weights_sha256": hashlib.sha256(data.tobytes()).hexdigest(),
            "source_graph_unchanged": True,
            "weight_model": "presynaptic NT sign * contact count * 0.03 rate units",
            "biological_validation": False,
            "adoption_status": "RESEARCH_CANDIDATE",
        }
        audit["audit_hash"] = digest(audit)
    weights = csr_matrix(
        (data, graph.indices.copy(), graph.indptr.copy()), shape=(graph.n, graph.n)
    )
    return weights, audit
