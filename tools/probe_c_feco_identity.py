"""Build and apply an explicit, unvalidated FeCO homology input candidate.

The reviewed PNG root and the Lee annotation root are joined as strings.
Annotation row IDs and cell_ids_v2 IDs belong to different namespaces.
This utility never promotes a morphological match to measured BANC tuning.
"""

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

import numpy as np

FEATURES = {"claw_ext": "claw_negative", "claw_flx": "claw_positive"}
OLD_FEATURES = {"SNpp50": "claw_positive", "SNpp51": "claw_negative"}
SCHEMA = "flylab.feco-homology-input-candidate.v1"


def build_candidate(port_audit, reviewed, directory, annotations):
    """Keep only unique exact-root annotations; retain exclusions explicitly."""
    ports = set(port_audit["external_claw_root_ids"])
    metadata = {row["root_888"]: row for row in port_audit["root_alignment"]}
    if (
        len(metadata) != len(port_audit["root_alignment"])
        or not ports <= metadata.keys()
    ):
        raise ValueError("Duplicate or absent current port metadata")
    assignments = []
    exclusions = []
    for root in sorted(ports):
        old = metadata[root]
        files = [
            row
            for row in directory
            if re.search(r"_root_id_(\d+)_", row["filename"])
            and re.search(r"_root_id_(\d+)_", row["filename"])[1] == root
        ]
        decisions = [
            row for row in reviewed if row["query_id"] == root and row["valid"] == "t"
        ]
        short_ids = {row["match_id"] for row in decisions}
        reason = None
        if len(files) != 1:
            reason = "reviewed_filename_missing_or_ambiguous"
        elif not old["fanc_match"] or short_ids != {old["fanc_match"]}:
            reason = "reviewed_short_id_missing_or_conflicting"
        if reason:
            exclusions.append({"banc_root": root, "reason": reason})
            continue
        match = re.search(r"_hit_id_(m?\d+)_", files[0]["filename"])
        if match is None or match[1].startswith("m"):
            exclusions.append({"banc_root": root, "reason": "missing_or_mirrored_root"})
            continue
        fanc_root = match[1]
        # Deliberately never join annotation['id'] to old['fanc_match'].
        functional = [
            row
            for row in annotations
            if row["pt_root_id"] == fanc_root and row["valid"] == "t"
        ]
        if len(functional) != 1 or functional[0]["cell_type"] not in FEATURES:
            exclusions.append(
                {"banc_root": root, "reason": "functional_root_missing_or_ambiguous"}
            )
            continue
        row = functional[0]
        assignments.append(
            {
                "banc_root": root,
                "banc_type": old["cell_type"],
                "previous_feature": OLD_FEATURES[old["cell_type"]],
                "candidate_feature": FEATURES[row["cell_type"]],
                "fanc_cell_ids_v2_id": old["fanc_match"],
                "fanc_root": fanc_root,
                "lee_annotation_row_id": row["id"],
                "lee_subtype": row["cell_type"],
                "reviewed_filename": files[0]["filename"],
                "review_confidence": None,
                "reviewed_rows": decisions,
            }
        )
    return {
        "schema": SCHEMA,
        "evidence_status": "morphological_homology_input_hypothesis",
        "biological_validation": False,
        "reference_segmentation_version": "not established by archived filename",
        "review_confidence_note": "ZIP flattens folders; leading numeral is not review confidence",
        "graph_hash": port_audit["graph_hash"],
        "port_count": len(ports),
        "assignments": assignments,
        "excluded": exclusions,
        "previous_features": {
            root: OLD_FEATURES[metadata[root]["cell_type"]] for root in sorted(ports)
        },
    }


def remap_inputs(root_ids, input_indices, input_values, candidate):
    """Change only declared cells, using saved curves for the requested channel.

    All pre-existing cells in a channel must have the same saved input curve.
    This keeps the previously measured pose, amplitude, delay, and removal
    timing fixed, and avoids recomputing a superficially matched stimulus.
    """
    roots = [str(value) for value in root_ids]
    indices = np.asarray(input_indices)
    values = np.asarray(input_values)
    if (
        candidate.get("schema") != SCHEMA
        or candidate.get("biological_validation") is not False
        or len(set(roots)) != len(roots)
        or indices.ndim != 1
        or indices.dtype.kind not in "iu"
        or len(set(indices.tolist())) != len(indices)
        or (indices < 0).any()
        or (indices >= len(roots)).any()
        or values.ndim != 2
        or values.shape[1] != len(indices)
        or not np.isfinite(values).all()
        or (values < 0).any()
    ):
        raise ValueError("Invalid candidate or saved input alignment")
    columns = {roots[int(index)]: col for col, index in enumerate(indices)}
    previous = candidate["previous_features"]
    if (
        len(previous) != candidate["port_count"]
        or not previous.keys() <= columns.keys()
    ):
        raise ValueError("Saved input is missing a declared claw port")
    curves = {}
    for root, feature in previous.items():
        if feature not in FEATURES.values():
            raise ValueError("Unrecognized previous claw feature")
        curve = values[:, columns[root]]
        if feature in curves and not np.array_equal(curves[feature], curve):
            raise ValueError("Saved input varies within the claimed feature")
        curves[feature] = curve.copy()
    replacements = {}
    for row in candidate["assignments"]:
        root = row["banc_root"]
        feature = row["candidate_feature"]
        if (
            root in replacements
            or root not in previous
            or row["previous_feature"] != previous[root]
            or FEATURES.get(row["lee_subtype"]) != feature
            or feature not in curves
        ):
            raise ValueError("Ambiguous or inconsistent candidate assignment")
        replacements[root] = feature
    result = values.copy()
    for root, feature in replacements.items():
        result[:, columns[root]] = curves[feature]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity", required=True, type=Path)
    parser.add_argument("--previous-audit", required=True, type=Path)
    args = parser.parse_args()
    paths = {
        "ports": args.previous_audit / "current-port-audit.json",
        "annotations": args.previous_audit / "feco_annotation_table.csv",
        "reviewed": args.identity / "banc_fanc_reviewed_matches.csv",
        "directory": args.identity / "zip-directory.csv",
    }
    source = {}
    for key, path in paths.items():
        if path.suffix == ".csv":
            with path.open(encoding="utf8", newline="") as file:
                source[key] = list(csv.DictReader(file))
        else:
            source[key] = json.loads(path.read_text(encoding="utf8"))
    candidate = build_candidate(
        source["ports"], source["reviewed"], source["directory"], source["annotations"]
    )
    candidate["sources"] = {
        key: {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for key, path in paths.items()
    }
    candidate["builder_sha256"] = hashlib.sha256(
        Path(__file__).read_bytes()
    ).hexdigest()
    out = args.identity / "claw-input-candidate.json"
    with out.open("x", encoding="utf8") as file:
        json.dump(candidate, file, indent=2, allow_nan=False)
    print(
        json.dumps(
            {
                "ports": candidate["port_count"],
                "assignments": len(candidate["assignments"]),
                "changed": sum(
                    row["candidate_feature"] != row["previous_feature"]
                    for row in candidate["assignments"]
                ),
                "sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
            }
        )
    )


if __name__ == "__main__":
    main()
