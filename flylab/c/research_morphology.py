"""Explicit research-only extension of E6 sizes from the frozen E17 audit."""

import json
from pathlib import Path

import numpy as np

from .integrity import file_hash


def extend_sizes(baseline, measured, predictions, eligible, median):
    """Preserve measured sizes; only fill declared, unmeasured candidate cells."""
    base = np.asarray(baseline)
    measured = np.asarray(measured)
    predicted = np.asarray(predictions)
    eligible = np.asarray(eligible)
    if (
        base.ndim != 1
        or base.dtype != np.float32
        or measured.dtype != np.bool_
        or eligible.dtype != np.bool_
        or any(a.shape != base.shape for a in (measured, predicted, eligible))
    ):
        raise ValueError("Aligned float32 sizes and boolean provenance masks required")
    if not np.isfinite(base).all() or np.any(base <= 0):
        raise ValueError("Invalid baseline cell sizes")
    if isinstance(median, bool) or not np.isfinite(median) or median <= 0:
        raise ValueError("Positive reference area median required")
    fill = eligible & ~measured
    if not np.isfinite(predicted[fill]).all() or np.any(predicted[fill] <= 0):
        raise ValueError("Invalid eligible area prediction")
    sizes = base.copy()
    sizes[fill] = predicted[fill] / median
    if not np.isfinite(sizes).all() or np.any(sizes <= 0):
        raise ValueError("Invalid converted cell size")
    return sizes, fill


def load_morphology_sizes(graph, rate_reference, reference, expected_manifest_sha256):
    """Require pinned artifacts, matching IDs/areas, and supported regions.

    This returns a separately identified numerical hypothesis. It establishes no
    membrane-area unit conversion and is never selected by the product backend.
    """
    reference, rate_reference = Path(reference), Path(rate_reference)
    manifest_path = reference / "manifest.json"
    if (
        not isinstance(expected_manifest_sha256, str)
        or len(expected_manifest_sha256) != 64
        or file_hash(manifest_path) != expected_manifest_sha256
    ):
        raise ValueError("E17 manifest checksum mismatch")
    manifest = json.loads(manifest_path.read_text())
    expected_files = {
        "spec.json",
        "report.json",
        "alignment-and-predictions.npz",
        "runner.py",
    }
    if set(manifest["files"]) != expected_files:
        raise ValueError("Unexpected E17 artifact inventory")
    for name, expected in manifest["files"].items():
        if file_hash(reference / name) != expected:
            raise ValueError(f"E17 artifact checksum mismatch: {name}")
    spec = json.loads((reference / "spec.json").read_text())
    report = json.loads((reference / "report.json").read_text())
    prior = json.loads((rate_reference / "spec.json").read_text())
    if (
        spec["graph_hash"] != graph.hash
        or prior["graph_hash"] != graph.hash
        or report["status"] != "COMPLETE"
        or report["validity"] != "valid"
        or report["holdout_metrics"]["supported"] is not True
        or report["neural_parameters_modified"] is not False
        or spec["runner_sha256"] != manifest["files"]["runner.py"]
    ):
        raise ValueError("A valid matching frozen E17 audit is required")
    with np.load(rate_reference / "size-alignment.npz", allow_pickle=False) as data:
        baseline, measured, areas = (
            data[k].copy() for k in ("sizes", "measured", "area_um2")
        )
    with np.load(
        reference / "alignment-and-predictions.npz", allow_pickle=False
    ) as data:
        if not np.array_equal(data["root_ids"], [n["root_id"] for n in graph.nodes]):
            raise ValueError("E17 root order differs from the complete graph")
        if not np.array_equal(data["reference_area_um2"], areas, equal_nan=True):
            raise ValueError("E6/E17 original surface areas disagree")
        eligible = data["empirical_domain_eligible"].copy()
        regions = data["regions"].copy()
        known = np.array(
            [n["source_annotations"].get("region") or "unknown" for n in graph.nodes]
        )
        if not np.array_equal(regions, known):
            raise ValueError("E17 graph region provenance mismatch")
        validated = [
            name for name, row in report["regions"].items() if row["prediction_support"]
        ]
        expected_eligibility = data["within_training_feature_range"] & np.isin(
            regions, validated
        )
        if not np.array_equal(eligible, expected_eligibility):
            raise ValueError("Unsupported E17 region or feature-domain prediction")
        sizes, fill = extend_sizes(
            baseline,
            measured,
            data["predicted_area_um2"],
            eligible,
            prior["normalization_area_median_um2"],
        )
    audit = {
        "profile": "morphology_area",
        "e17_manifest_sha256": expected_manifest_sha256,
        "e6_alignment_sha256": file_hash(rate_reference / "size-alignment.npz"),
        "measured_size_preserved": bool(
            np.array_equal(sizes[measured], baseline[measured])
        ),
        "outside_fill_preserved": bool(np.array_equal(sizes[~fill], baseline[~fill])),
        "measured": int(measured.sum()),
        "predicted": int(fill.sum()),
        "assumed_baseline": int((~measured & ~fill).sum()),
        "fill_by_region": {
            name: int((fill & (regions == name)).sum()) for name in np.unique(regions)
        },
        "normalization_area_median_um2": prior["normalization_area_median_um2"],
        "coefficients_used_without_refitting": report["coefficients"],
        "surface_area_prediction_is_empirical": True,
        "electrical_size_validation": False,
    }
    if int(fill.sum()) != report["eligible_without_reference_area"]:
        raise ValueError("E17 eligible prediction count changed")
    return sizes, fill, audit
