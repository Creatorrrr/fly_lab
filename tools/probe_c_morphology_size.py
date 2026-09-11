"""E17: held-out empirical area prediction; never modifies neural parameters."""

import argparse
import csv
import hashlib
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flylab.c.graph import GraphStore
from flylab.c.integrity import file_hash, write_json

RAW_SHA256 = "86ccf5df0c67419f8c5f43e93a7ed38d23a080e9f7fde26737290252f3780098"
AREA_SHA256 = "6f87bf62227e160754523418bfbac35fc8971a90e4ed54574bf769b08332e285"
GRAPH_HASH = "9bd8468f5e5e40c3e6daaffde9f444466825ba8e69189f1e347568bff85920ff"


def bucket(root, purpose):
    return (
        int.from_bytes(
            hashlib.sha256(f"E17-{purpose}-v1:{root}".encode("ascii")).digest()[:8],
            "big",
        )
        % 5
    )


def validate_ids(ids):
    if any(
        not isinstance(s, str) or not re.fullmatch(r"[1-9][0-9]{0,19}", s) for s in ids
    ):
        raise ValueError("Root IDs must be exact decimal strings")
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate root ID; refuse ambiguous morphology join")


def positive(values):
    return np.isfinite(values) & (values > 0)


def metrics(log_observed, log_predicted, log_baseline):
    error = log_predicted - log_observed
    rmse = float(np.sqrt(np.mean(error**2)))
    baseline_rmse = float(np.sqrt(np.mean((log_baseline - log_observed) ** 2)))
    within = float(np.mean(np.abs(error) <= np.log(2)))
    return {
        "n": len(error),
        "log_rmse": rmse,
        "baseline_log_rmse": baseline_rmse,
        "log_rmse_ratio": rmse / baseline_rmse if baseline_rmse > 1e-12 else None,
        "within_factor_2": within,
        "median_multiplicative_error": float(np.exp(np.median(np.abs(error)))),
        "supported": baseline_rmse > 1e-12
        and rmse <= 0.8 * baseline_rmse
        and within >= 0.9,
    }


def fit_proxy(ids, length_um, volume_nm3, area_um2, regions):
    """Fit train-only models; return all splits and predictions for audit.

    The held-out area array is never read during model selection or fitting.
    Missing morphology/labels remain explicit, including unlabeled regions.
    """
    validate_ids(ids)
    n = len(ids)
    length, volume, area = (
        np.asarray(v, dtype=np.float64) for v in (length_um, volume_nm3, area_um2)
    )
    regions = np.asarray(regions, dtype=str)
    if any(v.shape != (n,) for v in (length, volume, area, regions)):
        raise ValueError("Aligned morphology columns must match the root roster")
    morphology_valid = positive(length) & positive(volume)
    labeled = morphology_valid & positive(area)
    partition = np.array([bucket(s, "holdout") for s in ids], dtype=np.int8)
    folds = np.array([bucket(s, "cv") for s in ids], dtype=np.int8)
    train = labeled & (partition != 0)
    holdout = labeled & (partition == 0)
    if train.sum() < 1000 or holdout.sum() < 300:
        raise ValueError("E17 minimum of 1000 train / 300 holdout labels not met")
    # Compute valid morphology features for labeled and unlabeled cells together;
    # use only training rows to select/fix the model and its feature domain.
    features = np.full((n, 3), np.nan)
    features[morphology_valid] = np.column_stack(
        (
            np.ones(morphology_valid.sum()),
            np.log(length[morphology_valid]),
            np.log(volume[morphology_valid] / 1e9),
        )
    )
    log_area = np.full(n, np.nan)
    log_area[labeled] = np.log(area[labeled])
    candidates = {"log_length": 2, "log_length_volume": 3}
    cv = {}
    for name, columns in candidates.items():
        errors = []
        for fold in range(5):
            fitting = train & (folds != fold)
            testing = train & (folds == fold)
            if not fitting.any() or not testing.any():
                raise ValueError("Empty training CV fold")
            coefficients, _, rank, _ = np.linalg.lstsq(
                features[fitting, :columns], log_area[fitting], rcond=None
            )
            if rank != columns:
                raise ValueError("Rank deficient training morphology design")
            residual = features[testing, :columns] @ coefficients - log_area[testing]
            errors.append(
                {
                    "fold": fold,
                    "n": int(testing.sum()),
                    "log_mse": float(np.mean(residual**2)),
                }
            )
        cv[name] = {
            "folds": errors,
            "mean_log_mse": float(np.mean([r["log_mse"] for r in errors])),
        }
    selected = min(candidates, key=lambda name: cv[name]["mean_log_mse"])
    columns = candidates[selected]
    coefficients, _, rank, _ = np.linalg.lstsq(
        features[train, :columns], log_area[train], rcond=None
    )
    if rank != columns:
        raise ValueError("Rank deficient final training design")
    predicted_log = np.full(n, np.nan)
    predicted_log[morphology_valid] = (
        features[morphology_valid, :columns] @ coefficients
    )
    predicted = np.full(n, np.nan)
    predicted[morphology_valid] = np.exp(predicted_log[morphology_valid])
    if not np.all(positive(predicted[morphology_valid])):
        raise ValueError("Non-finite or nonpositive size prediction")
    baseline_area = float(np.median(area[train]))
    log_baseline = np.log(baseline_area)
    lower = features[train, 1:].min(axis=0)
    upper = features[train, 1:].max(axis=0)
    in_domain = morphology_valid & np.all(
        (features[:, 1:] >= lower) & (features[:, 1:] <= upper), axis=1
    )
    global_metrics = metrics(log_area[holdout], predicted_log[holdout], log_baseline)
    regional = {}
    validated_regions = []
    for region in sorted(set(regions)):
        members = regions == region
        test = members & holdout
        enough = int((members & train).sum()) >= 100 and int(test.sum()) >= 30
        score = (
            metrics(log_area[test], predicted_log[test], log_baseline)
            if test.any()
            else None
        )
        supported = bool(enough and score["supported"])
        if supported:
            validated_regions.append(region)
        regional[region] = {
            "graph_cells": int(members.sum()),
            "morphology_available": int((members & morphology_valid).sum()),
            "labeled": int((members & labeled).sum()),
            "train": int((members & train).sum()),
            "holdout": int(test.sum()),
            "within_training_feature_range": int((members & in_domain).sum()),
            "holdout_metrics": score,
            "sufficient_sample_count": enough,
            "prediction_support": supported,
        }
    eligible = (
        in_domain & np.isin(regions, validated_regions) & global_metrics["supported"]
    )
    arrays = {
        "root_ids": np.asarray(ids),
        "regions": regions,
        "length_um": length,
        "volume_nm3": volume,
        "reference_area_um2": area,
        "predicted_area_um2": predicted,
        "morphology_valid": morphology_valid,
        "labeled": labeled,
        "partition_bucket": partition,
        "training_cv_fold": folds,
        "train": train,
        "holdout": holdout,
        "within_training_feature_range": in_domain,
        "empirical_domain_eligible": eligible,
    }
    report = {
        "status": "COMPLETE",
        "validity": "valid",
        "n": n,
        "morphology_available": int(morphology_valid.sum()),
        "reference_areas_available": int(positive(area).sum()),
        "labeled_intersection": int(labeled.sum()),
        "train": int(train.sum()),
        "holdout": int(holdout.sum()),
        "cv": cv,
        "selected_model": selected,
        "coefficients": coefficients.tolist(),
        "coefficient_order": ["intercept", "log_length_um", "log_volume_um3"][:columns],
        "constant_baseline_area_um2": baseline_area,
        "training_log_feature_bounds": {
            "lower": lower.tolist(),
            "upper": upper.tolist(),
        },
        "holdout_metrics": global_metrics,
        "regions": regional,
        "empirical_domain_eligible": int(eligible.sum()),
        "eligible_without_reference_area": int((eligible & ~positive(area)).sum()),
        "physiological_area_conversion_validated": False,
        "neural_parameters_modified": False,
        "implementation_adoption": "not_applicable",
        "F3_F6": "fail",
    }
    return report, arrays


def run(graph_path, raw_path, area_path, out):
    from pyarrow import feather

    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    (out / "runner.py").write_bytes(Path(__file__).read_bytes())
    spec = {
        "test": "E17 empirical morphology size proxy",
        "graph_hash": GRAPH_HASH,
        "sources": {"raw": str(raw_path), "area": str(area_path)},
        "raw_sha256": RAW_SHA256,
        "area_sha256": AREA_SHA256,
        "runner_sha256": file_hash(Path(__file__)),
        "goal_plan_sha256": file_hash(Path("GOAL_PLAN.md")),
        "split": "SHA256(E17-holdout-v1:<root>) first 8 bytes big-endian mod 5 == 0 held out",
        "cv_split": "Train only, SHA256(E17-cv-v1:<root>) first 8 bytes big-endian mod 5",
        "models": [
            "log_area ~ 1 + log_length",
            "log_area ~ 1 + log_length + log_volume_um3",
        ],
        "selection": "minimum unweighted mean of five training fold log-MSEs",
        "criteria": {
            "maximum_baseline_log_rmse_ratio": 0.8,
            "minimum_within_factor_2": 0.9,
        },
        "wall_limit_s": 180,
        "neural_model_seconds": 0,
        "physics_model_seconds": 0,
        "biological_validation": False,
    }
    write_json(out / "spec.json", spec)
    if file_hash(raw_path) != RAW_SHA256 or file_hash(area_path) != AREA_SHA256:
        raise ValueError("Morphology/area source checksum mismatch")
    graph = GraphStore.load(graph_path)
    if graph.hash != GRAPH_HASH:
        raise ValueError("Fixed BANC graph required")
    columns = feather.read_table(
        raw_path, columns=["root_888", "l2_cable_length_um", "volume_nm3", "region"]
    ).to_pydict()
    raw_ids = [str(value) for value in columns["root_888"]]
    validate_ids(raw_ids)
    raw_index = {root: i for i, root in enumerate(raw_ids)}
    with area_path.open(encoding="utf8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or len(set(reader.fieldnames)) != len(
            reader.fieldnames
        ):
            raise ValueError("Duplicate or missing area table headers")
        rows = list(reader)
    validate_ids([row["pt_root_id"] for row in rows])
    area_by_id = {}
    for row in rows:
        try:
            area_by_id[row["pt_root_id"]] = float(row["surf_area_um2"])
        except ValueError:
            area_by_id[row["pt_root_id"]] = np.nan
    ids = [node["root_id"] for node in graph.nodes]
    if any(root not in raw_index for root in ids):
        raise ValueError("Graph root missing from pinned raw morphology roster")
    source_rows = np.array([raw_index[root] for root in ids], dtype=np.int64)
    length = np.asarray(columns["l2_cable_length_um"], dtype=np.float64)[source_rows]
    volume = np.asarray(columns["volume_nm3"], dtype=np.float64)[source_rows]
    regions = [columns["region"][index] or "unknown" for index in source_rows]
    area = np.array([area_by_id.get(root, np.nan) for root in ids])
    report, arrays = fit_proxy(ids, length, volume, area, regions)
    arrays["raw_source_row"] = source_rows
    np.savez_compressed(out / "alignment-and-predictions.npz", **arrays)
    report.update(
        wall_seconds=time.perf_counter() - started,
        raw_rows=len(raw_ids),
        raw_region_counts=dict(Counter(regions)),
    )
    if report["wall_seconds"] > spec["wall_limit_s"]:
        raise TimeoutError("E17 cumulative CPU wall limit")
    write_json(out / "report.json", report)
    # A subsequent neural comparison requires this explicit, immutable input
    # manifest; emitting it here keeps the two research CLIs directly usable.
    write_json(
        out / "manifest.json",
        {
            "schema": "E17-frozen-artifacts-v1",
            "files": {
                name: file_hash(out / name)
                for name in (
                    "spec.json",
                    "report.json",
                    "alignment-and-predictions.npz",
                    "runner.py",
                )
            },
        },
    )
    print(json.dumps(report, allow_nan=False), flush=True)
    print("manifest_sha256", file_hash(out / "manifest.json"), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--graph",
        type=Path,
        default=Path("data/acquisitions/banc888-windows-20260910/bundle"),
    )
    parser.add_argument(
        "--raw",
        type=Path,
        default=Path(
            "data/acquisitions/banc888-windows-20260910/raw/banc_888_meta.feather"
        ),
    )
    parser.add_argument(
        "--areas",
        type=Path,
        default=Path(
            "verification/body-control-20260911/rate-reference/source/data/banc t1 premotor/wTable_20260217_fullData_consistentColumns.csv"
        ),
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        run(args.graph, args.raw, args.areas, args.out)
    except Exception as exc:
        if (
            args.out.exists()
            and not (args.out / "report.json").exists()
            and not (args.out / "invalid.json").exists()
        ):
            write_json(
                args.out / "invalid.json",
                {"validity": "invalid", "error": f"{type(exc).__name__}: {exc}"},
            )
        raise
