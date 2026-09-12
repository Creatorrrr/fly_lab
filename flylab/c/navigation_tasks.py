"""Physical goal-arrival contract, independent of neural controller choices."""

import numpy as np

from .integrity import digest

CRITERIA = {
    "schema": "flylab.banc-navigation-criteria.v1",
    "arrival_deadline_s": 30.0,
    "radius_mm": 2.0,
    "settle_s": 0.5,
    "measure_s": 1.0,
    "stop_path_mm": 0.5,
    "control_dt_s": 0.005,
}
CASES = (
    ("forward-42", 42, (8.0, 0.0)),
    ("left-42", 42, (6.0, -6.0)),
    ("right-42", 42, (6.0, 6.0)),
    ("behind-42", 42, (-6.0, 0.0)),
    ("forward-7", 7, (8.0, 0.0)),
    ("right-7", 7, (6.0, 6.0)),
)


def evaluate_navigation(trace, target_xz):
    """A fly must reach AND remain at a target; pass-through is insufficient.

    The target is visible to the evaluator. Policy state such as 'arrived' is
    deliberately ignored: only time-stamped physical measurements count.
    """
    target = np.asarray(target_xz, dtype=float)
    if target.shape != (2,) or not np.isfinite(target).all():
        raise ValueError("Finite target X/Z pair required")
    result = {
        "criteria": CRITERIA.copy(),
        "criteria_hash": digest(CRITERIA),
        "target_xz_mm": target.tolist(),
        "task_status": "INCOMPLETE",
        "technical_status": "INCOMPLETE",
        "reasons": [],
        "biological_validation": False,
    }
    if len(trace) < 2:
        return result
    t = np.array([r["simTime"] for r in trace], float)
    p = np.array([r["position"] for r in trace], float)
    if (
        p.shape != (len(trace), 3)
        or not np.isfinite(p).all()
        or not np.isfinite(t).all()
    ):
        result.update(
            technical_status="FAIL",
            task_status="FAIL",
            reasons=["Nonfinite physical observations"],
        )
        return result
    t = t - t[0]
    dt = np.diff(t)
    if not np.allclose(dt, CRITERIA["control_dt_s"], atol=1e-8, rtol=0):
        result["reasons"].append("Missing or nonmonotonic physical samples")
        return result
    d = np.linalg.norm(p[:, [0, 2]] - target, axis=1)
    distance = np.linalg.norm(np.diff(p[:, [0, 2]], axis=0), axis=1)
    path = np.r_[0.0, np.cumsum(distance)]
    result.update(
        elapsed_s=float(t[-1]),
        initial_distance_mm=float(d[0]),
        final_distance_mm=float(d[-1]),
        minimum_distance_mm=float(d.min()),
        horizontal_path_mm=float(path[-1]),
    )
    if any(r.get("fault") or r.get("contact") for r in trace):
        result.update(
            technical_status="FAIL",
            task_status="FAIL",
            reasons=["Fault or non-foot physical contact"],
        )
        return result
    result["technical_status"] = "PASS"
    if d[0] <= CRITERIA["radius_mm"]:
        result.update(task_status="NOT_APPLICABLE", reasons=["No approach opportunity"])
        return result
    settle = round(CRITERIA["settle_s"] / CRITERIA["control_dt_s"])
    measure = round(CRITERIA["measure_s"] / CRITERIA["control_dt_s"])
    for i in np.flatnonzero(
        (d <= CRITERIA["radius_mm"]) & (t <= CRITERIA["arrival_deadline_s"] + 1e-8)
    ):
        a, b = i + settle, i + settle + measure
        if b >= len(trace):
            continue
        if (
            np.all(d[i : b + 1] <= CRITERIA["radius_mm"])
            and path[b] - path[a] <= CRITERIA["stop_path_mm"]
        ):
            result.update(
                task_status="PASS",
                arrival_s=float(t[i]),
                stop_path_mm=float(path[b] - path[a]),
                stop_maximum_distance_mm=float(d[i : b + 1].max()),
            )
            return result
    if t[-1] + 1e-8 >= sum(
        CRITERIA[k] for k in ("arrival_deadline_s", "settle_s", "measure_s")
    ):
        result.update(
            task_status="FAIL", reasons=["No arrival with sustained physical stop"]
        )
    return result
