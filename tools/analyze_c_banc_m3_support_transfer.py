"""E122 read-only transfer audit of three frozen, recorded M3 trajectories.

The two decoder passes only transform stored motor rates. They do not advance a
neural network or body, and the resulting descriptions are not gait gates.
"""

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

for _thread_var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_thread_var] = "1"

import numpy as np
from scipy.ndimage import gaussian_filter1d
from threadpoolctl import threadpool_info

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flylab.c.storage import StateStore
from tools.analyze_c_banc_biological_gait import analyze, load_model

LEGS = ("lf", "lm", "lh", "rf", "rm", "rh")
DESIGN_SHA = "907d07ec6fbee795f991504472732790e171324c24c0e5f0a841ac2b0a1e776f"
FREEZE_SHA = "f6792d4ff585668e2cf327ba6fa7f5ad3e986dfb9d27d26f9dfdc03fb672a5f4"
BODY_SHA = "1b2a89084311ebd2ccbccbe2c57f2ec29fa937e115bb6cd4c7f6cd16ba7c2e6f"
DEFAULT_FREEZE = ROOT / "verification/banc-natural-gait-m3-20260919/e122-frozen-design.json"
DEFAULT_OUTPUT = ROOT / "verification/banc-natural-gait-m3-20260919/e122-support-transfer"


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite_array(value, shape, name):
    a = np.asarray(value, dtype=np.float64)
    require(a.shape == shape and np.isfinite(a).all(), f"Invalid {name}: {a.shape}")
    return a


def optional_float(value):
    value = float(value)
    return value if np.isfinite(value) else None


def deadline_check(start, limit, reserve=0.0, action="continuation"):
    elapsed = time.perf_counter() - start
    if elapsed > limit - reserve:
        raise TimeoutError(f"E122 {action} exceeded internal limit/reserve: {elapsed:.3f}s")
    return elapsed


def frozen_inputs(path):
    require(file_sha(path) == FREEZE_SHA, "E122 freeze SHA changed")
    freeze = json.loads(Path(path).read_text(encoding="utf-8"))
    require(freeze.get("schema") == "flylab.m3-e122-frozen-support-transfer.v1", "Unexpected freeze schema")
    require(freeze.get("design_sha256") == DESIGN_SHA, "E122 design SHA changed in freeze")
    require(file_sha(ROOT / freeze["design"]) == DESIGN_SHA, "E122 design changed")
    require(freeze.get("owned_files") == ["tools/analyze_c_banc_m3_support_transfer.py", "tests/test_c_banc_support_transfer.py"], "Owned file contract changed")
    require(freeze.get("not_adoption") is True and freeze.get("whole_goal_complete") is False, "Outcome contract changed")
    for section in ("input_sha256", "reused_source_sha256"):
        for relative, digest in freeze[section].items():
            name = Path(relative)
            require(not name.is_absolute() and ".." not in name.parts, f"Unsafe frozen path {relative}")
            require(file_sha(ROOT / name) == digest, f"Frozen {section} mismatch: {relative}")
    for files in freeze["preservation_sha256"].values():
        for relative, digest in files.items():
            name = Path(relative)
            require(not name.is_absolute() and ".." not in name.parts, f"Unsafe preservation path {relative}")
            require(file_sha(ROOT / name) == digest, f"Preserved artifact mismatch: {relative}")
    return freeze


def verify_trace(payload, rows, model):
    require(len(rows) == 2001, "Expected 2001 control states")
    ticks = np.asarray([row["control_tick"] for row in rows])
    require(np.array_equal(ticks, np.arange(2001)), "Control ticks do not start at zero or have gaps")
    t = finite_array([row["time_s"] for row in rows], (2001,), "control clock")
    require(np.allclose(t, np.arange(2001) * .005, atol=2e-8, rtol=0), "Unexpected control clock")
    require(payload["body_hash"] == BODY_SHA and model["source"]["body_hash"] == BODY_SHA, "Body hash changed")
    request = finite_array([row["joint_targets_rad"] for row in rows], (2001, 42), "requested joints")
    actual = finite_array([row["joint_angles_rad"] for row in rows], (2001, 42), "measured joints")
    qpos = [np.asarray(row["qpos"], dtype=float) for row in rows]
    qvel = [np.asarray(row["qvel"], dtype=float) for row in rows]
    require(all(a.shape == qpos[0].shape and np.isfinite(a).all() for a in qpos)
            and all(a.shape == qvel[0].shape and np.isfinite(a).all() for a in qvel), "Invalid physical qpos/qvel")
    require(np.array_equal(model["t"], t) and model["position"].shape == (2001, 3), "Existing model clock/geometry differs")
    boundary_load = finite_array([row["feet"]["normal_load_bw"] for row in rows], (2001, 6), "boundary load")
    boundary_support = np.asarray([row["feet"]["load_bearing"] for row in rows], dtype=bool)
    require(boundary_support.shape == (2001, 6) and np.array_equal(boundary_support, boundary_load >= .02), "Boundary support disagrees with 0.02 BW")
    require(all(row.get("fault") is None for row in rows), "Recorded body fault")
    return t, request, actual, boundary_load, boundary_support


def verify_start_anchor(start_state, row):
    require(start_state["control_tick"] == 0 and start_state["network"]["tick"] == 0, "Nonzero start checkpoint")
    network_rate = np.asarray(start_state["network"]["rate"], dtype=float)
    require(network_rate.ndim == 1 and network_rate.size >= 391
            and np.isfinite(network_rate).all() and np.all(network_rate == 0),
            "Actual start neural rate is not finite zero")
    require(start_state["body"]["modelHash"] == BODY_SHA, "Start body hash differs")
    state_vector = np.asarray(start_state["body"]["state"])
    npos = len(row["qpos"]); nvel = len(row["qvel"])
    require(np.array_equal(state_vector[1:1+npos], np.asarray(row["qpos"]))
            and np.array_equal(state_vector[1+npos:1+npos+nvel], np.asarray(row["qvel"])),
            "Start physical state differs")
    require(np.array_equal(np.asarray(start_state["body"]["action"]["joint_angles"]),
                           np.asarray(row["joint_targets_rad"])),
            "Start held body action differs from trace initial request")


def verify_shared_anchor(anchor, baseline, label):
    require(all(np.array_equal(anchor[key], baseline[key])
                for key in ("qpos", "qvel", "joint_angles")),
            f"{label} initial physical state differs from E85")


def native_intervals(native):
    """Native row n belongs to the 50 samples ending at control tick ceil(n/50)."""
    times = finite_array(native["time_s"], (100000,), "native clock")
    require(np.allclose(times, np.arange(1, 100001) * .0001, atol=2e-8, rtol=0), "Native clock has gap/shift")
    loads = finite_array(native["normal_load_bw"], (100000, 6), "native loads")
    require(np.all(loads >= 0), "Negative normal load")
    slip = np.asarray(native["slip_mm_s"], dtype=float)
    require(slip.shape == (100000, 6), "Native slip shape differs")
    if "slip_valid" in native:
        require(np.asarray(native["slip_valid"]).dtype == np.dtype(bool), "Saved slip_valid is not Boolean")
        valid = np.asarray(native["slip_valid"], dtype=bool)
        require(valid.shape == slip.shape and np.array_equal(valid, np.isfinite(slip)), "Saved slip_valid disagrees with finite slip")
        provenance = "recorded"
    else:
        valid = np.isfinite(slip)
        provenance = "derived_isfinite_E85"
    supported = loads >= .02
    require(not np.any(supported & ~valid), "Supported native slip unavailable")
    require(np.isfinite(native["upright"]).all() and np.asarray(native["upright"]).shape == (100000,), "Invalid native upright")
    require(np.asarray(native["nonfoot_contact"]).shape == (100000,), "Invalid native contact shape")
    return times, loads, slip, valid, supported, provenance


def native_group_stats(times, loads, slip, supported, state_indices):
    """Group the native (t-1,t] intervals by their endpoint state, without interpolation."""
    indices = np.asarray(state_indices, dtype=int)
    require(np.all((indices >= 1) & (indices <= 2000)), "Invalid native endpoint index")
    sample = (indices[:, None] - 1) * 50 + np.arange(50)[None, :]
    selected_load = loads[sample]
    selected_slip = slip[sample]
    selected_support = supported[sample]
    require(len(sample.ravel()) == len(indices) * 50, "Native interval count differs")
    valid_load = np.where(selected_support, selected_load, 0.0)
    weight = float(valid_load.sum())
    # Unsupported NaNs are excluded by indexing, never converted into zero-valued slip.
    weighted = float(np.sum(selected_load[selected_support] * selected_slip[selected_support])) if weight else 0.0
    return {
        "control_endpoints": int(len(indices)), "native_samples": int(sample.size),
        "first_native_s": float(times[sample[0, 0]]) if sample.size else None,
        "last_native_s": float(times[sample[-1, -1]]) if sample.size else None,
        "support_fraction": float(selected_support.mean()) if sample.size else None,
        "support_load_sum_bw": weight,
        "supported_slip_load_weighted_mm_s": weighted / weight if weight else None,
        "supported_slip_samples": int(selected_support.sum()),
    }


def decoder_calibration(cal, meta, signal, anatomy, start):
    ids = np.asarray(cal["motor_ids"])
    observed = np.asarray(signal["observed_ids"])
    require(ids.shape == (391,) and len(np.unique(ids)) == 391, "Motor ID count/order invalid")
    require(observed.shape == (446,) and len(np.unique(observed)) == 446, "Observed ID count invalid")
    require(np.array_equal(observed, meta["observed_ids"]) and np.array_equal(ids, meta["motor_ids"]), "Signal/metadata motor IDs differ")
    roster = anatomy["motor_roster"]
    require(len(roster) == 391 and len(set(roster)) == 391
            and all(isinstance(value, str) and value.startswith("flywire:banc:888:")
                    and value.rsplit(":", 1)[-1].isdigit() for value in roster),
            "Anatomy ordered FlyWire root roster invalid")
    require(anatomy.get("graph_hash") == start["graph_hash"], "Anatomy graph identity differs")
    pool = finite_array(cal["pool"], (96, 391), "pool")
    membership = np.asarray(cal["membership"], dtype=int)
    require(membership.shape == (391,) and np.all((membership >= 0) & (membership < 96)), "Membership invalid")
    require(np.all(pool >= 0) and np.allclose(pool.sum(axis=1), 1, atol=1e-12, rtol=0), "Pool must be nonnegative and row-normalized")
    require(np.array_equal(np.sum(pool > 0, axis=0), np.ones(391, dtype=int))
            and np.array_equal(np.argmax(pool, axis=0), membership), "Each motor must contribute to exactly its declared muscle")
    muscles = anatomy["muscles"]
    require(len(muscles) == 96, "Anatomy muscle count differs")
    leg_positions = [LEGS.index(item["leg"]) if item.get("leg") in LEGS else -1 for item in muscles]
    require(sorted(set(leg_positions)) == list(range(6)) and leg_positions == sorted(leg_positions), "Anatomical leg order differs")
    for i, item in enumerate(muscles):
        require(bool(item["muscle"]), "Anatomical muscle name missing")
        require(np.array_equal(np.asarray(item["columns"], dtype=int), np.flatnonzero(pool[i] > 0)), f"Muscle {i} columns differ")
    sc = start["adapter"]["calibration"]
    for key in ("motor_ids", "moment", "rest", "pool", "ltm"):
        require(np.array_equal(np.asarray(cal[key]), np.asarray(sc[key])), f"Start adapter {key} differs from calibration")
    require(start["adapter"]["groups"] == muscles, "Start anatomy groups differ")
    require(np.all(np.asarray(start["adapter"]["activation"]) == 0)
            and np.all(np.asarray(start["adapter"]["original"]["original"]["filtered"]) == 0), "Decoder initial activation/filter is nonzero")
    moment = finite_array(cal["moment"], (42, 96), "moment")
    rest = finite_array(cal["rest"], (42,), "rest")
    ltm = np.asarray(cal["ltm"], dtype=int)
    expected_ltm = [next((i for i, item in enumerate(muscles)
                          if item["leg"] == leg and item["muscle"] == "long_tendon_muscle"), -1)
                    for leg in LEGS]
    require(ltm.shape == (6,) and np.array_equal(ltm, expected_ltm), "Long tendon muscle order differs")
    return ids, observed, pool, moment, rest, ltm, muscles, roster


def decode_recorded(rates, pool, moment, rest, ltm, progress=None):
    """Exactly one 2000-step EMA pass over a recorded 391-MN matrix."""
    rates = finite_array(rates, (2000, 391), "recorded motor rates")
    alpha = -np.expm1(-.005 / .01)
    filtered = np.empty_like(rates)
    last = np.zeros(391, dtype=float)
    for i in range(2000):
        last += alpha * (rates[i] - last)
        filtered[i] = last
        if progress is not None:
            progress["decoder_steps"] += 1
    activation = filtered @ pool.T
    request = rest[None, :] + .02 * (activation @ moment.T)
    tendon = .2 * np.tanh(activation[:, ltm] / 10)
    return filtered, activation, request, tendon


def decoder_errors(activation, request, tendon, signal, rows, moment, rest):
    stored_activation = finite_array(signal["muscle_activation"], (2000, 96), "stored activation")
    recorded_request = finite_array([r["joint_targets_rad"] for r in rows[1:]], (2000, 42), "stored decoder request")
    recorded_tendon = finite_array([[r["tendon_inputs"][f"{leg}_tarsus"] for leg in LEGS] for r in rows[1:]], (2000, 6), "stored tendon")
    checks = {
        "activation": float(np.max(np.abs(activation - stored_activation))),
        "request": float(np.max(np.abs(request - recorded_request))),
        "tendon": float(np.max(np.abs(tendon - recorded_tendon))),
        "stored_activation_to_request": float(np.max(np.abs(rest + .02 * (stored_activation @ moment.T) - recorded_request))),
    }
    return checks


def compare_decoder(filtered, activation, request, tendon, signal, rows, moment, rest):
    checks = decoder_errors(activation, request, tendon, signal, rows, moment, rest)
    require(all(np.isfinite(list(checks.values()))) and max(checks.values()) <= 1e-10,
            f"Recorded decoder mismatch: {checks}")
    return checks


def group_masks(speed, window):
    return {
        "forward_gt_0p5": window & (speed > .5),
        "slow_0_to_0p5": window & (speed >= 0) & (speed <= .5),
        "backward_lt_0": window & (speed < 0),
    }


def rms(values, axis=0):
    return np.sqrt(np.mean(np.square(values), axis=axis))


def tracking_stats(request, measured, mask):
    if not np.any(mask):
        return {"samples": 0, "available": False}
    q = request[mask]
    a = measured[mask]
    delta = q - a
    dynamic_q = rms(q - q.mean(axis=0))
    dynamic_a = rms(a - a.mean(axis=0))
    ratio = np.divide(dynamic_a, dynamic_q, out=np.full(42, np.nan),
                      where=np.isfinite(dynamic_q) & (dynamic_q > 0))
    aggregate = lambda x: float(np.sqrt(np.mean(np.square(x))))
    per_joint = {
        "rms_rad": rms(delta).tolist(),
        "max_abs_rad": np.max(np.abs(delta), axis=0).tolist(),
        "p95_abs_rad": np.percentile(np.abs(delta), 95, axis=0).tolist(),
        "request_dynamic_rms_rad": dynamic_q.tolist(),
        "measured_dynamic_rms_rad": dynamic_a.tolist(),
        "measured_to_requested_dynamic_ratio": [optional_float(v) for v in ratio],
    }
    return {
        "samples": int(mask.sum()), "available": True,
        "all_joint_rms_rad": aggregate(delta),
        "by_leg": {leg: {"rms_rad": aggregate(delta[:, j*7:(j+1)*7]),
                          "max_abs_rad": float(np.max(np.abs(delta[:, j*7:(j+1)*7]))),
                          "p95_abs_rad": float(np.percentile(np.abs(delta[:, j*7:(j+1)*7]), 95))}
                   for j, leg in enumerate(LEGS)},
        "per_joint": per_joint,
    }


def contribution_stats(filtered, activation, pool, moment, rest, mask, muscles, ids):
    if not np.any(mask):
        return {"samples": 0, "available": False}
    # Each component is factorized: 42x391 or 42x96, never time x joint x motor.
    c_mn = .02 * (moment @ pool)
    c_muscle = .02 * moment
    fm = filtered[mask]
    am = activation[mask]
    mean_mn = c_mn * fm.mean(axis=0)[None, :]
    dyn_mn = np.abs(c_mn) * rms(fm - fm.mean(axis=0))[None, :]
    mean_muscle = c_muscle * am.mean(axis=0)[None, :]
    dyn_muscle = np.abs(c_muscle) * rms(am - am.mean(axis=0))[None, :]
    linear_mean = mean_mn.sum(axis=1)
    require(np.allclose(linear_mean, mean_muscle.sum(axis=1), atol=1e-10, rtol=0), "Factorized MN/muscle sums differ")
    def leaders(mean, labels):
        flat = np.argsort(np.abs(mean).ravel())[-12:][::-1]
        return [{"joint": int(index // mean.shape[1]), "source": labels[int(index % mean.shape[1])],
                 "signed_mean_rad": float(mean.ravel()[index])} for index in flat]
    return {
        "samples": int(mask.sum()), "available": True,
        "rest_rad": rest.tolist(), "linear_net_mean_rad": linear_mean.tolist(),
        "request_mean_rad": (rest + linear_mean).tolist(),
        "positive_mean_sum_rad": np.maximum(mean_mn, 0).sum(axis=1).tolist(),
        "negative_mean_sum_rad": np.minimum(mean_mn, 0).sum(axis=1).tolist(),
        "mn_signed_mean_rad_42x391": mean_mn.tolist(),
        "mn_centered_rms_rad_42x391": dyn_mn.tolist(),
        "muscle_signed_mean_rad_42x96": mean_muscle.tolist(),
        "muscle_centered_rms_rad_42x96": dyn_muscle.tolist(),
        "largest_mn_mean": leaders(mean_mn, [str(i) for i in ids]),
        "largest_muscle_mean": leaders(mean_muscle, [f"{m['leg']}:{m['muscle']}" for m in muscles]),
    }


def physical_stats(rows, model, request, measured, boundary_load, boundary_support, native, decoder=None):
    t = model["t"]
    speed = model["speed"]
    window = (t >= .5) & (t < 10)
    require(int(window.sum()) == 1900, "Control analysis window differs from 1900 states")
    position = finite_array([r["body"]["position"] for r in rows], (2001, 3), "body position")
    basis = finite_array([r["body"]["basis"] for r in rows], (2001, 3, 3), "body basis")
    tip = finite_array([[r["body"]["legs"][leg][-1] for leg in LEGS] for r in rows], (2001, 6, 3), "tarsus5 origin")
    ego = np.einsum("tlj,tjk->tlk", tip - position[:, None, :], basis)
    ap_velocity = gaussian_filter1d(np.gradient(ego[:, :, 0], .005, axis=0), .02/.005, axis=0, mode="nearest")
    times, loads, slip, valid, supported, provenance = native
    masks = group_masks(speed, window)
    result = {}
    for name, mask in {"all": window, **masks}.items():
        selected = np.flatnonzero(mask)
        native_summary = native_group_stats(times, loads, slip, supported, selected)
        load = boundary_load[mask]
        forward_foot = ap_velocity[mask] > 0
        support = boundary_support[mask]
        weight = float(load[support].sum())
        foot_weighted = float(np.sum(load[support] * ap_velocity[mask][support]) / weight) if weight else None
        result[name] = {
            "control_states": int(mask.sum()), "speed_mean_mm_s": float(speed[mask].mean()) if mask.any() else None,
            "tracking": tracking_stats(request, measured, mask),
            "boundary_support_fraction": float(support.mean()) if mask.any() else None,
            "boundary_forward_moving_foot_fraction_given_support": float((support & forward_foot).sum()/support.sum()) if support.any() else None,
            "boundary_load_weighted_foot_ap_velocity_mm_s": foot_weighted,
            "tarsus5_body_ap_mean_mm": ego[mask, :, 0].mean(axis=0).tolist() if mask.any() else None,
            "tarsus5_body_height_mean_mm": ego[mask, :, 2].mean(axis=0).tolist() if mask.any() else None,
            "tarsus5_body_ap_velocity_mean_mm_s": ap_velocity[mask].mean(axis=0).tolist() if mask.any() else None,
            "native_interval": native_summary,
        }
        if decoder is not None:
            filtered, activation, pool, moment, rest, muscles, ids = decoder
            result[name]["contributions"] = contribution_stats(filtered, activation, pool, moment, rest, mask[1:], muscles, ids)
    f7 = (times >= .5) & (times < 10)
    selected = supported & f7[:, None]
    weight = float(loads[selected].sum())
    result["native_F7_fixed_window"] = {
        "window": "[0.5,10) native, distinct from control endpoint intervals (.495,9.995]",
        "native_samples": int(f7.sum()), "support_samples": int(selected.sum()),
        "supported_slip_load_weighted_mm_s": float(np.sum(loads[selected] * slip[selected]) / weight) if weight else None,
        "support_load_sum_bw": weight,
    }
    result["native_slip_valid_provenance"] = provenance
    return result, {"t": t, "speed": speed, "tracking_rms_rad": np.sqrt(np.mean((request-measured)**2, axis=1)),
                    "supported_forward_foot_fraction": np.divide((boundary_support & (ap_velocity > 0)).sum(axis=1),
                                                                  boundary_support.sum(axis=1),
                                                                  out=np.full(2001, np.nan), where=boundary_support.sum(axis=1)>0)}


def plot_comparison(out, curves):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True, constrained_layout=True)
    primary = None
    try:
        for label, data in curves.items():
            mask = (data["t"] >= .5) & (data["t"] <= 10)
            for axis, key in zip(axes, ("speed", "tracking_rms_rad", "supported_forward_foot_fraction")):
                axis.plot(data["t"][mask], data[key][mask], linewidth=.8, label=label)
        axes[0].set_ylabel("Thorax forward (mm/s)")
        axes[1].set_ylabel("42-joint tracking RMS (rad)")
        axes[2].set_ylabel("Forward tarsus5 among supported feet")
        axes[2].set_xlabel("Recorded time (s)")
        axes[2].set_xlim(.5, 10)
        for axis in axes:
            axis.grid(alpha=.2)
            axis.legend()
        fig.savefig(out / "comparison.png", dpi=130)
    except BaseException as error:
        primary = error
    finally:
        try:
            plt.close(fig)
        except BaseException as close_error:
            if primary is None:
                primary = close_error
            else:
                primary.add_note(f"Secondary E122 plot close failure: {type(close_error).__name__}: {close_error}")
    if primary is not None:
        raise primary.with_traceback(primary.__traceback__)


def save_decoder_artifacts(out, label, raw, checks, registry):
    """Independent writes retain a completed pass even if one receipt fails."""
    failures = []
    try:
        np.savez(out / f"{label}-decoder-raw.npz", **raw)
        registry["raw_saved"].add(label)
    except BaseException as error:
        failures.append(("decoder raw", error))
    try:
        save_json(out / f"{label}-decoder-pass.json", {"stored_rate_rows": 2000, "motor_units": 391,
                    "decoder_attempts": 1, "decoder_completed": 1, "decoder_steps": 2000,
                    "actual_neural_s": 0, "actual_physical_s": 0,
                    "raw_saved": label in registry["raw_saved"]})
    except BaseException as error:
        failures.append(("decoder pass receipt", error))
    try:
        save_json(out / f"{label}-decoder-comparison-raw.json", checks)
    except BaseException as error:
        failures.append(("decoder comparison raw", error))
    if failures:
        first_operation, primary = failures[0]
        primary.add_note(f"Primary E122 persistence operation: {first_operation}")
        for operation, secondary in failures[1:]:
            primary.add_note(f"Secondary E122 {operation} failure: {type(secondary).__name__}: {secondary}")
        raise primary.with_traceback(primary.__traceback__)


def one_dataset(label, spec, out, start, resources, registry, baseline_anchor=None):
    deadline_check(start, resources["internal_limit_s"], resources["new_dataset_or_decoder_reserve_s"], "new dataset")
    source = ROOT / spec["directory"]
    payload = json.loads((source / "trace.json").read_text(encoding="utf-8"))
    rows = payload["samples"]
    with np.load(source / "native-contact.npz", allow_pickle=False) as stored:
        native_data = {key: stored[key] for key in stored.files}
    # Save as soon as a dataset is read. A later gate failure leaves inspectable records.
    np.savez(out / f"{label}-recorded-raw.npz",
             control_time_s=np.asarray([r["time_s"] for r in rows]),
             requested_rad=np.asarray([r["joint_targets_rad"] for r in rows]),
             measured_rad=np.asarray([r["joint_angles_rad"] for r in rows]),
             boundary_load_bw=np.asarray([r["feet"]["normal_load_bw"] for r in rows]),
             boundary_support=np.asarray([r["feet"]["load_bearing"] for r in rows]),
             body_position_mm=np.asarray([r["body"]["position"] for r in rows]),
             body_basis=np.asarray([r["body"]["basis"] for r in rows]),
             tarsus5_origin_mm=np.asarray([[r["body"]["legs"][leg][-1] for leg in LEGS] for r in rows]),
             native_time_s=native_data["time_s"], native_load_bw=native_data["normal_load_bw"],
             native_slip_mm_s=native_data["slip_mm_s"],
             native_slip_valid=np.asarray(native_data.get("slip_valid", np.isfinite(native_data["slip_mm_s"]))))
    model = load_model(source / "trace.json")
    t, request, measured, boundary_load, boundary_support = verify_trace(payload, rows, model)
    anchor = {key: np.asarray(rows[0][field], dtype=float)
              for key, field in (("qpos", "qpos"), ("qvel", "qvel"), ("joint_angles", "joint_angles_rad"))}
    if baseline_anchor is not None:
        verify_shared_anchor(anchor, baseline_anchor, label)
    native = native_intervals(native_data)
    original_analysis, _, _ = analyze(model)
    result = {"role": spec["role"], "source_dir": spec["directory"],
              "original_analyze_unchanged": original_analysis,
              "original_result_unchanged": json.loads((source / "result.json").read_text(encoding="utf-8")),
              "decoder": None, "initial_99": {"rows": 99, "tracking": tracking_stats(request, measured,
                  np.isin(np.arange(2001), np.arange(1, 100)))}}
    decoder = None
    if label in ("E113", "E121"):
        deadline_check(start, resources["internal_limit_s"], resources["new_dataset_or_decoder_reserve_s"], "decoder pass")
        with np.load(source / "recruitment-signals.npz", allow_pickle=False) as z:
            signal = {key: z[key] for key in z.files}
        with np.load(source / "recruitment-metadata.npz", allow_pickle=False) as z:
            meta = {key: z[key] for key in z.files}
        with np.load(source / "calibration.npz", allow_pickle=False) as z:
            cal = {key: z[key] for key in z.files}
        anatomy = json.loads((source / "muscle-provenance.json").read_text(encoding="utf-8"))
        start_state = StateStore.load(source / "start")
        verify_start_anchor(start_state, rows[0])
        ids, observed, pool, moment, rest, ltm, muscles, roster = decoder_calibration(cal, meta, signal, anatomy, start_state)
        require(np.array_equal(np.asarray(rows[0]["motor_rates"], dtype=float), np.zeros(391)), "Nonzero trace start motor rate")
        signal_rate = finite_array(signal["rate"], (2000, 446), "observed rate")
        require(np.allclose(signal["time_s"], t[1:], atol=2e-8, rtol=0), "Neural observation clock differs")
        columns = np.asarray([int(np.flatnonzero(observed == motor_id)[0]) for motor_id in ids])
        recorded_rate = finite_array([row["motor_rates"] for row in rows[1:]], (2000, 391), "trace motor rate")
        require(np.all(signal_rate[:, columns] >= 0) and np.all(recorded_rate >= 0), "Negative recorded 391-MN input rate")
        require(np.array_equal(signal_rate[:, columns], recorded_rate), "Motor ID order or stored trace rate differs")
        deadline_check(start, resources["internal_limit_s"], resources["new_dataset_or_decoder_reserve_s"], "actual decoder entry")
        registry["decoder_attempts"] += 1
        filtered, activation, recreated, tendon = decode_recorded(
            signal_rate[:, columns], pool, moment, rest, ltm, registry)
        registry["decoder_completed"] += 1
        raw = dict(motor_ids=ids, observed_ids=observed, recorded_rate=recorded_rate,
                   filtered_rate=filtered, activation=activation, recreated_request=recreated,
                   recreated_tendon=tendon, factorized_mn_42x391=.02*(moment@pool),
                   factorized_muscle_42x96=.02*moment, rest=rest)
        registry["raw_arrays"][label] = raw
        checks = decoder_errors(activation, recreated, tendon, signal, rows, moment, rest)
        registry["comparison_errors"][label] = checks
        save_decoder_artifacts(out, label, raw, checks, registry)
        require(all(np.isfinite(list(checks.values()))) and max(checks.values()) <= 1e-10,
                f"Recorded decoder mismatch: {checks}")
        result["decoder"] = {"reproduced": True, "steps": 2000, "max_abs_error": checks,
                             "motor_ids": ids.tolist(), "muscles": muscles}
        decoder = (filtered, activation, pool, moment, rest, muscles, roster)
    result["groups"], curve = physical_stats(rows, model, request, measured, boundary_load, boundary_support, native, decoder)
    save_json(out / f"{label}-summary.json", result)
    deadline_check(start, resources["internal_limit_s"], action=f"{label} final save")
    return result, curve, anchor


def run(output=DEFAULT_OUTPUT, freeze_path=DEFAULT_FREEZE):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing existing E122 output: {output}")
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    primary = None
    secondary = []
    summary = {"experiment": "M3-E122-support-transfer", "validity": "incomplete",
               "candidate_adopted": False, "F6_exposed": False, "whole_goal_complete": False,
               "actual_neural_s": 0, "actual_physical_s": 0, "fit_calls": 0, "LP_calls": 0,
               "decoder_calls": 0, "decoder_attempts": 0, "decoder_completed": 0,
               "decoder_steps": 0, "datasets": {}}
    curves = {}
    anchors = {}
    registry = {"decoder_attempts": 0, "decoder_completed": 0, "decoder_steps": 0,
                "raw_arrays": {}, "raw_saved": set(), "comparison_errors": {}}
    limit_s = 25.0
    try:
        save_json(output / "execution-raw.json", {"status": "started", "freeze": str(freeze_path),
                                                   "source_sha256": file_sha(__file__)})
        freeze = frozen_inputs(freeze_path)
        resources = freeze["resources"]
        limit_s = float(resources["internal_limit_s"])
        pools = threadpool_info()
        summary["cpu_runtime"] = {"requested_threads": 1,
                                  "environment": {key: os.environ.get(key) for key in
                                                  ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
                                  "observed_threadpools": pools}
        require(all(pool.get("num_threads") == 1 for pool in pools), "Actual CPU threadpool is not single-threaded")
        require(resources["prior_cumulative_rounded_s"] + resources["external_limit_s"] <= resources["shared_limit_s"], "Shared budget contract differs")
        summary["resource_contract"] = resources
        summary["interpretation_limit"] = (
            "E85 is an external teacher with fixed .2 tendon commands, while E113/E121 six leg tendons "
            "follow recorded muscle activation. Same initial body and 42-joint tracking do not isolate tendon effects; "
            "speed groups and linear contributions are descriptive, not causal or F4/F5/F8 gates."
        )
        for label in ("E85", "E113", "E121"):
            value, curve, anchor = one_dataset(label, freeze["datasets"][label], output, started,
                                               resources, registry, anchors.get("E85"))
            summary["datasets"][label] = {"summary_file": f"{label}-summary.json", "role": value["role"]}
            curves[label] = curve
            anchors[label] = anchor
            save_json(output / "execution-raw.json", summary)
        summary["decoder_attempts"] = registry["decoder_attempts"]
        summary["decoder_completed"] = registry["decoder_completed"]
        summary["decoder_calls"] = registry["decoder_attempts"]
        summary["decoder_steps"] = registry["decoder_steps"]
        summary["decoder_raw_saved"] = sorted(registry["raw_saved"])
        summary["decoder_comparison_errors_available"] = registry["comparison_errors"]
        require(summary["decoder_attempts"] == 2 and summary["decoder_completed"] == 2
                and summary["decoder_steps"] == 4000, "Decoder pass count differs")
        summary["validity"] = "valid_recorded_transfer_diagnostic"
        summary["wall_s_before_plot"] = time.perf_counter() - started
        np.savez(output / "plot-curves-raw.npz",
                 **{f"{label}_{key}": values for label, curve in curves.items()
                    for key, values in curve.items()})
        save_json(output / "summary.json", summary)
        plot_comparison(output, curves)
        deadline_check(started, resources["internal_limit_s"], action="plot save/close")
        summary["wall_s"] = time.perf_counter() - started
        summary["resource_limit_met"] = True
        save_json(output / "summary.json", summary)
        deadline_check(started, resources["internal_limit_s"], action="final summary save")
    except BaseException as error:
        primary = error
        summary["validity"] = "incomplete"
        summary["primary_error"] = {"type": type(error).__name__, "message": str(error),
                                     "traceback": traceback.format_exc()}
    finally:
        summary["decoder_attempts"] = registry["decoder_attempts"]
        summary["decoder_completed"] = registry["decoder_completed"]
        summary["decoder_calls"] = registry["decoder_attempts"]
        summary["decoder_steps"] = registry["decoder_steps"]
        summary["decoder_raw_saved"] = sorted(registry["raw_saved"])
        summary["decoder_comparison_errors_available"] = registry["comparison_errors"]
        summary["wall_s"] = time.perf_counter() - started
        if primary is not None:
            for label, arrays in registry["raw_arrays"].items():
                if label not in registry["raw_saved"]:
                    try:
                        recovery = output / f"{label}-decoder-raw-recovery"
                        recovery.mkdir(exist_ok=True)
                        for name, array in arrays.items():
                            np.save(recovery / f"{name}.npy", array, allow_pickle=False)
                        registry["raw_saved"].add(label)
                    except BaseException as recovery_error:
                        secondary.append({"operation": f"recover {label} decoder raw",
                                          "type": type(recovery_error).__name__, "message": str(recovery_error)})
            summary["decoder_raw_saved"] = sorted(registry["raw_saved"])
            summary["resource_limit_met"] = summary["wall_s"] <= limit_s
            for name, value in (("execution-raw.json", summary), ("summary.json", summary)):
                try:
                    save_json(output / name, value)
                except BaseException as secondary_error:
                    secondary.append({"operation": f"save {name}", "type": type(secondary_error).__name__,
                                      "message": str(secondary_error)})
            if secondary:
                try:
                    save_json(output / "secondary-errors.json", secondary)
                except BaseException as secondary_error:
                    secondary.append({"operation": "save secondary-errors.json", "type": type(secondary_error).__name__,
                                      "message": str(secondary_error)})
            summary["wall_s"] = time.perf_counter() - started
            if summary["wall_s"] > limit_s:
                summary["resource_limit_met"] = False
            try:
                save_json(output / "summary.json", summary)
            except BaseException as secondary_error:
                secondary.append({"operation": "save final elapsed summary.json", "type": type(secondary_error).__name__,
                                  "message": str(secondary_error)})
            if time.perf_counter() - started > limit_s and summary["resource_limit_met"]:
                summary["resource_limit_met"] = False
                summary["wall_s"] = time.perf_counter() - started
                try:
                    save_json(output / "summary.json", summary)
                except BaseException as secondary_error:
                    secondary.append({"operation": "save overrun summary.json", "type": type(secondary_error).__name__,
                                      "message": str(secondary_error)})
            if secondary:
                for note in secondary:
                    primary.add_note(f"Secondary E122 failure: {note}")
    if primary is not None:
        raise primary.with_traceback(primary.__traceback__)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--freeze", type=Path, default=DEFAULT_FREEZE)
    args = parser.parse_args()
    print(json.dumps(run(args.output, args.freeze), indent=2))
