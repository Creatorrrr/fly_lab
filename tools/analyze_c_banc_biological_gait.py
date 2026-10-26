"""Offline empirical gait comparison; never tunes or advances the controller.

Positions and geometric angles are measured, not joint-command surrogates.
Analysis constants are frozen in BANC_BIOLOGICAL_GAIT_PROTOCOL_20260919.md.
No statistic here establishes biological equivalence from a single fly clip.
"""

import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, hilbert, savgol_filter

LEGS = ("lf", "lm", "lh", "rf", "rm", "rh")
PAIRS = ((0, 3), (1, 4), (2, 5), (0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5))
ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, allow_nan=False), encoding="utf-8"
    )


def runs(mask):
    edges = np.diff(np.r_[False, np.asarray(mask, dtype=bool), False].astype(int))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def validate_clock(t):
    t = np.asarray(t, dtype=float)
    if t.ndim != 1 or len(t) < 4 or not np.isfinite(t).all():
        raise ValueError("A finite sampled clock with at least four points is required")
    dt = round(float(np.median(np.diff(t))), 12)
    if dt <= 0 or not np.allclose(np.diff(t), dt, atol=1e-9, rtol=0):
        raise ValueError("Noncontiguous clock: never bridge missing samples")
    return dt


def geometric_knee(points):
    """Interior CTr--FTi--TiTa angle; radians or neutral-pose conventions unused."""
    a = points[:, :, 1] - points[:, :, 2]
    b = points[:, :, 3] - points[:, :, 2]
    den = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    if np.any(den <= 1e-12):
        raise ValueError("Degenerate femur or tibia in coordinate recording")
    return np.degrees(np.arccos(np.clip(np.sum(a * b, axis=-1) / den, -1, 1)))


def position_forward_speed(position, forward, dt):
    velocity = np.gradient(position, dt, axis=0)
    return gaussian_filter1d(
        np.sum(velocity * forward, axis=1), 0.02 / dt, mode="nearest"
    )


def load_model(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = payload["samples"]
    t = np.array([r["time_s"] for r in rows])
    dt = validate_clock(t)
    if not np.isclose(dt, 0.005) or any(
        r["control_tick"] != rows[0]["control_tick"] + i for i, r in enumerate(rows)
    ):
        raise ValueError("Expected complete 5 ms control-boundary recording")
    basis = np.array([r["body"]["basis"] for r in rows])
    if not np.allclose(np.einsum("tji,tjk->tik", basis, basis), np.eye(3), atol=1e-9):
        raise ValueError("Body basis is not orthogonal")
    origin = np.array([r["body"]["position"] for r in rows])
    points = np.array([[r["body"]["legs"][leg] for leg in LEGS] for r in rows])
    # Row vectors projected onto the recorded forward/lateral/up basis columns.
    ego = np.einsum("tlpj,tjk->tlpk", points - origin[:, None, None], basis)
    lengths = np.median(
        np.linalg.norm(np.diff(ego, axis=2), axis=-1).sum(axis=-1), axis=0
    )
    brake_times = [
        r["time_s"] for r in rows if (r.get("navigation") or {}).get("brake_request")
    ]
    end = min(t[-1], 10.0, min(brake_times) if brake_times else np.inf)
    window = (t >= 0.5) & (t < end)
    speed = position_forward_speed(origin, basis[:, :, 0], dt)
    yaw = np.unwrap([r["body"]["yaw"] for r in rows])
    yaw_rate = gaussian_filter1d(np.gradient(yaw, dt), 0.02 / dt, mode="nearest")
    return {
        "t": t,
        "dt": dt,
        "ap": ego[:, :, -1, 0] / lengths,
        "knee": geometric_knee(ego),
        "window": window,
        "moving": window & (speed > 0.5),
        "straight": np.abs(yaw_rate) < 0.5,
        "speed": speed,
        "yaw_rate": yaw_rate,
        "contact": np.array([r["feet"]["load_bearing"] for r in rows], dtype=bool),
        "position": origin,
        "upright": np.array([r["upright"] for r in rows]),
        "faults": [r["fault"] for r in rows if r["fault"]],
        "lengths": lengths,
        "start_s": 0.5,
        "end_s": float(end),
        "source": {
            "path": str(Path(path).resolve()),
            "sha256": sha256(path),
            "kind": "model",
            "body_hash": payload["body_hash"],
            "model_hash": payload["model_hash"],
            "adapter_hash": payload["adapter_hash"],
            "graph_hash": payload["graph_hash"],
            "parameters": payload["parameters"],
            "foot": "tarsus5 origin proxy",
            "length_unit": "mm",
            "speed_source": "Thorax coordinate derivative, projected onto physical forward axis, Gaussian sigma 20 ms; not boundary body.speed",
        },
    }


def load_reference(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len({r["video_id"] for r in rows}) != 1:
        raise ValueError("One continuous raw reference video expected")
    frame = np.array([int(r["frame_idx"]) for r in rows])
    if not np.all(np.diff(frame) == 1):
        raise ValueError("Reference frames have a gap")
    t = (frame - frame[0]) / 120.0
    dt = validate_clock(t)
    points = np.array(
        [
            [
                [
                    [float(r[f"{leg.upper()}-{joint}_{axis}"]) for axis in "xyz"]
                    for joint in ("ThC", "CTr", "FTi", "TiTa", "Claw")
                ]
                for leg in LEGS
            ]
            for r in rows
        ]
    )
    thorax = np.array([[float(r[f"Th_{axis}"]) for axis in "xyz"] for r in rows])
    points -= thorax[:, None, None]
    lengths = np.median(
        np.linalg.norm(np.diff(points, axis=2), axis=-1).sum(axis=-1), axis=0
    )
    return {
        "t": t,
        "dt": dt,
        "ap": points[:, :, -1, 0] / lengths,
        "knee": geometric_knee(points),
        "window": np.ones(len(t), dtype=bool),
        "moving": np.ones(len(t), dtype=bool),
        "straight": np.ones(len(t), dtype=bool),
        "speed": None,
        "yaw_rate": None,
        "lengths": lengths,
        "start_s": 0.0,
        "end_s": float(t[-1]),
        "source": {
            "path": str(Path(path).resolve()),
            "sha256": sha256(path),
            "kind": "empirical_reference",
            "doi": "10.7910/DVN/3MCEYR",
            "video_ids": [rows[0]["video_id"]],
            "frames": frame.tolist(),
            "sampling_hz": 120,
            "length_unit": "unverified annotation units; normalized by leg length",
            "independence": "Single short clip used in NMF template development; not a held-out population",
        },
    }


def phase_and_events(data, amplitude_floor=0.01, prominence=0.02):
    """Do not concatenate motion bouts or compute phase for motionless legs."""
    t, dt, ap = data["t"], data["dt"], data["ap"]
    phi = np.full(ap.shape, np.nan)
    events = [[] for _ in LEGS]
    eligible = np.zeros(len(t), dtype=bool)
    bouts = []
    for lo, hi in runs(data["moving"]):
        if (hi - lo - 1) * dt < 0.3 - 1e-9:
            continue
        eligible[lo:hi] = True
        bouts.append(
            {
                "start_s": float(t[lo]),
                "end_s": float(t[hi - 1]),
                "samples": int(hi - lo),
            }
        )
        x = ap[lo:hi]
        z = hilbert(x - x.mean(axis=0), axis=0)
        valid = np.abs(z) >= amplitude_floor
        trim = max(1, int(np.ceil(0.02 / dt)))
        valid[:trim] = False
        valid[-trim:] = False
        phi[lo:hi] = np.where(valid, np.angle(z), np.nan)
        # Largest odd window <= 25 ms; reference needs >=3 points for a filter.
        win = int(np.floor(0.025 / dt + 1e-9))
        win -= 1 - win % 2
        smooth = (
            savgol_filter(x, win, min(2, win - 1), axis=0) if win >= 3 else x.copy()
        )
        for leg in range(6):
            peaks, _ = find_peaks(
                smooth[:, leg],
                distance=max(1, int(np.ceil(0.04 / dt))),
                prominence=prominence,
            )
            for a, b in itertools.pairwise(peaks):
                p = int(a + np.argmin(smooth[a : b + 1, leg]))
                if p in (a, b):
                    continue
                cycle = {
                    "aep_s": float(t[lo + a]),
                    "pep_s": float(t[lo + p]),
                    "next_aep_s": float(t[lo + b]),
                    "period_s": float((b - a) * dt),
                    "retraction_s": float((p - a) * dt),
                    "protraction_s": float((b - p) * dt),
                    "retraction_fraction": float((p - a) / (b - a)),
                    "excursion_leg_lengths": float(smooth[a, leg] - smooth[p, leg]),
                    "mean_forward_mm_s": float(
                        np.mean(data["speed"][lo + a : lo + b + 1])
                    )
                    if data["speed"] is not None
                    else None,
                    "straight_fraction": float(
                        data["straight"][lo + a : lo + b + 1].mean()
                    ),
                }
                events[leg].append(cycle)
    return phi, events, eligible, bouts


def percentiles(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return (
        {
            "p05": float(np.percentile(x, 5)),
            "median": float(np.median(x)),
            "p95": float(np.percentile(x, 95)),
        }
        if len(x)
        else None
    )


def circular_summary(cycles):
    cycles = np.asarray(cycles, dtype=float)
    cycles = cycles[np.isfinite(cycles)]
    if not len(cycles):
        return {
            "samples": 0,
            "mean_cycles": None,
            "resultant_length": None,
            "mean_antiphase_distance_cycles": None,
        }
    z = np.mean(np.exp(2j * np.pi * cycles))
    return {
        "samples": len(cycles),
        "mean_cycles": float((np.angle(z) / (2 * np.pi)) % 1),
        "resultant_length": float(abs(z)),
        "mean_antiphase_distance_cycles": float(
            np.mean(np.abs(((cycles - 0.5 + 0.5) % 1) - 0.5))
        ),
    }


def phase_summary(phi, mask, dt):
    pairs = {}
    for a, b in PAIRS:
        delta = (phi[:, b] - phi[:, a]) / (2 * np.pi)
        values = delta[mask & np.isfinite(delta)]
        item = circular_summary(values)
        item["valid_duration_s"] = item["samples"] * dt
        pairs[f"{LEGS[a]}-{LEGS[b]}"] = item
    return pairs


def event_summary(events):
    result = {}
    for leg, cycles in zip(LEGS, events):
        periods = [c["period_s"] for c in cycles]
        result[leg] = {
            "complete_cycles": len(cycles),
            "recurrence_evaluable": len(cycles) >= 3,
            "period_s": percentiles(periods),
            "retraction_s": percentiles([c["retraction_s"] for c in cycles]),
            "protraction_s": percentiles([c["protraction_s"] for c in cycles]),
            "retraction_fraction": percentiles(
                [c["retraction_fraction"] for c in cycles]
            ),
            "excursion_leg_lengths": percentiles(
                [c["excursion_leg_lengths"] for c in cycles]
            ),
            "period_cv": float(np.std(periods) / np.mean(periods)) if periods else None,
            "cycles": cycles,
        }
    return result


def analyze(data):
    phi, events, eligible, bouts = phase_and_events(data)
    moving, window, dt = data["moving"], data["window"], data["dt"]
    result = {
        "source": data["source"],
        "window": {
            "start_s": data["start_s"],
            "end_s": data["end_s"],
            "sampled_duration_s": float(np.trapezoid(window.astype(float), data["t"])),
            "moving_duration_s": float(np.trapezoid(moving.astype(float), data["t"])),
            "eligible_bout_duration_s": float(
                np.trapezoid(eligible.astype(float), data["t"])
            ),
            "bouts": bouts,
            "straight_eligible_duration_s": float(
                np.trapezoid((eligible & data["straight"]).astype(float), data["t"])
            ),
            "longest_forward_bout_s": max(
                ((b - a - 1) * dt for a, b in runs(moving)), default=0.0
            ),
        },
        "pair_phase": phase_summary(phi, eligible, dt),
        "straight_pair_phase": phase_summary(phi, eligible & data["straight"], dt),
        "events": event_summary(events),
        "knee_interior_deg": {
            leg: percentiles(data["knee"][window, i]) for i, leg in enumerate(LEGS)
        },
        "foot_ap_leg_lengths": {
            leg: percentiles(data["ap"][window, i]) for i, leg in enumerate(LEGS)
        },
        "median_leg_lengths": data["lengths"].tolist(),
        "statistical_unit": "A recorded trajectory, not independent time points or individual flies",
    }
    result["sensitivity"] = {}
    for floor in (0.005, 0.02):
        alt, _, _, _ = phase_and_events(data, amplitude_floor=floor)
        result["sensitivity"][f"phase_amplitude_{floor}"] = phase_summary(
            alt, eligible, dt
        )
    for prom in (0.01, 0.04):
        _, alt_events, _, _ = phase_and_events(data, prominence=prom)
        result["sensitivity"][f"event_prominence_{prom}"] = {
            leg: {
                "complete_cycles": len(ev),
                "period_s": percentiles([c["period_s"] for c in ev]),
            }
            for leg, ev in zip(LEGS, alt_events)
        }
    if data["speed"] is not None:
        result["forward_speed_mm_s"] = percentiles(data["speed"][window])
        result["yaw_rate_rad_s"] = percentiles(data["yaw_rate"][window])
        pos = data["position"][window][:, [0, 2]]
        result["physical"] = {
            "minimum_upright": float(data["upright"][window].min()),
            "faults": data["faults"],
            "path_mm": float(np.linalg.norm(np.diff(pos, axis=0), axis=1).sum()),
            "net_mm": float(np.linalg.norm(pos[-1] - pos[0])),
        }
        contact = data["contact"][window]
        result["contact_boundary_descriptive"] = {
            "note": "Load includes adhesion; 5 ms boundary observations; transitions are not steps",
            "duty": {leg: float(contact[:, i].mean()) for i, leg in enumerate(LEGS)},
            "transitions": {
                leg: int(np.count_nonzero(np.diff(contact[:, i])))
                for i, leg in enumerate(LEGS)
            },
        }
        bins = ((0.5, 2), (2, 5), (5, 10), (10, 20), (20, float("inf")))
        result["speed_bins"] = {}
        for lo, hi in bins:
            selected = [
                [c for c in ev if lo <= c["mean_forward_mm_s"] < hi] for ev in events
            ]
            # Descriptive only: repeated cycles from the same deterministic run.
            result["speed_bins"][f"{lo}-{hi}"] = {
                leg: {
                    "cycles": len(ev),
                    "retraction_s": percentiles([c["retraction_s"] for c in ev]),
                    "protraction_s": percentiles([c["protraction_s"] for c in ev]),
                }
                for leg, ev in zip(LEGS, selected)
            }
    return result, phi, events


def plot_trial(data, result, phi, out, name):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t, win = data["t"], data["window"]
    fig, axes = plt.subplots(
        4, 1, figsize=(13, 10), sharex=True, constrained_layout=True
    )
    for i, leg in enumerate(LEGS):
        axes[0].plot(t[win], data["ap"][win, i], label=leg.upper(), lw=0.8)
    axes[0].legend(ncol=6, fontsize=9)
    axes[0].set_ylabel("Foot AP / leg length")
    for a, b in PAIRS[:3]:
        d = ((phi[:, b] - phi[:, a]) / (2 * np.pi)) % 1
        axes[1].plot(
            t[win], d[win], ".", ms=1.5, label=f"{LEGS[a].upper()}-{LEGS[b].upper()}"
        )
    axes[1].axhline(0.5, color="black", ls="--", lw=1)
    axes[1].set(ylim=(0, 1), ylabel="Left/right phase (cycles)")
    axes[1].legend(ncol=3, fontsize=9)
    if data["speed"] is not None:
        axes[2].plot(t[win], data["speed"][win], color="#236990")
        axes[2].axhline(0.5, color="gray", ls="--", lw=1)
        axes[2].set_ylabel("Forward speed (mm/s)")
        axes[3].imshow(
            data["contact"][win].T,
            extent=(t[win][0], t[win][-1], 5.5, -0.5),
            aspect="auto",
            interpolation="none",
            cmap="Greys",
            vmin=0,
            vmax=1,
        )
        axes[3].set_yticks(range(6), [leg.upper() for leg in LEGS])
        axes[3].set_ylabel("Load including adhesion")
    else:
        for i, leg in enumerate(LEGS):
            axes[2].plot(t, data["knee"][:, i], lw=1, label=leg.upper())
        axes[2].set_ylabel("Knee interior (degrees)")
        axes[3].text(
            0.5,
            0.5,
            "One fly, 0.3 s. No world velocity or contact-force ground truth.\nReference clip only; not a population acceptance envelope.",
            ha="center",
            va="center",
            transform=axes[3].transAxes,
        )
        axes[3].set_yticks([])
    axes[-1].set_xlabel("Recorded time (s)")
    axes[0].set_title(
        f"{name} | frozen M2 biological gait audit"
        if data["speed"] is not None
        else f"{name} | empirical 3D annotated walking"
    )
    fig.savefig(out / f"{name}.png", dpi=160)
    plt.close(fig)


def run(args):
    args.out.mkdir(parents=True, exist_ok=False)
    protocol = ROOT / "docs/BANC_BIOLOGICAL_GAIT_PROTOCOL_20260919.md"
    lock = json.loads((args.evidence / "protocol-lock.json").read_text())
    if sha256(protocol) != lock["protocol_sha256"]:
        raise ValueError("Protocol changed after it was frozen")
    note = ROOT / "docs/BANC_BIOLOGICAL_GAIT_MEASUREMENT_NOTE_20260919.md"
    note_lock = json.loads((args.evidence / "measurement-note-lock.json").read_text())
    if sha256(note) != note_lock["sha256"]:
        raise ValueError("Measurement correction changed after it was frozen")
    base = ROOT / "verification/banc-gait-quality-m2-20260912"
    paths = {
        "development": base / "e76-calibration/p03/development/development/trace.json",
        "forward": base / "e76-q4-remaining/forward-42/trace.json",
        "left": base / "e76-q4-remaining/left-42/trace.json",
        "right": base / "e76-q4-remaining/right-42/trace.json",
        "rear": base / "e76-calibration/p03/rear/development/trace.json",
        "confirmation_a": base / "e76-q3-goal1/development/trace.json",
        "confirmation_b": base / "e76-q3-goal2/development/trace.json",
        "fresh_far_forward": args.evidence / "fresh-far-forward/development/trace.json",
    }
    outputs = {}
    ref = load_reference(
        args.evidence / "reference/annotated_3d_walking_kinematics.csv"
    )
    result, phi, _events = analyze(ref)
    write_json(args.out / "empirical_reference.json", result)
    plot_trial(ref, result, phi, args.out, "empirical_reference")
    outputs["empirical_reference"] = result
    for name, path in paths.items():
        data = load_model(path)
        for key, expected in lock["model_parameters"].items():
            if data["source"]["parameters"][key] != expected:
                raise ValueError(f"{name}: wrong frozen model parameter {key}")
        result, phi, _events = analyze(data)
        exploratory_data = {**data, "moving": data["window"]}
        exploratory, ephi, _ = analyze(exploratory_data)
        exploratory["interpretation"] = (
            "Post-hoc descriptive analysis of all pre-brake samples; includes backward/stationary/turning time; not the preregistered moving-bout comparison"
        )
        result["exploratory_all_prebrake"] = exploratory
        outputs[name] = result
        write_json(args.out / f"{name}.json", result)
        np.savez_compressed(
            args.out / f"{name}-measured.npz",
            time_s=data["t"],
            foot_ap_leg_lengths=data["ap"],
            knee_interior_deg=data["knee"],
            phase_rad=phi,
            moving=data["moving"],
            straight=data["straight"],
        )
        plot_trial(data, result, phi, args.out, name)
        plot_trial(
            exploratory_data,
            exploratory,
            ephi,
            args.out,
            name + "_exploratory_all_prebrake",
        )
        print(
            json.dumps(
                {
                    "name": name,
                    "window": result["window"],
                    "phase": {
                        k: v
                        for k, v in result["pair_phase"].items()
                        if k in ("lf-rf", "lm-rm", "lh-rh")
                    },
                    "complete_cycles": {
                        k: v["complete_cycles"] for k, v in result["events"].items()
                    },
                }
            ),
            flush=True,
        )
    write_json(args.out / "results.json", outputs)
    write_json(
        args.out / "analysis-manifest.json",
        {
            "protocol_sha256": sha256(protocol),
            "analysis_sha256": sha256(__file__),
            "measurement_note_sha256": sha256(note),
            "inputs": {name: result["source"] for name, result in outputs.items()},
            "verdict": "No automatic biological-equivalence PASS; consult measured evidence and report",
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evidence",
        type=Path,
        default=ROOT / "verification/banc-biological-gait-20260919",
    )
    parser.add_argument("--out", type=Path, required=True)
    run(parser.parse_args())
