"""E352 fixed read-only cycle coverage; inherited H348 remains partial."""

from __future__ import annotations

import time

ENTRY = time.perf_counter()
import argparse
import hashlib
import importlib.util
import itertools
import json
import math
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
BASE = "verification/banc-natural-gait-m3-20260919"
OWNED = (
    "tools/analyze_c_banc_m3_cycle_coverage.py",
    "tests/test_c_banc_cycle_coverage.py",
)
REGISTRATION = BASE + "/parent-e352-cycle-input-registration.json"
REGISTRATION_SHA = "09a577326954cbf1c72585f7ab814ec2ea9c950af57108e5a7c7f0e579f1634c"
REFLECTION = BASE + "/parent-e352-reflected-design-registration.json"
REFLECTION_SHA = "aebfadfa97d792563f532144fc57c460e337ebea5fe54a4121c2527452100174"
HELPER = "tools/recover_c_banc_m3_source_pose_report.py"
HELPER_SHA = "a9a5804b9b48adffb41c43c91931c5d1664b43e17624f1f0c70478a7105c1a04"
GAIT = "tools/analyze_c_banc_biological_gait.py"
GAIT_SHA = "bcceb5d1e9f26293ffc800d09a1fbf063a1ab784d73cf5489b74c57ecdd14760"
OUTPUT = BASE + "/e352-cycle-coverage"
LEGS = ("lf", "lm", "lh", "rf", "rm", "rh")
ARMS = ("B", "I", "C")
CHECKS = (
    "CPU",
    "ruff",
    "standard_events",
    "short_runs",
    "whole_window",
    "phase_timing",
    "native_context",
    "publication",
    "deadline",
    "metadata_guards",
    "six_path_reader",
    "selected_headers",
)
_HELPER = None


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def helper():
    global _HELPER
    if _HELPER is None:
        file = ROOT / HELPER
        require(sha(file.read_bytes()) == HELPER_SHA, "frozen reader source")
        spec = importlib.util.spec_from_file_location("e352_frozen_primitives", file)
        value = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(value)
        _HELPER = value
    return _HELPER


def gait():
    require(sha((ROOT / GAIT).read_bytes()) == GAIT_SHA, "frozen analyzer source")
    from tools import analyze_c_banc_biological_gait as value

    return value


def own_sources(root=ROOT):
    return {name: sha(helper().safe(name, root).read_bytes()) for name in OWNED}


class SixReader:
    """Own gate capability, exact six descriptors, frozen once-byte cache only."""

    def __init__(self, descriptors, *, root=ROOT, timely=lambda: None, generated=False):
        self.descriptors = {v["path"]: dict(v) for v in descriptors}
        require(len(self.descriptors) == 6, "exact six NPZ paths")
        require(
            all(
                v["path"].endswith(".npz")
                and type(v["bytes"]) is int
                and type(v["sha256"]) is str
                and len(v["sha256"]) == 64
                for v in descriptors
            ),
            "finite pinned six descriptors",
        )
        self.reader = helper().EvidenceReader(root, timely=timely)
        self.authorized = False
        if generated:
            resolved = Path(root).resolve()
            require(
                not resolved.is_relative_to(ROOT)
                or resolved.relative_to(ROOT)
                .as_posix()
                .startswith(BASE + "/e352-sol-"),
                "generated fixture root only",
            )
            self.authorized = True

    def npz(self, spec, roster, *, selected=None):
        require(self.authorized, "E352 gate before numeric bytes")
        require(
            self.descriptors.get(spec["path"]) == spec, "exact six-path descriptor only"
        )
        self.reader.array_authorized = True
        return self.reader.npz(spec, roster, selected=selected)


def metadata_preflight(root=ROOT, *, reader=None):
    p = helper()
    reader = reader or p.EvidenceReader(root)
    reg = reader.json(
        {"path": REGISTRATION, "bytes": 19431, "sha256": REGISTRATION_SHA}
    )
    file = p.safe(REFLECTION, root)
    reflection = reader.json(
        {"path": REFLECTION, "bytes": file.stat().st_size, "sha256": REFLECTION_SHA}
    )
    require(
        reg["schema"] == "flylab.m3.e352.cycle-input-registration.v1"
        and reflection["reflected_M1_M2_M3"] is True
        and reflection["registration"]["sha256"] == REGISTRATION_SHA,
        "E352 reflected metadata",
    )
    for spec in (reg["design"], reg["actual_review"], reg["review_md"]):
        reader.bytes(spec)
    for name, digest in reg["consumed_sources"].items():
        require(name.endswith(".py"), "source PY allowlist")
        file = p.safe(name, root)
        reader.bytes({"path": name, "bytes": file.stat().st_size, "sha256": digest})
    specs = list(reg["original_payloads"].values()) + list(reg["raw"].values())
    require(
        len(specs) == 6 and set(reg["allowed_npz_paths"]) == {s["path"] for s in specs},
        "exact six registered paths",
    )
    for spec in specs:
        require(
            type(spec["sha256"]) is str
            and len(spec["sha256"]) == 64
            and p.safe(spec["path"], root).stat().st_size == spec["bytes"],
            "stat-only raw descriptor",
        )
    inherited = {
        name: reader.json(spec) for name, spec in reg["inherited_provenance"].items()
    }
    final = inherited[BASE + "/parent-e351-final-decision.json"]
    report = inherited[final["report"]["path"]]
    identity = inherited[final["identity"]["path"]]
    require(
        final["report"] == reg["inherited_provenance"][final["report"]["path"]]
        and final["identity"] == reg["inherited_provenance"][final["identity"]["path"]],
        "E351 decision descriptor linkage",
    )
    require(
        final["diagnostic_completion_qualified"] is True
        and final["H348"] == reg["inherited_H348"] == report["diagnostic_H348"]
        and report["diagnostic_verified"] is True
        and report["external_completion_qualified"] is False
        and report["body_evidence_qualified"] is False,
        "separate E351 final decision, preserved provisional scope",
    )
    require(
        identity["qualified"] is True
        and identity["actual_supervisor_exit_code"] == 0
        and identity["all_owned_pids_ended"] is True
        and identity["errors"] == []
        and identity["owned_pids"]
        and all(
            identity["present_after_return"][str(pid)] is False
            and math.isfinite(identity["pid_create_times"][str(pid)])
            for pid in identity["owned_pids"]
        ),
        "E351 actual identity closure",
    )
    for spec in reg["raw"].values():
        require(
            report["consumed_input_digests"][spec["path"]] == spec["sha256"]
            and spec["digest_provenance"]
            == "E351_qualified_recovery_time_digest_not_acquisition_time",
            "qualified recovery-time raw digest",
        )
    # No preserved scientific trace arithmetic in this metadata-only admission.
    return {
        "registration": reg,
        "reader": reader,
        "final": final,
        "inherited_report": report,
        "descriptors": specs,
    }


def make_freeze(qualification, *, root=ROOT, context=None):
    p = helper()
    context = context or metadata_preflight(root)
    reader = context["reader"]
    file = p.safe(qualification, root)
    spec = {
        "path": qualification,
        "bytes": file.stat().st_size,
        "sha256": sha(file.read_bytes()),
    }
    qual = reader.json(spec)
    own = own_sources(root)
    require(
        qual.get("schema") == "flylab.m3.e352.root-qualification.v1"
        and qual.get("source_sha256") == own
        and qual.get("passed") is True
        and type(qual.get("original_calls")) is int
        and qual["original_calls"] == 0
        and all(qual.get("checks", {}).get(k) is True for k in CHECKS),
        "current E352 ROOT qualification",
    )
    execution = qual["execution"]
    contract, supervision, full = (
        reader.json(execution[k]) for k in ("contract", "supervision", "full_exit")
    )
    require(
        all(contract["source_sha256"].get(k) == v for k, v in own.items())
        and contract["original_calls"] == 0
        and supervision["success"] is True
        and supervision["original_calls"] == 0
        and supervision["contract_sha256"] == execution["contract"]["sha256"]
        and full["exit_code"] == 0
        and full["first_error"] is None
        and full["secondary_errors"] == []
        and full["all_owned_ended"] is True
        and full["owned_identities"]
        and all(
            x["close"] == "ended"
            and type(x["pid"]) is int
            and type(x["create_time"]) in (float, int)
            and math.isfinite(x["create_time"])
            for x in full["owned_identities"]
        )
        and len(contract["commands"]) == len(supervision["attempts"]) > 0
        and all(
            a["argv"] == c["argv"] and a["exit_code"] == 0 and a["killed"] is False
            for a, c in zip(supervision["attempts"], contract["commands"])
        )
        and type(full["charged_s"]) in (float, int)
        and math.isfinite(full["charged_s"])
        and 0 < full["charged_s"] <= 20,
        "current source argv full-exit/PID/cost qualification",
    )
    return {
        "schema": "flylab.m3.e352.cycle-freeze.v1",
        "source_sha256": own,
        "registration_sha256": REGISTRATION_SHA,
        "reflection_sha256": REFLECTION_SHA,
        "design": context["registration"]["design"],
        "consumed_sources": context["registration"]["consumed_sources"],
        "qualification": spec,
        "prospective": context["registration"]["prospective"],
        "original_authorized": False,
    }


def original_argv(freeze, review):
    return [
        ".venv/Scripts/python.exe",
        "-X",
        "utf8",
        "-B",
        OWNED[0],
        "--analyze",
        "--freeze",
        freeze,
        "--review",
        review,
        "--output",
        OUTPUT,
    ]


def original_gate(freeze_name, review_name, *, root=ROOT, output=OUTPUT, reader=None):
    p = helper()
    context = metadata_preflight(root, reader=reader)
    reader = context["reader"]

    def read(name):
        file = p.safe(name, root)
        return reader.json({"path": name, "bytes": file.stat().st_size, "sha256": None})

    freeze, review = read(freeze_name), read(review_name)
    require(
        freeze
        == make_freeze(freeze["qualification"]["path"], root=root, context=context),
        "exact current E352 freeze",
    )
    own, digest = own_sources(root), reader.digests[freeze_name]
    require(
        review.get("schema") == "flylab.m3.e352.root-original-review.v1"
        and review.get("approved") is True
        and review.get("source_sha256") == own
        and review.get("freeze_sha256") == digest
        and review.get("registration_sha256") == REGISTRATION_SHA
        and review.get("reflection_sha256") == REFLECTION_SHA
        and review.get("output") == OUTPUT == output,
        "E352 private approval",
    )
    reservation = reader.json(review["reservation"])
    expected = {
        "schema": "flylab.m3.e352.root-reservation.v1",
        "cumulative_before_s": 9845.069767399496,
        "cumulative_limit_s": 9900,
        "reserved_s": 6,
        "audit_wall_s": 6,
        "audit_internal_s": 5,
        "original_calls": 1,
        "retry_count": 0,
        "body_calls": 0,
        "neural_calls": 0,
        "physics_calls": 0,
        "CPG_calls": 0,
        "GPU_calls": 0,
        "source_sha256": own,
        "freeze_sha256": digest,
        "registration_sha256": REGISTRATION_SHA,
        "reflection_sha256": REFLECTION_SHA,
        "review_path": review_name,
        "output": OUTPUT,
        "argv": original_argv(freeze_name, review_name),
    }
    require(
        all(
            type(reservation.get(k)) is type(v) and reservation.get(k) == v
            for k, v in expected.items()
        ),
        "exact E352 once cumulative argv reservation",
    )
    require(
        not p.safe(output, root).exists()
        and shutil.disk_usage(root).free >= 512 << 20
        and all(
            os.environ.get(k) == "1"
            for k in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        ),
        "fresh output CPU1 free disk",
    )
    adapter = SixReader(context["descriptors"], root=root, timely=reader.timely)
    adapter.reader = (
        reader  # Same once cache; no old helper globals/gate/full-recovery mutation.
    )
    adapter.authorized = True
    context["arrays"] = adapter
    return context


def local_geometry(data, lo, hi):
    """Same fixed SG/peak settings; no Hilbert mask as event eligibility."""
    import numpy as np
    from scipy.signal import find_peaks, savgol_filter

    t, x, dt = data["t"], data["ap"][lo:hi], data["dt"]
    win = math.floor(0.025 / dt + 1e-9)
    win -= 1 - win % 2
    result = []
    for leg in range(6):
        row = {
            "raw_start": float(x[0, leg]),
            "raw_end": float(x[-1, leg]),
            "raw_range": float(np.ptp(x[:, leg])),
            "filter_unavailable_mask": hi - lo < win,
            "reason": None,
            "peaks": None,
            "pairs": None,
        }
        if hi - lo < win:
            row["reason"] = "fewer_samples_than_fixed_SG_window"
        else:
            smooth = (
                savgol_filter(x[:, leg], win, min(2, win - 1))
                if win >= 3
                else x[:, leg].copy()
            )
            peaks, _ = find_peaks(
                smooth, distance=max(1, math.ceil(0.04 / dt)), prominence=0.02
            )
            row["peaks"] = [
                {
                    "index": int(lo + a),
                    "time_s": float(t[lo + a]),
                    "value": float(smooth[a]),
                }
                for a in peaks
            ]
            row["pairs"] = []
            for a, b in itertools.pairwise(peaks):
                valley = int(a + np.argmin(smooth[a : b + 1]))
                row["pairs"].append(
                    {
                        "a": int(lo + a),
                        "p": int(lo + valley),
                        "b": int(lo + b),
                        "aep_s": float(t[lo + a]),
                        "pep_s": float(t[lo + valley]),
                        "next_aep_s": float(t[lo + b]),
                        "interior_minimum": valley not in (a, b),
                    }
                )
        result.append(row)
    return result


def coverage(data):
    import numpy as np

    g = gait()
    require(
        np.isfinite(data["t"]).all()
        and np.isfinite(data["ap"]).all()
        and np.isfinite(data["speed"]).all()
        and math.isclose(data["dt"], 0.005, abs_tol=1e-12),
        "finite 5ms geometry/speed",
    )
    require(
        np.array_equal(data["moving"], data["window"] & (data["speed"] > 0.5)),
        "actual frozen moving definition",
    )
    phi, events, eligible, bouts = g.phase_and_events(data)
    standard = g.analyze(data)[0]
    require(not np.isinf(phi).any(), "unexpected phase Inf")
    runs = []
    for lo, hi in g.runs(data["moving"]):
        lo, hi = int(lo), int(hi)
        duration = (hi - lo - 1) * data["dt"]
        is_eligible = duration >= 0.3 - 1e-9
        require(
            bool(eligible[lo:hi].all()) == is_eligible,
            "actual analyzer eligible linkage",
        )
        geometry = local_geometry(data, lo, hi)
        counts = []
        for leg in range(6):
            official = [
                e
                for e in events[leg]
                if data["t"][lo] <= e["aep_s"] and e["next_aep_s"] <= data["t"][hi - 1]
            ]
            counts.append(len(official))
            if is_eligible:
                require(
                    len(official)
                    == sum(p["interior_minimum"] for p in geometry[leg]["pairs"]),
                    "local vs actual official events",
                )
        runs.append(
            {
                "lo": lo,
                "hi": hi,
                "start_s": float(data["t"][lo]),
                "end_s": float(data["t"][hi - 1]),
                "duration_s": float(duration),
                "eligible": is_eligible,
                "standard_count_by_leg": dict(zip(LEGS, counts)),
                "geometry": dict(zip(LEGS, geometry)),
            }
        )
    indices = np.flatnonzero(data["window"])
    require(
        len(indices) > 0 and np.all(np.diff(indices) == 1),
        "contiguous nonempty diagnostic window",
    )
    whole = local_geometry(data, int(indices[0]), int(indices[-1]) + 1)
    for row in whole:
        for pair in row["pairs"] or []:
            a, b = pair["a"], pair["b"]
            inside = next((r for r in runs if r["lo"] <= a and b < r["hi"]), None)
            pair.update(
                classification="one_eligible_run"
                if inside and inside["eligible"]
                else "one_short_run"
                if inside
                else "crosses_nonmoving_or_multiple_runs",
                standard_event_identity_not_assumed=True,
                contributes_to_standard_count=False,
            )
    phase = [
        [float(v) if math.isfinite(float(v)) else None for v in row] for row in phi
    ]
    return {
        "standard": standard,
        "actual_bouts": bouts,
        "moving_runs": runs,
        "whole_window_geometry": dict(zip(LEGS, whole)),
        "phase_rad": phase,
        "phase_available_mask": np.isfinite(phi).tolist(),
        "phase_unavailable_count": int(np.isnan(phi).sum()),
        "event_eligibility_uses_phase_mask": False,
        "window_half_open": True,
        "boundary_peak_censoring": True,
        "whole_window_filter_context_differs": True,
    }


def phase_progress(phi, ticks):
    import numpy as np

    phi, ticks = np.asarray(phi, float), np.asarray(ticks)
    require(
        phi.ndim == 1
        and len(phi) == len(ticks)
        and len(phi) > 0
        and np.isfinite(phi).all()
        and np.all(np.diff(ticks) == 1),
        "contiguous finite source phase",
    )
    delta = (np.diff(phi) + 0.5) % 1 - 0.5
    cumulative = np.r_[0.0, np.cumsum(delta)]
    bins = {
        "blend_internal": [],
        "fully_shaped_internal": [],
        "crosses_blend_boundary": [],
    }
    for a, b, step in zip(ticks[:-1], ticks[1:], delta):
        key = (
            "blend_internal"
            if 101 <= a and b <= 119
            else "fully_shaped_internal"
            if a >= 120
            else "crosses_blend_boundary"
        )
        bins[key].append(
            {"from_tick": int(a), "to_tick": int(b), "signed_cycles": float(step)}
        )
    return {
        "start_tick": int(ticks[0]),
        "end_tick": int(ticks[-1]),
        "unit": "source_output_embedding_cycles",
        "net_cycles": math.fsum(float(v) for v in delta),
        "positive_variation_cycles": math.fsum(max(float(v), 0) for v in delta),
        "negative_variation_cycles": math.fsum(max(-float(v), 0) for v in delta),
        "span_cycles": float(np.ptp(cumulative)),
        "max_abs_step_cycles": float(np.abs(delta).max()) if len(delta) else None,
        "increments": delta.tolist(),
        "circular_wrap_to_ticks": [
            int(ticks[k + 1]) for k in np.flatnonzero(np.abs(np.diff(phi)) > 0.5)
        ],
        "half_turn_tie_to_ticks": [
            int(ticks[k + 1]) for k in np.flatnonzero(np.abs(delta) == 0.5)
        ],
        "direction_ambiguous": bool((np.abs(delta) == 0.5).any()),
        "multi_turn_alias_not_excluded": True,
        "blend_bins": bins,
    }


def native_validate(native, *, expected_count=15000):
    import numpy as np

    require(
        set(native)
        == {
            "time_s",
            "normal_load_bw",
            "slip_mm_s",
            "slip_valid",
            "upright",
            "nonfoot_contact",
        },
        "exact native six fields",
    )
    for k, value in native.items():
        shape = (
            (expected_count, 6)
            if k in ("normal_load_bw", "slip_mm_s", "slip_valid")
            else (expected_count,)
        )
        dtype = np.dtype(
            "bool" if k in ("slip_valid", "nonfoot_contact") else "float64"
        )
        require(value.shape == shape and value.dtype == dtype, "native shape/dtype")
        require(
            not np.isinf(value).any()
            and (k == "slip_mm_s" or np.isfinite(value).all()),
            "native unexpected nonfinite",
        )
    ids = np.rint(native["time_s"] / 1e-4).astype(np.int64)
    require(
        np.array_equal(ids, np.arange(1, expected_count + 1))
        and np.allclose(native["time_s"], ids * 1e-4, rtol=0, atol=2e-8),
        "native rounded contiguous clock tolerance",
    )
    require(
        (native["normal_load_bw"] >= 0).all()
        and np.array_equal(native["slip_valid"], np.isfinite(native["slip_mm_s"])),
        "native load/slip mask consistency",
    )
    return ids


def contact_summary(native, selected):
    import numpy as np

    load, slip, valid = (
        native[k][selected] for k in ("normal_load_bw", "slip_mm_s", "slip_valid")
    )
    count = len(load)
    support = load >= 0.02
    missing = int(np.count_nonzero(support & ~valid))
    weight = math.fsum(float(v) for v in load[support])
    reason = (
        "no_native_samples"
        if not count
        else "supported_slip_missing"
        if missing
        else "zero_support"
        if weight == 0
        else None
    )
    weighted = (
        None
        if reason
        else math.fsum(
            float(w) * float(v) for w, v in zip(load[support], slip[support])
        )
        / weight
    )
    histogram = (
        np.bincount(support.sum(axis=1), minlength=7) if count else np.zeros(7, int)
    )
    return {
        "samples": count,
        "support_leg_counts": histogram.tolist(),
        "support_leg_occupancy": [
            float(v / count) if count else None for v in histogram
        ],
        "leg_support_fraction": {
            leg: float(support[:, j].mean()) if count else None
            for j, leg in enumerate(LEGS)
        },
        "weighted_slip_mm_s": weighted,
        "unavailable_reason": reason,
        "supported_slip_missing_count": missing,
        "invalid_slip_count": int((~valid).sum()),
        "minimum_source_upright": float(native["upright"][selected].min())
        if count
        else None,
        "nonfoot_contact_count": int(native["nonfoot_contact"][selected].sum()),
        "contact_is_not_biological_stance": True,
    }


def speed_context(data, native, *, expected_count=15000):
    import numpy as np

    ids = native_validate(native, expected_count=expected_count)
    t = np.asarray(data["t"])
    ticks = np.rint(t / 0.005).astype(int)
    require(
        np.array_equal(ticks, np.arange(len(t)))
        and np.allclose(t, ticks * 0.005, rtol=0, atol=2e-8)
        and np.isfinite(data["speed"]).all(),
        "control rounded contiguous clock",
    )
    window = (ticks >= 100) & (ticks < 300)
    require(
        np.array_equal(window, data["window"]),
        "half-open standard window vs boundary300",
    )
    moving = data["speed"] > 0.5
    contexts, union = [], np.zeros(len(ids), bool)
    for k in np.flatnonzero(window[:-1] & window[1:] & (moving[:-1] != moving[1:])) + 1:
        anchor_id = int(ticks[k]) * 50
        lo, hi = max(5000, anchor_id - 200), min(15000, anchor_id + 200)
        selected = (ids >= lo) & (ids < hi)
        union |= selected
        contexts.append(
            {
                "bracket_s": [float(t[k - 1]), float(t[k])],
                "anchor_s": float(t[k]),
                "entering_moving": bool(moving[k]),
                "context_half_open_s": [lo * 1e-4, hi * 1e-4],
                "first_native_index": int(ids[selected][0]) if selected.any() else None,
                "last_native_index": int(ids[selected][-1]) if selected.any() else None,
                "summary": contact_summary(native, selected),
            }
        )
    index = np.flatnonzero(window)
    return {
        "sampled_crossings": contexts,
        "union_summary": contact_summary(native, union),
        "union_indices": np.flatnonzero(union).astype(int).tolist(),
        "union_double_counts": False,
        "window_censoring": {
            "first_in_window_moving": bool(moving[index[0]]),
            "last_in_window_moving": bool(moving[index[-1]]),
            "censoring_is_not_speed_crossing": True,
        },
        "time_alignment": "rounded clocks abs tolerance2e-8; actual native samples, clipped half-open contexts",
    }


def calculate(context):
    import numpy as np

    p = helper()
    reg, reader, arrays = context["registration"], context["reader"], context["arrays"]
    require(arrays.authorized, "own E352 numeric gate")
    payloads = {}
    for key, spec in reg["original_payloads"].items():
        roster = {
            k: (tuple(v["shape"]), v["dtype"])
            for k, v in spec["selected_members"].items()
        }
        payloads[key] = arrays.npz(spec, roster, selected=spec["selected_members"])
        require(
            all(np.isfinite(v).all() for v in payloads[key].values()),
            "finite selected source members",
        )
    control, reference = payloads["control"], payloads["reference"]
    require(
        np.array_equal(control["control_tick"], np.arange(301))
        and np.allclose(
            control["physics_time_s"], np.arange(301) * 0.005, atol=2e-8, rtol=0
        ),
        "control source clock",
    )
    plan = arrays.npz(
        reg["raw"]["shape_plan"],
        {
            **{arm: ((301, 42), "float64") for arm in ARMS},
            "z": ((201, 6, 2), "float64"),
            "R": ((201, 6), "float64"),
            "phi": ((201, 6), "float64"),
        },
    )
    require(all(np.isfinite(v).all() for v in plan.values()), "finite plan")
    plan_error = p.independent_plan(control, reference, plan)
    result = {}
    identities = None
    reader.json(reg["preserved_JSON"][reg["worker_folder"] + "/force-policy.json"])
    native_roster = {
        k: (
            (15000, 6)
            if k in ("normal_load_bw", "slip_mm_s", "slip_valid")
            else (15000,),
            "bool" if k in ("slip_valid", "nonfoot_contact") else "float64",
        )
        for k in (
            "time_s",
            "normal_load_bw",
            "slip_mm_s",
            "slip_valid",
            "upright",
            "nonfoot_contact",
        )
    }
    for arm in ARMS:
        reader.timely()

        def preserved(name):
            return reader.json(reg["preserved_JSON"][reg["worker_folder"] + "/" + name])

        trace = preserved(arm + "/trace.json")
        ledger, evaluation, observer = (
            preserved(arm + "/" + name)
            for name in ("receipt.json", "evaluation.json", "source-pose.json")
        )
        identity = {
            key: trace[key]
            for key in (
                "body_hash",
                "model_hash",
                "adapter_hash",
                "graph_hash",
                "parameters",
            )
        }
        if identities is None:
            identities = identity
        require(identity == identities, "same source identities, neural labels only")
        rows = trace["samples"]
        require(
            len(rows) == 301
            and trace["body_hash"] == reg["source_model_hash"]
            and ledger["status"] == "completed",
            "complete fixed trace/receipt",
        )
        for c, row in enumerate(rows):
            require(
                type(row["control_tick"]) is int
                and row["control_tick"] == c
                and math.isclose(row["time_s"], 0.005 * c, rel_tol=0, abs_tol=2e-8)
                and math.isclose(
                    row["physics_time_s"], 0.005 * c, rel_tol=0, abs_tol=2e-8
                )
                and row["neural_request"] is None
                and row["fault"] is None,
                "finite exact trace clock/fault",
            )
            for name, shape in (("qpos", None), ("qvel", None), ("ctrl", None)):
                value = np.asarray(row[name])
                require(
                    value.ndim == 1 and np.isfinite(value).all(),
                    "finite actual trace state",
                )
            points = np.asarray([row["body"]["legs"][leg] for leg in LEGS])
            require(
                points.shape == (6, 8, 3) and np.isfinite(points).all(),
                "actual finite all8 landmarks",
            )
            if c:
                p.exact(
                    np.asarray(row["joint_targets_rad"]),
                    plan[arm][c],
                    "actual planned trace command",
                )
        data = p.gait_model(
            trace,
            p.safe(reg["worker_folder"] + "/" + arm + "/trace.json", reader.root),
            reader.digests[reg["worker_folder"] + "/" + arm + "/trace.json"],
        )
        detail = coverage(data)
        inherited_gait = context["inherited_report"]["independent_metrics"][arm]["gait"]
        require(
            p.lower(detail["standard"]) == evaluation["gait"] == inherited_gait,
            "actual standard analyzer vs E351 and disk metrics",
        )
        native = arrays.npz(reg["raw"][arm + "_native"], native_roster)
        native_validate(native)
        progress = {}
        for leg, name in enumerate(LEGS):
            progress[name] = phase_progress(plan["phi"][:, leg], np.arange(100, 301))
        run_phase = []
        for run in detail["moving_runs"]:
            if run["eligible"]:
                lo, hi = max(run["lo"], 100), min(run["hi"], 301)
                run_phase.append(
                    {
                        "lo": lo,
                        "hi": hi,
                        "source_LF": phase_progress(
                            plan["phi"][lo - 100 : hi - 100, 0], np.arange(lo, hi)
                        ),
                        "geometry": run["geometry"],
                        "standard_count_by_leg": run["standard_count_by_leg"],
                    }
                )
        result[arm] = {
            "coverage": detail,
            "source_phase_all_legs": progress,
            "shared_C_source_is_LF_label": arm == "C",
            "eligible_run_source_LF": run_phase,
            "fully_shaped_source_LF": phase_progress(
                plan["phi"][20:, 0], np.arange(120, 301)
            ),
            "native_context": speed_context(data, native) if arm == "C" else None,
            "source_pose_health_inherited_not_reaudited": True,
            "source_pose_metadata_digest": reader.digests[
                reg["worker_folder"] + "/" + arm + "/source-pose.json"
            ],
            "receipt_status": ledger["status"],
            "observer_count_inherited": observer["count"],
        }
    return {
        "schema": "flylab.m3.e352.cycle-coverage-report.v1",
        "diagnostic_verified": True,
        "analysis_completed": True,
        "external_completion_qualified": False,
        "body_evidence_qualified": False,
        "inherited_H348": reg["inherited_H348"],
        "original_worker_status": reg["original_worker_status"],
        "candidate_adopted": False,
        "F1_F8_complete": False,
        "biology_proven": False,
        "new_body_neural_physics_CPG_GPU_calls": 0,
        "direct_recalculation": reg["direct"],
        "inherited_scope": reg["inherited"],
        "E351_final_decision": context["final"],
        "source_sha256": own_sources(reader.root),
        "registration_sha256": REGISTRATION_SHA,
        "reflection_sha256": REFLECTION_SHA,
        "consumed_sources": reg["consumed_sources"],
        "consumed_input_digests": reader.digests,
        "consumed_read_counts": reader.read_counts,
        "selected_members_opened": [list(v) for v in reader.opened_members],
        "raw_digest_provenance": {
            k: v.get("digest_provenance", "historical_selected")
            for k, v in reg["raw"].items()
        },
        "maximum_plan_error": plan_error,
        "arms": result,
        "first_error": None,
        "secondary_errors": [],
        "no_causal_or_biological_stance_claim": True,
    }


def error_report(error, *, stage="calculation_or_publication", context=None):
    item = helper().failure(stage, error)
    return {
        "schema": "flylab.m3.e352.cycle-coverage-report.v1",
        "diagnostic_verified": False,
        "analysis_completed": False,
        "external_completion_qualified": False,
        "body_evidence_qualified": False,
        "inherited_H348": context["registration"]["inherited_H348"]
        if context
        else None,
        "original_worker_status": "technical_invalid",
        "arms": None,
        "candidate_adopted": False,
        "F1_F8_complete": False,
        "biology_proven": False,
        "new_body_neural_physics_CPG_GPU_calls": 0,
        "first_error": item,
        "secondary_errors": [],
        "late_error_has_precedence": True,
    }


def publish_outcome(output, calculate_fn, *, timely=lambda: None, context=None):
    output.mkdir(parents=True, exist_ok=False)
    calculated = False
    try:
        report = calculate_fn()
        calculated = True
        helper().publish(output / "report.json", report, timely=timely)
        timely()
        return report, 0
    except Exception as error:  # noqa: BLE001 -- owned calculation/close boundary
        report = error_report(error, context=context)
        report["analysis_completed"] = calculated
        if context:
            report["consumed_input_digests"] = context["reader"].digests
            report["consumed_read_counts"] = context["reader"].read_counts
        report["provisional_report_present"] = (output / "report.json").exists()
        try:
            helper().publish(output / "failure.json", report, timely=timely)
        except Exception as secondary:  # noqa: BLE001 -- late failure remains nonzero
            report["secondary_errors"].append(
                helper().failure("failure_publication", secondary)
            )
        return report, 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", required=True)
    parser.add_argument("--qualification")
    parser.add_argument("--review")
    parser.add_argument("--output", default=OUTPUT)
    parser.add_argument("--analyze", action="store_true")
    args = parser.parse_args(argv)
    if args.analyze:
        require(
            list(sys.argv[1:] if argv is None else argv)
            == original_argv(args.freeze, args.review)[5:],
            "exact sole E352 CLI",
        )

    def timely():
        if time.perf_counter() - ENTRY > 5:
            raise TimeoutError("E352 inclusive import/read/calc/fsync/close5s")
        import psutil

        if psutil.Process().memory_info().rss > 2 << 30:
            raise MemoryError("host2GiB")

    reader = helper().EvidenceReader(timely=timely)
    if not args.analyze:
        require(args.qualification is not None, "passive freeze qualification required")
        value = make_freeze(
            args.qualification, context=metadata_preflight(reader=reader)
        )
        helper().publish(
            helper().safe(args.freeze), value, timely=timely, count_folder=False
        )
        return 0
    context = None
    try:
        context = original_gate(
            args.freeze, args.review, output=args.output, reader=reader
        )
        report, code = publish_outcome(
            helper().safe(args.output),
            lambda: calculate(context),
            timely=timely,
            context=context,
        )
    except Exception as error:  # noqa: BLE001 -- metadata rejection before numeric access
        report, code = error_report(error, stage="gate", context=context), 1
    if code:
        print(json.dumps(helper().lower(report), allow_nan=False), file=sys.stderr)
    return code


if __name__ == "__main__":
    sys.exit(main())
