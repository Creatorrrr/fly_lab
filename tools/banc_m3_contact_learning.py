"""Fixed contact loss, selected-input plan and saved arithmetic; no top-level numerics."""
from __future__ import annotations

import hashlib
import json
import math
import zipfile

SCHEMA = "flylab.m3.contact-learning"
STATE_KEYS = ("rate", "adaptation", "bank", "original_activation")
STARTS = (100, 164, 228, 292)
REPORT_CAP = 1 << 20
RECEIPTS_CAP = 8 << 20


def require(ok, label):
    if not ok:
        raise ValueError(label)


def sha(a):
    return hashlib.sha256(a.tobytes(order="C")).hexdigest()


def member_bytes(shape, dtype, cap):
    widths = {"<f4": 4, "float32": 4, "<f8": 8, "float64": 8, "<i4": 4, "int32": 4,
              "<i8": 8, "int64": 8, "|b1": 1, "bool": 1, "|u1": 1, "uint8": 1,
              "<u2": 2, "uint16": 2, "<u4": 4, "uint32": 4, "<u8": 8, "uint64": 8,
              "|i1": 1, "int8": 1, "<i2": 2, "int16": 2, "<f2": 2, "float16": 2}
    require(dtype in widths and type(shape) in (list, tuple) and 0 <= len(shape) <= 4
            and all(type(x) is int and x >= 0 for x in shape), "finite numeric shape/dtype")
    size = math.prod(shape) * widths[dtype]
    require(size <= cap, "selected member64MiB")
    return size


def metadata_fields(binding):
    require(type(binding["selected_schema"]) is dict and all(v.get("finite") is True and v.get("fortran_order", False) is False
            for v in binding["selected_schema"].values()), "pinned finite selected C-order schema")
    return {k: (tuple(v["shape"]), v["dtype"]) for k, v in binding["selected_schema"].items()}


def wire(value, cap):
    from tools.verify_c_banc_m3_candidate_body_handoff import normalize

    raw = (json.dumps(normalize(value), sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n").encode("utf8")
    require(len(raw) <= cap, "finite saved JSON reader cap")
    return raw


class PhaseBuffers:
    """Zero storage is never evidence: each returned field has its own valid bitmap."""
    def __init__(self, raw, result, np):
        self.raw, self.result, self.np = raw, result, np

    def allocate(self, key, shape, dtype="float64"):
        require(key not in self.raw, "phase allocated once")
        self.raw[key] = self.np.zeros(shape, dtype=dtype)
        self.raw[key + ".valid"] = self.np.zeros(shape[0], dtype=bool)
        require(len(self.raw) <= 256 and self.raw[key].nbytes <= 64 << 20, "bounded phase members")

    def returned(self, key, index, value):
        array, bitmap = self.raw[key], self.raw[key + ".valid"]
        require(not bitmap[index] and value.shape == array[index].shape and value.dtype == array.dtype,
                "only actual same-schema returned phase fields")
        array[index] = value
        bitmap[index] = True
        self.result.setdefault("retained_phase_fields", {})[key] = {"returned": int(bitmap.sum()), "capacity": len(bitmap),
            "valid_bitmap": key + ".valid", "invalid_storage_is_evidence": False}


def endpoint_fields(binding):
    chunk, recruitment = binding["chunks"][-1], binding["recruitment"]
    return ((chunk, {k: metadata_fields(chunk)[k] for k in ("control_tick", "network_tick", "rate", "adaptation")}),
            (recruitment, {k: metadata_fields(recruitment)[k] for k in ("control_tick", "network_tick", "bank", "original_activation", "complete", "bank_available", "original_activation_available")}))


def compare_E318_endpoint(raw, chunk, recruitment, np):
    require(np.array_equal(chunk["control_tick"], np.arange(76, 101, dtype=np.int64))
            and np.array_equal(chunk["network_tick"], chunk["control_tick"] * 20)
            and np.array_equal(recruitment["control_tick"], np.arange(301, dtype=np.int64))
            and np.array_equal(recruitment["network_tick"], recruitment["control_tick"] * 20)
            and all(bool(recruitment[k][99]) for k in ("complete", "bank_available", "original_activation_available")),
            "actual admitted E318 endpoint99 clocks/availability")
    indices = np.flatnonzero((chunk["control_tick"] == 99) & (chunk["network_tick"] == 1980))
    require(len(indices) == 1 and int(indices[0]) == 23, "unique tick99 endpoint; never chunk last tick100")
    index = int(indices[0])
    expected = {"rate": chunk["rate"][index], "adaptation": chunk["adaptation"][index],
                "bank": recruitment["bank"][99], "original_activation": recruitment["original_activation"][99]}
    checks = {}
    for key, target in expected.items():
        actual = raw["neutral_endpoint-" + key]
        require(actual.dtype == target.dtype and actual.shape == target.shape and np.isfinite(actual).all(), "saved actual baseline endpoint " + key)
        delta = actual.astype(np.float64) - target.astype(np.float64)
        maximum = float(np.max(np.abs(delta)))
        atol = 0 if key in ("rate", "adaptation") else 1e-10
        passed = bool(np.array_equal(actual, target)) if atol == 0 else maximum <= atol
        checks[key] = {"passed": passed, "exact": bool(np.array_equal(actual, target)), "max_abs": maximum,
                       "RMS": float(np.sqrt(np.mean(delta * delta))), "atol": atol, "rtol": 0, "count": int(delta.size)}
        require(passed, "independent saved baseline E318 endpoint " + key)
    return checks


def secant_diagnostic(slope, secant):
    difference = abs(secant - slope)
    return {"absolute_discrepancy": difference, "relative_discrepancy": difference / max(abs(slope), 1e-300),
            "secant_agreement_atol": 1e-6, "secant_agreement_rtol": .01,
            "secant_agreement": difference <= 1e-6 + .01 * abs(slope), "secant_agreement_is_gate": False}


def whole_npy_descriptor(spec, cap):
    """Whole-file cap includes the header; call before digest or header I/O."""
    from tools.banc_m3_saved_original_admission import descriptor
    descriptor(spec, cap)
    require(type(cap) is int and cap == 160 << 20, "unchanged whole NPY160MiB")
    require(type(spec["path"]) is str and spec["path"].endswith(".npy"), "plain whole NPY descriptor")


def whole_npy_bytes(spec, shape, dtype, version, fortran, header_bytes, stat_bytes, cap):
    """Pure header/size arithmetic; never allocates or reads an array payload."""
    whole_npy_descriptor(spec, cap)
    require(version in ((1, 0), (2, 0)), "body numeric header version")
    prefix = 10 if version == (1, 0) else 12
    require(type(header_bytes) is int and prefix < header_bytes <= prefix + 4096, "body bounded4096 header")
    require(fortran is False, "body selected numeric C order")
    require(type(stat_bytes) is int and stat_bytes == spec["bytes"], "body exact NPY stat size")
    size = member_bytes(shape, dtype, cap)
    require(size + header_bytes == spec["bytes"], "body exact preallocation NPY size")
    return size

def body_plan(admission, saved, candidate_spec, np):
    from tools import verify_c_banc_m3_coordination_parameter_direction as core
    inventory = admission.inputs["body_inventory"]["inventory"]
    selected = inventory["selected_archives"]
    files = {k: v for k, v in inventory["files"].items() if k != selected["audited_candidate"]["path"]}
    require(len(files) == 134, "same generic actual body source134")
    total, entries = 0, []
    for name, spec in files.items():
        admission.fence("pre_array")
        whole = {"path": name, "bytes": spec["bytes"], "sha256": spec["sha256"]}
        if name.endswith(".npy"):
            whole_npy_descriptor(whole, admission.limits["input_file_cap_bytes"])
        p, stamp = saved.base.digest(whole)
        if p.suffix == ".npy":
            with p.open("rb") as stream:
                version = np.lib.format.read_magic(stream)
                require(version in ((1, 0), (2, 0)), "body numeric header version")
                reader = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
                shape, fortran, dtype = reader(stream, max_header_size=4096)
                require(dtype.kind in "bifu", "body selected numeric scalar dtype")
                size = whole_npy_bytes(whole, shape, dtype.str, version, fortran, stream.tell(),
                                       p.stat().st_size, admission.limits["input_file_cap_bytes"])
            # digest alone does not register inputs: retain the planned whole NPY
            # in the unchanged SavedInputs posthash ledger, including its stamp.
            saved.base.ledger.append((whole, stamp))
            _, after = saved.base.digest(whole)
            require(after == stamp, "body NPY header/digest stamp unchanged")
        elif p.suffix == ".npz":
            binding = next(v for v in selected.values() if v["path"] == name)
            size = saved.arrays(binding, metadata_fields(binding), plan_only=True)["expanded_bytes"]
        else:
            size = spec["bytes"]  # conservative metadata/text cost retained under host cap
        total += size
        entries.append({"path": core.safe(admission.root, name).relative_to(admission.root).as_posix(), "planned_bytes": size})
    from tools.banc_m3_contact_runtime_journal import phase
    with phase(admission, "candidate-header", opaque=True):
        candidate = saved.arrays(candidate_spec, plan_only=True)["expanded_bytes"]
    require(total + candidate <= admission.limits["selected_expanded_cap_bytes"]
            and 3 * (total + candidate) + (96 << 20) <= admission.limits["host_bytes"], "all selected body512MiB and temporary host plan before existing assembly")
    return {"entries": entries, "candidate_nbytes": candidate, "selected_upper_bytes": total + candidate,
            "body_loader": "existing ray.arm_data header/digest/checked-whole gates unchanged", "MJB_reads": 0}


def preplan(admission, metadata, *, audit=False):
    from tools import analyze_c_banc_m3_startup_tracking as tracking
    from tools import verify_c_banc_m3_live_replay as replay
    from tools.observe_c_banc_m3_local_foot_geometry import pose_fields
    inv, limits = admission.inputs["legacy_inventory"], admission.limits
    entries = []

    def file(spec, fields):
        require(type(spec["bytes"]) is int and 0 < spec["bytes"] <= limits["input_file_cap_bytes"], "selected file160MiB")
        members = {k: member_bytes(shape, dtype, limits["member_cap_bytes"]) for k, (shape, dtype) in fields.items()}
        entries.append({"path": spec["path"], "members": members, "bytes": sum(members.values())})

    for role in replay.PACKET_ROLES:
        binding, fields = inv["bindings"][role], replay.PACKET_SCHEMA[role]
        require(metadata_fields(binding) == fields, "exact pinned legacy packet schema " + role)
        file(binding, fields)
    capture = admission.roster["neutral_capture"]
    file(capture["geometry"], pose_fields(metadata["dimensions"]))
    file(capture["control"], {k: (tuple(301 if x is None else x for x in shape), dtype) for k, (shape, dtype) in tracking.CONTROL_FIELDS.items()})
    if not audit:
        file(capture["context"], tracking.CONTEXT_FIELDS)
        e318 = admission.inputs["e318_startup"]
        for binding in (*e318["chunks"], e318["controls"], e318["recruitment"]):
            file(binding, metadata_fields(binding))
        whole_upper = sum(v["bytes"] for v in e318["whole_files"].values())
        require(all(type(v["bytes"]) is int and 0 < v["bytes"] <= limits["member_cap_bytes"] for v in e318["whole_files"].values()), "whole control0 finite files before restore")
        entries.append({"path": "control0-whole-122-header-plan", "members": {}, "bytes": whole_upper})
    else:
        for binding, fields in endpoint_fields(admission.inputs["e318_startup"]):
            file(binding, fields)
        binding = admission.inputs["e318_startup"]["controls"]
        file(binding, {k: metadata_fields(binding)[k] for k in ("control_tick", "network_tick", "command", "tendon")})
    startup = admission.inputs["body_inventory"]["inventory"]["selected_archives"]["startup"]
    file(startup, metadata_fields(startup))
    total = sum(e["bytes"] for e in entries)
    # Chunks, concatenate, selected copies and retained99 fields are host costs,
    # separate from the selected file payload bound and the actual work receipts.
    transient = 3 * 100 * 158706 * 4 + 3 * 99 * 158706 * 4 if not audit else 0
    output_upper = 128 << 20
    json_transient = 64 << 20 if audit else 0  # bounded8MiB complete receipts, including decoded objects
    require(total <= limits["selected_expanded_cap_bytes"] and 3 * total + transient + output_upper + json_transient <= limits["host_bytes"], "combined selected512MiB and temporary host plan before arrays")
    return {"entries": entries, "selected_upper_bytes": total, "transient_copy_upper_bytes": transient,
            "output_arrays_upper_bytes": output_upper, "receipt_JSON_host_upper_bytes": json_transient, "startup99_drive_bytes": 62847576, "MJB_reads": 0}


class Saved:
    def __init__(self, admission, np):
        from tools.analyze_c_banc_m3_startup_tracking import SavedInputs
        self.base, self.q, self.np, self.actual_bytes = SavedInputs(admission, np), admission, np, 0

    def json(self, spec):
        return self.base.json(spec)

    def arrays(self, spec, fields=None, *, plan_only=False):
        np = self.np
        self.q.fence("pre_array")
        path, stamp = self.base.digest(spec)
        self.base.ledger.append((spec, stamp))
        headers, total = {}, 0
        with zipfile.ZipFile(path) as archive:
            infos = archive.infolist()
            require(len(infos) <= 256 and len({x.filename for x in infos}) == len(infos), "finite selected archive")
            names = list(fields) if fields is not None else [x.filename.removesuffix(".npy") for x in infos]
            for key in names:
                require('/' not in key and '\\' not in key, "flat numeric member")
                info = archive.getinfo(key + ".npy")
                require(not info.flag_bits & 1 and info.compress_type in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED), "numeric ZIP compression")
                with archive.open(info) as stream:
                    version = np.lib.format.read_magic(stream)
                    require(version in ((1, 0), (2, 0)), "numeric header version")
                    reader = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
                    shape, fortran, dtype = reader(stream, max_header_size=4096)
                    require(not fortran and dtype.kind in "bifu", "selected numeric C order")
                    size = member_bytes(shape, dtype.str, self.q.limits["member_cap_bytes"])
                    require(info.file_size == stream.tell() + size, "exact preallocation NPY payload")
                    if fields is not None:
                        expected_shape, expected_dtype = fields[key]
                        require(dtype == np.dtype(expected_dtype) and len(shape) == len(expected_shape)
                                and all(a == b or b is None for a, b in zip(shape, expected_shape, strict=True)), "selected schema " + key)
                    headers[key] = (shape, dtype, size)
                    total += size
            require(self.actual_bytes + total <= self.q.limits["selected_expanded_cap_bytes"], "actual combined selected header sum before allocation")
            if plan_only:
                return {"member_bytes": {k: v[2] for k, v in headers.items()}, "expanded_bytes": total}
            result = {}
            for key, (shape, dtype, size) in headers.items():
                self.q.check("science")
                with archive.open(key + ".npy") as stream:
                    value = np.lib.format.read_array(stream, allow_pickle=False, max_header_size=4096)
                require(value.shape == shape and value.dtype == dtype and value.nbytes == size, "actual selected bytes reconciliation")
                if key != "slip_mm_s":
                    require(np.isfinite(value).all(), "finite saved " + key)
                value.flags.writeable = False
                result[key] = value
            self.actual_bytes += total
            if "slip_mm_s" in result:
                from tools.analyze_c_banc_m3_startup_tracking import (
                    validate_native_slip,
                )
                validate_native_slip(result, np)
        self.base.digest(spec)
        return result

    def posthash(self):
        return self.base.posthash()


def geometry_context(pose, metadata, saved_q, contract, np):
    from tools.observe_c_banc_m3_local_foot_geometry import ANCHORS, CONTROLS
    mapping = metadata["control_mapping"]
    require(mapping["activation_tau_s"] == 0 and mapping["interpolation"] == "held raw ctrl"
            and mapping["ctrl_to_q_mapping"]["status"] == "VERIFIED_UNIT_GEAR_JOINT_HINGE"
            and mapping["control_clamp"]["status"] == "VERIFIED_COMPILED_FLAG"
            and metadata["control_end_verified"] is True and all(metadata["control_boundary_verified"].values())
            and pose["valid"].all() and pose["jacobian_valid"].all() and pose["interval_valid"].all(), "qualified fixed held/unit/clamp geometry")
    require(tuple(pose["control"]) == CONTROLS and tuple(pose["anchor_control"]) == ANCHORS, "observed geometry boundary IDs")
    dofs, qpos = np.asarray(metadata["active_dof_ids"], np.int64), np.asarray(metadata["active_qpos_ids"], np.int64)
    require(dofs.shape == qpos.shape == (42,) and len(np.unique(dofs)) == len(np.unique(qpos)) == 42, "ordered active42 columns")
    indices = np.asarray([CONTROLS.index(c) for c in ANCHORS], np.int64)
    ends = indices + 1
    loaded = (pose["normal_load_bw"][indices] >= .02) & (pose["normal_load_bw"][ends] >= .02)
    unloaded = (pose["normal_load_bw"][indices] < .02) & (pose["normal_load_bw"][ends] < .02)
    require(int(loaded.sum()) == contract["loss"]["NL"] == 12 and int(unloaded.sum()) == contract["loss"]["NU"] == 12,
            "fixed global loaded12/unloaded12; transition UNOBSERVED")
    for j, c in enumerate(ANCHORS):
        raw = pose["interval_raw_ctrl_rad"][j]
        require(np.array_equal(raw, np.broadcast_to(raw[0], raw.shape))
                and np.array_equal(raw[0], saved_q[c + 1]), "all50 observed interval raw control equals saved c+1 command")
    return {"J": np.take(pose["jacobian_full_nv"], dofs, axis=-1), "qpos": pose["qpos"][indices][:, qpos],
            "anterior": pose["thorax_points_mm"][indices, 1] - pose["thorax_points_mm"][indices, 0],
            "loaded": loaded, "unloaded": unloaded, "limits": np.asarray(mapping["actuator_ctrlrange_rad"], np.float64),
            "limited": np.asarray(mapping["actuator_ctrllimited"], bool), "clamp_enabled": mapping["control_clamp"]["enabled"],
            "recorded_load_bearing": pose["recorded_load_bearing"][indices], "geometry_metadata": metadata}


def projection(q, frozen, window, np):
    enabled = frozen["limited"] & bool(frozen["clamp_enabled"])
    ranges = frozen["limits"]
    effective = np.where(enabled, np.clip(q, ranges[:, 0], ranges[:, 1]), q)
    derivative = np.where(enabled, (q > ranges[:, 0]) & (q < ranges[:, 1]), True).astype(np.float64)
    d = frozen["J"][window] @ (effective - frozen["qpos"][window])
    return d, derivative, {"clip_derivative": derivative.tolist(), "clip_lower_headroom": (q - ranges[:, 0]).tolist(),
                           "clip_upper_headroom": (ranges[:, 1] - q).tolist()}


def qualified_projection(measured, frozen, analysis, np):
    require(analysis["qualified"] is True and analysis["common_values_exact"] is True,
            "independent qualified geometry before learning")
    legs = ("lf", "lm", "lh", "rf", "rm", "rh")
    rows = analysis["local_table"]
    require(len(rows) == 24 and measured.shape == (4, 6, 3), "fixed24 qualified command projections")
    differences = []
    for j, c in enumerate(STARTS):
        for leg, name in enumerate(legs):
            row = rows[6 * j + leg]
            require(row["anchor_control"] == c and row["leg"] == name
                    and row["command_status"] == "VERIFIED_ACTUAL_HELD_CLAMP"
                    and row["category"] == ("loaded_loaded" if frozen["loaded"][j, leg] else "unloaded_unloaded"),
                    "qualified local row identity and actual contact category")
            expected = row["J_command"]
            differences.extend((float(measured[j, leg] @ frozen["anterior"][j]) - expected["start_thorax_AP_mm"],
                                float(measured[j, leg, 2]) - expected["world_vertical_mm"]))
    delta = np.asarray(differences, np.float64)
    maximum = float(np.max(np.abs(delta)))
    require(np.isfinite(delta).all() and maximum <= 1e-9, "actual saved-command projection matches qualified geometry")
    return {"passed": True, "max_abs_mm": maximum, "RMS_mm": float(np.sqrt(np.mean(delta * delta))),
            "count": len(differences), "atol_mm": 1e-9, "rtol": 0, "scope": "saved command AP/worldZ only; no candidate geometry"}


def window_loss(q, frozen, d0, window, coefficients, np):
    require(q.shape == (4, 42) and q.dtype == np.float64 and np.isfinite(q).all(), "finite first-only four-control command")
    d, derivative, branches = projection(q[0], frozen, window, np)
    anterior, loaded, unloaded = frozen["anterior"][window], frozen["loaded"][window], frozen["unloaded"][window]
    ap, z = d @ anterior, d[:, 2]
    pa, pz, delta = np.maximum(ap, 0), np.maximum(z, 0), d - d0[window]
    s2, nl, nu = coefficients["s_mm"] ** 2, coefficients["NL"], coefficients["NU"]
    stance = float((pa[loaded] ** 2 + pz[loaded] ** 2).sum() / (nl * s2))
    keep_numerator = float((delta[unloaded] ** 2).sum())
    keep = keep_numerator / (3 * nu * s2)
    grad_d = np.zeros((6, 3), np.float64)
    grad_d[loaded] = 2 * (pa[loaded, None] * anterior + pz[loaded, None] * np.asarray([0., 0., 1.])) / (nl * s2)
    grad_d[unloaded] = 2 * coefficients["keep_weight"] * delta[unloaded] / (3 * nu * s2)
    gq = np.zeros((4, 42), np.float64)
    gq[0] = derivative * np.einsum("lci,lc->i", frozen["J"][window], grad_d)
    branches.update(AP_active=(ap > 0).tolist(), Z_active=(z > 0).tolist(), SSP_internal_branches="NOT_OBSERVED")
    require(math.isfinite(stance + keep) and np.isfinite(gq).all(), "finite contact loss/analytic seed")
    return {"stance": stance, "keep": keep, "total": stance + coefficients["keep_weight"] * keep,
            "unloaded_squared_mm": keep_numerator, "branches": branches}, gq


def evaluate(q, frozen, d0, coefficients, np):
    rows = [window_loss(q[i], frozen, d0, i, coefficients, np)[0] for i in range(4)]
    result = {"stance": sum(x["stance"] for x in rows), "keep": sum(x["keep"] for x in rows), "total": sum(x["total"] for x in rows),
            "unloaded_RMS_mm": math.sqrt(sum(x["unloaded_squared_mm"] for x in rows) / (3 * coefficients["NU"])), "windows": rows}
    require(all(math.isfinite(result[k]) for k in ("stance", "keep", "total", "unloaded_RMS_mm")), "finite global loss")
    return result


def finite_probe_pass(base, plus, minus, represented):
    return bool(represented and plus["total"] < base["total"] and plus["total"] < minus["total"])


def eligible(base, current, startup, representation, correspondence, coefficients):
    return bool(representation and correspondence and startup["passed"] and current["stance"] < base["stance"]
                and current["total"] < base["total"] and current["unloaded_RMS_mm"] <= coefficients["unloaded_RMS_max_mm"])


def work_summary(receipts, contract, *, evaluated, probes, complete=False):
    from tools.verify_c_banc_m3_coordination_parameter_direction import work_counts
    result = work_counts(receipts)
    require(all(type(v) is int and v >= 0 for v in result.values()), "actual typed partial counts")
    require(result["forward_controls_requested"] <= contract["work"]["max_forward_controls"]
            and result["recompute_controls_requested"] <= 16 and result["backward_blocks_requested"] <= 16, "fixed actual work caps")
    confirmed = {k: 0 for k in ("forward_controls", "forward_neural_ticks", "recompute_controls", "recompute_neural_ticks", "backward_blocks", "backward_block_ticks")}
    for receipt in receipts.values():
        if receipt.get("confirmed"):
            for k in confirmed:
                confirmed[k] += receipt["confirmed"][k]
    if complete:
        expected = 115 + 16 * probes + 230 * evaluated
        for key, value in (("forward_controls", expected), ("forward_neural_ticks", expected * 20),
                           ("recompute_controls", 16), ("recompute_neural_ticks", 320), ("backward_blocks", 16), ("backward_block_ticks", 320)):
            require(result[key + "_returned"] == confirmed[key] == value, "complete actual returned/device-confirmed " + key)
            if key != "backward_block_ticks":
                require(result[key + "_requested"] == value, "complete requested work " + key)
    return {**result, "confirmed": confirmed, "body_constructors": 0, "physics_steps": 0, "warmup_ticks": 0,
            "capture_ticks": 0, "backward_block_ticks_are_additional_forward": False}
