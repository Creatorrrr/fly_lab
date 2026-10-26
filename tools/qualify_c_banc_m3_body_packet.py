"""E338 independent metadata qualification and generated-only packet decoding.

No replay adapter exists here. ZIP metadata CRC is observed, not independently
computed. Hashing and ZIP buffering can read payload bytes; no metadata path
interprets array values. ROOT alone determines external engineering qualification.
"""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import math
import os
import re
import struct
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import psutil

ROOT = Path(__file__).resolve().parents[1]
BASE = "verification/banc-natural-gait-m3-20260919"
OUTPUT = BASE + "/e338-body-packet-preflight"
OWNED = ("tools/qualify_c_banc_m3_body_packet.py", "tests/test_c_banc_body_packet.py")
SCHEMA = "flylab.m3.e338.body-packet.freeze.v2"
PREVIOUS = BASE + "/e337-current-shape-counterfactual-freeze.json"
DECISION = BASE + "/parent-e337-final-decision.json"
DESIGN = "docs/BANC_M3_E338_BODY_PACKET_PREFLIGHT_DESIGN_20261001.md"
ASTRA = BASE + "/parent-e338-astra-design-review.md"
REFLECTION = BASE + "/parent-e338-astra-reflection.json"
CLARIFICATION = BASE + "/parent-e338-slip-clarification.json"
FIXED = {
    PREVIOUS: "f3cebc2d934de7c6a803b647e89802ee5248bff2de770a02c39d238c5ae1c177",
    DECISION: "c476cb52f867747ef2318398a5f8a5c7e5a03af176836bf6c7f068fd2310fe82",
    DESIGN: "37194e392d71e89e979b2de61393f46eb034932b6a6d3dd0987ae87d49f7e534",
    ASTRA: "5b8e2dbf0ca8c8aeb85feee89c0ed9b9df6d7095885ca0fe2dfff8c7e9624647",
    REFLECTION: "32c80e319be43fba9e869e420a0a9a720bfccc2f8c32ac5d5739244a2ceab1e5",
    CLARIFICATION: "ff0b3cf81b1cc259c3eeeaab2b8576a089f285c9a0dab53e2db8f07d037865f6",
}
R1_DESIGN = "docs/BANC_M3_E338_R1_SELECTED_INVENTORY_DESIGN_20261001.md"
R1_ASTRA = BASE + "/parent-e338-r1-astra-review.md"
R1_REFLECTION = BASE + "/parent-e338-r1-astra-reflection.json"
ACTUAL_FAILURE = BASE + "/parent-e338-actual-failure-final-decision.json"
R1_FIXED = {
    R1_DESIGN: "57822f0dca4a073fab714c56ab442c5c5eac78e0deed563a815acd98ed65e945",
    R1_ASTRA: "6a47f266fb023693ad24fed824672ceaffd0dcfab79e6bb96f2194f630b3d797",
    R1_REFLECTION: "6e6a873ac6a482f65487a2d024c2a9bf68e699654a61251266cf222fc8db7684",
    ACTUAL_FAILURE: "1db5cb57b3fbf90340deef6791da1ab6c399c51f84ed4dd9e198d5da3f6b4b89",
}
# These are immutable original identities, not caller-controlled deny lists.
ORIGINALS = {
    "control": (BASE + "/e335-smooth-secant-body/control-raw.npz",
                "35e0de41eefa01b2445bd77ab8740dc718403c65751a0486e57024bb97e5b072"),
    "native": (BASE + "/e335-smooth-secant-body/native-contact.npz",
               "23a61e13e4e1a452e8f6be1a957595dd44b3b273d9608c4ccc8468716e5255cc"),
    "reference": (BASE + "/e332-smooth-temporal-saved/reference.npz",
                  "140e23c865d7e110cd93699b41612c1ffb02eb88dc100ac435c4dcba5acf6c01"),
    "live": (BASE + "/e336-saved-gait-transfer/live-smooth.npz",
             "4d2ef1a8a22243b905adea9420ef7230e93e51d19d01e73615c37e84b9953ef4"),
    "startup": (BASE + "/e103-startup-preserving-recruitment/startup-inputs.npz",
                "a1b7d06907f8c2ac3f4eaee1c4a70445bc81ed3b1c0d124d77d214c77c90ad82"),
}
SPECS = {
    "control": {"command": ((301, 42), "<f8"), "control_tick": ((301,), "<i8"),
                "physics_time_s": ((301,), "<f8")},
    "native": {"time_s": ((15000,), "<f8"), "normal_load_bw": ((15000, 6), "<f8"),
               "slip_mm_s": ((15000, 6), "<f8"), "slip_valid": ((15000, 6), "|b1"),
               "upright": ((15000,), "<f8"), "nonfoot_contact": ((15000,), "|b1")},
    "reference": {"curves": ((6, 128, 7), "<f8")},
    "live": {"q": ((201, 42), "<f8"), "z": ((201, 6, 2), "<f8"),
             "R": ((201, 6), "<f8"), "phase_phase": ((201, 6), "<f8"),
             "phase_valid": ((200, 6), "|b1")},
    "startup": {"target_command": ((99, 42), "<f8"),
                "target_tendon": ((99, 6), "<f8")},
}
MEMBER_FIELDS = frozenset(("member", "dtype", "shape", "fortran_order", "npy_version",
                           "header_bytes", "zip_compression", "compressed_bytes",
                           "uncompressed_bytes", "crc32"))
LIMITS = {"science_wall_s": 20, "internal_wall_s": 25, "finalization_s": 5,
          "supervisor_wall_s": 30, "host_bytes": 2 << 30, "output_bytes": 2 << 20,
          "cpu_threads": 1, "gpu_jobs": 0, "retry_count": 0, "jobs": 1}
MAX_MEMBERS = 128
MAX_UNCOMPRESSED = 128 << 20
MAX_ARCHIVE_BYTES = 128 << 20
MAX_HEADER = 10000
MAX_FAILURE_DETAIL = 64 << 10


def _record(error, stage):
    return {"stage": stage, "type": type(error).__name__, "message": str(error)[:2048]}


def _bounded_detail(detail):
    detail = copy.deepcopy(detail)
    detail.setdefault("metadata_truncated", False)
    if len(json.dumps(detail, ensure_ascii=True, allow_nan=False).encode("utf8")) <= MAX_FAILURE_DETAIL:
        return detail
    omitted = {}
    observation = detail.get("failure_observation", {})
    descriptor = observation.get("descriptor", {})
    if "members" in descriptor:
        omitted["observed_members"] = len(descriptor["members"])
        descriptor["members"] = {}
    comparison = detail.get("comparison") or {}
    if "field_differences" in comparison:
        omitted["field_differences"] = len(comparison["field_differences"])
        comparison["field_differences"] = {}
    detail.update(metadata_truncated=True, omitted_items=omitted,
                  truncation_reason="structured metadata exceeded64KiB; member detail omitted")
    if len(json.dumps(detail, ensure_ascii=True, allow_nan=False).encode("utf8")) > MAX_FAILURE_DETAIL:
        for key in ("actual_roster", "expected_roster", "missing_recorded_members", "unrecorded_members"):
            if key in comparison:
                omitted[key] = len(comparison[key])
                comparison[key] = []
        for key in ("unrecorded_inherited_members", "selected_schema"):
            if key in observation:
                omitted[key] = len(observation[key])
                del observation[key]
    return detail


class PacketFailure(ValueError):
    def __init__(self, first_error, observation, *, comparison=None, secondary_errors=()):
        super().__init__(first_error["message"])
        self.detail = _bounded_detail({"first_error": first_error, "secondary_errors": list(secondary_errors),
                                      "descriptor_scope": observation["descriptor_scope"],
                                      "comparison": comparison, "failure_observation": observation})


class PacketTimeout(PacketFailure, TimeoutError):
    pass


def _scope(scope, descriptor):
    if scope not in ("complete", "historical_selected", "observation"):
        raise ValueError("unknown descriptor scope")
    if scope == "complete" and descriptor is None:
        return "observation"  # Existing no-descriptor API compatibility, never historical fallback.
    if (scope == "historical_selected" and descriptor is None
            or scope == "observation" and descriptor is not None):
        raise ValueError("descriptor/scope combination differs")
    return scope


def safe(name, root=ROOT):
    if not isinstance(name, str) or not name or "\\" in name:
        raise ValueError("canonical repository-relative path required")
    path = Path(name)
    if path.is_absolute() or any(p in ("", ".", "..") for p in name.split("/")):
        raise ValueError("unsafe relative path")
    resolved = (Path(root) / path).resolve()
    if not resolved.is_relative_to(Path(root).resolve()):
        raise ValueError("path escapes root")
    return resolved


def read(path):
    return json.loads(Path(path).read_text("utf8"))


def check_sha(value):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("lowercase SHA256 required")
    return value


def sha256(path, timely=lambda: None):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while True:
            timely()
            block = stream.read(1 << 20)
            if not block:
                break
            digest.update(block)
    timely()
    return digest.hexdigest()


def dtype_metadata(value):
    # Historical nominal descriptors explicitly mean little endian here.
    nominal = {"float64": "<f8", "float32": "<f4", "int64": "<i8",
               "int32": "<i4", "bool": "|b1"}
    if not isinstance(value, str) or not value:
        raise ValueError("dtype string required")
    value = nominal.get(value, value)
    if value.startswith("="):
        raise ValueError("host-native byte order forbidden")
    dtype = np.dtype(value)
    if (dtype.hasobject or dtype.fields is not None or dtype.subdtype is not None
            or dtype.kind not in "bifu" or dtype.itemsize not in (1, 2, 4, 8)):
        raise ValueError("scalar numeric/bool dtype required")
    if dtype.itemsize > 1 and not value.startswith("<"):
        raise ValueError("explicit little-endian dtype required")
    return dtype.str


def member_schema(key, spec):
    if (not isinstance(key, str) or not 0 < len(key) <= 128
            or re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]*", key) is None
            or not isinstance(spec, dict) or set(spec) != MEMBER_FIELDS):
        raise ValueError("strict member schema required")
    if spec["member"] != key + ".npy":
        raise ValueError("member name differs")
    normalized = copy.deepcopy(spec)
    normalized["dtype"] = dtype_metadata(spec["dtype"])
    shape = spec["shape"]
    if (not isinstance(shape, list) or len(shape) > 8
            or any(type(v) is not int or v < 0 for v in shape)):
        raise ValueError("bounded shape required")
    if type(spec["fortran_order"]) is not bool:
        raise ValueError("boolean ordering required")
    if (not isinstance(spec["npy_version"], list)
            or any(type(v) is not int for v in spec["npy_version"])
            or spec["npy_version"] not in ([1, 0], [2, 0])):
        raise ValueError("supported NPY version required")
    for field in ("header_bytes", "zip_compression", "compressed_bytes",
                  "uncompressed_bytes", "crc32"):
        if type(spec[field]) is not int or spec[field] < 0:
            raise ValueError("nonnegative integer metadata required: " + field)
    if (spec["zip_compression"] not in (0, 8) or spec["crc32"] > 0xFFFFFFFF
            or not 10 <= spec["header_bytes"] <= MAX_HEADER + 12
            or spec["uncompressed_bytes"] > MAX_UNCOMPRESSED
            or spec["compressed_bytes"] > MAX_ARCHIVE_BYTES):
        raise ValueError("metadata bound exceeded")
    expected = math.prod(shape) * np.dtype(normalized["dtype"]).itemsize
    if spec["uncompressed_bytes"] != expected + spec["header_bytes"]:
        raise ValueError("declared NPY size differs")
    return normalized


def archive_schema(descriptor):
    if not isinstance(descriptor, dict) or set(descriptor) != {"sha256", "bytes", "members"}:
        raise ValueError("strict nested archive descriptor required")
    check_sha(descriptor["sha256"])
    if type(descriptor["bytes"]) is not int or not 0 < descriptor["bytes"] <= MAX_ARCHIVE_BYTES:
        raise ValueError("bounded archive byte length required")
    members = descriptor["members"]
    if not isinstance(members, dict) or not 0 < len(members) <= MAX_MEMBERS:
        raise ValueError("bounded full member roster required")
    normalized = {key: member_schema(key, spec) for key, spec in members.items()}
    if sum(v["uncompressed_bytes"] for v in normalized.values()) > MAX_UNCOMPRESSED:
        raise ValueError("declared archive decompression bound exceeded")
    return {"sha256": descriptor["sha256"], "bytes": descriptor["bytes"], "members": normalized}


def _exact_read(stream, count):
    value = stream.read(count)
    if len(value) != count:
        raise ValueError("truncated NPY header")
    return value


def _header(stream):
    if _exact_read(stream, 6) != b"\x93NUMPY":
        raise ValueError("NPY magic required")
    version = tuple(_exact_read(stream, 2))
    if version not in ((1, 0), (2, 0)):
        raise ValueError("unsupported NPY version")
    size = struct.unpack("<H" if version == (1, 0) else "<I",
                         _exact_read(stream, 2 if version == (1, 0) else 4))[0]
    if not 0 < size <= MAX_HEADER:
        raise ValueError("bounded NPY header required")
    text = _exact_read(stream, size).decode("latin1")
    if not text.endswith("\n"):
        raise ValueError("NPY header newline required")
    tree = ast.parse(text.strip(), mode="eval")
    if (not isinstance(tree.body, ast.Dict) or len(tree.body.keys) != 3
            or {ast.literal_eval(k) for k in tree.body.keys} != {"descr", "fortran_order", "shape"}):
        raise ValueError("strict NPY header mapping required")
    fields = ast.literal_eval(tree)
    if type(fields["fortran_order"]) is not bool or not isinstance(fields["shape"], tuple):
        raise ValueError("typed NPY shape/order required")
    return fields, list(version), stream.tell()


def scan_archive(path, *, expected_sha256, descriptor=None, descriptor_scope="complete", timely=lambda: None):
    """One header pass, bounded metadata observation, then mandatory posthash."""
    path = Path(path)
    check_sha(expected_sha256)
    scope = _scope(descriptor_scope, descriptor)
    members = {}
    observation = {"descriptor_scope": scope, "descriptor": {"members": members},
                   "full_header_scan_completed": False, "posthash_verified": False,
                   "metadata_qualified": False, "header_passes": 0,
                   "historical_selected_metadata_match": None, "inherited_roster_match": None,
                   "unrecorded_inherited_members": None, "numeric_decode_calls": 0,
                   "crc_policy": "observed_zip_field_only"}
    try:
        normalized = archive_schema(descriptor) if descriptor is not None else None
        if normalized is not None and normalized["sha256"] != expected_sha256:
            raise ValueError("descriptor/source pin differs")
        timely()
        length = path.stat().st_size
        observation["descriptor"]["bytes"] = length
        if not 0 < length <= MAX_ARCHIVE_BYTES:
            raise ValueError("archive length limit")
        if normalized is not None and length != normalized["bytes"]:
            raise ValueError("archive byte length differs")
        before = sha256(path, timely)
        observation.update(prehash_sha256=before)
        observation["descriptor"]["sha256"] = before
        if before != expected_sha256:
            raise ValueError("archive SHA differs")
        with zipfile.ZipFile(path) as archive:
            infos = archive.infolist()
            if not 0 < len(infos) <= MAX_MEMBERS:
                raise ValueError("archive member count limit")
            if sum(info.file_size for info in infos) > MAX_UNCOMPRESSED:
                raise ValueError("archive decompression limit")
            observation["header_passes"] = 1
            for info in infos:
                timely()
                name = info.filename
                if (not name.endswith(".npy") or "/" in name or "\\" in name
                        or ":" in name or info.flag_bits & 1):
                    raise ValueError("unsafe ZIP member")
                key = name[:-4]
                if key in members:
                    raise ValueError("duplicate ZIP member")
                if info.compress_type not in (0, 8):
                    raise ValueError("unsupported ZIP compression")
                with archive.open(info) as stream:
                    fields, version, header_bytes = _header(stream)
                spec = {"member": name, "dtype": fields["descr"], "shape": list(fields["shape"]),
                        "fortran_order": fields["fortran_order"], "npy_version": version,
                        "header_bytes": header_bytes, "zip_compression": info.compress_type,
                        "compressed_bytes": info.compress_size, "uncompressed_bytes": info.file_size,
                        "crc32": info.CRC}
                members[key] = member_schema(key, spec)
        observation.update(full_header_scan_completed=True,
                           header_bytes=sum(v["header_bytes"] for v in members.values()))
    except Exception as error:  # Partial unsafe observations must remain failures.
        cls = PacketTimeout if isinstance(error, TimeoutError) else PacketFailure
        raise cls(_record(error, "scan"), observation) from error
    comparison = None
    first = None
    secondary = []
    if normalized is not None:
        recorded = normalized["members"]
        missing, unrecorded = sorted(set(recorded) - set(members)), sorted(set(members) - set(recorded))
        differences = {key: {field: {"expected": recorded[key][field], "observed": members[key][field]}
                             for field in MEMBER_FIELDS if recorded[key][field] != members[key][field]}
                       for key in sorted(set(recorded) & set(members))}
        differences = {key: value for key, value in differences.items() if value}
        comparison = {"descriptor_scope": scope, "expected_roster": sorted(recorded),
                      "actual_roster": sorted(members), "missing_recorded_members": missing,
                      "unrecorded_members": unrecorded, "field_differences": differences}
        if missing or differences or (scope == "complete" and unrecorded):
            first = _record(ValueError("full inherited archive metadata differs (" + scope + ")"), "comparison")
    try:
        after = sha256(path, timely)
        observation["posthash_sha256"] = after
        if after != before or path.stat().st_size != length:
            raise ValueError("archive posthash differs")
        timely()
        observation["posthash_verified"] = True
    except Exception as error:  # noqa: BLE001 -- Preserve comparison first; posthash errors are secondary.
        record = _record(error, "posthash")
        if first is None:
            first = record
        else:
            secondary.append(record)
    if first is not None:
        cls = PacketTimeout if first["type"] == "TimeoutError" else PacketFailure
        raise cls(first, observation, comparison=comparison, secondary_errors=secondary)
    observation.update(metadata_qualified=True,
                       historical_selected_metadata_match=True if scope == "historical_selected" else None,
                       inherited_roster_match=True if scope == "complete" else None,
                       unrecorded_inherited_members=(comparison["unrecorded_members"] if comparison else None))
    return observation


def selected_schema(role):
    if role not in SPECS:
        raise ValueError("unknown archive role")
    return {key: {"shape": list(shape), "dtype": dtype, "finite": key != "slip_mm_s"}
            for key, (shape, dtype) in SPECS[role].items()}


def qualify_role(path, role, *, expected_sha256, descriptor=None, descriptor_scope="complete", timely=lambda: None):
    specs = selected_schema(role)
    if descriptor_scope == "historical_selected" and role in ("live", "startup"):
        raise ValueError("live/startup cannot have historical selected descriptors")
    try:
        receipt = scan_archive(path, expected_sha256=expected_sha256, descriptor=descriptor,
                               descriptor_scope=descriptor_scope, timely=timely)
    except PacketFailure as error:
        error.detail["failure_observation"]["role"] = role
        error.detail = _bounded_detail(error.detail)
        raise
    members = receipt["descriptor"]["members"]
    for key, spec in specs.items():
        if key not in members or any(members[key][field] != spec[field] for field in ("shape", "dtype")):
            receipt.update(role=role, metadata_qualified=False)
            comparison = {"selected_key": key, "expected": spec, "observed": members.get(key)}
            raise PacketFailure(_record(ValueError("fixed selected role schema differs: " + key), "role_schema"),
                                receipt, comparison=comparison)
    receipt.update(role=role, selected_schema=specs, leg_order=["lf", "lm", "lh", "rf", "rm", "rh"])
    return receipt


def decode_generated(path, role, keys, *, descriptor, timely=lambda: None, reads=None):
    """Reject original identities; validate full metadata before selected values."""
    specs = selected_schema(role)
    if (not isinstance(keys, (list, tuple)) or not keys or len(set(keys)) != len(keys)
            or any(not isinstance(k, str) or k not in specs for k in keys)):
        raise ValueError("requested keys must be a nonempty role subset")
    if "slip_mm_s" in keys and not {"slip_valid", "normal_load_bw"}.issubset(keys):
        raise ValueError("slip companions must be explicitly requested")
    path = Path(path).resolve()
    if path in {(ROOT / name).resolve() for name, _ in ORIGINALS.values()}:
        raise ValueError("original path cannot be decoded")
    normalized = archive_schema(descriptor)
    forbidden = {digest for _, digest in ORIGINALS.values()}
    if normalized["sha256"] in forbidden or sha256(path, timely) in forbidden:
        raise ValueError("original SHA cannot be decoded, including copies")
    qualify_role(path, role, expected_sha256=normalized["sha256"], descriptor=descriptor,
                 descriptor_scope="complete", timely=timely)
    values = {}
    with np.load(path, allow_pickle=False) as archive:
        for key in keys:
            timely()
            value = archive[key]
            if reads is not None:
                reads.append(key)
            spec = specs[key]
            if list(value.shape) != spec["shape"] or value.dtype.str != spec["dtype"]:
                raise ValueError("decoded selected dtype/shape differs")
            if key != "slip_mm_s" and not np.isfinite(value).all():
                raise ValueError("selected nonfinite value")
            values[key] = value
    if "normal_load_bw" in values and np.any(values["normal_load_bw"] < 0):
        raise ValueError("negative normal load")
    if "slip_mm_s" in values:
        slip, valid, load = (values[k] for k in ("slip_mm_s", "slip_valid", "normal_load_bw"))
        if (not np.array_equal(np.isfinite(slip), valid) or np.isinf(slip).any()
                or not np.isnan(slip[~valid]).all() or np.any((load >= .02) & ~valid)):
            raise ValueError("slip validity/load semantics differ")
    if sha256(path, timely) != normalized["sha256"]:
        raise ValueError("decode posthash differs")
    timely()
    return values


def _merge(pins, additions):
    for name, digest in additions.items():
        safe(name)
        check_sha(digest)
        if name in pins and pins[name] != digest:
            raise ValueError("conflicting inherited pin: " + name)
        pins[name] = digest


def _pinned_json(name, pin, root, timely=lambda: None):
    path = safe(name, root)
    if sha256(path, timely) != pin:
        raise ValueError("fixed JSON identity differs: " + name)
    return read(path)


def validate_bindings(roles):
    if not isinstance(roles, dict) or set(roles) != set(ORIGINALS):
        raise ValueError("exact five role bindings required")
    for role, (name, pin) in ORIGINALS.items():
        binding = roles[role]
        expected_scope = "historical_selected" if role in ("control", "native", "reference") else "observation"
        if (not isinstance(binding, dict) or set(binding) != {"path", "sha256", "descriptor", "descriptor_scope"}
                or binding["path"] != name or binding["sha256"] != pin
                or binding["descriptor_scope"] != expected_scope):
            raise ValueError("fixed role/path/pin/scope differs: " + role)
        if expected_scope == "historical_selected":
            normalized = archive_schema(binding["descriptor"])
            if normalized["sha256"] != pin:
                raise ValueError("role descriptor pin differs")
        elif binding["descriptor"] is not None:
            raise ValueError("live/startup observation requires no inherited descriptor")


def make_freeze(*, generated_proof, root=ROOT, timely=lambda: None):
    """Read frozen JSON only; inherited full-file hashing belongs to ROOT gate.

    generated_proof is a current ROOT-created qualification JSON. It must pin
    owned2 and every generated command/contract/supervision/terminal proof file.
    Mutable progress/checkpoint/cost files are deliberately excluded.
    """
    previous = _pinned_json(PREVIOUS, FIXED[PREVIOUS], root, timely)
    decision = _pinned_json(DECISION, FIXED[DECISION], root, timely)
    reflection = _pinned_json(REFLECTION, FIXED[REFLECTION], root, timely)
    if (not reflection["all_required_items_reflected"]
            or reflection["reflected_design_sha256"] != FIXED[DESIGN]
            or reflection["review_sha256"] != FIXED[ASTRA]):
        raise ValueError("current reflected Astra review required")
    for name, pin in {**FIXED, **R1_FIXED}.items():
        if sha256(safe(name, root), timely) != pin:
            raise ValueError("immutable design/history pin differs: " + name)
    r1 = _pinned_json(R1_REFLECTION, R1_FIXED[R1_REFLECTION], root, timely)
    if (r1.get("schema") != "flylab.m3.e338.r1.root-design-reflection.v1"
            or r1.get("all_required_items_reflected") is not True
            or r1.get("reflected_design_sha256") != R1_FIXED[R1_DESIGN]
            or r1.get("review_sha256") != R1_FIXED[R1_ASTRA]
            or set(r1.get("accepted", {})) != {"M1", "M2", "M3", "M4", "M5"}
            or r1.get("not_accepted") != {} or not r1.get("provenance_sha256")):
        raise ValueError("exact current R1 reflection required")
    actual_failure = _pinned_json(ACTUAL_FAILURE, R1_FIXED[ACTUAL_FAILURE], root, timely)
    current = {name: sha256(safe(name, root), timely) for name in OWNED}
    proof = read(safe(generated_proof, root))
    if (proof.get("schema") != "flylab.m3.e338.generated-qualification.v2"
            or proof.get("owned_sha256") != current or proof.get("qualified") is not True
            or proof.get("original_calls") != 0 or not proof.get("evidence_sha256")
            or proof.get("execution_stage") != "ONE_R1"
            or proof.get("r1_design_sha256") != R1_FIXED[R1_DESIGN]
            or not any("e338" in name and "r1" in name and "supervision" in name
                       for name in proof["evidence_sha256"])):
        raise ValueError("current generated qualification required")
    sources = previous["source_sha256"].copy()
    inputs = previous["input_sha256"].copy()
    _merge(sources, current)
    immutable = {**FIXED, **R1_FIXED}
    _merge(inputs, immutable)
    _merge(inputs, decision["evidence_sha256"])
    _merge(inputs, actual_failure["evidence_sha256"])
    _merge(inputs, r1["provenance_sha256"])
    _merge(inputs, {generated_proof: sha256(safe(generated_proof, root), timely)})
    _merge(inputs, proof["evidence_sha256"])
    roles = {}
    for role, (name, pin) in ORIGINALS.items():
        if inputs.get(name) != pin:
            raise ValueError("original role pin differs")
        historical = role in ("control", "native", "reference")
        if historical and name not in previous["input_members"]:
            raise ValueError("historical descriptor required: " + role)
        descriptor = previous["input_members"][name] if historical else None
        if historical:
            archive_schema(descriptor)
            if descriptor["sha256"] != pin:
                raise ValueError("original descriptor pin differs")
        roles[role] = {"path": name, "sha256": pin, "descriptor": descriptor,
                       "descriptor_scope": "historical_selected" if historical else "observation"}
    validate_bindings(roles)
    return {"schema": SCHEMA, "output": OUTPUT, "limits": LIMITS.copy(), "roles": roles,
            "source_sha256": sources, "input_sha256": inputs, "owned_sha256": current,
            "generated_proof": generated_proof, "inherited_source_sha256": previous["source_sha256"],
            "inherited_input_sha256": previous["input_sha256"], "original_numeric_authorized": False,
            "candidate_adopted": False, "F1_F8_complete": False}


def gate(freeze_path, freeze_sha256, review_path, review_sha256, *, root=ROOT, timely=lambda: None):
    timely()
    path = safe(freeze_path, root)
    if sha256(path, timely) != check_sha(freeze_sha256):
        raise ValueError("freeze byte pin differs")
    freeze = read(path)
    if (freeze.get("schema") != SCHEMA or freeze.get("output") != OUTPUT
            or freeze.get("limits") != LIMITS or freeze.get("original_numeric_authorized") is not False):
        raise ValueError("fixed metadata freeze contract required")
    validate_bindings(freeze.get("roles"))
    expected = make_freeze(generated_proof=freeze["generated_proof"], root=root, timely=timely)
    if freeze != expected:
        raise ValueError("exact current immutable closure required")
    review_file = safe(review_path, root)
    if sha256(review_file, timely) != check_sha(review_sha256):
        raise ValueError("ROOT review byte pin differs")
    review = read(review_file)
    if (review.get("schema") != "flylab.m3.e338.source-review.v2"
            or review.get("qualified") is not True or review.get("freeze_sha256") != freeze_sha256
            or review.get("owned_sha256") != freeze["owned_sha256"]):
        raise ValueError("ROOT current source review required")
    for pins in (freeze["source_sha256"], freeze["input_sha256"]):
        for name, digest in pins.items():
            timely()
            if sha256(safe(name, root), timely) != digest:
                raise ValueError("immutable source/input pin differs: " + name)
    timely()
    if sha256(path, timely) != freeze_sha256 or sha256(review_file, timely) != review_sha256:
        raise ValueError("gate terminal posthash differs")
    return freeze


class Clock:
    def __init__(self, s0=None):
        self.s0 = time.perf_counter() if s0 is None else s0
        self.deadline = self.s0 + LIMITS["science_wall_s"]
        self.final_begin = None
        self.last_rss = self.s0 - .5
        self.peak_rss = 0

    def check(self):
        now = time.perf_counter()
        if now > self.deadline:
            raise TimeoutError("absolute worker deadline exceeded")
        if now - self.last_rss >= .5:
            self.peak_rss = max(self.peak_rss, psutil.Process(os.getpid()).memory_info().rss)
            self.last_rss = now
            if self.peak_rss > LIMITS["host_bytes"]:
                raise MemoryError("worker RSS limit exceeded")

    def finalize(self):
        self.final_begin = time.perf_counter()
        self.deadline = min(self.s0 + LIMITS["internal_wall_s"],
                            self.final_begin + LIMITS["finalization_s"])
        self.check()


def put_json(path, value):
    path = Path(path)
    pending = path.with_name(path.name + ".pending")
    with pending.open("w", encoding="utf8", newline="\n") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(pending, path)


def _error(result, stage, error):
    record = _record(error, stage)
    if isinstance(error, PacketFailure):
        record = error.detail["first_error"]
        result["failure_detail"] = error.detail
        result["failure_role"] = error.detail["failure_observation"].get("role")
    if result["first_error"] is None:
        result["first_error"] = record
    else:
        result["secondary_errors"].append(record)
    if isinstance(error, PacketFailure):
        result["secondary_errors"].extend(error.detail["secondary_errors"])
    result.update(internal_qualified=False, publication_agreement=False,
                  classification="technical_invalid")


def publish(output, result, clock):
    """Three agreeing terminal sinks; failures retain first error and revoke.

    Revocation uses all sinks even after the deadline. Its success is not claimed
    as agreement; stale disk success always loses to the returned error/exit1.
    """
    output = Path(output)
    try:
        clock.check()
        result["publication_agreement"] = True
        files = {"result.json": result, "receipt.json": copy.deepcopy(result)}
        for name, value in files.items():
            clock.check()
            put_json(output / name, value)
            clock.check()
        manifest = {"schema": "flylab.m3.e338.publication.v2", "files": {
            name: {"sha256": sha256(output / name, clock.check), "bytes": (output / name).stat().st_size}
            for name in files}}
        put_json(output / "output-manifest.json", manifest)
        clock.check()
        if sum(p.stat().st_size for p in output.iterdir() if p.is_file()) > LIMITS["output_bytes"]:
            raise ValueError("output byte limit exceeded")
        for name, value in files.items():
            if (read(output / name) != value
                    or sha256(output / name, clock.check) != manifest["files"][name]["sha256"]):
                raise ValueError("last terminal readback differs")
            clock.check()
        if read(output / "output-manifest.json") != manifest:
            raise ValueError("last manifest readback differs")
        clock.check()
        return result
    except Exception as error:  # noqa: BLE001 -- Any terminal sink failure must revoke success.
        _error(result, "publication", error)
        for name in ("result.json", "receipt.json", "output-manifest.json"):
            try:
                put_json(output / name, result)
            except Exception as secondary:  # noqa: BLE001 -- Preserve every failed revocation sink.
                _error(result, "revocation:" + name, secondary)
        return result


def metadata_worker(freeze_path, freeze_pin, review_path, review_pin, *, root=ROOT):
    # s0 precedes gate; hashing uses cheap wall checks only. Outer supervisor
    # samples the owned process tree RSS at 100 ms; no per-block tree walks.
    clock = Clock()
    result = {"schema": "flylab.m3.e338.body-packet.result.v2", "classification": "technical_invalid",
              "internal_qualified": False, "engineering_qualified": False,
              "external_qualification_required": True, "publication_agreement": False,
              "first_error": None, "secondary_errors": [], "archives": {},
              "selected_numeric_reads": 0, "numeric_analysis_calls": 0,
              "body_constructors": 0, "neural_calls": 0, "CUDA_calls": 0,
              "original_numeric_calls": 0, "H337": "not_evaluated", "candidate_adopted": False,
              "F1_F8_complete": False, "science_deadline": clock.deadline,
              "s0": clock.s0, "RSS_policy": "outer_owned_tree_100ms; internal_self_500ms",
              "hash_payload_bytes_read_possible": True, "crc_independently_recomputed": False}
    output = safe(OUTPUT, root)
    try:
        if output.exists():
            raise ValueError("fresh literal output required")
        output.mkdir()
        freeze = gate(freeze_path, freeze_pin, review_path, review_pin, root=root, timely=clock.check)
        for role, binding in freeze["roles"].items():
            result["archives"][role] = qualify_role(safe(binding["path"], root), role,
                                                    expected_sha256=binding["sha256"],
                                                    descriptor=binding["descriptor"],
                                                    descriptor_scope=binding["descriptor_scope"], timely=clock.check)
        clock.check()
        result.update(internal_qualified=True, classification="metadata_valid_pending_external")
    except Exception as error:  # noqa: BLE001 -- A preflight failure cannot bypass terminal invalidation.
        _error(result, "preflight", error)
    try:
        clock.finalize()
    except Exception as error:  # noqa: BLE001 -- Deadline/resource failures require invalid terminals.
        _error(result, "finalization", error)
    result.update(final_begin=clock.final_begin, final_deadline=clock.deadline)
    # Never overwrite a pre-existing directory, including failed prior attempts.
    if not output.exists() or (result["first_error"] and result["first_error"]["message"] == "fresh literal output required"):
        return result
    return publish(output, result, clock)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", required=True)
    parser.add_argument("--freeze-sha256", required=True)
    parser.add_argument("--review", required=True)
    parser.add_argument("--review-sha256", required=True)
    args = parser.parse_args()
    result = metadata_worker(args.freeze, args.freeze_sha256, args.review, args.review_sha256)
    print(json.dumps(result, ensure_ascii=True, allow_nan=False))
    return 0 if result["internal_qualified"] and result["publication_agreement"] else 1


if __name__ == "__main__":
    sys.exit(main())
