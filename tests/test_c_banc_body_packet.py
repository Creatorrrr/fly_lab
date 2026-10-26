"""Generated-only E338 regressions; originals are never opened by these tests."""

import builtins
import copy
import io
import struct
import zipfile
from pathlib import Path

import numpy as np
import pytest

from tools import qualify_c_banc_m3_body_packet as packet


def write_archive(path, arrays, *, compression=8):
    with zipfile.ZipFile(path, "w", compression=compression) as archive:
        for key, value in arrays.items():
            stream = io.BytesIO()
            np.save(stream, value, allow_pickle=False)
            archive.writestr(key + ".npy", stream.getvalue())
    return packet.scan_archive(path, expected_sha256=packet.sha256(path))["descriptor"]


def arrays_for(role):
    arrays = {key: np.zeros(shape, dtype=dtype) for key, (shape, dtype) in packet.SPECS[role].items()}
    if role == "native":
        arrays["slip_mm_s"][:] = np.nan
    arrays["unselected_neuron"] = np.zeros((2, 3), dtype="<f4")
    return arrays


@pytest.fixture
def reference(tmp_path):
    path = tmp_path / "reference.npz"
    return path, write_archive(path, arrays_for("reference"))


def test_actual_old_nested_roster_failure_then_new_approval(tmp_path):
    from tools import probe_c_banc_m3_current_shape_counterfactual as old

    path = tmp_path / "control.npz"
    descriptor = write_archive(path, arrays_for("control"))
    frozen = {"input_members": {path.name: descriptor}}
    with pytest.raises(ValueError, match="full frozen archive key roster differs"):
        old.Dependencies.selected(path.name, ("command",), frozen, tmp_path, lambda: None, [])
    reads = []
    value = packet.decode_generated(path, "control", ["command"], descriptor=descriptor, reads=reads)
    assert reads == ["command"] and set(value) == {"command"}
    assert descriptor.keys() == {"sha256", "bytes", "members"}


@pytest.mark.parametrize("role", list(packet.SPECS))
def test_all_fixed_roles_and_selected_only(tmp_path, role, monkeypatch):
    path = tmp_path / (role + ".npz")
    descriptor = write_archive(path, arrays_for(role))
    requested = list(packet.SPECS[role])
    actual_get = np.lib.npyio.NpzFile.__getitem__
    observed = []

    def selected_only(self, key):
        assert key in requested, "unselected values reached"
        observed.append(key)
        return actual_get(self, key)

    monkeypatch.setattr(np.lib.npyio.NpzFile, "__getitem__", selected_only)
    result = packet.decode_generated(path, role, requested, descriptor=descriptor)
    assert set(result) == set(requested) and observed == requested


@pytest.mark.parametrize("field,value", [
    ("shape", [1]), ("dtype", ">f8"), ("fortran_order", True),
    ("header_bytes", 129), ("npy_version", [2, 0]), ("zip_compression", 0),
    ("compressed_bytes", 1), ("uncompressed_bytes", 1), ("crc32", 1),
    ("member", "wrong.npy"), ("crc32", True), ("shape", [True]),
    ("fortran_order", 0), ("npy_version", [True, 0]),
])
def test_exact_all_member_metadata(reference, field, value):
    path, descriptor = reference
    changed = copy.deepcopy(descriptor)
    changed["members"]["curves"][field] = value
    with pytest.raises((ValueError, TypeError)):
        packet.qualify_role(path, "reference", expected_sha256=descriptor["sha256"], descriptor=changed)


@pytest.mark.parametrize("mutation", ["missing_field", "extra_field", "missing_member", "extra_member",
                                      "missing_container", "extra_container", "sha", "bytes", "bytes_bool"])
def test_strict_nested_roster(reference, mutation):
    path, descriptor = reference
    changed = copy.deepcopy(descriptor)
    if mutation == "missing_field":
        del changed["members"]["curves"]["crc32"]
    elif mutation == "extra_field":
        changed["members"]["curves"]["unexpected"] = 1
    elif mutation == "missing_member":
        del changed["members"]["unselected_neuron"]
    elif mutation == "extra_member":
        changed["members"]["extra"] = copy.deepcopy(changed["members"]["unselected_neuron"])
        changed["members"]["extra"]["member"] = "extra.npy"
    elif mutation == "missing_container":
        del changed["bytes"]
    elif mutation == "extra_container":
        changed["unexpected"] = 1
    elif mutation == "sha":
        changed["sha256"] = "0" * 64
    elif mutation == "bytes":
        changed["bytes"] += 1
    else:
        changed["bytes"] = True
    with pytest.raises(ValueError):
        packet.scan_archive(path, expected_sha256=descriptor["sha256"], descriptor=changed)


def raw_npy(shape=(1,), descr="<f8", fortran=False, version=(1, 0), payload=None):
    header = repr({"descr": descr, "fortran_order": fortran, "shape": shape}).encode("latin1") + b"\n"
    if payload is None:
        try:
            payload = bytes(np.dtype(descr).itemsize * int(np.prod(shape)))
        except (TypeError, ValueError):
            payload = b""
    return b"\x93NUMPY" + bytes(version) + struct.pack("<H", len(header)) + header + payload


@pytest.mark.parametrize("member", ["../x.npy", "a/x.npy", "a\\x.npy", "C:x.npy", "x.txt", ".npy"])
def test_unsafe_names(tmp_path, member):
    path = tmp_path / "bad.npz"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(member, raw_npy())
    with pytest.raises(ValueError):
        packet.scan_archive(path, expected_sha256=packet.sha256(path))


@pytest.mark.parametrize("content", [raw_npy(descr=">f8"), raw_npy(descr="|O"),
                                      raw_npy(descr=[("x", "<f8")]), raw_npy(descr=("<f8", (2,))),
                                      raw_npy(shape=(-1,), payload=b""), raw_npy(shape=(True,)),
                                      raw_npy(version=(3, 0)), b"wrong", raw_npy(payload=b"")])
def test_unsafe_headers(tmp_path, content):
    path = tmp_path / "bad.npz"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("value.npy", content)
    with pytest.raises((ValueError, TypeError, SyntaxError)):
        packet.scan_archive(path, expected_sha256=packet.sha256(path))


def test_duplicate_limits_and_bounded_header(tmp_path):
    path = tmp_path / "bad.npz"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("x.npy", raw_npy())
        with pytest.warns(UserWarning):
            archive.writestr("x.npy", raw_npy())
    with pytest.raises(ValueError, match="duplicate"):
        packet.scan_archive(path, expected_sha256=packet.sha256(path))
    with zipfile.ZipFile(path, "w") as archive:
        for i in range(129):
            archive.writestr(str(i) + ".npy", raw_npy())
    with pytest.raises(ValueError, match="member count"):
        packet.scan_archive(path, expected_sha256=packet.sha256(path))
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("x.npy", b"\x93NUMPY\x01\x00" + struct.pack("<H", 10001))
    with pytest.raises(ValueError, match="bounded NPY header"):
        packet.scan_archive(path, expected_sha256=packet.sha256(path))
    # Change the declared uncompressed central-directory size, without allocating128MiB.
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("x.npy", raw_npy())
    data = bytearray(path.read_bytes())
    offset = data.index(b"PK\x01\x02")
    struct.pack_into("<I", data, offset + 24, packet.MAX_UNCOMPRESSED + 1)
    path.write_bytes(data)
    with pytest.raises(ValueError, match="decompression limit"):
        packet.scan_archive(path, expected_sha256=packet.sha256(path))


def test_fortran_nominal_endian_and_singleton(tmp_path):
    path = tmp_path / "reference.npz"
    arrays = arrays_for("reference")
    arrays["curves"] = np.asfortranarray(arrays["curves"])
    descriptor = write_archive(path, arrays)
    descriptor["members"]["curves"]["dtype"] = "float64"
    result = packet.decode_generated(path, "reference", ["curves"], descriptor=descriptor)
    assert result["curves"].flags.f_contiguous
    path = tmp_path / "singleton.npz"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("singleton.npy", raw_npy(shape=(1, 3), fortran=True))
    observed = packet.scan_archive(path, expected_sha256=packet.sha256(path))["descriptor"]
    with np.load(path, allow_pickle=False) as archive:
        singleton = archive["singleton"]
    assert singleton.flags.c_contiguous and singleton.flags.f_contiguous
    assert observed["members"]["singleton"]["fortran_order"] is True
    changed = copy.deepcopy(observed)
    changed["members"]["singleton"]["fortran_order"] = False
    with pytest.raises(ValueError, match="metadata differs"):
        packet.scan_archive(path, expected_sha256=observed["sha256"], descriptor=changed)


def test_subset_and_original_rejection_before_np_load(reference, monkeypatch):
    path, descriptor = reference
    monkeypatch.setattr(np, "load", lambda *a, **k: pytest.fail("np.load reached"))
    for role, keys in (("reference", ["unselected_neuron"]), ("native", ["slip_mm_s"]),
                       ("reference", ["curves", "curves"])):
        with pytest.raises(ValueError):
            packet.decode_generated(path, role, keys, descriptor=descriptor)
    name, pin = packet.ORIGINALS["reference"]
    with pytest.raises(ValueError, match="original path"):
        packet.decode_generated(packet.ROOT / name, "reference", ["curves"], descriptor=descriptor)
    changed = copy.deepcopy(descriptor)
    changed["sha256"] = pin
    with pytest.raises(ValueError, match="original SHA"):
        packet.decode_generated(path, "reference", ["curves"], descriptor=changed)
    actual_sha = packet.sha256
    monkeypatch.setattr(packet, "sha256", lambda p, timely=lambda: None: pin)
    with pytest.raises(ValueError, match="original SHA"):
        packet.decode_generated(path, "reference", ["curves"], descriptor=descriptor)
    monkeypatch.setattr(packet, "sha256", actual_sha)


@pytest.mark.parametrize("defect", ["negative_load", "inf", "valid_nan", "invalid_finite", "loaded_invalid"])
def test_slip_semantics(tmp_path, defect):
    arrays = arrays_for("native")
    if defect == "negative_load":
        arrays["normal_load_bw"][0, 0] = -.01
    elif defect == "inf":
        arrays["slip_mm_s"][0, 0] = np.inf
    elif defect == "valid_nan":
        arrays["slip_valid"][0, 0] = True
    elif defect == "invalid_finite":
        arrays["slip_mm_s"][0, 0] = 0
    else:
        arrays["normal_load_bw"][0, 0] = .02
    path = tmp_path / "native.npz"
    descriptor = write_archive(path, arrays)
    with pytest.raises(ValueError):
        packet.decode_generated(path, "native", ["slip_mm_s", "slip_valid", "normal_load_bw"], descriptor=descriptor)


def test_posthash_and_single_pass(reference, monkeypatch):
    path, descriptor = reference
    original_sha = packet.sha256
    original_header = packet._header
    calls = []

    def changed(p, timely=lambda: None):
        calls.append("hash")
        return original_sha(p, timely) if calls.count("hash") == 1 else "0" * 64

    def header(stream):
        calls.append("header")
        return original_header(stream)

    monkeypatch.setattr(packet, "sha256", changed)
    monkeypatch.setattr(packet, "_header", header)
    with pytest.raises(ValueError, match="posthash differs"):
        packet.scan_archive(path, expected_sha256=descriptor["sha256"], descriptor=descriptor)
    assert calls.count("header") == len(descriptor["members"]) and calls.count("hash") == 2


def test_metadata_worker_observable_forbidden_sentinels(tmp_path, monkeypatch):
    roles = {}
    for role in packet.SPECS:
        name = role + ".npz"
        descriptor = write_archive(tmp_path / name, arrays_for(role))
        roles[role] = {"path": name, "sha256": descriptor["sha256"],
                       "descriptor": descriptor if role in ("control", "native", "reference") else None,
                       "descriptor_scope": "historical_selected" if role in ("control", "native", "reference") else "observation"}
    output = packet.safe(packet.OUTPUT, tmp_path)
    output.parent.mkdir(parents=True)
    called = []

    def forbidden(*args, **kwargs):
        called.append("decoder")
        raise AssertionError("forbidden numeric decoder")

    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.startswith(("flylab", "torch", "cupy", "mujoco", "flygym")):
            called.append(name)
            raise AssertionError("forbidden body/neural/CUDA import")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setattr(np, "load", forbidden)
    monkeypatch.setattr(packet, "decode_generated", forbidden)
    monkeypatch.setattr(packet, "gate", lambda *a, **k: {"roles": roles})
    result = packet.metadata_worker("generated.json", "0" * 64, "review.json", "0" * 64, root=tmp_path)
    assert result["internal_qualified"] and result["publication_agreement"] and called == []
    assert not result["engineering_qualified"] and result["external_qualification_required"]
    assert all(v["header_passes"] == 1 for v in result["archives"].values())
    assert result["archives"]["live"]["inherited_roster_match"] is None
    assert packet.read(output / "result.json") == packet.read(output / "receipt.json") == result


def terminal():
    return {"first_error": None, "secondary_errors": [], "internal_qualified": True,
            "publication_agreement": False, "classification": "metadata_valid_pending_external"}


@pytest.mark.parametrize("fault", ["publish", "readback", "manifest_readback", "after_last_read", "all_sinks"])
def test_publication_fail_closed(tmp_path, monkeypatch, fault):
    result = terminal()
    clock = packet.Clock()
    clock.finalize()
    original_write, original_read = packet.put_json, packet.read
    writes, reads = [], []

    def writer(path, value):
        writes.append(Path(path).name)
        if fault == "all_sinks" or (fault == "publish" and len(writes) == 1):
            raise OSError("injected sink failure")
        return original_write(path, value)

    def reader(path):
        name = Path(path).name
        reads.append(name)
        value = original_read(path)
        if fault == "readback" and name == "receipt.json":
            value["internal_qualified"] = False
        if fault == "manifest_readback" and name == "output-manifest.json":
            value["files"] = {}
        if fault == "after_last_read" and name == "output-manifest.json":
            clock.deadline = 0
        return value

    monkeypatch.setattr(packet, "put_json", writer)
    monkeypatch.setattr(packet, "read", reader)
    packet.publish(tmp_path, result, clock)
    assert not result["internal_qualified"] and not result["publication_agreement"]
    assert result["first_error"]["stage"] == "publication"
    if fault == "all_sinks":
        assert len(result["secondary_errors"]) == 3 and not list(tmp_path.iterdir())
    else:
        assert not original_read(tmp_path / "result.json")["internal_qualified"]


def test_deadline_budget_first_error_and_fresh_output(tmp_path, monkeypatch):
    now = [100.]
    monkeypatch.setattr(packet.time, "perf_counter", lambda: now[0])
    clock = packet.Clock()
    now[0] = 123.
    clock.finalize()
    assert clock.deadline == 125.
    now[0] = 126.
    with pytest.raises(TimeoutError):
        clock.check()
    clock.finalize = lambda: None
    result = terminal()
    packet._error(result, "first", ValueError("preserved"))
    packet.publish(tmp_path, result, clock)
    assert result["first_error"]["message"] == "preserved" and result["secondary_errors"]
    output = packet.safe(packet.OUTPUT, tmp_path)
    output.mkdir(parents=True)
    marker = output / "old.txt"
    marker.write_text("preserve", "utf8")
    result = packet.metadata_worker("unused", "0" * 64, "unused", "0" * 64, root=tmp_path)
    assert result["first_error"]["message"] == "fresh literal output required"
    assert list(output.iterdir()) == [marker] and marker.read_text() == "preserve"


@pytest.mark.parametrize("name", ["../x", "a//x", "a/./x", "a\\x", "", "C:/x"])
def test_safe_path_binding(tmp_path, name):
    with pytest.raises(ValueError):
        packet.safe(name, tmp_path)


def test_science_timeout_and_full_hash_deadline(reference):
    path, descriptor = reference
    calls = []

    def timeout():
        calls.append(1)
        if len(calls) == 3:
            raise TimeoutError("generated science timeout")

    with pytest.raises(TimeoutError, match="science timeout"):
        packet.qualify_role(path, "reference", expected_sha256=descriptor["sha256"], timely=timeout)


def control_seven():
    return {"command": np.zeros((301, 42), "<f8"), "tendon": np.zeros((301, 6), "<f8"),
            "motor_rate32": np.zeros((301, 391), "<f4"), "control_tick": np.zeros(301, "<i8"),
            "network_tick": np.zeros(301, "<i8"), "physics_time_s": np.zeros(301, "<f8"),
            "activation": np.zeros((301, 96), "<f8")}


@pytest.fixture
def seven(tmp_path):
    path = tmp_path / "control-seven.npz"
    full = write_archive(path, control_seven())
    selected = copy.deepcopy(full)
    del selected["members"]["activation"]
    return path, full, selected


def test_seven_recorded_six_default_failure_historical_approval(seven, monkeypatch):
    path, full, recorded = seven
    with pytest.raises(packet.PacketFailure) as failed:
        packet.scan_archive(path, expected_sha256=full["sha256"], descriptor=recorded)
    detail = failed.value.detail
    assert detail["comparison"]["actual_roster"] == sorted(full["members"])
    assert len(detail["comparison"]["actual_roster"]) == 7
    assert len(detail["comparison"]["expected_roster"]) == 6
    assert detail["comparison"]["missing_recorded_members"] == []
    assert detail["comparison"]["unrecorded_members"] == ["activation"]
    assert detail["comparison"]["field_differences"] == {}
    assert detail["failure_observation"]["posthash_verified"]
    assert not detail["failure_observation"]["metadata_qualified"]
    monkeypatch.setattr(np, "load", lambda *a, **k: pytest.fail("metadata decoder reached"))
    historical = packet.qualify_role(path, "control", expected_sha256=full["sha256"], descriptor=recorded,
                                     descriptor_scope="historical_selected")
    assert historical["historical_selected_metadata_match"] is True
    assert historical["inherited_roster_match"] is None
    assert historical["unrecorded_inherited_members"] == ["activation"]
    assert historical["descriptor"] == full
    with pytest.raises(packet.PacketFailure):
        packet.decode_generated(path, "control", ["command"], descriptor=recorded)


def test_full_control_decoder_never_activation(seven, monkeypatch):
    path, full, _ = seven
    get = np.lib.npyio.NpzFile.__getitem__
    seen = []

    def selected(self, key):
        seen.append(key)
        assert key not in ("activation", "motor_rate32")
        return get(self, key)

    monkeypatch.setattr(np.lib.npyio.NpzFile, "__getitem__", selected)
    assert set(packet.decode_generated(path, "control", ["command"], descriptor=full)) == {"command"}
    assert seen == ["command"]


@pytest.mark.parametrize("scope,with_descriptor,historical,roster,unrecorded", [
    ("complete", True, None, True, []), ("complete", False, None, None, None),
    ("observation", False, None, None, None), ("historical_selected", True, True, None, []),
])
def test_scope_truth_table(reference, scope, with_descriptor, historical, roster, unrecorded):
    path, full = reference
    receipt = packet.scan_archive(path, expected_sha256=full["sha256"], descriptor=full if with_descriptor else None,
                                  descriptor_scope=scope)
    assert receipt["historical_selected_metadata_match"] is historical
    assert receipt["inherited_roster_match"] is roster
    assert receipt["unrecorded_inherited_members"] == unrecorded
    assert receipt["descriptor_scope"] == (scope if with_descriptor else "observation")


@pytest.mark.parametrize("scope,descriptor", [("wrong", None), ("historical_selected", None),
                                              ("observation", {"invalid": True})])
def test_invalid_scope_before_io(tmp_path, monkeypatch, scope, descriptor):
    monkeypatch.setattr(packet, "sha256", lambda *a, **k: pytest.fail("hash before scope rejection"))
    with pytest.raises(ValueError):
        packet.scan_archive(tmp_path / "unused.npz", expected_sha256="0" * 64,
                            descriptor=descriptor, descriptor_scope=scope)


@pytest.mark.parametrize("role", ["live", "startup"])
def test_role_scope_rejection_before_io(tmp_path, monkeypatch, role):
    monkeypatch.setattr(packet, "sha256", lambda *a, **k: pytest.fail("hash before role/scope rejection"))
    with pytest.raises(ValueError):
        packet.qualify_role(tmp_path / "unused.npz", role, expected_sha256="0" * 64,
                            descriptor={"symbolic": True}, descriptor_scope="historical_selected")


@pytest.mark.parametrize("defect", ["missing", "crc", "dtype", "order", "shape", "version", "size", "pin"])
def test_historical_recorded_metadata_must_match(seven, defect):
    path, full, recorded = seven
    if defect == "missing":
        extra = copy.deepcopy(recorded["members"]["control_tick"])
        extra["member"] = "absent.npy"
        recorded["members"]["absent"] = extra
    elif defect == "crc":
        recorded["members"]["command"]["crc32"] ^= 1
    elif defect == "dtype":
        recorded["members"]["command"]["dtype"] = "<i8"
    elif defect == "order":
        recorded["members"]["command"]["fortran_order"] = True
    elif defect == "shape":
        recorded["members"]["command"]["shape"] = [42, 301]
    elif defect == "version":
        recorded["members"]["command"]["npy_version"] = [2, 0]
    elif defect == "size":
        recorded["bytes"] += 1
    else:
        recorded["sha256"] = "0" * 64
    with pytest.raises(packet.PacketFailure) as failed:
        packet.scan_archive(path, expected_sha256=full["sha256"], descriptor=recorded,
                            descriptor_scope="historical_selected")
    assert not failed.value.detail["failure_observation"]["metadata_qualified"]
    if defect in ("crc", "dtype", "order", "shape", "version", "missing"):
        assert failed.value.detail["failure_observation"]["posthash_verified"]


@pytest.mark.parametrize("secondary", ["posthash", "timeout"])
def test_comparison_first_posthash_secondary(seven, monkeypatch, secondary):
    path, full, recorded = seven
    actual_sha = packet.sha256
    count = []

    def sha(path, timely=lambda: None):
        count.append(1)
        if len(count) == 2:
            if secondary == "timeout":
                raise TimeoutError("posthash time exhausted")
            return "0" * 64
        return actual_sha(path, timely)

    monkeypatch.setattr(packet, "sha256", sha)
    with pytest.raises(packet.PacketFailure) as failed:
        packet.qualify_role(path, "control", expected_sha256=full["sha256"], descriptor=recorded)
    detail = failed.value.detail
    assert detail["first_error"]["stage"] == "comparison"
    assert detail["secondary_errors"][0]["stage"] == "posthash"
    assert detail["failure_observation"]["full_header_scan_completed"]
    assert not detail["failure_observation"]["posthash_verified"]
    assert detail["failure_observation"]["role"] == "control"


def test_unrecorded_unsafe_partial_and_role_failure(tmp_path):
    path = tmp_path / "unsafe.npz"
    full = write_archive(path, {"curves": np.zeros((6, 128, 7), "<f8")})
    with zipfile.ZipFile(path, "a") as archive:
        archive.writestr("unsafe.npy", raw_npy(descr=">f8"))
    full.update(sha256=packet.sha256(path), bytes=path.stat().st_size)
    with pytest.raises(packet.PacketFailure) as failed:
        packet.qualify_role(path, "reference", expected_sha256=full["sha256"], descriptor=full,
                            descriptor_scope="historical_selected")
    observation = failed.value.detail["failure_observation"]
    assert not observation["full_header_scan_completed"] and not observation["posthash_verified"]
    assert set(observation["descriptor"]["members"]) == {"curves"}
    path = tmp_path / "role-wrong.npz"
    full = write_archive(path, {"curves": np.zeros((1, 1), "<f8")})
    with pytest.raises(packet.PacketFailure) as failed:
        packet.qualify_role(path, "reference", expected_sha256=full["sha256"], descriptor=full)
    detail = failed.value.detail
    assert detail["first_error"]["stage"] == "role_schema"
    assert detail["failure_observation"]["role"] == "reference"
    assert detail["failure_observation"]["posthash_verified"]
    assert not detail["failure_observation"]["metadata_qualified"]


def test_failure_detail_size_bound():
    member = {"long": "x" * 1000}
    observation = {"descriptor_scope": "complete", "descriptor": {"members": {str(i): member for i in range(128)}},
                   "full_header_scan_completed": True, "posthash_verified": False, "metadata_qualified": False}
    failed = packet.PacketFailure({"stage": "comparison", "type": "ValueError", "message": "first remains"}, observation)
    encoded = packet.json.dumps(failed.detail, ensure_ascii=True).encode("utf8")
    assert len(encoded) <= 64 << 10
    assert failed.detail["metadata_truncated"] and failed.detail["omitted_items"]["observed_members"] == 128
    assert failed.detail["first_error"]["message"] == "first remains"


@pytest.fixture
def symbolic_freeze_namespace(tmp_path, monkeypatch):
    """Real make_freeze JSON/pin tests, never an actual original closure.

    ORIGINALS stay unchanged. Their paths/SHA occur only as symbolic JSON data;
    metadata files and owned2 stubs exist only below this generated temp root.
    """
    def save(name, value):
        path = packet.safe(name, tmp_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(value, str):
            path.write_text(value, "utf8")
        else:
            packet.put_json(path, value)
        return packet.sha256(path)

    for name in packet.OWNED:
        save(name, "symbolic current owned source")
    old_design = save(packet.DESIGN, "symbolic FIRST design")
    old_review = save(packet.ASTRA, "symbolic FIRST review")
    reflection = {"all_required_items_reflected": True, "reflected_design_sha256": old_design,
                  "review_sha256": old_review}
    old_reflection = save(packet.REFLECTION, reflection)
    source_pin = save("generated-source.txt", "synthetic inherited source; no original source access")
    artifact_pin = save("generated-historical-evidence.json", {"symbolic": True})
    inputs = {name: pin for name, pin in packet.ORIGINALS.values()}
    inventory = {}
    for role in ("control", "native", "reference"):
        name, pin = packet.ORIGINALS[role]
        members = {}
        for key, (shape, dtype) in packet.SPECS[role].items():
            size = packet.math.prod(shape) * np.dtype(dtype).itemsize + 128
            members[key] = {"member": key + ".npy", "dtype": dtype, "shape": list(shape),
                            "fortran_order": False, "npy_version": [1, 0], "header_bytes": 128,
                            "zip_compression": 0, "compressed_bytes": size, "uncompressed_bytes": size,
                            "crc32": 0}
        inventory[name] = {"sha256": pin, "bytes": 100, "members": members}
    previous = {"source_sha256": {"generated-source.txt": source_pin}, "input_sha256": inputs,
                "input_members": inventory}
    fixed = {packet.PREVIOUS: save(packet.PREVIOUS, previous),
             packet.DECISION: save(packet.DECISION, {"evidence_sha256": {"generated-historical-evidence.json": artifact_pin}}),
             packet.DESIGN: old_design, packet.ASTRA: old_review, packet.REFLECTION: old_reflection,
             packet.CLARIFICATION: save(packet.CLARIFICATION, {"symbolic": True})}
    r1_design = save(packet.R1_DESIGN, "symbolic reflected R1 design")
    r1_review = save(packet.R1_ASTRA, "symbolic actual R1 review")
    r1_reflection = {"schema": "flylab.m3.e338.r1.root-design-reflection.v1",
                     "all_required_items_reflected": True, "reflected_design_sha256": r1_design,
                     "review_sha256": r1_review, "accepted": {"M" + str(i): "reflected" for i in range(1, 6)},
                     "not_accepted": {}, "provenance_sha256": {"generated-historical-evidence.json": artifact_pin}}
    r1_fixed = {packet.R1_DESIGN: r1_design, packet.R1_ASTRA: r1_review,
                packet.R1_REFLECTION: save(packet.R1_REFLECTION, r1_reflection),
                packet.ACTUAL_FAILURE: save(packet.ACTUAL_FAILURE, {"evidence_sha256": {
                    "generated-historical-evidence.json": artifact_pin}})}
    supervision = "e338-r1-supervision.json"
    supervisor_pin = save(supervision, {"symbolic_generated_namespace": True})
    proof = {"schema": "flylab.m3.e338.generated-qualification.v2", "qualified": True,
             "execution_stage": "ONE_R1", "r1_design_sha256": r1_design, "original_calls": 0,
             "owned_sha256": {name: packet.sha256(packet.safe(name, tmp_path)) for name in packet.OWNED},
             "evidence_sha256": {supervision: supervisor_pin}}
    save("proof.json", proof)
    monkeypatch.setattr(packet, "FIXED", fixed)
    monkeypatch.setattr(packet, "R1_FIXED", r1_fixed)
    actual_sha = packet.sha256
    forbidden_paths = {packet.safe(name, tmp_path) for name, _ in packet.ORIGINALS.values()}
    forbidden_paths |= {(packet.ROOT / name).resolve() for name, _ in packet.ORIGINALS.values()}

    def guarded_sha(path, timely=lambda: None):
        assert Path(path).resolve() not in forbidden_paths, "original archive/full closure accessed"
        return actual_sha(path, timely)

    monkeypatch.setattr(packet, "sha256", guarded_sha)
    return {"root": tmp_path, "save": save, "previous": previous, "reflection": r1_reflection,
            "proof": proof, "fixed": fixed, "r1_fixed": r1_fixed}


def test_real_make_freeze_scope_and_reflection(symbolic_freeze_namespace):
    fixture = symbolic_freeze_namespace
    freeze = packet.make_freeze(generated_proof="proof.json", root=fixture["root"])
    assert freeze["schema"].endswith("v2")
    packet.validate_bindings(freeze["roles"])
    for role, (name, pin) in packet.ORIGINALS.items():
        binding = freeze["roles"][role]
        assert binding["path"] == name and binding["sha256"] == pin
        assert binding["descriptor_scope"] == ("historical_selected" if role in ("control", "native", "reference") else "observation")
        assert (binding["descriptor"] is None) == (role in ("live", "startup"))
    for name in fixture["r1_fixed"]:
        assert freeze["input_sha256"][name] == fixture["r1_fixed"][name]
    assert "generated-historical-evidence.json" in freeze["input_sha256"]


@pytest.mark.parametrize("fault", ["missing_descriptor", "null_descriptor", "original_pin", "reflection_design",
                                   "reflection_review", "reflection_M", "proof_v1", "proof_current", "proof_FIRST", "metadata_pin"])
def test_real_make_freeze_failures(symbolic_freeze_namespace, fault):
    fixture = symbolic_freeze_namespace
    if fault in ("missing_descriptor", "null_descriptor", "original_pin"):
        name = packet.ORIGINALS["control"][0]
        if fault == "missing_descriptor":
            del fixture["previous"]["input_members"][name]
        elif fault == "null_descriptor":
            fixture["previous"]["input_members"][name] = None
        else:
            fixture["previous"]["input_sha256"][name] = "0" * 64
        fixture["fixed"][packet.PREVIOUS] = fixture["save"](packet.PREVIOUS, fixture["previous"])
    elif fault.startswith("reflection"):
        if fault == "reflection_design":
            fixture["reflection"]["reflected_design_sha256"] = "0" * 64
        elif fault == "reflection_review":
            fixture["reflection"]["review_sha256"] = "0" * 64
        else:
            del fixture["reflection"]["accepted"]["M5"]
        fixture["r1_fixed"][packet.R1_REFLECTION] = fixture["save"](packet.R1_REFLECTION, fixture["reflection"])
    elif fault.startswith("proof"):
        if fault == "proof_v1":
            fixture["proof"]["schema"] = "flylab.m3.e338.generated-qualification.v1"
        elif fault == "proof_current":
            fixture["proof"]["owned_sha256"][packet.OWNED[0]] = "0" * 64
        else:
            fixture["proof"]["execution_stage"] = "FIRST"
        fixture["save"]("proof.json", fixture["proof"])
    else:
        fixture["save"](packet.R1_DESIGN, "changed metadata text without updated pin")
    with pytest.raises(ValueError):
        packet.make_freeze(generated_proof="proof.json", root=fixture["root"])


@pytest.mark.parametrize("fault", ["v1", "scope_missing", "scope_wrong", "scope_unknown", "path", "pin", "live_descriptor"])
def test_gate_rejects_before_leaf_hash(symbolic_freeze_namespace, monkeypatch, fault):
    fixture = symbolic_freeze_namespace
    freeze = packet.make_freeze(generated_proof="proof.json", root=fixture["root"])
    if fault == "v1":
        freeze["schema"] = "flylab.m3.e338.body-packet.freeze.v1"
    elif fault == "scope_missing":
        del freeze["roles"]["control"]["descriptor_scope"]
    elif fault in ("scope_wrong", "scope_unknown"):
        freeze["roles"]["control"]["descriptor_scope"] = "complete" if fault == "scope_wrong" else "invalid"
    elif fault == "path":
        freeze["roles"]["control"]["path"] = "wrong.npz"
    elif fault == "pin":
        freeze["roles"]["control"]["sha256"] = "0" * 64
    else:
        freeze["roles"]["live"]["descriptor"] = freeze["roles"]["control"]["descriptor"]
    fixture["save"]("freeze.json", freeze)
    freeze_path = packet.safe("freeze.json", fixture["root"])
    pin = packet.sha256(freeze_path)
    actual_sha = packet.sha256
    reached = []

    def only_freeze(path, timely=lambda: None):
        reached.append(Path(path))
        assert Path(path) == freeze_path, "leaf/current closure hash reached before invalid gate"
        return actual_sha(path, timely)

    monkeypatch.setattr(packet, "sha256", only_freeze)
    with pytest.raises(ValueError):
        packet.gate("freeze.json", pin, "unread-review.json", "0" * 64, root=fixture["root"])
    assert reached == [freeze_path]


@pytest.mark.parametrize("fault", ["comparison", "secondary_timeout", "secondary_posthash", "role", "readback", "unsafe"])
def test_generated_worker_failure_observation_publication(tmp_path, monkeypatch, fault):
    # This local generated gate is deliberately distinct from real make_freeze
    # tests and never represents an original source/input closure qualification.
    path = tmp_path / "generated-control.npz"
    full = write_archive(path, control_seven())
    recorded = copy.deepcopy(full)
    del recorded["members"]["activation"]
    role = "control"
    if fault == "role":
        role = "reference"
        full = write_archive(path, {"curves": np.zeros((1, 1), "<f8")})
        recorded = full
    elif fault == "unsafe":
        with zipfile.ZipFile(path, "a") as archive:
            archive.writestr("unrecorded_unsafe.npy", raw_npy(descr="|O"))
        full.update(sha256=packet.sha256(path), bytes=path.stat().st_size)
        recorded.update(sha256=full["sha256"], bytes=full["bytes"])
    binding = {"path": path.name, "sha256": full["sha256"], "descriptor": recorded,
               "descriptor_scope": "historical_selected" if fault == "unsafe" else "complete"}
    monkeypatch.setattr(packet, "gate", lambda *a, **k: {"roles": {role: binding}})
    monkeypatch.setattr(np, "load", lambda *a, **k: pytest.fail("numeric decoder reached"))
    monkeypatch.setattr(packet, "decode_generated", lambda *a, **k: pytest.fail("selected decoder reached"))
    actual_sha, actual_import, actual_read = packet.sha256, builtins.__import__, packet.read
    original_paths = {(packet.ROOT / name).resolve() for name, _ in packet.ORIGINALS.values()}
    count = []

    def sha(candidate, timely=lambda: None):
        assert Path(candidate).resolve() not in original_paths, "original file/full closure accessed"
        if Path(candidate) == path:
            count.append(1)
            if len(count) == 2 and fault == "secondary_timeout":
                raise TimeoutError("posthash injected timeout")
            if len(count) == 2 and fault == "secondary_posthash":
                return "0" * 64
        return actual_sha(candidate, timely)

    def imports(name, *args, **kwargs):
        assert not name.startswith(("flylab", "torch", "cupy", "mujoco", "flygym")), "forbidden import"
        return actual_import(name, *args, **kwargs)

    def read(path):
        value = actual_read(path)
        if fault == "readback" and Path(path).name == "receipt.json":
            value["first_error"] = None
        return value

    monkeypatch.setattr(packet, "sha256", sha)
    monkeypatch.setattr(builtins, "__import__", imports)
    monkeypatch.setattr(packet, "read", read)
    output = packet.safe(packet.OUTPUT, tmp_path)
    output.parent.mkdir(parents=True)
    result = packet.metadata_worker("unused", "0" * 64, "unused", "0" * 64, root=tmp_path)
    assert result["schema"].endswith("v2") and not result["internal_qualified"]
    assert result["failure_role"] == role and not result["engineering_qualified"]
    assert "failure_detail" in actual_read(output / "result.json")
    assert actual_read(output / "result.json")["first_error"] == result["first_error"]
    if fault in ("comparison", "secondary_timeout", "secondary_posthash", "readback"):
        assert result["first_error"]["stage"] == "comparison"
        assert result["failure_detail"]["comparison"]["unrecorded_members"] == ["activation"]
    if fault.startswith("secondary"):
        assert result["secondary_errors"][0]["stage"] == "posthash"
        assert not result["failure_detail"]["failure_observation"]["posthash_verified"]
    if fault == "readback":
        assert not result["publication_agreement"] and result["secondary_errors"][-1]["stage"] == "publication"
    if fault == "unsafe":
        assert not result["failure_detail"]["failure_observation"]["full_header_scan_completed"]
