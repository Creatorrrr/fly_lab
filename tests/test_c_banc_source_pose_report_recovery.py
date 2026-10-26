"""E351 small generated publication/admission/readers; original arrays untouched."""

import copy
import io
import json
import subprocess
import sys
import zipfile

import numpy as np
import pytest

from tools import recover_c_banc_m3_source_pose_report as p


def test_standalone_entrypoint_resolves_current_workspace_gait(tmp_path):
    code = """
import runpy, sys
from pathlib import Path
p = runpy.run_path(sys.argv[1])
from tools import analyze_c_banc_biological_gait as gait
assert Path(gait.__file__).resolve() == Path(p['ROOT'], 'tools/analyze_c_banc_biological_gait.py')
assert p['sha'](Path(gait.__file__).read_bytes()) == 'bcceb5d1e9f26293ffc800d09a1fbf063a1ab784d73cf5489b74c57ecdd14760'
print('standalone current workspace import passed; original arrays/body/NN/GPU zero')
"""
    job = subprocess.run(
        [sys.executable, "-X", "utf8", "-B", "-c", code, str(p.ROOT / p.OWNED[0])],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
        timeout=3,
    )
    assert job.returncode == 0, job.stderr
    assert "standalone current workspace import passed" in job.stdout


@pytest.mark.parametrize("raw", [b'{"value":1e999}', b'{"value":-1e999}'])
def test_json_exponent_overflow_rejected_without_arrays(raw):
    with pytest.raises(ValueError, match="nonfinite JSON numeric"):
        p.strict_json(raw)


def metrics(*, health=True, missing=False):
    from tools import analyze_c_banc_biological_gait as gait

    start, end = gait.runs(np.array([False] + [True] * 65 + [False]))[0]
    bout = (end - start - 1) * 0.005  # actual NumPy endpoints, Python dt.
    result = {}
    for arm, forward, slip in (("B", 1.0, 2.0), ("I", 1.2, 2.0), ("C", 1.8, 1.7)):
        result[arm] = {
            "full_status": "evaluated",
            "physical": {
                "forward_mm": forward,
                "slip_mm_s": None if missing else slip,
                "measured_native_healthy": health,
                "all_control_faults_clear": True,
            },
            "gait": {
                "window": {"longest_forward_bout_s": bout},
                "events": {leg: {"complete_cycles": 0} for leg in p.LEGS},
            },
        }
    assert isinstance(bout, np.floating)
    return result


@pytest.mark.parametrize("health", [True, np.bool_(True)])
def test_actual_frozen_hypothesis_then_owned_publication(tmp_path, health):
    values = metrics(health=health)
    original = copy.deepcopy(values)
    typed = p.diagnostic_hypothesis(values)
    assert type(typed["gates"]["bout"]) is np.bool_
    assert typed["gates"]["health"] is (health is True)
    with pytest.raises(TypeError):
        json.dumps(typed, allow_nan=False)
    report, code = p.publish_outcome(
        tmp_path / "out",
        lambda: {
            "diagnostic_verified": True,
            "diagnostic_H348": p.diagnostic_hypothesis(values),
        },
    )
    assert code == 0 and values == original
    saved = json.loads((tmp_path / "out" / "report.json").read_text())
    assert saved == p.lower(report)
    assert saved["diagnostic_H348"]["gates"]["health"] is (health is True)
    assert saved["diagnostic_H348"]["verdict"] == "partial_or_inconclusive"


def test_missing_metric_stays_none_not_zero(tmp_path):
    result, code = p.publish_outcome(
        tmp_path / "out",
        lambda: {
            "diagnostic_verified": True,
            "diagnostic_H348": p.diagnostic_hypothesis(metrics(missing=True)),
        },
    )
    assert code == 0
    assert result["diagnostic_H348"] == {
        "verdict": "not_evaluated",
        "supported": None,
        "gates": None,
    }


def test_exact_scalar_types_and_large_integer():
    value = {
        "values": [
            None,
            True,
            2,
            0.5,
            "s",
            np.bool_(True),
            np.int64(-8),
            np.uint64(2**64 - 1),
            np.float16(0.25),
            np.float32(0.2),
            np.float64(0.3),
        ]
    }
    actual = p.lower(value)
    assert actual["values"][7] == 2**64 - 1
    assert actual["values"][9] == float(np.float32(0.2))
    assert [type(v) for v in actual["values"]] == [
        type(None),
        bool,
        int,
        float,
        str,
        bool,
        int,
        int,
        float,
        float,
        float,
    ]


@pytest.mark.parametrize(
    "bad",
    [
        float("nan"),
        float("inf"),
        np.float16("nan"),
        np.float32("inf"),
        np.longdouble(0.3),
        np.array(1.0),
        np.zeros(2),
        1j,
        np.complex64(1j),
        object(),
        (1, 2),
        {1: "bad key"},
        {np.str_("bad"): 1},
    ],
)
def test_unsupported_publication_has_strict_error_artifact(tmp_path, bad):
    report, code = p.publish_outcome(
        tmp_path / "out",
        lambda: {
            "diagnostic_verified": True,
            "independent_metrics": bad,
            "diagnostic_H348": p.diagnostic_hypothesis(metrics()),
        },
    )
    assert code == 1 and not report["diagnostic_verified"]
    assert report["independent_metrics"] is None
    assert not (tmp_path / "out" / "report.json").exists()
    saved = json.loads((tmp_path / "out" / "failure.json").read_text())
    assert saved["first_error"]["type"] in ("TypeError", "ValueError")
    assert saved["diagnostic_H348"]["verdict"] == "technical_invalid"


def test_caps_before_publish_and_after_close_late_success_not_eligible(tmp_path):
    tmp_path.joinpath("pub").mkdir()
    with pytest.raises(ValueError, match="before publication"):
        p.publish(tmp_path / "pub" / "too.json", {"x": "a" * 200}, output_cap=100)
    assert not (tmp_path / "pub" / "too.json").exists()
    ticks = []

    def deadline():
        ticks.append(1)
        if len(ticks) >= 2:
            raise TimeoutError("generated actual-close deadline")

    report, code = p.publish_outcome(
        tmp_path / "late",
        lambda: {
            "diagnostic_verified": True,
            "diagnostic_H348": p.diagnostic_hypothesis(metrics()),
        },
        timely=deadline,
    )
    assert (
        code == 1
        and not report["diagnostic_verified"]
        and report["late_error_has_precedence"]
    )
    assert (
        report["provisional_report_present"]
        and (tmp_path / "late" / "report.json").exists()
    )
    assert report["first_error"]["type"] == "TimeoutError"
    assert report["errors"][-1]["stage"] == "failure_publication"


def test_passive_freeze_does_not_charge_unowned_sibling_bytes(tmp_path):
    (tmp_path / "old.bin").write_bytes(b"x" * 1000)
    p.publish(
        tmp_path / "freeze.json", {"frozen": True}, output_cap=100, count_folder=False
    )
    assert json.loads((tmp_path / "freeze.json").read_text()) == {"frozen": True}


def admission_fixture():
    reg = p.strict_json((p.ROOT / p.REGISTRATION).read_bytes())
    rule = reg["admission"]
    sources = {"generated.py": "fixed"}
    worker = {
        "schema": rule["worker_schema"],
        "source_sha256": sources,
        "status": "technical_invalid",
        "analysis_completed": False,
        "first_error": rule["worker_singleton_error"],
        "errors": [rule["worker_singleton_error"]],
        "H348": rule["worker_H348"],
        "arms": {},
        "neural_calls": 0,
        "CPG_advance_calls": 0,
        "candidate_adopted": False,
        "F1_F8_complete": False,
        "biology_proven": False,
    }
    for arm in p.ARMS:
        worker["arms"][arm] = {
            "status": "completed",
            "first_error": None,
            "errors": [],
            "cleanup_completed": True,
            "constructor_returned": True,
            "controls_requested": 300,
            "controls_returned": 300,
            "native_samples_acquired": 15000,
            "controls_unreturned": 0,
            "baseline_full_exact": arm == "B",
            "prefix_checked_before_c101": arm != "B",
        }
        for name, count, before, after in (
            ("constructor", 2000, 0.0, 0.2),
            ("scientific_physics", 15000, 0.2, 1.7),
        ):
            worker["arms"][arm][name] = {
                "tracking_available": True,
                "method_restored": True,
                "absolute_clock_known": True,
                "steps_requested": count,
                "steps_returned": count,
                "absolute_before_s": before,
                "absolute_after_s": after,
                "actual_advance_s": after - before,
                "clock_read_errors": [rule["allowed_constructor_probe_error"]]
                if name == "constructor"
                else [],
            }
    audit = {
        "schema": "flylab.m3.e350.bounded-independent-audit-failure.v1",
        "passed": False,
        "body_evidence_qualified": False,
        "H348": "not_evaluated",
        "worker_first_error": rule["worker_singleton_error"],
        "audit_first_error": {
            "type": "ValueError",
            "message": "independent H348 conjunction",
        },
    }
    return worker, audit, rule, sources


def test_exact_known_publication_admission():
    p.admit(*admission_fixture())


@pytest.mark.parametrize(
    "bad",
    [
        "error",
        "partial",
        "roster",
        "cleanup",
        "science_clock",
        "probe",
        "clock",
        "audit",
    ],
)
def test_narrow_admission_rejects_other_failures(bad):
    worker, audit, rule, sources = admission_fixture()
    arm = worker["arms"]["I"]
    if bad == "error":
        worker["errors"].append({"stage": "extra"})
    elif bad == "partial":
        arm["status"] = "scientific_partial"
    elif bad == "roster":
        worker["arms"] = {"I": arm, "B": worker["arms"]["B"], "C": worker["arms"]["C"]}
    elif bad == "cleanup":
        arm["cleanup_completed"] = False
    elif bad == "science_clock":
        arm["scientific_physics"]["clock_read_errors"] = [{"type": "AttributeError"}]
    elif bad == "probe":
        arm["constructor"]["clock_read_errors"] = [
            {
                "stage": "before_init_or_science",
                "type": "AttributeError",
                "message": "different",
            }
        ]
    elif bad == "clock":
        arm["scientific_physics"]["absolute_after_s"] = float("inf")
    else:
        audit["passed"] = True
    with pytest.raises(ValueError):
        p.admit(worker, audit, rule, sources)


def generated_selected(tmp_path, *, extra=True):
    file = tmp_path / "selected.npz"
    arrays = {"command": np.zeros((2, 3), dtype=np.float64)}
    if extra:
        arrays["unopened"] = np.array([object()], dtype=object)
    np.savez_compressed(file, **arrays)
    raw = file.read_bytes()
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        info = archive.getinfo("command.npy")
        with archive.open(info) as stream:
            version = np.lib.format.read_magic(stream)
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
            header = stream.tell()
    member = {
        "member": "command.npy",
        "dtype": str(dtype),
        "shape": list(shape),
        "fortran_order": fortran,
        "npy_version": list(version),
        "header_bytes": header,
        "zip_compression": info.compress_type,
        "compressed_bytes": info.compress_size,
        "uncompressed_bytes": info.file_size,
        "crc32": info.CRC,
    }
    return {"path": "selected.npz", "bytes": len(raw), "sha256": p.sha(raw)}, {
        "command": member
    }


def test_selected_headers_unopened_extra_cache_and_pre_gate_zero_reads(
    tmp_path, monkeypatch
):
    spec, selected = generated_selected(tmp_path)
    blocked = p.EvidenceReader(tmp_path)
    with pytest.raises(ValueError, match="private gate"):
        blocked.npz(spec, {"command": ((2, 3), "float64")}, selected=selected)
    assert blocked.read_counts == {}
    calls = []
    original = zipfile.ZipFile.open

    def opened(archive, name, *args, **kw):
        calls.append(name.filename if hasattr(name, "filename") else name)
        return original(archive, name, *args, **kw)

    monkeypatch.setattr(zipfile.ZipFile, "open", opened)
    reader = p.EvidenceReader(tmp_path, generated=True)
    arrays = reader.npz(spec, {"command": ((2, 3), "float64")}, selected=selected)
    reader.bytes(spec, array=True)
    assert calls == ["command.npy"] and reader.read_counts == {"selected.npz": 1}
    assert np.array_equal(arrays["command"], np.zeros((2, 3)))


@pytest.mark.parametrize("bad", ["hash", "header", "dtype", "crc"])
def test_registered_selected_metadata_rejects(tmp_path, bad):
    spec, selected = generated_selected(tmp_path)
    if bad == "hash":
        spec["sha256"] = "0" * 64
    elif bad == "header":
        selected["command"]["header_bytes"] += 1
    elif bad == "dtype":
        selected["command"]["dtype"] = "int64"
    else:
        selected["command"]["crc32"] ^= 1
    roster = {"command": ((2, 3), selected["command"]["dtype"])}
    with pytest.raises(ValueError):
        p.EvidenceReader(tmp_path, generated=True).npz(spec, roster, selected=selected)


def test_raw_roster_infinities_and_nan_exact(tmp_path):
    np.savez_compressed(tmp_path / "raw.npz", x=np.array([np.inf]))
    raw = (tmp_path / "raw.npz").read_bytes()
    spec = {"path": "raw.npz", "bytes": len(raw), "sha256": None}
    with pytest.raises(ValueError, match="Inf"):
        p.EvidenceReader(tmp_path, generated=True).npz(spec, {"x": ((1,), "float64")})
    for a, b in (([np.inf], [np.inf]), ([np.nan, 1.0], [1.0, np.nan])):
        with pytest.raises(ValueError):
            p.exact(np.array(a), np.array(b), "generated", nan=True)
    p.exact(np.array([np.nan, 1.0]), np.array([np.nan, 1.0]), "generated", nan=True)
    with pytest.raises(ValueError, match="size"):
        p.EvidenceReader(tmp_path, generated=True).bytes(spec, array=True, cap=1)


def test_on_disk_evaluation_and_worker_link_both_required():
    physical = {"forward_mm": 1.0, "slip_mm_s": None, "measured_native_healthy": True}
    measured = {"window": {"longest_forward_bout_s": np.float64(0.32)}}
    saved = {"full_status": "evaluated", "physical": physical, "gait": measured}
    p.compare_evaluation(saved, copy.deepcopy(saved), physical, measured)
    changed = copy.deepcopy(saved)
    changed["physical"]["forward_mm"] = 1.01
    with pytest.raises(ValueError, match="linkage"):
        p.compare_evaluation(saved, changed, physical, measured)
    with pytest.raises(ValueError, match="independent physical"):
        p.compare_evaluation(changed, changed, physical, measured)


def test_actual_registered_metadata_admission_stat_only_no_array_hash(monkeypatch):
    original = p.EvidenceReader.bytes
    names = []

    def checked(reader, spec, **kw):
        assert not spec["path"].endswith((".npz", ".npy")) and not kw.get("array")
        names.append(spec["path"])
        return original(reader, spec, **kw)

    monkeypatch.setattr(p.EvidenceReader, "bytes", checked)
    context = p.metadata_preflight()
    assert context["worker"]["status"] == "technical_invalid"
    assert not context["reader"].array_authorized
    assert context["old_audit"]["passed"] is False and names


def test_private_gate_bad_freeze_before_array_capability(tmp_path, monkeypatch):
    (tmp_path / "freeze.json").write_text(
        json.dumps({"qualification": {"path": "q.json"}})
    )
    (tmp_path / "review.json").write_text("{}")
    reader = p.EvidenceReader(tmp_path)
    monkeypatch.setattr(p, "metadata_preflight", lambda *args, **kw: {"reader": reader})
    monkeypatch.setattr(p, "make_freeze", lambda *args, **kw: {"different": True})
    with pytest.raises(ValueError, match="exact E351 freeze"):
        p.original_gate("freeze.json", "review.json", root=tmp_path)
    assert not reader.array_authorized
    assert p.original_argv("F.json", "R.json")[5:] == [
        "--recover",
        "--freeze",
        "F.json",
        "--review",
        "R.json",
        "--output",
        p.OUTPUT,
    ]


def test_cached_gait_model_matches_actual_pinned_load_model(tmp_path):
    from tools import analyze_c_banc_biological_gait as gait

    points = [[i * 0.1, (i % 2) * 0.03, (i % 3) * 0.04] for i in range(8)]
    rows = [
        {
            "time_s": c * 0.005,
            "control_tick": c,
            "body": {
                "basis": np.eye(3).tolist(),
                "position": [c * 0.005, 0.0, 0.0],
                "yaw": 0.0,
                "legs": {leg: points for leg in p.LEGS},
            },
            "feet": {"load_bearing": [True] * 6},
            "upright": 1.0,
            "fault": None,
        }
        for c in range(20)
    ]
    payload = {
        "samples": rows,
        "body_hash": "b",
        "model_hash": "m",
        "adapter_hash": "a",
        "graph_hash": "g",
        "parameters": {},
    }
    file = tmp_path / "trace.json"
    file.write_text(json.dumps(payload))
    actual, expected = (
        p.gait_model(payload, file, p.sha(file.read_bytes())),
        gait.load_model(file),
    )
    assert set(actual) == set(expected)
    for key in actual:
        if isinstance(actual[key], np.ndarray):
            assert np.array_equal(actual[key], expected[key])
        else:
            assert actual[key] == expected[key]


@pytest.mark.parametrize("bad", [None, "gravity", "precision"])
def test_fixed_configuration_metadata_without_runtime_import(bad):
    reg = p.strict_json((p.ROOT / p.REGISTRATION).read_bytes())
    file = p.ROOT / reg["worker_folder"] / "B" / "receipt.json"
    ledger = p.strict_json(
        file.read_bytes()
    )  # Existing finite configuration JSON only.
    config, before = (
        copy.deepcopy(ledger["configuration"]),
        ledger["configuration_before_configure"],
    )
    if bad == "gravity":
        config["config"]["gravity"] = 2.0
    elif bad == "precision":
        config["physics_identity"]["precision"] = "float32"
    if bad:
        with pytest.raises(ValueError, match="pinned fixed"):
            p.fixed_configuration(p.EvidenceReader(), reg, config, before)
    else:
        p.fixed_configuration(p.EvidenceReader(), reg, config, before)


def test_cli_requires_exact_sole_argv_before_gate(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("gate must not be reached for nonsole argv")

    monkeypatch.setattr(p, "original_gate", refuse)
    with pytest.raises(ValueError, match="sole E351 recovery CLI"):
        p.main(["--recover", "--freeze", "F.json", "--review", "R.json"])
