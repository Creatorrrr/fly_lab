"""Synthetic two-payload CPU tests; original arrays and worker imports forbidden."""

import copy
import hashlib
import io
import json
import math
import zipfile
from pathlib import Path

import numpy as np
import pytest

from tools import analyze_c_banc_m3_startup_partial as p


def pin(path, root):
    value = path.read_bytes()
    return {
        "path": path.relative_to(root).as_posix(),
        "bytes": len(value),
        "sha256": hashlib.sha256(value).hexdigest(),
    }


def fixture(root, *, error=0.05, nonfinite=False):
    # This registered JSON contains descriptors only; no referenced original is opened.
    d = json.loads((p.ROOT / p.DESCRIPTOR).read_text("utf-8"))
    base = root / "e347-sol-packet"
    base.mkdir()
    sources, snapshots = {}, {}
    for index, name in enumerate(d["worker_source_sha256"]):
        source = base / ("source" + str(index) + ".py")
        snapshot = base / ("snapshot" + str(index) + ".py")
        source.write_text("# synthetic inert source " + str(index), "utf-8")
        snapshot.write_bytes(source.read_bytes())
        sources[source.relative_to(root).as_posix()] = pin(source, root)["sha256"]
        snapshots[source.relative_to(root).as_posix()] = pin(snapshot, root)
    d["worker_source_sha256"], d["worker_source_snapshots"] = sources, snapshots
    control = {
        name: np.zeros(shape, dtype=dtype) for name, (shape, dtype) in p.CONTROL.items()
    }
    startup = {
        name: np.zeros(shape, dtype=dtype) for name, (shape, dtype) in p.STARTUP.items()
    }
    control["control_tick"][:] = np.arange(100)
    control["network_tick"][:] = 20 * np.arange(100)
    control["physics_time_s"][:] = 0.005 * np.arange(100)
    control["command"][0, :] = 99  # Must exclude boundary row0.
    control["command"][45, 2] = float("nan") if nonfinite else error
    for label, arrays, extras in (
        ("control", control, ("theta", "selected_weights")),
        ("startup", startup, ("unselected_extra",)),
    ):
        path = base / (label + ".npz")
        with zipfile.ZipFile(path, "w") as archive:
            for name, value in arrays.items():
                stream = io.BytesIO()
                np.save(stream, value, allow_pickle=False)
                archive.writestr(name + ".npy", stream.getvalue())
            for name in extras:
                archive.writestr(name + ".npy", b"MUST NOT OPEN OR PARSE THIS HEADER")
        spec = d["files"]["control"] if label == "control" else d["startup"]
        spec.update(pin(path, root))
    d["payload_whitelist"] = [d["files"]["control"]["path"], d["startup"]["path"]]
    metrics = {
        "q_rms": abs(error) / math.sqrt(99 * 42),
        "q_max": abs(error),
        "tendon_rms": 0.0,
        "tendon_max": 0.0,
        "controls": 99,
        "passed": abs(error) <= 0.04,
    }
    counters = {
        "completed_rows": 100,
        "controls_completed": 99,
        "controls_host_returned": 99,
        "controls_requested": 99,
        "neural_requested": 1980,
        "neural_host_returned": 1980,
        "neural_device_confirmed": 1980,
        "physics_requested": 4950,
        "physics_host_returned": 4950,
        "physics_device_confirmed": 4950,
        "observer_samples": 4950,
        "neural_unreturned_unknown": False,
        "physics_unreturned_unknown": False,
    }
    transfer = {
        key: True
        for key in (
            "pointer_version_guards_passed",
            "source_inverse_whole_exact",
            "source_boundary_command_and_tendon_exact",
        )
    }
    result = {
        "source_sha256": sources,
        "counters": counters,
        "startup": metrics,
        "effective_qualified": False,
        "candidate_adopted": False,
        "F1_F8_complete": False,
        "biology_proven": False,
        "transfer": transfer,
    }

    def write(spec, name, value):
        path = base / name
        path.write_text(json.dumps(value, allow_nan=False), "utf-8")
        spec.update(pin(path, root))

    for key in ("result", "receipt", "stdout"):
        write(d["files"][key], key + ".json", result)
    write(d["files"]["transfer"], "transfer.json", transfer)
    write(d["files"]["trace"], "trace.json", {"inherited_only": True})
    write(
        d["files"]["selected_ledger"],
        "ledger.json",
        {
            "inputs": [
                {"path": spec["path"], "expected_sha256": spec["sha256"]}
                for spec in (d["startup"], d["candidate"])
            ]
        },
    )
    write(
        d["freeze"],
        "freeze.json",
        {
            "source_sha256": sources,
            "inventory": {
                "selected_archives": {
                    "startup": d["startup"],
                    "audited_candidate": d["candidate"],
                }
            },
        },
    )
    sampler = {
        "owned_thread_alive": False,
        "sampler_errors": [],
        "sample_count": 3,
        "sampled_additional_peak_bytes": 0,
    }
    for key, spec in d["completion_metadata"].items():
        value = sampler if key == "sampler" else {}
        if key == "ROOT_correction":
            value = {
                "original_conservative80_preserved": True,
                "startup_worker_reported_fail_preserved": True,
                "sampler_actual_completion": sampler,
            }
        write(spec, key + ".json", value)
    oracle = base / "oracle.py"
    oracle.write_text(
        "# inert provenance; independent fsum implemented in analyzer", "utf-8"
    )
    d["immutable_metric_oracle"] = pin(oracle, root)
    names = {
        "control": "control-raw.npz",
        "native": "native-contact.npz",
        "result": "result.json",
        "receipt": "receipts.json",
        "transfer": "transfer-chunks.json",
        "trace": "trace.json",
        "selected_ledger": "selected-ledger.json",
    }
    write(
        d["manifest"],
        "manifest.json",
        {
            "files": {
                name: {k: d["files"][key][k] for k in ("bytes", "sha256")}
                for key, name in names.items()
            }
        },
    )
    dossier = [
        d["manifest"],
        *(v for key, v in d["files"].items() if key not in ("control", "native")),
        *d["completion_metadata"].values(),
        *d["worker_source_snapshots"].values(),
        d["freeze"],
        d["immutable_metric_oracle"],
    ]
    d["metadata_dossier"] = dossier
    return d, control, startup


def test_partial_refused_startup_completed_analysis_and_exact_locations(
    tmp_path, monkeypatch
):
    d, _, _ = fixture(tmp_path)
    opened = []
    actual_open = Path.open

    def guarded_open(path, *args, **kwargs):
        if path.suffix in (".npy", ".npz"):
            assert path.relative_to(tmp_path).as_posix() in d["payload_whitelist"]
            opened.append(path.name)
        return actual_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    actual_member_open = zipfile.ZipFile.open
    decoded_members = []

    def guarded_member_open(archive, member, *args, **kwargs):
        name = member.filename if isinstance(member, zipfile.ZipInfo) else member
        assert name in {key + ".npy" for key in (*p.CONTROL, *p.STARTUP)}
        decoded_members.append(name)
        return actual_member_open(archive, member, *args, **kwargs)

    monkeypatch.setattr(zipfile.ZipFile, "open", guarded_member_open)
    report = p.analyze(tmp_path, d, generated=True)
    assert report["analysis_completed"] is True, report
    assert report["startup_passed"] is False
    assert report["command_minus_target"]["maximum_location"] == {
        "control_tick": 45,
        "column_zero_based": 2,
        "signed_delta": 0.05,
        "absolute_error": 0.05,
    }
    assert math.isclose(
        report["metrics"]["q_rms"], 0.05 / math.sqrt(99 * 42), abs_tol=1e-15
    )
    assert opened == ["control.npz", "startup.npz"]
    assert len(decoded_members) == 7
    assert (
        report["later_gait"] == "not_tested"
        and report["external_completion_qualified"] is False
    )
    assert (
        report["candidate_adopted"]
        is report["F1_F8_complete"]
        is report["biology_proven"]
        is False
    )
    assert len(json.dumps(report)) < 1048576


def test_aligned_control0_excluded_and_gate_boundary(tmp_path):
    d, control, startup = fixture(tmp_path, error=0.04)
    report = p.analyze(tmp_path, d, generated=True)
    assert report["analysis_completed"] and report["startup_passed"]
    assert report["metrics"]["q_max"] == 0.04
    shifted = copy.deepcopy(control)
    shifted["network_tick"] = shifted["network_tick"] + 20
    with pytest.raises(ValueError, match="clocks"):
        p.calculate(shifted, startup, {"passed": True})


def test_main_slow_publication_close_returns_nonzero(tmp_path, monkeypatch, capsys):
    """Actual main on generated files; close crosses10s after a write below10s."""
    import sys
    from types import SimpleNamespace

    d, _, _ = fixture(tmp_path)
    descriptor = tmp_path / "e347-sol-packet/descriptor.json"
    descriptor.write_text(json.dumps(d), "utf-8")
    registration = tmp_path / "e347-sol-packet/registration.json"
    registration.write_text("{}", "utf-8")
    output = tmp_path / "e347-sol-packet/slow-result.json"
    clock = [9.0]
    monkeypatch.setattr(p, "ROOT", tmp_path)
    monkeypatch.setattr(p, "ENTRY_S0", 0)
    monkeypatch.setattr(p, "time", SimpleNamespace(perf_counter=lambda: clock[0]))
    monkeypatch.setattr(p, "DESCRIPTOR", descriptor.relative_to(tmp_path).as_posix())
    monkeypatch.setattr(p, "DESCRIPTOR_SHA", pin(descriptor, tmp_path)["sha256"])
    monkeypatch.setattr(
        p, "REGISTRATION", registration.relative_to(tmp_path).as_posix()
    )
    monkeypatch.setattr(p, "REGISTRATION_SHA", pin(registration, tmp_path)["sha256"])
    monkeypatch.setattr(
        sys,
        "argv",
        ["generated-main-test", "--output", output.relative_to(tmp_path).as_posix()],
    )
    actual_open = Path.open

    class SlowClose:
        def __init__(self, stream):
            self.stream = stream

        def __enter__(self):
            return self

        def write(self, payload):
            clock[0] = 9.6
            return self.stream.write(payload)

        def __exit__(self, *args):
            self.stream.close()
            clock[0] = 10.05

    def injected_open(path, *args, **kwargs):
        assert path.resolve().is_relative_to(tmp_path.resolve()), (
            "no original metadata/payload access"
        )
        stream = actual_open(path, *args, **kwargs)
        return SlowClose(stream) if path == output and args == ("xb",) else stream

    monkeypatch.setattr(Path, "open", injected_open)
    assert p.main() == 1
    saved = json.loads(output.read_text("utf-8"))
    assert saved["analysis_completed"] is True
    assert saved["external_completion_qualified"] is False
    assert "numerical calculation only" in saved["analysis_completion_scope"]
    assert "after publication/stream close" in capsys.readouterr().err


@pytest.mark.parametrize(
    "mutation",
    [
        "badpin",
        "latecompletion",
        "extra_decode",
        "nonfinite",
        "worker_mismatch",
        "duplicate_dossier",
        "path_escape",
    ],
)
def test_synthetic_fail_closed(tmp_path, mutation):
    d, _, _ = fixture(tmp_path, nonfinite=mutation == "nonfinite")
    if mutation == "badpin":
        d["files"]["result"]["sha256"] = "0" * 64
    elif mutation == "latecompletion":
        spec = d["completion_metadata"]["sampler"]
        path = tmp_path / spec["path"]
        value = json.loads(path.read_text())
        value["owned_thread_alive"] = True
        path.write_text(json.dumps(value), "utf-8")
        spec.update(pin(path, tmp_path))
    elif mutation == "extra_decode":
        d["files"]["control"]["decode_members"].append("theta")
    elif mutation == "worker_mismatch":
        for key in ("result", "receipt", "stdout"):
            spec = d["files"][key]
            path = tmp_path / spec["path"]
            value = json.loads(path.read_text())
            value["startup"]["q_max"] = 0.049
            path.write_text(json.dumps(value), "utf-8")
            spec.update(pin(path, tmp_path))
        spec = d["manifest"]
        path = tmp_path / spec["path"]
        manifest = json.loads(path.read_text())
        for key, name in (("result", "result.json"), ("receipt", "receipts.json")):
            manifest["files"][name] = {
                k: d["files"][key][k] for k in ("bytes", "sha256")
            }
        path.write_text(json.dumps(manifest), "utf-8")
        spec.update(pin(path, tmp_path))
    elif mutation == "duplicate_dossier":
        d["metadata_dossier"].append(d["metadata_dossier"][0])
    elif mutation == "path_escape":
        d["files"]["control"]["path"] = "../control.npz"
        d["payload_whitelist"][0] = "../control.npz"
    report = p.analyze(tmp_path, d, generated=True)
    assert report["analysis_completed"] is False and report["first_error"], report
    assert report["candidate_adopted"] is False
