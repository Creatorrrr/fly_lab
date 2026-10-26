"""Independent CPU diagnosis of two pinned partial-startup payloads only."""

import time

ENTRY_S0 = time.perf_counter()
import argparse
import hashlib
import io
import json
import math
import os
import sys
import zipfile
from pathlib import Path

for _key in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_key] = "1"
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = "verification/banc-natural-gait-m3-20260919"
REGISTRATION = BASE + "/parent-e347-reflected-design-registration-v2.json"
REGISTRATION_SHA = "a54654d3dd8de45025a0a997b7d07cd3618853a51714edaad23b297063bf0c11"
DESCRIPTOR = BASE + "/parent-e347-partial-input-v2.json"
DESCRIPTOR_SHA = "91dc9a20430e1dc18b0e64b327c59381ef461ba265e45764e77ab6c94af25e40"
CONTROL = {
    "command": ((100, 42), "<f8"),
    "tendon": ((100, 6), "<f8"),
    "control_tick": ((100,), "<i8"),
    "network_tick": ((100,), "<i8"),
    "physics_time_s": ((100,), "<f8"),
}
STARTUP = {"target_command": ((99, 42), "<f8"), "target_tendon": ((99, 6), "<f8")}
LIMITS = {"q_rms": 0.018, "q_max": 0.04, "tendon_rms": 0.009, "tendon_max": 0.025}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def safe(root, name):
    require(
        isinstance(name, str) and name and "\\" not in name and "\x00" not in name,
        "canonical relative path",
    )
    path = Path(name)
    require(
        not path.is_absolute() and ".." not in path.parts and path.as_posix() == name,
        "path traversal/alias",
    )
    target = Path(root) / path
    require(
        target.resolve() == Path(root).resolve() / path and not target.is_symlink(),
        "symlink/path escape",
    )
    return target


def bounded_bytes(root, spec, limit, timely):
    timely()
    require(
        type(spec["bytes"]) is int and 0 < spec["bytes"] <= limit,
        "bounded typed pin bytes",
    )
    path = safe(root, spec["path"])
    with path.open("rb") as stream:
        payload = stream.read(spec["bytes"] + 1)
    timely()
    require(
        len(payload) == spec["bytes"]
        and hashlib.sha256(payload).hexdigest() == spec["sha256"],
        "exact saved bytes/SHA",
    )
    return payload


def strict_json(payload):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate JSON key")
            result[key] = value
        return result

    def constant(value):
        raise ValueError("nonfinite JSON constant " + value)

    return json.loads(payload, object_pairs_hook=pairs, parse_constant=constant)


def pin_identity(spec):
    return {key: spec[key] for key in ("path", "bytes", "sha256")}


def verify_metadata(root, descriptor, *, timely=lambda: None, generated=False):
    """Finite explicit dossier; never follows references in its JSON objects."""
    require(
        descriptor["schema"] == "flylab.m3.e347.root-partial-input.v2", "v2 descriptor"
    )
    require(
        descriptor["fixed_rows"] == 100
        and descriptor["fixed_controls"] == 99
        and descriptor["fixed_neural_ticks"] == 1980
        and descriptor["fixed_physics_steps"] == 4950
        and descriptor["neural_per_control"] == 20
        and descriptor["physics_per_control"] == 50
        and descriptor["control_dt_s"] == 0.005
        and descriptor["metric_report_abs_tolerance"] == 1e-12
        and descriptor["metric_report_rel_tolerance"] == descriptor["gate_epsilon"] == 0
        and descriptor["physics_clock_abs_tolerance_s"] == 2e-8
        and descriptor["startup_limits"]
        == {"startup_" + key: value for key, value in LIMITS.items()},
        "fixed diagnostic math/clock",
    )
    require(
        descriptor["external_completion_qualified"] is False
        and descriptor["candidate_adopted"] is False
        and descriptor["F1_F8_complete"] is False
        and descriptor["biology_proven"] is False
        and descriptor["E346_technical_status"] == "technical_invalid",
        "no promotion",
    )
    control, startup = descriptor["files"]["control"], descriptor["startup"]
    require(
        set(control["decode_members"]) == set(CONTROL)
        and len(control["decode_members"]) == 5
        and set(startup["decode_members"]) == set(STARTUP)
        and len(startup["decode_members"]) == 2,
        "exact five/two decoded members",
    )
    for spec, schema in ((control, CONTROL), (startup, STARTUP)):
        for name, (shape, dtype) in schema.items():
            require(
                spec["selected_schema"][name]
                == {"shape": list(shape), "dtype": dtype, "finite": True},
                "selected schema",
            )
    require(
        descriptor["payload_whitelist"] == [control["path"], startup["path"]]
        and control["path"] != startup["path"]
        and descriptor["original_payload_limit_bytes"] == 6 * 1048576
        and descriptor["output_limit_bytes"] == 1048576,
        "two exact payloads/quota",
    )
    require(
        descriptor["candidate"]["fresh_payload_access_allowed"] is False
        and descriptor["files"]["native"]["fresh_payload_access_allowed"] is False,
        "provenance-only payloads",
    )
    specs = [descriptor["manifest"]]
    specs.extend(
        v for key, v in descriptor["files"].items() if key not in ("control", "native")
    )
    specs.extend(descriptor["completion_metadata"].values())
    specs.extend(descriptor["worker_source_snapshots"].values())
    specs.extend((descriptor["freeze"], descriptor["immutable_metric_oracle"]))
    require(descriptor["metadata_dossier"] == specs, "finite ordered metadata dossier")
    paths = [v["path"] for v in specs] + [control["path"], startup["path"]]
    require(
        len({v.casefold() for v in paths}) == len(paths),
        "distinct dossier/payload paths",
    )
    if generated:
        require(
            all(Path(v).parts[0].startswith("e347-sol-") for v in paths),
            "generated-only paths",
        )
    saved = {}
    for spec in specs:
        path = spec["path"]
        require(Path(path).suffix in (".json", ".py", ".md"), "metadata/text only")
        saved[path] = bounded_bytes(root, spec, 1048576, timely)
    for name, sha in descriptor["worker_source_sha256"].items():
        require(name.endswith(".py"), "worker source text only")
        snapshot = descriptor["worker_source_snapshots"][name]
        require(snapshot["sha256"] == sha, "source/snapshot pin equality")
        spec = {"path": name, "bytes": snapshot["bytes"], "sha256": sha}
        require(
            bounded_bytes(root, spec, 1048576, timely) == saved[snapshot["path"]],
            "source equals snapshot",
        )
    files = descriptor["files"]
    require(
        saved[files["result"]["path"]]
        == saved[files["receipt"]["path"]]
        == saved[files["stdout"]["path"]],
        "result=receipt=stdout bytes",
    )
    result = strict_json(saved[files["result"]["path"]])
    require(
        result["source_sha256"] == descriptor["worker_source_sha256"],
        "worker source map",
    )
    counters = result["counters"]
    for key, value in {
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
    }.items():
        require(
            type(counters[key]) is int and counters[key] == value,
            "inherited partial counters",
        )
    require(
        counters["neural_unreturned_unknown"] is False
        and counters["physics_unreturned_unknown"] is False
        and result["effective_qualified"] is False
        and result["candidate_adopted"] is False
        and result["F1_F8_complete"] is False
        and result["biology_proven"] is False,
        "worker no promotion",
    )
    manifest = strict_json(saved[descriptor["manifest"]["path"]])
    names = {
        "control": "control-raw.npz",
        "native": "native-contact.npz",
        "result": "result.json",
        "receipt": "receipts.json",
        "transfer": "transfer-chunks.json",
        "trace": "trace.json",
        "selected_ledger": "selected-ledger.json",
    }
    require(set(manifest["files"]) == set(names.values()), "manifest7")
    for key, name in names.items():
        require(
            manifest["files"][name] == {k: files[key][k] for k in ("bytes", "sha256")},
            "manifest metadata binding",
        )
    freeze = strict_json(saved[descriptor["freeze"]["path"]])
    require(
        freeze["source_sha256"] == descriptor["worker_source_sha256"],
        "freeze worker sources",
    )
    for role, spec in (
        ("startup", startup),
        ("audited_candidate", descriptor["candidate"]),
    ):
        require(
            pin_identity(freeze["inventory"]["selected_archives"][role])
            == pin_identity(spec),
            "frozen target/candidate provenance",
        )
    ledger = strict_json(saved[files["selected_ledger"]["path"]])["inputs"]
    require(
        len({v["path"] for v in ledger}) == len(ledger),
        "unique inherited selected ledger",
    )
    for spec in (startup, descriptor["candidate"]):
        entries = [v for v in ledger if v["path"] == spec["path"]]
        require(
            len(entries) == 1 and entries[0]["expected_sha256"] == spec["sha256"],
            "selected ledger provenance",
        )
    transfer = strict_json(saved[files["transfer"]["path"]])
    require(
        transfer == result["transfer"]
        and transfer["pointer_version_guards_passed"] is True
        and transfer["source_inverse_whole_exact"] is True
        and transfer["source_boundary_command_and_tendon_exact"] is True,
        "inherited transfer identity",
    )
    completion = descriptor["completion_metadata"]
    sampler = strict_json(saved[completion["sampler"]["path"]])
    require(
        sampler["owned_thread_alive"] is False
        and sampler["sampler_errors"] == []
        and type(sampler["sample_count"]) is int
        and sampler["sample_count"] > 0,
        "late sampler completion",
    )
    correction = strict_json(saved[completion["ROOT_correction"]["path"]])
    require(
        correction["original_conservative80_preserved"] is True
        and correction["startup_worker_reported_fail_preserved"] is True
        and all(
            correction["sampler_actual_completion"][key] == sampler[key]
            for key in correction["sampler_actual_completion"]
        ),
        "ROOT correction/late completion binding",
    )
    # Remaining pinned reports are recorded as inherited, never promoted or recursively opened.
    return result


def load_selected(root, spec, schema, *, timely=lambda: None):
    payload = bounded_bytes(root, spec, 6 * 1048576, timely)
    selected = {}
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        names = archive.namelist()
        require(
            len(set(names)) == len(names)
            and all(
                "/" not in name and "\\" not in name and name.endswith(".npy")
                for name in names
            ),
            "NPZ duplicate/member traversal",
        )
        require(
            {name + ".npy" for name in schema} <= set(names), "selected members present"
        )
        for name, (shape, dtype) in schema.items():
            timely()
            member = archive.getinfo(name + ".npy")
            expected = math.prod(shape) * np.dtype(dtype).itemsize
            require(
                member.file_size <= expected + 1024 and member.file_size > expected,
                "bounded selected member",
            )
            # Only these selected members are opened, including headers.
            with archive.open(member) as stream:
                raw = stream.read(expected + 1025)
            stream = io.BytesIO(raw)
            version = np.lib.format.read_magic(stream)
            require(version in ((1, 0), (2, 0)), "selected NPY version")
            reader = (
                np.lib.format.read_array_header_1_0
                if version == (1, 0)
                else np.lib.format.read_array_header_2_0
            )
            found_shape, fortran, found_dtype = reader(stream, max_header_size=1024)
            require(
                found_shape == shape
                and found_dtype.str == dtype
                and not found_dtype.hasobject,
                "bounded selected header schema",
            )
            data = stream.read(expected + 1)
            require(len(data) == expected, "exact selected numeric bytes")
            value = np.frombuffer(data, dtype=found_dtype).reshape(
                shape, order="F" if fortran else "C"
            )
            require(
                value.shape == shape
                and value.dtype.str == dtype
                and np.isfinite(value).all(),
                "typed finite selected array",
            )
            selected[name] = value
    return selected


def distribution(delta):
    rows, columns = delta.shape
    values = [[float(v) for v in row] for row in delta]
    rms = math.sqrt(math.fsum(v * v for row in values for v in row) / (rows * columns))
    row, column = max(
        ((i, j) for i in range(rows) for j in range(columns)),
        key=lambda index: abs(values[index[0]][index[1]]),
    )
    return {
        "rms": rms,
        "max": abs(values[row][column]),
        "maximum_location": {
            "control_tick": row + 1,
            "column_zero_based": column,
            "signed_delta": values[row][column],
            "absolute_error": abs(values[row][column]),
        },
        "per_column": [
            {
                "column_zero_based": j,
                "rms": math.sqrt(
                    math.fsum(values[i][j] ** 2 for i in range(rows)) / rows
                ),
                "max": max(abs(values[i][j]) for i in range(rows)),
            }
            for j in range(columns)
        ],
        "time_distribution": [
            {
                "control_tick": i + 1,
                "physics_time_s": 0.005 * (i + 1),
                "rms": math.sqrt(math.fsum(v * v for v in values[i]) / columns),
                "max": max(abs(v) for v in values[i]),
            }
            for i in range(rows)
        ],
    }


def calculate(control, startup, worker_startup):
    for arrays, schema in ((control, CONTROL), (startup, STARTUP)):
        require(set(arrays) == set(schema), "exact selected mathematical inputs")
        for name, (shape, dtype) in schema.items():
            value = arrays[name]
            require(
                value.shape == shape
                and value.dtype.str == dtype
                and np.isfinite(value).all(),
                "exact typed finite math inputs",
            )
    require(
        np.array_equal(control["control_tick"], np.arange(100, dtype=np.int64))
        and np.array_equal(control["network_tick"], 20 * np.arange(100, dtype=np.int64))
        and np.all(np.abs(control["physics_time_s"] - 0.005 * np.arange(100)) <= 2e-8),
        "fixed100 clocks; no shifting",
    )
    command = distribution(control["command"][1:100] - startup["target_command"])
    tendon = distribution(control["tendon"][1:100] - startup["target_tendon"])
    metrics = {
        "q_rms": command["rms"],
        "q_max": command["max"],
        "tendon_rms": tendon["rms"],
        "tendon_max": tendon["max"],
    }
    require(
        all(
            type(worker_startup[key]) in (int, float)
            and math.isfinite(worker_startup[key])
            and math.isclose(value, worker_startup[key], rel_tol=0, abs_tol=1e-12)
            for key, value in metrics.items()
        ),
        "independent worker metric match abs1e-12/rel0",
    )
    passed = all(value <= LIMITS[key] for key, value in metrics.items())
    require(
        worker_startup["passed"] is passed and worker_startup["controls"] == 99,
        "worker startup verdict/controls",
    )
    return {
        "metrics": metrics,
        "startup_passed": passed,
        "command_minus_target": command,
        "tendon_minus_target": tendon,
        "limits": dict(LIMITS),
    }


def analyze(root, descriptor, *, timely=lambda: None, generated=False):
    report = {
        "schema": "flylab.m3.e347.partial-startup-analysis.v1",
        "analysis_completed": False,
        "analysis_completion_scope": "numerical calculation only; full process exit qualification requires external supervisor",
        "startup_passed": None,
        "external_completion_qualified": False,
        "E346_technical_status": "technical_invalid",
        "candidate_adopted": False,
        "F1_F8_complete": False,
        "biology_proven": False,
        "later_gait": "not_tested",
        "fresh_candidate_native_whole_reads": 0,
        "claim_scope": "saved command-target and tendon-input error; physical joints/forces/causality not proven",
    }
    try:
        result = verify_metadata(root, descriptor, timely=timely, generated=generated)
        require(
            sum(
                spec["bytes"]
                for spec in (descriptor["files"]["control"], descriptor["startup"])
            )
            <= 6 * 1048576,
            "two payload combined quota",
        )
        control = load_selected(
            root, descriptor["files"]["control"], CONTROL, timely=timely
        )
        startup = load_selected(root, descriptor["startup"], STARTUP, timely=timely)
        report.update(calculate(control, startup, result["startup"]))
        timely()
        report.update(
            analysis_completed=True,
            first_error=None,
            candidate_identity_scope="inherited pinned freeze/ledger/transfer metadata only",
            metadata_pins=[pin_identity(v) for v in descriptor["metadata_dossier"]],
            payload_pins=[
                pin_identity(descriptor["files"]["control"]),
                pin_identity(descriptor["startup"]),
            ],
        )
    except Exception as error:  # noqa: BLE001 -- Refusal is diagnostic failure, never promotion.
        report["first_error"] = {
            "type": type(error).__name__,
            "message": str(error)[:2048],
        }
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--descriptor", default=DESCRIPTOR)
    parser.add_argument("--descriptor-sha256", default=DESCRIPTOR_SHA)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    def timely():
        if time.perf_counter() - ENTRY_S0 >= 9.5:
            raise TimeoutError("absolute internal10s deadline reserves publication")

    require(
        args.descriptor == DESCRIPTOR and args.descriptor_sha256 == DESCRIPTOR_SHA,
        "fixed original v2 descriptor",
    )
    for name, sha in ((REGISTRATION, REGISTRATION_SHA), (DESCRIPTOR, DESCRIPTOR_SHA)):
        timely()
        require(
            hashlib.sha256(safe(ROOT, name).read_bytes()).hexdigest() == sha,
            "registered original metadata pin",
        )
    descriptor = strict_json(safe(ROOT, DESCRIPTOR).read_bytes())
    output = safe(ROOT, args.output)
    require(not output.exists() and output.parent.is_dir(), "fresh output")
    report = analyze(ROOT, descriptor, timely=timely)
    report["elapsed_before_publication_s"] = time.perf_counter() - ENTRY_S0
    payload = (json.dumps(report, indent=2, allow_nan=False) + "\n").encode()
    require(len(payload) <= 1048576, "total diagnostic output1MiB")
    timely()
    with output.open("xb") as stream:
        stream.write(payload)
    elapsed_after_close_s = time.perf_counter() - ENTRY_S0
    if elapsed_after_close_s >= 10:
        sys.stderr.write(
            json.dumps(
                {
                    "type": "TimeoutError",
                    "message": "internal10s exceeded after publication/stream close",
                    "elapsed_after_close_s": elapsed_after_close_s,
                }
            )
            + "\n"
        )
        return 1
    return 0 if report["analysis_completed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
