"""One fixed fresh generated native20 diagnostic; never whole-pytest Q/F."""

# ruff: noqa: E402
from __future__ import annotations

import time

ENTRY = time.perf_counter()
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = Path("C:/Users/ckthd/AppData/Local/Temp/e357-root-diag-v1")
for name in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[name] = "1"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools import verify_c_banc_m3_reference_parameter_direction as m


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args(argv)
    output = Path(args.output_root)
    m.require(
        output.is_absolute()
        and ".." not in output.parts
        and output == OUTPUT_ROOT
        and output.resolve() == OUTPUT_ROOT
        and not output.resolve().is_relative_to(ROOT.resolve())
        and not output.exists(),
        "fixed canonical fresh generated temp only",
    )
    output.mkdir()
    trace = m.PhaseTrace(
        scope="generated-native20-diagnostic-precontext", cuda_events=True
    )
    summary = {
        "schema": "flylab.m3.e357.generated-diagnostic.v1",
        "qualified": False,
        "original_calls": 0,
        "pytest_case_execution": None,
        "nopytest_case_execution": True,
        "Q": False,
        "F": False,
        "H356": "not_evaluated",
        "source_sha256": m.own_sources(),
        "diagnostic_source_sha256": m.sha(Path(__file__).read_bytes()),
        "registration_sha256": m.REGISTRATION_SHA,
        "sampler_scope": "before_scientific_import_and_first_CUDA",
        "sampler_start": None,
        "sampler_closed": None,
        "first_error": None,
        "secondary_errors": [],
        "work": None,
        "preparation_work": "16 forward controls/320 neural ticks, separate from collect",
        "timing_scope": "CPU perf_counter through sampler close, excludes summary publication; CUDA stream times after existing confirm",
    }
    watch = m.generated_gpu_watch()
    clock = m.Clock(entry=ENTRY, internal=40, science=38)

    def failure(error):
        row = {
            "type": type(error).__name__,
            "message": str(error)[:1024],
            "last_phase": trace.rows[-1] if trace.rows else None,
        }
        if summary["first_error"] is None:
            summary["first_error"] = row
        else:
            summary["secondary_errors"].append(row)
        summary["qualified"] = False

    def timely():
        clock.check("science")
        m.generated_sampler_check(watch.snapshot())
        # CPU tree RSS is distinct from sampled whole-device GPU observations.
        m.require(
            core.usage("cpu")["tree_rss_bytes"] <= 2 << 30, "generated tree RSS2GiB"
        )

    try:
        m.metadata_preflight()
        trace.emit("job-sampler", "start")
        summary["sampler_start"] = watch.start()
        m.generated_sampler_check(summary["sampler_start"])
        trace.emit("job-sampler", "started")
        trace.emit("scientific-import", "start")
        import numpy as np
        import torch

        core = m.native()
        torch.set_num_threads(1)
        m.require(
            torch.get_num_threads() == 1 and torch.cuda.is_available(),
            "mandatory generated CUDA/thread1",
        )
        trace.emit("scientific-import", "done")
        dimensions = core.TestDimensions(steps_per_control=20)
        timely()
        trace.emit("initial-fixture", "preparing")
        fixture = core.synthetic_fixture(output, dimensions=dimensions, device="cuda")
        trace.emit("initial-fixture", "done")
        timely()
        trace.emit("packet-load", "preparing")
        packet = core.load_packet(
            output,
            fixture["inventory"],
            dimensions=dimensions,
            metadata=fixture["metadata"],
        )
        trace.emit("packet-load", "done")
        target = np.stack([packet[f"endpoint-{c}"]["q"] for c in m.WINDOWS]) + 0.01
        raw, receipts, record = m.collect(
            packet,
            target_q=target,
            device="cuda",
            dimensions=dimensions,
            timely=timely,
            progress=trace,
        )
        work = core.work_counts(receipts)
        m.require(
            work["forward_controls_returned"] == 32
            and work["forward_neural_ticks_returned"] == 640
            and work["recompute_controls_returned"] == 16
            and work["recompute_neural_ticks_returned"] == 320
            and work["backward_blocks_returned"] == 16
            and record["neural_backward_returned"] == 4,
            "mandatory actual native20 generated diagnostic work",
        )
        summary["work"] = work
        trace.emit("publication", "start", work=work)
        result = m.publish_generated(
            output / "generated-complete",
            raw,
            receipts,
            record,
            packet["ledger"],
            dimensions=dimensions,
            timely=timely,
        )
        m.require(
            result["qualified"] is True
            and result["analysis"]["candidate_runtime_adoption"] is False,
            "closed generated diagnostic publication only",
        )
        trace.emit("publication", "complete", work=work)
        summary["qualified"] = True
    except BaseException as error:  # noqa: BLE001 -- Keep scientific/publication failure and owned closure.
        failure(error)
    finally:
        try:
            trace.emit("job-sampler", "close-start")
            summary["sampler_closed"] = watch.close()
            m.generated_sampler_check(summary["sampler_closed"], closed=True)
            trace.emit("job-sampler", "closed")
        except BaseException as error:  # noqa: BLE001 -- Never mask the original failure.
            failure(error)
        summary["actual_entrypoint_elapsed_s"] = time.perf_counter() - ENTRY
        summary["phases"] = trace.rows
        try:
            m.require(
                summary["source_sha256"] == m.own_sources(), "same diagnostic source2"
            )
            m.require(
                summary["diagnostic_source_sha256"]
                == m.sha(Path(__file__).read_bytes()),
                "same diagnostic entrypoint source",
            )
            clock.check()
        except BaseException as error:  # noqa: BLE001 -- Final source/time closure is mandatory.
            failure(error)
        try:
            # Existing strict JSON/fsync/close/reread publisher, bounded emergency closure.
            closing_clock = m.Clock()
            store = m.Store(output / "closed-diagnostic", closing_clock)
            store.put("audit.json", m.json_bytes(summary))
        except BaseException as error:  # noqa: BLE001 -- Failed summary publication must return nonzero.
            failure(error)
    print(
        m.json_bytes(
            {
                "stage": "diagnostic-return",
                "qualified": summary["qualified"],
                "first_error": summary["first_error"],
                "secondary_errors": summary["secondary_errors"],
                "summary": str(output / "closed-diagnostic" / "audit.json"),
            }
        ).decode("utf8"),
        flush=True,
    )
    return 0 if summary["qualified"] and summary["first_error"] is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
