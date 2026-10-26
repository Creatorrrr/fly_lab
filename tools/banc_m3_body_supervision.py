"""E345 bounded orchestration of immutable E267; metadata only before spawn."""

# ruff: noqa: BLE001 -- emergency accounting must survive supervisor failure.
import time

ENTRY_S0 = time.perf_counter()
import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[1]
BASE = "verification/banc-natural-gait-m3-20260919"
SUPERVISOR = BASE + "/parent-e267-supervise.py"
SUPERVISOR_SHA = "1d8682505f8060d3d8bcb50508811b9c4da531bdbf07c96200065bcc3324df8d"
SELF = "tools/banc_m3_body_supervision.py"
MAX_REPORT = 256 * 1024


def need(condition, message):
    if not condition:
        raise ValueError(message)


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def safe(root, name):
    need(isinstance(name, str) and name and "\x00" not in name, "path string")
    path = Path(name.replace("\\", "/"))
    need(not path.is_absolute() and ".." not in path.parts, "relative owned path")
    target = Path(root) / path
    need(
        target.resolve().is_relative_to(Path(root).resolve())
        and not target.is_symlink(),
        "owned path containment",
    )
    return target


def digest(path, timely=lambda: None):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(128 * 1024):
            timely()
            result.update(block)
    return result.hexdigest()


def read_json(path):
    need(path.stat().st_size <= MAX_REPORT, "bounded metadata/report")
    return json.loads(path.read_text("utf-8"))


def pinned(root, descriptor, timely):
    path = safe(root, descriptor["path"])
    need(digest(path, timely) == descriptor["sha256"], "metadata pin")
    return read_json(path)


def prevalidate(root, contract, authority, *, receipt, pids, timely=lambda: None):
    """Authority is ROOT-owned, externally SHA-pinned, never inferred from contract."""
    generated = authority.get("generated_cpu") is True
    phase = authority["phase"]
    limit = authority["phase_limit_s"]
    need(
        finite(limit) and 0.8 <= limit <= (10 if generated else 80), "finite phase wall"
    )
    need(
        contract["phase"] == phase and len(contract["commands"]) == 1, "sole phase/job"
    )
    job = contract["commands"][0]
    need(
        job["argv"] == authority["argv"]
        and isinstance(job["argv"], list)
        and all(isinstance(arg, str) and arg for arg in job["argv"]),
        "sole exact argv",
    )
    need(
        finite(job["wall_s"])
        and job["wall_s"] == authority["job_wall_s"]
        and 0 < job["wall_s"] <= limit,
        "exact wall",
    )
    need(
        type(contract["max_tree_rss_bytes"]) is int
        and contract["max_tree_rss_bytes"] == authority["max_tree_rss_bytes"]
        and 0 < contract["max_tree_rss_bytes"] <= 6 * 1024**3,
        "exact RSS",
    )
    calls = contract["original_calls"]
    need(
        type(calls) is int and calls == authority["original_calls"] and calls in (0, 1),
        "typed original_calls",
    )
    before, ceiling = contract["cumulative_before_s"], contract["cumulative_limit_s"]
    need(
        finite(before)
        and finite(ceiling)
        and 0 <= before < ceiling
        and before == authority["cumulative_before_s"]
        and ceiling == authority["cumulative_limit_s"]
        and before + limit <= ceiling,
        "typed cumulative linkage",
    )
    need(
        contract.get("retry_count") == 0 and type(contract.get("retry_count")) is int,
        "retry0",
    )
    need(
        contract["source_sha256"].get(SUPERVISOR) == SUPERVISOR_SHA,
        "immutable E267 pin",
    )
    need(
        contract["source_sha256"].get(SELF) == digest(safe(root, SELF), timely),
        "wrapper source pin",
    )
    names = [
        receipt,
        pids,
        contract["supervisor_output"],
        job["stdout"],
        job["stderr"],
        authority["supervisor_stdout"],
        authority["supervisor_stderr"],
    ]
    names.extend(authority.get("fresh_outputs", []))
    need(len(set(names)) == len(names), "distinct output paths")
    for name in names:
        path = safe(root, name)
        need(not path.exists() and path.parent.is_dir(), "fresh output paths")
        if generated:
            relative = path.relative_to(safe(root, BASE)).parts
            need(
                relative and relative[0].startswith(("e345-sol-", "e346-sol-")),
                "generated CPU namespace",
            )
    if generated:
        need(
            authority.get("science_original_calls") == 0
            and job["argv"][:4] == [".venv/Scripts/python.exe", "-X", "utf8", "-B"],
            "CPU fixture declaration",
        )
        script = job["argv"][4]
        need(
            script.startswith((BASE + "/e345-sol-", BASE + "/e346-sol-"))
            and script.endswith(".py"),
            "fixed generated CPU script",
        )
        need(script in contract["source_sha256"], "CPU script source pin")
    else:
        need(
            phase in ("original_body", "original_audit")
            and limit == (80 if phase == "original_body" else 12)
            and job["wall_s"] == limit
            and calls == 1,
            "original phase caps",
        )
        reservation = pinned(root, authority["reservation"], timely)
        ledger = pinned(root, authority["ledger"], timely)
        need(
            reservation.get("schema") == "flylab.m3.e346.root-reservation.v1"
            and reservation.get("approved") is True
            and reservation.get("retry_count") == 0
            and reservation["original_cumulative_s"] == 9700.57798869951
            and reservation["original_limit_s"] == ceiling == 9900
            and reservation["reserved_s"] == 92
            and reservation["body_s"] == 80
            and reservation["audit_s"] == 12,
            "active E345 reservation",
        )
        need(
            ledger["cumulative_before_s"] == before
            and ledger["cumulative_limit_s"] == ceiling
            and ledger["reservation_sha256"] == authority["reservation"]["sha256"]
            and ledger["phase"] == phase
            and ledger["reserved_s"] >= limit,
            "current ledger reservation linkage",
        )
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from tools import verify_c_banc_m3_common_descent_body as body

        body.validate_registered_inventory(root, timely=timely)
        need(
            all(
                contract["source_sha256"].get(k) == v
                for k, v in reservation["source_sha256"].items()
            ),
            "reserved source pins",
        )
        need(
            set(body.source_map(root)) <= set(contract["source_sha256"]),
            "owned source scope",
        )
        if phase == "original_body":
            need(
                job["argv"][:6]
                == [
                    ".venv/Scripts/python.exe",
                    "-X",
                    "utf8",
                    "-B",
                    body.OWNED[0],
                    "--run",
                ]
                and len(job["argv"]) == 10
                and job["argv"][6] == "--freeze-sha256"
                and job["argv"][8] == "--review-sha256",
                "fixed original body argv",
            )
            freeze = pinned(
                root, {"path": body.FREEZE, "sha256": job["argv"][7]}, timely
            )
            review = pinned(
                root, {"path": body.REVIEW, "sha256": job["argv"][9]}, timely
            )
            need(
                freeze["source_sha256"] == body.source_map(root)
                and freeze["reservation"] == authority["reservation"]
                and review.get("approved") is True
                and review.get("blocking_defects") == []
                and review["freeze_sha256"] == job["argv"][7]
                and review["source_sha256"] == freeze["source_sha256"],
                "freeze/source approval",
            )
            need(body.OUTPUT in names, "fresh original body output")
    for name, sha in contract["source_sha256"].items():
        timely()
        # Native PY pins may be absolute; scientific arrays are never source inputs.
        path = Path(name)
        need(path.suffix in (".py", ".json", ".md", ".toml"), "text source whitelist")
        target = path if path.is_absolute() else safe(root, name)
        need(not target.is_symlink() and digest(target, timely) == sha, "source bytes")
    return limit


def supervise(
    contract_name,
    authority_name,
    authority_sha256,
    receipt,
    pids,
    *,
    root=ROOT,
    started=None,
):
    """Returns bounded receipt; always records emergency without E267 JSON."""
    start = time.perf_counter() if started is None else started
    known, errors, first = {}, [], None
    process, contract, authority, raw = None, None, None, None
    cap, peak, spawned, unknown = 0, 0, 0, False
    exit_code = None
    report = {
        "schema": "flylab.m3.e345.wrapper-receipt.v1",
        "technical_valid": False,
        "scientific_verdict": "not_tested",
        "science_original_calls": 0,
    }
    report["wrapper_identity"] = {
        "pid": psutil.Process().pid,
        "create_time": psutil.Process().create_time(),
    }
    cleanup_until = None

    def timely():
        if cap and time.perf_counter() - start >= cap - min(1.0, cap * 0.25):
            raise TimeoutError("wrapper absolute deadline reserves cleanup/publication")

    def capture(item):
        nonlocal unknown
        try:
            created = item.create_time()
            need(finite(created) and created > 0, "PID creation time")
            need(item.pid in known or len(known) < 512, "bounded known PID roster")
            known.setdefault(item.pid, created)
        except psutil.NoSuchProcess:
            unknown = True
        except Exception as error:
            unknown = True
            if len(errors) < 16:
                errors.append(str(error)[:256])

    try:
        authority = pinned(
            root, {"path": authority_name, "sha256": authority_sha256}, timely
        )
        declared_cap = authority["phase_limit_s"]
        need(finite(declared_cap) and 0.8 <= declared_cap <= 80, "bounded entry phase")
        cap = declared_cap
        timely()
        contract_path = safe(root, contract_name)
        need(
            digest(contract_path, timely) == authority["contract_sha256"],
            "exact contract pin",
        )
        contract = read_json(contract_path)
        prevalidate(
            root, contract, authority, receipt=receipt, pids=pids, timely=timely
        )
        report.update(
            phase=contract["phase"],
            original_calls=contract["original_calls"],
            cumulative_before_s=contract["cumulative_before_s"],
            cumulative_limit_s=contract["cumulative_limit_s"],
            contract_sha256=authority["contract_sha256"],
            generated_cpu=authority.get("generated_cpu") is True,
            science_original_calls=0
            if authority.get("generated_cpu") is True
            else None,
        )
        argv = [
            ".venv/Scripts/python.exe",
            "-X",
            "utf8",
            "-B",
            SUPERVISOR,
            contract_name,
        ]
        report["supervisor_argv"] = argv
        with (
            safe(root, authority["supervisor_stdout"]).open("xb") as out,
            safe(root, authority["supervisor_stderr"]).open("xb") as err,
        ):
            timely()
            process = subprocess.Popen(argv, cwd=root, stdout=out, stderr=err)
            spawned = 1
            owner = psutil.Process(process.pid)
            capture(owner)
            while process.poll() is None:
                timely()
                try:
                    for child in owner.children(recursive=True):
                        capture(child)
                except psutil.NoSuchProcess:
                    pass
                rss = 0
                for pid, created in list(known.items()):
                    try:
                        item = psutil.Process(pid)
                        if item.create_time() == created:
                            rss += item.memory_info().rss
                    except psutil.NoSuchProcess:
                        pass
                peak = max(peak, rss)
                need(rss <= contract["max_tree_rss_bytes"], "wrapper tree RSS")
                time.sleep(0.01)
            exit_code = process.returncode
        timely()
        raw = read_json(safe(root, contract["supervisor_output"]))
        if set(raw.get("owned_pids", [])) - set(known):
            unknown = True
        need(
            raw["schema"] == "flylab.m3.e267.root-supervision.v1"
            and raw["contract_sha256"] == authority["contract_sha256"]
            and raw["phase"] == contract["phase"]
            and raw["original_calls"] == contract["original_calls"],
            "E267 receipt linkage",
        )
        need(
            len(raw["attempts"]) == 1
            and raw["attempts"][0]["argv"] == contract["commands"][0]["argv"]
            and finite(raw["elapsed_s"])
            and 0 <= raw["elapsed_s"] <= time.perf_counter() - start
            and set(raw["owned_pids"]) == set(raw["attempts"][0]["pids"])
            and raw["owned_pids"]
            and set(raw["owned_pids"]) <= set(known),
            "E267 measured identities/delta",
        )
        if contract["original_calls"]:
            need(
                raw["cumulative_before_s"] == contract["cumulative_before_s"]
                and raw["cumulative_limit_s"] == contract["cumulative_limit_s"]
                and raw["cumulative_after_s"]
                == contract["cumulative_before_s"] + raw["elapsed_s"],
                "E267 truthy cumulative finally",
            )
        need(
            exit_code == 0
            and raw["success"] is True
            and raw["attempts"][0]["exit_code"] == 0
            and raw["attempts"][0]["killed"] is False,
            "E267 successful process closure",
        )
        report["technical_valid"] = True
    except Exception as error:
        first = {"type": type(error).__name__, "message": str(error)[:2048]}
    finally:
        cleanup_until = start + cap - 0.05 if cap else time.perf_counter() + 0.2
        identities = []
        cleanup_kills = []
        for pid, created in reversed(list(known.items())):
            state = "unknown"
            try:
                item = psutil.Process(pid)
                if item.create_time() != created:
                    state = "identity_reused"
                else:
                    if item.is_running():
                        item.kill()
                        cleanup_kills.append(pid)
                        item.wait(
                            timeout=max(
                                0, min(0.15, cleanup_until - time.perf_counter())
                            )
                        )
                    state = "ended"
            except psutil.NoSuchProcess:
                state = "ended"
            except Exception as error:
                if len(errors) < 16:
                    errors.append(str(error)[:256])
            identities.append({"pid": pid, "create_time": created, "close": state})
        if process is not None:
            try:
                process.wait(
                    timeout=max(0, min(0.1, cleanup_until - time.perf_counter()))
                )
                exit_code = process.returncode
            except Exception as error:
                errors.append(str(error)[:256])
        closed = (
            bool(identities)
            and all(v["close"] == "ended" for v in identities)
            and not unknown
        )
        elapsed = time.perf_counter() - start
        report.update(
            first_error=first,
            secondary_errors=errors[:16],
            spawned_supervisors=spawned,
            supervisor_exit_code=exit_code,
            supervisor_report_present=raw is not None,
            raw_E267_elapsed_s=None if raw is None else raw.get("elapsed_s"),
            owned_identities=identities,
            cleanup_killed_pids=cleanup_kills,
            all_owned_ended=closed,
            unknown_identity_or_close=unknown or (bool(spawned) and not closed),
            whole_elapsed_s=elapsed,
            charged_s=max(elapsed, cap if spawned and not closed else 0),
            max_tree_rss_bytes=peak,
            phase_limit_s=cap,
        )
        if first or cleanup_kills or (spawned and not closed) or elapsed > cap:
            report["technical_valid"] = False
        report["verdict"] = (
            "engineering_pass" if report["technical_valid"] else "technical_invalid"
        )
        if "cumulative_before_s" in report:
            report["cumulative_after_s"] = (
                report["cumulative_before_s"] + report["charged_s"]
            )
        # Exclusive bounded publication is independent of a supervisor report.
        for name, value in (
            (
                pids,
                {
                    "schema": "flylab.m3.e345.pid-identities.v1",
                    "identities": identities,
                    "all_owned_ended": closed,
                    "unknown_identity_or_close": report["unknown_identity_or_close"],
                },
            ),
            (receipt, report),
        ):
            payload = (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()
            need(len(payload) <= MAX_REPORT, "bounded emergency publication")
            with safe(root, name).open("xb") as stream:
                stream.write(payload)
        # Include first publication time in final measured receipt and cost.
        elapsed = time.perf_counter() - start
        report["whole_elapsed_s"] = elapsed
        report["charged_s"] = max(report["charged_s"], elapsed)
        if "cumulative_before_s" in report:
            report["cumulative_after_s"] = (
                report["cumulative_before_s"] + report["charged_s"]
            )
        if elapsed > cap:
            report["technical_valid"] = False
            report["verdict"] = "technical_invalid"
        safe(root, receipt).write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", "utf-8"
        )
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("contract")
    parser.add_argument("--authority", required=True)
    parser.add_argument("--authority-sha256", required=True)
    parser.add_argument("--receipt", required=True)
    parser.add_argument("--pids", required=True)
    args = parser.parse_args()
    report = supervise(
        args.contract,
        args.authority,
        args.authority_sha256,
        args.receipt,
        args.pids,
        started=ENTRY_S0,
    )
    return 0 if report["technical_valid"] else 1


if __name__ == "__main__":
    sys.exit(main())
