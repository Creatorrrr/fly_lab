"""Cross-check campaign journals against committed, hashed batch checkpoints."""

import math

from .integrity import digest, read_json
from .storage import StateStore


def checkpoint_metadata(path, graph_hash, binding_hash):
    """Read and verify checkpoint bytes without allocating a CUDA session."""
    try:
        saved = StateStore.load(path, max_files=4096)
        manifest = read_json(path / "manifest.json")
        if (
            saved.get("schema") != "flylab.full-batch-state.v1"
            or saved.get("graph_hash") != graph_hash
            or saved.get("binding_hash") != binding_hash
        ):
            raise ValueError("batch identity mismatch")
        engines = saved.get("engines", {})
        keys = [str(i) for i in range(len(engines))]
        statuses = saved.get("status", {})
        if (
            not 1 <= len(keys) <= 32
            or set(engines) != set(keys)
            or set(statuses) != set(keys)
            or any(s not in ("RUNNING", "CANCELLED") for s in statuses.values())
        ):
            raise ValueError("invalid campaign checkpoint worlds or statuses")
        active = [key for key in keys if statuses[key] == "RUNNING"]
        if saved.get("active") != active:
            raise ValueError("active world order mismatch")
        metadata = {}
        for key, engine in engines.items():
            tick = engine.get("control_tick")
            neural, encoder, body = engine["neural"], engine["encoder"], engine["body"]
            if (
                engine.get("schema") != "flylab.checkpoint.v3"
                or engine.get("graph_hash") != graph_hash
                or engine.get("binding_hash") != binding_hash
                or neural.get("graph_hash") != graph_hash
                or encoder.get("binding_hash") != binding_hash
                or neural.get("backend") != "exp_lif_cuda"
            ):
                raise ValueError("engine identity mismatch")
            if (
                type(tick) is not int
                or tick < 0
                or neural.get("tick") != tick * 50
                or encoder.get("control_tick") != tick
                or body.get("walk_ticks") != tick
                or engine.get("sensor_tick") != max(0, (tick - 1) * 50)
                or not math.isclose(
                    body["state"][0] - body["t0"], tick * 0.005, rel_tol=0, abs_tol=1e-8
                )
            ):
                raise ValueError("engine checkpoint clock mismatch")
            if (
                body.get("fault")
                or body.get("backend") != "flygym-warp-batch-view-v1"
                or body.get("modelHash") != engine.get("body_model_hash")
                or engine.get("config_hash") != digest(engine.get("config"))
                or engine.get("environment_hash") != digest(engine.get("world"))
            ):
                raise ValueError("engine body/configuration mismatch")
            metadata[key] = dict(
                status=statuses[key],
                tick=tick * 50,
                seed=engine["seed"],
                mode=engine["mode"],
                body_options=engine["body_options"],
                world=engine["world"],
                model_hash=engine["body_model_hash"],
                t0=body["t0"],
            )
        physics = saved.get("physics")
        if active:
            if (
                not isinstance(physics, dict)
                or physics.get("schema") != "flylab.warp-batch.v1"
                or physics.get("worlds") != len(active)
                or any(
                    metadata[key]["model_hash"] != physics.get("model_hash")
                    for key in active
                )
            ):
                raise ValueError("physics identity mismatch")
            ticks, times = physics["controller"]["ticks"], physics["arrays"]["time"]
            if ticks.shape != (len(active),) or times.shape != (len(active),):
                raise ValueError("physics clock inventory mismatch")
            for row, key in enumerate(active):
                if int(ticks[row]) != metadata[key]["tick"] or not math.isclose(
                    float(times[row]),
                    metadata[key]["t0"] + int(ticks[row]) * 0.0001,
                    rel_tol=1e-6,
                    abs_tol=1e-6,
                ):
                    raise ValueError("physics checkpoint clock mismatch")
        elif physics is not None:
            raise ValueError("inactive checkpoint has physics")
        return dict(hash=manifest["state_hash"], worlds=metadata)
    except (
        OSError,
        KeyError,
        TypeError,
        AttributeError,
        IndexError,
        ValueError,
    ) as exc:
        raise ValueError(f"Invalid campaign checkpoint {path.name}: {exc}") from exc


def validate_resume(campaign):
    s, jobs = campaign.state, campaign.jobs
    if (
        s.get("schema") != "flylab.cuda-campaign-state.v1"
        or s.get("identity") != campaign.identity
        or s.get("graph_hash") != campaign.graph.hash
        or s.get("binding_hash") != campaign.bindings.hash
        or s.get("spec") != campaign.spec
        or s.get("status") not in ("QUEUED", "RUNNING", "PAUSED", "COMPLETE", "FAILED")
    ):
        raise ValueError("Invalid campaign state identity")
    ledger, cancelled = s.get("jobs"), s.get("cancelled_ids")
    if (
        not isinstance(ledger, dict)
        or set(ledger) != set(jobs)
        or not isinstance(cancelled, list)
        or any(k not in jobs for k in cancelled)
        or len(cancelled) != len(set(cancelled))
    ):
        raise ValueError("Invalid campaign job inventory")
    capacity = s.get("capacity", campaign.limit)
    if type(capacity) is not int or not 1 <= capacity <= campaign.spec.get(
        "max_worlds", 32
    ):
        raise ValueError("Invalid campaign capacity")
    target = campaign.steps * 50
    for key, job in ledger.items():
        if (
            not isinstance(job, dict)
            or job.get("group") != jobs[key]["group"]
            or job.get("status")
            not in ("QUEUED", "RUNNING", "COMPLETE", "CANCELLED", "FAILED")
            or job.get("fault")
        ):
            raise ValueError("Invalid campaign job state")
        tick = job.get("tick")
        if type(tick) is not int or not 0 <= tick <= target or tick % 50:
            raise ValueError("Invalid campaign job clock")
        if (
            (job["status"] == "QUEUED" and (tick != 0 or key in cancelled))
            or (job["status"] == "COMPLETE" and tick != target)
            or (job["status"] == "CANCELLED" and key not in cancelled)
        ):
            raise ValueError("Campaign job status/clock mismatch")

    completed, pending = s.get("completed_chunks"), s.get("pending_chunks", [])
    current = s.get("current")
    if (
        not isinstance(completed, list)
        or not isinstance(pending, list)
        or (current is not None and not isinstance(current, dict))
    ):
        raise ValueError("Invalid campaign chunks")
    chunks = [(c, True) for c in completed] + [
        (c, False) for c in ([current] if current is not None else []) + pending
    ]
    assigned, serials, checkpoints = set(), set(), {}
    for chunk, terminal in chunks:
        if not isinstance(chunk, dict):
            raise ValueError("Invalid campaign chunk")
        keys = chunk.get("jobs", [])
        if (
            not isinstance(keys, list)
            or not keys
            or len(keys) != len(set(keys))
            or any(
                k not in jobs or jobs[k]["group"] != chunk.get("group") for k in keys
            )
            or assigned.intersection(keys)
        ):
            raise ValueError("Invalid or duplicate campaign chunk jobs")
        # Resolve before opening any checkpoint; both completed and pending
        # references obey the same output-directory boundary.
        if not isinstance(chunk.get("checkpoint"), str):
            raise ValueError("Uncommitted campaign cannot resume")
        path = (campaign.out / chunk["checkpoint"]).resolve()
        root = (campaign.out / "checkpoints").resolve()
        if (
            not root.is_relative_to(campaign.out.resolve())
            or path == root
            or not path.is_relative_to(root)
        ):
            raise ValueError("Campaign checkpoint escapes output directory")
        for field in ("controls", "checkpoint_controls"):
            if (
                type(chunk.get(field)) is not int
                or not 0 <= chunk[field] <= campaign.steps
            ):
                raise ValueError("Invalid campaign checkpoint clock")
        if chunk["checkpoint_controls"] > chunk["controls"] or (
            terminal and chunk["checkpoint_controls"] != chunk["controls"]
        ):
            raise ValueError("Campaign committed clock mismatch")
        serial = chunk.get("serial")
        if type(serial) is not int or serial < 1 or serial in serials:
            raise ValueError("Invalid campaign chunk serial")
        serials.add(serial)
        if path not in checkpoints:
            checkpoints[path] = checkpoint_metadata(
                path, campaign.graph.hash, campaign.bindings.hash
            )
        checkpoint = checkpoints[path]
        if chunk.get("checkpoint_hash", checkpoint["hash"]) != checkpoint["hash"]:
            raise ValueError("Campaign checkpoint hash mismatch")
        worlds = chunk.get("checkpoint_worlds", [str(i) for i in range(len(keys))])
        if (
            not isinstance(worlds, list)
            or len(worlds) != len(keys)
            or len(set(worlds)) != len(worlds)
            or any(w not in checkpoint["worlds"] for w in worlds)
            or (
                "checkpoint_worlds" not in chunk
                and len(keys) != len(checkpoint["worlds"])
            )
        ):
            raise ValueError("Invalid campaign checkpoint world mapping")
        models = sorted({checkpoint["worlds"][world]["model_hash"] for world in worlds})
        if len(models) != 1 or chunk.get("model_hashes") != models:
            raise ValueError("Campaign model hash mismatch")
        for key, world in zip(keys, worlds):
            saved, requested, recorded = (
                checkpoint["worlds"][world],
                jobs[key],
                ledger[key],
            )
            options = dict(requested["body_options"], render_camera=True)
            if (
                saved["seed"] != requested["seed"]
                or saved["mode"] != requested["mode"]
                or saved["body_options"] != options
                or saved["world"] != requested["world"]
            ):
                raise ValueError("Campaign checkpoint job identity mismatch: " + key)
            if saved["tick"] > chunk["checkpoint_controls"] * 50 or (
                saved["status"] == "RUNNING"
                and saved["tick"] != chunk["checkpoint_controls"] * 50
            ):
                raise ValueError("Campaign checkpoint progress mismatch")
            if terminal:
                expected = "COMPLETE" if saved["status"] == "RUNNING" else "CANCELLED"
                if (
                    recorded["status"] != expected
                    or recorded["tick"] != saved["tick"]
                    or (expected == "COMPLETE" and saved["tick"] != target)
                ):
                    raise ValueError(
                        "Campaign completion has no matching checkpoint progress"
                    )
            else:
                # An interrupted journal can be ahead of its last checkpoint.
                # Only RUNNING -> CANCELLED is possible after that boundary;
                # run() rolls clocks back to the verified checkpoint on restore.
                allowed = (
                    ("RUNNING", "CANCELLED")
                    if saved["status"] == "RUNNING"
                    else ("CANCELLED",)
                )
                if (
                    recorded["status"] not in allowed
                    or not saved["tick"] <= recorded["tick"] <= chunk["controls"] * 50
                    or (
                        recorded["status"] == "RUNNING"
                        and recorded["tick"] != chunk["controls"] * 50
                    )
                    or (
                        saved["status"] == "CANCELLED"
                        and recorded["tick"] != saved["tick"]
                    )
                ):
                    raise ValueError("Campaign current job/checkpoint mismatch")
        assigned.update(keys)
    if type(s.get("serial")) is not int or s["serial"] < max(serials, default=0):
        raise ValueError("Invalid campaign serial")
    for key, job in ledger.items():
        if key not in assigned and not (
            job["tick"] == 0 and job["status"] in ("QUEUED", "CANCELLED")
        ):
            raise ValueError("Campaign job has no checkpoint evidence: " + key)
    all_terminal = all(
        j["status"] in ("COMPLETE", "CANCELLED") for j in ledger.values()
    )
    if s["status"] == "COMPLETE" and (
        current is not None or pending or not all_terminal
    ):
        raise ValueError("Campaign COMPLETE contradicts pending work")
    if s["status"] == "QUEUED" and (
        chunks
        or any(
            j["tick"] or j["status"] not in ("QUEUED", "CANCELLED")
            for j in ledger.values()
        )
    ):
        raise ValueError("Campaign QUEUED contradicts progress")
    if s["status"] == "FAILED":
        raise ValueError("A faulted campaign cannot resume")
