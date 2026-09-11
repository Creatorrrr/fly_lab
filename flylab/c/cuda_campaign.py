"""Persistent homogeneous-model CUDA queues with explicit OOM backoff.

No job, recording channel or neuron is dropped to make an allocation fit.
An interrupted period is failed; only committed control boundaries resume.
"""

from dataclasses import asdict
import gc
import json
import math
from pathlib import Path
import time

from .batch import BatchSession
from .integrity import checked_name, digest, read_json, write_json
from .cuda_campaign_state import validate_resume
from ..body_options import BodyOptions
from ..sensors import default_world, validate_world
from ..common import clone


def allocation_failure(exc):
    if isinstance(exc, MemoryError):
        return True
    name = type(exc).__name__
    text = str(exc).lower()
    return name == "OutOfMemoryError" or (
        "cuda" in text
        and any(
            s in text for s in ("out of memory", "out_of_memory", "error_out_of_memory")
        )
    )


def release_unused_cuda():
    gc.collect()
    try:
        import cupy as cp

        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    except (ImportError, RuntimeError):
        pass


class AllocationDeferred(RuntimeError):
    """No simulation period started; a later run can retry safely."""


class Campaign:
    def __init__(
        self, graph, bindings, spec, out, *, resume=False, session_type=BatchSession
    ):
        if (
            set(spec) - {"schema", "jobs", "seconds", "max_worlds", "record_channels"}
            or spec.get("schema") != "flylab.cuda-campaign.v1"
        ):
            raise ValueError("Invalid CUDA campaign schema")
        self.graph, self.bindings, self.session_type = graph, bindings, session_type
        self.spec = clone(spec)
        seconds = spec.get("seconds", 0.05)
        if (
            type(seconds) not in (int, float)
            or not math.isfinite(seconds)
            or not 0.005 <= seconds <= 3600
        ):
            raise ValueError("Invalid campaign duration")
        self.steps = round(seconds / 0.005)
        if abs(self.steps * 0.005 - seconds) > 1e-9:
            raise ValueError(
                "Campaign duration must contain integral 5 ms periods in .005..3600 s"
            )
        self.limit = spec.get("max_worlds", 32)
        self.channels = spec.get("record_channels", 512)
        if (
            type(self.limit) is not int
            or not 1 <= self.limit <= 32
            or type(self.channels) is not int
            or not 0 <= self.channels <= 512
        ):
            raise ValueError("Use 1..32 worlds and 0..512 recording channels")
        jobs = spec.get("jobs")
        if not isinstance(jobs, list) or not 1 <= len(jobs) <= 4096:
            raise ValueError("Use 1..4096 campaign jobs")
        self.jobs = {}
        for job in jobs:
            if not isinstance(job, dict) or set(job) - {
                "id",
                "seed",
                "mode",
                "body_options",
                "world",
            }:
                raise ValueError("Invalid campaign job")
            key = checked_name(job.get("id"))
            seed = job.get("seed", 42)
            mode = job.get("mode", "C_SHADOW")
            if key in self.jobs or type(seed) is not int or not 0 <= seed < 2**32:
                raise ValueError("Duplicate job or invalid seed")
            if mode not in ("C_STRICT", "C_SHADOW", "C_ASSISTED"):
                raise ValueError("Invalid campaign mode")
            options = asdict(BodyOptions.parse(job.get("body_options")))
            world = clone(
                validate_world(
                    default_world() if job.get("world") is None else job["world"]
                )
            )
            self.jobs[key] = dict(
                id=key,
                seed=seed,
                mode=mode,
                body_options=options,
                world=world,
                group=digest(dict(mode=mode, body_options=options, world=world)),
            )
        self.identity = digest(
            dict(
                graph=graph.hash,
                bindings=bindings.hash,
                jobs=self.jobs,
                steps=self.steps,
                max_worlds=self.limit,
                record_channels=self.channels,
            )
        )
        self.out = Path(out)
        self.path = self.out / "campaign.json"
        self.session = None
        if resume:
            self.state = read_json(self.path)
            if self.state.get("identity") != self.identity:
                raise ValueError("Campaign identity mismatch")
            self._validate_resume()
            self.state.setdefault("pending_chunks", [])
            self.limit = self.state.get("capacity", self.limit)
        else:
            self.out.mkdir(parents=True, exist_ok=False)
            self.state = dict(
                schema="flylab.cuda-campaign-state.v1",
                identity=self.identity,
                status="QUEUED",
                graph_hash=graph.hash,
                binding_hash=bindings.hash,
                spec=spec,
                jobs={
                    key: dict(status="QUEUED", tick=0, group=j["group"])
                    for key, j in self.jobs.items()
                },
                current=None,
                serial=0,
                allocation_attempts=[],
                completed_chunks=[],
                pending_chunks=[],
                cancelled_ids=[],
            )
            self._save()

    def _validate_resume(self):
        validate_resume(self)

    def _save(self):
        write_json(self.path, self.state)

    def cancel(self, ids):
        if not isinstance(ids, list) or any(key not in self.jobs for key in ids):
            raise ValueError("Unknown campaign cancellation job")
        self.state["cancelled_ids"] = sorted(
            set(self.state["cancelled_ids"]) | set(ids)
        )
        for key in ids:
            if self.state["jobs"][key]["status"] == "QUEUED":
                self.state["jobs"][key]["status"] = "CANCELLED"
        self._save()

    def _split_current(self):
        source = self.state["current"]
        size = max(1, len(source["jobs"]) // 2)
        self.limit = min(self.limit, size)
        self.state["capacity"] = self.limit
        worlds = source.get(
            "checkpoint_worlds", [str(i) for i in range(len(source["jobs"]))]
        )
        chunks = []
        for offset in range(0, len(worlds), size):
            self.state["serial"] += 1
            chunk = clone(source)
            chunk.update(
                jobs=source["jobs"][offset : offset + size],
                checkpoint_worlds=worlds[offset : offset + size],
                serial=self.state["serial"],
                record_segment=0,
                saves=0,
            )
            chunks.append(chunk)
        self.state["current"] = chunks[0]
        self.state["pending_chunks"] = chunks[1:] + self.state["pending_chunks"]
        self.state["status"] = "PAUSED"
        self._save()

    def _restore_current(self):
        while True:
            current = self.state["current"]
            attempt = dict(
                phase="restore", jobs=list(current["jobs"]), worlds=len(current["jobs"])
            )
            try:
                kwargs = (
                    {"worlds": current["checkpoint_worlds"]}
                    if "checkpoint_worlds" in current
                    else {}
                )
                self.session = self.session_type.restore(
                    self.graph,
                    self.bindings,
                    self.out / current["checkpoint"],
                    **kwargs,
                )
            except Exception as exc:
                oom = allocation_failure(exc)
                error = f"{type(exc).__name__}: {exc}"
                attempt.update(status="OOM" if oom else "FAILED", error=error)
                self.state["allocation_attempts"].append(attempt)
                if not oom:
                    raise
            else:
                attempt["status"] = "READY"
                self.state["allocation_attempts"].append(attempt)
                current["controls"] = current["checkpoint_controls"]
                self._sync()
                return
            # Leave the exception scope before collecting failed constructors.
            release_unused_cuda()
            if len(current["jobs"]) == 1:
                raise AllocationDeferred(error)
            self._split_current()

    def _start(self):
        if self.state["current"] is None and self.state["pending_chunks"]:
            self.state["current"] = self.state["pending_chunks"].pop(0)
        if self.state["current"] is not None:
            self._restore_current()
            current = self.state["current"]
        else:
            pending = [
                key for key, s in self.state["jobs"].items() if s["status"] == "QUEUED"
            ]
            if not pending:
                return False
            group = self.jobs[pending[0]]["group"]
            selected = [key for key in pending if self.jobs[key]["group"] == group][
                : self.limit
            ]
            first = self.jobs[selected[0]]
            while True:
                attempt = dict(
                    phase="create", jobs=list(selected), worlds=len(selected)
                )
                try:
                    self.session = self.session_type(
                        self.graph,
                        self.bindings,
                        [self.jobs[k]["seed"] for k in selected],
                        mode=first["mode"],
                        body_options=first["body_options"],
                        world=first["world"],
                    )
                    attempt["status"] = "READY"
                    self.state["allocation_attempts"].append(attempt)
                    break
                except Exception as exc:
                    oom = allocation_failure(exc)
                    error = f"{type(exc).__name__}: {exc}"
                    attempt.update(status="OOM" if oom else "FAILED", error=error)
                    self.state["allocation_attempts"].append(attempt)
                    if not oom:
                        raise
                release_unused_cuda()
                if len(selected) == 1:
                    raise AllocationDeferred(error)
                self.limit = max(1, len(selected) // 2)
                selected = selected[: self.limit]
                self.state["capacity"] = self.limit
                self._save()
            self.state["serial"] += 1
            current = dict(
                jobs=selected,
                group=group,
                serial=self.state["serial"],
                controls=0,
                checkpoint=None,
                model_hashes=sorted(
                    {e.body.model_hash for e in self.session.engines.values()}
                ),
            )
            if len(current["model_hashes"]) != 1:
                raise RuntimeError("Compiled campaign model hashes disagree")
            self.state["current"] = current
            self._checkpoint()
        current["record_segment"] = current.get("record_segment", 0) + 1
        # Commit the segment number before touching its directory. An interrupted
        # recording start must not reuse an existing segment on the next resume.
        self._sync()
        self.state["status"] = "RUNNING"
        self.state.pop("waiting_for_memory", None)
        self._save()
        ids = [n["id"] for n in self.graph.nodes[: min(self.channels, self.graph.n)]]
        for i, key in enumerate(current["jobs"]):
            status = self.session.status[str(i)]
            self.state["jobs"][key]["status"] = status
            if self.channels and status == "RUNNING":
                engine = self.session.engines[str(i)]
                engine.start_recording(
                    self.out
                    / "recordings"
                    / key
                    / f"segment-{current['serial']}-{current['record_segment']}",
                    ids,
                )
                engine.event(
                    "campaign_resume",
                    dict(
                        campaign=self.identity,
                        job=key,
                        chunk=current["serial"],
                        checkpoint=current["checkpoint"],
                        checkpoint_controls=current["checkpoint_controls"],
                        regroup_origin=self.session.regroup_origin,
                    ),
                )
        self.state["status"] = "RUNNING"
        self._save()
        return True

    def _checkpoint(self):
        current = self.state["current"]
        relative = (
            Path("checkpoints")
            / f"chunk-{current['serial']}-step-{current['controls']}-save-{current.get('saves', 0)}"
        )
        # A crash between saving a directory and committing campaign.json can
        # leave an orphan; never overwrite or collide with that checkpoint.
        while (self.out / relative).exists():
            current["saves"] = current.get("saves", 0) + 1
            relative = (
                Path("checkpoints")
                / f"chunk-{current['serial']}-step-{current['controls']}-save-{current['saves']}"
            )
        saved = self.session.checkpoint(self.out / relative)
        current["checkpoint"] = relative.as_posix()
        current["checkpoint_hash"] = saved["state_hash"]
        current["saves"] = current.get("saves", 0) + 1
        current["checkpoint_controls"] = current["controls"]
        current.pop("checkpoint_worlds", None)
        if self.session.regroup_origin:
            current["regroup_origin"] = clone(self.session.regroup_origin)

    def _sync(self):
        if self.session is None or not self.state.get("current"):
            return
        for i, key in enumerate(self.state["current"]["jobs"]):
            e = self.session.engines[str(i)]
            self.state["jobs"][key].update(
                status=self.session.status[str(i)],
                tick=e.tick,
                fault=e.fault or e.body.fault,
            )

    def run(self, *, max_controls=None):
        if max_controls is not None and (
            type(max_controls) is not int or max_controls < 1
        ):
            raise ValueError("Positive max_controls required")
        self._validate_resume()
        used = 0
        started = time.perf_counter()
        try:
            while self._start():
                current = self.state["current"]
                while current["controls"] < self.steps:
                    requests = self.out / "cancel.json"
                    if requests.is_file():
                        self.cancel(json.loads(requests.read_text(encoding="utf-8")))
                    for i, key in enumerate(current["jobs"]):
                        if key in self.state["cancelled_ids"] and self.session.status[
                            str(i)
                        ] in ("RUNNING", "PAUSED"):
                            self.session.cancel(str(i))
                    if not self.session.active:
                        break
                    self.session.step(1)
                    current["controls"] += 1
                    used += 1
                    self._sync()
                    if (
                        max_controls is not None
                        and used >= max_controls
                        and current["controls"] < self.steps
                    ):
                        self._checkpoint()
                        self.state["status"] = "PAUSED"
                        self._save()
                        return self.state
                self._sync()
                self._checkpoint()
                self.session.close()
                self.session = None
                for key in current["jobs"]:
                    if self.state["jobs"][key]["status"] == "RUNNING":
                        self.state["jobs"][key]["status"] = "COMPLETE"
                self.state["completed_chunks"].append(dict(current))
                self.state["current"] = None
                self._save()
                release_unused_cuda()
                if max_controls is not None and used >= max_controls:
                    self.state["status"] = (
                        "PAUSED"
                        if self.state["pending_chunks"]
                        or any(
                            j["status"] == "QUEUED" for j in self.state["jobs"].values()
                        )
                        else self._completed_status()
                    )
                    self._save()
                    return self.state
            self.state["status"] = self._completed_status()
            self._save()
            return self.state
        except AllocationDeferred as exc:
            self.state.update(
                status="PAUSED",
                waiting_for_memory=dict(
                    reason=str(exc),
                    retry="Run again with --resume; committed checkpoints are preserved",
                ),
            )
            self._save()
            return self.state
        except BaseException as exc:
            try:
                self._sync()
            except Exception as sync_error:
                self.state["sync_error"] = str(sync_error)
            self.state.update(status="FAILED", error=f"{type(exc).__name__}: {exc}")
            self._save()
            raise
        finally:
            if self.session is not None:
                try:
                    self.session.close()
                except Exception as close_error:
                    self.state.update(status="FAILED", close_error=str(close_error))
                    # Keep an existing execution exception as the primary cause.
                    if "error" not in self.state:
                        raise
                finally:
                    self.session = None
                    self._save()
            self.state["last_run_wall_s"] = time.perf_counter() - started
            self._save()

    def _completed_status(self):
        if self.state["current"] is not None or self.state["pending_chunks"]:
            raise ValueError("Campaign still has committed work to resume")
        evidence = {
            key for chunk in self.state["completed_chunks"] for key in chunk["jobs"]
        }
        for key, job in self.state["jobs"].items():
            if (
                job["status"] == "COMPLETE"
                and job["tick"] == self.steps * 50
                and key in evidence
            ):
                continue
            if (
                job["status"] == "CANCELLED"
                and key in self.state["cancelled_ids"]
                and (job["tick"] == 0 or key in evidence)
            ):
                continue
            raise ValueError("Campaign has unfinished or unproven work: " + key)
        self.state.pop("waiting_for_memory", None)
        return "COMPLETE"
