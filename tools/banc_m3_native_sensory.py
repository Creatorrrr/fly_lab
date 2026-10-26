"""E195 research adapter: causally integrate native hook/club features."""

# Cleanup catches preserve the primary model exception.
# ruff: noqa: BLE001

from __future__ import annotations

import copy
import hashlib
import json
import math
from typing import Any

import numpy as np

from flylab.c.neuromuscular import typed_receptor_feature
from tools.banc_m3_monotone_recruitment import MonotoneRecruitmentTrial
from tools.verify_c_banc_observation import exact

ADAPTER_SCHEMA = "flylab.m3.e195.native-sensory.adapter.v1"
TRIAL_SCHEMA = "flylab.m3.e195.native-sensory.trial.v1"
PATCH_SCHEMA = "flylab.m3.e195.native-sensory.patch.v1"
CONTROL_DT = 0.005
NATIVE_DT = 0.0001
SAMPLES = 50
KINDS = frozenset(("hook", "club"))
ALGORITHM = "post-step-shadow-state-feature-mean-before-boundary-filter"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def candidate_identity(motor_identity: str) -> str:
    require(isinstance(motor_identity, str) and len(motor_identity) == 64,
            "complete source motor identity required")
    payload = {"schema": ADAPTER_SCHEMA, "source_motor_identity": motor_identity,
               "algorithm": ALGORITHM, "control_dt_s": CONTROL_DT,
               "native_dt_s": NATIVE_DT, "samples_per_control": SAMPLES}
    return hashlib.sha256(json.dumps(payload, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def native_feature(kind: str, cell: str, angle: float, velocity: float,
                   lowpass: float, spec: dict) -> float:
    require(kind in KINDS and all(math.isfinite(x) for x in (angle, velocity, lowpass)),
            "finite native hook/club sample required")
    return typed_receptor_feature(kind, cell, angle, velocity, lowpass, 0., 0., spec)


def validate_trial_boundary(control_tick: Any, encode_count: Any,
                            sample_count: Any) -> None:
    require(type(control_tick) is int and control_tick >= 0
            and type(encode_count) is int and encode_count == control_tick
            and type(sample_count) is int
            and (sample_count == 0 if control_tick == 0 else sample_count == SAMPLES),
            "native/control boundary clock mismatch")


class NativeVelocityAdapter:
    """Composition keeps `.original` bound to the original RateBodyAdapter."""

    schema = ADAPTER_SCHEMA

    def __init__(self, motor: Any):
        require(hasattr(motor, "identity") and hasattr(motor, "original")
                and hasattr(motor.original, "_receptor_keys"),
                "complete monotone motor and RateBodyAdapter required")
        self._motor = motor
        self.identity = candidate_identity(motor.identity)
        leaf = motor.original
        self._ports = np.asarray([i for i, feature in enumerate(leaf._port_features)
                                  if leaf._receptor_keys[feature][1] in KINDS], np.int64)
        require(self._ports.size > 0 and len(set(self._ports.tolist())) == len(self._ports),
                "native hook/club ports required")
        self._sum = np.zeros(len(self._ports), np.float64)
        self._duration = 0.
        self._count = 0
        self._start_s = 0.
        self._end_s = 0.
        self._encode_count = 0
        self._last_mean = np.zeros(len(self._ports), np.float64)
        self._rows: list[dict[str, Any]] = []

    def __getattr__(self, name: str) -> Any:
        return getattr(self._motor, name)

    @property
    def native_ports(self) -> np.ndarray:
        return self._ports.copy()

    @property
    def native_rows(self) -> list[dict[str, Any]]:
        return self._rows

    def record_failure(self, time_s: float, error: BaseException) -> None:
        self._rows.append({"native_index": len(self._rows),
                           "control_tick": self._encode_count,
                           "substep": self._count + 1, "time_s": time_s,
                           "complete": False, "error": repr(error)})

    def record(self, time_s: float, flexion: Any, velocity: Any) -> None:
        """Called once after each original physics step, before next encode."""
        row: dict[str, Any] = {"native_index": len(self._rows),
                               "control_tick": self._encode_count,
                               "substep": self._count + 1, "complete": False}
        self._rows.append(row)
        try:
            q = np.asarray(flexion)
            v = np.asarray(velocity)
            row["time_s"] = float(time_s)
            require(q.shape == v.shape == (6,) and np.isfinite(q).all()
                    and np.isfinite(v).all(), "finite six-knee native geometry required")
            expected = ((self._encode_count - 1) * CONTROL_DT
                        + (self._count + 1) * NATIVE_DT)
            require(self._encode_count > 0 and math.isfinite(time_s)
                    and abs(time_s - expected) <= 2e-8,
                    "native sample clock/order")
            require(self._count < SAMPLES, "more than 50 native samples before encode")
            previous = self._end_s
            sample_dt = time_s - previous
            require(0 < sample_dt <= 2 * NATIVE_DT, "native elapsed sample duration")
            leaf = self._motor.original
            lowpass = leaf.velocity_lowpass[leaf.joints[:, 5]].copy()
            values = np.empty(len(self._ports), np.float64)
            for j, port in enumerate(self._ports):
                leg, kind, cell = leaf._receptor_keys[leaf._port_features[port]]
                values[j] = native_feature(kind, cell, q[leg], v[leg], lowpass[leg], leaf.spec)
            require(np.isfinite(values).all() and np.all((values >= 0) & (values <= 1)),
                    "native receptor feature range")
            self._sum += sample_dt * values
            self._duration += sample_dt
            self._count += 1
            if self._count == 1:
                self._start_s = previous
            self._end_s = time_s
            row.update(interval_dt_s=sample_dt, flexion_rad=q.copy(),
                       velocity_rad_s=v.copy(), velocity_lowpass_rad_s=lowpass,
                       features=values.copy(), complete=True)
        except BaseException as error:
            row["error"] = repr(error)
            raise

    def encode(self, *, sensory_cut: bool = False, dt: float = CONTROL_DT) -> Any:
        require(dt == CONTROL_DT, "fixed E195 encode clock")
        leaf = self._motor.original
        old = leaf.filtered.copy()
        if self._encode_count == 0:
            require(self._count == 0 and self._duration == 0,
                    "tick zero must have no previous physics interval")
            drive = self._motor.encode(sensory_cut=sensory_cut, dt=dt)
        else:
            require(self._count == SAMPLES and abs(self._duration - CONTROL_DT) <= 2e-8
                    and abs(self._start_s - (self._encode_count - 1) * CONTROL_DT) <= 2e-8
                    and abs(self._end_s - self._encode_count * CONTROL_DT) <= 2e-8,
                    "only complete preceding 50-sample interval may be encoded")
            mean = self._sum / self._duration
            drive = self._motor.encode(sensory_cut=sensory_cut, dt=dt)
            alpha = 1 - math.exp(-dt / 0.02)
            leaf.filtered[self._ports] = old[self._ports] + alpha * (
                leaf.sensory_gain * mean - old[self._ports])
            leaf.last_features[self._ports] = mean
            self._last_mean = mean.copy()
            self._sum.fill(0)
            self._count = 0
            self._duration = 0.
            self._start_s = self._end_s
        self._encode_count += 1
        return drive

    def snapshot(self) -> dict[str, Any]:
        return {"schema": self.schema, "hash": self.identity,
                "source_motor_identity": self._motor.identity,
                "motor": self._motor.snapshot(), "native_ports": self._ports.copy(),
                "sum": self._sum.copy(), "duration_s": self._duration,
                "sample_count": self._count, "start_s": self._start_s,
                "end_s": self._end_s, "encode_count": self._encode_count,
                "last_mean": self._last_mean.copy()}

    def restore(self, saved: dict[str, Any]) -> None:
        require(isinstance(saved, dict) and set(saved) == set(self.snapshot()),
                "complete native adapter snapshot required")
        require(saved["schema"] == self.schema and saved["hash"] == self.identity
                and saved["source_motor_identity"] == self._motor.identity
                and np.array_equal(saved["native_ports"], self._ports),
                "native adapter identity/port mismatch")
        count, encoded = saved["sample_count"], saved["encode_count"]
        require(type(count) is int and 0 <= count <= SAMPLES
                and type(encoded) is int and 0 <= encoded <= 10**10,
                "native sample/encode count")
        summ = np.asarray(saved["sum"])
        mean = np.asarray(saved["last_mean"])
        duration = saved["duration_s"]
        require(type(duration) in (float, int) and math.isfinite(duration)
                and 0 <= duration <= CONTROL_DT + 2e-8,
                "native weighted duration")
        require(summ.shape == mean.shape == self._sum.shape
                and summ.dtype == mean.dtype == np.float64
                and np.isfinite(summ).all() and np.isfinite(mean).all()
                and np.all((summ >= 0) & (summ <= duration + 1e-14))
                and np.all((mean >= 0) & (mean <= 1)), "native feature accumulator")
        start, end = saved["start_s"], saved["end_s"]
        require(all(type(x) in (float, int) and math.isfinite(x)
                    for x in (duration, start, end))
                and abs(duration - count * NATIVE_DT) <= 2e-8,
                "native weighted duration")
        if count:
            require(encoded > 0 and abs(start - (encoded - 1) * CONTROL_DT) <= 2e-8
                    and abs(end - (start + duration)) <= 2e-8,
                    "pending native interval clock")
        else:
            expected_end = (encoded - 1) * CONTROL_DT if encoded else 0.
            require(duration == 0 and np.array_equal(summ, np.zeros_like(summ))
                    and abs(start - expected_end) <= 2e-8
                    and abs(end - expected_end) <= 2e-8
                    and abs(start - end) <= 1e-14
                    and (encoded > 0 or np.array_equal(mean, np.zeros_like(mean))),
                    "empty native interval state/clock")
        before = self._motor.snapshot()
        try:
            self._motor.restore(saved["motor"])
            self._sum[:] = summ
            self._last_mean[:] = mean
            self._duration, self._count = float(duration), count
            self._start_s, self._end_s = float(start), float(end)
            self._encode_count = encoded
        except BaseException as error:
            try:
                self._motor.restore(before)
            except BaseException as rollback:
                error.add_note("motor rollback failed: " + repr(rollback))
            raise


class NativeSensoryTrial(MonotoneRecruitmentTrial):
    schema = TRIAL_SCHEMA

    def __init__(self, base: MonotoneRecruitmentTrial):
        require(type(base) is MonotoneRecruitmentTrial and not base.closed,
                "complete original monotone trial required")
        self.__dict__.update(base.__dict__)
        original_adapter = self.adapter
        self._native_restore = None
        try:
            candidate = NativeVelocityAdapter(original_adapter)
            self.native_sensory_patch = {"schema": PATCH_SCHEMA,
                "source_adapter_identity": original_adapter.identity,
                "adapter_identity": candidate.identity,
                "algorithm": ALGORITHM, "control_dt_s": CONTROL_DT,
                "native_dt_s": NATIVE_DT, "samples_per_control": SAMPLES}
            self.adapter = candidate
            self._install_native_observer()
            base.closed = True
        except BaseException as error:
            base.closed = True
            try:
                self.close()
            except BaseException as cleanup:
                error.add_note("candidate cleanup failed: " + repr(cleanup))
            raise

    def _install_native_observer(self) -> None:
        body = self.body
        existed = "_step_physics" in vars(body)
        prior = vars(body).get("_step_physics")
        original = body._step_physics
        shadow = copy.copy(body)
        shadow.d = body.mj.MjData(body.m)
        mj = body.mj
        state_type = mj.mjtState.mjSTATE_INTEGRATION
        state = np.empty(mj.mj_stateSize(body.m, state_type))
        actor_after = np.empty_like(state)

        def observe_step():
            original()
            before_rows = len(self.adapter.native_rows)
            try:
                mj.mj_getState(body.m, body.d, state, state_type)
                mj.mj_setState(body.m, shadow.d, state, state_type)
                mj.mj_forward(body.m, shadow.d)
                knee = shadow.knee_kinematics(include_direction=False)
                mj.mj_getState(body.m, body.d, actor_after, state_type)
                require(np.array_equal(state, actor_after),
                        "shadow observation changed actor integration state")
                self.adapter.record(body.physics_time(), knee["flexion_rad"],
                                    knee["velocity_rad_s"])
            except BaseException as error:
                if len(self.adapter.native_rows) == before_rows:
                    try:
                        self.adapter.record_failure(body.physics_time(), error)
                    except BaseException as cleanup:
                        error.add_note("native failure record failed: " + repr(cleanup))
                raise

        body._step_physics = observe_step

        def undo():
            if existed:
                body._step_physics = prior
            else:
                delattr(body, "_step_physics")

        self._native_restore = undo

    def snapshot(self) -> dict[str, Any]:
        return {**super().snapshot(),
                "native_sensory_patch": copy.deepcopy(self.native_sensory_patch)}

    @classmethod
    def from_checkpoint(cls, graph: Any, saved: dict[str, Any]) -> NativeSensoryTrial:
        require(isinstance(saved, dict) and saved.get("schema") == cls.schema,
                "explicit native sensory checkpoint required")
        patch = saved.get("native_sensory_patch")
        adapter = saved.get("adapter")
        require(isinstance(patch, dict) and isinstance(adapter, dict)
                and patch.get("schema") == PATCH_SCHEMA
                and adapter.get("schema") == ADAPTER_SCHEMA,
                "complete native sensory provenance required")
        require(patch == {"schema": PATCH_SCHEMA,
                "source_adapter_identity": adapter.get("source_motor_identity"),
                "adapter_identity": adapter.get("hash"), "algorithm": ALGORITHM,
                "control_dt_s": CONTROL_DT, "native_dt_s": NATIVE_DT,
                "samples_per_control": SAMPLES}, "native patch changed")
        validate_trial_boundary(saved.get("control_tick"), adapter.get("encode_count"),
                                adapter.get("sample_count"))
        source = copy.deepcopy(saved)
        source.pop("native_sensory_patch")
        source["schema"] = MonotoneRecruitmentTrial.schema
        source["adapter"] = adapter["motor"]
        base = MonotoneRecruitmentTrial.from_checkpoint(graph, source)
        trial = None
        try:
            exact(source, base.snapshot(), "full underlying monotone source")
            trial = cls(base)
            trial.adapter.restore(adapter)
            validate_trial_boundary(trial.control_tick, trial.adapter._encode_count,
                                    trial.adapter._count)
            exact(saved, trial.snapshot(), "full native sensory checkpoint")
            return trial
        except BaseException as error:
            owner = trial if trial is not None else base
            try:
                if not owner.closed:
                    owner.close()
            except BaseException as cleanup:
                error.add_note("body cleanup failed: " + repr(cleanup))
            raise

    def close(self) -> None:
        try:
            if self._native_restore is not None:
                restore, self._native_restore = self._native_restore, None
                restore()
        finally:
            super().close()
