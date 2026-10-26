"""Passive E223 RK-stage input recorder, outside the neural checkpoint.

The observer only writes its own CUDA buffers.  The existing signed CSR,
neuronal state, effector destinations, and E223 checkpoint are unchanged.
"""

from __future__ import annotations

import numpy as np

from tools.banc_m3_neural_current_effector import NeuralCurrentEffectorNetwork

CHANNELS = (
    "positive_recurrent", "negative_magnitude", "drive",
    "effector64", "fast64", "slow64",
)
DTYPES = (np.float32, np.float32, np.float32,
          np.float64, np.float64, np.float64)
MAX_STAGES = 7920
MAX_FREE_MOTORS = 343
MAX_BUFFER_BYTES = 268435456


def observation_plan(capacity: int, free_count: int) -> dict:
    """Validate exact bounded buffer sizes before any device allocation."""
    if (type(capacity) is not int or type(free_count) is not int
            or not 1 <= capacity <= MAX_STAGES
            or not 1 <= free_count <= MAX_FREE_MOTORS):
        raise ValueError("Bounded positive stage capacity and free count required")
    shapes = ((capacity, free_count),) * 4 + ((capacity, free_count, 3),) * 2
    channel_bytes = sum(int(np.prod(shape)) * np.dtype(dtype).itemsize
                        for shape, dtype in zip(shapes, DTYPES))
    auxiliary_bytes = free_count * np.dtype(np.int64).itemsize + 8
    planned = channel_bytes + auxiliary_bytes
    if planned > MAX_BUFFER_BYTES:
        raise ValueError("Observation buffers exceed 256 MiB")
    return {
        "capacity": capacity,
        "free_count": free_count,
        "shapes": dict(zip(CHANNELS, shapes)),
        "dtypes": dict(zip(CHANNELS, DTYPES)),
        "channel_bytes": channel_bytes,
        "auxiliary_bytes": auxiliary_bytes,
        "planned_bytes": planned,
    }


def validate_observation_roster(matrix, motor_ids, free_motor_ids):
    """Reject malformed ownership before allocating any observation storage."""
    shape = getattr(matrix, "shape", None)
    motor = np.asarray(motor_ids)
    free = np.asarray(free_motor_ids)
    if (not isinstance(shape, tuple) or len(shape) != 2
            or type(shape[0]) is not int or shape[0] < 2
            or shape[1] != shape[0]
            or motor.dtype != np.int64 or motor.ndim != 1
            or free.dtype != np.int64 or free.ndim != 1
            or len(motor) < 2 or not 1 <= len(free) <= MAX_FREE_MOTORS
            or len(np.unique(motor)) != len(motor)
            or len(np.unique(free)) != len(free)
            or np.any((motor < 0) | (motor >= shape[0]))
            or not np.array_equal(free, motor[np.isin(motor, free)])):
        raise ValueError("Ordered int64 free motor subset and square network required")
    return len(free)


def stage_record_sources(name, tensors):
    """Select the actual E221 stage inputs, rejecting a changed ABI."""
    expected = {"native_stage": 27, "native_finish": 22}
    if name not in expected or len(tensors) != expected[name]:
        raise ValueError("Unexpected E221 native stage/finish argument layout")
    return tensors[0], tensors[1], tensors[2], tensors[3], *tensors[15:18]


_RECORD_SOURCE = r"""
extern "C" __global__ void record_effector_stage(
    const float* positive, const float* negative, const float* drive,
    const int* free_ids, const double* effector, const double* fast,
    const double* slow, float* out_positive, float* out_negative,
    float* out_drive, double* out_effector, double* out_fast,
    double* out_slow, long long* counters, const int* armed,
    int* overflow, int capacity, int free_count) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= free_count || armed[0] == 0) return;
    long long row = counters[j];
    counters[j] = row + 1;
    if (row >= capacity) { atomicExch(overflow, 1); return; }
    long long place = row * free_count + j;
    out_positive[place] = positive[j];
    out_negative[place] = negative[j];
    out_drive[place] = drive[free_ids[j]];
    out_effector[place] = effector[j];
    for (int k = 0; k < 3; ++k) {
        long long coordinate = 3 * place + k;
        out_fast[coordinate] = fast[3*j+k];
        out_slow[coordinate] = slow[3*j+k];
    }
}
"""


class EffectorStageObserverNetwork(NeuralCurrentEffectorNetwork):
    """Same E223 model with a one-epoch, write-only RK-stage observation."""

    def __init__(self, *args, observation_capacity=MAX_STAGES, **kwargs):
        import torch

        if not args:
            raise ValueError("Square complete-neuron matrix required")
        free_count = validate_observation_roster(
            args[0], kwargs.get("motor_ids"), kwargs.get("free_motor_ids"))
        plan = observation_plan(observation_capacity, free_count)
        if (kwargs.get("device") != "cuda"
                or kwargs.get("cuda_implementation") != "packed"
                or kwargs.get("dt") != .00025
                or not torch.cuda.is_available()):
            raise ValueError("Stage observer requires packed CUDA dt=.00025")
        self._construction_stream = int(torch.cuda.current_stream().cuda_stream)
        free_info, _ = torch.cuda.mem_get_info()
        if free_info < plan["planned_bytes"]:
            raise MemoryError("Insufficient free GPU memory for observation buffers")
        self._observation_plan = plan
        self._observation_values = {}
        torch_dtype = {np.float32: torch.float32, np.float64: torch.float64}
        for channel in CHANNELS:
            value = torch.empty(plan["shapes"][channel],
                                dtype=torch_dtype[plan["dtypes"][channel]],
                                device="cuda")
            value.fill_(float("nan"))
            self._observation_values[channel] = value
        self._observation_counters = torch.zeros(plan["free_count"],
                                                  dtype=torch.int64, device="cuda")
        self._observation_armed = torch.zeros(1, dtype=torch.int32, device="cuda")
        self._observation_overflow = torch.zeros(1, dtype=torch.int32, device="cuda")
        self._observation_pointer_ids = tuple(
            value.data_ptr() for value in self._observation_values.values()
        ) + (self._observation_counters.data_ptr(),
             self._observation_armed.data_ptr(), self._observation_overflow.data_ptr())
        self._observation_ever_armed = False
        self._observation_active = False
        self._observation_stopped = False
        self._observation_start_tick = None
        self._observation_end_tick = None
        self._observation_stream = None
        super().__init__(*args, **kwargs)
        self._observation_source_pointers = {
            key: getattr(self, key).data_ptr()
            for key in ("_positive", "_negative", "drive", "_free_row",
                        "_free_rate64", "_free_fast64", "_free_slow64",
                        "_stage_rate64", "_stage_fast64", "_stage_slow64")
        }
        self._observation_capture_complete = True
        self._assert_observation_layout()
        self._assert_observation_sources()

    def _init_fusion(self):
        super()._init_fusion()
        self._observation_kernel = self._cupy.RawKernel(
            _RECORD_SOURCE, "record_effector_stage", options=("--fmad=false",))
        self._observation_kernel.compile()

    def _assert_observation_layout(self):
        tensors = tuple(self._observation_values.values()) + (
            self._observation_counters, self._observation_armed,
            self._observation_overflow,
        )
        plan = self._observation_plan
        torch_dtype = {np.float32: self.torch.float32,
                       np.float64: self.torch.float64}
        if (tuple(value.data_ptr() for value in tensors)
                != self._observation_pointer_ids
                or tuple(self._observation_values) != CHANNELS
                or any(tuple(self._observation_values[channel].shape)
                       != plan["shapes"][channel]
                       or self._observation_values[channel].dtype
                       != torch_dtype[plan["dtypes"][channel]]
                       for channel in CHANNELS)
                or tuple(self._observation_counters.shape) != (plan["free_count"],)
                or self._observation_counters.dtype != self.torch.int64
                or any(tuple(value.shape) != (1,) or value.dtype != self.torch.int32
                       for value in (self._observation_armed,
                                     self._observation_overflow))
                or any(value.device.type != "cuda" for value in tensors)):
            raise ValueError("Captured observation buffer pointer or layout changed")

    def _current_stream_id(self):
        return int(self.torch.cuda.current_stream().cuda_stream)

    def _require_stream(self):
        if (getattr(self, "_observation_capture_complete", False)
                and self._current_stream_id() != self._construction_stream):
            raise ValueError("Stage observation construction CUDA stream changed")

    def _assert_observation_sources(self):
        if not hasattr(self, "_observation_source_pointers"):
            return  # Constructor warmup/capture precedes the fixed-source receipt.
        f = self._observation_plan["free_count"]
        expected = {
            "_positive": ((f,), self.torch.float32),
            "_negative": ((f,), self.torch.float32),
            "drive": ((self.n,), self.torch.float32),
            "_free_row": ((f,), self.torch.int32),
            "_free_rate64": ((f,), self.torch.float64),
            "_free_fast64": ((f, 3), self.torch.float64),
            "_free_slow64": ((f, 3), self.torch.float64),
            "_stage_rate64": ((f,), self.torch.float64),
            "_stage_fast64": ((f, 3), self.torch.float64),
            "_stage_slow64": ((f, 3), self.torch.float64),
        }
        if any(getattr(self, key).data_ptr() != pointer
               or tuple(getattr(self, key).shape) != expected[key][0]
               or getattr(self, key).dtype != expected[key][1]
               or getattr(self, key).device.type != "cuda"
               for key, pointer in self._observation_source_pointers.items()):
            raise ValueError("Captured observation source pointer or layout changed")

    def _launch_native(self, name, tensors, step):
        sources = stage_record_sources(name, tensors)
        f = self._observation_plan["free_count"]
        expected = (
            ((f,), self.torch.float32), ((f,), self.torch.float32),
            ((self.n,), self.torch.float32), ((f,), self.torch.int32),
            ((f,), self.torch.float64), ((f, 3), self.torch.float64),
            ((f, 3), self.torch.float64),
        )
        if any(tuple(value.shape) != shape or value.dtype != dtype
               or value.device.type != "cuda"
               for value, (shape, dtype) in zip(sources, expected)):
            raise ValueError("E221 native observation source layout changed")
        self._require_stream()
        if self._observation_ever_armed:
            self._assert_observation_layout()
            self._assert_observation_sources()
        kernel = self._observation_kernel
        values = self._observation_values
        kernel(((self._observation_plan["free_count"] + 127) // 128,), (128,),
               tuple(np.uint64(value.data_ptr()) for value in sources)
               + tuple(np.uint64(values[channel].data_ptr()) for channel in CHANNELS)
               + (np.uint64(self._observation_counters.data_ptr()),
                  np.uint64(self._observation_armed.data_ptr()),
                  np.uint64(self._observation_overflow.data_ptr()),
                  np.int32(self._observation_plan["capacity"]),
                  np.int32(self._observation_plan["free_count"])))
        return super()._launch_native(name, tensors, step)

    def _advance(self, drive, steps, *, readout_indices=None):
        self._require_stream()
        self._assert_observation_layout()
        self._assert_observation_sources()
        return super()._advance(drive, steps, readout_indices=readout_indices)

    def arm_observation(self):
        if self._observation_ever_armed or self._observation_stopped:
            raise ValueError("Stage observation may be armed only once")
        if (not getattr(self, "_observation_capture_complete", False)
                or self.torch.cuda.is_current_stream_capturing()):
            raise ValueError("Stage observation requires completed graph capture")
        self._require_stream()
        self._assert_observation_layout()
        self._assert_hot_path_layout()
        self._assert_observation_sources()
        self.torch.cuda.current_stream().synchronize()
        self._observation_start_tick = int(self.tick)
        self._observation_stream = self._current_stream_id()
        self._observation_ever_armed = True
        self._observation_active = True
        self._observation_armed.fill_(1)
        self.torch.cuda.current_stream().synchronize()

    def disable_observation(self):
        self._require_stream()
        if self._observation_stopped:
            return
        if not self._observation_ever_armed:
            raise ValueError("Stage observation has not been armed")
        self._assert_observation_layout()
        self._assert_observation_sources()
        stream = self.torch.cuda.current_stream()
        stream.synchronize()
        self._observation_armed.zero_()
        stream.synchronize()
        self._observation_end_tick = int(self.tick)
        self._observation_active = False
        self._observation_stopped = True

    def _require_stopped(self):
        if not self._observation_stopped:
            raise ValueError("Stop stage observation before host readback")
        self._require_stream()
        self._assert_observation_layout()
        self._assert_observation_sources()

    def copy_values(self, channel):
        self._require_stopped()
        if channel not in CHANNELS:
            raise ValueError("Unknown observation channel")
        return self._observation_values[channel].detach().cpu().numpy().copy()

    def copy_counters(self):
        self._require_stopped()
        return self._observation_counters.detach().cpu().numpy().copy()

    def copy_overflow(self):
        self._require_stopped()
        return int(self._observation_overflow.detach().cpu().numpy()[0])

    def copy_armed(self):
        self._require_stopped()
        return int(self._observation_armed.detach().cpu().numpy()[0])

    def observation_static_metadata(self):
        plan = self._observation_plan
        return {
            "channels": CHANNELS,
            "dtypes": {key: np.dtype(value).name
                       for key, value in plan["dtypes"].items()},
            "shapes": plan["shapes"].copy(),
            "capacity": plan["capacity"],
            "free_count": plan["free_count"],
            "planned_bytes": plan["planned_bytes"],
            "start_tick": self._observation_start_tick,
            "end_tick": self._observation_end_tick,
            "construction_stream": self._construction_stream,
            "stream": self._observation_stream,
            "armed_once": self._observation_ever_armed,
            "stopped": self._observation_stopped,
        }

    def reset(self):
        if self._observation_ever_armed:
            raise ValueError("Reset after arming stage observation is prohibited")
        self._require_stream()
        return super().reset()

    def restore(self, saved):
        if self._observation_ever_armed:
            raise ValueError("Restore after arming stage observation is prohibited")
        self._require_stream()
        return super().restore(saved)


__all__ = (
    "CHANNELS", "EffectorStageObserverNetwork", "observation_plan",
    "stage_record_sources", "validate_observation_roster",
)
