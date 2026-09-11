"""Explicit MuJoCo-Warp candidate with synchronized CPU observation mirrors.

The first integration preserves the existing 10 kHz CPU reflex controller.
It is an instrumentable correctness path, not an assertion of GPU speedup.
No state is integrated on the CPU after the common neutral settling phase.
"""
from dataclasses import fields, is_dataclass
from importlib.metadata import version
import copy
import hashlib
import json
import numpy as np
from .body import FlyGymBody
from .physics import PhysicsProfile, profile_values, warp_option_metadata
from . import PHYSICS_DT


class WarpBody(FlyGymBody):
    backend = 'flygym-2.1.0-mujoco-warp-3.9.0'

    def __init__(self, seed, world, config, *, physics_profile=None, **kwargs):
        self.gpu_data = None
        profile = profile_values(physics_profile or PhysicsProfile(backend='warp'))
        if profile.backend != 'warp':
            raise ValueError('WarpBody requires a warp physics profile')
        import warp as wp
        import mujoco_warp as mjw
        if version('warp-lang')!='1.14.0' or version('mujoco-warp')!='3.9.0':
            raise RuntimeError('Install the pinned physics packages from requirements-warp.txt')
        if not wp.is_cuda_available():
            raise RuntimeError('Warp physics requires an available NVIDIA CUDA device')
        self.wp, self.mjw = wp, mjw
        self.device = wp.get_device('cuda:0')
        self.stream = wp.Stream(self.device)
        self.step_graph = self.forward_graph = None
        self.gpu_ticks = 0
        self.capacity_peak = {'contacts': 0, 'constraints': 0, 'broadphase': 0}
        self.runtime_identity = dict(warp=version('warp-lang'), mujoco_warp=version('mujoco-warp'),
                                     device=self.device.name, architecture=self.device.arch,
                                     cuda_driver=list(wp.get_cuda_driver_version()))
        super().__init__(seed, world, config, physics_profile=profile, **kwargs)
        self.model_hash = hashlib.sha256(json.dumps(dict(model=self.model_hash,
            warp=self.runtime_identity), sort_keys=True).encode()).hexdigest()
        with wp.ScopedStream(self.stream):
            self.gpu_model = mjw.put_model(self.m)
            self.gpu_options = warp_option_metadata(self.gpu_model)
            self.gpu_data = self._put_data()
            # Compile kernels against private dynamic state, before clock start.
            mjw.step(self.gpu_model, self.gpu_data)
            mjw.forward(self.gpu_model, self.gpu_data)
            wp.synchronize_stream(self.stream)
            self.gpu_data = self._put_data()
            if profile.cuda_graph:
                with wp.ScopedCapture() as step_capture:
                    mjw.step(self.gpu_model, self.gpu_data)
                self.step_graph = step_capture.graph
                with wp.ScopedCapture() as forward_capture:
                    mjw.forward(self.gpu_model, self.gpu_data)
                self.forward_graph = forward_capture.graph
        self._sync_observations()

    def _put_data(self):
        return self.mjw.put_data(self.m, self.d, nworld=1,
            njmax=self.physics_profile.max_constraints, nconmax=self.physics_profile.max_contacts)

    def _upload_inputs(self):
        for name in ('ctrl', 'xfrc_applied', 'qfrc_applied'):
            value = np.asarray(getattr(self.d, name), dtype=np.float32)[None]
            getattr(self.gpu_data, name).assign(value)

    def _sync_observations(self):
        wp, d = self.wp, self.gpu_data
        with wp.ScopedStream(self.stream):
            wp.synchronize_stream(self.stream)
            counts = dict(contacts=int(d.nacon.numpy()[0]), constraints=int(d.nefc.numpy().max()),
                          broadphase=int(d.ncollision.numpy()[0]))
            for name, value in counts.items():
                self.capacity_peak[name] = max(self.capacity_peak[name], value)
            if counts['contacts'] > d.naconmax or counts['broadphase'] > d.naconmax or counts['constraints'] > d.njmax:
                self.fault = 'Warp contact/constraint buffer overflow; result rejected'
                raise RuntimeError(self.fault)
            # get_data_into clamps overflows; the check above must precede it.
            self.mjw.get_data_into(self.d, self.m, d)
            # Upstream does not copy contact.exclude into reused CPU storage.
            # Only contacts with constraint rows are active for body force queries.
            n = self.d.ncon
            # MuJoCo 3.9 keeps legacy geom1/geom2 fields separately from geom.
            # mjw.get_data_into only fills geom, including after reallocation.
            self.d.contact.geom1[:n] = self.d.contact.geom[:n, 0]
            self.d.contact.geom2[:n] = self.d.contact.geom[:n, 1]
            self.d.contact.exclude[:n] = (self.d.contact.efc_address[:n] < 0).astype(np.int32)
        # Public time is an integer tick clock, not an accumulated float32 sum.
        self.d.time = self.t0 + self.gpu_ticks * PHYSICS_DT

    def _step_physics(self):
        if self.gpu_data is None:
            return super()._step_physics()  # common CPU neutral settling only
        with self.wp.ScopedStream(self.stream):
            self._upload_inputs()
            self.gpu_data.time.assign(np.array([self.t0 + self.gpu_ticks * PHYSICS_DT], np.float32))
            if self.step_graph is None:
                self.mjw.step(self.gpu_model, self.gpu_data)
            else:
                self.wp.capture_launch(self.step_graph)
        self.gpu_ticks += 1
        self._sync_observations()

    def _forward_physics(self):
        if self.gpu_data is None:
            return super()._forward_physics()
        with self.wp.ScopedStream(self.stream):
            if self.forward_graph is None:
                self.mjw.forward(self.gpu_model, self.gpu_data)
            else:
                self.wp.capture_launch(self.forward_graph)
        self._sync_observations()

    def physics_time(self):
        return self.gpu_ticks * PHYSICS_DT

    def set_config(self, config):
        super().set_config(config)
        if self.gpu_data is not None:
            # Model constants change; captured graphs must be rebuilt before use.
            with self.wp.ScopedStream(self.stream):
                self.wp.synchronize_stream(self.stream)
                self.gpu_model = self.mjw.put_model(self.m)
                self.gpu_options = warp_option_metadata(self.gpu_model)
            self.step_graph = self.forward_graph = None

    def physics_identity(self):
        identity=super().physics_identity()
        if self.gpu_data is not None:
            identity['cpu_mirror_options']={k:identity[k] for k in self.gpu_options}
            identity.update(self.gpu_options)
        return dict(identity, precision='float32', runtime=self.runtime_identity,
                    observation_source='MuJoCo-Warp with synchronized CPU mirror',
                    settling_backend='CPU with matching physics options',
                    controller_backend='CPU hybrid reflex at each physics tick',
                    experimental=True, capacity_peak=dict(self.capacity_peak))

    def _device_arrays(self):
        result = {}
        def collect(obj, prefix=''):
            for field in fields(obj):
                value = getattr(obj, field.name)
                key = prefix + field.name
                if isinstance(value, self.wp.array):
                    result[key] = value
                elif is_dataclass(value):
                    collect(value, key + '.')
        collect(self.gpu_data)
        return result

    def snapshot(self):
        # Save device state including warm-start and solver/contact workspace.
        # Inactive workspace may be uninitialized; only integration state is
        # subject to finiteness checks. Shapes/dtypes of every buffer are checked.
        self.wp.synchronize_stream(self.stream)
        result = super().snapshot()
        result['warp'] = dict(runtime=copy.deepcopy(self.runtime_identity), ticks=self.gpu_ticks,
            arrays={name: value.numpy().copy() for name, value in self._device_arrays().items()})
        return result

    def restore(self, state):
        gpu = state.get('warp', {})
        if gpu.get('runtime') != self.runtime_identity:
            raise ValueError('Warp checkpoint runtime mismatch')
        ticks = gpu.get('ticks')
        if type(ticks) is not int or ticks < 0:
            raise ValueError('Invalid Warp physics tick')
        arrays = self._device_arrays()
        saved = gpu.get('arrays', {})
        if set(saved) != set(arrays):
            raise ValueError('Warp checkpoint array inventory mismatch')
        for name, target in arrays.items():
            value = np.asarray(saved[name])
            expected = target.numpy()
            if value.shape != expected.shape or value.dtype != expected.dtype:
                raise ValueError('Warp checkpoint shape/dtype mismatch: ' + name)
            if name in ('qpos', 'qvel', 'act', 'ctrl', 'qacc_warmstart') and not np.isfinite(value).all():
                raise ValueError('Non-finite Warp integration state: ' + name)
        super().restore(state)
        with self.wp.ScopedStream(self.stream):
            for name, target in arrays.items():
                target.assign(saved[name])
        self.gpu_ticks = ticks
        self._sync_observations()

    def close(self):
        if hasattr(self, 'stream'):
            self.wp.synchronize_stream(self.stream)
        super().close()
