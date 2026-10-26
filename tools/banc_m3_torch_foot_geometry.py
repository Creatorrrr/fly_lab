"""Fixed-context, differentiable FlyBody foot geometry for offline M3 fitting.

The extractor reads one neutral MuJoCo configuration through a *separate*
MjData. Evaluation uses only float64 Torch operations. It does not control or
advance a body and cannot represent a changing passive-joint context.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

LEGS = ('lf', 'lm', 'lh', 'rf', 'rm', 'rh')
ARRAY_NAMES = ('reference_qpos', 'active_joint_ids', 'active_qpos_ids',
               'tip_body_ids', 'tree_joint_ids', 'tree_frontend_indices',
               'leg_frontend_indices', 'passive_joint_ids', 'passive_qpos_ids',
               'passive_qpos_values', 'reference_angles', 'axes', 'anchors', 'tips')
INTEGER_ARRAYS = set(ARRAY_NAMES[1:9])
SCHEMA = 'flylab.m3.e111.torch-foot-geometry.v1'


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=False, allow_nan=False).encode('utf-8')


def _geometry_hash(metadata, arrays):
    digest = hashlib.sha256(_canonical(metadata))
    for name in ARRAY_NAMES:
        array = arrays[name]
        digest.update(name.encode('ascii') + b'\0')
        digest.update(array.dtype.str.encode('ascii') + b'\0')
        digest.update(_canonical(array.shape))
        digest.update(array.tobytes(order='C'))
    return digest.hexdigest()


def _validated_arrays(source):
    if set(source) != set(ARRAY_NAMES):
        raise ValueError('Incomplete or unexpected geometry arrays')
    arrays = {}
    for name in ARRAY_NAMES:
        array = np.asarray(source[name])
        expected = np.int64 if name in INTEGER_ARRAYS else np.float64
        if array.dtype != expected:
            raise ValueError(f'{name} must have exact {expected} dtype')
        arrays[name] = np.ascontiguousarray(array.copy())
    for name in ('active_joint_ids', 'active_qpos_ids', 'reference_angles'):
        if arrays[name].shape != (42,):
            raise ValueError(f'{name} must have 42 frontend entries')
    for name in ('tree_joint_ids', 'tree_frontend_indices',
                 'leg_frontend_indices'):
        if arrays[name].shape != (6, 7):
            raise ValueError(f'{name} must have six seven-hinge chains')
    for name in ('tip_body_ids',):
        if arrays[name].shape != (6,):
            raise ValueError(f'{name} must have six tips')
    for name in ('axes', 'anchors'):
        if arrays[name].shape != (6, 7, 3):
            raise ValueError(f'{name} must have six seven-hinge vectors')
    if arrays['tips'].shape != (6, 3):
        raise ValueError('Six reference tip points required')
    passive = arrays['passive_joint_ids']
    if passive.ndim != 1 or arrays['passive_qpos_ids'].shape != passive.shape or \
            arrays['passive_qpos_values'].shape != passive.shape:
        raise ValueError('Passive context arrays must align')
    reference = arrays['reference_qpos']
    if reference.ndim != 1 or reference.size == 0:
        raise ValueError('One complete fixed qpos context required')
    if any(not np.isfinite(arrays[name]).all() for name in ARRAY_NAMES
           if name not in INTEGER_ARRAYS):
        raise ValueError('Nonfinite fixed geometry')
    for name in INTEGER_ARRAYS:
        if (arrays[name] < 0).any():
            raise ValueError(f'{name} must be nonnegative')
    if (arrays['active_qpos_ids'] >= reference.size).any() or \
            (arrays['passive_qpos_ids'] >= reference.size).any():
        raise ValueError('Joint qpos address outside reference context')
    if len(np.unique(arrays['active_joint_ids'])) != 42 or \
            len(np.unique(arrays['active_qpos_ids'])) != 42 or \
            len(np.unique(arrays['tip_body_ids'])) != 6:
        raise ValueError('Repeated active joint, qpos address or tip')
    if not np.array_equal(arrays['reference_angles'],
                          reference[arrays['active_qpos_ids']]):
        raise ValueError('Frontend reference angles differ from context')
    if not np.array_equal(arrays['passive_qpos_values'],
                          reference[arrays['passive_qpos_ids']]):
        raise ValueError('Passive positions differ from fixed context')
    tree = arrays['tree_frontend_indices']
    if not np.array_equal(np.sort(tree.ravel()), np.arange(42)) or \
            not np.array_equal(arrays['tree_joint_ids'],
                               arrays['active_joint_ids'][tree]):
        raise ValueError('Tree chains must map every frontend hinge exactly once')
    if not np.array_equal(arrays['leg_frontend_indices'], np.sort(tree, axis=1)):
        raise ValueError('Leg API order must be frontend order')
    if not np.allclose(np.linalg.norm(arrays['axes'], axis=-1), 1.,
                       rtol=0., atol=1e-10):
        raise ValueError('Reference hinge axes must be unit vectors')
    for array in arrays.values():
        array.flags.writeable = False
    return arrays


class TorchFootGeometry:
    """One exact qpos context; leg inputs follow sorted frontend indices."""

    def __init__(self, metadata, arrays, *, device='cpu'):
        self.arrays = _validated_arrays(arrays)
        self.metadata = dict(metadata)
        if self.metadata.get('schema') != SCHEMA or not isinstance(
                self.metadata.get('model_hash'), str) or not self.metadata['model_hash']:
            raise ValueError('Original model identity and geometry schema required')
        if self.metadata.get('legs') != list(LEGS) or \
                not isinstance(self.metadata.get('joint_names'), list) or \
                len(self.metadata['joint_names']) != 42:
            raise ValueError('Explicit frontend leg/joint names required')
        if not isinstance(self.metadata.get('thorax_body_id'), int) or \
                self.metadata['thorax_body_id'] < 0:
            raise ValueError('Thorax body identity required')
        self.geometry_hash = _geometry_hash(self.metadata, self.arrays)
        tree = self.arrays['tree_frontend_indices']
        leg = self.arrays['leg_frontend_indices']
        self._tree_local = tuple(tuple(int(np.flatnonzero(leg[i] == j)[0])
                                       for j in tree[i]) for i in range(6))
        self._tree_global = tuple(tuple(int(j) for j in row) for row in tree)
        requested_device = torch.device(device)
        self._axes = torch.as_tensor(self.arrays['axes'].copy(),
                                     dtype=torch.float64, device=requested_device)
        self._anchors = torch.as_tensor(self.arrays['anchors'].copy(),
                                        dtype=torch.float64, device=requested_device)
        self._tips = torch.as_tensor(self.arrays['tips'].copy(),
                                     dtype=torch.float64, device=requested_device)
        self._reference_angles = torch.as_tensor(
            self.arrays['reference_angles'].copy(), dtype=torch.float64,
            device=requested_device)
        # Bare "cuda" and "cpu:0" aliases must compare to the actual allocation.
        self.device = self._axes.device

    def to(self, device):
        """Stage fixed coefficients once; subsequent evaluations stay on device."""
        return TorchFootGeometry(self.metadata, self.arrays, device=device)

    def save(self, path):
        payload = {'metadata_json': np.array(_canonical(self.metadata).decode()),
                   'geometry_hash': np.array(self.geometry_hash)}
        payload.update(self.arrays)
        np.savez_compressed(Path(path), **payload)

    @classmethod
    def load(cls, path, *, device='cpu'):
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != set(ARRAY_NAMES) | {'metadata_json', 'geometry_hash'}:
                raise ValueError('Geometry archive fields changed')
            metadata = json.loads(str(archive['metadata_json'].item()))
            digest = str(archive['geometry_hash'].item())
            arrays = {name: archive[name].copy() for name in ARRAY_NAMES}
        result = cls(metadata, arrays, device=device)
        if digest != result.geometry_hash:
            raise ValueError('Geometry archive content hash mismatch')
        return result

    def _input(self, commands, width):
        if not isinstance(commands, torch.Tensor) or commands.dtype != torch.float64 \
                or commands.device != self.device or commands.ndim != 2 \
                or commands.shape[1] != width:
            raise ValueError(f'Expected float64 tensor [N,{width}] on {self.device}')
        if not bool(torch.isfinite(commands).all()):
            raise ValueError('Nonfinite joint command')

    def _point(self, command, leg_index, indices):
        point = self._tips[leg_index].expand(command.shape[0], 3)
        for k in range(6, -1, -1):
            frontend = self._tree_global[leg_index][k]
            theta = (command[:, indices[k]]-
                     self._reference_angles[frontend]).unsqueeze(1)
            axis = self._axes[leg_index, k]
            anchor = self._anchors[leg_index, k]
            relative = point-anchor
            cosine = torch.cos(theta)
            sine = torch.sin(theta)
            point = anchor + (relative*cosine +
                torch.cross(axis.expand_as(relative), relative, dim=1)*sine +
                axis*torch.sum(relative*axis, dim=1, keepdim=True)*(1-cosine))
        return point

    def full(self, commands):
        """Absolute frontend [N,42] radians -> thorax-frame [N,6,3] mm."""
        self._input(commands, 42)
        return torch.stack([self._point(commands, leg, self._tree_global[leg])
                            for leg in range(6)], dim=1)

    def leg(self, commands, leg):
        """Absolute [N,7] radians in that leg's frontend order -> [N,3] mm."""
        if not isinstance(leg, int) or not 0 <= leg < 6:
            raise ValueError('Leg index must be 0..5')
        self._input(commands, 7)
        return self._point(commands, leg, self._tree_local[leg])


def extract_from_body(body, reference_qpos):
    """Read actual tree/kinematics in a shadow MjData; never change body.d."""
    model, mj = body.m, body.mj
    qref = np.asarray(reference_qpos)
    if qref.dtype != np.float64 or qref.shape != (model.nq,) or \
            not np.isfinite(qref).all():
        raise ValueError('Exact finite original float64 qpos context required')
    active = np.asarray(body.joint_ids, dtype=np.int64)
    addresses = np.asarray(body.qpos_ids, dtype=np.int64)
    tips = np.asarray(body._tip_ids, dtype=np.int64)
    thorax = int(body.thorax)
    if active.shape != (42,) or tips.shape != (6,) or \
            addresses.shape != (42,) or len(np.unique(active)) != 42:
        raise ValueError('Original 42-hinge frontend and six tips required')
    if not np.array_equal(model.jnt_qposadr[active], addresses) or \
            any(int(model.jnt_type[j]) != int(mj.mjtJoint.mjJNT_HINGE)
                for j in active):
        raise ValueError('Every active frontend coordinate must be a hinge')
    active_map = {int(j): i for i, j in enumerate(active)}
    chains, passive = [], set()
    for tip in tips:
        bodies, cursor = [], int(tip)
        while cursor != thorax and cursor > 0:
            bodies.append(cursor)
            cursor = int(model.body_parentid[cursor])
        if cursor != thorax or not bodies:
            raise ValueError('Tip must descend uniquely from thorax')
        joint_chain = []
        for segment in reversed(bodies):
            first, count = int(model.body_jntadr[segment]), int(model.body_jntnum[segment])
            for joint in range(first, first+count):
                if joint in active_map:
                    joint_chain.append(joint)
                else:
                    kind = int(model.jnt_type[joint])
                    if kind not in (int(mj.mjtJoint.mjJNT_HINGE),
                                    int(mj.mjtJoint.mjJNT_SLIDE)):
                        raise ValueError('Unsupported passive joint in foot chain')
                    passive.add(joint)
        if len(joint_chain) != 7:
            raise ValueError('Exactly seven actual ancestor hinges per foot required')
        chains.append(joint_chain)
    tree_joint = np.asarray(chains, dtype=np.int64)
    tree_frontend = np.asarray([[active_map[int(j)] for j in row]
                                for row in tree_joint], dtype=np.int64)
    if not np.array_equal(np.sort(tree_frontend.ravel()), np.arange(42)):
        raise ValueError('Active hinge shared across legs or absent from a tip')
    data = mj.MjData(model)
    data.qpos[:] = qref
    mj.mj_kinematics(model, data)
    mj.mj_comPos(model, data)
    origin = data.xpos[thorax].copy()
    rotation = data.xmat[thorax].reshape(3, 3).copy()
    axes = np.asarray(data.xaxis[tree_joint] @ rotation, dtype=np.float64)
    anchors = np.asarray((data.xanchor[tree_joint]-origin) @ rotation,
                         dtype=np.float64)
    points = np.asarray((data.xpos[tips]-origin) @ rotation,
                        dtype=np.float64)
    passive_ids = np.asarray(sorted(passive), dtype=np.int64)
    passive_qpos_ids = np.asarray(model.jnt_qposadr[passive_ids], dtype=np.int64)
    arrays = {
        'reference_qpos': qref.copy(), 'active_joint_ids': active.copy(),
        'active_qpos_ids': addresses.copy(), 'tip_body_ids': tips.copy(),
        'tree_joint_ids': tree_joint, 'tree_frontend_indices': tree_frontend,
        'leg_frontend_indices': np.sort(tree_frontend, axis=1),
        'passive_joint_ids': passive_ids, 'passive_qpos_ids': passive_qpos_ids,
        'passive_qpos_values': qref[passive_qpos_ids].copy(),
        'reference_angles': qref[addresses].copy(),
        'axes': axes, 'anchors': anchors, 'tips': points}
    metadata = {'schema': SCHEMA, 'model_hash': str(body.model_hash),
                'legs': list(LEGS), 'joint_names': list(body.joint_names),
                'thorax_body_id': thorax,
                'context': 'original E32 neutral; all nonfrontend qpos fixed',
                'units': {'position': 'mm', 'angle': 'rad'}}
    return TorchFootGeometry(metadata, arrays)
