"""Synthetic CPU-only E111 geometry tests; no FlyBody or CUDA model creation."""

from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from tools import verify_c_banc_m3_e111_geometry as verifier
from tools.banc_m3_torch_foot_geometry import (
    LEGS,
    SCHEMA,
    TorchFootGeometry,
    extract_from_body,
)


def synthetic_geometry():
    reference = np.linspace(-.2, .2, 42, dtype=np.float64)
    tree = np.array([[7*i+2, 7*i+0, 7*i+6, 7*i+1, 7*i+5,
                      7*i+3, 7*i+4] for i in range(6)], dtype=np.int64)
    axes = np.zeros((6, 7, 3), dtype=np.float64)
    axes[:, :, 2] = 1.
    axes[:, 1, :] = [1., 0., 0.]
    anchors = np.zeros((6, 7, 3), dtype=np.float64)
    anchors[:, 1, 0] = 1.
    tips = np.tile(np.array([[1., 2., 3.]], dtype=np.float64), (6, 1))
    tips[:, 0] += np.arange(6)
    arrays = {
        'reference_qpos': reference,
        'active_joint_ids': np.arange(100, 142, dtype=np.int64),
        'active_qpos_ids': np.arange(42, dtype=np.int64),
        'tip_body_ids': np.arange(200, 206, dtype=np.int64),
        'tree_joint_ids': np.arange(100, 142, dtype=np.int64)[tree],
        'tree_frontend_indices': tree,
        'leg_frontend_indices': np.sort(tree, axis=1),
        'passive_joint_ids': np.empty(0, dtype=np.int64),
        'passive_qpos_ids': np.empty(0, dtype=np.int64),
        'passive_qpos_values': np.empty(0, dtype=np.float64),
        'reference_angles': reference.copy(),
        'axes': axes, 'anchors': anchors, 'tips': tips,
    }
    metadata = {'schema': SCHEMA, 'model_hash': 'synthetic-only',
                'legs': list(LEGS),
                'joint_names': [f'joint_{i}' for i in range(42)],
                'thorax_body_id': 1, 'context': 'synthetic',
                'units': {'position': 'mm', 'angle': 'rad'}}
    return TorchFootGeometry(metadata, arrays)


class TorchFootGeometryTests(unittest.TestCase):
    def setUp(self):
        self.geometry = synthetic_geometry()

    def test_reference_and_ordered_two_hinge_rotation(self):
        geom = self.geometry
        base = torch.tensor(geom.arrays['reference_angles'][None],
                            dtype=torch.float64)
        np.testing.assert_allclose(geom.full(base).numpy(),
                                   geom.arrays['tips'][None], rtol=0, atol=1e-14)
        command = base.clone()
        first, second = geom.arrays['tree_frontend_indices'][0, :2]
        command[0, first] += np.pi/2
        command[0, second] += np.pi/2
        # First apply the distal X hinge through (1,0,0), then the proximal Z
        # hinge through the origin; the reverse order gives a different point.
        tip = geom.arrays['tips'][0]
        distal = np.array([tip[0], -tip[2], tip[1]])
        expected = np.array([-distal[1], distal[0], distal[2]])
        actual = geom.full(command).numpy()
        np.testing.assert_allclose(actual[0, 0], expected, rtol=0, atol=1e-12)
        np.testing.assert_allclose(actual[0, 1:], geom.arrays['tips'][1:],
                                   rtol=0, atol=1e-12)

    def test_frontend_leg_mapping_and_cross_leg_gradient(self):
        geom = self.geometry
        q = np.tile(geom.arrays['reference_angles'], (3, 1))
        q[:, geom.arrays['tree_frontend_indices'][2, 0]] += [.1, -.2, .3]
        full = torch.tensor(q, dtype=torch.float64, requires_grad=True)
        point = geom.full(full)
        local = torch.tensor(q[:, geom.arrays['leg_frontend_indices'][2]],
                             dtype=torch.float64, requires_grad=True)
        np.testing.assert_allclose(geom.leg(local, 2).detach().numpy(),
                                   point[:, 2].detach().numpy(), rtol=0, atol=1e-14)
        derivative, = torch.autograd.grad(point[:, 2, 0].sum(), full)
        foreign = np.setdiff1d(np.arange(42),
                               geom.arrays['leg_frontend_indices'][2])
        self.assertTrue(torch.equal(derivative[:, foreign],
                                    torch.zeros_like(derivative[:, foreign])))

    def test_autograd_matches_finite_difference(self):
        geom = self.geometry
        q = torch.tensor(geom.arrays['reference_angles'][None].copy(),
                         dtype=torch.float64, requires_grad=True)
        joint = int(geom.arrays['tree_frontend_indices'][0, 0])
        derivative, = torch.autograd.grad(geom.full(q)[0, 0, 0], q)
        h = 1e-6
        plus = q.detach().clone(); plus[0, joint] += h
        minus = q.detach().clone(); minus[0, joint] -= h
        estimate = (geom.full(plus)[0, 0, 0]-geom.full(minus)[0, 0, 0])/(2*h)
        self.assertAlmostEqual(derivative[0, joint].item(), estimate.item(), places=8)

    def test_shape_dtype_device_nonfinite_and_leg_index(self):
        geom = self.geometry
        good = torch.zeros((2, 42), dtype=torch.float64)
        for bad in (good[:, :41], good.float(), good[0],
                    good.clone().index_fill(1, torch.tensor([0]), float('nan'))):
            with self.assertRaises(ValueError):
                geom.full(bad)
        with self.assertRaises(ValueError):
            geom.leg(torch.zeros((2, 7), dtype=torch.float64), 6)
        with self.assertRaises(ValueError):
            geom.leg(torch.full((2, 7), float('inf'), dtype=torch.float64), 0)

    def test_allocated_device_canonicalizes_cpu_alias(self):
        geometry = self.geometry.to('cpu:0')
        self.assertEqual(geometry.device, torch.device('cpu'))
        input_tensor = torch.zeros((2, 42), dtype=torch.float64)
        self.assertEqual(geometry.full(input_tensor).shape, (2, 6, 3))

    def test_archive_roundtrip_and_tamper_detection(self):
        geom = self.geometry
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'synthetic.npz'
            geom.save(path)
            restored = TorchFootGeometry.load(path)
            self.assertEqual(geom.geometry_hash, restored.geometry_hash)
            command = torch.zeros((3, 42), dtype=torch.float64)
            self.assertTrue(torch.equal(geom.full(command), restored.full(command)))
            with np.load(path, allow_pickle=False) as archive:
                payload = {key: archive[key].copy() for key in archive.files}
            payload['tips'][0, 0] += 1e-3
            np.savez_compressed(path, **payload)
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                TorchFootGeometry.load(path)

    def test_reject_invalid_chain_mapping_and_context(self):
        geom = self.geometry
        arrays = {key: value.copy() for key, value in geom.arrays.items()}
        arrays['tree_frontend_indices'][1, 0] = arrays['tree_frontend_indices'][0, 0]
        with self.assertRaises(ValueError):
            TorchFootGeometry(geom.metadata, arrays)
        arrays = {key: value.copy() for key, value in geom.arrays.items()}
        arrays['reference_angles'][0] += .1
        with self.assertRaises(ValueError):
            TorchFootGeometry(geom.metadata, arrays)

    def test_extractor_reads_shadow_data_and_actual_tree_order(self):
        # Model-free tiny stand-in: one thorax and six one-segment descendants.
        # Seven joints on each descendant, frontend actuator order permuted.
        frontend = np.arange(42, dtype=np.int64)[::-1]
        model = SimpleNamespace(nq=42, body_parentid=np.array([0, 0, 1, 1, 1, 1, 1, 1]),
            body_jntadr=np.array([0, 0, 0, 7, 14, 21, 28, 35]),
            body_jntnum=np.array([0, 0, 7, 7, 7, 7, 7, 7]),
            jnt_qposadr=np.arange(42, dtype=np.int64),
            jnt_type=np.ones(42, dtype=np.int64))
        class FakeData:
            def __init__(self, unused):
                self.qpos = np.zeros(42, dtype=np.float64)
                self.xpos = np.zeros((8, 3), dtype=np.float64)
                self.xpos[2:] = np.arange(6)[:, None]
                self.xmat = np.tile(np.eye(3).ravel(), (8, 1))
                self.xaxis = np.tile([0., 0., 1.], (42, 1))
                self.xanchor = np.zeros((42, 3), dtype=np.float64)
        fake_mj = SimpleNamespace(MjData=FakeData,
            mj_kinematics=lambda m, d: None, mj_comPos=lambda m, d: None,
            mjtJoint=SimpleNamespace(mjJNT_HINGE=1, mjJNT_SLIDE=2))
        body = SimpleNamespace(m=model, mj=fake_mj, joint_ids=frontend,
            qpos_ids=frontend, _tip_ids=np.arange(2, 8), thorax=1,
            model_hash='fake', joint_names=[str(i) for i in frontend])
        geom = extract_from_body(body, np.zeros(42, dtype=np.float64))
        np.testing.assert_array_equal(geom.arrays['tree_joint_ids'][0], np.arange(7))
        np.testing.assert_array_equal(geom.arrays['tree_frontend_indices'][0],
                                      np.arange(41, 34, -1))
        self.assertEqual(geom.metadata['model_hash'], 'fake')
        self.assertEqual(geom.arrays['tips'].shape, (6, 3))

    def test_validator_closes_constructed_body_when_initial_snapshot_fails(self):
        class Body:
            closed = False
            d = SimpleNamespace(time=.2)

            def snapshot(self):
                raise OSError('synthetic snapshot failure')

            def close(self):
                self.closed = True

        body = Body()
        fake_inputs = (np.zeros((1, 42), dtype=np.float64),
                       np.zeros(42), np.zeros(42), {'synthetic': True})
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with (patch.object(verifier, 'input_commands', return_value=fake_inputs),
                  patch.object(verifier.StateStore, 'load',
                               return_value={'body': {'modelHash': 'fake'}}),
                  patch.object(verifier, 'new_body_owned', return_value=body),
                  self.assertRaisesRegex(OSError, 'synthetic snapshot failure')):
                verifier.run(root, root/'single-output')
            self.assertTrue(body.closed)
            cleanup = json.loads((root/'single-output/cleanup.json').read_text())
            self.assertTrue(cleanup['clock_exact'])
            self.assertTrue(cleanup['closed'])
            self.assertIsNone(cleanup['snapshot_exact'])
            error = json.loads((root/'single-output/execution-error.json').read_text())
            self.assertEqual(error['error']['type'], 'OSError')

    def test_owned_constructor_closes_on_restore_failure_and_keeps_primary(self):
        class Body:
            close_attempted = False

            def restore(self, unused):
                raise ValueError('synthetic restore failure')

            def close(self):
                self.close_attempted = True
                raise OSError('synthetic close failure')

        body = Body()
        saved = {'seed': 1, 'world': {}, 'parameters': {}, 'body': {}}
        noop = lambda *args, **kwargs: None
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            with (patch.object(verifier.observation, 'configure_torque_capacity', noop),
                  patch.object(verifier.observation, 'configure_activation_trajectory', noop),
                  patch.object(verifier.observation, 'configure_active_length', noop),
                  patch.object(verifier.observation, 'configure_finite_pad_contact', noop),
                  patch.object(verifier.observation, 'configure_adhesion_units', noop),
                  self.assertRaisesRegex(ValueError, 'synthetic restore failure')):
                verifier.new_body_owned(saved, output,
                    body_factory=lambda *args, **kwargs: body)
            self.assertTrue(body.close_attempted)
            record = json.loads((output/'construction-error.json').read_text())
            self.assertEqual(record['error']['type'], 'ValueError')
            self.assertEqual(record['close_error']['type'], 'OSError')

    def test_native_prefix_save_failure_preserves_native_error(self):
        class BrokenGeometry:
            def evaluate(self, unused, *, jacobian):
                raise ValueError('synthetic native failure')

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            with (patch.object(verifier, 'StaticFootGeometry',
                               return_value=BrokenGeometry()),
                  patch.object(verifier.np, 'savez_compressed',
                               side_effect=OSError('synthetic prefix save failure')),
                  self.assertRaisesRegex(ValueError, 'synthetic native failure')):
                verifier.native_values(object(), np.zeros(42),
                    np.zeros((1, 42)), time.perf_counter(), output)
            record = json.loads((output/'native-prefix-error.json').read_text())
            self.assertEqual(record['native_error']['type'], 'ValueError')
            self.assertEqual(record['persistence_error']['type'], 'OSError')
            self.assertEqual(record['completed_rows'], 0)


if __name__ == '__main__':
    unittest.main()
