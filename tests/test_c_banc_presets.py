"""Named experiment admission and truthful labels after checkpoint restoration."""
import copy
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from flylab.c import PROTOCOL
from flylab.c.banc_presets import observed_target_preset, resolve_target_preset
from flylab.c.banc_walking import BancWalkingParameters
from flylab.c.server import CDispatcher


class BancPresetContracts(unittest.TestCase):
    def request(self, preset='m2-calibrated-v1'):
        return dict(preset=preset, seed=7, navigation=dict(target_xz_mm=[6, -3]))

    def test_named_model_is_resolved_by_actual_rpc_without_touching_main(self):
        dispatcher = CDispatcher.__new__(CDispatcher)
        dispatcher.engine = None
        dispatcher.graph = object()
        dispatcher.banc_walking = None
        dispatcher.load = Mock()
        dispatcher.banc_walking_result = lambda: {'observation': {}}
        fake = SimpleNamespace(close=Mock())
        with patch('flylab.c.banc_walking.BancWalkingSession', return_value=fake) as session:
            result = dispatcher.handle(dict(protocol=PROTOCOL, requestId=1,
                op='banc_walking_init', payload=self.request()))
        args = session.call_args.kwargs
        self.assertEqual(args['seed'], 7)
        self.assertEqual(args['navigation'], dict(target_xz_mm=[6., -3.], forward_drive=5000.))
        p = BancWalkingParameters(**args['parameters'])
        self.assertEqual(p.joint_gain_multipliers, (1., 1., 1.15, 1., 1., 1., 1.))
        self.assertEqual((p.adaptation_gain, p.adaptation_tau_s, p.joint_activation_tau_s), (8., .3, .09))
        self.assertTrue(p.fast_coxa_activation and p.coxa_geometry and p.ltm_tendons)
        self.assertIs(dispatcher.banc_walking, fake)
        self.assertIsNone(dispatcher.engine)
        self.assertIsNone(result['result']['source_checkpoint'])

    def test_invalid_or_mixed_preset_rejects_before_loading_or_replacing(self):
        bad = [self.request(x) for x in ('../unknown', '', None, [], 1)]
        bad += [dict(self.request(), parameters={}), dict(self.request(), navigation={}),
                dict(self.request(), navigation=dict(target_xz_mm=[6,-3], forward_drive=5000)),
                dict(self.request(), navigation=dict(target_xz_mm=[float('nan'),0]))]
        for payload in bad:
            d = CDispatcher.__new__(CDispatcher)
            d.engine = None
            d.banc_walking = object()
            original = d.banc_walking
            d.load = Mock(side_effect=AssertionError('Must reject before loading'))
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                d.handle(dict(protocol=PROTOCOL, requestId=1, op='banc_walking_init', payload=payload))
            self.assertIs(d.banc_walking, original)
            d.load.assert_not_called()

    def test_legacy_request_is_not_reinterpreted_and_cached_definitions_are_not_mutable(self):
        legacy = dict(parameters=dict(neural_dt_s=.0001), navigation=None)
        self.assertIs(resolve_target_preset(legacy), legacy)
        a = resolve_target_preset(self.request())
        a['parameters']['motor_gain'] = .123
        b = resolve_target_preset(self.request())
        self.assertEqual(b['parameters']['motor_gain'], .002)

    def test_current_name_comes_from_restored_parameters_and_navigation(self):
        for preset in ('m1-target-v1', 'm2-calibrated-v1'):
            request = resolve_target_preset(self.request(preset))
            observation = dict(parameters=request['parameters'], navigation={'target_xz_mm':[6,-3]},
                               descending_drive=5000.)
            saved = copy.deepcopy(observation)
            self.assertEqual(observed_target_preset(observation)['id'], preset)
            self.assertEqual(observation, saved)
            observation['parameters']['motor_gain'] *= 1.01
            self.assertIsNone(observed_target_preset(observation)['id'])
            observation = copy.deepcopy(saved)
            observation['descending_drive'] = 0.
            self.assertIsNone(observed_target_preset(observation)['id'])
        old = copy.deepcopy(saved)
        old['parameters']['neural_dt_s'] = .0001
        self.assertIsNone(observed_target_preset(old)['id'])
        self.assertEqual(old['parameters']['neural_dt_s'], .0001)
        self.assertIsNone(observed_target_preset({'time_s': .01})['id'])


if __name__ == '__main__':
    unittest.main()
