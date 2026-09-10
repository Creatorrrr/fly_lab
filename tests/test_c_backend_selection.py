import unittest
from unittest.mock import patch
from flylab.c.backend_selection import resolve_backend


class BackendSelectionTests(unittest.TestCase):
    def test_auto_uses_cuda_then_mps_then_cpu(self):
        states={'exp_lif_cpu_reference':{'available':True},'exp_lif_cuda':{'available':True},'exp_lif_mps':{'available':True}}
        with patch('flylab.c.backend_selection.backend_availability',return_value=states):
            self.assertEqual(resolve_backend(),'exp_lif_cuda')
            states['exp_lif_cuda']['available']=False
            self.assertEqual(resolve_backend(),'exp_lif_mps')
            states['exp_lif_mps']['available']=False
            self.assertEqual(resolve_backend(),'exp_lif_cpu_reference')

    def test_explicit_backend_is_preserved_without_auto_fallback(self):
        with patch('flylab.c.backend_selection.backend_availability',side_effect=AssertionError('Unexpected probe')):
            self.assertEqual(resolve_backend('exp_lif_cpu_reference'),'exp_lif_cpu_reference')
            self.assertEqual(resolve_backend('exp_lif_cuda'),'exp_lif_cuda')
            with self.assertRaises(ValueError):resolve_backend('unknown')
