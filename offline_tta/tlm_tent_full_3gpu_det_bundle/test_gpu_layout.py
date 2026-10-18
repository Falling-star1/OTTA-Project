import unittest
from unittest.mock import patch
import torch
import full_core
import run_dataset

try:
    import gpu_layout
except ModuleNotFoundError:
    gpu_layout = None


class LayoutTest(unittest.TestCase):
    def test_three_physical_devices_are_required_and_distinct(self):
        self.assertIsNotNone(gpu_layout, 'Three-GPU selection validation is missing')
        self.assertEqual(gpu_layout.physical_devices('2,4,7', 3), [2, 4, 7])
        for text in ['0,1', '0,1,1', '0,00,1', '0,1,2,3', '0,x,2', '0,1,-2']:
            with self.assertRaises(ValueError):
                gpu_layout.physical_devices(text, 3)

    def test_preflight_requires_positive_gradient_on_every_selected_device(self):
        report = {'changed_elements': 5, 'gradient_l1_by_device': {'cuda:0': 1., 'cuda:1': 2., 'cuda:2': 3.}}
        full_core.validate_preflight_update(report, 3)
        for gradients in [{'cuda:0': 1., 'cuda:1': 2.}, {'cuda:0': 1., 'cuda:1': 2., 'cuda:2': 0.},
                          {'cuda:0': 1., 'cuda:1': 2., 'cuda:2': float('nan')}]:
            with self.assertRaises(ValueError):
                full_core.validate_preflight_update({**report, 'gradient_l1_by_device': gradients}, 3)

    def test_rng_continuation_probe_includes_third_device(self):
        with patch.object(torch.cuda, 'device_count', return_value=3), \
             patch.object(torch, 'rand', return_value=torch.tensor([1., 2., 3.])) as rand:
            report = run_dataset.rng_probe()
        requested = [call.kwargs.get('device') for call in rand.call_args_list]
        self.assertEqual(requested, [None, 'cuda:0', 'cuda:1', 'cuda:2'])
        self.assertEqual(len(report['cuda']), 3)


if __name__ == '__main__':
    unittest.main(verbosity=2)
