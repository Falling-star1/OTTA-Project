import os
import unittest
from unittest.mock import patch
import torch
import run_dataset

try:
    import reproducibility
except ModuleNotFoundError:
    reproducibility = None


class DeterminismTest(unittest.TestCase):
    def test_strict_mode_requires_cublas_workspace_before_cuda_use(self):
        self.assertIsNotNone(reproducibility, 'Deterministic computation setup is missing')
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, 'CUBLAS_WORKSPACE_CONFIG'):
                reproducibility.configure()

    def test_configure_enables_strict_algorithms_without_changing_seed(self):
        self.assertIsNotNone(reproducibility, 'Deterministic computation setup is missing')
        old = torch.are_deterministic_algorithms_enabled()
        warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
        rng = torch.get_rng_state().clone()
        try:
            with patch.dict(os.environ, {'CUBLAS_WORKSPACE_CONFIG': ':4096:8'}):
                effective = reproducibility.configure()
            self.assertTrue(torch.are_deterministic_algorithms_enabled())
            self.assertFalse(torch.is_deterministic_algorithms_warn_only_enabled())
            self.assertTrue(torch.equal(torch.get_rng_state(), rng))
            self.assertEqual(effective, {'algorithms': True, 'warn_only': False, 'cublas_workspace': ':4096:8'})
        finally:
            torch.use_deterministic_algorithms(old, warn_only=warn_only)

    def test_tiny_parameter_mismatch_is_still_rejected_and_reports_path(self):
        a = torch.tensor([0.001, 0.002], dtype=torch.float32)
        b = a.clone()
        b[0] = torch.nextafter(b[0], torch.tensor(float('inf')))
        with self.assertRaisesRegex(ValueError, r'root\.layer.*changed_elements=1'):
            run_dataset.assert_same_tree({'layer': a}, {'layer': b})


if __name__ == '__main__':
    unittest.main(verbosity=2)
