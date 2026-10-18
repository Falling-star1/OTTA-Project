"""Strict deterministic kernels; set the cuBLAS environment in the launcher."""
import os
import torch
from job_state import require

REQUESTED = {'algorithms': True, 'warn_only': False, 'cublas_workspace': ':4096:8'}


def configure(expected=None):
    require(expected is None or expected == REQUESTED, 'Unsupported deterministic configuration')
    require(os.environ.get('CUBLAS_WORKSPACE_CONFIG') == REQUESTED['cublas_workspace'],
            'Set CUBLAS_WORKSPACE_CONFIG=:4096:8 before launching Python')
    torch.use_deterministic_algorithms(True, warn_only=False)
    return {'algorithms': torch.are_deterministic_algorithms_enabled(),
            'warn_only': torch.is_deterministic_algorithms_warn_only_enabled(),
            'cublas_workspace': os.environ['CUBLAS_WORKSPACE_CONFIG']}
