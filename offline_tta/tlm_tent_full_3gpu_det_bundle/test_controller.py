"""Controller bootstrap recovery without loading a model or accessing CUDA."""
import json
import io
from contextlib import redirect_stdout
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import run_tent_full as controller
import run_dataset as worker
from job_state import claim_directory, atomic_json, sha256


class ControllerTest(unittest.TestCase):
    def test_missing_resume_directory_is_rejected_without_creating_it(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / 'results' / 'typo'
            cfg = {'result_root': str(Path(folder) / 'results')}
            with patch.object(controller, 'configuration', return_value=(cfg, {})), \
                 patch.object(controller, 'check_environment'), \
                 patch.object(controller, 'protocol_id', return_value='p'), \
                 patch('sys.argv', ['run_tent_full.py', 'resume', '--output', str(output)]):
                with self.assertRaisesRegex(ValueError, 'existing run directory'):
                    controller.main()
            self.assertFalse(output.exists())

    def test_resume_identity_only_root_and_child_after_bootstrap_interruption(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            output = root / 'results' / 'old-run'
            child = output / 'tiny'
            preflight = root / 'preflight'
            preflight.mkdir()
            report = preflight / 'report.json'
            report.write_text('{}')
            cfg = {'result_root': str(root / 'results'), 'preflight_root': str(preflight),
                   'receipt': str(root / 'receipt.json'), 'samples_per_dataset': 3}
            spec = {'datasets': [{'alias': 'tiny'}]}
            atomic_json(cfg['receipt'], {'protocol_id': 'p', 'status': 'PASS',
                        'reports': {'tiny': {'path': str(report), 'sha256': sha256(report)}}})
            claim_directory(output, {'job_id': worker.dataset_id('p', output, 'full'), 'protocol_id': 'p', 'kind': 'full'})
            identifier = worker.dataset_id('p', child, 'tiny')
            claim_directory(child, {'job_id': identifier, 'protocol_id': 'p', 'dataset': 'tiny', 'mode': 'full'})
            commands = []

            class Process:
                def __init__(self, command):
                    commands.append(command)
                    for name in ['source.jsonl', 'tent.jsonl']:
                        (child / name).write_text('three simulated rows\n')
                    summary = {'dataset': 'tiny', 'n': 3, 'metric': 'exact_match',
                               'value_0_to_1': 1., 'generation_limit_hits': 0, 'empty_predictions': 0}
                    atomic_json(child / 'metrics.json', {'source': summary, 'tent': summary,
                                'adaptation': {'processed': 3, 'optimizer_steps': 3}, 'frozen_base_unchanged': True,
                                'source_sha256': sha256(child / 'source.jsonl'), 'tent_sha256': sha256(child / 'tent.jsonl')})
                    atomic_json(child / 'run_manifest.json', {'job_id': identifier, 'protocol_id': 'p',
                                'complete': True, 'status': 'PASS'})

                def wait(self):
                    return 0

            with patch.object(controller, 'configuration', return_value=(cfg, spec)), \
                 patch.object(controller, 'check_environment'), \
                 patch.object(controller, 'protocol_id', return_value='p'), \
                 patch.object(controller.signal, 'signal'), \
                 patch.object(controller.subprocess, 'Popen', side_effect=Process), \
                 patch('sys.argv', ['run_tent_full.py', 'resume', '--output', str(output)]), \
                 redirect_stdout(io.StringIO()):
                controller.main()
            self.assertIn('--resume', commands[0])
            self.assertTrue(json.loads((output / 'run_manifest.json').read_text())['complete'])

    def test_child_recovers_identity_only_directory_and_marks_pause(self):
        from full_job import PausedRun
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            output = root / 'run' / 'tiny'
            cfg = {'result_root': str(root)}
            identifier = worker.dataset_id('p', output, 'tiny')
            claim_directory(output, {'job_id': identifier, 'protocol_id': 'p', 'dataset': 'tiny', 'mode': 'full'})
            with patch.object(worker, 'configuration', return_value=(cfg, {})), \
                 patch.object(worker, 'protocol_id', return_value='p'), \
                 patch.object(worker.signal, 'signal'), \
                 patch.object(worker, 'Runtime') as runtime, \
                 patch.object(worker, 'run_phases', side_effect=PausedRun('test pause')), \
                 patch('sys.argv', ['run_dataset.py', '--mode', 'full', '--dataset', 'tiny', '--output', str(output), '--resume']), \
                 redirect_stdout(io.StringIO()):
                runtime.return_value.physical = [0, 1]
                runtime.return_value.determinism = {'algorithms': True, 'warn_only': False, 'cublas_workspace': ':4096:8'}
                runtime.return_value.device_map = {'a': '0', 'b': '1'}
                with self.assertRaises(SystemExit) as stopped:
                    worker.main()
            self.assertEqual(stopped.exception.code, 75)
            self.assertEqual(json.loads((output / 'run_manifest.json').read_text())['status'], 'PAUSED')


if __name__ == '__main__':
    unittest.main(verbosity=2)
