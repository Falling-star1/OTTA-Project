import hashlib
from pathlib import Path
import tempfile
import unittest
import torch
from test_full_core import tiny_model
from tent_text import frozen_digest, make_optimizer, snapshot_adapters
from full_core import entropy_step

try:
    import full_job as job
except ModuleNotFoundError as exc:
    if exc.name != 'full_job':
        raise
    job = None


class FakeRuntime:
    def __init__(self, crash_after=None):
        self.model = tiny_model()
        self.optimizer = make_optimizer(self.model, 1e-3)
        self.rows = [{'instruction': 'q' + str(i), 'input': '', 'output': 'answer'} for i in range(3)]
        self.cfg = {'samples_per_dataset': 3, 'checkpoint_interval': 2,
                    'adapt_max_new_tokens': 80, 'eval_max_new_tokens': 512,
                    'preflight_frozen_base_sha256': frozen_digest(self.model)}
        self.alias = 'tiny'
        self.metric = 'exact_match'
        self.history = []
        self.phase = 'source'
        self.steps = 0
        self.crash_after = crash_after

    def generate(self, record, limit):
        assert set(record) == {'instruction', 'input'}
        if self.phase == 'adapt' and self.crash_after is not None and self.steps == self.crash_after:
            raise RuntimeError('simulated hard interruption before next sample')
        self.history.append((self.phase, record['instruction']))
        ids = torch.tensor([[1, 2, 3, 4, 5]])
        return ids, 3, {'predict': 'answer', 'prompt': record['instruction'],
                         'original_prompt_sha256': hashlib.sha256(record['instruction'].encode()).hexdigest(),
                         'used_prompt_tokens': 3, 'input_truncated': False,
                         'generated_tokens': 2, 'hit_generation_limit': False}

    def score(self, prediction, label):
        return {'score': float(prediction == label), 'metric': self.metric, 'extracted_answer': prediction}

    def enable_training(self):
        self.phase = 'adapt'
        self.model.train()

    def enable_evaluation(self):
        self.phase = 'tent'
        self.model.eval()

    def step(self, ids, prompt):
        self.model.train()
        report = entropy_step(self.model, self.optimizer, ids, prompt, [], None)
        self.steps += 1
        return report

    def save_adapter(self, directory):
        Path(directory).mkdir(exist_ok=True)

    def base_digest(self):
        return frozen_digest(self.model)


class ImplementationTest(unittest.TestCase):
    def test_full_phase_runner_exists(self):
        self.assertIsNotNone(job, 'Full phase runner is not implemented')


@unittest.skipIf(job is None, 'waiting for phase runner')
class FullJobTest(unittest.TestCase):
    def test_wrong_base_cannot_generate_or_reuse_source_predictions(self):
        with tempfile.TemporaryDirectory() as folder:
            runtime = FakeRuntime()
            runtime.cfg['preflight_frozen_base_sha256'] = 'wrong-model'
            with self.assertRaises(ValueError):
                job.run_phases(runtime, Path(folder), 'job', lambda: False, lambda *a, **k: None)
            self.assertEqual(runtime.history, [], 'Source was generated before checking actual weights')

    def test_graceful_pause_resumes_without_repeating_committed_updates(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            expected = FakeRuntime()
            job.run_phases(expected, root / 'expected', 'expected', lambda: False, lambda *a, **k: None)
            first = FakeRuntime()
            with self.assertRaises(job.PausedRun):
                job.run_phases(first, root / 'resumed', 'resumed', lambda: first.steps == 2, lambda *a, **k: None)
            fresh = FakeRuntime()
            result = job.run_phases(fresh, root / 'resumed', 'resumed', lambda: False, lambda *a, **k: None)
            self.assertEqual([q for phase, q in first.history + fresh.history if phase == 'adapt'], ['q0', 'q1', 'q2'])
            self.assertEqual(result['adaptation']['processed'], 3)
            for name, value in snapshot_adapters(fresh.model).items():
                self.assertTrue(torch.equal(value, snapshot_adapters(expected.model)[name]))

    def test_crash_rolls_back_uncommitted_update_and_final_state_matches(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            expected = FakeRuntime()
            job.run_phases(expected, root / 'expected', 'expected', lambda: False, lambda *a, **k: None)
            interrupted = FakeRuntime(crash_after=1)
            with self.assertRaises(RuntimeError):
                job.run_phases(interrupted, root / 'actual', 'actual', lambda: False, lambda *a, **k: None)
            self.assertFalse(any(phase == 'tent' for phase, _ in interrupted.history))
            fresh = FakeRuntime()
            result = job.run_phases(fresh, root / 'actual', 'actual', lambda: False, lambda *a, **k: None)
            self.assertEqual([q for phase, q in fresh.history if phase == 'adapt'], ['q0', 'q1', 'q2'])
            self.assertEqual(result['source']['n'], 3)
            self.assertEqual(result['tent']['n'], 3)
            for name, value in snapshot_adapters(fresh.model).items():
                self.assertTrue(torch.equal(value, snapshot_adapters(expected.model)[name]))


if __name__ == '__main__':
    unittest.main(verbosity=2)
