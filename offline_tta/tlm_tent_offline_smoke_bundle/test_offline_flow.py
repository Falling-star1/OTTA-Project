"""Offline order, label isolation and matching Source comparisons, CPU only."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

try:
    import offline_flow as flow
except ModuleNotFoundError as exc:
    if exc.name != 'offline_flow':
        raise
    flow = None


class ImplementationTest(unittest.TestCase):
    def test_workflow_exists(self):
        self.assertIsNotNone(flow, 'Offline workflow has not been implemented yet')


@unittest.skipIf(flow is None, 'waiting for implementation')
class OfflineOrderTest(unittest.TestCase):
    def test_all_updates_precede_evaluation_and_adapt_never_sees_labels(self):
        events = []
        rows = [{'instruction': 'q' + str(i), 'input': '', 'output': 'SECRET' + str(i)} for i in range(3)]

        def adapt(index, unlabeled):
            self.assertEqual(set(unlabeled), {'instruction', 'input'})
            self.assertNotIn('SECRET', str(unlabeled))
            events.append(('adapt', index))
            return {'optimizer_steps': 1}

        def predict(index, unlabeled):
            self.assertEqual(set(unlabeled), {'instruction', 'input'})
            self.assertEqual(sum(kind == 'adapt' for kind, _ in events), len(rows))
            events.append(('predict', index))
            return {'predict': 'answer' + str(index)}

        def score(prediction, label):
            events.append(('score', label))
            return {'score': 1., 'metric': 'exact_match', 'extracted_answer': prediction}

        adapted, evaluated = flow.run_offline(rows, adapt, predict, score)
        self.assertEqual(events[:3], [('adapt', 0), ('adapt', 1), ('adapt', 2)])
        self.assertEqual(len(adapted), 3)
        self.assertEqual([r['source_index'] for r in evaluated], [0, 1, 2])
        self.assertEqual([r['label'] for r in evaluated], [r['output'] for r in rows])

    def test_update_failure_prevents_any_evaluation(self):
        events = []

        def adapt(index, unlabeled):
            if index == 1:
                raise FloatingPointError('bad update')
            return {'optimizer_steps': 1}

        def predict(*args):
            events.append('predict')

        with self.assertRaises(FloatingPointError):
            flow.run_offline([{'instruction': 'q', 'input': '', 'output': 'a'}] * 2,
                             adapt, predict, lambda *_: {'score': 0})
        self.assertEqual(events, [])

    def test_missing_update_cannot_be_reported_as_success(self):
        with self.assertRaises(ValueError):
            flow.run_offline([{'instruction': 'q', 'input': '', 'output': 'a'}],
                             lambda *_: {'optimizer_steps': 0}, lambda *_: {}, lambda *_: {})

    def test_empty_dataset_is_rejected(self):
        with self.assertRaises(ValueError):
            flow.run_offline([], lambda *_: {}, lambda *_: {}, lambda *_: {})

    def test_reference_access_is_forbidden_until_offline_boundary(self):
        phase = {'evaluation': False}
        events = []

        class PhaseGuardedRow(dict):
            def __getitem__(self, key):
                if key == 'output' and not phase['evaluation']:
                    raise AssertionError('Reference read during adaptation')
                return super().__getitem__(key)

        rows = [PhaseGuardedRow(instruction='q', input='', output='answer') for _ in range(2)]

        def adapt(index, record):
            events.append('adapt')
            return {'optimizer_steps': 1}

        def boundary():
            self.assertEqual(events, ['adapt', 'adapt'])
            events.append('boundary')
            phase['evaluation'] = True

        def predict(index, record):
            self.assertTrue(phase['evaluation'])
            return {'predict': 'answer'}

        flow.run_offline(rows, adapt, predict, lambda *_: {'score': 1.}, before_evaluation=boundary)
        self.assertEqual(events, ['adapt', 'adapt', 'boundary'])

    def test_failed_boundary_prevents_prediction(self):
        def boundary():
            raise RuntimeError('frozen model hash changed')

        def predict(*args):
            self.fail('Prediction should not start after failed freeze check')

        with self.assertRaises(RuntimeError):
            flow.run_offline([{'instruction': 'q', 'input': '', 'output': 'a'}],
                             lambda *_: {'optimizer_steps': 1}, predict, lambda *_: {},
                             before_evaluation=boundary)


@unittest.skipIf(flow is None, 'waiting for implementation')
class BaselineTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.expected = {'model': '/model', 'source_commit': 'fixed', 'max_input_tokens': 1024,
                         'eval_max_new_tokens': 512, 'seed': 42, 'samples_per_dataset': 2,
                         'versions': {'torch': '2.5.1+cu118'}}
        self.spec = {'datasets': [{'alias': 'tiny', 'metric': 'exact_match'}]}
        self.rows = {'tiny': [{'instruction': 'q' + str(i), 'input': '', 'output': str(i)} for i in range(2)]}
        self.manifest = {'complete': True, 'method': 'Source; all parameters frozen',
                         'model': '/model', 'source_commit': 'fixed', 'max_input_tokens': 1024,
                         'max_new_tokens': 512, 'seed': 42, 'dtype': 'float16',
                         'decoding': 'greedy', 'repetition_penalty': 1., 'batch_size': 1,
                         'prompt_template': 'TLM qwen', 'source_indices_per_dataset': [0, 1],
                         'torch': '2.5.1+cu118'}
        self.predictions = [{'source_index': i, 'prompt': 'q' + str(i), 'predict': str(i), 'label': str(i),
                             'original_prompt_sha256': hashlib.sha256(('q' + str(i)).encode()).hexdigest(),
                             'input_truncated': False, 'hit_generation_limit': False, 'metric': 'exact_match',
                             'generated_tokens': 3, 'score': 1.} for i in range(2)]
        self.metrics = [{'dataset': 'tiny', 'n': 2, 'metric': 'exact_match', 'value_0_to_1': 1.,
                         'truncated_inputs': 0, 'generation_limit_hits': 0, 'empty_predictions': 0}]
        self.write_files()

    def tearDown(self):
        self.temp.cleanup()

    def write_files(self):
        (self.root / 'run_manifest.json').write_text(json.dumps(self.manifest), encoding='utf-8')
        (self.root / 'metrics.json').write_text(json.dumps(self.metrics), encoding='utf-8')
        (self.root / 'tiny.jsonl').write_text('\n'.join(json.dumps(r) for r in self.predictions), encoding='utf-8')

    def validate(self, hashes=None):
        return flow.validate_source(self.root, self.expected, self.spec, self.rows, hashes)

    def test_valid_source_is_accepted_and_delta_does_not_require_improvement(self):
        source = self.validate()
        result = flow.comparison_row(source['tiny']['summary'],
                                     {**self.metrics[0], 'value_0_to_1': .5})
        self.assertEqual(result['delta_tent_minus_source'], -.5)

    def test_wrong_output_budget_or_incomplete_manifest_is_rejected(self):
        for key, value in [('max_new_tokens', 128), ('complete', False)]:
            original = self.manifest[key]
            self.manifest[key] = value
            self.write_files()
            with self.assertRaises(ValueError):
                self.validate()
            self.manifest[key] = original

    def test_index_prompt_fingerprint_label_and_metric_must_match(self):
        for key, wrong in [('source_index', 99), ('prompt', 'different'),
                           ('original_prompt_sha256', 'bad'), ('label', 'different'), ('metric', 'rougeLsum')]:
            original = self.predictions[0][key]
            self.predictions[0][key] = wrong
            self.write_files()
            with self.assertRaises(ValueError):
                self.validate()
            self.predictions[0][key] = original

    def test_aggregate_metric_must_match_per_row_scores(self):
        self.metrics[0]['value_0_to_1'] = .25
        self.write_files()
        with self.assertRaises(ValueError):
            self.validate()

    def test_changed_source_bytes_are_rejected(self):
        hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in self.root.iterdir()}
        self.validate(hashes)
        (self.root / 'tiny.jsonl').write_text('changed', encoding='utf-8')
        with self.assertRaises(ValueError):
            self.validate(hashes)


if __name__ == '__main__':
    unittest.main(verbosity=2)
