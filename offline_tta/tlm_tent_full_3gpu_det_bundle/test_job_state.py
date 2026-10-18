import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch

try:
    import job_state as state
except ModuleNotFoundError as exc:
    if exc.name != 'job_state':
        raise
    state = None


class ImplementationTest(unittest.TestCase):
    def test_storage_exists(self):
        self.assertIsNotNone(state, 'Checkpoint storage is not implemented')


@unittest.skipIf(state is None, 'waiting for state implementation')
class StorageTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_uninitialized_directory_can_be_claimed_after_bootstrap_crash(self):
        self.assertTrue(hasattr(state, 'claim_directory'), 'Durable directory identity is missing')
        directory = self.root / 'bootstrap'
        directory.mkdir()
        identity = {'job_id': 'job', 'protocol_id': 'protocol'}
        state.claim_directory(directory, identity)
        self.assertEqual(json.loads((directory / 'identity.json').read_text()), identity)
        state.claim_directory(directory, identity)
        with self.assertRaises(ValueError):
            state.claim_directory(directory, {'job_id': 'other', 'protocol_id': 'protocol'})

    def test_unknown_files_in_unclaimed_directory_are_preserved(self):
        self.assertTrue(hasattr(state, 'claim_directory'), 'Durable directory identity is missing')
        directory = self.root / 'foreign'
        directory.mkdir()
        (directory / 'user.txt').write_text('preserve')
        with self.assertRaises(ValueError):
            state.claim_directory(directory, {'job_id': 'job'})
        self.assertEqual((directory / 'user.txt').read_text(), 'preserve')

    def test_journal_repairs_only_partial_tail_and_rejects_duplicates(self):
        path = self.root / 'records.jsonl'
        journal = state.Journal(path, 'job')
        journal.append({'source_index': 0, 'predict': 'a'})
        good_size = journal.offset
        with path.open('ab') as stream:
            stream.write(b'{"source_index":1')
        recovered = state.Journal(path, 'job')
        self.assertEqual(recovered.offset, good_size)
        self.assertEqual(len(recovered.records), 1)
        with self.assertRaises(ValueError):
            recovered.append({'source_index': 0})
        with self.assertRaises(ValueError):
            state.Journal(path, 'foreign')

    def test_journal_rejects_middle_corruption_without_truncating_it(self):
        path = self.root / 'bad.jsonl'
        path.write_bytes(b'not json\n{"source_index":1}\n')
        before = path.read_bytes()
        with self.assertRaises(ValueError):
            state.Journal(path, 'job')
        self.assertEqual(path.read_bytes(), before)

    def test_commit_pointer_survives_crash_and_rolls_back_uncommitted_log(self):
        journal = state.Journal(self.root / 'adapt.jsonl', 'job')
        store = state.CheckpointStore(self.root / 'checkpoint', 'job')
        journal.append({'source_index': 0})
        store.commit({'next_index': 1, 'log_offset': journal.offset, 'value': torch.tensor([1.])})
        journal.append({'source_index': 1})
        with patch.object(state, 'atomic_json', side_effect=OSError('simulated pointer crash')):
            with self.assertRaises(OSError):
                store.commit({'next_index': 2, 'log_offset': journal.offset, 'value': torch.tensor([2.])})
        committed = store.load()
        self.assertEqual(committed['next_index'], 1)
        self.assertEqual(committed['value'].item(), 1.)
        journal.truncate_to(committed['log_offset'], committed['next_index'])
        self.assertEqual(len(journal.records), 1)
        journal.append({'source_index': 1})
        store.commit({'next_index': 2, 'log_offset': journal.offset, 'value': torch.tensor([2.])})
        self.assertEqual(store.load()['next_index'], 2)

    def test_foreign_job_or_changed_checkpoint_is_rejected(self):
        store = state.CheckpointStore(self.root, 'job')
        store.commit({'next_index': 0, 'log_offset': 0, 'x': torch.tensor([1])})
        with self.assertRaises(ValueError):
            state.CheckpointStore(self.root, 'other').load()
        pointer = json.loads((self.root / 'commit.json').read_text())
        (self.root / pointer['filename']).write_bytes(b'changed')
        with self.assertRaises(ValueError):
            store.load()

    def test_model_optimizer_and_rng_continue_exactly_after_restore(self):
        from test_full_core import tiny_model
        from tent_text import make_optimizer, snapshot_adapters
        from full_core import entropy_step
        model = tiny_model()
        opt = make_optimizer(model, 1e-3)
        ids = torch.tensor([[1, 2, 3, 4, 5]])
        entropy_step(model, opt, ids, 3, [], None)
        saved = state.capture_training(model, opt, 1, 0)
        store = state.CheckpointStore(self.root, 'job')
        store.commit(saved)
        rng_expected = (random.random(), np.random.rand(), torch.rand(3))
        entropy_step(model, opt, ids, 3, [], None)
        expected = snapshot_adapters(model)
        expected_opt = opt.state_dict()
        fresh = tiny_model()
        fresh_opt = make_optimizer(fresh, 1e-3)
        state.restore_training(fresh, fresh_opt, store.load())
        actual_rng = (random.random(), np.random.rand(), torch.rand(3))
        self.assertEqual(rng_expected[:2], actual_rng[:2])
        self.assertTrue(torch.equal(rng_expected[2], actual_rng[2]))
        entropy_step(fresh, fresh_opt, ids, 3, [], None)
        for name, value in snapshot_adapters(fresh).items():
            self.assertTrue(torch.equal(value, expected[name]), name)
        actual_opt = fresh_opt.state_dict()
        for key, values in expected_opt['state'].items():
            for name, value in values.items():
                self.assertTrue(torch.equal(value, actual_opt['state'][key][name]))


if __name__ == '__main__':
    unittest.main(verbosity=2)
