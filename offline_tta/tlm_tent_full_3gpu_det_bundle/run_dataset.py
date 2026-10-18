"""One fresh-model process for either a long-input preflight or a full dataset."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import sys
import traceback
from full_job import PausedRun, run_phases
from full_runtime import Runtime, configuration, protocol_id
from job_state import (Journal, CheckpointStore, atomic_json, capture_training,
                       restore_training, cpu_tree, require, claim_directory)
from offline_flow import unlabeled


def dataset_id(protocol, directory, alias):
    return hashlib.sha256((protocol + '|' + str(Path(directory).resolve()) + '|' + alias).encode()).hexdigest()


def assert_same_tree(left, right, path='root'):
    import torch
    if isinstance(left, torch.Tensor):
        require(isinstance(right, torch.Tensor) and left.shape == right.shape and left.dtype == right.dtype,
                'Restored tensor metadata differs at ' + path)
        if not torch.equal(left, right):
            changed = int(torch.count_nonzero(left != right))
            maximum = float((left.double() - right.double()).abs().max())
            raise ValueError('Restored tensor continuation differs at ' + path +
                             ': changed_elements=' + str(changed) + ', max_abs_difference=' + str(maximum))
    elif isinstance(left, dict):
        require(left.keys() == right.keys(), 'Restored state keys differ')
        for key in left:
            assert_same_tree(left[key], right[key], path + '.' + str(key))
    elif isinstance(left, (tuple, list)):
        require(len(left) == len(right), 'Restored state length differs')
        for index, (a, b) in enumerate(zip(left, right)):
            assert_same_tree(a, b, path + '[' + str(index) + ']')
    else:
        require(left == right, 'Restored scalar/RNG differs')


def rng_probe():
    import numpy as np
    import torch
    return {'python': random.random(), 'numpy': float(np.random.rand()), 'torch': torch.rand(3),
            'cuda': [torch.rand(3, device='cuda:' + str(i)).cpu() for i in range(torch.cuda.device_count())]}


def preflight(rt, output, job_id, event):
    import torch
    from full_core import validate_preflight_update
    from tent_text import make_optimizer, snapshot_adapters
    index = rt.cfg['preflight_indices'][rt.alias]
    row = unlabeled(rt.rows[index])
    before = rt.base_digest()
    require(before == rt.cfg['preflight_frozen_base_sha256'], 'Wrong frozen model fingerprint')
    _, _, source = rt.generate(row, rt.cfg['eval_max_new_tokens'])
    atomic_json(output / 'source_probe.json', {'dataset_source_index': index, **source})
    event('long_source_generation', prompt_tokens=source['used_prompt_tokens'], generated_tokens=source['generated_tokens'])
    rt.enable_training()
    journal = Journal(output / 'adaptation.jsonl', job_id)
    store = CheckpointStore(output / 'checkpoints', job_id)
    ids, prompt, prefix = rt.generate(row, rt.cfg['adapt_max_new_tokens'])
    first = rt.step(ids, prompt, measure_after=True)
    validate_preflight_update(first, rt.cfg['gpu_count'])
    journal.append({'source_index': 0, 'dataset_source_index': index, **first})
    committed = capture_training(rt.model, rt.optimizer, 1, journal.offset)
    store.commit(committed)
    draws_expected = rng_probe()
    sequence, length, _ = rt.generate(row, rt.cfg['adapt_max_new_tokens'])
    second = rt.step(sequence, length, measure_after=True)
    journal.append({'source_index': 1, 'dataset_source_index': index, **second})
    expected_adapter = snapshot_adapters(rt.model)
    expected_optimizer = cpu_tree(rt.optimizer.state_dict())
    # Discard a new optimizer's empty state, restore the committed step-one state,
    # and roll the uncommitted second log record back before repeating that step.
    rt.optimizer = make_optimizer(rt.model, rt.cfg['learning_rates'][rt.alias])
    saved = store.load()
    journal.truncate_to(saved['log_offset'], saved['next_index'])
    restore_training(rt.model, rt.optimizer, saved)
    assert_same_tree(committed['adapters'], snapshot_adapters(rt.model), 'restored_adapters')
    assert_same_tree(committed['optimizer'], cpu_tree(rt.optimizer.state_dict()), 'restored_Adam')
    event('checkpoint_restored_before_replay', adapter='EXACT_MATCH', Adam='EXACT_MATCH', next_index=saved['next_index'])
    assert_same_tree(draws_expected, rng_probe())
    resumed_sequence, resumed_length, _ = rt.generate(row, rt.cfg['adapt_max_new_tokens'])
    require(torch.equal(sequence, resumed_sequence) and length == resumed_length, 'Restored greedy prefix differs')
    replay = rt.step(resumed_sequence, resumed_length, measure_after=True)
    assert_same_tree(expected_adapter, snapshot_adapters(rt.model))
    assert_same_tree(expected_optimizer, cpu_tree(rt.optimizer.state_dict()))
    journal.append({'source_index': 1, 'dataset_source_index': index, **replay})
    store.commit(capture_training(rt.model, rt.optimizer, 2, journal.offset))
    after = rt.base_digest()
    require(before == after, 'Frozen base changed during long-input preflight')
    result = {'dataset': rt.alias, 'dataset_source_index': index, 'source_prompt_info': source,
              'first_update': first, 'second_update': replay, 'checkpoint_continuation': 'EXACT_MATCH',
              'rng_restore': 'EXACT_MATCH', 'frozen_base_unchanged': True, 'peak_memory_mib': rt.peak_memory(),
              'determinism': rt.determinism}
    atomic_json(output / 'preflight_report.json', result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['preflight', 'full'], required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    cfg, spec = configuration()
    protocol = protocol_id(cfg, spec)
    output = Path(args.output).resolve()
    root = Path(cfg['preflight_root'] if args.mode == 'preflight' else cfg['result_root']).resolve()
    require(output.is_relative_to(root), 'Output outside configured /data root')
    identifier = dataset_id(protocol, output, args.dataset)
    claim_directory(output, {'job_id': identifier, 'protocol_id': protocol,
                             'dataset': args.dataset, 'mode': args.mode})
    manifest_path = output / 'run_manifest.json'
    if args.resume:
        require(args.mode == 'full', 'Preflight does not resume')
        if manifest_path.exists():
            previous = json.loads(manifest_path.read_text())
            require(previous['job_id'] == identifier and previous['protocol_id'] == protocol, 'Cannot resume another protocol/job')
    else:
        require(not manifest_path.exists(), 'Existing dataset run requires explicit resume')
    manifest = {'job_id': identifier, 'protocol_id': protocol, 'complete': False, 'status': 'RUNNING',
                'mode': args.mode, 'dataset': args.dataset, 'pid': os.getpid(), 'config': cfg,
                'started_at': datetime.now(timezone.utc).isoformat(), 'resume': args.resume,
                'reference_answers_used_for_adaptation': False}
    stop = {'requested': False}

    def on_signal(signum, frame):
        stop['requested'] = True

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)

    def event(phase, **fields):
        payload = {'time': datetime.now(timezone.utc).isoformat(), 'dataset': args.dataset, 'phase': phase, **fields}
        line = json.dumps(payload, ensure_ascii=False)
        with (output / 'events.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(line + '\n')
        # All events are saved; console progress is reduced for a long job.
        if phase not in ('source', 'adapted', 'evaluated') or fields['completed'] % 10 == 0:
            print(line, flush=True)

    atomic_json(output / 'run_manifest.json', manifest)
    try:
        event('loading_fresh_model')
        runtime = Runtime(cfg, spec, args.dataset)
        manifest.update({'physical_gpus': runtime.physical, 'device_map': runtime.device_map, 'determinism': runtime.determinism})
        atomic_json(output / 'run_manifest.json', manifest)
        if args.mode == 'preflight':
            require(not stop['requested'], 'Preflight interrupted before validation')
            preflight(runtime, output, identifier, event)
        else:
            run_phases(runtime, output, identifier, lambda: stop['requested'], event)
        manifest.update({'complete': True, 'status': 'PASS', 'finished_at': datetime.now(timezone.utc).isoformat(),
                         'peak_memory_mib': runtime.peak_memory()})
        atomic_json(output / 'run_manifest.json', manifest)
        print(('TENT_LONG_DATASET=PASS ' if args.mode == 'preflight' else 'TENT_FULL_DATASET=PASS ') + args.dataset, flush=True)
    except PausedRun as exc:
        manifest.update({'complete': False, 'status': 'PAUSED', 'reason': str(exc)})
        atomic_json(output / 'run_manifest.json', manifest)
        print('TENT_DATASET_STOPPED_AT_SAFE_BOUNDARY=' + str(exc), flush=True)
        raise SystemExit(75)
    except Exception as exc:
        manifest.update({'complete': False, 'status': 'FAILED', 'error_type': type(exc).__name__, 'error': str(exc)})
        atomic_json(output / 'run_manifest.json', manifest)
        stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
        (output / ('failure_' + stamp + '.txt')).write_text(traceback.format_exc(), encoding='utf-8')
        print('TENT_DATASET=FAIL ' + args.dataset, flush=True)
        raise


if __name__ == '__main__':
    main()
