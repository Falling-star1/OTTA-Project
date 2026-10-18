"""Testable Source -> all adaptation -> final evaluation with exact commit cursors."""
import hashlib
import math
from pathlib import Path
from job_state import Journal, CheckpointStore, atomic_json, capture_training, restore_training, require, sha256
from offline_flow import summarize, text_prompt, unlabeled


class PausedRun(Exception):
    pass


def validate_predictions(journal, rows):
    require(len(journal.records) <= len(rows), 'Too many stored predictions')
    for index, record in enumerate(journal.records):
        prompt_sha = hashlib.sha256(text_prompt(rows[index]).encode('utf-8')).hexdigest()
        require(record['original_prompt_sha256'] == prompt_sha, 'Stored prediction prompt changed')
        require(record['label'] == rows[index]['output'], 'Stored reference does not match fixed data')
        require(math.isfinite(record['score']) and 0 <= record['score'] <= 1, 'Invalid stored score')


def run_phases(runtime, output, job_id, stop_requested, event):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    cfg, rows = runtime.cfg, runtime.rows
    n = cfg['samples_per_dataset']
    require(len(rows) == n, 'Dataset size differs from frozen run configuration')
    before = runtime.base_digest()
    require(before == cfg['preflight_frozen_base_sha256'], 'Base differs from the verified model')
    source = Journal(output / 'source.jsonl', job_id + ':source')
    adaptation = Journal(output / 'adaptation.jsonl', job_id + ':adapt')
    evaluation = Journal(output / 'tent.jsonl', job_id + ':tent')
    store = CheckpointStore(output / 'checkpoints', job_id + ':adapt')
    if store.pointer.exists() or adaptation.records or evaluation.records:
        require(len(source.records) == n, 'Adaptation/evaluation exists before Source completion')
    validate_predictions(source, rows)
    validate_predictions(evaluation, rows)
    for index in range(len(source.records), n):
        if stop_requested():
            raise PausedRun('Paused in Source at ' + str(index))
        _, _, prediction = runtime.generate(unlabeled(rows[index]), cfg['eval_max_new_tokens'])
        label = rows[index]['output']
        source.append({'source_index': index, **prediction, 'label': label, **runtime.score(prediction['predict'], label)})
        event('source', completed=index + 1, total=n)
    source_metrics = summarize(runtime.alias, runtime.metric, source.records)
    atomic_json(output / 'source_metrics.json', source_metrics)
    event('source_complete', **source_metrics)
    runtime.enable_training()
    if store.pointer.exists():
        payload = store.load()
        adaptation.truncate_to(payload['log_offset'], payload['next_index'])
        restore_training(runtime.model, runtime.optimizer, payload)
        cursor = payload['next_index']
        event('training_restored', next_index=cursor)
    else:
        require(not adaptation.records, 'Adaptation log has no committed initial checkpoint')
        cursor = 0
        store.commit(capture_training(runtime.model, runtime.optimizer, 0, 0))
    require(0 <= cursor <= n, 'Training cursor outside dataset')
    require(not evaluation.records or cursor == n, 'Evaluation exists before all updates committed')
    resumed_from = cursor
    for index in range(cursor, n):
        if stop_requested():
            store.commit(capture_training(runtime.model, runtime.optimizer, index, adaptation.offset))
            raise PausedRun('Paused before adaptation input ' + str(index))
        sequence, prompt_length, prefix = runtime.generate(unlabeled(rows[index]), cfg['adapt_max_new_tokens'])
        report = runtime.step(sequence, prompt_length)
        require(report['optimizer_steps'] == 1, 'Input did not execute one Adam step')
        adaptation.append({'source_index': index, 'prompt_sha256': prefix['original_prompt_sha256'],
                           'generated_prefix': prefix['predict'], 'prefix_tokens': prefix['generated_tokens'],
                           'input_truncated': prefix['input_truncated'], **report})
        cursor = index + 1
        event('adapted', completed=cursor, total=n, entropy=report['entropy_before'], zero_update=report['zero_update'])
        if cursor % cfg['checkpoint_interval'] == 0 or cursor == n or stop_requested():
            store.commit(capture_training(runtime.model, runtime.optimizer, cursor, adaptation.offset))
            event('training_committed', next_index=cursor)
        if stop_requested():
            raise PausedRun('Paused after committed update ' + str(cursor))
    require(store.load()['next_index'] == n and len(adaptation.records) == n, 'All training inputs must be committed before evaluation')
    after = runtime.base_digest()
    require(before == after, 'Frozen base changed during adaptation')
    runtime.optimizer.zero_grad(set_to_none=True)
    runtime.enable_evaluation()
    runtime.save_adapter(output / 'adapter')
    event('all_updates_committed_starting_evaluation', optimizer_updates=n)
    for index in range(len(evaluation.records), n):
        if stop_requested():
            raise PausedRun('Paused in final evaluation at ' + str(index))
        _, _, prediction = runtime.generate(unlabeled(rows[index]), cfg['eval_max_new_tokens'])
        label = rows[index]['output']
        evaluation.append({'source_index': index, **prediction, 'label': label, **runtime.score(prediction['predict'], label)})
        event('evaluated', completed=index + 1, total=n)
    tent_metrics = summarize(runtime.alias, runtime.metric, evaluation.records)
    result = {'source': source_metrics, 'tent': tent_metrics,
              'adaptation': {'processed': n, 'optimizer_steps': sum(r['optimizer_steps'] for r in adaptation.records),
                             'zero_updates': sum(r['zero_update'] for r in adaptation.records), 'resumed_from': resumed_from},
              'frozen_base_sha256_before': before, 'frozen_base_sha256_after': after,
              'frozen_base_unchanged': True, 'source_sha256': sha256(source.path), 'tent_sha256': sha256(evaluation.path)}
    atomic_json(output / 'metrics.json', result)
    return result
