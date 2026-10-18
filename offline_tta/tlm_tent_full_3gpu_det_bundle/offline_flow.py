"""Offline ordering and fixed Source comparison. No torch or training labels API."""
import hashlib
import json
import math
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def unlabeled(record):
    return {'instruction': record['instruction'], 'input': record['input']}


def text_prompt(record):
    return record['instruction'] + ('\n' + record['input'] if record['input'] else '')


def run_offline(rows, adapt, predict, score, on_prediction=None, before_evaluation=None):
    """Complete every update first; references are accessed only after prediction."""
    require(bool(rows), 'Empty offline dataset')
    updates = []
    for index, row in enumerate(rows):
        update = adapt(index, unlabeled(row))
        require(update.get('optimizer_steps') == 1, 'Every input must perform exactly one update')
        updates.append({'source_index': index, **update})
    if before_evaluation is not None:
        before_evaluation()
    evaluated = []
    for index, row in enumerate(rows):
        prediction = predict(index, unlabeled(row))
        label = row['output']
        scored = score(prediction['predict'], label)
        value = scored['score']
        require(math.isfinite(value) and 0. <= value <= 1., 'Invalid evaluation score')
        result = {**prediction, **scored, 'source_index': index, 'label': label}
        if on_prediction is not None:
            on_prediction(result)
        evaluated.append(result)
    return updates, evaluated


def summarize(alias, metric, predictions):
    require(bool(predictions), 'No predictions')
    scores = [row['score'] for row in predictions]
    require(all(math.isfinite(s) and 0. <= s <= 1. for s in scores), 'Invalid per-row score')
    return {'dataset': alias, 'n': len(predictions), 'metric': metric,
            'value_0_to_1': sum(scores) / len(scores),
            'truncated_inputs': sum(bool(row['input_truncated']) for row in predictions),
            'generation_limit_hits': sum(bool(row['hit_generation_limit']) for row in predictions),
            'empty_predictions': sum(not row['predict'].strip() for row in predictions)}


def validate_source(directory, expected, spec, rows_by_alias, hashes=None):
    directory = Path(directory)
    names = {'run_manifest.json', 'metrics.json'} | {item['alias'] + '.jsonl' for item in spec['datasets']}
    if hashes is not None:
        require(set(hashes) == names, 'Source checksum inventory mismatch')
        for name in names:
            require(sha256(directory / name) == hashes[name], 'Source bytes changed: ' + name)
    manifest = json.loads((directory / 'run_manifest.json').read_text(encoding='utf-8'))
    n = expected['samples_per_dataset']
    required = {'complete': True, 'model': expected['model'], 'source_commit': expected['source_commit'],
                'max_input_tokens': expected['max_input_tokens'], 'max_new_tokens': expected['eval_max_new_tokens'],
                'seed': expected['seed'], 'dtype': 'float16', 'decoding': 'greedy', 'repetition_penalty': 1.,
                'batch_size': 1, 'prompt_template': 'TLM qwen',
                'source_indices_per_dataset': list(range(n)), 'torch': expected['versions']['torch']}
    for key, value in required.items():
        require(manifest.get(key) == value, 'Source configuration mismatch: ' + key)
    require(manifest.get('method', '').startswith('Source;'), 'Expected frozen Source baseline')
    metrics = json.loads((directory / 'metrics.json').read_text(encoding='utf-8'))
    require(len(metrics) == len(spec['datasets']), 'Unexpected Source metric count')
    by_alias = {m['dataset']: m for m in metrics}
    require(set(by_alias) == {item['alias'] for item in spec['datasets']}, 'Unexpected Source datasets')
    results = {}
    for item in spec['datasets']:
        alias = item['alias']
        predictions = [json.loads(line) for line in (directory / (alias + '.jsonl')).read_text(encoding='utf-8').splitlines()]
        require(len(predictions) == len(rows_by_alias[alias]) == n, 'Unexpected Source sample count')
        for index, (record, prediction) in enumerate(zip(rows_by_alias[alias], predictions)):
            prompt = text_prompt(record)
            require(prediction['source_index'] == index, 'Source sample index mismatch')
            require(not prediction['input_truncated'] and prediction['prompt'] == prompt, 'Source prompt mismatch')
            require(prediction['original_prompt_sha256'] == hashlib.sha256(prompt.encode('utf-8')).hexdigest(),
                    'Source prompt fingerprint mismatch')
            require(prediction['label'] == record['output'], 'Source reference mismatch')
            require(prediction['metric'] == item['metric'], 'Source metric mismatch')
            require(prediction['hit_generation_limit'] == (prediction['generated_tokens'] >= expected['eval_max_new_tokens']),
                    'Source generation-cap flag mismatch')
        measured = summarize(alias, item['metric'], predictions)
        for key, value in measured.items():
            actual = by_alias[alias].get(key)
            if key == 'value_0_to_1':
                require(isinstance(actual, (int, float)) and math.isfinite(actual) and abs(value - actual) < 1e-12,
                        'Source aggregate score mismatch')
            else:
                require(actual == value, 'Source aggregate metadata mismatch: ' + key)
        results[alias] = {'summary': measured, 'predictions': predictions}
    return results


def comparison_row(source, tent):
    for key in ('dataset', 'n', 'metric'):
        require(source[key] == tent[key], 'Incomparable results: ' + key)
    require(all(math.isfinite(x) and 0. <= x <= 1. for x in
                [source['value_0_to_1'], tent['value_0_to_1']]), 'Invalid aggregate score')
    return {'dataset': source['dataset'], 'n': source['n'], 'metric': source['metric'],
            'source': source['value_0_to_1'], 'tent': tent['value_0_to_1'],
            'delta_tent_minus_source': tent['value_0_to_1'] - source['value_0_to_1'],
            'source_generation_limit_hits': source['generation_limit_hits'],
            'tent_generation_limit_hits': tent['generation_limit_hits'],
            'source_empty_predictions': source['empty_predictions'],
            'tent_empty_predictions': tent['empty_predictions']}
