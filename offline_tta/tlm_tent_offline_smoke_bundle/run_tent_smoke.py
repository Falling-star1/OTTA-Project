"""Orchestrate three fresh processes, then compare with the pinned Source run."""
from datetime import datetime, timedelta, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

from offline_flow import comparison_row, require, sha256, summarize, validate_source

HERE = Path(__file__).resolve().parent


def main():
    cfg = json.loads((HERE / 'config.json').read_text())
    spec = json.loads((HERE / 'selected_datasets.json').read_text())
    stamp = datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S_%f')
    output = Path(cfg['result_root']) / stamp
    output.mkdir(parents=True, exist_ok=False)
    manifest = {'complete': False, 'status': 'RUNNING', 'config': cfg,
                'purpose': '3x16 engineering comparison, not full paper reproduction',
                'process_isolation': 'one fresh Python process per dataset', 'completed_datasets': [],
                'code_sha256': {p.name: sha256(p) for p in HERE.iterdir() if p.is_file()}}

    def save():
        (output / 'run_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')

    print('OUTPUT_DIR=' + str(output), flush=True)
    save()
    try:
        require(Path(sys.prefix).resolve() == Path(cfg['environment']).resolve(), 'Wrong Python environment')
        require({n: importlib.metadata.version(n) for n in cfg['versions']} == cfg['versions'], 'Wrong package versions')
        require(spec['source_commit'] == cfg['source_commit'], 'Wrong data source commit')
        require({item['alias'] for item in spec['datasets']} == set(cfg['learning_rates']), 'Dataset/LR inventory mismatch')
        rows_by_alias = {}
        for item in spec['datasets']:
            path = Path(cfg['data_root']) / item['filename']
            require(sha256(path) == item['sha256'], 'Data changed: ' + item['alias'])
            rows = json.loads(path.read_text())
            require(len(rows) == item['records'] == 5000, 'Unexpected dataset size')
            rows_by_alias[item['alias']] = rows[:cfg['samples_per_dataset']]
        source = validate_source(cfg['source_reference'], cfg, spec, rows_by_alias, cfg['source_files_sha256'])
        (output / 'source_summary.json').write_text(json.dumps({a: v['summary'] for a, v in source.items()}, indent=2) + '\n')
        print('PINNED_SOURCE_COMPARISON=PASS', flush=True)
        comparisons, adapter_initializations, worker_pids = [], [], []
        for item in spec['datasets']:
            alias = item['alias']
            dataset_dir = output / alias
            print(json.dumps({'phase': 'starting_fresh_dataset_process', 'dataset': alias,
                              'learning_rate': cfg['learning_rates'][alias]}), flush=True)
            result = subprocess.run([sys.executable, '-u', str(HERE / 'run_tent_dataset.py'),
                                     '--dataset', alias, '--output', str(dataset_dir)], check=False)
            require(result.returncode == 0, 'Dataset child failed: ' + alias + ', exit=' + str(result.returncode))
            child = json.loads((dataset_dir / 'run_manifest.json').read_text())
            require(child.get('complete') and child.get('status') == 'PASS', 'Incomplete dataset: ' + alias)
            require(child['optimizer_updates'] == child['evaluation_count'] == cfg['samples_per_dataset'], 'Incomplete update/evaluation counts')
            require(child['frozen_base_unchanged'] and child['fresh_model_adapter_optimizer'], 'Model isolation/freeze checks failed')
            require(child['frozen_base_sha256_before'] == child['frozen_base_sha256_after'] == cfg['preflight_frozen_base_sha256'],
                    'Base model fingerprint mismatch')
            adapter_initializations.append(child['initial_adapter_sha256'])
            worker_pids.append(child['pid'])
            predictions = [json.loads(line) for line in (dataset_dir / 'predictions.jsonl').read_text(encoding='utf-8').splitlines()]
            require(len(predictions) == cfg['samples_per_dataset'], 'Wrong number of predictions')
            for index, (prediction, baseline) in enumerate(zip(predictions, source[alias]['predictions'])):
                require(prediction['source_index'] == index, 'Evaluation order changed')
                for key in ('prompt', 'original_prompt_sha256', 'label', 'metric', 'used_prompt_tokens', 'input_truncated'):
                    require(prediction[key] == baseline[key], 'Incomparable prediction: ' + key)
                require(prediction['hit_generation_limit'] == (prediction['generated_tokens'] >= cfg['eval_max_new_tokens']),
                        'Generation cap metadata mismatch')
            metrics = json.loads((dataset_dir / 'metrics.json').read_text())
            require(metrics == summarize(alias, item['metric'], predictions), 'Worker metrics inconsistent with predictions')
            require((dataset_dir / 'adapter/adapter_model.safetensors').is_file(), 'Missing adapter checkpoint')
            comparisons.append(comparison_row(source[alias]['summary'], metrics))
            manifest['completed_datasets'].append(alias)
            save()
        require(len(set(adapter_initializations)) == 1, 'Fresh adapters were not initialized identically')
        require(len(set(worker_pids)) == len(spec['datasets']), 'Dataset process IDs should differ')
        (output / 'comparison.json').write_text(json.dumps(comparisons, indent=2) + '\n')
        table = ['| Dataset | Metric | n | Source | Tent | Tent - Source | Source cap hits | Tent cap hits |',
                 '|---|---|---:|---:|---:|---:|---:|---:|']
        for row in comparisons:
            table.append('| {dataset} | {metric} | {n} | {source:.6f} | {tent:.6f} | {delta_tent_minus_source:+.6f} | {source_generation_limit_hits} | {tent_generation_limit_hits} |'.format(**row))
            print(json.dumps({'phase': 'comparison', **row}), flush=True)
        (output / 'comparison.md').write_text('\n'.join(table) + '\n')
        manifest.update({'complete': True, 'status': 'PASS', 'dataset_worker_pids': worker_pids,
                         'identical_fresh_adapter_initialization': True})
        save()
        print('TENT_OFFLINE_SMOKE=PASS', flush=True)
    except Exception as exc:
        manifest.update({'complete': False, 'status': 'FAILED', 'error_type': type(exc).__name__, 'error': str(exc)})
        save()
        (output / 'failure.txt').write_text(traceback.format_exc(), encoding='utf-8')
        print('TENT_OFFLINE_SMOKE=FAIL', flush=True)
        raise


if __name__ == '__main__':
    main()
