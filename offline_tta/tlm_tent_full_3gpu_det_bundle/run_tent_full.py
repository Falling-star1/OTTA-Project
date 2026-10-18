"""Preflight receipt gate and resumable fresh-process full experiment controller."""
import argparse
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import traceback
from full_runtime import HERE, check_environment, configuration, protocol_id
from job_state import atomic_json, require, sha256, claim_directory
from offline_flow import comparison_row
from run_dataset import dataset_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['preflight', 'run', 'resume'])
    parser.add_argument('--output')
    args = parser.parse_args()
    cfg, spec = configuration()
    check_environment(cfg)
    protocol = protocol_id(cfg, spec)
    kind = 'preflight' if args.action == 'preflight' else 'full'
    base_root = Path(cfg['preflight_root'] if kind == 'preflight' else cfg['result_root']).resolve()
    if args.action == 'resume':
        require(bool(args.output), 'Resume requires the exact OUTPUT_DIR')
        output = Path(args.output).resolve()
        require(output != base_root and output.is_relative_to(base_root), 'Resume path outside result root')
        require(output.is_dir(), 'Resume requires an existing run directory')
    else:
        require(args.output is None, 'New runs use a new timestamp directory')
        stamp = datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S_%f')
        output = base_root / stamp
        require(not output.exists(), 'New run directory already exists')
    root_id = dataset_id(protocol, output, kind)
    claim_directory(output, {'job_id': root_id, 'protocol_id': protocol, 'kind': kind})
    if (output / 'run_manifest.json').exists():
        old = json.loads((output / 'run_manifest.json').read_text())
        require(old['protocol_id'] == protocol and old['kind'] == kind, 'Resume protocol changed')
    if kind == 'full':
        receipt = json.loads(Path(cfg['receipt']).read_text())
        require(receipt['protocol_id'] == protocol and receipt['status'] == 'PASS', 'Run the matching long-input preflight first')
        reports = receipt['reports']
        require(set(reports) == {i['alias'] for i in spec['datasets']}, 'Incomplete preflight receipt')
        for item in reports.values():
            path = Path(item['path']).resolve()
            require(path.is_relative_to(Path(cfg['preflight_root']).resolve()) and sha256(path) == item['sha256'], 'Preflight report changed')
    manifest = {'kind': kind, 'protocol_id': protocol, 'config': cfg, 'complete': False, 'pid': os.getpid(),
                'status': 'RUNNING', 'completed_datasets': [], 'started_at': datetime.now(timezone.utc).isoformat()}
    stop, active = {'requested': False}, {'process': None}

    def on_signal(signum, frame):
        stop['requested'] = True
        if active['process'] is not None and active['process'].poll() is None:
            active['process'].send_signal(signal.SIGTERM)

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    print('OUTPUT_DIR=' + str(output), flush=True)
    atomic_json(output / 'run_manifest.json', manifest)
    try:
        for item in spec['datasets']:
            if stop['requested']:
                raise InterruptedError('Stopped between datasets')
            alias = item['alias']
            child_dir = output / alias
            expected_id = dataset_id(protocol, child_dir, alias)
            previous = None
            existing_child = child_dir.exists()
            claim_directory(child_dir, {'job_id': expected_id, 'protocol_id': protocol,
                                        'dataset': alias, 'mode': kind})
            if (child_dir / 'run_manifest.json').exists():
                previous = json.loads((child_dir / 'run_manifest.json').read_text())
                require(previous['job_id'] == expected_id and previous['protocol_id'] == protocol, 'Child protocol mismatch')
            if previous is None or not previous.get('complete'):
                cmd = [sys.executable, '-u', str(HERE / 'run_dataset.py'), '--mode', kind,
                       '--dataset', alias, '--output', str(child_dir)]
                if existing_child:
                    require(kind == 'full', 'Retry failed preflight with a new preflight run')
                    cmd.append('--resume')
                active['process'] = subprocess.Popen(cmd)
                status = active['process'].wait()
                active['process'] = None
                if status == 75 or stop['requested']:
                    raise InterruptedError('Dataset stopped at a safe boundary')
                require(status == 0, 'Dataset failed: ' + alias + ', exit=' + str(status))
            child = json.loads((child_dir / 'run_manifest.json').read_text())
            require(child['complete'] and child['status'] == 'PASS', 'Dataset did not finish')
            manifest['completed_datasets'].append(alias)
            atomic_json(output / 'run_manifest.json', manifest)
        if kind == 'preflight':
            reports = {item['alias']: {'path': str(output / item['alias'] / 'preflight_report.json'),
                       'sha256': sha256(output / item['alias'] / 'preflight_report.json')} for item in spec['datasets']}
            for entry in reports.values():
                report = json.loads(Path(entry['path']).read_text())
                require(report['checkpoint_continuation'] == report['rng_restore'] == 'EXACT_MATCH' and report['frozen_base_unchanged'],
                        'Preflight recovery/freeze check failed')
            atomic_json(cfg['receipt'], {'status': 'PASS', 'protocol_id': protocol, 'reports': reports})
        else:
            comparisons = []
            for item in spec['datasets']:
                directory = output / item['alias']
                metrics = json.loads((directory / 'metrics.json').read_text())
                require(metrics['source']['n'] == metrics['tent']['n'] == metrics['adaptation']['processed'] ==
                        metrics['adaptation']['optimizer_steps'] == cfg['samples_per_dataset'], 'Incomplete 5000-sample result')
                require(metrics['frozen_base_unchanged'], 'Base changed')
                require(sha256(directory / 'source.jsonl') == metrics['source_sha256'] and
                        sha256(directory / 'tent.jsonl') == metrics['tent_sha256'], 'Completed prediction journal changed')
                comparisons.append(comparison_row(metrics['source'], metrics['tent']))
            atomic_json(output / 'comparison.json', comparisons)
            table = ['| Dataset | Metric | n | Source | Tent | Delta |', '|---|---|---:|---:|---:|---:|']
            for row in comparisons:
                table.append('| {dataset} | {metric} | {n} | {source:.6f} | {tent:.6f} | {delta_tent_minus_source:+.6f} |'.format(**row))
                print(json.dumps({'phase': 'comparison', **row}), flush=True)
            (output / 'comparison.md').write_text('\n'.join(table) + '\n', encoding='utf-8')
        manifest.update({'complete': True, 'status': 'PASS', 'finished_at': datetime.now(timezone.utc).isoformat()})
        atomic_json(output / 'run_manifest.json', manifest)
        print('TENT_FULL_PREFLIGHT=PASS' if kind == 'preflight' else 'TENT_FULL_RUN=PASS', flush=True)
    except InterruptedError as exc:
        manifest.update({'complete': False, 'status': 'PAUSED', 'reason': str(exc)})
        atomic_json(output / 'run_manifest.json', manifest)
        print('RESUME_OUTPUT_DIR=' + str(output), flush=True)
        raise SystemExit(75)
    except Exception as exc:
        manifest.update({'complete': False, 'status': 'FAILED', 'error': str(exc), 'error_type': type(exc).__name__})
        atomic_json(output / 'run_manifest.json', manifest)
        (output / 'failure.txt').write_text(traceback.format_exc(), encoding='utf-8')
        raise


if __name__ == '__main__':
    main()
