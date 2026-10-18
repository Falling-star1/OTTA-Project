"""Validate one real, unlabeled entropy update; preserve frozen upstream/model."""
import csv
from datetime import datetime, timedelta, timezone
import gc
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import traceback

HERE = Path(__file__).resolve().parent


def check(condition, message):
    if not condition:
        raise RuntimeError(message)


def file_sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    cfg = json.loads((HERE / 'config.json').read_text(encoding='utf-8'))
    stamp = datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S_%f')
    output = Path(cfg['result_root']) / stamp
    output.mkdir(parents=True, exist_ok=False)
    manifest = {'complete': False, 'status': 'RUNNING', 'config': cfg,
                'purpose': 'One unlabeled update; not a task score or full reproduction',
                'reference_answers_used': False,
                'implementation_note': 'TLM specifies 80-token entropy; Tent adapter parameter set not confirmed by author code',
                'files': {p.name: file_sha(p) for p in HERE.iterdir() if p.is_file()}}

    def save_manifest():
        (output / 'run_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')

    def event(phase, **fields):
        payload = {'phase': phase, **fields}
        line = json.dumps(payload, ensure_ascii=False)
        print(line, flush=True)
        with (output / 'events.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(line + '\n')

    print('OUTPUT_DIR=' + str(output), flush=True)
    save_manifest()
    try:
        check(Path(sys.prefix).resolve() == Path(cfg['environment']).resolve(), 'Wrong Python environment: ' + sys.prefix)
        versions = {name: importlib.metadata.version(name) for name in cfg['versions']}
        check(versions == cfg['versions'], 'Pinned environment mismatch: ' + str(versions))
        manifest['versions'] = versions
        repo = Path(cfg['repository'])
        version = json.loads((repo / 'SOURCE_VERSION.json').read_text())
        check(version['commit'] == cfg['source_commit'], 'Wrong TLM source commit')
        spec = json.loads((HERE / 'selected_datasets.json').read_text())
        check(spec['source_commit'] == cfg['source_commit'], 'Wrong dataset specification')
        selected = next(item for item in spec['datasets'] if item['alias'] == cfg['dataset_alias'])
        dataset_path = Path(cfg['data_root']) / selected['filename']
        check(file_sha(dataset_path) == selected['sha256'], 'Dataset hash mismatch')
        rows = json.loads(dataset_path.read_text())
        check(len(rows) == selected['records'] == 5000, 'Unexpected dataset size')
        row = rows[cfg['source_index']]
        # Remove reference fields before any adaptation-related function is called.
        unlabeled = {'instruction': row['instruction'], 'input': row['input']}
        del row, rows
        visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        check(bool(re.fullmatch(r'\d+,\d+', visible)), 'Select exactly two physical GPU IDs')
        physical = [int(value) for value in visible.split(',')]
        check(len(set(physical)) == 2, 'GPU IDs must differ')
        gpu_csv = subprocess.check_output([
            'nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu',
            '--format=csv,noheader,nounits'], text=True)
        states = {int(r[0]): (int(r[1]), int(r[2])) for r in csv.reader(io.StringIO(gpu_csv))}
        for gpu in physical:
            check(gpu in states and states[gpu][0] < 128 and states[gpu][1] < 5,
                  'GPU busy or unavailable; choose another pair: ' + str((gpu, states.get(gpu))))
        event('input_and_environment_verified', physical_gpus=physical)

        import torch
        from transformers import GenerationConfig
        from tent_text import (adapt_once, encode_prompt, frozen_digest, install_lora,
                               is_adapter_name, make_optimizer, reset_adapters,
                               snapshot_adapters, trainable_parameters)
        sys.path.insert(0, str(repo / 'src'))
        from llamafactory.chat import ChatModel

        check(torch.cuda.is_available() and torch.cuda.device_count() == 2, 'Expected two visible CUDA devices')
        torch.manual_seed(cfg['seed'])
        event('loading_model', dtype='float16')
        chat = ChatModel({'model_name_or_path': cfg['model'], 'template': 'qwen',
                          'infer_backend': 'huggingface', 'infer_dtype': 'float16',
                          'flash_attn': 'sdpa', 'finetuning_type': 'full',
                          'trust_remote_code': False, 'do_sample': False,
                          'temperature': 0., 'max_new_tokens': cfg['adapt_max_new_tokens'],
                          'cutoff_len': cfg['max_input_tokens']})
        base = chat.engine.model
        mapping = {str(k): str(v) for k, v in base.hf_device_map.items()}
        check(set(mapping.values()) == {'0', '1'}, 'Expected GPU-only model dispatch: ' + str(mapping))
        check(not any(p.requires_grad for p in base.parameters()), 'Expected frozen Source model on load')
        check(all(p.dtype == torch.float16 for p in base.parameters() if p.is_floating_point()),
              'Expected FP16 base parameters')
        tokenizer, template = chat.engine.tokenizer, chat.engine.template
        prompt, prompt_ids, prompt_info = encode_prompt(
            unlabeled, tokenizer, template, chat.engine.generating_args['default_system'], cfg['max_input_tokens'])
        check(not prompt_info['input_truncated'], 'Preflight sample unexpectedly requires input truncation')
        stop_ids = template.get_stop_token_ids(tokenizer)
        check(bool(stop_ids), 'TLM template has no stop IDs')
        model = install_lora(base, cfg['lora_rank'], cfg['lora_alpha'])
        check(cfg['lora_dropout'] == 0., 'This preflight requires zero LoRA dropout')
        pairs = trainable_parameters(model)
        parameter_count = sum(p.numel() for _, p in pairs)
        check(parameter_count == cfg['expected_trainable_parameters'], 'Unexpected adapter parameter count')
        check({str(p.device) for _, p in pairs} == {'cuda:0', 'cuda:1'}, 'Adapters must span both GPUs')
        for name, p in pairs:
            if '.lora_B.' in name:
                check(int(torch.count_nonzero(p)) == 0, 'LoRA B must initialize to zero')
        parameter_info = [{'name': n, 'shape': list(p.shape), 'device': str(p.device),
                           'dtype': str(p.dtype), 'numel': p.numel()} for n, p in pairs]
        manifest.update({'physical_gpus': physical, 'device_map': mapping,
                         'trainable_parameters': parameter_count, 'parameter_info': parameter_info,
                         'dataset_file_sha256': selected['sha256'], 'prompt_info': prompt_info,
                         'stop_ids': stop_ids, 'pad_id': tokenizer.pad_token_id})
        (output / 'trainable_parameters.json').write_text(json.dumps(parameter_info, indent=2) + '\n', encoding='utf-8')
        save_manifest()
        event('adapters_verified', trainable_parameters=parameter_count, tensors=len(pairs))
        for item in parameter_info:
            event('trainable_parameter', **item)
        inputs = torch.tensor([prompt_ids], device=model.get_input_embeddings().weight.device)
        config = GenerationConfig(max_new_tokens=cfg['adapt_max_new_tokens'], do_sample=False,
                                  num_beams=1, repetition_penalty=1., eos_token_id=stop_ids,
                                  pad_token_id=tokenizer.pad_token_id)
        # Generation has no autograd graph; adaptation below differentiates a recomputed forward.
        with torch.no_grad():
            sequence = model.generate(input_ids=inputs, attention_mask=torch.ones_like(inputs),
                                      generation_config=config, use_cache=True).detach().clone()
        generated = sequence[0, len(prompt_ids):]
        sampled = {'source_index': cfg['source_index'], 'prompt': prompt,
                   'generated_prefix': tokenizer.decode(generated, skip_special_tokens=True),
                   'generated_token_ids': generated.cpu().tolist(), 'generated_tokens': len(generated),
                   'prompt_info': prompt_info, 'reference_answer_included': False}
        (output / 'sampled_prefix.json').write_text(json.dumps(sampled, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        event('prefix_generated', generated_tokens=len(generated))
        del generated, inputs
        gc.collect()
        torch.cuda.empty_cache()
        model.config.use_cache = False
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        check(model.is_gradient_checkpointing, 'Gradient checkpointing was not enabled')
        initial = snapshot_adapters(model)
        optimizer = make_optimizer(model, cfg['learning_rate'])
        event('hashing_frozen_base_before')
        before = frozen_digest(model)
        event('performing_single_entropy_update')
        report = adapt_once(model, optimizer, sequence, len(prompt_ids), stop_ids, tokenizer.pad_token_id)
        check(set(report['gradient_l1_by_device']) == {'cuda:0', 'cuda:1'}, 'Missing GPU gradient participation')
        event('hashing_frozen_base_after')
        after = frozen_digest(model)
        check(before == after, 'Frozen base changed during update')
        check(all(p.grad is None for n, p in model.named_parameters() if not is_adapter_name(n)),
              'A frozen parameter received gradients')
        report.update({'frozen_base_unchanged': True, 'frozen_base_sha256_before': before,
                       'frozen_base_sha256_after': after})
        reset_adapters(model, optimizer, initial)
        restored = snapshot_adapters(model)
        check(not optimizer.state, 'Adam state did not reset')
        check(all(torch.equal(restored[n], value) for n, value in initial.items()), 'Adapter reset failed')
        report['adapter_and_optimizer_reset'] = 'PASS'
        for gpu in range(2):
            torch.cuda.synchronize(gpu)
        report['peak_memory_mib'] = [round(torch.cuda.max_memory_allocated(i) / 1024**2, 1) for i in range(2)]
        (output / 'step_report.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
        event('single_step_verified', **report)
        manifest.update({'complete': True, 'status': 'PASS'})
        save_manifest()
        print('TENT_SINGLE_STEP=PASS', flush=True)
    except Exception as exc:
        manifest.update({'complete': False, 'status': 'FAILED', 'error_type': type(exc).__name__, 'error': str(exc)})
        save_manifest()
        (output / 'failure.txt').write_text(traceback.format_exc(), encoding='utf-8')
        print('TENT_SINGLE_STEP=FAIL', flush=True)
        raise


if __name__ == '__main__':
    main()
