"""Fresh-process offline adaptation and evaluation for exactly one dataset."""
import argparse
import csv
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
import time
import traceback

from offline_flow import require, run_offline, sha256, summarize, text_prompt, unlabeled

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    cfg = json.loads((HERE / 'config.json').read_text())
    spec = json.loads((HERE / 'selected_datasets.json').read_text())
    item = next(item for item in spec['datasets'] if item['alias'] == args.dataset)
    alias = item['alias']
    output = Path(args.output).resolve()
    require(output.is_relative_to(Path(cfg['result_root']).resolve()), 'Output must remain in the configured /data result root')
    output.mkdir(parents=True, exist_ok=False)
    manifest = {'complete': False, 'status': 'RUNNING', 'phase': 'checking',
                'method': cfg['method'], 'dataset': alias, 'pid': os.getpid(), 'config': cfg,
                'optimizer_updates': 0, 'evaluation_count': 0,
                'fresh_model_adapter_optimizer': True, 'reference_answers_used_for_adaptation': False,
                'source_indices': list(range(cfg['samples_per_dataset'])),
                'code_sha256': {p.name: sha256(p) for p in HERE.iterdir() if p.is_file()}}

    def save():
        (output / 'run_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')

    def event(phase, **fields):
        line = json.dumps({'dataset': alias, 'phase': phase, **fields}, ensure_ascii=False)
        print(line, flush=True)
        with (output / 'events.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(line + '\n')

    save()
    try:
        require(Path(sys.prefix).resolve() == Path(cfg['environment']).resolve(), 'Wrong project environment')
        versions = {name: importlib.metadata.version(name) for name in cfg['versions']}
        require(versions == cfg['versions'], 'Pinned package versions do not match')
        require(spec['source_commit'] == cfg['source_commit'], 'Wrong data commit')
        repo = Path(cfg['repository'])
        require(json.loads((repo / 'SOURCE_VERSION.json').read_text())['commit'] == cfg['source_commit'], 'Wrong TLM commit')
        for name, expected in cfg['source_files_sha256'].items():
            require(sha256(Path(cfg['source_reference']) / name) == expected, 'Source changed: ' + name)
        path = Path(cfg['data_root']) / item['filename']
        require(sha256(path) == item['sha256'], 'Dataset hash mismatch')
        all_rows = json.loads(path.read_text())
        require(len(all_rows) == item['records'] == 5000, 'Unexpected full dataset length')
        rows = all_rows[:cfg['samples_per_dataset']]
        del all_rows
        source_predictions = [json.loads(line) for line in
                              (Path(cfg['source_reference']) / (alias + '.jsonl')).read_text().splitlines()]
        require(len(rows) == len(source_predictions) == 16, 'Expected exactly 16 samples')
        for index, row in enumerate(rows):
            require(source_predictions[index]['prompt'] == text_prompt(row), 'Source prompts do not match')
        visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        require(bool(re.fullmatch(r'\d+,\d+', visible)), 'Choose two physical GPU IDs')
        physical = [int(value) for value in visible.split(',')]
        require(len(set(physical)) == 2, 'GPU IDs must differ')
        # NVML utilization can retain the previous child's last sample briefly.
        for attempt in range(10):
            raw = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu',
                                           '--format=csv,noheader,nounits'], text=True)
            states = {int(r[0]): (int(r[1]), int(r[2])) for r in csv.reader(io.StringIO(raw))}
            ready = all(gpu in states and states[gpu][0] < 128 and states[gpu][1] < 5 for gpu in physical)
            if ready:
                break
            event('waiting_for_idle_gpus', attempt=attempt + 1, states={g: states.get(g) for g in physical})
            time.sleep(1)
        require(ready, 'Selected GPUs are occupied or unavailable; no model loaded')

        import torch
        import nltk
        from rouge_score import rouge_scorer
        from transformers import GenerationConfig
        from tent_text import (adapt_once, encode_prompt, frozen_digest, install_lora,
                               is_adapter_name, make_optimizer, snapshot_adapters, trainable_parameters)
        sys.path.insert(0, str(repo / 'src'))
        sys.path.insert(0, str(repo / 'scripts/eval'))
        from llamafactory.chat import ChatModel
        from eval_utils import extract_gsm8k_answer_number

        require(torch.cuda.is_available() and torch.cuda.device_count() == 2, 'Expected two CUDA devices')
        torch.manual_seed(cfg['seed'])
        event('loading_fresh_model', physical_gpus=physical)
        chat = ChatModel({'model_name_or_path': cfg['model'], 'template': 'qwen',
                          'infer_backend': 'huggingface', 'infer_dtype': 'float16', 'flash_attn': 'sdpa',
                          'finetuning_type': 'full', 'trust_remote_code': False, 'do_sample': False,
                          'temperature': 0., 'max_new_tokens': cfg['eval_max_new_tokens'],
                          'cutoff_len': cfg['max_input_tokens']})
        base = chat.engine.model
        mapping = {str(k): str(v) for k, v in base.hf_device_map.items()}
        require(set(mapping.values()) == {'0', '1'}, 'Model must be entirely on two GPUs')
        require(not any(p.requires_grad for p in base.parameters()), 'Base must load frozen')
        require(all(p.dtype == torch.float16 for p in base.parameters() if p.is_floating_point()), 'Expected FP16 base')
        tokenizer, template = chat.engine.tokenizer, chat.engine.template
        system = chat.engine.generating_args['default_system']
        stop_ids = template.get_stop_token_ids(tokenizer)
        require(bool(stop_ids), 'No generation stop tokens')
        model = install_lora(base, cfg['lora_rank'], cfg['lora_alpha'])
        require(cfg['lora_dropout'] == 0., 'Expected zero LoRA dropout')
        require(not any(isinstance(module, torch.nn.Dropout) and module.p != 0. for module in model.modules()),
                'Unexpected stochastic dropout in this deterministic comparison')
        pairs = trainable_parameters(model)
        require(sum(p.numel() for _, p in pairs) == cfg['expected_trainable_parameters'], 'Unexpected trainable parameter count')
        require({str(p.device) for _, p in pairs} == {'cuda:0', 'cuda:1'}, 'Adapters must span two GPUs')
        require(all(int(torch.count_nonzero(p)) == 0 for n, p in pairs if '.lora_B.' in n), 'Expected zero-initialized B')
        initial = snapshot_adapters(model)
        initial_hash = hashlib.sha256()
        for name, value in initial.items():
            initial_hash.update(name.encode())
            initial_hash.update(value.numpy().tobytes())
        manifest.update({'physical_gpus': physical, 'versions': versions, 'device_map': mapping,
                         'initial_adapter_sha256': initial_hash.hexdigest(), 'learning_rate': cfg['learning_rates'][alias],
                         'trainable_parameters': cfg['expected_trainable_parameters']})
        del initial
        parameters = [{'name': n, 'shape': list(p.shape), 'device': str(p.device), 'dtype': str(p.dtype)} for n, p in pairs]
        (output / 'trainable_parameters.json').write_text(json.dumps(parameters, indent=2) + '\n')
        optimizer = make_optimizer(model, cfg['learning_rates'][alias])
        scorer = rouge_scorer.RougeScorer(['rougeLsum'], use_stemmer=True, split_summaries=True)
        require(len(nltk.sent_tokenize('One sentence. Another sentence.')) == 2, 'Offline NLTK resources missing')

        def generate(index, record, limit):
            prompt, ids, info = encode_prompt(record, tokenizer, template, system, cfg['max_input_tokens'])
            source = source_predictions[index]
            require(info['original_prompt_sha256'] == source['original_prompt_sha256'], 'Prompt fingerprint differs from Source')
            require(info['used_prompt_tokens'] == source['used_prompt_tokens'], 'Encoded prompt length differs from Source')
            require(prompt == source['prompt'], 'Encoded user content differs from Source')
            inputs = torch.tensor([ids], device=model.get_input_embeddings().weight.device)
            model.eval()
            gen_config = GenerationConfig(max_new_tokens=limit, do_sample=False, num_beams=1,
                                          repetition_penalty=1., eos_token_id=stop_ids, pad_token_id=tokenizer.pad_token_id)
            with torch.no_grad():
                sequence = model.generate(input_ids=inputs, attention_mask=torch.ones_like(inputs),
                                          generation_config=gen_config, use_cache=True).detach().clone()
            generated = sequence[0, len(ids):]
            prediction = tokenizer.decode(generated, skip_special_tokens=True)
            record = {'prompt': prompt, 'predict': prediction, **info,
                      'generated_tokens': len(generated), 'hit_generation_limit': len(generated) >= limit}
            return sequence, len(ids), record

        sequence, _, probe = generate(0, unlabeled(rows[0]), cfg['eval_max_new_tokens'])
        (output / 'source_probe.json').write_text(json.dumps(probe, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        require(probe['predict'] == source_predictions[0]['predict'], 'Zero-LoRA Source probe differs; stop before adaptation')
        del sequence, probe
        gc.collect()
        torch.cuda.empty_cache()
        event('source_probe_matches', source_index=0)
        event('hashing_frozen_base_before')
        before = frozen_digest(model)
        require(before == cfg['preflight_frozen_base_sha256'], 'Base weights differ from verified single-step model')
        manifest['frozen_base_sha256_before'] = before
        manifest['phase'] = 'adapting'
        save()
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        require(model.is_gradient_checkpointing, 'Checkpointing was not enabled')
        model.config.use_cache = False

        def adapt(index, record):
            sequence, prompt_length, prefix = generate(index, record, cfg['adapt_max_new_tokens'])
            model.train()
            report = adapt_once(model, optimizer, sequence, prompt_length, stop_ids, tokenizer.pad_token_id)
            require(set(report['gradient_l1_by_device']) == {'cuda:0', 'cuda:1'}, 'A GPU received no adapter gradient')
            result = {'source_index': index, 'generated_prefix': prefix['predict'],
                      'prefix_tokens': prefix['generated_tokens'], 'prompt_sha256': prefix['original_prompt_sha256'], **report}
            with (output / 'adaptation.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(result, ensure_ascii=False) + '\n')
            manifest['optimizer_updates'] += 1
            save()
            event('adapted', completed=manifest['optimizer_updates'], total=16,
                  entropy_before=report['entropy_before'], entropy_after=report['entropy_after'])
            return report

        def before_evaluation():
            require(manifest['optimizer_updates'] == 16, 'All updates must finish before evaluation')
            optimizer.zero_grad(set_to_none=True)
            event('hashing_frozen_base_after')
            after = frozen_digest(model)
            require(before == after, 'Frozen base changed during adaptation')
            require(all(p.grad is None for n, p in model.named_parameters() if not is_adapter_name(n)), 'Base received gradients')
            manifest.update({'frozen_base_sha256_after': after, 'frozen_base_unchanged': True, 'phase': 'evaluating'})
            model.gradient_checkpointing_disable()
            model.eval()
            model.config.use_cache = True
            model.save_pretrained(str(output / 'adapter'), safe_serialization=True, save_embedding_layers=False)
            save()
            event('all_updates_complete_starting_evaluation', optimizer_updates=16)

        def predict(index, record):
            _, _, prediction = generate(index, record, cfg['eval_max_new_tokens'])
            return prediction

        def score(prediction, label):
            if item['metric'] == 'exact_match':
                extracted = extract_gsm8k_answer_number(prediction)
                value = float(extracted == label.strip('\n'))
            else:
                extracted = None
                processed = '\n'.join(nltk.sent_tokenize(prediction.replace('<n>', ' ')))
                value = scorer.score(label, processed)['rougeLsum'].fmeasure
            return {'score': value, 'metric': item['metric'], 'extracted_answer': extracted}

        def record_prediction(record):
            with (output / 'predictions.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + '\n')
            manifest['evaluation_count'] += 1
            save()
            event('evaluated', completed=manifest['evaluation_count'], total=16)

        updates, predictions = run_offline(rows, adapt, predict, score, record_prediction, before_evaluation)
        require(len(updates) == len(predictions) == 16, 'Incomplete dataset run')
        metrics = summarize(alias, item['metric'], predictions)
        (output / 'metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
        for gpu in range(2):
            torch.cuda.synchronize(gpu)
        manifest.update({'complete': True, 'status': 'PASS', 'phase': 'complete',
                         'peak_memory_mib': [round(torch.cuda.max_memory_allocated(g) / 1024**2, 1) for g in range(2)]})
        save()
        event('dataset_complete', **metrics)
        print('TENT_DATASET=PASS ' + alias, flush=True)
    except Exception as exc:
        manifest.update({'status': 'FAILED', 'complete': False, 'error_type': type(exc).__name__, 'error': str(exc)})
        save()
        (output / 'failure.txt').write_text(traceback.format_exc(), encoding='utf-8')
        print('TENT_DATASET=FAIL ' + alias, flush=True)
        raise


if __name__ == '__main__':
    main()
