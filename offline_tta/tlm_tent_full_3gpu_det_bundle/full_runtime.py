"""Pinned local Qwen runtime shared by long-input checks and full jobs."""
import csv
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import time
from job_state import require, sha256
from gpu_layout import physical_devices, logical_devices
from reproducibility import configure

HERE = Path(__file__).resolve().parent


def configuration():
    return json.loads((HERE / 'config.json').read_text()), json.loads((HERE / 'selected_datasets.json').read_text())


def protocol_id(cfg, spec):
    code = {p.name: sha256(p) for p in sorted(HERE.glob('*.py'))}
    payload = {'config': cfg, 'datasets': spec, 'code': code}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def check_environment(cfg):
    require(Path(sys.prefix).resolve() == Path(cfg['environment']).resolve(), 'Activate the project environment')
    require({n: importlib.metadata.version(n) for n in cfg['versions']} == cfg['versions'], 'Pinned package version mismatch')
    version = json.loads((Path(cfg['repository']) / 'SOURCE_VERSION.json').read_text())
    require(version['commit'] == cfg['source_commit'], 'Wrong frozen TLM commit')


class Runtime:
    def __init__(self, cfg, spec, alias):
        check_environment(cfg)
        self.determinism = configure(cfg['determinism'])
        require(spec['source_commit'] == cfg['source_commit'], 'Wrong dataset source commit')
        self.cfg, self.alias = cfg, alias
        item = next(item for item in spec['datasets'] if item['alias'] == alias)
        path = Path(cfg['data_root']) / item['filename']
        require(sha256(path) == item['sha256'], 'Dataset SHA256 changed')
        self.rows = json.loads(path.read_text())
        require(len(self.rows) == item['records'] == cfg['samples_per_dataset'] == 5000, 'Expected all 5000 records')
        self.metric = item['metric']
        visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        self.physical = physical_devices(visible, cfg['gpu_count'])
        self.device_count = cfg['gpu_count']
        for attempt in range(10):
            raw = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu',
                                           '--format=csv,noheader,nounits'], text=True)
            states = {int(r[0]): (int(r[1]), int(r[2])) for r in csv.reader(io.StringIO(raw))}
            ready = all(g in states and states[g][0] < 128 and states[g][1] < 5 for g in self.physical)
            if ready:
                break
            print(json.dumps({'phase': 'waiting_for_idle_gpus', 'attempt': attempt + 1}), flush=True)
            time.sleep(1)
        require(ready, 'Selected GPUs are busy or unavailable')
        import torch
        import numpy as np
        import nltk
        from rouge_score import rouge_scorer
        from tent_text import install_lora, make_optimizer, trainable_parameters
        sys.path.insert(0, str(Path(cfg['repository']) / 'src'))
        sys.path.insert(0, str(Path(cfg['repository']) / 'scripts/eval'))
        from llamafactory.chat import ChatModel
        from eval_utils import extract_gsm8k_answer_number
        require(torch.cuda.is_available() and torch.cuda.device_count() == self.device_count, 'Visible CUDA count differs from configuration')
        random.seed(cfg['seed'])
        np.random.seed(cfg['seed'])
        torch.manual_seed(cfg['seed'])
        chat = ChatModel({'model_name_or_path': cfg['model'], 'template': 'qwen', 'infer_backend': 'huggingface',
                          'infer_dtype': 'float16', 'flash_attn': 'sdpa', 'finetuning_type': 'full',
                          'trust_remote_code': False, 'do_sample': False, 'temperature': 0.,
                          'max_new_tokens': cfg['eval_max_new_tokens'], 'cutoff_len': cfg['max_input_tokens']})
        base = chat.engine.model
        self.device_map = {str(k): str(v) for k, v in base.hf_device_map.items()}
        require(set(self.device_map.values()) == {str(i) for i in range(self.device_count)}, 'Model must be entirely on all selected GPUs')
        require(not any(p.requires_grad for p in base.parameters()), 'Base must load frozen')
        require(all(p.dtype == torch.float16 for p in base.parameters() if p.is_floating_point()), 'Base must be FP16')
        self.model = install_lora(base, cfg['lora_rank'], cfg['lora_alpha'])
        pairs = trainable_parameters(self.model)
        require(sum(p.numel() for _, p in pairs) == cfg['expected_trainable_parameters'], 'Wrong adapter count')
        require({str(p.device) for _, p in pairs} == logical_devices(self.device_count), 'Adapters must span all selected GPUs')
        require(all(int(torch.count_nonzero(p)) == 0 for name, p in pairs if '.lora_B.' in name), 'LoRA B must initialize to zero')
        require(cfg['lora_dropout'] == 0. and not any(isinstance(m, torch.nn.Dropout) and m.p != 0. for m in self.model.modules()),
                'Expected deterministic zero-dropout model')
        self.tokenizer, self.template = chat.engine.tokenizer, chat.engine.template
        self.system = chat.engine.generating_args['default_system']
        self.stop_ids = self.template.get_stop_token_ids(self.tokenizer)
        require(bool(self.stop_ids), 'Missing stop tokens')
        self.optimizer = make_optimizer(self.model, cfg['learning_rates'][alias])
        self.extract_number = extract_gsm8k_answer_number
        self.scorer = rouge_scorer.RougeScorer(['rougeLsum'], use_stemmer=True, split_summaries=True)
        require(len(nltk.sent_tokenize('One sentence. Another sentence.')) == 2, 'Offline NLTK resources missing')

    def generate(self, record, limit):
        import torch
        require(torch.are_deterministic_algorithms_enabled() and not torch.is_deterministic_algorithms_warn_only_enabled(),
                'Strict deterministic algorithms must remain enabled')
        from transformers import GenerationConfig
        from tent_text import encode_prompt
        prompt, ids, info = encode_prompt(record, self.tokenizer, self.template, self.system, self.cfg['max_input_tokens'])
        inputs = torch.tensor([ids], device=self.model.get_input_embeddings().weight.device)
        self.model.eval()
        config = GenerationConfig(max_new_tokens=limit, do_sample=False, num_beams=1, repetition_penalty=1.,
                                  eos_token_id=self.stop_ids, pad_token_id=self.tokenizer.pad_token_id)
        with torch.no_grad():
            sequence = self.model.generate(input_ids=inputs, attention_mask=torch.ones_like(inputs),
                                           generation_config=config, use_cache=True).detach().clone()
        generated = sequence[0, len(ids):]
        record = {'prompt': prompt, 'predict': self.tokenizer.decode(generated, skip_special_tokens=True), **info,
                  'generated_tokens': len(generated), 'hit_generation_limit': len(generated) >= limit}
        return sequence, len(ids), record

    def enable_training(self):
        self.model.config.use_cache = False
        self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        require(self.model.is_gradient_checkpointing, 'Gradient checkpointing not enabled')
        self.model.train()

    def enable_evaluation(self):
        self.model.gradient_checkpointing_disable()
        self.model.config.use_cache = True
        self.model.eval()

    def step(self, ids, prompt_length, measure_after=False):
        from full_core import entropy_step
        self.model.train()
        report = entropy_step(self.model, self.optimizer, ids, prompt_length, self.stop_ids,
                              self.tokenizer.pad_token_id, measure_after=measure_after)
        require(set(report['gradient_l1_by_device']) == logical_devices(self.device_count), 'Missing device gradient participation')
        return report

    def score(self, prediction, label):
        if self.metric == 'exact_match':
            extracted = self.extract_number(prediction)
            value = float(extracted == label.strip('\n'))
        else:
            import nltk
            extracted = None
            processed = '\n'.join(nltk.sent_tokenize(prediction.replace('<n>', ' ')))
            value = self.scorer.score(label, processed)['rougeLsum'].fmeasure
        return {'score': value, 'metric': self.metric, 'extracted_answer': extracted}

    def base_digest(self):
        from tent_text import frozen_digest
        return frozen_digest(self.model)

    def save_adapter(self, directory):
        self.model.save_pretrained(str(directory), safe_serialization=True, save_embedding_layers=False)

    def peak_memory(self):
        import torch
        for gpu in range(self.device_count):
            torch.cuda.synchronize(gpu)
        return [round(torch.cuda.max_memory_allocated(g) / 1024**2, 1) for g in range(self.device_count)]
