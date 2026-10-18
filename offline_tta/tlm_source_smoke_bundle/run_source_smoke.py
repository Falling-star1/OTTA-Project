"""Frozen Qwen / TLM text-template smoke run: 16 samples per selected dataset."""
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys

HERE=Path(__file__).resolve().parent
REPO=Path('/home/chuwenjing/TTA_LLM/TLM')
MODEL=Path('/data/chuwenjing/models/Qwen2.5-7B-Instruct')
SPEC=json.loads((HERE/'selected_datasets.json').read_text(encoding='utf-8'))
DATA=Path('/data/chuwenjing/datasets/AdaptEval')/SPEC['source_commit'][:12]
MAX_INPUT=1024
MAX_NEW=128
assert json.loads((REPO/'SOURCE_VERSION.json').read_text())['commit']==SPEC['source_commit']
registry=json.loads((DATA/'dataset_info.json').read_text())
work=[]
for item in SPEC['datasets']:
    raw=(DATA/item['filename']).read_bytes()
    assert hashlib.sha256(raw).hexdigest()==item['sha256'],item['filename']
    rows=json.loads(raw)
    smoke=json.loads((DATA/registry[item['alias']+'_smoke16']['file_name']).read_text())
    assert len(rows)==5000 and len(smoke)==16 and smoke==rows[:16]
    work.append((item,smoke))

physical=[int(x) for x in os.environ['CUDA_VISIBLE_DEVICES'].split(',')]
assert len(physical)==2 and len(set(physical))==2
gpu_rows=csv.reader(io.StringIO(subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)))
gpu_states={int(r[0]):(int(r[1]),int(r[2])) for r in gpu_rows}
for gpu in physical:
    assert gpu in gpu_states and gpu_states[gpu][0]<128 and gpu_states[gpu][1]<5,('GPU busy; choose a different pair',gpu,gpu_states.get(gpu))

import torch
import nltk
from rouge_score import rouge_scorer
from transformers import GenerationConfig
sys.path.insert(0,str(REPO/'src'))
sys.path.insert(0,str(REPO/'scripts/eval'))
from llamafactory.chat import ChatModel
from eval_utils import extract_gsm8k_answer_number

torch.manual_seed(42)
assert torch.cuda.is_available() and torch.cuda.device_count()==2
assert len(nltk.sent_tokenize('One sentence. Another sentence.'))==2
chat=ChatModel({'model_name_or_path':str(MODEL),'template':'qwen','infer_backend':'huggingface',
                'infer_dtype':'float16','flash_attn':'sdpa','finetuning_type':'full',
                'trust_remote_code':False,'do_sample':False,'temperature':0.0,
                'max_new_tokens':MAX_NEW,'cutoff_len':MAX_INPUT})
model=chat.engine.model
tokenizer=chat.engine.tokenizer
template=chat.engine.template
assert not any(p.requires_grad for p in model.parameters())
assert next(model.parameters()).dtype==torch.float16
device_map={str(k):str(v) for k,v in model.hf_device_map.items()}
assert len(set(device_map.values()))==2 and not {'cpu','disk'}.intersection(device_map.values())
generation=GenerationConfig(max_new_tokens=MAX_NEW,do_sample=False,num_beams=1,
                            repetition_penalty=1.0,eos_token_id=template.get_stop_token_ids(tokenizer),
                            pad_token_id=tokenizer.pad_token_id)
scorer=rouge_scorer.RougeScorer(['rougeLsum'],use_stemmer=True,split_summaries=True)

def encode(content):
    return template.encode_oneturn(tokenizer,[{'role':'user','content':content},{'role':'assistant','content':''}],
                                    system=chat.engine.generating_args['default_system'])[0]

def prepare_prompt(row):
    original=row['instruction']+('\n'+row['input'] if row['input'] else '')
    used=original
    ids=encode(used)
    original_length=len(ids)
    if original_length>MAX_INPUT:
        content_ids=tokenizer.encode(original,add_special_tokens=False)
        while len(ids)>MAX_INPUT:
            remove=max(1,len(ids)-MAX_INPUT)
            content_ids=content_ids[:-remove]
            if not content_ids:raise RuntimeError('Prompt budget too small for the TLM template')
            used=tokenizer.decode(content_ids,skip_special_tokens=False)
            ids=encode(used)
    return original,used,ids,original_length

stamp=datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S_%f')
output=Path('/data/chuwenjing/results/source_smoke/qwen25-7b')/stamp
output.mkdir(parents=True,exist_ok=False)
manifest={'purpose':'48-sample smoke test; not a full reproduction result','method':'Source; all parameters frozen',
          'model':str(MODEL),'source_commit':SPEC['source_commit'],'physical_gpus':physical,
          'dtype':'float16','max_input_tokens':MAX_INPUT,'max_new_tokens':MAX_NEW,'seed':42,
          'decoding':'greedy','repetition_penalty':1.0,'batch_size':1,
          'prompt_template':'TLM qwen','text_entry':'TLM template + model.generate',
          'source_indices_per_dataset':list(range(16)),'device_map':device_map,
          'torch':torch.__version__,'complete':False}
(output/'run_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
print('OUTPUT_DIR='+str(output),flush=True)
summary=[]
for item,rows in work:
    scores=[]
    empty=truncated=at_limit=0
    with (output/(item['alias']+'.jsonl')).open('x',encoding='utf-8') as stream:
        for index,row in enumerate(rows):
            original,prompt,ids,original_length=prepare_prompt(row)
            inputs=torch.tensor([ids],device=model.get_input_embeddings().weight.device)
            with torch.inference_mode():
                generated=model.generate(input_ids=inputs,attention_mask=torch.ones_like(inputs),generation_config=generation)
            tokens=generated[0,len(ids):]
            prediction=tokenizer.decode(tokens,skip_special_tokens=True)
            label=row['output']
            if item['metric']=='exact_match':
                extracted=extract_gsm8k_answer_number(prediction)
                value=float(extracted==label.strip('\n'))
            else:
                extracted=None
                processed='\n'.join(nltk.sent_tokenize(prediction.replace('<n>',' ')))
                value=scorer.score(label,processed)['rougeLsum'].fmeasure
            truncated+=int(original_length>len(ids))
            at_limit+=int(len(tokens)>=MAX_NEW)
            empty+=int(not prediction.strip())
            scores.append(value)
            result={'source_index':index,'prompt':prompt,'predict':prediction,'label':label,
                    'original_prompt_sha256':hashlib.sha256(original.encode('utf-8')).hexdigest(),
                    'original_prompt_tokens':original_length,'used_prompt_tokens':len(ids),
                    'input_truncated':original_length>len(ids),'generated_tokens':len(tokens),
                    'hit_generation_limit':len(tokens)>=MAX_NEW,'metric':item['metric'],
                    'score':value,'extracted_answer':extracted}
            stream.write(json.dumps(result,ensure_ascii=False)+'\n')
            stream.flush()
            print(json.dumps({'dataset':item['alias'],'completed':index+1,'total':16}),flush=True)
    summary.append({'dataset':item['alias'],'n':len(scores),'metric':item['metric'],
                    'value_0_to_1':sum(scores)/len(scores),'truncated_inputs':truncated,
                    'generation_limit_hits':at_limit,'empty_predictions':empty})
    print(json.dumps(summary[-1]),flush=True)
manifest['complete']=True
manifest['peak_memory_mib']=[round(torch.cuda.max_memory_allocated(i)/1024**2,1) for i in range(2)]
(output/'run_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
(output/'metrics.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
print('SOURCE_SMOKE=PASS',flush=True)
