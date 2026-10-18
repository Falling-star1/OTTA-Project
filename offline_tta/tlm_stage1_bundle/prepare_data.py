"""Copy verified official AdaptEval files into /data without changing upstream."""
import hashlib
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
REPO = Path('/home/chuwenjing/TTA_LLM/TLM')
BUNDLE = Path(sys.argv[1]).resolve()
spec = json.loads((HERE/'selected_datasets.json').read_text(encoding='utf-8'))
version = json.loads((REPO/'SOURCE_VERSION.json').read_text(encoding='utf-8'))
if version['commit'] != spec['source_commit']:
    raise RuntimeError('Unexpected TLM source revision')
DATA = Path('/data/chuwenjing/datasets/AdaptEval') / spec['source_commit'][:12]
NLTK = Path('/data/chuwenjing/cache/nltk_data')

def digest(raw):
    return hashlib.sha256(raw).hexdigest()

def write_if_new(path, raw):
    if path.exists():
        if path.read_bytes() != raw:
            raise RuntimeError(f'Refusing to overwrite a different existing file: {path}')
        return
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary = path.with_name(path.name+f'.tmp-{os.getpid()}')
    with temporary.open('xb') as stream:
        stream.write(raw)
    temporary.replace(path)

pending = []
registry = {}
provenance = []
for item in spec['datasets']:
    source = REPO/'data/AdaptEval'/item['filename']
    raw = source.read_bytes()
    if digest(raw) != item['sha256']:
        raise RuntimeError(f'Source hash mismatch: {source}')
    records = json.loads(raw)
    if len(records) != item['records']:
        raise RuntimeError(f'Unexpected sample count: {source}')
    if any(not isinstance(r.get('output'),str) or not r['output'].strip() for r in records):
        raise RuntimeError(f'Missing evaluation references: {source}')
    pending.append((DATA/item['filename'],raw))
    smoke_name = item['filename'].replace('_random_5k.json','_smoke16.json')
    smoke = (json.dumps(records[:16],ensure_ascii=False,indent=2)+'\n').encode('utf-8')
    pending.append((DATA/smoke_name,smoke))
    registry[item['alias']] = {'file_name':item['filename']}
    registry[item['alias']+'_smoke16'] = {'file_name':smoke_name}
    provenance.append({**item,'source_file':str(source),'copied_file':str(DATA/item['filename']),
                       'smoke_file':str(DATA/smoke_name),'smoke_records':16,
                       'smoke_source_indices':list(range(16)),'smoke_sha256':digest(smoke)})
for item in json.loads((BUNDLE/'nltk-manifest.json').read_text(encoding='utf-8')):
    raw = (BUNDLE/item['path']).read_bytes()
    if digest(raw) != item['sha256']:
        raise RuntimeError('NLTK resource hash mismatch: '+item['path'])
    pending.append((NLTK/Path(item['path']).relative_to('nltk_data'),raw))
for path,value in [(DATA/'dataset_info.json',registry),
                   (DATA/'provenance.json',{'source_commit':spec['source_commit'],'datasets':provenance})]:
    pending.append((path,(json.dumps(value,ensure_ascii=False,indent=2)+'\n').encode('utf-8')))
# Validate existing destinations before writing any new files.
for path,raw in pending:
    if path.exists() and path.read_bytes()!=raw:
        raise RuntimeError(f'Destination already has different content: {path}')
for path,raw in pending:
    write_if_new(path,raw)
    if digest(path.read_bytes()) != digest(raw):
        raise RuntimeError(f'Copy verification failed: {path}')

import nltk
nltk.data.path.insert(0,str(NLTK))
assert nltk.sent_tokenize('This is one sentence. Here is another.') == ['This is one sentence.','Here is another.']
from rouge_score import rouge_scorer
scorer=rouge_scorer.RougeScorer(['rougeLsum'],use_stemmer=True,split_summaries=True)
score=scorer.score('A cat sits on the mat. It is sleeping.', 'A cat sits on the mat. It is sleeping.')['rougeLsum'].fmeasure
assert abs(score-1.0)<1e-12
print(json.dumps({'data_directory':str(DATA),'full_datasets':3,'records_each':5000,
                  'smoke_datasets':3,'smoke_records_each':16,'nltk_directory':str(NLTK),
                  'offline_sentence_split_and_rougeLsum':'PASS'}),flush=True)
print('TLM_STAGE1_DATA=PASS',flush=True)
