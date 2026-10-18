"""Atomic two-slot training commits and durable ordered JSONL journals."""
import hashlib
import json
import os
from pathlib import Path
import random
import numpy as np
import torch
from tent_text import snapshot_adapters, trainable_parameters


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def sync_directory(directory):
    if os.name != 'nt':
        descriptor = os.open(str(directory), os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    with temp.open('wb') as stream:
        stream.write((json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode('utf-8'))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)
    sync_directory(path.parent)


def claim_directory(directory, identity):
    """Recover an empty bootstrap directory; never adopt unidentified user files."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / 'identity.json'
    if target.exists():
        require(json.loads(target.read_text(encoding='utf-8')) == identity, 'Directory identity differs')
    else:
        require({p.name for p in directory.iterdir()} <= {'identity.json.tmp'}, 'Unidentified directory contains files')
        atomic_json(target, identity)
    return directory


class Journal:
    def __init__(self, path, job_id):
        self.path = Path(path)
        self.job_id = job_id
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            with self.path.open('xb') as stream:
                stream.flush()
                os.fsync(stream.fileno())
            sync_directory(self.path.parent)
        raw = self.path.read_bytes()
        end = len(raw) if raw.endswith(b'\n') else raw.rfind(b'\n') + 1
        self.records = self._parse(raw[:end])
        if end != len(raw):
            self._truncate(end)

    def _parse(self, raw):
        rows = []
        for index, line in enumerate(raw.splitlines()):
            try:
                row = json.loads(line)
            except (ValueError, UnicodeError) as exc:
                raise ValueError('Corrupt complete journal record: ' + str(index)) from exc
            require(row.get('job_id') == self.job_id, 'Journal belongs to another job')
            require(type(row.get('source_index')) is int and row['source_index'] == index, 'Journal duplicate/gap/index mismatch')
            rows.append(row)
        return rows

    @property
    def offset(self):
        return self.path.stat().st_size

    def _truncate(self, offset):
        with self.path.open('r+b') as stream:
            stream.truncate(offset)
            stream.flush()
            os.fsync(stream.fileno())

    def append(self, record):
        require(type(record.get('source_index')) is int and record['source_index'] == len(self.records), 'Journal append index mismatch')
        require('job_id' not in record or record['job_id'] == self.job_id, 'Foreign record')
        row = {**record, 'job_id': self.job_id}
        line = (json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n').encode('utf-8')
        with self.path.open('ab') as stream:
            stream.write(line)
            stream.flush()
            os.fsync(stream.fileno())
        self.records.append(row)

    def truncate_to(self, offset, expected_count):
        raw = self.path.read_bytes()
        require(type(offset) is int and 0 <= offset <= len(raw), 'Checkpoint log offset outside journal')
        require(offset == 0 or raw[:offset].endswith(b'\n'), 'Checkpoint offset is not a record boundary')
        rows = self._parse(raw[:offset])
        require(len(rows) == expected_count, 'Checkpoint cursor and journal length differ')
        self._truncate(offset)
        self.records = rows


class CheckpointStore:
    def __init__(self, directory, job_id):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.job_id = job_id
        self.pointer = self.directory / 'commit.json'

    def _pointer(self):
        pointer = json.loads(self.pointer.read_text(encoding='utf-8'))
        require(pointer['job_id'] == self.job_id, 'Checkpoint belongs to another protocol/job')
        require(pointer['slot'] in (0, 1) and pointer['filename'] == 'state_' + str(pointer['slot']) + '.pt', 'Invalid checkpoint slot')
        return pointer

    def commit(self, payload):
        require(type(payload['next_index']) is int and payload['next_index'] >= 0, 'Invalid checkpoint cursor')
        require(type(payload['log_offset']) is int and payload['log_offset'] >= 0, 'Invalid log offset')
        old = self._pointer() if self.pointer.exists() else None
        if old is not None:
            require(payload['next_index'] >= old['next_index'], 'Cannot commit a backward training cursor')
        slot = 1 - old['slot'] if old else 0
        filename = 'state_' + str(slot) + '.pt'
        target = self.directory / filename
        temp = self.directory / (filename + '.tmp')
        state = {**payload, 'job_id': self.job_id, 'format_version': 1}
        with temp.open('wb') as stream:
            torch.save(state, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, target)
        sync_directory(self.directory)
        atomic_json(self.pointer, {'job_id': self.job_id, 'slot': slot, 'filename': filename,
                                   'sha256': sha256(target), 'next_index': state['next_index'], 'log_offset': state['log_offset']})

    def load(self):
        pointer = self._pointer()
        target = self.directory / pointer['filename']
        require(sha256(target) == pointer['sha256'], 'Checkpoint SHA256 mismatch')
        payload = torch.load(target, map_location='cpu', weights_only=True)
        require(payload.get('job_id') == self.job_id and payload.get('format_version') == 1, 'Invalid checkpoint identity/version')
        require(payload['next_index'] == pointer['next_index'] and payload['log_offset'] == pointer['log_offset'], 'Checkpoint/pointer cursor mismatch')
        return payload


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(item) for item in value)
    return value


def capture_training(model, optimizer, next_index, log_offset):
    numpy = np.random.get_state()
    return {'next_index': next_index, 'log_offset': log_offset,
            'adapters': snapshot_adapters(model), 'trainable_names': [n for n, _ in trainable_parameters(model)],
            'optimizer': cpu_tree(optimizer.state_dict()), 'python_rng': random.getstate(),
            'numpy_rng': [numpy[0], numpy[1].tolist(), numpy[2], numpy[3], numpy[4]],
            'torch_rng': torch.get_rng_state(),
            'cuda_rng': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_training(model, optimizer, payload):
    pairs = trainable_parameters(model)
    require([n for n, _ in pairs] == payload['trainable_names'], 'Trainable parameter order changed')
    require(set(payload['adapters']) == {n for n, _ in pairs}, 'Adapter state names changed')
    for name, parameter in pairs:
        saved = payload['adapters'][name]
        require(saved.shape == parameter.shape and saved.dtype == parameter.dtype and bool(torch.isfinite(saved).all()), 'Invalid adapter tensor: ' + name)
    with torch.no_grad():
        for name, parameter in pairs:
            parameter.copy_(payload['adapters'][name].to(parameter.device))
    optimizer.load_state_dict(payload['optimizer'])
    optimizer.zero_grad(set_to_none=True)
    for values in optimizer.state.values():
        for value in values.values():
            if isinstance(value, torch.Tensor):
                require(bool(torch.isfinite(value).all()), 'Nonfinite restored Adam state')
    random.setstate(payload['python_rng'])
    numpy = payload['numpy_rng']
    np.random.set_state((numpy[0], np.asarray(numpy[1], dtype=np.uint32), numpy[2], numpy[3], numpy[4]))
    torch.set_rng_state(payload['torch_rng'])
    if payload['cuda_rng']:
        require(torch.cuda.is_available() and len(payload['cuda_rng']) == torch.cuda.device_count(), 'CUDA RNG device count mismatch')
        torch.cuda.set_rng_state_all(payload['cuda_rng'])
