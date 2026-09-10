"""Durable, provider-independent checkpoints for the frozen Long experiment."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter

MODEL = 'Qwen/Qwen3-4B'
GENERATION = dict(max_new_tokens=900, do_sample=False, enable_thinking=False,
                  precision='bfloat16', attention='sdpa', batch_size=1)


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


def digest(data):
    return hashlib.sha256(json.dumps(data, ensure_ascii=False, sort_keys=True,
                                     separators=(',', ':')).encode()).hexdigest()


class ModalCheckpointProvider:
    """Only public messages reach remote inference; IDs remain cache metadata."""
    def __init__(self, remote, directory, config, on_checkpoint=None):
        self.remote, self.directory, self.config = remote, Path(directory), config
        self.on_checkpoint = on_checkpoint
        self.case = None
        self.case_calls = []

    def begin_case(self, case):
        self.case, self.case_calls = case, []

    def complete(self, messages):
        from language.long_parser import ProviderResponse
        if self.case is None:
            raise RuntimeError('begin_case required')
        key = digest(dict(case=self.case, config=self.config, messages=messages))
        path = self.directory / (key + '.json')
        if path.exists():
            saved = json.loads(path.read_text())
        else:
            started = perf_counter()
            saved = self.remote(messages, key)
            saved = dict(saved, request_roundtrip_ms=(perf_counter()-started)*1000,
                         case_id=self.case)
            # Persist even invalid/truncated JSON before LongParser's validation.
            atomic_json(path, saved)
        if saved['request_digest'] != key or saved['model'] != MODEL:
            raise RuntimeError('Checkpoint identity mismatch')
        if saved['revision'] != self.config['revision']:
            raise RuntimeError('Checkpoint model revision mismatch')
        self.case_calls.append(saved)
        if self.on_checkpoint:
            self.on_checkpoint()
        return ProviderResponse(saved['text'], saved['input_tokens'],
                                saved['output_tokens'], MODEL)
