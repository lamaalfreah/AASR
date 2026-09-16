"""Router V2 input freeze: only existing Router development data, never CORE/TEST."""
import hashlib
import json
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
SOURCE = BASE/'experiments/E5_lightweight_adaptive/step3a_router_data'
OUT = BASE/'experiments/E5_lightweight_adaptive/router_v2'
SHORT = BASE/'experiments/E5_lightweight_adaptive/step2_neural_short'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def write(path,value):
    path=Path(path); temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    temp.replace(path)


def verify():
    frozen=read(OUT/'input_freeze.json')
    for path,expected in frozen['sha256'].items():
        assert sha(BASE/path)==expected,'Frozen input changed: '+path
    return frozen


def prepare():
    if OUT.exists():
        raise RuntimeError('Router V2 directory exists; do not recreate inputs')
    data=rows(SOURCE/'router_train.jsonl')+rows(SOURCE/'router_dev.jsonl')
    assert len(data)==2400 and len({r['id'] for r in data})==2400
    assert all(r['provenance']['source_split']=='train' for r in data)
    excluded=set(read(SOURCE/'anchor_split.json')['excluded_core_anchors'])
    assert not {r['provenance']['source_anchor'] for r in data}&excluded
    checks=read(SOURCE/'manifest.json')['frozen_sources']
    checks.update(read(SOURCE/'pipeline_freeze.json')['code_sha256'])
    for path,digest in checks.items():assert sha(BASE/path)==digest,path
    for path in SOURCE.rglob('*'):
        if path.is_file():checks[str(path.relative_to(BASE))]=sha(path)
    OUT.mkdir()
    write(OUT/'input_freeze.json',dict(sha256=checks,n=2400,
        policy='Merge Router TRAIN/DEV for anchor-grouped development only. No CORE EVAL or TEST reads.',
        excluded_core_anchors=len(excluded),new_questions=0))
    print('Router V2: 2400 frozen development rows; existing Short and Long outcomes retained')


if __name__=='__main__':prepare()
