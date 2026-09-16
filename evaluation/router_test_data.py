"""Final TEST preparation and integrity checks; no training or data repair."""
import gzip
import json
import hashlib
from pathlib import Path
from collections import Counter
from language.modal_checkpoint import atomic_json, digest, MODEL, GENERATION

BASE=Path(__file__).resolve().parents[1]
ROOT=BASE/'experiments/E5_lightweight_adaptive'
ROUTER=ROOT/'router_v2'
OUT=ROOT/'router_v2_test'
SHORT=ROOT/'step2_neural_short'
TEST=Path('/Users/joudalrubaish/Tuwaiq-AI /ArabicSpatialReasoning/split_dataset/test.jsonl.gz')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def rows(path):return [json.loads(line) for line in Path(path).read_text().splitlines()]
def write_rows(path,data):Path(path).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in data))


def verify():
    frozen=read(OUT/'pre_test_freeze.json')
    assert sha(TEST)==frozen['test_sha256']
    for path,value in frozen['sha256'].items():assert sha(BASE/path)==value,path
    assert sha(OUT/'public_inputs.jsonl')==frozen['public_sha256']
    assert sha(OUT/'test_records.jsonl')==frozen['records_sha256']
    assert sha(OUT/'inference_config.json')==frozen['config_sha256']
    return frozen


def prepare():
    from evaluation.router_data import compact_context
    if OUT.exists():raise RuntimeError('TEST output already exists; do not prepare again')
    router_freeze=read(ROUTER/'freeze_manifest.json')
    assert router_freeze['selected_model']=='small_mlp' and router_freeze['threshold']==.91
    checks=read(ROUTER/'input_freeze.json')['sha256']
    checks.update(router_freeze['code_sha256'])
    for path,value in router_freeze['output_sha256'].items():checks[str((ROUTER/path).relative_to(BASE))]=value
    checks[str((ROUTER/'freeze_manifest.json').relative_to(BASE))]=sha(ROUTER/'freeze_manifest.json')
    for path,value in checks.items():assert sha(BASE/path)==value,path
    with gzip.open(TEST,'rt',encoding='utf-8') as stream:data=[json.loads(line) for line in stream]
    assert len(data)==2332 and len({r['id'] for r in data})==2332
    assert all(r['split']=='test' for r in data)
    dev=rows(ROOT/'step3a_router_data/router_dataset.jsonl')
    assert not {r['wikidata_id'] for r in data}&{r['provenance']['source_anchor'] for r in dev}
    public=[dict(id=r['id'],question=r['question'],context=compact_context(r['context'])) for r in data]
    OUT.mkdir()
    write_rows(OUT/'test_records.jsonl',data);write_rows(OUT/'public_inputs.jsonl',public)
    test_hash=sha(TEST)
    config=dict(model=MODEL,revision='1cfa9a7208912126459214e8b04321603b3df60c',generation=GENERATION,
        stage='frozen_router_v2_final_test',test_sha256=test_hash,
        prompt_sha256=sha(BASE/'language/long_parser.py'))
    atomic_json(OUT/'inference_config.json',config)
    atomic_json(OUT/'pre_test_freeze.json',dict(n=2332,test_sha256=test_hash,sha256=checks,
        public_sha256=sha(OUT/'public_inputs.jsonl'),records_sha256=sha(OUT/'test_records.jsonl'),
        config_sha256=sha(OUT/'inference_config.json'),router_threshold=.91,
        remote_namespace='router_v2_test_'+test_hash[:16],
        policy='All original TEST rows retained, no repairs or exclusions; models/features/calibration/threshold frozen; exact answer matching via frozen comparator',
        routing_metric_policy='No native gold StructuredQuery; answer-based SHORT preference, LONG rescue, FAILURE both wrong; exclude FAILURE only from binary routing metrics',
        breakdown_policy='Native task_type and difficulty only; no language_register field',
        original_geo_parsed_context_preserved=True,test_dev_anchor_disjoint=True))
    print(json.dumps(dict(n=len(data),anchors=len({r['wikidata_id'] for r in data}),public_bytes=(OUT/'public_inputs.jsonl').stat().st_size,
        tasks=dict(Counter(r['task_type'] for r in data)),difficulty=dict(Counter(r['difficulty'] for r in data))),ensure_ascii=False))


if __name__=='__main__':prepare()
