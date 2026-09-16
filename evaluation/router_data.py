"""Step 3A TRAIN-only construction, provenance, scoring and inference-safe features."""
from __future__ import annotations
import argparse
from collections import Counter
from dataclasses import asdict
import gzip
import hashlib
import json
from pathlib import Path
import re
import string

from language.modal_checkpoint import atomic_json, digest
from spatial import parse, parse_context, execute
from spatial.identity import resolve
from spatial.normalization import answer_matches
from spatial.schema import OPERATIONS, EntityReference
from spatial.context_parser import ART_MARK, REG_MARK

BASE = Path(__file__).resolve().parents[1]
OUT = BASE/'experiments/E5_lightweight_adaptive/step3a_router_data'
SHORT = BASE/'experiments/E5_lightweight_adaptive/step2_neural_short'
CORE_SOURCES = BASE/'experiments/E5_lightweight_adaptive/step2_short_long/challenge/source_contexts.jsonl'
REGISTERS = ('natural_msa', 'conversational', 'light_saudi', 'indirect_compositional')
SEED = 'aasr-step3a-v1'


def read_jsonl(path):
    with Path(path).open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(query):
    return json.loads(json.dumps(asdict(query))) if query else None


def frozen_sources():
    # Read only freeze metadata; never open CORE EVAL or TEST data/predictions.
    prior = json.loads((BASE/'experiments/E5_lightweight_adaptive/step2_long/freeze_manifest.json').read_text())
    checks = {p: h for p, h in prior['frozen_step1_files'].items()}
    checks['language/long_parser.py'] = prior['long_prompt_source_sha256']
    checks['language/neural_short.py'] = prior['neural_source_sha256']
    checks[str((SHORT/'intent_bigru.pt').relative_to(BASE))] = prior['neural_checkpoint_sha256']
    config = json.loads((BASE/'experiments/E5_lightweight_adaptive/step2_long/modal_experiment.json').read_text())
    checks.update(config['code'])
    checks[str((SHORT/'model_config.json').relative_to(BASE))] = sha(SHORT/'model_config.json')
    for relative, expected in checks.items():
        if sha(BASE/relative) != expected:
            raise RuntimeError('Frozen source mismatch: '+relative)
    return checks


def verify_manifest():
    manifest = json.loads((OUT/'manifest.json').read_text())
    for relative, expected in manifest['frozen_sources'].items():
        if sha(BASE/relative) != expected:
            raise RuntimeError('Frozen source mismatch: '+relative)
    for name, expected in manifest['dataset_sha256'].items():
        if sha(OUT/name) != expected:
            raise RuntimeError('Dataset changed: '+name)
    return manifest


def slots(query):
    quote = lambda text: '«'+text+'»'
    value = {'a': quote(query.origin.name)}
    for key, target in zip(('e',) if len(query.targets)==1 else ('e1','e2'), query.targets):
        value[key] = quote(target.name)
    if len(query.categories)==1:
        value['c'] = query.categories[0]
    if len(query.categories)==2:
        value.update(c1=query.categories[0], c2=query.categories[1])
    if query.first_hop_category:
        value.update(c1=query.first_hop_category, c2=query.second_hop_category)
    if query.radius_km is not None:
        value['r'] = f'{query.radius_km:g} كم'
    if query.direction:
        value['d'] = query.direction
    return value


def compact_context(text):
    compact = text.partition(ART_MARK)[0]+ART_MARK+'\n[الخلفية النصية غير لازمة للاستدلال المكاني]\n'+REG_MARK+text.partition(REG_MARK)[2]
    if parse_context(compact) != parse_context(text):
        raise ValueError('Context compaction changed visible geometry')
    return compact


def build(train_path):
    train_path = Path(train_path).resolve()
    if train_path.name != 'train.jsonl.gz':
        raise ValueError('Only original train.jsonl.gz is permitted')
    if (OUT/'manifest.json').exists():
        verify_manifest()
        print('Existing frozen Step 3A dataset verified; no regeneration')
        return
    checks = frozen_sources()
    bank_path = Path(__file__).with_name('router_paraphrases.json')
    bank = json.loads(bank_path.read_text())
    assert set(bank)==set(OPERATIONS)
    assert all(set(b)==set(REGISTERS) and all(len(v)==4 for v in b.values()) for b in bank.values())
    core_ids = {r['source_record_id'] for r in read_jsonl(CORE_SOURCES)}
    with gzip.open(train_path, 'rt', encoding='utf-8') as stream:
        train = [json.loads(line) for line in stream]
    if any(r['split']!='train' for r in train):
        raise ValueError('Non-TRAIN source refused')
    matched = {r['id'] for r in train if r['id'] in core_ids}
    if matched != core_ids:
        raise RuntimeError('Cannot resolve every CORE source ID within TRAIN')
    excluded = {r['wikidata_id'] for r in train if r['id'] in core_ids}
    available = sorted({r['wikidata_id'] for r in train if r['wikidata_id'] not in excluded},
                       key=lambda a: digest([SEED,'anchor',a]))
    # Anchor assignment precedes record selection, register assignment and inference.
    train_anchors = set(available[:round(.8*len(available))])
    pools = {(task, split): [] for task in OPERATIONS for split in ('train','dev')}
    rejected = Counter()
    for row in train:
        if row['wikidata_id'] in excluded:
            continue
        try:
            context = parse_context(row['context'])
            query = parse(row['question'], context)
            result = execute(query, context)
            if result.status!='success' or not answer_matches(result.answer, row['answer']):
                rejected['original_executor_or_gold_mismatch'] += 1
                continue
            if query.operation != row['task_type']:
                raise ValueError('Original task/query mismatch')
            if any('\n' in n or n!=n.strip() for n in [query.origin.name]+[t.name for t in query.targets]):
                rejected['nonliteral_clean_name'] += 1
                continue
        except (ValueError, TypeError, KeyError):
            rejected['original_query_not_reconstructable'] += 1
            continue
        split = 'train' if row['wikidata_id'] in train_anchors else 'dev'
        pools[(query.operation,split)].append((row,query))
    items = []
    seen_questions = set()
    for task in OPERATIONS:
        for split, quota in (('train',240),('dev',60)):
            pool = sorted(pools[(task,split)],key=lambda pair:digest([SEED,task,pair[0]['id']]))
            if len(pool)<quota:
                raise RuntimeError(f'Insufficient eligible {task}/{split}: {len(pool)} < {quota}')
            accepted = 0
            for row,query in pool:
                if accepted == quota:
                    break
                index = accepted
                register = REGISTERS[index%4]
                template_index = (index//4)%4
                template = bank[task][register][template_index]
                fields = slots(query)
                used = {key for _,key,_,_ in string.Formatter().parse(template) if key}
                if used != set(fields):
                    raise ValueError(f'Template changed protected slot set: {task}/{register}/{template_index}')
                question = template.format(**fields)
                if question in seen_questions:
                    rejected['duplicate_rendered_question'] += 1
                    continue
                seen_questions.add(question)
                accepted += 1
                context_text = compact_context(row['context'])
                rid = 'router-'+digest([SEED,row['id'],register,template_index])[:20]
                items.append(dict(id=rid, source_split='train', source_record_id=row['id'],
                    source_anchor=row['wikidata_id'], router_split=split, task_type=task,
                    language_register=register, wording_family=f'{task}:{register}:{template_index}',
                    question=question, context=context_text, original_question=row['question'],
                    original_context_sha256=hashlib.sha256(row['context'].encode()).hexdigest(),
                    expected_structured_query=canonical(query), gold_answer=row['answer'],
                    semantic_verification='Protected slot equality and original frozen execution/gold agreement'))
            if accepted != quota:
                raise RuntimeError(f'Insufficient distinct questions for {task}/{split}: {accepted}')
    assert len(items)==2400 and len({r['id'] for r in items})==2400
    assert len({r['source_record_id'] for r in items})==2400
    assert len({r['question'] for r in items})==2400
    actual_train = {r['source_anchor'] for r in items if r['router_split']=='train'}
    actual_dev = {r['source_anchor'] for r in items if r['router_split']=='dev'}
    assert not actual_train & actual_dev and not (actual_train|actual_dev)&excluded
    OUT.mkdir(parents=True,exist_ok=True)
    write_jsonl(OUT/'questions.jsonl',items)
    write_jsonl(OUT/'public_inputs.jsonl',[{k:r[k] for k in ('id','question','context')} for r in items])
    atomic_json(OUT/'anchor_split.json',dict(excluded_core_anchors=sorted(excluded),
        assigned_train_anchors=sorted(train_anchors),assigned_dev_anchors=sorted(set(available)-train_anchors),
        represented_train_anchors=sorted(actual_train),represented_dev_anchors=sorted(actual_dev)))
    atomic_json(OUT/'paraphrase_bank.json',bank)
    manifest = dict(stage='3A',seed=SEED,source_train_path=str(train_path),source_train_sha256=sha(train_path),
        core_source_ids_sha256=sha(CORE_SOURCES),excluded_core_anchors=len(excluded),n=2400,
        counts_by_split=dict(Counter(r['router_split'] for r in items)),
        counts_by_task=dict(Counter(r['task_type'] for r in items)),
        counts_by_register=dict(Counter(r['language_register'] for r in items)),
        represented_train_anchors=len(actual_train),represented_dev_anchors=len(actual_dev),
        anchor_disjoint=True,unique_source_records=2400,source_rejections=dict(rejected),
        source_policy='Original TRAIN only, excluding every CORE source anchor; only exactly reconstructable original gold cases.',
        split_policy='80/20 eligible anchor assignment before sampling; exact 1920/480 question quotas, 60/15 per task/register cell.',
        wording_policy='128 new agent-authored generic Arabic templates (4 per task/register), instantiated without semantic slot changes. Not user logs or independently human-reviewed.',
        frozen_sources=checks,
        dataset_sha256={name:sha(OUT/name) for name in ('questions.jsonl','public_inputs.jsonl','anchor_split.json','paraphrase_bank.json')})
    atomic_json(OUT/'manifest.json',manifest)
    print(json.dumps({k:v for k,v in manifest.items() if k not in ('frozen_sources','dataset_sha256')},ensure_ascii=False,indent=2))


def safe_features(question, context, prediction, probabilities):
    import math
    query = prediction.query
    # The frozen extractor raises before returning partial slots. Do not invent
    # a missing-slot count; expose only its documented unresolved slot group.
    slot_errors = {'count_origin_not_explicit_unique':['origin'], 'origin_not_explicit':['origin'],
        'wrong_named_target_count':['targets'],'category_slot_count':['categories'],
        'two_category_slot_count':['categories'],'missing_or_conflicting_radius':['radius_km'],
        'direction_slot_count':['direction']}
    missing = [] if query else slot_errors.get(prediction.reason)
    refs = (query.origin,)+query.targets if query else tuple(
        EntityReference(name,'unspecified') for name in sorted(
            {p.name for p in context.candidates}|({context.anchor.name} if context.anchor else set())) if name in question)
    ambiguous = sum(resolve(ref,context).status=='AMBIGUOUS' for ref in refs)
    ranked = sorted(probabilities,reverse=True)
    constraint_count = (len(query.targets)+len(query.categories)+int(query.radius_km is not None)+
        int(query.direction is not None)+int(query.first_hop_category is not None)+int(query.second_hop_category is not None)) if query else None
    return dict(short_confidence=float(prediction.confidence),
        short_entropy=-sum(p*math.log(p) for p in probabilities if p>0),
        short_top1_top2_margin=ranked[0]-ranked[1],query_valid=query is not None,
        missing_slots=missing,entity_ambiguity_count=ambiguous,question_length_chars=len(question),
        question_length_words=len(question.split()),predicted_operation=prediction.operation,
        constraint_count=constraint_count)


def score(query, execution, item):
    return dict(query_exact=canonical(query)==item['expected_structured_query'],
        answer_correct=bool(execution and execution.status=='success' and answer_matches(execution.answer,item['gold_answer'])))


def label(short_correct,long_correct):
    return 'SHORT' if short_correct else 'LONG' if long_correct else 'FAILURE'


if __name__=='__main__':
    cli=argparse.ArgumentParser()
    cli.add_argument('--train-path',required=True,type=Path)
    build(cli.parse_args().train_path)
