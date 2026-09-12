#!/usr/bin/env python3
"""Build TRAIN-only controlled Arabic paraphrases; never change frozen runtime."""
from __future__ import annotations
import csv
from dataclasses import asdict
import gzip
import hashlib
import json
from pathlib import Path
import string
import sys

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
from spatial import execute, parse, parse_context
from spatial.normalization import answer_matches
from spatial.schema import OPERATIONS

OUT = BASE/'experiments/E5_lightweight_adaptive/step2_short_long'
# Disjoint wording families and anchor groups; fixed before parser experiments.
DEV_INDICES = {0,1,2,4,5,7,8,10,12,14,16,17,19,20,22,24,26,28,29,31,32,34,36,38}
PLURALS={'مستشفى':'المستشفيات','عيادة':'العيادات','صيدلية':'الصيدليات','مدرسة':'المدارس','جامعة':'الجامعات','كلية':'الكليات','مطعم':'المطاعم','مقهى':'المقاهي','بنك':'البنوك','محطة وقود':'محطات الوقود','مطعم وجبات سريعة':'مطاعم الوجبات السريعة','صراف آلي':'الصرافات الآلية','روضة أطفال':'رياض الأطفال','مكان عبادة':'أماكن العبادة'}
WORDS = {1:'كيلومتر واحد',1.5:'كيلومتر ونصف',2:'كيلومترين',3:'ثلاثة كيلومترات',
         5:'خمسة كيلومترات',8:'ثمانية كيلومترات'}


def write_jsonl(path, rows):
    with path.open('w',encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row,ensure_ascii=False)+'\n')


def read_jsonl(path):
    with path.open(encoding='utf-8') as f:
        return [json.loads(line) for line in f]


def frozen_hashes():
    prior=json.loads((BASE/'experiments/E5_lightweight_adaptive/step1_structured_baseline/metrics.json').read_text())['runtime_source_sha256']
    for name,expected in prior.items():
        if hashlib.sha256((BASE/name).read_bytes()).hexdigest()!=expected:
            raise RuntimeError('Frozen Step-1 source changed: '+name)
    return prior


def train_records(path=None):
    path=path or BASE/'data/raw/train.jsonl.gz'
    if path.name!='train.jsonl.gz':
        raise ValueError('Only original TRAIN source may be used')
    with gzip.open(path,'rt',encoding='utf-8') as f:
        for line in f:
            row=json.loads(line)
            if row['split']!='train':
                raise ValueError('Non-train row refused')
            yield row


def register_level(i):
    register='natural_msa' if i<16 else 'conversational' if i<28 else 'light_saudi' if i<36 else 'indirect_compositional'
    level='L1' if i<8 or 28<=i<32 else 'L2' if 16<=i<24 or 32<=i<36 else 'L3' if 8<=i<12 or 24<=i<28 else 'L4'
    return register,level


def slots_for(query,i):
    quote=lambda s:'«'+s+'»'
    fields={'a':quote(query.origin.name)}
    for k,target in zip(('e',) if len(query.targets)==1 else ('e1','e2'),query.targets):
        fields[k]=quote(target.name)
    if len(query.categories)==1:
        fields['c']=query.categories[0]
        fields['cp']=PLURALS[query.categories[0]]
    if len(query.categories)==2:
        fields.update(c1=query.categories[0],c2=query.categories[1],p1=PLURALS[query.categories[0]],p2=PLURALS[query.categories[1]])
    if query.first_hop_category:fields.update(c1=query.first_hop_category,c2=query.second_hop_category)
    if query.radius_km is not None:
        # Explicit equivalent unit expressions only; no perturbation or rounding.
        fields['r']=WORDS[query.radius_km] if i%2 else f'{query.radius_km:g} كم'
    if query.direction:fields['d']=query.direction
    return fields


def semantic_check(item,source,template,i):
    ctx=parse_context(source['context']); expected=parse(source['question'],ctx)
    if json.dumps(asdict(expected),sort_keys=True)!=json.dumps(item['expected_structured_query'],sort_keys=True):
        raise ValueError('Expected query changed')
    if item['source_record_id']!=source['id'] or source['split']!='train':
        raise ValueError('Wrong source')
    fields=slots_for(expected,i)
    used={key for _,key,_,_ in string.Formatter().parse(template) if key}
    canonical=lambda key:{'cp':'c','p1':'c1','p2':'c2'}.get(key,key)
    if {canonical(key) for key in used}!={canonical(key) for key in fields}:
        raise ValueError('Template omitted/added a semantic slot')
    if item['challenge_question']!=template.format(**fields):
        raise ValueError('Unreviewed challenge wording change')
    for key in used:
        if fields[key] not in item['challenge_question']:
            raise ValueError('Lost semantic slot')
    result=execute(expected,ctx)
    if result.status!='success' or not answer_matches(result.answer,item['gold_answer']):
        raise ValueError('Frozen execution does not preserve gold')
    if item['gold_answer']!=source['answer']:
        raise ValueError('Gold modified')
    return {'same_origin':True,'same_targets':True,'same_categories':True,'same_radius':True,
            'same_direction':True,'same_hops':True,'same_constraints':True,'same_frozen_answer':True}


def main():
    hashes=frozen_hashes()
    bank=json.loads((OUT/'challenge/paraphrase_bank.json').read_text())
    assert set(bank)==set(OPERATIONS) and all(len(v)==40 for v in bank.values())
    with (BASE/'experiments/E5_lightweight_adaptive/data_audit/reconstructability_records.csv').open() as f:
        eligible={r['id'] for r in csv.DictReader(f) if r['split']=='train' and r['class']=='A'}
    selected={}; seen_anchors=set(); skipped=0
    for row in train_records():
        if row['id'] not in eligible:continue
        # Offline grouping provenance only. Never passed to either language parser.
        anchor=row['wikidata_id']; split='dev' if int(hashlib.sha256(anchor.encode()).hexdigest(),16)%5<3 else 'eval'
        task=row['task_type']; key=(task,split)
        indices=[i for i in range(40) if (i in DEV_INDICES)==(split=='dev')]
        if len(selected.get(key,[]))>=len(indices) or anchor in seen_anchors:continue
        q=parse(row['question'],row['context'])
        # Names must remain visible verbatim; no ugly multiline name editing.
        names=[q.origin.name]+[r.name for r in q.targets]
        if any('\n' in name or name!=name.strip() for name in names):
            skipped+=1;continue
        selected.setdefault(key,[]).append(row);seen_anchors.add(anchor)
        if sum(map(len,selected.values()))==320:break
    if sum(map(len,selected.values()))!=320:
        raise RuntimeError('Insufficient distinct eligible TRAIN anchors')
    items=[];contexts=[];review=[]
    for task in OPERATIONS:
        for split in ('dev','eval'):
            indices=[i for i in range(40) if (i in DEV_INDICES)==(split=='dev')]
            for i,row in zip(indices,selected[(task,split)]):
                q=parse(row['question'],row['context']);reg,level=register_level(i)
                question=bank[task][i].format(**slots_for(q,i))
                cid=f'core-{task}-{i+1:02d}'
                item={'challenge_id':cid,'source_record_id':row['id'],'task_type':task,
                      'original_question':row['question'],'challenge_question':question,
                      'language_register':reg,'complexity_level':level,'challenge_split':split,
                      'expected_structured_query':asdict(q),'gold_answer':row['answer'],
                      'review_status':'agent_reviewed_and_structurally_validated',
                      'wording_family':f'{task}:{i+1:02d}'}
                checks=semantic_check(item,row,bank[task][i],i)
                item['semantic_checks']=checks
                items.append(item)
                # Compact, inference-only context: preserve original visible anchor,
                # registry and exact coordinate strings; omit irrelevant article.
                from spatial.context_parser import ART_MARK,REG_MARK
                before=row['context'].partition(ART_MARK)[0]
                registry=row['context'].partition(REG_MARK)[2]
                compact=before+ART_MARK+'\n[خلفية المقال محذوفة من نسخة الاستدلال المختصرة]\n'+REG_MARK+registry
                assert parse_context(compact)==parse_context(row['context'])
                contexts.append({'source_record_id':row['id'],'context':compact,
                                 'original_context_sha256':hashlib.sha256(row['context'].encode()).hexdigest()})
                review.append({'challenge_id':cid,'status':'accepted','reason':'All protected slots preserved; author semantic reading of wording family; frozen expected query agrees with source answer',
                               'independent_human_review':False})
    assert len({i['challenge_question'] for i in items})==320
    write_jsonl(OUT/'challenge/core_challenge.jsonl',items)
    for split in ('dev','eval'):write_jsonl(OUT/f'challenge/core_{split}.jsonl',[i for i in items if i['challenge_split']==split])
    write_jsonl(OUT/'challenge/source_contexts.jsonl',contexts)
    with (OUT/'challenge/generation_review.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(review[0]));w.writeheader();w.writerows(review)
    manifest={'count':320,'dev':192,'eval':128,'unique_source_records':320,'unique_source_anchors':320,
              'source':'TRAIN only; audit class A','frozen_step1_sha256':hashes,
              'bank_sha256':hashlib.sha256((OUT/'challenge/paraphrase_bank.json').read_bytes()).hexdigest(),
              'accepted':320,'rejected_paraphrases':0,'revised_paraphrases':0,
              'source_names_skipped_for_multiline_or_edge_whitespace':skipped,
              'review_limit':'Author/agent semantic review plus slot and execution checks; not independent native-speaker review. Constructed paraphrases, not real user logs.',
              'source_selection':'First eligible distinct TRAIN anchors within deterministic hash partition; not a random representative sample.',
              'split_policy':'Anchor-disjoint and wording-family-disjoint; source IDs also disjoint. All split/selection decisions fixed before Long calls.'}
    (OUT/'challenge/manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in manifest.items() if k!='frozen_step1_sha256'},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
