#!/usr/bin/env python3
"""Frozen Short vs exact-model Long. CORE EVAL never feeds prompts or retries."""
from __future__ import annotations
import argparse
from collections import Counter
import csv
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
from time import perf_counter
BASE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(BASE))
from evaluation.build_language_challenge import OUT,read_jsonl,frozen_hashes
from spatial import parse,parse_context,execute
from spatial.schema import QueryError,Location
from spatial.normalization import answer_matches
from spatial.extensions import SiteQuery,execute_site_query
from language.long_parser import LongParser,PROMPT
from language.qwen_provider import QwenProvider,ProviderUnavailable,MODEL


def csv_write(path,rows,fields):
    with path.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)


def pct(n,d):return round(100*n/d,6) if d else None


def latency(values):
    if not values:return {'mean_ms':None,'p50_ms':None,'p95_ms':None}
    a=sorted(values)
    def p(q):
        x=(len(a)-1)*q;i=int(x)
        return a[i]+(a[min(i+1,len(a)-1)]-a[i])*(x-i)
    return {'mean_ms':statistics.mean(a),'p50_ms':p(.5),'p95_ms':p(.95)}


def short_infer(question,context):
    began=perf_counter();query=None
    try:
        query=parse(question,context);result=execute(query,context)
        status=result.status;answer=result.answer
    except QueryError as exc:status=exc.status;answer=None
    return query,status,answer,(perf_counter()-began)*1000


def score_short(item,context):
    query,status,answer,ms=short_infer(item['challenge_question'],context)
    return {'short_parse_success':query is not None,
            'short_query_exact':query is not None and json.loads(json.dumps(asdict(query)))==item['expected_structured_query'],
            'short_correct':answer is not None and status=='success' and answer_matches(answer,item['gold_answer']),
            'short_status':status,'short_latency_ms':ms}


def score_long(item,context,parser):
    began=perf_counter()
    # Only explicit safe inputs enter the parser; no item dictionary passes through.
    outcome=parser.parse(item['challenge_question'],context)
    query=outcome.query
    result=execute(query,context) if query is not None else None
    status=result.status if result else outcome.status
    return {'long_parse_success':query is not None,
            'long_query_exact':query is not None and json.loads(json.dumps(asdict(query)))==item['expected_structured_query'],
            'long_correct':bool(result and result.status=='success' and answer_matches(result.answer,item['gold_answer'])),
            'long_status':status,'long_latency_ms':(perf_counter()-began)*1000,
            'long_parser_latency_ms':outcome.latency_ms,'llm_calls':outcome.llm_calls,
            'input_tokens':outcome.input_tokens,'output_tokens':outcome.output_tokens,
            'invalid_responses':outcome.invalid_responses}


def summarize(rows,prefix):
    available=[r for r in rows if r.get(prefix+'_correct') is not None]
    n=len(available)
    if not n:return {'evaluated':0,'status':'not_run','accuracy_percent':None,'query_exact_percent':None,
                     'parse_success_percent':None,'abstention_percent':None,**latency([])}
    return {'evaluated':n,'status':'evaluated','correct':sum(r[prefix+'_correct'] for r in available),
            'accuracy_percent':pct(sum(r[prefix+'_correct'] for r in available),n),
            'query_exact_percent':pct(sum(r[prefix+'_query_exact'] for r in available),n),
            'parse_success_percent':pct(sum(r[prefix+'_parse_success'] for r in available),n),
            'abstention_percent':pct(sum(r[prefix+'_status']!='success' for r in available),n),
            'status_counts':dict(Counter(r[prefix+'_status'] for r in available)),
            **latency([r[prefix+'_latency_ms'] for r in available])}


def paired(rows):
    evaluated=[r for r in rows if r.get('long_correct') is not None]
    shortfail=sum(not r['short_correct'] for r in evaluated)
    shortok=len(evaluated)-shortfail
    rescue=sum(not r['short_correct'] and r['long_correct'] for r in evaluated)
    regress=sum(r['short_correct'] and not r['long_correct'] for r in evaluated)
    return {'paired_evaluated':len(evaluated),'short_wrong_long_correct':rescue if evaluated else None,
            'opportunity_percent':pct(rescue,len(evaluated)), 'long_rescue_percent':pct(rescue,shortfail),
            'long_regression_percent':pct(regress,shortok),
            'long_regression_denominator':shortok,
            'outcomes':dict(Counter(r['outcome'] for r in evaluated))}


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--live-long',action='store_true')
    args=cli.parse_args()
    frozen_hashes()
    sources={r['source_record_id']:parse_context(r['context']) for r in read_jsonl(OUT/'challenge/source_contexts.jsonl')}
    items=read_jsonl(OUT/'challenge/core_challenge.jsonl')
    result_dir=OUT/'results';result_dir.mkdir(exist_ok=True)
    manifest_hash=hashlib.sha256((OUT/'challenge/core_challenge.jsonl').read_bytes()).hexdigest()
    prompt_hash=hashlib.sha256(PROMPT.encode()).hexdigest()
    availability={'requested_model':MODEL,'account_region_verified':False,'status':'not_verified',
                  'reason':'Live evaluation not requested'}
    parser=None
    if args.live_long:
        try:
            provider=QwenProvider()
            availability=provider.verify_model();availability['status']='verified'
            parser=LongParser(provider)
        except ProviderUnavailable as exc:
            availability['reason']=str(exc);availability['status']='blocked'
    (result_dir/'provider_availability.json').write_text(json.dumps(availability,indent=2)+'\n')
    # Allow interruption recovery without replaying paid calls. Cache is keyed to
    # exact challenge, prompt and parser/provider source hashes; no cached targets.
    code_hash=hashlib.sha256(b''.join((BASE/'language'/p).read_bytes() for p in ('long_parser.py','qwen_provider.py'))).hexdigest()
    cache_path=result_dir/'long_progress.jsonl';cache={}
    if cache_path.exists():
        for r in read_jsonl(cache_path):
            if (r['challenge_sha256'],r['prompt_sha256'],r['code_sha256'])!=(manifest_hash,prompt_hash,code_hash):
                raise RuntimeError('Existing Long results belong to a different frozen experiment')
            cache[r['challenge_id']]=r['scores']
    rows=[]
    for item in sorted(items,key=lambda r:(r['challenge_split']!='dev',r['challenge_id'])):
        row={k:item[k] for k in ('challenge_id','task_type','language_register','complexity_level','challenge_split')}
        ctx=sources[item['source_record_id']]
        row.update(score_short(item,ctx))
        if item['challenge_id'] in cache:
            row.update(cache[item['challenge_id']])
        elif parser:
            try:
                score=score_long(item,ctx,parser)
                with cache_path.open('a',encoding='utf-8') as f:
                    f.write(json.dumps({'challenge_id':item['challenge_id'],'challenge_sha256':manifest_hash,
                        'prompt_sha256':prompt_hash,'code_sha256':code_hash,'scores':score})+'\n')
                row.update(score)
            except ProviderUnavailable as exc:
                availability.update(status='blocked',reason=str(exc));parser=None
                (result_dir/'provider_availability.json').write_text(json.dumps(availability,indent=2)+'\n')
        if 'long_correct' not in row:
            row.update({k:None for k in ('long_parse_success','long_query_exact','long_correct','long_latency_ms',
                       'long_parser_latency_ms','llm_calls','input_tokens','output_tokens','invalid_responses')})
            row['long_status']='not_run'
        row['outcome']=('short_'+('correct' if row['short_correct'] else 'wrong')+'/long_'+
                        ('correct' if row['long_correct'] else 'wrong')) if row['long_correct'] is not None else 'not_evaluated'
        rows.append(row)
        if parser and len(rows)%16==0:print(f'Long complete: {len(rows)}/320',flush=True)
    csv_write(result_dir/'short_vs_long.csv',rows,list(rows[0]))
    prior=json.loads((BASE/'experiments/E5_lightweight_adaptive/step1_structured_baseline/metrics.json').read_text())
    short={'original_validation':prior['splits']['validation'],
           'original_validation_provenance':'Reused frozen Step-1 full-split execution. No re-audit. Independent full structured-query EM labels do not exist for all original records; parse and operation agreement are 100%, query EM is not claimed.',
           'core':{split:summarize([r for r in rows if r['challenge_split']==split],'short') for split in ('dev','eval')},
           'llm_calls':0,'core_timing_scope':'Pre-parsed immutable contexts shared by both parsers. Short latency includes query parsing and execution if parsed; Long includes request, retry, validation and execution. Original-validation timing includes context parsing and is not directly comparable.'}
    long={'model':MODEL,'provider_status':availability,'core':{split:summarize([r for r in rows if r['challenge_split']==split],'long') for split in ('dev','eval')},
          'paired':{split:paired([r for r in rows if r['challenge_split']==split]) for split in ('dev','eval')},
          'challenge_sha256':manifest_hash,'prompt_sha256':prompt_hash,'code_sha256':code_hash,
          'llm_calls':sum(r['llm_calls'] or 0 for r in rows),
          'input_tokens':sum(r['input_tokens'] for r in rows) if all(r['input_tokens'] is not None for r in rows) else None,
          'output_tokens':sum(r['output_tokens'] for r in rows) if all(r['output_tokens'] is not None for r in rows) else None,
          'actual_provider_cost':None,'estimated_provider_cost':None,'cost_note':'No invented prices; account/region-specific rates and usage required.',
          'latency_ratio':None,'verdict':'ROUTER NOT YET JUSTIFIED'}
    if all(r['long_correct'] is not None for r in rows):
        long['latency_ratio']=statistics.mean(r['long_latency_ms'] for r in rows)/statistics.mean(r['short_latency_ms'] for r in rows)
        if paired(rows)['short_wrong_long_correct']:
            long['verdict']='ROUTER PROMISING BUT NEEDS MORE DATA'
    for name,metric in [('short_metrics',short),('long_metrics',long)]:
        (result_dir/(name+'.json')).write_text(json.dumps(metric,ensure_ascii=False,indent=2)+'\n')
    for dimension,file in [('task_type','per_task'),('complexity_level','per_complexity'),('language_register','per_register')]:
        summaries=[]
        for split in ('dev','eval'):
            for value in sorted({r[dimension] for r in rows}):
                subset=[r for r in rows if r[dimension]==value and r['challenge_split']==split]
                if not subset:continue
                ss,ll=summarize(subset,'short'),summarize(subset,'long')
                summaries.append(dict(split=split,group=value,n=len(subset),short_accuracy=ss['accuracy_percent'],
                   long_accuracy=ll['accuracy_percent'],short_parse=ss['parse_success_percent'],long_parse=ll['parse_success_percent'],
                   short_query_exact=ss['query_exact_percent'],long_query_exact=ll['query_exact_percent'],
                   short_abstention=ss['abstention_percent'],long_abstention=ll['abstention_percent'],
                   **{k:v for k,v in paired(subset).items() if k!='outcomes'}))
        csv_write(result_dir/(file+'.csv'),summaries,list(summaries[0]))
    extension=[]
    for item in read_jsonl(OUT/'challenge/extension_challenge.jsonl'):
        ctx=item['inference_context'];q=dict(item['expected_extension_query']);q['required_data']=tuple(q['required_data'])
        result=execute_site_query(SiteQuery(**q),tuple(Location(**p) for p in ctx['supplied_sites']),
                                  tuple(Location(**p) for p in ctx['listed_facilities']),Location(**ctx['anchor']))
        extension.append({'challenge_id':item['challenge_id'],'intent_kind':item['intent_kind'],
                          'status':result.status,'reason':result.reason,'selected_ids':'|'.join(result.selected_ids),
                          'expected_contract_agrees':result.status==item['expected_status'] and list(result.selected_ids)==item['expected_selected_ids'],
                          'evaluation_mode':'oracle_intent_contract_only_not_language_accuracy'})
    csv_write(result_dir/'extension_results.csv',extension,list(extension[0]))
    frozen_hashes()
    print(json.dumps({'short_core':short['core'],'long_status':availability,'extension_statuses':dict(Counter(r['status'] for r in extension))},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
