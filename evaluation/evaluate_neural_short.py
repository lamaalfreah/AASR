#!/usr/bin/env python3
"""Evaluate fixed neural Short once on validation, CORE DEV, CORE EVAL; no TEST."""
from __future__ import annotations
from collections import Counter
from dataclasses import asdict
import csv,gzip,hashlib,json,statistics,sys,time
from pathlib import Path
BASE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(BASE))
from language.neural_short import NeuralShortParser,tokenize
from spatial import parse_context,execute,parse as rule_baseline_0
from spatial.context_parser import parse_context_fields
from spatial.normalization import answer_matches
from evaluation.build_language_challenge import frozen_hashes,OUT as CHALLENGE
OUT=BASE/'experiments/E5_lightweight_adaptive/step2_neural_short'


def read_jsonl(path):
    with path.open(encoding='utf-8') as f:
        for line in f:yield json.loads(line)


def csv_write(path,rows):
    if not rows:return
    with path.open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def canonical(q):return json.loads(json.dumps(asdict(q)))

def pct(n,d):return round(100*n/d,6) if d else None


def timing(values):
    a=sorted(values)
    def p(q):
        x=(len(a)-1)*q;i=int(x)
        return a[i]+(a[min(i+1,len(a)-1)]-a[i])*(x-i)
    return {'mean_ms':statistics.mean(a),'p50_ms':p(.5),'p95_ms':p(.95)}


def summary(rows):
    n=len(rows);recon=[r for r in rows if r['reconstructable']]
    return {'n':n,'intent_correct':sum(r['intent_correct'] for r in rows),
            'intent_accuracy_percent':pct(sum(r['intent_correct'] for r in rows),n),
            'query_exact_correct':sum(r['query_exact'] for r in rows),
            'structured_query_exact_match_percent':pct(sum(r['query_exact'] for r in rows),n),
            'answer_correct':sum(r['answer_correct'] for r in rows),
            'end_to_end_accuracy_percent':pct(sum(r['answer_correct'] for r in rows),n),
            'query_produced':sum(r['query_produced'] for r in rows),
            'query_produced_percent':pct(sum(r['query_produced'] for r in rows),n),
            'abstentions':sum(r['status']!='success' for r in rows),
            'abstention_percent':pct(sum(r['status']!='success' for r in rows),n),
            'reconstructable_n':len(recon),'reconstructable_accuracy_percent':pct(sum(r['answer_correct'] for r in recon),len(recon)),
            'failure_counts':dict(Counter(r['failure_stage'] for r in rows if not r['answer_correct'])),
            'pipeline_latency':timing([r['pipeline_ms'] for r in rows]),
            'parser_latency':timing([r['parser_ms'] for r in rows])}


def public_inference(parser,question,context_text):
    started=time.perf_counter();ctx=parse_context(context_text);t=time.perf_counter()
    prediction=parser.predict(question,ctx);parser_ms=(time.perf_counter()-t)*1000
    result=execute(prediction.query,ctx) if prediction.query is not None else None
    return ctx,prediction,result,(time.perf_counter()-started)*1000,parser_ms


def main():
    import torch
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    frozen_hashes()
    train=json.loads((OUT/'training_metrics.json').read_text())
    source_hash=hashlib.sha256((BASE/'language/neural_short.py').read_bytes()).hexdigest()
    if source_hash!=train['short_source_sha256']:raise RuntimeError('Short source changed after training freeze')
    if (OUT/'metrics.json').exists():raise RuntimeError('Evaluation already exists; refuse accidental repeated held-out evaluation')
    challenge_hashes={name:hashlib.sha256((CHALLENGE/'challenge'/name).read_bytes()).hexdigest()
                      for name in ('core_dev.jsonl','core_eval.jsonl','core_challenge.jsonl','source_contexts.jsonl')}
    load_started=time.perf_counter();parser=NeuralShortParser(OUT);load_seconds=time.perf_counter()-load_started
    with (BASE/'experiments/E5_lightweight_adaptive/data_audit/reconstructability_records.csv').open() as f:
        recon={r['id']:r['class']=='A' for r in csv.DictReader(f) if r['split']=='validation'}
    contexts={r['source_record_id']:r['context'] for r in read_jsonl(CHALLENGE/'challenge/source_contexts.jsonl')}
    all_rows=[];scope_metrics={};predictions=[]
    for scope in ('original_validation','core_dev','core_eval'):
        parse_context_fields.cache_clear();scope_start=time.perf_counter();rows=[]
        if scope=='original_validation':
            stream=gzip.open(BASE/'data/raw/validation.jsonl.gz','rt',encoding='utf-8')
            iterator=(json.loads(line) for line in stream)
        else:
            stream=None;iterator=read_jsonl(CHALLENGE/'challenge'/(scope+'.jsonl'))
        for item in iterator:
            original=scope=='original_validation'
            question=item['question'] if original else item['challenge_question']
            text=item['context'] if original else contexts[item['source_record_id']]
            # Runtime takes only question + context; read labels for scoring AFTER it.
            ctx,prediction,result,pipeline_ms,parser_ms=public_inference(parser,question,text)
            expected=canonical(rule_baseline_0(question,ctx)) if original else item['expected_structured_query']
            intent_correct=prediction.operation==item['task_type']
            query_exact=prediction.query is not None and canonical(prediction.query)==expected
            gold=item['answer'] if original else item['gold_answer']
            correct=bool(result and result.status=='success' and answer_matches(result.answer,gold))
            status=result.status if result else prediction.status
            reason=(result.reason if result else prediction.reason) or ''
            stage=('none' if correct else 'intent' if not intent_correct else 'slot_extraction' if prediction.query is None
                   else 'structured_query' if not query_exact else 'identity' if result and result.stage=='identity'
                   else 'executor_or_label')
            row={'scope':scope,'id':item['id'] if original else item['challenge_id'],'task_type':item['task_type'],
                 'language_register':'original_template' if original else item['language_register'],
                 'complexity_level':'original_template' if original else item['complexity_level'],
                 'predicted_intent':prediction.operation,'confidence':prediction.confidence,
                 'intent_correct':intent_correct,'query_exact':query_exact,'query_produced':prediction.query is not None,
                 'answer_correct':correct,'status':status,'failure_stage':stage,'reason':reason,
                 'reconstructable':recon[item['id']] if original else True,
                 'pipeline_ms':pipeline_ms,'parser_ms':parser_ms,
                 'oov_tokens':sum(w not in parser.vocabulary for w in tokenize(question)),
                 'tokens':len(tokenize(question))}
            rows.append(row)
            if not correct or not query_exact:
                predictions.append({'scope':scope,'id':row['id'],'predicted_query':canonical(prediction.query) if prediction.query else None,
                                    'status':status,'failure_stage':stage,'reason':reason})
        if stream:stream.close()
        metric=summary(rows);metric['wall_seconds']=time.perf_counter()-scope_start
        metric['oov_token_percent']=pct(sum(r['oov_tokens'] for r in rows),sum(r['tokens'] for r in rows))
        scope_metrics[scope]=metric;all_rows.extend(rows)
        print(json.dumps({'scope':scope,**metric},indent=2),flush=True)
    csv_write(OUT/'per_example.csv',all_rows)
    for name,key in [('per_task','task_type'),('per_register','language_register'),('per_complexity','complexity_level')]:
        grouped=[]
        for scope in scope_metrics:
            for value in sorted({r[key] for r in all_rows if r['scope']==scope}):
                s=summary([r for r in all_rows if r['scope']==scope and r[key]==value])
                grouped.append({'scope':scope,'group':value,**{k:v for k,v in s.items() if not isinstance(v,dict)}})
        csv_write(OUT/(name+'.csv'),grouped)
    failures=Counter((r['scope'],r['task_type'],r['failure_stage'],r['reason']) for r in all_rows if not r['answer_correct'])
    csv_write(OUT/'failure_summary.csv',[dict(zip(('scope','task_type','stage','reason','count'),(*key,count))) for key,count in sorted(failures.items())])
    with (OUT/'failure_predictions.jsonl').open('w',encoding='utf-8') as f:
        for record in predictions:f.write(json.dumps(record,ensure_ascii=False)+'\n')
    metrics={'scopes':scope_metrics,'model_load_seconds':load_seconds,'parameter_count':train['parameter_count'],
             'training_seconds':train['training_seconds'],'cpu_threads':2,'device':'cpu',
             'evaluation_protocol':'Single frozen run; original validation then CORE DEV then CORE EVAL. No retraining, vocabulary changes, threshold selection or slot-rule changes from these results.',
             'query_reference':'Original validation: frozen audited Rule Baseline 0 query, an agreement metric. CORE: preserved source expected_structured_query.',
             'source_hash':source_hash,'challenge_hashes':challenge_hashes,'model_checkpoint_sha256':train['checkpoint_sha256'],
             'inference_inputs':['question','context'],'provider_calls':0,'test_accessed':False,
             'limitation':'CORE was agent-authored in earlier work; source anchors excluded from this model fit. No independent native-speaker review.'}
    (OUT/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
    assert all(hashlib.sha256((CHALLENGE/'challenge'/name).read_bytes()).hexdigest()==value for name,value in challenge_hashes.items())
    frozen_hashes()


if __name__=='__main__':main()
