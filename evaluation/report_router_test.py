"""Freeze TEST routes before Long outcomes, then report once from checkpoints."""
from collections import Counter
import json
import fcntl
from time import perf_counter
from evaluation.router_test_data import BASE,OUT,ROUTER,read,rows,sha,verify
from language.modal_checkpoint import atomic_json,digest


def verify_evaluator():
    for path,value in read(OUT/'evaluation_code_freeze.json').items():assert sha(BASE/path)==value,path


def routes():
    from language.router_v2 import RouterV2
    from threadpoolctl import threadpool_limits
    verify();verify_evaluator()
    public=rows(OUT/'public_inputs.jsonl')
    saved={p.stem:read(p) for p in (OUT/'short_predictions').glob('*.json')}
    assert set(saved)=={r['id'] for r in public} and len(saved)==2332
    with (OUT/'router.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        model_hash=sha(ROUTER/'router_v2.joblib')
        began=perf_counter();router=RouterV2(ROUTER/'router_v2.joblib');load_ms=(perf_counter()-began)*1000
        assert router.threshold==.91
        with threadpool_limits(limits=1):
            for item in public:
                path=OUT/'router_predictions'/(item['id']+'.json')
                features=saved[item['id']]['features']
                if path.exists():
                    old=read(path);assert old['model_sha256']==model_hash and old['feature_digest']==digest(features)
                    assert old['threshold']==.91
                    continue
                began=perf_counter();p=float(router.long_probabilities([features])[0]);ms=(perf_counter()-began)*1000
                atomic_json(path,dict(id=item['id'],route='LONG' if p>=.91 else 'SHORT',long_probability=p,
                    threshold=.91,router_ms=ms,model_sha256=model_hash,feature_digest=digest(features)))
        if not (OUT/'router_runtime.json').exists():atomic_json(OUT/'router_runtime.json',dict(model_load_ms=load_ms))
    verify();verify_evaluator()
    print('Frozen TEST Router decisions:',len(list((OUT/'router_predictions').glob('*.json'))))


def report():
    import numpy as np
    from language.long_parser import query_from_dict,LongParser
    from language.modal_checkpoint import ModalCheckpointProvider
    from spatial import parse_context,execute
    from spatial.normalization import answer_matches
    from evaluation.train_lightweight_router import evaluate_routes
    verify();verify_evaluator()
    if (OUT/'summary.json').exists():raise RuntimeError('Final TEST report already exists; refusing repeated evaluation')
    gold=rows(OUT/'test_records.jsonl');public={r['id']:r for r in rows(OUT/'public_inputs.jsonl')}
    short={p.stem:read(p) for p in (OUT/'short_predictions').glob('*.json')}
    long={p.stem:read(p) for p in (OUT/'long_predictions').glob('*.json')}
    route={p.stem:read(p) for p in (OUT/'router_predictions').glob('*.json')}
    assert set(short)==set(long)==set(route)==set(public) and len(short)==2332
    config=read(OUT/'inference_config.json')
    def no_generation(*args):raise RuntimeError('Verification requested a new Long generation')
    provider=ModalCheckpointProvider(no_generation,OUT/'long_calls',config);parser=LongParser(provider)
    immutable={str(p.relative_to(OUT)):sha(p) for folder in ('short_predictions','long_predictions','router_predictions','long_calls') for p in (OUT/folder).glob('*.json')}
    scoring=[];details=[];short_ms=[];long_ms=[];router_ms=[];feature_ms=[];engine_ms=[];long_geo_ms=[]
    for item in gold:
        key=item['id'];ctx=parse_context(public[key]['context']);s=short[key];l=long[key];r=route[key]
        assert s['config_digest']==l['config_digest']==digest(config)
        assert r['feature_digest']==digest(s['features']) and r['model_sha256']==sha(ROUTER/'router_v2.joblib') and r['threshold']==.91
        provider.begin_case(key);parsed=parser.parse(public[key]['question'],ctx)
        from dataclasses import asdict
        serialized=json.loads(json.dumps(asdict(parsed.query))) if parsed.query else None
        assert serialized==l['query'] and parsed.llm_calls==l['llm_calls'] and parsed.invalid_responses==l['invalid_responses']
        outcomes={};geo={}
        for name,prediction in [('short',s),('long',l)]:
            began=perf_counter();query=query_from_dict(prediction['query']) if prediction['query'] else None
            execution=execute(query,ctx) if query else None;geo[name]=(perf_counter()-began)*1000
            correct=bool(execution and execution.status=='success' and answer_matches(execution.answer,item['answer']))
            outcomes[name]=dict(answer_correct=correct,status=execution.status if execution else prediction['status'],
                reason=execution.reason if execution else prediction.get('reason',prediction.get('message')),
                answer=asdict(execution.answer) if execution and execution.answer else None)
        label='SHORT' if outcomes['short']['answer_correct'] else 'LONG' if outcomes['long']['answer_correct'] else 'FAILURE'
        scoring.append(dict(id=key,label=label,outcomes=outcomes))
        details.append(dict(id=key,task=item['task_type'],difficulty=item['difficulty'],label=label,route=r['route'],
            short=outcomes['short'],long=outcomes['long'],adaptive_answer_correct=outcomes[r['route'].lower()]['answer_correct']))
        short_ms.append(s['parser_ms']+geo['short']);feature_ms.append(s['short_with_features_ms']+geo['short'])
        long_ms.append(l['inference_ms']+geo['long']);router_ms.append(r['router_ms']);long_geo_ms.append(geo['long'])
    decisions=np.array([route[r['id']]['route']=='LONG' for r in scoring])
    metrics=evaluate_routes(scoring,decisions)
    breakdowns={}
    for field in ('task_type','difficulty'):
        breakdowns[field]={}
        for value in sorted({r[field] for r in gold}):
            indexes=[i for i,r in enumerate(gold) if r[field]==value]
            breakdowns[field][value]=evaluate_routes([scoring[i] for i in indexes],decisions[indexes])
    def latency(values):
        a=np.array(values,dtype=float)
        return dict(mean_ms=float(a.mean()),p50_ms=float(np.percentile(a,50)),p95_ms=float(np.percentile(a,95)),total_seconds=float(a.sum()/1000))
    raw_long=np.array([long[r['id']]['inference_ms'] for r in scoring])
    projected=np.array(feature_ms)+np.array(router_ms)+np.where(decisions,np.array(long_ms),0)
    calls=[read(p) for p in (OUT/'long_calls').glob('*.json')]
    sessions=[read(p) for p in (OUT/'long_sessions').glob('*.json')]
    performance=dict(short=latency(short_ms),short_with_features=latency(feature_ms),router=latency(router_ms),
        long=latency(long_ms),projected_adaptive=latency(projected),
        long_generation_calls=len(calls),long_retry_calls=sum(v['llm_calls']-1 for v in long.values()),
        adaptive_generation_calls=sum(long[r['id']]['llm_calls'] for r,u in zip(scoring,decisions) if u),
        avoided_long_inference_seconds=float(raw_long[~decisions].sum()/1000),
        avoided_long_inference_fraction=float(raw_long[~decisions].sum()/raw_long.sum()),
        long_session_wall_seconds=sum(s['wall_seconds'] for s in sessions),
        actual_gpu=sorted({c['gpu'] for c in calls}),model_load_seconds=[s['runtime']['model_load_seconds'] for s in sessions if 'runtime' in s],
        actual_monetary_cost=None,
        note='Adaptive latency is a projection from this evaluation’s component timings. It excludes cold start, idle billing and full RPC overhead. Actual evaluation runs Long on all TEST rows for baselines/oracle; savings apply to hypothetical adaptive-only inference.')
    for path,value in immutable.items():assert sha(OUT/path)==value,path
    verify();verify_evaluator()
    summary=dict(n=2332,model='frozen calibrated Small MLP ensemble',threshold=.91,
        metrics=metrics,performance=performance,breakdowns=breakdowns,
        routing_labels='Answer-correctness SHORT preference; LONG if Short wrong/Long correct; FAILURE both wrong. Binary metrics exclude FAILURE because TEST has no native gold StructuredQuery.',
        language_register='unavailable in native TEST; not inferred',
        short_schema_valid=sum(s['query'] is not None for s in short.values()),long_schema_valid=sum(l['query'] is not None for l in long.values()),
        long_invalid_responses=sum(l['invalid_responses'] for l in long.values()),
        label_counts=dict(Counter(r['label'] for r in scoring)),
        verification=dict(all_frozen_inputs_unchanged=True,all_test_rows_retained=True,completed_predictions_unchanged=True,
            long_checkpoint_replay_verified=2332,replay_new_generations=0,router_inference_once_per_row=True,
            no_gold_in_model_or_router_inputs=True,no_retraining_or_tuning=True))
    atomic_json(OUT/'summary.json',summary)
    (OUT/'per_example.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in details))
    def pct(x):return f'{100*x:.2f}%' if x is not None else 'N/A'
    def table(headers,data):return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+['| '+' | '.join(map(str,r))+' |' for r in data])
    lines=['**Final Router V2 TEST evaluation — all 2,332 examples**','',
        'Frozen calibrated Small MLP ensemble; LONG iff probability >= 0.91. Original TEST questions, answers and split retained. No tuning, retraining, filtering, or repairs after TEST access.','',
        table(['Policy','Correct / 2332','Accuracy'],[[name,metrics[count],pct(metrics[acc])] for name,count,acc in [
            ('Always Short','always_short_correct','always_short_accuracy'),('Always Long','always_long_correct','always_long_accuracy'),
            ('Router V2','adaptive_answer_correct','adaptive_answer_accuracy'),('Answer oracle','oracle_correct','oracle_accuracy')]]),'',
        table(['Metric','Result'],[[k,pct(metrics[k])] for k in ('short_route_rate','long_route_rate','routing_accuracy','long_precision','long_recall')]),'',
        f'Short/Long answer-route counts: {metrics["short_route_count"]}/{metrics["long_route_count"]}. Short inference always runs to provide Router features. Long is an additional call on LONG routes. Rescues: {metrics["long_rescues"]}; regressions: {metrics["long_regressions"]}; unnecessary Long calls: {metrics["unnecessary_long_calls"]}.','',
        f'Routing metrics use {metrics["binary_n"]} non-FAILURE rows, with {metrics["failure_n"]} both-answer-wrong rows analyzed separately. TEST lacks native gold StructuredQueries, so routing labels use final-answer correctness rather than the development exact-query-plus-answer criterion. Confusion: `{json.dumps(metrics["binary_confusion"])}`.','',
        '**Latency and computation**','',
        table(['Component/policy','Mean ms','P50 ms','P95 ms'],[[name,f'{performance[key]["mean_ms"]:.3f}',f'{performance[key]["p50_ms"]:.3f}',f'{performance[key]["p95_ms"]:.3f}'] for name,key in [
            ('Always Short (parser + Geo)','short'),('Short with Router features + Geo','short_with_features'),('Router decision','router'),
            ('Always Long (inference + Geo)','long'),('Projected adaptive','projected_adaptive')]]),'',
        f'Long generation calls: {len(calls)}, including {performance["long_retry_calls"]} retries. Adaptive-only generation calls would be {performance["adaptive_generation_calls"]}. Avoided Long inference: {performance["avoided_long_inference_seconds"]:.2f} seconds ({pct(performance["avoided_long_inference_fraction"])}). GPU: {", ".join(performance["actual_gpu"])}; BF16, existing asar-hf-cache only. Monetary cost unavailable.','',performance['note'],'']
    for field,title in [('task_type','Task'),('difficulty','Native difficulty')]:
        lines += ['**'+title+' breakdown**','',table([title,'N','Short','Long','Router V2','Oracle','Long route rate'],[
            [name,m['n'],pct(m['always_short_accuracy']),pct(m['always_long_accuracy']),pct(m['adaptive_answer_accuracy']),pct(m['oracle_accuracy']),pct(m['long_route_rate'])]
            for name,m in breakdowns[field].items()]),'']
    lines += ['Language register is unavailable in TEST. Native easy/medium/hard difficulty is reported as supplied; CORE complexity labels are not invented. All breakdown labels are used only for scoring/reporting.','',
        'Short and Long predictions were committed to the existing Modal checkpoint volume before completion was counted. Local Router decisions were atomically checkpointed before Long outcomes were available. Resume skips completed predictions. Full local replay verified all Long checkpoints without new generation.','',
        'The original split is disjoint by wikidata anchor ID from Router development data. The pre-existing split audit reports some coordinate aliases across splits; ID-disjointness is not a guarantee of geographic-coordinate separation. Original template wording also differs from Router development paraphrases. No TEST rows were removed.','',
        'All frozen component and dataset hashes passed. No CORE rerun, training, tuning, push, or merge. Artifacts: `summary.json`, `per_example.jsonl`, prediction/checkpoint directories, `pre_test_freeze.json`, `evaluation_code_freeze.json`, `final_manifest.json`.','',
        '**Stopped after the final TEST report.**','']
    (OUT/'REPORT.md').write_text('\n'.join(lines))
    atomic_json(OUT/'final_manifest.json',dict(input_freeze_sha256=sha(OUT/'pre_test_freeze.json'),
        evaluation_code_freeze_sha256=sha(OUT/'evaluation_code_freeze.json'),predictions_sha256=immutable,
        reports_sha256={name:sha(OUT/name) for name in ('summary.json','per_example.jsonl','REPORT.md')}))
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':
    import sys
    {'routes':routes,'report':report}[sys.argv[1]]()
