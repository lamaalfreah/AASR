"""One fixed Router evaluation on saved CORE EVAL outcomes; never fits models."""
import csv
import hashlib
import json
from pathlib import Path
from time import perf_counter

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE / 'experiments/E5_lightweight_adaptive'
ROUTER = ROOT / 'step3b_router'
CORE = ROOT / 'step2_short_long/challenge'
LONG = ROOT / 'step2_long'
SHORT = ROOT / 'step2_neural_short'
OUT = ROOT / 'step3c_router_core_eval'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read(p):
    return json.loads(p.read_text())


def rows(p):
    return [json.loads(line) for line in p.read_text().splitlines()]


def write(p, obj):
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def verify(checks):
    for name, expected in checks.items():
        assert sha(BASE/name) == expected, 'Frozen file changed: '+name


def prepare():
    if OUT.exists():
        raise RuntimeError('Evaluation directory already exists; refusing a second setup')
    freeze = read(ROUTER/'freeze_manifest.json')
    assert freeze['selected_model'] == 'small_mlp'
    assert freeze['threshold'] == 0.30140149024077767
    checks = dict(freeze['frozen_component_hashes'], **freeze['code_sha256'])
    for name, digest in freeze['artifact_sha256'].items():
        checks[str((ROUTER/name).relative_to(BASE))] = digest
    checks[str((ROUTER/'freeze_manifest.json').relative_to(BASE))] = sha(ROUTER/'freeze_manifest.json')
    verify(checks)
    prior = read(LONG/'freeze_manifest.json')
    assert sha(CORE/'core_eval.jsonl') == prior['challenge_files']['core_eval.jsonl']
    assert sha(SHORT/'per_example.csv') == prior['neural_predictions_sha256']
    assert sha(CORE/'source_contexts.jsonl') == prior['challenge_files']['source_contexts.jsonl']
    inputs = rows(CORE/'core_eval.jsonl')
    assert len(inputs) == 128 and all(r['challenge_split']=='eval' for r in inputs)
    long_paths = sorted((LONG/'modal_predictions/eval').glob('*.json'))
    assert len(long_paths) == 128
    assert {read(p)['challenge_id'] for p in long_paths} == {r['challenge_id'] for r in inputs}
    context = {r['source_record_id']:r['context'] for r in rows(CORE/'source_contexts.jsonl')}
    for p in [CORE/'core_eval.jsonl',CORE/'source_contexts.jsonl',SHORT/'per_example.csv',*long_paths,
              BASE/'evaluation/router_data.py',BASE/'evaluation/modal_router_data.py']:
        checks[str(p.relative_to(BASE))] = sha(p)
    stage3a = read(ROOT/'step3a_router_data/pipeline_freeze.json')
    verify(stage3a['code_sha256'])
    OUT.mkdir()
    public = [dict(id=r['challenge_id'],question=r['challenge_question'],context=context[r['source_record_id']]) for r in inputs]
    (OUT/'public_inputs.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in public))
    write(OUT/'pre_eval_freeze.json',dict(sha256=checks,router_threshold=freeze['threshold'],
        router_model=freeze['selected_model'],n=128,
        policy='One fixed threshold; existing Short/Long outcomes; CPU Short feature augmentation only; no fitting or Long generation',
        public_inputs_sha256=sha(OUT/'public_inputs.jsonl')))
    print('Verified frozen Small MLP and threshold; prepared 128 public feature inputs')


def evaluate():
    import numpy as np
    from threadpoolctl import threadpool_limits
    from language.lightweight_router import LightweightRouter
    from evaluation.train_lightweight_router import evaluate_routes
    from language.long_parser import query_from_dict
    from spatial import parse_context, execute
    from spatial.normalization import answer_matches

    if (OUT/'evaluation_started.json').exists():
        raise RuntimeError('Evaluation has already started; no repeat evaluation allowed')
    freeze = read(OUT/'pre_eval_freeze.json'); verify(freeze['sha256'])
    response = read(OUT/'short_feature_response.json')
    feature_rows = {r['id']:r for r in response['predictions']}
    core = rows(CORE/'core_eval.jsonl')
    public = {r['id']:r for r in rows(OUT/'public_inputs.jsonl')}
    assert sha(OUT/'public_inputs.jsonl') == freeze['public_inputs_sha256']
    with (SHORT/'per_example.csv').open() as stream:
        short = {r['id']:r for r in csv.DictReader(stream) if r['scope']=='core_eval'}
    long = {r['challenge_id']:r for r in [read(p) for p in (LONG/'modal_predictions/eval').glob('*.json')]}
    assert set(feature_rows)==set(short)==set(long)==set(public) and len(short)==128
    scoring=[]
    boolval=lambda s: s=='True'
    max_confidence_difference=0.0
    for item in core:
        key=item['challenge_id']; saved=short[key]; features=feature_rows[key]; previous=long[key]
        assert features['operation']==saved['predicted_intent']
        diff=abs(features['features']['short_confidence']-float(saved['confidence']))
        max_confidence_difference=max(diff,max_confidence_difference)
        assert diff <= 1e-6, (key,'Short confidence mismatch',diff)
        assert (features['query'] is not None)==boolval(saved['query_produced'])
        if features['query'] is None:
            assert features['reason']==saved['reason']
        # Verify replay parity only; never replace any saved outcomes.
        assert (features['query']==item['expected_structured_query'])==boolval(saved['query_exact'])
        query=query_from_dict(features['query']) if features['query'] else None
        execution=execute(query,parse_context(public[key]['context'])) if query else None
        correct=bool(execution and execution.status=='success' and answer_matches(execution.answer,item['gold_answer']))
        assert correct==boolval(saved['answer_correct'])
        assert previous['short_query_exact']==boolval(saved['query_exact'])
        assert previous['short_answer_correct']==boolval(saved['answer_correct'])
        s=boolval(saved['query_exact']) and boolval(saved['answer_correct'])
        l=previous['long_query_exact'] and previous['long_answer_correct']
        scoring.append(dict(id=key,features=features['features'],label='SHORT' if s else 'LONG' if l else 'FAILURE',
            outcomes=dict(short=dict(query_exact=boolval(saved['query_exact']),answer_correct=boolval(saved['answer_correct'])),
                          long=dict(query_exact=previous['long_query_exact'],answer_correct=previous['long_answer_correct']))))
    write(OUT/'evaluation_started.json',dict(n=128,model='small_mlp',threshold=freeze['router_threshold'],
        pre_eval_freeze_sha256=sha(OUT/'pre_eval_freeze.json'),short_feature_response_sha256=sha(OUT/'short_feature_response.json')))
    began=perf_counter(); router=LightweightRouter(ROUTER/'router.joblib'); load_ms=(perf_counter()-began)*1000
    assert router.threshold == freeze['router_threshold']
    predictions=[]; overhead=[]
    with threadpool_limits(limits=1):
        for row in scoring:
            began=perf_counter()
            p=float(router.long_probabilities([row['features']])[0])
            elapsed=(perf_counter()-began)*1000;overhead.append(elapsed)
            route='LONG' if p>=router.threshold else 'SHORT'
            predictions.append(dict(id=row['id'],route=route,long_probability=p,router_ms=elapsed,
                label=row['label'],adaptive_answer_correct=row['outcomes'][route.lower()]['answer_correct']))
    (OUT/'predictions.jsonl').write_text(''.join(json.dumps(p)+'\n' for p in predictions))
    use_long=np.array([p['route']=='LONG' for p in predictions])
    metrics=evaluate_routes(scoring,use_long)
    # Saved latency projection: no new end-to-end/GPU run is implied.
    s_ms=np.array([float(short[r['id']]['pipeline_ms']) for r in scoring])
    l_ms=np.array([long[r['id']]['long_inference_ms'] for r in scoring])
    roundtrip=np.array([long[r['id']]['long_roundtrip_ms'] for r in scoring])
    feature_ms=np.array([feature_rows[r['id']]['short_with_feature_ms'] for r in scoring])
    def timing(values):
        return dict(mean_ms=float(np.mean(values)),p50_ms=float(np.percentile(values,50)),
                    p95_ms=float(np.percentile(values,95)),sum_ms=float(np.sum(values)))
    adaptive=s_ms+np.array(overhead)+np.where(use_long,l_ms,0)
    total_calls=sum(long[r['id']]['llm_calls'] for r in scoring)
    selected_calls=sum(long[r['id']]['llm_calls'] for r,u in zip(scoring,use_long) if u)
    perf=dict(router_load_ms=load_ms,router_observed=timing(overhead),frozen_short=timing(s_ms),
        saved_long_inference=timing(l_ms),projected_adaptive=timing(adaptive),
        projected_adaptive_with_remote_roundtrip=timing(s_ms+np.array(overhead)+np.where(use_long,roundtrip,0)),
        observed_cpu_short_with_features=timing(feature_ms),
        projected_adaptive_using_observed_feature_time=timing(feature_ms+np.array(overhead)+np.where(use_long,l_ms,0)),
        saved_long_generation_calls=total_calls,projected_adaptive_long_generation_calls=selected_calls,
        avoided_long_inference_seconds=float(l_ms[~use_long].sum()/1000),
        avoided_long_inference_fraction=float(l_ms[~use_long].sum()/l_ms.sum()),
        new_long_generations=0,new_gpu_calls=0,new_cpu_short_feature_passes=128,
        actual_monetary_cost=None,
        limitation='Latency projection from saved CPU Short and L4 Long timings plus local Router timings, not a measured deployed service. Excludes Long cold-start, idle billing, and full orchestration; currencies unavailable.')
    verify(freeze['sha256'])
    result=dict(model='small_mlp',threshold=router.threshold,metrics=metrics,performance=perf,
        verification=dict(all_frozen_hashes_unchanged=True,n=128,short_feature_prediction_parity=True,
            max_short_confidence_difference=max_confidence_difference,gold_not_used_as_router_features=True,
            eval_runs=1,refit=False,threshold_search=False,test_not_accessed=True))
    write(OUT/'summary.json',result)
    def pct(x):return f'{100*x:.2f}%' if x is not None else 'N/A'
    def table(headers,data):
        return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+['| '+' | '.join(map(str,r))+' |' for r in data])
    lines=['**Final frozen Router CORE EVAL — 128 examples**','',
        f'Frozen Small MLP, LONG iff score >= `{router.threshold:.17g}`. One Router inference per example. No fitting, threshold selection, or changes after CORE results. Original frozen Short/Long outcomes supply all accuracy scores.','',
        table(['Policy','Correct / 128','Accuracy'],[[name,metrics[count],pct(metrics[acc])] for name,count,acc in [
            ('Always Short','always_short_correct','always_short_accuracy'),('Always Long','always_long_correct','always_long_accuracy'),
            ('Adaptive Router','adaptive_answer_correct','adaptive_answer_accuracy'),('Answer oracle','oracle_correct','oracle_accuracy')]]),'',
        table(['Metric','Result'],[[k,pct(metrics[k])] for k in ('routing_accuracy','long_precision','long_recall','short_route_rate','long_route_rate')]),'',
        f'Binary routing metrics exclude {metrics["failure_n"]} FAILURE rows and use {metrics["binary_n"]} SHORT/LONG rows, retaining the Step 3A exact-query-plus-answer label definition and SHORT preference. All 128 rows count toward final-answer accuracy and route rates. Binary confusion: `{json.dumps(metrics["binary_confusion"])}`.','',
        f'Long rescues: {metrics["long_rescues"]}; regressions: {metrics["long_regressions"]}; unnecessary Long calls: {metrics["unnecessary_long_calls"]}. Regression rate among Short-answer-correct examples: {pct(metrics["long_regression_rate_among_short_correct"])}. FAILURE analysis: `{json.dumps(metrics["failure_analysis"])}`.','',
        'Short/Long route rates describe the selected answer path. Short inference is required on 100% of inputs to compute Router features; Long is an additional call only for LONG routes.','',
        '**Latency and cost implication**','',
        table(['Measurement','Mean ms'],[[name,f'{perf[key]["mean_ms"]:.3f}'] for name,key in [
            ('Frozen Short pipeline','frozen_short'),('Saved Long inference','saved_long_inference'),('Observed local Router','router_observed'),
            ('Projected adaptive (saved Short + Router + selected Long)','projected_adaptive'),
            ('CPU Short plus feature extraction, observed','observed_cpu_short_with_features'),
            ('Projected adaptive with observed feature extraction','projected_adaptive_using_observed_feature_time')]]),'',
        f'Projected Long generation calls (including existing retries): {selected_calls}/{total_calls}; avoided saved Long inference time: {perf["avoided_long_inference_seconds"]:.2f} seconds ({pct(perf["avoided_long_inference_fraction"])}). These are inference-work savings, not a measured bill reduction. Cost is unavailable. The evaluation itself made zero Long generations and zero GPU calls.','',
        perf['limitation'],'',
        '**Feature provenance and checks**','',
        'Original CORE Short predictions did not save entropy/top-two margin. One CPU pass of the unchanged frozen Short model and the exact Step 3A feature extractor supplied the missing inference-safe features. Operation, confidence (tolerance 1e-6), query availability, query exactness, and final-answer correctness matched all original frozen Short predictions. Original outcomes were not overwritten. Only public question/context fields reached the CPU function; only the frozen feature dictionary reached Router inference.','',
        'All Router, threshold, Short, Long, schema, Geo Engine, CORE inputs, and saved prediction hashes verified unchanged. No TEST data read. Router retains its recorded DEV training iteration-limit warning; no convergence adjustment or retraining occurred.','',
        'Artifacts: `summary.json`, `predictions.jsonl`, `short_feature_response.json`, `pre_eval_freeze.json`, and `freeze_manifest.json`. Stopped after this final evaluation.','']
    (OUT/'REPORT.md').write_text('\n'.join(lines))
    write(OUT/'freeze_manifest.json',dict(input_checks=freeze['sha256'],
        code_sha256={str(Path(__file__).relative_to(BASE)):sha(Path(__file__)),
                     'evaluation/modal_router_eval_features.py':sha(BASE/'evaluation/modal_router_eval_features.py')},
        outputs={p.name:sha(p) for p in OUT.iterdir() if p.is_file()}))
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    import sys
    {'prepare':prepare,'evaluate':evaluate}[sys.argv[1]]()
