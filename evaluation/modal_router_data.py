"""Step 3A paired inference only; reuse the frozen Long class and CPU Short.

modal run --profile joudalrubaish -m evaluation.modal_router_data::router_data
"""
import json
from pathlib import Path
from time import perf_counter, time
import fcntl
import modal

from language.modal_long_backend import app, image, QwenLong, prepare_weights, primitive_payload
from language.modal_checkpoint import ModalCheckpointProvider, atomic_json, digest
from language.long_parser import LongParser, query_from_dict
from evaluation.router_data import (BASE, OUT, SHORT, read_jsonl, write_jsonl, verify_manifest,
    canonical, safe_features, score, label, sha)
from spatial import parse_context, execute

short_image = image
if modal.is_local():
    short_image = (image.add_local_python_source('spatial', 'evaluation')
        .add_local_file(SHORT/'intent_bigru.pt', '/short/intent_bigru.pt')
        .add_local_file(SHORT/'model_config.json', '/short/model_config.json'))


@app.function(image=short_image, cpu=2, memory=4096, timeout=1800, retries=0,
              max_containers=1)
def short_inference(public_inputs: list, artifact_hashes: dict):
    """No gold, source anchors, splits, task labels, or register reach the model."""
    import torch
    from language.neural_short import NeuralShortParser
    for name, expected in artifact_hashes.items():
        if sha(Path('/short')/name) != expected:
            raise RuntimeError('Remote Short artifact mismatch')
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    started = perf_counter()
    parser = NeuralShortParser('/short')
    load_seconds = perf_counter()-started
    captured = {}
    # Observe the existing forward pass; do not replace or rerun its classifier.
    def capture(module, inputs, output):
        captured['probs'] = output.detach().softmax(-1)[0].tolist()
    handle = parser.model.register_forward_hook(capture)
    predictions = []
    try:
        for item in public_inputs:
            if set(item) != {'id','question','context'}:
                raise ValueError('Non-public Short input fields')
            began = perf_counter()
            context = parse_context(item['context'])
            prediction = parser.predict(item['question'], context)
            probabilities = captured['probs']
            if abs(max(probabilities)-prediction.confidence)>1e-7:
                raise RuntimeError('Observed probability differs from frozen prediction')
            features = safe_features(item['question'],context,prediction,probabilities)
            predictions.append(dict(id=item['id'],operation=prediction.operation,
                query=canonical(prediction.query),status=prediction.status,reason=prediction.reason,
                features=features,short_with_feature_ms=(perf_counter()-began)*1000))
    finally:
        handle.remove()
    return primitive_payload(dict(predictions=predictions,load_seconds=load_seconds,
        wall_seconds=perf_counter()-started,torch_version=str(torch.__version__),device='cpu'))


def progress(status, session=None, reason=None):
    paths=list((OUT/'paired_predictions').glob('*.json'))
    atomic_json(OUT/'progress.json',dict(status=status,completed=len(paths),total=2400,
        remaining=2400-len(paths),generation_checkpoints=len(list((OUT/'long_calls').glob('*.json'))),
        updated_unix=time(),session=session,reason=reason))


def finish(primary_correctness):
    from collections import Counter
    import csv
    items={r['id']:r for r in read_jsonl(OUT/'questions.jsonl')}
    paired=[json.loads(p.read_text()) for p in sorted((OUT/'paired_predictions').glob('*.json'))]
    if len(paired)!=2400 or {r['id'] for r in paired}!=set(items):
        raise RuntimeError('Step 3A incomplete; refusing final Router dataset')
    allowed={'short_confidence','short_entropy','short_top1_top2_margin','query_valid','missing_slots',
        'entity_ambiguity_count','question_length_chars','question_length_words','predicted_operation','constraint_count'}
    dataset=[]
    for row in paired:
        source=items[row['id']]
        if set(row['features'])!=allowed:
            raise RuntimeError('Router feature allowlist mismatch')
        strict=label(row['short']['query_exact'] and row['short']['answer_correct'],
                     row['long']['query_exact'] and row['long']['answer_correct'])
        answers=label(row['short']['answer_correct'],row['long']['answer_correct'])
        dataset.append(dict(id=row['id'],features=row['features'],label=strict if primary_correctness=='exact_and_answer' else answers,
            strict_label=strict,answer_only_label=answers,
            provenance={k:source[k] for k in ('source_split','source_record_id','source_anchor','router_split','task_type','language_register','wording_family')},
            outcomes={'short':row['short'],'long':row['long']}))
    write_jsonl(OUT/'router_dataset.jsonl',dataset)
    for split in ('train','dev'):
        selected=[r for r in dataset if r['provenance']['router_split']==split]
        write_jsonl(OUT/f'router_{split}.jsonl',selected)
        write_jsonl(OUT/f'router_{split}_features.jsonl',[{'id':r['id'],'features':r['features'],'label':r['label']} for r in selected])
    groups=[]
    for split in ('all','train','dev'):
        for task in sorted({r['provenance']['task_type'] for r in dataset}):
            for register in sorted({r['provenance']['language_register'] for r in dataset}):
                selected=[r for r in dataset if (split=='all' or r['provenance']['router_split']==split)
                          and r['provenance']['task_type']==task and r['provenance']['language_register']==register]
                counts=Counter(r['label'] for r in selected)
                groups.append(dict(split=split,task=task,register=register,n=len(selected),
                    **{k:counts[k] for k in ('SHORT','LONG','FAILURE')}))
    with (OUT/'task_register_distribution.csv').open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(groups[0]));writer.writeheader();writer.writerows(groups)
    availability={key:sum(r['features'][key] is not None for r in dataset) for key in sorted(allowed)}
    sessions=[json.loads(p.read_text()) for p in (OUT/'sessions').glob('*.json')]
    calls=[json.loads(p.read_text()) for p in (OUT/'long_calls').glob('*.json')]
    counts=Counter(r['label'] for r in dataset)
    summary=dict(n=2400,primary_correctness=primary_correctness,
        label_counts={k:counts[k] for k in ('SHORT','LONG','FAILURE')},
        strict_label_counts=dict(Counter(r['strict_label'] for r in dataset)),
        answer_only_label_counts=dict(Counter(r['answer_only_label'] for r in dataset)),
        counts_by_split=dict(Counter(r['provenance']['router_split'] for r in dataset)),
        labels_by_split={s:dict(Counter(r['label'] for r in dataset if r['provenance']['router_split']==s)) for s in ('train','dev')},
        counts_by_task=dict(Counter(r['provenance']['task_type'] for r in dataset)),
        counts_by_register=dict(Counter(r['provenance']['language_register'] for r in dataset)),
        feature_nonnull_counts=availability,long_generation_calls=len(calls),
        long_retry_calls=sum(r['long']['llm_calls']-1 for r in paired),
        long_inference_seconds=sum(c['inference_ms'] for c in calls)/1000,
        long_session_wall_seconds=sum(s.get('long_wall_seconds',0) for s in sessions),
        sessions=sessions,gpus=sorted({c['gpu'] for c in calls}),
        frozen_components_verified=True,router_trained=False)
    atomic_json(OUT/'summary.json',summary)
    return summary


@app.local_entrypoint()
def router_data(primary_correctness: str = 'exact_and_answer'):
    if primary_correctness not in ('exact_and_answer','answer_only'):
        raise ValueError('Unknown correctness criterion')
    verify_manifest()
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'.run.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        _run(primary_correctness)


def _run(primary_correctness):
    started=perf_counter()
    session=dict(app_id=app.app_id,started_unix=time(),primary_correctness=primary_correctness)
    session_path=OUT/'sessions'/(app.app_id+'.json')
    long_started=None
    try:
        original=json.loads((BASE/'experiments/E5_lightweight_adaptive/step2_long/modal_experiment.json').read_text())
        config=dict(original,stage='3A',dataset_sha256=sha(OUT/'questions.jsonl'))
        config_path=OUT/'inference_config.json'
        if config_path.exists() and json.loads(config_path.read_text())!=config:
            raise RuntimeError('Frozen Step 3A inference configuration changed')
        atomic_json(config_path,config)
        items=read_jsonl(OUT/'questions.jsonl')
        if len(list((OUT/'paired_predictions').glob('*.json')))==2400:
            finish(primary_correctness);progress('complete');return
        short_path=OUT/'short_predictions.jsonl'
        if not short_path.exists():
            response=short_inference.remote(read_jsonl(OUT/'public_inputs.jsonl'),
                {name:sha(SHORT/name) for name in ('intent_bigru.pt','model_config.json')})
            write_jsonl(short_path,response.pop('predictions'))
            atomic_json(OUT/'short_runtime.json',response)
        short={r['id']:r for r in read_jsonl(short_path)}
        if set(short)!={r['id'] for r in items}:
            raise RuntimeError('Short coverage mismatch')
        print('Frozen CPU Short complete: 2400/2400',flush=True)
        long_started=perf_counter()
        session['cache']=prepare_weights.remote(config['revision'])
        remote=QwenLong(revision=config['revision'])
        session['runtime']=remote.metadata.remote()
        atomic_json(session_path,session)
        print('Frozen Long loaded:',json.dumps(session['runtime']),flush=True)
        provider=ModalCheckpointProvider(remote.generate.remote,OUT/'long_calls',config,
            lambda:progress('running',app.app_id))
        parser=LongParser(provider)
        for item in items:
            path=OUT/'paired_predictions'/(item['id']+'.json')
            if path.exists():
                if json.loads(path.read_text())['config_hash']!=digest(config):
                    raise RuntimeError('Paired prediction configuration mismatch')
                continue
            context=parse_context(item['context'])
            provider.begin_case(item['id'])
            outcome=parser.parse(item['question'],context)
            result=execute(outcome.query,context) if outcome.query else None
            short_row=short[item['id']]
            short_query=query_from_dict(short_row['query']) if short_row['query'] else None
            short_execution=execute(short_query,context) if short_query else None
            short_scores=score(short_query,short_execution,item)
            long_scores=score(outcome.query,result,item)
            short_scores.update(query=short_row['query'],status=short_execution.status if short_execution else short_row['status'])
            long_scores.update(query=canonical(outcome.query),status=result.status if result else outcome.status,
                reason=result.reason if result else outcome.message,llm_calls=outcome.llm_calls,
                invalid_responses=outcome.invalid_responses,inference_ms=sum(c['inference_ms'] for c in provider.case_calls))
            atomic_json(path,dict(id=item['id'],config_hash=digest(config),features=short_row['features'],
                short=short_scores,long=long_scores))
            progress('running',app.app_id)
            completed=len(list((OUT/'paired_predictions').glob('*.json')))
            if completed%25==0 or completed==2400:
                print(f'Step 3A checkpointed: {completed}/2400',flush=True)
        # Sample remote committed reads; complete local replay uses no GPU calls.
        calls=sorted((OUT/'long_calls').glob('*.json'))
        for path in (calls[0],calls[-1]):
            call=json.loads(path.read_text())
            saved=remote.read_checkpoint.remote(call['request_digest'])
            if saved!={k:v for k,v in call.items() if k not in ('case_id','request_roundtrip_ms')}:
                raise RuntimeError('Remote committed checkpoint mismatch')
        session['remote_checkpoint_sample_verified']=True
        session['status']='complete'
    except (Exception,KeyboardInterrupt) as exc:
        session.update(status='stopped',error=type(exc).__name__+': '+str(exc))
        print('Stopped:',session['error'],flush=True)
        raise
    finally:
        session['wall_seconds']=perf_counter()-started
        if long_started is not None:
            session['long_wall_seconds']=perf_counter()-long_started
        atomic_json(session_path,session)
        progress(session.get('status','stopped'),app.app_id,session.get('error'))
        verify_manifest()
    summary=finish(primary_correctness)
    print(json.dumps({k:summary[k] for k in ('n','label_counts','counts_by_split','long_generation_calls')},ensure_ascii=False),flush=True)
