"""Durable final TEST inference. Only public inputs reach Modal functions."""
import json
import modal
from pathlib import Path
from evaluation.modal_router_data import app,short_image
from language.modal_long_backend import QwenLong,prepare_weights,checkpoints,primitive_payload
from language.modal_checkpoint import atomic_json,ModalCheckpointProvider,digest
from evaluation.router_test_data import OUT,SHORT,verify,read,rows,sha

test_image=short_image
if modal.is_local():
    test_image=(test_image.add_local_file(OUT/'public_inputs.jsonl','/test/public.jsonl')
        .add_local_file(OUT/'inference_config.json','/test/config.json')
        .add_local_file(OUT/'remote_spec.json','/test/spec.json'))


def inputs():
    spec=read('/test/spec.json');config=read('/test/config.json')
    assert sha('/test/public.jsonl')==spec['public_sha256']
    assert sha('/test/config.json')==spec['config_sha256']
    public=rows('/test/public.jsonl')
    assert len(public)==2332 and all(set(r)=={'id','question','context'} for r in public)
    directory=Path('/checkpoints')/spec['remote_namespace']
    directory.mkdir(exist_ok=True)
    return public,config,spec,directory


def progress(directory,status,reason=None):
    from time import time
    value=dict(status=status,total=2332,short_completed=len(list((directory/'short').glob('*.json'))),
        long_completed=len(list((directory/'long').glob('*.json'))),
        generation_checkpoints=len(list((directory/'calls').glob('*.json'))),reason=reason,updated_unix=time())
    atomic_json(directory/'progress.json',value);checkpoints.commit()
    return value


def collect(directory,kind):
    return [read(p) for p in sorted((directory/kind).glob('*.json'))]


@app.function(image=test_image,volumes={'/checkpoints':checkpoints},cpu=2,memory=4096,
              timeout=86400,retries=0,max_containers=1)
def short_test():
    import torch,fcntl
    from time import perf_counter
    from dataclasses import asdict
    from language.neural_short import NeuralShortParser
    from spatial import parse_context
    from spatial.schema import OPERATIONS
    from evaluation.router_data import safe_features
    public,config,spec,directory=inputs()
    with (directory/'short.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if len(list((directory/'short').glob('*.json')))==2332:
            return primitive_payload(dict(predictions=collect(directory,'short'),progress=progress(directory,'short_complete')))
        for name,value in spec['short_artifact_hashes'].items():assert sha(Path('/short')/name)==value
        torch.set_num_threads(2);torch.set_num_interop_threads(1)
        began=perf_counter();parser=NeuralShortParser('/short');load=perf_counter()-began
        captured={}
        def capture(module,inp,output):captured['p']=output.detach().softmax(-1)[0].tolist()
        handle=parser.model.register_forward_hook(capture)
        try:
            for item in public:
                path=directory/'short'/(item['id']+'.json')
                if path.exists():
                    assert read(path)['config_digest']==digest(config)
                    continue
                started=perf_counter();context=parse_context(item['context'])
                prediction=parser.predict(item['question'],context)
                parser_ms=(perf_counter()-started)*1000
                features=safe_features(item['question'],context,prediction,captured['p'])
                features['operation_probabilities']=dict(zip(OPERATIONS,captured['p']))
                saved=primitive_payload(dict(id=item['id'],config_digest=digest(config),operation=prediction.operation,
                    query=asdict(prediction.query) if prediction.query else None,status=prediction.status,
                    reason=prediction.reason,features=features,parser_ms=parser_ms,
                    short_with_features_ms=(perf_counter()-started)*1000))
                atomic_json(path,saved)
                state=progress(directory,'short_running')
                if state['short_completed']%100==0:print('TEST Short checkpointed:',state['short_completed'],'/ 2332',flush=True)
        except Exception as exc:
            progress(directory,'short_stopped',type(exc).__name__+': '+str(exc));raise
        finally:handle.remove()
        atomic_json(directory/'short_runtime.json',dict(load_seconds=load,wall_seconds=perf_counter()-began,
            device='cpu',torch_version=str(torch.__version__)))
        state=progress(directory,'short_complete')
        return primitive_payload(dict(predictions=collect(directory,'short'),runtime=read(directory/'short_runtime.json'),progress=state))


@app.function(image=test_image,volumes={'/checkpoints':checkpoints},cpu=2,memory=4096,
              timeout=86400,retries=0,max_containers=1)
def long_test():
    import fcntl,uuid
    from time import perf_counter,time
    from dataclasses import asdict
    from language.long_parser import LongParser
    from spatial import parse_context
    public,config,spec,directory=inputs()
    with (directory/'long.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if len(list((directory/'long').glob('*.json')))==2332:
            return primitive_payload(dict(predictions=collect(directory,'long'),calls=collect(directory,'calls'),
                sessions=collect(directory,'sessions'),progress=progress(directory,'complete')))
        session_id=uuid.uuid4().hex;began=perf_counter()
        session=dict(session_id=session_id,started_unix=time())
        try:
            session['cache']=prepare_weights.remote(config['revision'])
            model=QwenLong(revision=config['revision'])
            session['runtime']=model.metadata.remote()
            atomic_json(directory/'sessions'/(session_id+'.json'),session);checkpoints.commit()
            print('TEST frozen Long loaded:',json.dumps(session['runtime']),flush=True)
            provider=ModalCheckpointProvider(model.generate.remote,directory/'calls',config,
                lambda:progress(directory,'long_running'))
            parser=LongParser(provider)
            for item in public:
                path=directory/'long'/(item['id']+'.json')
                if path.exists():
                    assert read(path)['config_digest']==digest(config)
                    continue
                provider.begin_case(item['id'])
                prediction=parser.parse(item['question'],parse_context(item['context']))
                saved=primitive_payload(dict(id=item['id'],config_digest=digest(config),
                    query=asdict(prediction.query) if prediction.query else None,status=prediction.status,message=prediction.message,
                    llm_calls=prediction.llm_calls,invalid_responses=prediction.invalid_responses,
                    input_tokens=prediction.input_tokens,output_tokens=prediction.output_tokens,
                    inference_ms=sum(c['inference_ms'] for c in provider.case_calls),
                    roundtrip_ms=sum(c['request_roundtrip_ms'] for c in provider.case_calls)))
                atomic_json(path,saved);state=progress(directory,'long_running')
                if state['long_completed']%25==0 or state['long_completed']==2332:
                    print('TEST Long checkpointed:',state['long_completed'],'/ 2332',flush=True)
            session['status']='complete'
        except Exception as exc:
            session.update(status='stopped',error=type(exc).__name__+': '+str(exc))
            progress(directory,'long_stopped',session['error']);raise
        finally:
            session['wall_seconds']=perf_counter()-began
            atomic_json(directory/'sessions'/(session_id+'.json'),session);checkpoints.commit()
        return primitive_payload(dict(predictions=collect(directory,'long'),calls=collect(directory,'calls'),
            sessions=collect(directory,'sessions'),progress=progress(directory,'complete')))


@app.function(image=test_image,volumes={'/checkpoints':checkpoints},cpu=1,memory=4096,timeout=300)
def saved_state(include_results:bool=False):
    checkpoints.reload()
    _,_,_,directory=inputs()
    value=dict(progress=read(directory/'progress.json') if (directory/'progress.json').exists() else dict(short_completed=0,long_completed=0,total=2332))
    if include_results:
        value.update(short=collect(directory,'short'),long=collect(directory,'long'),calls=collect(directory,'calls'),sessions=collect(directory,'sessions'))
    return primitive_payload(value)


def mirror(kind,data):
    target=OUT/kind;target.mkdir(exist_ok=True)
    for row in data:
        key=row.get('id') or row.get('request_digest') or row.get('session_id')
        path=target/(key+'.json')
        if path.exists():assert read(path)==row,'Attempt to overwrite frozen prediction: '+str(path)
        else:atomic_json(path,row)


@app.local_entrypoint()
def short_run():
    verify();response=short_test.remote();mirror('short_predictions',response['predictions'])
    atomic_json(OUT/'short_runtime.json',response.get('runtime',{}));atomic_json(OUT/'progress.json',response['progress']);verify()
    print('TEST Short saved locally:',len(response['predictions']))


@app.local_entrypoint()
def long_run():
    verify()
    assert len(list((OUT/'router_predictions').glob('*.json')))==2332,'Freeze Router decisions before Long inference'
    response=long_test.remote();mirror('long_predictions',response['predictions']);mirror('long_calls',response['calls'])
    for session in response['sessions']:atomic_json(OUT/'long_sessions'/(session['session_id']+'.json'),session)
    atomic_json(OUT/'progress.json',response['progress']);verify()
    print('TEST Long saved locally:',len(response['predictions']))


@app.local_entrypoint()
def status(sync:bool=False):
    verify();response=saved_state.remote(sync)
    if sync:
        mirror('short_predictions',response['short']);mirror('long_predictions',response['long']);mirror('long_calls',response['calls'])
        for session in response['sessions']:atomic_json(OUT/'long_sessions'/(session['session_id']+'.json'),session)
    atomic_json(OUT/'progress.json',response['progress']);print(json.dumps(response['progress']))
