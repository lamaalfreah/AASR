"""CPU observation of full frozen Short probability vector; no Long calls."""
from evaluation.modal_router_data import app, short_image
from evaluation.router_v2_data import OUT, SOURCE, SHORT, verify, rows, sha, write


@app.function(image=short_image,cpu=2,memory=4096,timeout=1800,retries=0,max_containers=1)
def probability_vectors(public_inputs,artifact_hashes):
    import torch
    from pathlib import Path
    from dataclasses import asdict
    from time import perf_counter
    from language.neural_short import NeuralShortParser
    from spatial import parse_context
    from spatial.schema import OPERATIONS
    from language.modal_long_backend import primitive_payload
    from evaluation.router_v2_data import sha
    for name,digest in artifact_hashes.items():assert sha(Path('/short')/name)==digest
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    started=perf_counter();parser=NeuralShortParser('/short');load=perf_counter()-started
    captured={}
    def capture(module,inputs,output):captured['p']=output.detach().softmax(-1)[0].tolist()
    handle=parser.model.register_forward_hook(capture)
    result=[]
    try:
        for item in public_inputs:
            assert set(item)=={'id','question','context'}
            prediction=parser.predict(item['question'],parse_context(item['context']))
            result.append(dict(id=item['id'],operation=prediction.operation,confidence=float(prediction.confidence),
                query=asdict(prediction.query) if prediction.query else None,status=prediction.status,reason=prediction.reason,
                operation_probabilities=dict(zip(OPERATIONS,captured['p']))))
    finally:handle.remove()
    return primitive_payload(dict(predictions=result,operation_order=list(OPERATIONS),device='cpu',
        torch_version=str(torch.__version__),load_seconds=load,wall_seconds=perf_counter()-started))


@app.local_entrypoint()
def v2_features():
    verify()
    target=OUT/'short_probability_vectors.json'
    if target.exists():raise RuntimeError('Probability vectors already saved; refusing repeated Short pass')
    public=rows(SOURCE/'public_inputs.jsonl');assert len(public)==2400
    response=probability_vectors.remote(public,{n:sha(SHORT/n) for n in ('intent_bigru.pt','model_config.json')})
    old={r['id']:r for r in rows(SOURCE/'short_predictions.jsonl')}
    assert {r['id'] for r in response['predictions']}==set(old)
    max_difference=0.0
    for row in response['predictions']:
        saved=old[row['id']]
        assert row['operation']==saved['operation'] and row['query']==saved['query']
        assert row['status']==saved['status'] and row['reason']==saved['reason']
        difference=abs(row['confidence']-saved['features']['short_confidence'])
        max_difference=max(max_difference,difference);assert difference<=1e-6
    response['verification']=dict(n=2400,all_frozen_short_predictions_match=True,max_confidence_difference=max_difference)
    verify();write(target,response)
    print(response['verification'])
