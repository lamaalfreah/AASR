"""CPU-only feature augmentation; reuse unchanged Step 3A extraction."""
import json
from evaluation.modal_router_data import app, short_inference
from evaluation.evaluate_frozen_router import OUT, SHORT, read, rows, sha, verify, write


@app.local_entrypoint()
def core_features():
    target=OUT/'short_feature_response.json'
    if target.exists():
        raise RuntimeError('Saved feature response exists; refusing repeated Short pass')
    freeze=read(OUT/'pre_eval_freeze.json');verify(freeze['sha256'])
    assert sha(OUT/'public_inputs.jsonl')==freeze['public_inputs_sha256']
    public=rows(OUT/'public_inputs.jsonl')
    assert len(public)==128 and all(set(r)=={'id','question','context'} for r in public)
    response=short_inference.remote(public,{name:sha(SHORT/name) for name in ('intent_bigru.pt','model_config.json')})
    verify(freeze['sha256'])
    write(target,response)
    print(json.dumps(dict(feature_rows=len(response['predictions']),device=response['device'],wall_seconds=response['wall_seconds'])))
