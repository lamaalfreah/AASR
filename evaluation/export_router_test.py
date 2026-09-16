"""Read-only batched checkpoint export; no prediction or evaluator changes."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import re
from evaluation.modal_router_test import app,test_image,checkpoints,inputs,mirror
from evaluation.router_test_data import OUT,verify,read
from language.modal_checkpoint import atomic_json


@app.function(image=test_image,volumes={'/checkpoints':checkpoints},cpu=2,memory=4096,timeout=300,retries=0)
def inventory():
    checkpoints.reload();_,_,_,directory=inputs()
    return dict(progress=read(directory/'progress.json'),
        files={kind:[p.stem for p in sorted((directory/kind).glob('*.json'))] for kind in ('long','calls','sessions')})


@app.function(image=test_image,volumes={'/checkpoints':checkpoints},cpu=2,memory=4096,timeout=300,retries=0)
def batch(kind:str,names:list):
    assert kind in ('long','calls','sessions') and len(names)<=32
    assert all(re.fullmatch('[a-f0-9]{16,64}',name) for name in names)
    checkpoints.reload();_,_,_,directory=inputs()
    with ThreadPoolExecutor(max_workers=8) as pool:
        return list(pool.map(read,[directory/kind/(name+'.json') for name in names]))


@app.local_entrypoint()
def sync():
    verify();state=inventory.remote()
    print('Committed TEST checkpoint:',state['progress'],flush=True)
    mapping={'long':'long_predictions','calls':'long_calls','sessions':'long_sessions'}
    for kind,folder in mapping.items():
        existing={p.stem for p in (OUT/folder).glob('*.json')}
        names=[name for name in state['files'][kind] if kind=='sessions' or name not in existing]
        for index in range(0,len(names),32):
            result=batch.remote(kind,names[index:index+32])
            if kind=='sessions':
                for row in result:atomic_json(OUT/folder/(row['session_id']+'.json'),row)
            else:mirror(folder,result)
            print('Exported',kind,min(index+32,len(names)),'/',len(names),flush=True)
    state['progress']['local_long_completed']=len(list((OUT/'long_predictions').glob('*.json')))
    atomic_json(OUT/'progress.json',state['progress']);verify()
    print('Checkpoint export complete',state['progress'],flush=True)
