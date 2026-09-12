#!/usr/bin/env python3
"""Fixed CPU training on original TRAIN only, with anchor-disjoint internal holdout."""
from __future__ import annotations
from collections import Counter
import gzip,hashlib,json,os,random,sys,time
from pathlib import Path
BASE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(BASE))
from language.neural_short import SEED,MAX_TOKENS,tokenize,encode,make_model
from spatial.schema import OPERATIONS
from evaluation.build_language_challenge import frozen_hashes,OUT as CHALLENGE
OUT=BASE/'experiments/E5_lightweight_adaptive/step2_neural_short'


def main():
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    frozen_hashes();OUT.mkdir(exist_ok=True)
    if (OUT/'intent_bigru.pt').exists():raise RuntimeError('Existing trained checkpoint found; refuse an accidental retrain')
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    torch.manual_seed(SEED);np.random.seed(SEED);random.seed(SEED)
    torch.use_deterministic_algorithms(True)
    began=time.perf_counter()
    # Read only IDs from CORE; no challenge question/intent/gold is loaded for fitting.
    challenge_ids=set()
    with (CHALLENGE/'challenge/source_contexts.jsonl').open() as f:
        for line in f:challenge_ids.add(json.loads(line)['source_record_id'])
    rows=[];excluded_anchors=set()
    with gzip.open(BASE/'data/raw/train.jsonl.gz','rt',encoding='utf-8') as f:
        for line in f:
            r=json.loads(line)
            if r['split']!='train':raise ValueError('Non-train row refused')
            rows.append((r['id'],r['wikidata_id'],r['question'],r['task_type']))
            if r['id'] in challenge_ids:excluded_anchors.add(r['wikidata_id'])
    fit=[];holdout=[]
    for rid,anchor,question,task in rows:
        if anchor in excluded_anchors:continue
        subset=holdout if int(hashlib.sha256(('short-internal:'+anchor).encode()).hexdigest(),16)%10==0 else fit
        subset.append((question,OPERATIONS.index(task),anchor))
    vocab={'<pad>':0,'<unk>':1}
    counts=Counter(word for q,_,_ in fit for word in tokenize(q))
    for word in sorted(counts):vocab[word]=len(vocab)
    def encoded(data):return [(encode(q,vocab),label) for q,label,_ in data]
    def collate(batch):
        lengths=torch.tensor([len(ids) for ids,_ in batch]);tokens=torch.zeros(len(batch),int(lengths.max()),dtype=torch.long)
        for i,(ids,_) in enumerate(batch):tokens[i,:len(ids)]=torch.tensor(ids)
        return tokens,lengths,torch.tensor([label for _,label in batch])
    fit_loader=DataLoader(encoded(fit),batch_size=128,shuffle=True,collate_fn=collate,num_workers=0,
                          generator=torch.Generator().manual_seed(SEED))
    hold_loader=DataLoader(encoded(holdout),batch_size=128,collate_fn=collate,num_workers=0)
    model=make_model(len(vocab)).cpu()
    freq=Counter(y for _,y,_ in fit)
    weights=torch.tensor([len(fit)/(8*freq[i]) for i in range(8)],dtype=torch.float32)
    loss_fn=torch.nn.CrossEntropyLoss(weight=weights)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=.01)
    best_loss=float('inf');best_state=None;best_epoch=0;patience=0;history=[]
    train_started=time.perf_counter()
    for epoch in range(1,13):
        start=time.perf_counter();model.train();train_loss=train_correct=seen=0
        for x,lengths,y in fit_loader:
            optimizer.zero_grad(set_to_none=True);logits=model(x,lengths);loss=loss_fn(logits,y)
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);optimizer.step()
            train_loss+=float(loss)*len(y);train_correct+=int((logits.argmax(1)==y).sum());seen+=len(y)
        model.eval();hloss=hcorrect=hseen=0
        with torch.inference_mode():
            for x,lengths,y in hold_loader:
                logits=model(x,lengths);hloss+=float(loss_fn(logits,y))*len(y);hcorrect+=int((logits.argmax(1)==y).sum());hseen+=len(y)
        value=hloss/hseen
        entry={'epoch':epoch,'train_loss':train_loss/seen,'train_intent_accuracy':train_correct/seen,
               'internal_holdout_loss':value,'internal_holdout_intent_accuracy':hcorrect/hseen,
               'seconds':time.perf_counter()-start};history.append(entry)
        print(json.dumps(entry),flush=True)
        if value<best_loss-1e-5:
            best_loss=value;best_epoch=epoch;patience=0;best_state={k:v.detach().clone() for k,v in model.state_dict().items()}
        else:patience+=1
        if patience>=3:break
    training_seconds=time.perf_counter()-train_started
    torch.save(best_state,OUT/'intent_bigru.pt')
    config={'model':'single_layer_bidirectional_GRU','embedding_dim':48,'hidden_dim_per_direction':48,
            'dropout':.1,'max_tokens':MAX_TOKENS,'vocabulary':vocab,'operations':list(OPERATIONS)}
    (OUT/'model_config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2)+'\n')
    metrics={'seed':SEED,'device':'cpu','torch_threads':2,'torch_version':torch.__version__,
             'numpy_version':np.__version__,'source_train_records':len(rows),'excluded_core_source_anchors':len(excluded_anchors),
             'excluded_records_at_core_source_anchors':len(rows)-len(fit)-len(holdout),'fit_records':len(fit),'internal_holdout_records':len(holdout),
             'fit_anchors':len({a for _,_,a in fit}),'internal_holdout_anchors':len({a for _,_,a in holdout}),
             'anchor_disjoint':not ({a for _,_,a in fit}&{a for _,_,a in holdout}),
             'fit_task_counts':{OPERATIONS[k]:v for k,v in sorted(freq.items())},
             'parameter_count':sum(p.numel() for p in model.parameters()),'vocabulary_size':len(vocab),
             'selected_epoch':best_epoch,'epochs_run':len(history),'selection_rule':'Minimum original-TRAIN internal anchor-holdout loss; max 12 epochs, patience 3, improvement 1e-5. Fixed before validation/CORE evaluation.',
             'training_seconds':training_seconds,'total_preparation_training_seconds':time.perf_counter()-began,
             'history':history,'challenge_text_used_for_training':False,'validation_used_for_training':False,
             'core_source_anchors_excluded_from_fit_and_internal_holdout':True,
             'checkpoint_sha256':hashlib.sha256((OUT/'intent_bigru.pt').read_bytes()).hexdigest(),
             'short_source_sha256':hashlib.sha256((BASE/'language/neural_short.py').read_bytes()).hexdigest()}
    (OUT/'training_metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
    frozen_hashes();print(json.dumps({k:v for k,v in metrics.items() if k not in ('history','fit_task_counts')},indent=2))


if __name__=='__main__':main()
