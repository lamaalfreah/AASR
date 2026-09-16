"""Anchor-grouped nested CV; freeze Router V2 without CORE EVAL/TEST access."""
from collections import Counter
import importlib.metadata
import json
from pathlib import Path
from time import perf_counter
import warnings

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from threadpoolctl import threadpool_limits

from evaluation.router_v2_data import BASE,SOURCE,OUT,read,rows,write,sha,verify
from evaluation.train_lightweight_router import evaluate_routes
from language.router_v2 import V2_FEATURES,encode_v2,logit,CalibratedRouterEnsemble,RouterV2

SEED=20260914
MODEL_NAMES=('logistic_regression','hist_gradient_boosting','small_mlp')
THRESHOLDS=np.r_[np.linspace(0,1,101),np.nextafter(1.,np.inf)]


def make_model(name,seed):
    if name=='logistic_regression':return LogisticRegression(C=1.,max_iter=1000,random_state=seed)
    if name=='hist_gradient_boosting':return HistGradientBoostingClassifier(learning_rate=.08,max_iter=150,
        max_leaf_nodes=7,max_depth=3,min_samples_leaf=20,l2_regularization=1.,early_stopping=False,random_state=seed)
    if name=='small_mlp':return MLPClassifier(hidden_layer_sizes=(16,8),solver='lbfgs',alpha=.1,
        max_iter=3000,max_fun=60000,tol=1e-5,random_state=seed)
    raise ValueError(name)


def split_groups(data,n,seed):
    y=np.array([r['label'] for r in data]);groups=np.array([r['provenance']['source_anchor'] for r in data])
    splitter=StratifiedGroupKFold(n_splits=n,shuffle=True,random_state=seed)
    result=[]
    for train,test in splitter.split(np.zeros(len(data)),y,groups):
        assert not set(groups[train])&set(groups[test])
        result.append((train,test))
    assert sorted(np.concatenate([test for _,test in result]).tolist())==list(range(len(data)))
    return result


def fit_calibrated(data,name,seed):
    binary=[r for r in data if r['label']!='FAILURE']
    splits=split_groups(binary,3,seed)
    components=[];audit=[]
    for index,(train,cal) in enumerate(splits):
        fitting=[binary[i] for i in train];calibration=[binary[i] for i in cal]
        y=np.array([r['label']=='LONG' for r in fitting],dtype=int)
        cy=np.array([r['label']=='LONG' for r in calibration],dtype=int)
        assert len(set(y))==len(set(cy))==2
        prep=make_pipeline(DictVectorizer(sparse=False),SimpleImputer(strategy='median',add_indicator=True),StandardScaler())
        x=prep.fit_transform([encode_v2(r['features']) for r in fitting])
        model=make_model(name,seed+index)
        began=perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            model.fit(x,y,sample_weight=compute_sample_weight('balanced',y))
        raw=model.predict_proba(prep.transform([encode_v2(r['features']) for r in calibration]))[:,1]
        # Unweighted held-group sigmoid calibration restores observed binary prior.
        calibrator=LogisticRegression(C=1e6,solver='lbfgs',max_iter=1000,random_state=seed+index)
        calibrator.fit(logit(raw),cy)
        components.append(dict(preprocessor=prep,model=model,calibrator=calibrator))
        audit.append(dict(fit_n=len(fitting),calibration_n=len(calibration),fit_seconds=perf_counter()-began,
            fit_anchors=sorted({r['provenance']['source_anchor'] for r in fitting}),
            calibration_anchors=sorted({r['provenance']['source_anchor'] for r in calibration}),
            warnings=[str(w.message) for w in caught],iterations=int(np.asarray(model.n_iter_).max())))
    return CalibratedRouterEnsemble(components),audit


def summarize_folds(data,decisions,folds):
    metrics=evaluate_routes(data,decisions)
    fold_metrics=[evaluate_routes([data[i] for i in indexes],np.asarray(decisions)[indexes]) for indexes in folds]
    metrics['fold_accuracy_standard_error']=float(np.std([m['adaptive_answer_accuracy'] for m in fold_metrics],ddof=1)/np.sqrt(len(folds)))
    return metrics,fold_metrics


def robust_threshold(data,p,folds):
    candidates=[]
    for threshold in THRESHOLDS:
        result,by_fold=summarize_folds(data,p>=threshold,folds)
        candidates.append(dict(threshold=float(threshold),metrics=result,fold_metrics=by_fold))
    best=max(candidates,key=lambda r:(r['metrics']['adaptive_answer_accuracy'],-r['metrics']['long_route_rate'],
                                     -r['metrics']['long_regressions']))
    tolerance=best['metrics']['fold_accuracy_standard_error']
    floor=best['metrics']['adaptive_answer_accuracy']-tolerance
    eligible=[r for r in candidates if r['metrics']['adaptive_answer_accuracy']>=floor-1e-12]
    rank=lambda r:(r['metrics']['long_route_count'],r['metrics']['long_regressions'],-r['metrics']['adaptive_answer_accuracy'])
    optimum=min(rank(r) for r in eligible)
    plateau=[r for r in eligible if rank(r)==optimum]
    selected=plateau[len(plateau)//2]
    return dict(selected=selected,best_accuracy_threshold=best['threshold'],
        best_accuracy=best['metrics']['adaptive_answer_accuracy'],one_se=tolerance,accuracy_floor=floor,
        eligible_threshold_range=[min(r['threshold'] for r in eligible),max(r['threshold'] for r in eligible)],
        equivalent_selected_threshold_range=[plateau[0]['threshold'],plateau[-1]['threshold']],curve=candidates)


def calibration_metrics(data,p):
    indexes=[i for i,r in enumerate(data) if r['label']!='FAILURE']
    y=np.array([data[i]['label']=='LONG' for i in indexes],dtype=int);p=np.asarray(p)[indexes]
    ece=0.
    for low in np.linspace(0,.9,10):
        mask=(p>=low)&(p<(low+.1) if low<.9 else p<=1)
        if mask.any():ece+=float(mask.mean()*abs(p[mask].mean()-y[mask].mean()))
    return dict(n=len(y),brier=float(brier_score_loss(y,p)),log_loss=float(log_loss(y,p,labels=[0,1])),ece_10_bins=ece)


def protocol():
    return dict(seed=SEED,outer_folds=5,inner_threshold_folds=4,calibration_folds=3,
        grouping='source_anchor at every level; no shared anchor across fitting/calibration/validation',
        label_policy='Existing exact_and_answer SHORT/LONG labels; FAILURE excluded from model fit and calibration, retained for adaptive scoring',
        calibration='Three-member ensemble: each base model fits two anchor folds with balanced sample weights; unweighted sigmoid calibrator fits the third; average calibrated probabilities',
        threshold_policy='Fixed 0.01 grid plus always-Short boundary. Within one SE of best CV answer accuracy, minimize Long calls then regressions; middle of equivalent grid plateau.',
        model_policy='Within one SE of best nested-CV answer accuracy, minimize Long call rate then regressions; accuracy then fixed model order resolve ties.',
        final_threshold_policy='Same rule on selected-model pooled outer out-of-fold probabilities; descriptive selection metric, distinct from nested estimate',
        final_fit='Three-member grouped calibrated ensemble on merged development non-FAILURE rows',
        model_parameters={name:make_model(name,SEED).get_params() for name in MODEL_NAMES},
        input_feature_keys=sorted(V2_FEATURES),
        feature_policy='All eight named Short probabilities plus saved Short features; multi-hot missing-slot groups and their count; TRAIN-fold-only dictionary, median imputation, missingness indicators, scaling',
        no_new_questions=True,no_core_eval_or_test_access=True,no_long_generations=True)


def main():
    if (OUT/'freeze_manifest.json').exists():raise RuntimeError('Router V2 already frozen; refusing retraining')
    verify()
    plan=protocol();plan_path=OUT/'cv_protocol.json'
    plan['input_freeze_sha256']=sha(OUT/'input_freeze.json')
    plan['probability_vectors_sha256']=sha(OUT/'short_probability_vectors.json')
    plan['implementation_sha256']={p:sha(BASE/p) for p in ('evaluation/train_router_v2.py','language/router_v2.py')}
    if plan_path.exists():assert read(plan_path)==json.loads(json.dumps(plan))
    else:write(plan_path,plan)
    data=rows(SOURCE/'router_train.jsonl')+rows(SOURCE/'router_dev.jsonl')
    probability_response=read(OUT/'short_probability_vectors.json')
    probability_rows={r['id']:r for r in probability_response['predictions']}
    assert len(data)==2400 and {r['id'] for r in data}==set(probability_rows)
    for row in data:
        row['features']=dict(row['features'],operation_probabilities=probability_rows[row['id']]['operation_probabilities'])
        encode_v2(row['features'])
    outer=split_groups(data,5,SEED)
    fold_ids=np.empty(len(data),dtype=int)
    for number,(_,test) in enumerate(outer):fold_ids[test]=number
    write(OUT/'outer_fold_assignment.json',[dict(id=r['id'],anchor=r['provenance']['source_anchor'],fold=int(fold_ids[i])) for i,r in enumerate(data)])
    (OUT/'folds').mkdir(exist_ok=True)
    all_results={};calibrated_oof={};raw_oof={};nested_decisions={};audits={}
    started=perf_counter()
    with threadpool_limits(limits=1):
        for name in MODEL_NAMES:
            p=np.full(len(data),np.nan);raw=np.full(len(data),np.nan);decisions=np.zeros(len(data),dtype=bool)
            model_fold_results=[]
            for number,(outer_train,outer_test) in enumerate(outer):
                checkpoint=OUT/'folds'/f'{name}-{number}.joblib'
                if checkpoint.exists():
                    saved=joblib.load(checkpoint)
                    assert saved['protocol_sha256']==sha(plan_path)
                    assert saved['test_ids']==[data[i]['id'] for i in outer_test]
                else:
                    training=[data[i] for i in outer_train]
                    inner=split_groups(training,4,SEED+100+number)
                    inner_p=np.full(len(training),np.nan);inner_audits=[]
                    for j,(inner_train,inner_test) in enumerate(inner):
                        ensemble,audit=fit_calibrated([training[i] for i in inner_train],name,SEED+1000+number*100+j*10)
                        inner_p[inner_test]=ensemble.probabilities([training[i]['features'] for i in inner_test])
                        held={training[i]['provenance']['source_anchor'] for i in inner_test}
                        assert all(not held & (set(a['fit_anchors'])|set(a['calibration_anchors'])) for a in audit)
                        inner_audits.append(audit)
                    assert np.isfinite(inner_p).all()
                    selection=robust_threshold(training,inner_p,[test for _,test in inner])
                    threshold=selection['selected']['threshold']
                    ensemble,audit=fit_calibrated(training,name,SEED+2000+number)
                    held={data[i]['provenance']['source_anchor'] for i in outer_test}
                    assert all(not held & (set(a['fit_anchors'])|set(a['calibration_anchors'])) for a in audit)
                    features=[data[i]['features'] for i in outer_test]
                    saved=dict(protocol_sha256=sha(plan_path),test_ids=[data[i]['id'] for i in outer_test],
                        probabilities=ensemble.probabilities(features),raw_probabilities=ensemble.probabilities(features,calibrated=False),
                        threshold=threshold,selection=selection,fit_audit=audit,inner_fit_audits=inner_audits,
                        ensemble=ensemble)
                    joblib.dump(saved,checkpoint)
                p[outer_test]=saved['probabilities'];raw[outer_test]=saved['raw_probabilities']
                decisions[outer_test]=saved['probabilities']>=saved['threshold']
                fold_metric=evaluate_routes([data[i] for i in outer_test],decisions[outer_test])
                model_fold_results.append(dict(fold=number,threshold=saved['threshold'],metrics=fold_metric))
                print(name,'outer fold',number+1,'complete',json.dumps(dict(accuracy=fold_metric['adaptive_answer_accuracy'],long_rate=fold_metric['long_route_rate'],threshold=saved['threshold'])),flush=True)
            assert np.isfinite(p).all() and np.isfinite(raw).all()
            metrics,fold_metrics=summarize_folds(data,decisions,[test for _,test in outer])
            all_results[name]=dict(nested_cv=metrics,fold_results=model_fold_results,
                calibration=calibration_metrics(data,p),uncalibrated=calibration_metrics(data,raw))
            calibrated_oof[name]=p;raw_oof[name]=raw;nested_decisions[name]=decisions
            write(OUT/'cv_progress.json',dict(completed_models=list(all_results),results=all_results))
        best=max(MODEL_NAMES,key=lambda name:all_results[name]['nested_cv']['adaptive_answer_accuracy'])
        floor=all_results[best]['nested_cv']['adaptive_answer_accuracy']-all_results[best]['nested_cv']['fold_accuracy_standard_error']
        eligible=[name for name in MODEL_NAMES if all_results[name]['nested_cv']['adaptive_answer_accuracy']>=floor-1e-12]
        winner=min(eligible,key=lambda name:(all_results[name]['nested_cv']['long_route_rate'],all_results[name]['nested_cv']['long_regressions'],
            -all_results[name]['nested_cv']['adaptive_answer_accuracy'],MODEL_NAMES.index(name)))
        selection=robust_threshold(data,calibrated_oof[winner],[test for _,test in outer])
        threshold=selection['selected']['threshold']
        write(OUT/'final_threshold_selection.json',selection)
        final,final_audit=fit_calibrated(data,winner,SEED+3000)
        artifact=dict(ensemble=final,threshold=threshold,model_name=winner,feature_keys=sorted(V2_FEATURES),protocol=plan)
        joblib.dump(artifact,OUT/'router_v2.joblib')
        loaded=RouterV2(OUT/'router_v2.joblib')
        # Serialization verification only, never scored for model selection.
        sample=[r['features'] for r in data[:32]]
        assert np.allclose(loaded.long_probabilities(sample),final.probabilities(sample),rtol=0,atol=1e-12)
    records=[]
    for i,row in enumerate(data):
        records.append(dict(id=row['id'],anchor=row['provenance']['source_anchor'],fold=int(fold_ids[i]),label=row['label'],
            models={name:dict(probability=float(calibrated_oof[name][i]),raw_probability=float(raw_oof[name][i]),
                nested_route='LONG' if nested_decisions[name][i] else 'SHORT') for name in MODEL_NAMES}))
    (OUT/'oof_predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    write(OUT/'final_fit_audit.json',final_audit)
    warning_counts=Counter()
    for path in (OUT/'folds').glob('*.joblib'):
        saved=joblib.load(path)
        for audit in [saved['fit_audit'],*saved['inner_fit_audits']]:
            for component in audit:
                for warning in component['warnings']:warning_counts[warning.split('\n')[0]]+=1
    for component in final_audit:
        for warning in component['warnings']:warning_counts[warning.split('\n')[0]]+=1
    chosen_metrics=all_results[winner]['nested_cv']
    summary=dict(selected_model=winner,selected_threshold=threshold,models=all_results,
        model_selection=dict(best_accuracy_model=best,one_se_accuracy_floor=floor,eligible_models=eligible),
        selected_nested_cv=chosen_metrics,
        selected_threshold_pooled_oof=selection['selected']['metrics'],
        selected_threshold_note='Pooled OOF threshold-selection score is descriptive, not a nested held-out estimate.',
        label_counts=dict(Counter(r['label'] for r in data)),anchors=len({r['provenance']['source_anchor'] for r in data}),
        fit_warnings=dict(warning_counts),final_fit_warnings=[w for c in final_audit for w in c['warnings']],
        wall_seconds=perf_counter()-started,final_calibrated_components=3,new_questions=0,
        new_long_calls=0,new_cpu_short_feature_passes=2400,core_eval_or_test_accessed=False)
    write(OUT/'summary.json',summary)
    verify()
    write(OUT/'verification.json',dict(frozen_inputs_unchanged=True,all_2400_rows_scored_out_of_fold=True,
        anchor_disjoint_at_every_cv_and_calibration_level=True,no_failure_rows_fitted=True,
        preprocessing_fit_within_training_folds=True,calibration_unweighted_and_out_of_fit=True,
        serialized_model_probability_parity=True,core_eval_or_test_accessed=False))
    dependencies={n:importlib.metadata.version(n) for n in ('scikit-learn','numpy','scipy','joblib','threadpoolctl')}
    (OUT/'requirements.txt').write_text(''.join(f'{n}=={v}\n' for n,v in dependencies.items()))
    names=final.components[0]['preprocessor'].named_steps['dictvectorizer'].get_feature_names_out().tolist()
    write(OUT/'feature_schema.json',dict(input_keys=sorted(V2_FEATURES),example_encoded_names=names,
        probability_operation_names=probability_response['operation_order'],
        policy='Each ensemble member owns TRAIN-fold-fitted preprocessing. Dictionary categories may differ across members.'))
    report(summary)
    code=['language/router_v2.py','evaluation/router_v2_data.py','evaluation/modal_router_v2_features.py','evaluation/train_router_v2.py']
    write(OUT/'freeze_manifest.json',dict(selected_model=winner,threshold=threshold,input_freeze_sha256=sha(OUT/'input_freeze.json'),
        code_sha256={p:sha(BASE/p) for p in code},
        output_sha256={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file()},dependencies=dependencies))
    print(json.dumps(dict(selected_model=winner,threshold=threshold,nested_cv=chosen_metrics,fit_warnings=dict(warning_counts)),indent=2))


def report(summary):
    def pct(v):return f'{100*v:.2f}%' if v is not None else 'N/A'
    def table(headers,data):return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+['| '+' | '.join(map(str,r))+' |' for r in data])
    m=summary['selected_nested_cv']; pooled=summary['selected_threshold_pooled_oof']
    lines=['**Router V2 — grouped development CV and freeze**','',
        f'Selected `{summary["selected_model"]}`; calibrated LONG threshold `{summary["selected_threshold"]:.17g}`. Final artifact is a three-member calibrated ensemble of this lightweight model family.','',
        'Merged existing Router TRAIN and DEV: 2,400 rows across 768 anchors. Binary labels: 550 SHORT, 1,499 LONG; 351 FAILURE rows excluded from fitting/calibration but retained for final-answer scoring. No new questions. No CORE EVAL or TEST reads.','',
        'Five outer anchor folds estimate performance. Each outer training portion has four grouped inner folds for threshold selection; each inner estimator uses three further grouped folds for unweighted sigmoid calibration of class-balanced base fits. No held-out anchor enters fitting, preprocessing, calibration, or threshold selection for its outer prediction.','',
        'The predeclared one-standard-error rule selects lower Long-call rates, then lower regressions, among accuracy-eligible thresholds/models. All candidates use a fixed configuration. The final threshold uses pooled outer OOF scores; its apparent accuracy is a selection score and is reported separately from the nested estimate.','',
        table(['Model','Nested adaptive','Long route rate','Regressions','Routing accuracy','LONG precision','LONG recall'],[
            [name,pct(r['nested_cv']['adaptive_answer_accuracy']),pct(r['nested_cv']['long_route_rate']),r['nested_cv']['long_regressions'],
             pct(r['nested_cv']['routing_accuracy']),pct(r['nested_cv']['long_precision']),pct(r['nested_cv']['long_recall'])] for name,r in summary['models'].items()]),'',
        table(['Calibration (binary OOF)','Raw Brier','Calibrated Brier','Calibrated log loss','ECE, 10 bins'],[
            [name,f'{r["uncalibrated"]["brier"]:.4f}',f'{r["calibration"]["brier"]:.4f}',f'{r["calibration"]["log_loss"]:.4f}',f'{r["calibration"]["ece_10_bins"]:.4f}'] for name,r in summary['models'].items()]),'',
        table(['Policy, all 2400 development rows','Correct','Accuracy'],[
            [name,m[count],pct(m[accuracy])] for name,count,accuracy in (
                ('Always Short','always_short_correct','always_short_accuracy'),('Always Long','always_long_correct','always_long_accuracy'),
                ('Selected family, nested CV','adaptive_answer_correct','adaptive_answer_accuracy'),('Answer oracle','oracle_correct','oracle_accuracy'))]),'',
        f'Nested estimated Short/Long answer routes: {pct(m["short_route_rate"])} / {pct(m["long_route_rate"])}. Short inference must run for every question to supply features. Long is additional. Rescues: {m["long_rescues"]}; regressions: {m["long_regressions"]}; unnecessary Long calls: {m["unnecessary_long_calls"]}. Binary routing metrics exclude FAILURE. FAILURE nested analysis: `{json.dumps(m["failure_analysis"])}`.','',
        f'Final fixed-threshold pooled OOF selection score: {pooled["adaptive_answer_correct"]}/2400 ({pct(pooled["adaptive_answer_accuracy"])}), Long rate {pct(pooled["long_route_rate"])}. Do not present this as an independently evaluated final-model score. Final model fit uses merged development data only.','',
        '**Selected inference-safe features**','',
        'All eight named operation probabilities; confidence; entropy; top1-top2 margin; query validity; predicted operation; separate missing-slot group indicators and their count; entity ambiguity; question length in characters and words; constraint count. Missingness is retained and imputed within each fit. No gold task/register/complexity/anchor/outcome enters the feature dictionary. Missing-slot groups remain the frozen extractor diagnostics, not invented exact partial slots.','',
        'The full operation vector was observed in one CPU forward pass of unchanged Short on the existing questions. Operation, query, status, reason and confidence matched all frozen Short predictions. The original outcomes and feature values are retained. No Long generation or GPU calls were needed.','',
        '**Uncertainty and limits**','',
        f'Model-selection accuracy floor: {pct(summary["model_selection"]["one_se_accuracy_floor"])}. Eligible families: {", ".join(summary["model_selection"]["eligible_models"])}. Fold metrics and thresholds are in `summary.json`; complete threshold curves and calibration group audits are in `folds/`.','',
        'Nested scores estimate this training-and-threshold procedure; choosing a model family from three CV results still introduces selection uncertainty. Existing controlled wording families are shared across anchors, so this CV does not test unseen language styles. Calibration estimates the binary SHORT/LONG target among non-FAILURE cases; FAILURE is an out-of-target input, not an abstention class. No conclusion about production data sufficiency follows from these template-derived folds alone. No new data were added.','',
        f'Training warning counts: `{json.dumps(summary["fit_warnings"])}`. Final ensemble warnings: `{json.dumps(summary["final_fit_warnings"])}`. No repeated tuning was performed after inspecting results.','',
        'Frozen artifact: `router_v2.joblib`; preprocessing, three classifiers, calibrators and threshold are bundled. Versions and hashes are in `requirements.txt` and `freeze_manifest.json`. The selected ensemble is not integrated into the Geo Engine.','',
        '**Stopped after freezing Router V2. CORE EVAL results remain untouched.**','']
    (OUT/'REPORT.md').write_text('\n'.join(lines))


if __name__=='__main__':main()
