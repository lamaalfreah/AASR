"""Step 3B: fit on Router TRAIN, select on Router DEV, freeze without refitting."""
import hashlib
import importlib.metadata
import json
from collections import Counter
from pathlib import Path
from time import perf_counter
import warnings

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from threadpoolctl import threadpool_limits

from language.lightweight_router import FEATURES, LightweightRouter, encode_features

BASE = Path(__file__).resolve().parents[1]
SOURCE = BASE / 'experiments/E5_lightweight_adaptive/step3a_router_data'
OUT = BASE / 'experiments/E5_lightweight_adaptive/step3b_router'
SEED = 20260913


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def read_rows(name):
    return [json.loads(line) for line in (SOURCE / name).read_text().splitlines()]


def evaluate_routes(rows, use_long):
    n = len(rows)
    use_long = np.asarray(use_long, dtype=bool)
    short = np.array([r['outcomes']['short']['answer_correct'] for r in rows], dtype=bool)
    long = np.array([r['outcomes']['long']['answer_correct'] for r in rows], dtype=bool)
    binary = np.array([r['label'] != 'FAILURE' for r in rows])
    target = np.array([r['label'] == 'LONG' for r in rows])
    tp = int(np.sum(binary & use_long & target))
    fp = int(np.sum(binary & use_long & ~target))
    fn = int(np.sum(binary & ~use_long & target))
    tn = int(np.sum(binary & ~use_long & ~target))
    correct = np.where(use_long, long, short)
    def rate(a, b):
        return float(a / b) if b else None
    return dict(
        n=n, binary_n=int(binary.sum()), failure_n=int((~binary).sum()),
        routing_accuracy=rate(tp + tn, binary.sum()),
        long_precision=rate(tp, tp + fp), long_recall=rate(tp, tp + fn),
        binary_confusion=dict(short_to_short=tn, short_to_long=fp, long_to_short=fn, long_to_long=tp),
        short_route_count=int((~use_long).sum()), long_route_count=int(use_long.sum()),
        short_route_rate=rate((~use_long).sum(), n), long_route_rate=rate(use_long.sum(), n),
        # Short must run to produce features, even when the selected answer uses Long.
        short_inference_call_rate=1.0,
        adaptive_answer_correct=int(correct.sum()), adaptive_answer_accuracy=rate(correct.sum(), n),
        always_short_correct=int(short.sum()), always_short_accuracy=rate(short.sum(), n),
        always_long_correct=int(long.sum()), always_long_accuracy=rate(long.sum(), n),
        oracle_correct=int((short | long).sum()), oracle_accuracy=rate((short | long).sum(), n),
        unnecessary_long_calls=int((use_long & short).sum()),
        long_regressions=int((use_long & short & ~long).sum()),
        long_regression_rate_among_short_correct=rate((use_long & short & ~long).sum(), short.sum()),
        long_rescues=int((use_long & ~short & long).sum()),
        failure_analysis=dict(n=int((~binary).sum()), long_calls=int((use_long & ~binary).sum()),
            adaptive_answer_correct=int((correct & ~binary).sum()),
            always_short_correct=int((short & ~binary).sum()), always_long_correct=int((long & ~binary).sum()),
            oracle_correct=int(((short | long) & ~binary).sum())),
    )


def rank(metrics):
    # Declared before fitting: lexicographic priorities, no fitted utility weights.
    return (metrics['adaptive_answer_correct'], -metrics['unnecessary_long_calls'],
            -metrics['long_regressions'], -metrics['long_route_count'])


def select_threshold(rows, probabilities):
    thresholds = sorted(set([0.0, np.nextafter(1.0, np.inf), *map(float, probabilities)]))
    curve = []
    for threshold in thresholds:
        result = evaluate_routes(rows, probabilities >= threshold)
        curve.append(dict(threshold=threshold, **result))
    # Final threshold tie-break is deterministic; identical decisions have same objective.
    return max(curve, key=lambda m: (*rank(m), m['threshold'])), curve


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '|' + '|'.join(['---'] * len(headers)) + '|'] +
                     ['| ' + ' | '.join(map(str, row)) + ' |' for row in rows])


def main():
    if OUT.exists():
        raise RuntimeError('Step 3B output already exists; refusing to overwrite or retrain')
    train, dev = read_rows('router_train.jsonl'), read_rows('router_dev.jsonl')
    assert len(train) == 1920 and len(dev) == 480
    assert len({r['id'] for r in train + dev}) == 2400
    anchors = [{r['provenance']['source_anchor'] for r in rows} for rows in (train, dev)]
    assert not anchors[0] & anchors[1]
    excluded = set(json.loads((SOURCE / 'anchor_split.json').read_text())['excluded_core_anchors'])
    assert not (anchors[0] | anchors[1]) & excluded
    for split, rows in [('train', train), ('dev', dev)]:
        for row in rows:
            assert row['provenance']['source_split'] == 'train'
            assert row['provenance']['router_split'] == split
            assert row['label'] == row['strict_label']
            assert set(row['features']) == FEATURES
            short, long = [row['outcomes'][name] for name in ('short', 'long')]
            expected = 'SHORT' if short['query_exact'] and short['answer_correct'] else (
                'LONG' if long['query_exact'] and long['answer_correct'] else 'FAILURE')
            assert row['label'] == expected
    source_hashes = {str(p.relative_to(BASE)): sha(p) for p in SOURCE.rglob('*') if p.is_file()}
    frozen = json.loads((SOURCE / 'manifest.json').read_text())['frozen_sources']
    for path, digest in frozen.items():
        assert sha(BASE / path) == digest, path
    fitting = [r for r in train if r['label'] != 'FAILURE']
    y = np.array([r['label'] == 'LONG' for r in fitting], dtype=int)
    weights = compute_sample_weight('balanced', y)
    models = {
        'logistic_regression': LogisticRegression(C=1.0, max_iter=1000, random_state=SEED),
        'hist_gradient_boosting': HistGradientBoostingClassifier(
            learning_rate=0.08, max_iter=150, max_leaf_nodes=7, max_depth=3,
            min_samples_leaf=20, l2_regularization=1.0, early_stopping=False, random_state=SEED),
        'small_mlp': MLPClassifier(hidden_layer_sizes=(16, 8), solver='lbfgs',
            alpha=0.01, max_iter=1000, max_fun=25000, tol=1e-5, random_state=SEED),
    }
    OUT.mkdir()
    protocol = dict(seed=SEED, primary_label='Step 3A exact_and_answer SHORT/LONG',
        fit_scope='Router TRAIN excluding FAILURE; preprocessing fitted on the same rows',
        selection_scope='all Router DEV final-answer outcomes, including separately analyzed FAILURE',
        binary_metrics_scope='Router DEV excluding FAILURE',
        objective=['maximize adaptive final-answer correct count', 'minimize unnecessary Long calls (Short answer correct)',
                   'minimize Long regressions (Short answer correct, Long answer wrong)', 'minimize total Long calls'],
        model_tie_break=['logistic_regression', 'hist_gradient_boosting', 'small_mlp'],
        threshold_rule='LONG iff probability >= threshold; search each distinct DEV probability plus all-Long/all-Short',
        imbalance='inverse class-frequency sample weights, n/(2*class_n), for every model',
        feature_keys=sorted(FEATURES),
        preprocessing='TRAIN-only DictVectorizer, median imputation with missingness indicators, StandardScaler; unknown categories ignored',
        model_parameters={name: model.get_params() for name, model in models.items()},
        train_binary_counts=dict(Counter(r['label'] for r in fitting)),
        train_failure_excluded=len(train)-len(fitting), dev_counts=dict(Counter(r['label'] for r in dev)),
        train_anchors=len(anchors[0]), dev_anchors=len(anchors[1]), excluded_core_anchors=len(excluded),
        refit_on_dev=False, parser_generation_calls=0)
    write_json(OUT / 'selection_protocol.json', protocol)
    preprocessor = make_pipeline(DictVectorizer(sparse=False), SimpleImputer(strategy='median', add_indicator=True), StandardScaler())
    xtrain = preprocessor.fit_transform([encode_features(r['features']) for r in fitting])
    xdev = preprocessor.transform([encode_features(r['features']) for r in dev])
    results, probabilities = {}, {}
    with threadpool_limits(limits=1):
        for name, model in models.items():
            start = perf_counter()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                model.fit(xtrain, y, sample_weight=weights)
            seconds = perf_counter() - start
            p = model.predict_proba(xdev)[:, list(model.classes_).index(1)]
            probabilities[name] = p
            best, curve = select_threshold(dev, p)
            results[name] = dict(selected=best, at_default_threshold=evaluate_routes(dev, p >= .5),
                fit_seconds=seconds, warnings=[str(w.message) for w in caught],
                iterations=int(np.asarray(model.n_iter_).max()), threshold_candidates=len(curve))
            write_json(OUT / (name + '_threshold_curve.json'), curve)
            joblib.dump(dict(preprocessor=preprocessor, model=model, threshold=best['threshold'], model_name=name,
                             feature_keys=sorted(FEATURES)), OUT / (name + '.joblib'))
            print(name, json.dumps(dict(fit_seconds=seconds, threshold=best['threshold'],
                  adaptive_correct=best['adaptive_answer_correct'], long_calls=best['long_route_count'],
                  warnings=results[name]['warnings'])), flush=True)
    winner = max(models, key=lambda name: rank(results[name]['selected']))
    selected_path = OUT / 'router.joblib'
    selected_path.write_bytes((OUT / (winner + '.joblib')).read_bytes())
    runtime = LightweightRouter(selected_path)
    p = runtime.long_probabilities([r['features'] for r in dev])
    assert np.allclose(p, probabilities[winner], rtol=0, atol=1e-12)
    routes = runtime.route([r['features'] for r in dev])
    assert routes == np.where(probabilities[winner] >= results[winner]['selected']['threshold'], 'LONG', 'SHORT').tolist()
    selected = results[winner]['selected']
    dev_predictions = []
    for index, row in enumerate(dev):
        route = routes[index]
        dev_predictions.append(dict(id=row['id'], true_label=row['label'], route=route,
            long_probability=float(p[index]), adaptive_answer_correct=row['outcomes'][route.lower()]['answer_correct'],
            model_probabilities={name: float(values[index]) for name, values in probabilities.items()}))
    (OUT / 'dev_predictions.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in dev_predictions))
    breakdowns = {}
    for field in ('task_type', 'language_register'):
        breakdowns[field] = {}
        for value in sorted({r['provenance'][field] for r in dev}):
            indexes = [i for i,r in enumerate(dev) if r['provenance'][field] == value]
            breakdowns[field][value] = evaluate_routes([dev[i] for i in indexes], [routes[i] == 'LONG' for i in indexes])
    train_failures = [r for r in train if r['label'] == 'FAILURE']
    failure_routes = runtime.route([r['features'] for r in train_failures])
    summary = dict(selected_model=winner, selected_threshold=selected['threshold'], selected=selected,
        models=results, dev_breakdowns=breakdowns, train_failure_analysis=dict(
            used_for_fit=False, metrics=evaluate_routes(train_failures, [r == 'LONG' for r in failure_routes])),
        nonfailure_dev=evaluate_routes([r for r in dev if r['label'] != 'FAILURE'],
            [route == 'LONG' for r,route in zip(dev,routes) if r['label'] != 'FAILURE']),
        parser_generation_calls=0, gpu_calls=0, train_dev_anchor_disjoint=True, refit_on_dev=False)
    write_json(OUT / 'summary.json', summary)
    feature_names = list(preprocessor.named_steps['dictvectorizer'].get_feature_names_out())
    write_json(OUT / 'feature_schema.json', dict(input_keys=sorted(FEATURES), encoded_names=feature_names,
        transformed_dimensions=int(xtrain.shape[1]), train_fit_rows=len(fitting)))
    dependencies = {name: importlib.metadata.version(name) for name in ('scikit-learn','numpy','scipy','joblib','threadpoolctl')}
    (OUT / 'requirements.txt').write_text(''.join(f'{name}=={version}\n' for name,version in dependencies.items()))
    for path, digest in {**source_hashes, **frozen}.items():
        assert sha(BASE / path) == digest, path
    verification = dict(step3a_files_unchanged=len(source_hashes), frozen_components_unchanged=True,
        serialized_model_probability_parity=True, serialized_model_route_parity=True,
        core_eval_and_test_not_accessed=True, inference_safe_features_only=True,
        failure_excluded_from_fit=True, preprocessing_fit_on_train_only=True)
    write_json(OUT / 'verification.json', verification)
    pct = lambda value: f'{100*value:.2f}%' if value is not None else 'N/A'
    lines = ['**Step 3B — selected and frozen Router**', '',
        f'Selected `{winner}`, threshold `{selected["threshold"]:.17g}`. LONG iff score >= threshold. No refitting on DEV.', '',
        'Fitted 1,636 binary TRAIN rows (440 SHORT, 1,196 LONG); excluded 284 FAILURE rows. Binary DEV metrics use 413 rows (110 SHORT, 303 LONG). Final-answer selection and analysis use all 480 DEV rows, including 67 FAILURE rows.', '',
        'All three models use balanced inverse-frequency sample weights. Each has one fixed configuration. Preprocessing is fitted only on binary TRAIN. Full configurations and the predeclared lexicographic selection policy are in `selection_protocol.json`.', '',
        table(['Model','Threshold','Adaptive answer','Long route rate','Unnecessary Long','Regressions'],[
            [name, f'{r["selected"]["threshold"]:.6g}', pct(r['selected']['adaptive_answer_accuracy']),
             pct(r['selected']['long_route_rate']), r['selected']['unnecessary_long_calls'], r['selected']['long_regressions']]
            for name,r in results.items()]), '',
        table(['Selected Router metric','Value'], [[key,pct(selected[key])] for key in (
            'routing_accuracy','long_precision','long_recall','short_route_rate','long_route_rate','adaptive_answer_accuracy')]), '',
        table(['Policy, all 480 DEV','Correct','Accuracy'], [
            [name,selected[count],pct(selected[accuracy])] for name,count,accuracy in (
                ('Always Short','always_short_correct','always_short_accuracy'),
                ('Always Long','always_long_correct','always_long_accuracy'),
                ('Adaptive','adaptive_answer_correct','adaptive_answer_accuracy'),
                ('Answer oracle (either parser correct)','oracle_correct','oracle_accuracy'))]), '',
        'Short/Long route rates describe the selected answer path. Short inference actually runs on 100% of inputs to supply Router features; the Long route rate is the additional Long-call rate. Unnecessary Long means a Long call when Short already has a correct final answer. Regression means that call replaces a correct Short answer with a wrong Long answer.', '',
        f'FAILURE DEV analysis: `{json.dumps(selected["failure_analysis"])}`. FAILURE labels use exact-query-plus-answer correctness, so some can still have a correct final answer despite an inexact query. They are never relabeled or used for fitting.', '',
        f'Selected binary confusion matrix: `{json.dumps(selected["binary_confusion"])}`. Long rescues: {selected["long_rescues"]}; Long regressions: {selected["long_regressions"]}.', '',
        '**Limits and reproducibility**', '',
        'These are DEV selection results, not an unbiased held-out estimate. Thresholds were compared on DEV only, and DEV was not used for fitting, preprocessing, or refitting. No CORE EVAL or TEST data were read. Gold outcomes are used only for supervision/scoring; task/register labels only for reporting.', '',
        'Step 3A uses controlled Arabic templates with shared wording families across disjoint TRAIN/DEV anchors. This measures new-anchor generalization rather than unseen-style generalization. The binary Router has no FAILURE/abstention output. Scores are class-weighted classifier outputs, not calibrated probabilities.', '',
        'All frozen Step 3A data, Short, Long, prompts, schema, and Geo Engine hashes verified unchanged. No Short/Long generations or GPU calls were made in Step 3B. Training warnings, timings, default-threshold metrics, selected metrics and breakdowns are saved in `summary.json`.', '',
        'Frozen artifact: `router.joblib` contains preprocessing, model, threshold and feature allowlist. `freeze_manifest.json` records hashes. Load only trusted joblib artifacts with `language.lightweight_router.LightweightRouter`. Supply the existing Step 3A feature dictionary; gold fields are rejected.', '',
        'Stopped after selection and freeze. No CORE evaluation or application integration was launched.', '']
    (OUT / 'REPORT.md').write_text('\n'.join(lines))
    code_paths = ['language/lightweight_router.py','evaluation/train_lightweight_router.py','tests/test_lightweight_router.py']
    manifest = dict(stage='3B', selected_model=winner, threshold=selected['threshold'],
        source_hashes=source_hashes, frozen_component_hashes=frozen,
        code_sha256={path:sha(BASE/path) for path in code_paths},
        artifact_sha256={p.name:sha(p) for p in OUT.iterdir() if p.is_file()}, dependencies=dependencies,
        selection_dataset='Router DEV only', fitted_dataset='Router TRAIN non-FAILURE only')
    write_json(OUT / 'freeze_manifest.json', manifest)
    print(json.dumps(dict(selected_model=winner, selected=selected, verification=verification), indent=2))


if __name__ == '__main__':
    main()
