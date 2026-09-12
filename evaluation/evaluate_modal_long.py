"""Resumable DEV gate and held-out Long comparison; never reruns Neural Short."""
from __future__ import annotations
from collections import Counter
import csv
from dataclasses import asdict
import fcntl
import hashlib
import json
from pathlib import Path
import statistics
from time import perf_counter, time

from language.long_parser import LongParser
from language.modal_checkpoint import MODEL, GENERATION, ModalCheckpointProvider, atomic_json, digest
from spatial import parse_context, execute
from spatial.normalization import answer_matches

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE / 'experiments/E5_lightweight_adaptive'
OUT = ROOT / 'step2_long'
CHALLENGE = ROOT / 'step2_short_long/challenge'
SHORT = ROOT / 'step2_neural_short'
SMOKE_TASKS = ('nearest_category', 'cardinal_direction', 'two_hop_nearest')
SESSION_LIMIT_SECONDS = 45 * 60


def read_jsonl(path):
    with path.open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def frozen_check():
    manifest = json.loads((OUT / 'freeze_manifest.json').read_text())
    checks = {BASE / p: h for p, h in manifest['frozen_step1_files'].items()}
    checks.update({CHALLENGE / p: h for p, h in manifest['challenge_files'].items()})
    for path, key in [('language/neural_short.py', 'neural_source_sha256'),
                      ('language/long_parser.py', 'long_prompt_source_sha256')]:
        checks[BASE / path] = manifest[key]
    for filename, key in [('intent_bigru.pt', 'neural_checkpoint_sha256'),
                          ('per_example.csv', 'neural_predictions_sha256'),
                          ('metrics.json', 'neural_metrics_sha256')]:
        checks[SHORT / filename] = manifest[key]
    for path, expected in checks.items():
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError('Frozen artifact mismatch: ' + str(path.relative_to(BASE)))
    return len(checks)


def saved_rows(split):
    return [json.loads(p.read_text()) for p in sorted((OUT / 'modal_predictions' / split).glob('*.json'))]


def progress(status, reason=None):
    completed = len(saved_rows('eval'))
    value = dict(provider='Modal', model=MODEL, status=status, reason=reason,
                 smoke_completed=len(saved_rows('dev')), eval_completed=completed,
                 eval_total=128, eval_remaining=128-completed,
                 durable_generation_checkpoints=len(list((OUT/'modal_calls').glob('*.json'))),
                 updated_unix=time())
    atomic_json(OUT/'api_progress.json', value)
    return value


def percent(n, d):
    return 100*n/d if d else None


def latency(values):
    if not values:
        return dict(mean_ms=None, p50_ms=None, p95_ms=None)
    a = sorted(values)
    def percentile(q):
        position = (len(a)-1)*q
        i = int(position)
        return a[i] + (a[min(i+1, len(a)-1)]-a[i])*(position-i)
    return dict(mean_ms=statistics.mean(a), p50_ms=percentile(.5), p95_ms=percentile(.95))


def paired(rows, level):
    result = {key: 0 for key in ('both_correct', 'short_only', 'long_only', 'both_wrong')}
    for row in rows:
        s, l = row['short_'+level], row['long_'+level]
        key = 'both_correct' if s and l else 'short_only' if s else 'long_only' if l else 'both_wrong'
        result[key] += 1
    return dict(result, rescue_percent=percent(result['long_only'], result['long_only']+result['both_wrong']),
                regression_percent=percent(result['short_only'], result['short_only']+result['both_correct']))


def summarize(rows):
    value = dict(n=len(rows))
    for key in ('short_query_exact', 'short_answer_correct', 'long_intent_correct',
                'long_query_exact', 'long_answer_correct'):
        value[key+'_count'] = sum(r[key] for r in rows)
        value[key+'_percent'] = percent(value[key+'_count'], len(rows))
    for level in ('query_exact', 'answer_correct'):
        value[level+'_paired'] = paired(rows, level)
    return value


def write_csv(path, rows):
    if not rows:
        path.write_text('evaluated\n0\n')
        return
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def report(status, reason=None):
    rows = saved_rows('eval')
    calls = [json.loads(p.read_text()) for p in sorted((OUT/'modal_calls').glob('*.json'))]
    sessions = [json.loads(p.read_text()) for p in sorted((OUT/'modal_sessions').glob('*.json'))]
    metrics = summarize(rows)
    metrics.update(provider='Modal', model=MODEL, status=status, reason=reason,
                   eval_expected=128, completed_examples=len(rows), generation_calls=len(calls),
                   eval_generation_calls=sum(r['llm_calls'] for r in rows),
                   input_tokens=sum(c['input_tokens'] for c in calls),
                   output_tokens=sum(c['output_tokens'] for c in calls),
                   eval_input_tokens=sum(r['input_tokens'] for r in rows),
                   eval_output_tokens=sum(r['output_tokens'] for r in rows),
                   inference_latency=latency([r['long_inference_ms'] for r in rows]),
                   remote_roundtrip_latency=latency([r['long_roundtrip_ms'] for r in rows]),
                   generation_call_latency=latency([c['inference_ms'] for c in calls]),
                   sessions=sessions, actual_provider_cost=None,
                   cost_note='Monetary cost and billable GPU duration are not supplied by these inference responses; no estimate.',
                   gpu_types=sorted({c['gpu'] for c in calls}),
                   total_session_wall_seconds=sum(s.get('wall_seconds', 0) for s in sessions),
                   invalid_responses=sum(r['invalid_responses'] for r in rows),
                   status_counts=dict(Counter(r['long_status'] for r in rows)),
                   verdict='ROUTER NOT YET JUSTIFIED')
    if len(rows) == 128 and metrics['query_exact_paired']['long_only'] > metrics['query_exact_paired']['short_only']:
        metrics['verdict'] = 'ROUTER PROMISING BUT NEEDS MORE DATA'
    short_latency = statistics.mean(float(r['short_pipeline_ms']) for r in rows) if rows else None
    metrics['frozen_short_mean_ms'] = short_latency
    metrics['long_inference_to_short_pipeline_ratio'] = metrics['inference_latency']['mean_ms']/short_latency if rows else None
    metrics['answer_matches_with_wrong_query'] = sum(r['long_answer_correct'] and not r['long_query_exact'] for r in rows)
    metrics['frozen_files_verified'] = frozen_check()
    atomic_json(OUT/'long_metrics.json', metrics)
    write_csv(OUT/'short_vs_long.csv', rows)
    breakdowns = {}
    for key, filename in [('task_type', 'per_task'), ('language_register', 'per_register'),
                          ('complexity_level', 'per_complexity')]:
        groups = []
        for group in sorted({r[key] for r in rows}):
            summary = summarize([r for r in rows if r[key] == group])
            flat = {'group': group}
            for k, v in summary.items():
                if isinstance(v, dict):
                    flat.update({k+'_'+inner: count for inner, count in v.items()})
                else:
                    flat[k] = v
            groups.append(flat)
        write_csv(OUT/(filename+'.csv'), groups)
        breakdowns[filename] = groups
    def fmt(value):
        return 'unmeasured' if value is None else f'{value:.2f}'
    lines = ['# Step 2B — Modal Qwen3-4B versus frozen Neural Short', '',
             f'Status: **{status}**. {reason or ""}', '',
             f'**{len(rows)}/128 CORE EVAL examples completed.** {len(calls)} durable generation calls across DEV/EVAL, including schema retries.', '',
             'The official `Qwen/Qwen3-4B` weights are used without adapters, quantization or training. '
             'One L4 container is reused; bfloat16 and non-thinking greedy generation are fixed before EVAL. '
             'The existing prompt, schema, Geo Engine, Rule Baseline 0, Neural Short checkpoint/predictions and 320 questions remain frozen.', '',
             'Only question, exact visible candidate names, anchor name and category vocabulary reach the model. '
             'Gold labels and challenge metadata are used locally after inference for scoring. Spatial arithmetic stays in the frozen executor.', '',
             '## DEV gate', '',
             'Three DEV items selected before EVAL: first nearest-category, cardinal-direction and two-hop item in preserved file order. '
             'All must produce a validated query, exact reference query and matching answer through the frozen executor. '
             'At most one existing generic schema retry; no prompt tuning or EVAL-driven repair.', '']
    for row in saved_rows('dev'):
        lines.append(f'- {row["challenge_id"]}: status={row["long_status"]}, query exact={row["long_query_exact"]}, answer correct={row["long_answer_correct"]}, calls={row["llm_calls"]}.')
    lines += ['', '## Held-out results', '',
              '| System | Intent | Query exact match | End-to-end answer |',
              '|---|---:|---:|---:|',
              '| Frozen Neural Short (all 128) | 23.44% | 20.31% (26/128) | 23.44% (30/128) |',
              f'| Long (completed subset, N={len(rows)}) | {fmt(metrics["long_intent_correct_percent"])} | {fmt(metrics["long_query_exact_percent"])} | {fmt(metrics["long_answer_correct_percent"])} |', '',
              'Long percentages and paired counts below refer only to completed EVAL examples. Zero completed examples means unmeasured accuracy.', '',
              '| Level | Both correct | Short only | Long only | Both wrong | Rescue | Regression |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for level in ('query_exact', 'answer_correct'):
        p = metrics[level+'_paired']
        lines.append(f'| {level} | {p["both_correct"]} | {p["short_only"]} | {p["long_only"]} | {p["both_wrong"]} | {fmt(p["rescue_percent"])} | {fmt(p["regression_percent"])} |')
    lines += ['', 'Rescue = Long-only / all Short-wrong; regression = Short-only / all Short-correct. '
              'Abstentions count as incorrect. Query EM compares the complete canonical StructuredQuery, preserving exact entity text and order. '
              'Answer matching alone can hide wrong queries.', '',
              f'Long answers matching despite a wrong query: {metrics["answer_matches_with_wrong_query"]}.', '']
    for dimension, groups in breakdowns.items():
        lines += [f'## {dimension}', '', '| Group | N | Short query % | Long query % | Short answer % | Long answer % | Query rescue % | Answer rescue % |',
                  '|---|---:|---:|---:|---:|---:|---:|---:|']
        for g in groups:
            lines.append(f'| {g["group"]} | {g["n"]} | {fmt(g["short_query_exact_percent"])} | {fmt(g["long_query_exact_percent"])} | {fmt(g["short_answer_correct_percent"])} | {fmt(g["long_answer_correct_percent"])} | {fmt(g["query_exact_paired_rescue_percent"])} | {fmt(g["answer_correct_paired_rescue_percent"])} |')
        lines.append('')
    lines += ['## Runtime and preservation', '',
              f'Observed GPUs: {metrics["gpu_types"] or "not allocated"}. Model loading times and actual runtime versions are recorded per session in `long_metrics.json`. '
              f'Total measured session wall time: {fmt(metrics["total_session_wall_seconds"])} s; image build and local implementation are outside this timer.', '',
              f'EVAL inference latency including retries: mean {fmt(metrics["inference_latency"]["mean_ms"])} ms, '
              f'P50 {fmt(metrics["inference_latency"]["p50_ms"])} ms, P95 {fmt(metrics["inference_latency"]["p95_ms"])} ms. '
              'This uses saved remote measurements, so resumed cache replay does not create artificially fast inference times. '
              'Round-trip timings are also saved; replay after a lost response can understate network time.', '',
              f'All DEV/EVAL calls: {metrics["input_tokens"]} input and {metrics["output_tokens"]} output tokens. '
              f'EVAL alone: {metrics["eval_input_tokens"]} input and {metrics["eval_output_tokens"]} output tokens. '
              f'Long inference / frozen Short pipeline mean ratio: {fmt(metrics["long_inference_to_short_pipeline_ratio"])}. '
              'Timing scopes differ: Short is a saved local CPU pipeline measurement, Long is remote GPU generation including tokenization but excluding network and local Geo execution. '
              'This is not a controlled hardware speed comparison. Monetary cost is unavailable; no estimated dollar cost is reported.', '',
              'CPU-only weight preparation reuses `asar-hf-cache`. Raw generations are atomically persisted and committed in '
              '`aasr-step2b-long-checkpoints` before returning; local `modal_calls` and `modal_predictions` preserve progress. '
              'Batch size one avoids changing outputs through batching and permits immediate per-call checkpoints. '
              'No remote automatic retry, alternate GPU, model substitution or concurrent GPU containers. A 45-minute session guard stops further requests. '
              'A process-local file lock prevents duplicate launches from this workspace.', '',
              'Resume: `modal run -m language.modal_long_backend` from the repository root. Completed predictions and generation calls are reused. '
              'A failed recorded DEV gate stops again without GPU allocation; changing the experiment requires explicit review.', '',
              '## Router opportunity and next step', '', f'**{metrics["verdict"]}**', '',
              ('Long provides measured net exact-query rescue on this fixed challenge, but an oracle paired result does not establish a deployable router. '
               'The 128 synthetic EVAL questions are now evaluation evidence and must not be used to select thresholds. '
               'Use new TRAIN-derived/DEV paired outcomes and an independently reviewed held-out set for a future router study.'
               if metrics['verdict'] == 'ROUTER PROMISING BUT NEEDS MORE DATA' else
               'The current results do not establish sufficient measured complementary Long benefit. Resolve the recorded blocker or assess completed errors before any Router work.'), '',
              'Potential inference signals remain Short confidence, query validation, missing slots, identity ambiguity and question length. '
              'Saved Short confidence/status are available in the comparison CSV. Entropy and top-two margin were not saved, so no values or thresholds are invented. '
              'Gold task/difficulty, language register and assigned complexity are analysis strata only. No Router has been implemented.', '',
              'Evidence: `modal_experiment.json`, `modal_predictions/`, `modal_calls/`, `modal_sessions/`, '
              '`long_metrics.json`, `short_vs_long.csv`, and the three breakdown CSVs. '
              'Historical OpenRouter preflight files are retained as provenance; this run never uses OpenRouter or its credential.', '']
    (OUT/'README.md').write_text('\n'.join(lines), encoding='utf-8')
    return metrics


def infer(item, context, provider, parser, short, config_hash, split):
    path = OUT/'modal_predictions'/split/(item['challenge_id']+'.json')
    if path.exists():
        row = json.loads(path.read_text())
        if row['config_hash'] != config_hash:
            raise RuntimeError('Prediction configuration mismatch')
        return row
    provider.begin_case(item['challenge_id'])
    outcome = parser.parse(item['challenge_question'], context)
    query = outcome.query
    execution = execute(query, context) if query is not None else None
    predicted = json.loads(json.dumps(asdict(query))) if query else None
    short_row = short[item['challenge_id']]
    row = {k: item[k] for k in ('challenge_id', 'task_type', 'language_register', 'complexity_level')}
    row.update(config_hash=config_hash, split=split,
               long_intent_correct=bool(query and query.operation == item['task_type']),
               long_query_exact=predicted == item['expected_structured_query'],
               long_answer_correct=bool(execution and execution.status == 'success' and answer_matches(execution.answer, item['gold_answer'])),
               long_status=execution.status if execution else outcome.status,
               long_reason=execution.reason if execution else outcome.message,
               short_query_exact=short_row['query_exact'] == 'True',
               short_answer_correct=short_row['answer_correct'] == 'True',
               short_status=short_row['status'], short_confidence=float(short_row['confidence']),
               short_pipeline_ms=float(short_row['pipeline_ms']),
               llm_calls=outcome.llm_calls, invalid_responses=outcome.invalid_responses,
               input_tokens=outcome.input_tokens, output_tokens=outcome.output_tokens,
               long_inference_ms=sum(c['inference_ms'] for c in provider.case_calls),
               long_roundtrip_ms=sum(c['request_roundtrip_ms'] for c in provider.case_calls),
               predicted_query=predicted,
               predicted_answer=asdict(execution.answer) if execution and execution.answer else None)
    atomic_json(path, row)
    return row


def run(prepare_weights, model_class, app_id):
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT/'.modal_run.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return run_locked(prepare_weights, model_class, app_id)


def run_locked(prepare_weights, model_class, app_id):
    frozen_check()
    if len(saved_rows('eval')) == 128:
        progress('complete'); report('complete'); return
    prior_smoke = saved_rows('dev')
    if any(not r['long_query_exact'] or not r['long_answer_correct'] for r in prior_smoke):
        progress('smoke_failed', 'Preserved DEV smoke failure; no further inference')
        report('smoke_failed', 'Preserved DEV smoke failure; no further inference'); return
    started = perf_counter()
    session = dict(app_id=app_id, started_unix=time(), authentication_verified=True)
    session_path = OUT/'modal_sessions'/(app_id+'.json')
    status, reason = 'running', None
    try:
        path = OUT/'modal_experiment.json'
        config = json.loads(path.read_text()) if path.exists() else None
        code = {p: hashlib.sha256((BASE/p).read_bytes()).hexdigest() for p in
                ('language/long_parser.py', 'language/modal_long_backend.py', 'language/modal_checkpoint.py')}
        if config and (config['code'] != code or config['generation'] != GENERATION):
            raise RuntimeError('Frozen Modal configuration changed')
        prep = prepare_weights.remote(config['revision'] if config else '')
        session['preparation'] = prep
        if config is None:
            config = dict(model=MODEL, revision=prep['revision'], generation=GENERATION, code=code,
                          smoke_tasks=SMOKE_TASKS, frozen_manifest_sha256=hashlib.sha256((OUT/'freeze_manifest.json').read_bytes()).hexdigest())
            atomic_json(path, config)
        config_hash = digest(config)
        remote = model_class(revision=config['revision'])
        session['runtime'] = remote.metadata.remote()
        atomic_json(session_path, session)
        print('L4 loaded:', json.dumps(session['runtime']), flush=True)
        def checkpointed():
            progress('running')
        provider = ModalCheckpointProvider(remote.generate.remote, OUT/'modal_calls', config, checkpointed)
        parser = LongParser(provider)
        contexts = {r['source_record_id']: parse_context(r['context']) for r in read_jsonl(CHALLENGE/'source_contexts.jsonl')}
        with (SHORT/'per_example.csv').open() as stream:
            short = {r['id']: r for r in csv.DictReader(stream) if r['scope'] in ('core_dev', 'core_eval')}
        dev = read_jsonl(CHALLENGE/'core_dev.jsonl')
        smoke = [next(r for r in dev if r['task_type'] == task) for task in SMOKE_TASKS]
        for item in smoke:
            row = infer(item, contexts[item['source_record_id']], provider, parser, short, config_hash, 'dev')
            progress('smoke_running')
            print(f'DEV {item["challenge_id"]}: {row["long_status"]}; exact={row["long_query_exact"]}; answer={row["long_answer_correct"]}', flush=True)
            if not row['long_query_exact'] or not row['long_answer_correct']:
                status, reason = 'smoke_failed', 'DEV did not pass exact-query and frozen-engine answer checks; EVAL remains untouched'
                return
        print('DEV smoke passed. Beginning frozen CORE EVAL (128).', flush=True)
        # EVAL labels are first loaded only after the predeclared DEV gate succeeds.
        for item in read_jsonl(CHALLENGE/'core_eval.jsonl'):
            if perf_counter()-started > SESSION_LIMIT_SECONDS:
                status, reason = 'resource_stop', '45-minute session guard reached; resume retained progress'
                return
            row = infer(item, contexts[item['source_record_id']], provider, parser, short, config_hash, 'eval')
            progress('eval_running')
            # Detect unexpected container replacement after preserving its response.
            if any(c['container_session'] != session['runtime']['container_session'] for c in provider.case_calls
                   if c.get('container_session') and c.get('case_id') == item['challenge_id'] and
                   c.get('container_session') not in {s.get('runtime', {}).get('container_session') for s in
                       [json.loads(p.read_text()) for p in (OUT/'modal_sessions').glob('*.json')]}):
                raise RuntimeError('Unexpected extra GPU cold start')
            print(f'CORE EVAL checkpointed: {len(saved_rows("eval"))}/128', flush=True)
        status = 'complete'
    except (Exception, KeyboardInterrupt) as exc:
        status, reason = 'resource_stop', type(exc).__name__ + ': ' + str(exc)[:350]
        print('Stopped cleanly:', reason, flush=True)
    finally:
        session['wall_seconds'] = perf_counter()-started
        session['status'], session['reason'] = status, reason
        atomic_json(session_path, session)
        progress(status, reason)
        metrics = report(status, reason)
        print(json.dumps({'status': status, 'completed': metrics['completed_examples'],
                          'verdict': metrics['verdict']}, ensure_ascii=False), flush=True)
