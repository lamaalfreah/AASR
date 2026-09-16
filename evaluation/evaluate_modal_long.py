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

from language.long_parser import LongParser, parse_json
from language.modal_checkpoint import MODEL, GENERATION, ModalCheckpointProvider, atomic_json, digest
from spatial import parse_context, execute
from spatial.normalization import answer_matches
from spatial.schema import OPERATIONS

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE / 'experiments/E5_lightweight_adaptive'
OUT = ROOT / 'step2_long'
CHALLENGE = ROOT / 'step2_short_long/challenge'
SHORT = ROOT / 'step2_neural_short'
SMOKE_TASKS = OPERATIONS
SMOKE_PER_TASK = 2
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


def verify_eval_freeze():
    """The user accepted the recorded DEV result before any held-out inference."""
    acceptance = json.loads((OUT/'core_eval_freeze.json').read_text())
    if acceptance['authorization'] != 'User accepted DEV gate and authorized 128 CORE EVAL examples':
        raise RuntimeError('CORE EVAL authorization record mismatch')
    for relative, expected in acceptance['sha256'].items():
        if hashlib.sha256((BASE/relative).read_bytes()).hexdigest() != expected:
            raise RuntimeError('CORE EVAL frozen file changed: ' + relative)
    return acceptance


def audit_responses(calls, context):
    """Read returned text only; never repair it or feed scoring labels to inference."""
    names = {p.name for p in context.candidates}
    if context.anchor:
        names.add(context.anchor.name)
    invented = []
    failures = []
    raw_operation = None
    for call in calls:
        try:
            data = json.loads(call['text'])
            query = data.get('query') if isinstance(data, dict) else None
            raw_operation = query.get('operation') if isinstance(query, dict) else None
            if isinstance(query, dict):
                targets = query.get('targets')
                refs = [query.get('origin')] + (targets if isinstance(targets, list) else [])
                for ref in refs:
                    if isinstance(ref, dict) and isinstance(ref.get('name'), str) and ref['name'] not in names:
                        invented.append(ref['name'])
        except (ValueError, TypeError):
            raw_operation = None
        try:
            parse_json(call['text'], context)
        except (ValueError, TypeError, KeyError) as exc:
            failures.append(str(exc))
    return dict(raw_operation=raw_operation, invented_entity_names=invented,
                invented_entity_count=len(invented), validation_errors=failures)


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
    for name, count in (
        ('schema_valid', sum(r['predicted_query'] is not None for r in rows)),
        ('retried_examples', sum(r['llm_calls'] > 1 for r in rows)),
        ('invalid_output', sum(r['long_status'] == 'invalid_output' for r in rows)),
        ('abstention_or_failure', sum(r['long_status'] != 'success' for r in rows))):
        value[name+'_count'] = count
        value[name+'_percent'] = percent(count, len(rows))
    value['retry_calls'] = sum(max(0, r['llm_calls']-1) for r in rows)
    value['invented_entity_count'] = sum(r.get('invented_entity_count', 0) for r in rows)
    value['invented_entity_examples'] = sum(bool(r.get('invented_entity_count', 0)) for r in rows)
    value['raw_operation_correct_count'] = sum(r.get('raw_operation_correct', False) for r in rows)
    value['raw_operation_correct_percent'] = percent(value['raw_operation_correct_count'], len(rows))
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
    eval_sessions = [s for s in sessions if s.get('mode') == 'eval']
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
                   eval_session_wall_seconds=sum(s.get('wall_seconds', 0) for s in eval_sessions),
                   eval_sessions=eval_sessions,
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
             'Sixteen DEV items selected before EVAL: first two items for each supported operation in preserved file order. '
             'The user accepted the frozen result: 16 valid queries, 15 exact queries, 16 matching answers; '
             'one category-comparison operation confusion, four initial responses requiring retry, and verified checkpoint replay. '
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
    lines += ['## Validation and failures', '',
              f'Full strict validation: {metrics["schema_valid_count"]}/{len(rows)} ({fmt(metrics["schema_valid_percent"])}%). '
              f'Retried examples: {metrics["retried_examples_count"]} ({fmt(metrics["retried_examples_percent"])}%); retry calls: {metrics["retry_calls"]}. '
              f'Invalid responses across attempts: {metrics["invalid_responses"]}; final invalid-output examples: {metrics["invalid_output_count"]}. '
              f'Abstention/failure examples (non-success status): {metrics["abstention_or_failure_count"]}.', '',
              f'Invented entity references across all EVAL attempts: {metrics["invented_entity_count"]}, affecting {metrics["invented_entity_examples"]} examples. '
              'Names are checked against the supplied anchor/candidate names; repeated fabricated references are counted separately. '
              f'Final raw operation-label accuracy before validation: {metrics["raw_operation_correct_count"]}/{len(rows)} '
              f'({fmt(metrics["raw_operation_correct_percent"])}%). Main intent accuracy above requires an accepted query; rejected outputs count as wrong.', '',
              f'Status counts: `{json.dumps(metrics["status_counts"], sort_keys=True)}`.', '']
    for dimension, groups in breakdowns.items():
        lines += [f'## {dimension}', '', '| Group | N | Short query % | Long query % | Short answer % | Long answer % | Query rescue % | Answer rescue % |',
                  '|---|---:|---:|---:|---:|---:|---:|---:|']
        for g in groups:
            lines.append(f'| {g["group"]} | {g["n"]} | {fmt(g["short_query_exact_percent"])} | {fmt(g["long_query_exact_percent"])} | {fmt(g["short_answer_correct_percent"])} | {fmt(g["long_answer_correct_percent"])} | {fmt(g["query_exact_paired_rescue_percent"])} | {fmt(g["answer_correct_paired_rescue_percent"])} |')
        lines.append('')
    lines += ['## Runtime and preservation', '',
              f'Observed GPUs: {metrics["gpu_types"] or "not allocated"}. Model loading times and actual runtime versions are recorded per session in `long_metrics.json`. '
              f'Total measured session wall time: {fmt(metrics["total_session_wall_seconds"])} s; image build and local implementation are outside this timer.', '',
              f'CORE EVAL session wall time only: {fmt(metrics["eval_session_wall_seconds"])} s. '
              f'EVAL generation calls: {metrics["eval_generation_calls"]}, including {metrics["retry_calls"]} retry calls. '
              f'Frozen Short mean CPU pipeline latency: {fmt(metrics["frozen_short_mean_ms"])} ms.', '',
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
              'Resume: `modal run --profile joudalrubaish -m language.modal_long_backend --no-smoke-only` from the repository root. '
              'Completed predictions and generation calls are reused. The accepted DEV records and inference implementation are pinned by `core_eval_freeze.json`.', '',
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
    audit = audit_responses(provider.case_calls, context)
    row.update(audit, raw_operation_correct=audit['raw_operation'] == item['task_type'])
    atomic_json(path, row)
    return row


def run(prepare_weights, model_class, app_id, smoke_only=True):
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT/'.modal_run.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return run_locked(prepare_weights, model_class, app_id, smoke_only=smoke_only)


def run_locked(prepare_weights, model_class, app_id, smoke_only=True):
    frozen_check()
    if not smoke_only:
        verify_eval_freeze()
    if len(saved_rows('eval')) == 128:
        progress('complete'); report('complete'); return
    prior_smoke = saved_rows('dev')
    if smoke_only and any(not r['long_query_exact'] or not r['long_answer_correct'] for r in prior_smoke):
        progress('smoke_failed', 'Preserved DEV smoke failure; no further inference')
        report('smoke_failed', 'Preserved DEV smoke failure; no further inference'); return
    started = perf_counter()
    session = dict(app_id=app_id, started_unix=time(), authentication_verified=True,
                   mode='dev' if smoke_only else 'eval')
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
                          smoke_tasks=SMOKE_TASKS, smoke_per_task=SMOKE_PER_TASK,
                          frozen_manifest_sha256=hashlib.sha256((OUT/'freeze_manifest.json').read_bytes()).hexdigest())
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
            scopes = ('core_dev',) if smoke_only else ('core_dev', 'core_eval')
            short = {r['id']: r for r in csv.DictReader(stream) if r['scope'] in scopes}
        dev = read_jsonl(CHALLENGE/'core_dev.jsonl')
        smoke = [r for task in SMOKE_TASKS
                 for r in [item for item in dev if item['task_type'] == task][:SMOKE_PER_TASK]]
        if len(smoke) != len(SMOKE_TASKS) * SMOKE_PER_TASK:
            raise RuntimeError('Insufficient DEV examples for balanced smoke sample')
        for item in smoke if smoke_only else []:
            row = infer(item, contexts[item['source_record_id']], provider, parser, short, config_hash, 'dev')
            progress('smoke_running')
            print(f'DEV {item["challenge_id"]}: {row["long_status"]}; exact={row["long_query_exact"]}; answer={row["long_answer_correct"]}', flush=True)
            if not row['long_query_exact'] or not row['long_answer_correct']:
                status, reason = 'smoke_failed', 'DEV did not pass exact-query and frozen-engine answer checks; EVAL remains untouched'
                print(json.dumps(row, ensure_ascii=False), flush=True)
                if not smoke_only:
                    return
        if smoke_only:
            def no_inference(*args):
                raise RuntimeError('Resume unexpectedly requested inference')
            replay_provider = ModalCheckpointProvider(no_inference, OUT/'modal_calls', config)
            replay_parser = LongParser(replay_provider)
            for item in smoke:
                context = contexts[item['source_record_id']]
                saved = json.loads((OUT/'modal_predictions'/'dev'/(item['challenge_id']+'.json')).read_text())
                replay_provider.begin_case(item['challenge_id'])
                replay = replay_parser.parse(item['challenge_question'], context)
                replay_query = json.loads(json.dumps(asdict(replay.query))) if replay.query else None
                if replay_query != saved['predicted_query']:
                    raise RuntimeError('Generation checkpoint replay mismatch')
                replay_execution = execute(replay.query, context) if replay.query else None
                replay_status = replay_execution.status if replay_execution else replay.status
                replay_answer = asdict(replay_execution.answer) if replay_execution and replay_execution.answer else None
                if replay_status != saved['long_status'] or replay_answer != saved['predicted_answer']:
                    raise RuntimeError('Frozen executor replay mismatch')
                for call in replay_provider.case_calls:
                    durable = remote.read_checkpoint.remote(call['request_digest'])
                    expected = {k: v for k, v in call.items() if k not in ('case_id', 'request_roundtrip_ms')}
                    if durable != expected:
                        raise RuntimeError('Remote durable checkpoint mismatch')
                resumed = infer(item, context, replay_provider, replay_parser, short, config_hash, 'dev')
                if resumed != saved:
                    raise RuntimeError('Prediction checkpoint replay mismatch')
            session['resume_verification'] = dict(examples=len(smoke), local_generation_replay=True,
                prediction_replay=True, remote_committed_checkpoints=True, extra_generations=0)
            if status != 'smoke_failed':
                status = 'smoke_passed'
            print(f'DEV result: {status}; checkpoint replay passed. Stopping before CORE EVAL.', flush=True)
            return
        print('User-accepted frozen DEV gate verified. Beginning CORE EVAL (128).', flush=True)
        eval_items = read_jsonl(CHALLENGE/'core_eval.jsonl')
        eval_ids = {r['challenge_id'] for r in eval_items}
        if len(eval_items) != 128 or len(eval_ids) != 128 or eval_ids & {r['challenge_id'] for r in dev}:
            raise RuntimeError('CORE EVAL count, uniqueness or DEV disjointness failed')
        for item in eval_items:
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
        verify_eval_freeze()
    except (Exception, KeyboardInterrupt) as exc:
        status, reason = 'resource_stop', type(exc).__name__ + ': ' + str(exc)
        print('Stopped cleanly:', reason, flush=True)
    finally:
        session['wall_seconds'] = perf_counter()-started
        session['status'], session['reason'] = status, reason
        atomic_json(session_path, session)
        progress(status, reason)
        metrics = report(status, reason)
        print(json.dumps({'status': status, 'completed': metrics['completed_examples'],
                          'verdict': metrics['verdict']}, ensure_ascii=False), flush=True)
