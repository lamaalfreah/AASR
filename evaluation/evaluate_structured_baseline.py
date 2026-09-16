#!/usr/bin/env python3
"""CPU Step-1 evaluation. VALIDATION quality; optional TRAIN diagnostics. No TEST mode."""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import gzip
import hashlib
import json
from pathlib import Path
import platform
import statistics
import sys
from time import perf_counter

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
from spatial import run
from spatial.context_parser import parse_context_fields
from spatial.normalization import answer_matches


def infer_record(record):
    """Explicit gold firewall: runtime receives exactly two public input strings."""
    return run(record['question'], record['context'])


def percent(n, d):
    return round(100*n/d, 6) if d else None


def summarize(rows):
    n = len(rows)
    successful = [r for r in rows if r['status'] == 'success']
    reconstructable = [r for r in rows if r['audit_class'] == 'A']
    return {
        'total': n,
        'parse_success': sum(r['parsed'] for r in rows),
        'parse_success_percent': percent(sum(r['parsed'] for r in rows), n),
        'identity_status_counts': dict(Counter(r['identity_status'] for r in rows)),
        'execution_status_counts': dict(Counter(r['status'] for r in rows)),
        'executor_success': len(successful),
        'executor_success_percent': percent(len(successful), n),
        'correct': sum(r['correct'] for r in rows),
        'accuracy_percent': percent(sum(r['correct'] for r in rows), n),
        'answered_accuracy_percent': percent(sum(r['correct'] for r in successful), len(successful)),
        'identity_ambiguous': sum(r['identity_status'] == 'AMBIGUOUS' for r in rows),
        'identity_ambiguity_percent': percent(sum(r['identity_status'] == 'AMBIGUOUS' for r in rows), n),
        'ambiguous': sum(r['status'] == 'ambiguous' for r in rows),
        'ambiguity_percent': percent(sum(r['status'] == 'ambiguous' for r in rows), n),
        'abstentions': n-len(successful),
        'abstention_percent': percent(n-len(successful), n),
        'reconstructable_total': len(reconstructable),
        'reconstructable_correct': sum(r['correct'] for r in reconstructable),
        'reconstructable_accuracy_percent': percent(sum(r['correct'] for r in reconstructable), len(reconstructable)),
        'failure_stage_counts': dict(Counter(r['stage'] for r in rows if not r['correct'])),
    }


def csv_write(path, rows, fields):
    with path.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)


def evaluate_split(raw_dir, split, audit_classes):
    # Refuse before opening any file. TEST cannot be requested through CLI or API.
    if split not in ('train', 'validation'):
        raise ValueError('Only train and validation evaluation is permitted; test gold is locked')
    parse_context_fields.cache_clear()
    records, latencies = [], []
    started = perf_counter()
    with gzip.open(raw_dir / (split+'.jsonl.gz'), 'rt', encoding='utf-8') as stream:
        for line in stream:
            record = json.loads(line)
            began = perf_counter()
            result = infer_record(record)
            latencies.append((perf_counter()-began)*1000)
            # Gold and previous audit labels are read only after inference.
            key = (split, record['id'])
            if key not in audit_classes:
                raise ValueError('Record missing from saved audit classification')
            correct = result.status == 'success' and answer_matches(result.answer, record['answer'])
            resolutions = result.trace.get('identity_resolutions', [])
            identity = ('NOT_EVALUATED' if not resolutions else
                        'NOT_FOUND' if any(r['status'] == 'NOT_FOUND' for r in resolutions) else
                        'AMBIGUOUS' if any(r['status'] == 'AMBIGUOUS' for r in resolutions) else 'UNIQUE')
            stage = result.stage if result.status != 'success' else ('none' if correct else 'executor_answer')
            records.append({'split': split, 'id': record['id'], 'task_type': record['task_type'],
                'audit_class': audit_classes[key], 'parsed': result.operation is not None,
                'operation_matches_annotation': result.operation == record['task_type'],
                'identity_status': identity, 'status': result.status, 'correct': correct,
                'stage': stage, 'reason': result.reason or ('answer_mismatch' if not correct else ''),
                'diagnostics': '|'.join(sorted(set(result.trace.get('diagnostics', []))))})
    wall = perf_counter()-started
    metrics = summarize(records)
    ordered = sorted(latencies)
    def percentile(q):
        index = (len(ordered)-1)*q
        lower = int(index)
        return ordered[lower] + (ordered[min(lower+1, len(ordered)-1)]-ordered[lower])*(index-lower)
    metrics.update({
        'split': split, 'purpose': 'quality evaluation' if split == 'validation' else 'diagnostic only',
        'operation_annotation_agreement': sum(r['operation_matches_annotation'] for r in records),
        'runtime': {'total_wall_seconds': wall, 'total_inference_seconds': sum(latencies)/1000,
                    'mean_inference_ms': statistics.mean(latencies), 'p50_inference_ms': percentile(.5),
                    'p95_inference_ms': percentile(.95), 'mean_wall_ms_per_record': wall*1000/len(records),
                    'context_cache': parse_context_fields.cache_info()._asdict(),
                    'measurement': 'One streaming pass in original order; cache cleared per split, LRU 32. Inference includes parsing, identity, geometry and trace construction; wall includes gzip/JSON, scoring, and audit-class lookup. No warmup or repeated benchmark.'}})
    return metrics, records


def source_hashes():
    paths = sorted((BASE/'spatial').glob('*.py')) + [Path(__file__).resolve()]
    return {str(p.relative_to(BASE)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-dir', type=Path, default=BASE/'data/raw')
    parser.add_argument('--output-dir', type=Path, default=BASE/'experiments/E5_lightweight_adaptive/step1_structured_baseline')
    parser.add_argument('--include-train', action='store_true', help='Separate diagnostic; never a validation substitute')
    args = parser.parse_args()
    audit_path = BASE/'experiments/E5_lightweight_adaptive/data_audit/reconstructability_records.csv'
    with audit_path.open(encoding='utf-8', newline='') as f:
        classes = {(r['split'], r['id']): r['class'] for r in csv.DictReader(f)}
    frozen = source_hashes()
    metrics = {'test_gold_locked': True, 'device': 'CPU', 'python': platform.python_version(),
               'platform': platform.platform(), 'runtime_source_sha256': frozen,
               'audit_classes_sha256': hashlib.sha256(audit_path.read_bytes()).hexdigest(),
               'scoring': 'All rows denominator; abstentions count as incorrect. Reconstructable subset is saved audit class A, never an inference input. Entity whitespace only; strict direction labels; safe whole-string integer parsing; Arabic yes/no formatting.',
               'splits': {}}
    all_records, per_task = [], []
    for split in (['validation', 'train'] if args.include_train else ['validation']):
        result, records = evaluate_split(args.raw_dir, split, classes)
        metrics['splits'][split] = result; all_records.extend(records)
        for task in sorted({r['task_type'] for r in records}):
            summary = summarize([r for r in records if r['task_type'] == task])
            per_task.append(dict(split=split, task_type=task, **{k: v for k, v in summary.items() if not isinstance(v, dict)}))
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    if frozen != source_hashes():
        raise RuntimeError('Runtime source changed during evaluation')
    metrics['runtime_unchanged_during_evaluation'] = True
    out = args.output_dir; out.mkdir(parents=True, exist_ok=True)
    (out/'metrics.json').write_text(json.dumps(metrics, ensure_ascii=False, indent=2)+'\n')
    csv_write(out/'per_task_metrics.csv', per_task, list(per_task[0]))
    failures = [r for r in all_records if not r['correct']]
    grouped = Counter((r['split'], r['task_type'], r['stage'], r['status'], r['reason']) for r in failures)
    csv_write(out/'failure_summary.csv', [dict(zip(['split','task_type','stage','status','reason','count'], (*key,n)))
                                        for key, n in sorted(grouped.items())],
              ['split','task_type','stage','status','reason','count'])
    csv_write(out/'failure_records.csv', failures, list(all_records[0]))
    diagnostics = Counter((r['split'], code) for r in all_records for code in r['diagnostics'].split('|') if code)
    metrics['diagnostic_counts'] = {'|'.join(k): n for k,n in sorted(diagnostics.items())}
    (out/'metrics.json').write_text(json.dumps(metrics, ensure_ascii=False, indent=2)+'\n')


if __name__ == '__main__':
    main()
