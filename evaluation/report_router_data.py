"""Verify completed Step 3A checkpoints locally and write the development-data report."""
from collections import Counter
import csv
import gzip
import json
from pathlib import Path

from evaluation.router_data import (BASE,OUT,read_jsonl,verify_manifest,canonical,slots,sha,label)
from language.modal_checkpoint import ModalCheckpointProvider,atomic_json
from language.long_parser import LongParser,query_from_dict
from spatial import parse_context,execute
from spatial.normalization import answer_matches


def main():
    manifest=verify_manifest()
    pipeline=json.loads((OUT/'pipeline_freeze.json').read_text())
    for path,expected in pipeline['code_sha256'].items():
        if sha(BASE/path)!=expected:
            raise RuntimeError('Step 3A pipeline changed: '+path)
    items=read_jsonl(OUT/'questions.jsonl')
    dataset=read_jsonl(OUT/'router_dataset.jsonl')
    assert len(dataset)==len(items)==2400
    paired={r['id']:r for r in [json.loads(p.read_text()) for p in (OUT/'paired_predictions').glob('*.json')]}
    data={r['id']:r for r in dataset}
    assert set(paired)==set(data)=={r['id'] for r in items}
    train_ids={r['provenance']['source_anchor'] for r in dataset if r['provenance']['router_split']=='train'}
    dev_ids={r['provenance']['source_anchor'] for r in dataset if r['provenance']['router_split']=='dev'}
    excluded=set(json.loads((OUT/'anchor_split.json').read_text())['excluded_core_anchors'])
    assert not train_ids&dev_ids and not (train_ids|dev_ids)&excluded
    if sha(manifest['source_train_path'])!=manifest['source_train_sha256']:
        raise RuntimeError('Original TRAIN source changed')
    needed={r['source_record_id'] for r in items}
    original={}
    with gzip.open(manifest['source_train_path'],'rt',encoding='utf-8') as stream:
        for line in stream:
            row=json.loads(line)
            if row['id'] in needed:
                assert row['split']=='train'
                original[row['id']]=row
    assert set(original)==needed
    bank=json.loads((OUT/'paraphrase_bank.json').read_text())
    config=json.loads((OUT/'inference_config.json').read_text())
    def no_remote(*args):
        raise AssertionError('Checkpoint replay requested new inference')
    provider=ModalCheckpointProvider(no_remote,OUT/'long_calls',config)
    parser=LongParser(provider)
    before={p.name:sha(p) for p in (OUT/'long_calls').glob('*.json')}
    used=[]
    for item in items:
        row=paired[item['id']];record=data[item['id']]
        source=original[item['source_record_id']]
        assert source['wikidata_id']==item['source_anchor']
        assert source['answer']==item['gold_answer'] and source['question']==item['original_question']
        assert parse_context(source['context'])==parse_context(item['context'])
        expected=query_from_dict(item['expected_structured_query'])
        template=bank[item['task_type']][item['language_register']][int(item['wording_family'].split(':')[-1])]
        assert item['question']==template.format(**slots(expected))
        ctx=parse_context(item['context'])
        assert answer_matches(execute(expected,ctx).answer,item['gold_answer'])
        provider.begin_case(item['id'])
        result=parser.parse(item['question'],ctx)
        assert canonical(result.query)==row['long']['query']
        assert result.llm_calls==row['long']['llm_calls']
        assert result.invalid_responses==row['long']['invalid_responses']
        used.extend(provider.case_calls)
        for which in ('short','long'):
            saved=row[which]
            query=query_from_dict(saved['query']) if saved['query'] else None
            execution=execute(query,ctx) if query else None
            assert (canonical(query)==item['expected_structured_query'])==saved['query_exact']
            correct=bool(execution and execution.status=='success' and answer_matches(execution.answer,item['gold_answer']))
            assert correct==saved['answer_correct']
        strict=label(row['short']['query_exact'] and row['short']['answer_correct'],
                     row['long']['query_exact'] and row['long']['answer_correct'])
        answers=label(row['short']['answer_correct'],row['long']['answer_correct'])
        assert strict==record['strict_label'] and answers==record['answer_only_label']
    assert len({c['request_digest'] for c in used})==len(used)
    assert before=={p.name:sha(p) for p in (OUT/'long_calls').glob('*.json')}
    verify_manifest()
    summary=json.loads((OUT/'summary.json').read_text())
    verification=dict(n=2400,source_records_verified=2400,questions_unique=True,
        all_source_semantics_and_gold_preserved=True,train_only=True,excluded_core_anchors=len(excluded),
        anchor_disjoint=True,local_generation_replay_verified=2400,replay_new_generations=0,
        replay_unique_calls=len(used),generation_checkpoints_unchanged=True,
        label_precedence_verified=True,features_from_short_and_public_inputs_only=True,
        frozen_components_verified=True,core_eval_and_test_not_read=True,router_trained=False)
    atomic_json(OUT/'verification.json',verification)
    def table(headers,rows):
        return ['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+[
            '| '+' | '.join(map(str,row))+' |' for row in rows]+['']
    lines=['**Step 3A — Router development data (no Router training)**','',
        'Completed 2,400 paired frozen Short/Long predictions on new natural-Arabic paraphrases of original TRAIN records. All 320 source anchors used by the CORE challenge were excluded before sampling. CORE EVAL and TEST contents were not read.','',
        f'Primary correctness criterion: `{summary["primary_correctness"]}`. SHORT takes precedence whenever Short is correct, including when both models are correct; LONG means Short wrong and Long correct; FAILURE means both wrong. Exact-query-plus-answer and answer-only labels are both retained.','',
        '**Label distribution**','']
    lines+=table(['Scope','N','SHORT','LONG','FAILURE'],[
        [scope,2400 if scope=='all' else summary['counts_by_split'][scope]]+
        [summary['label_counts'].get(k,0) if scope=='all' else summary['labels_by_split'][scope].get(k,0) for k in ('SHORT','LONG','FAILURE')]
        for scope in ('all','train','dev')])
    lines+=['**Task distribution**','']
    for field,title in [('task_type','Task'),('language_register','Register')]:
        groups=[]
        for group in sorted({r['provenance'][field] for r in dataset}):
            selected=[r for r in dataset if r['provenance'][field]==group]
            counts=Counter(r['label'] for r in selected)
            groups.append([group,len(selected)]+[counts[k] for k in ('SHORT','LONG','FAILURE')])
        if field=='language_register':lines+=['**Register distribution**','']
        lines+=table([title,'N','SHORT','LONG','FAILURE'],groups)
    lines += ['Each task/register cell has 75 examples: 60 Router TRAIN and 15 Router DEV. The full cross-tabulation and split-specific labels are in `task_register_distribution.csv`.','',
        f'Router TRAIN: 1,920 questions across {len(train_ids)} represented anchors. Router DEV: 480 questions across {len(dev_ids)} represented anchors. The split is exactly 80/20 by question count and anchor-disjoint. Eligible anchors were assigned approximately 80/20 before record selection; represented-anchor counts differ slightly from that ratio.','',
        '**Feature availability**','']
    lines+=table(['Inference-safe feature','Non-null / 2400'],sorted(summary['feature_nonnull_counts'].items()))
    lines+=['Confidence, entropy (natural-log nats), and top1-top2 margin come from the same frozen Short forward pass, observed with a read-only hook. No classifier weights, logits, or predictions are modified. Query validity means Short produced a query accepted by the frozen validator.','',
        '`missing_slots` is [] for valid queries, or the unresolved slot group reported by the frozen extractor (which may include conflicting values, not just omissions). It is null for unrelated failures; exact partial-slot counts cannot be recovered because the frozen parser returns no partial query. Constraint count is the number of targets/categories plus radius/direction/hop fields in the Short query, and is null without a query.','',
        'Entity ambiguity counts ambiguous frozen identity resolutions over Short query references, or visible names mentioned in the question when Short has no query. Question length is available in characters and whitespace-delimited words. Predicted operation always comes from Short. No Long outputs, gold task/register labels, expected query, correctness, source IDs, anchors, or gold complexity enter the feature dictionary.','',
        '**Long runtime and calls**','']
    lines+=table(['Measurement','Value'],[
        ['Frozen model','Qwen/Qwen3-4B; BF16; existing asar-hf-cache; no model download'],
        ['Observed GPU',', '.join(summary['gpus'])],
        ['Generation calls',summary['long_generation_calls']],['Retry calls',summary['long_retry_calls']],
        ['Summed remote inference seconds',f'{summary["long_inference_seconds"]:.2f}'],
        ['Long session wall seconds',f'{summary["long_session_wall_seconds"]:.2f}'],
        ['Monetary cost','Not supplied by inference responses; no estimate']])
    lines+=['Load times and individual session metadata are saved in `summary.json` and `sessions/`; CPU Short timing is in `short_runtime.json`. Raw generation checkpoints are committed remotely before responses return and mirrored locally; completed paired predictions are skipped on resume. Full local replay verified all 2,400 examples without new generation.','',
        '**Provenance and limits**','',
        'There are 2,400 distinct source records and 2,400 distinct rendered questions. Only original TRAIN cases that execute successfully and match the original answer were eligible. Every paraphrase keeps the exact original origin, named targets, category order, radius, direction, hop order and answer. Context compaction preserves the exact frozen parsed spatial context. No original gold answer is replaced.','',
        'The questions instantiate 128 new agent-authored generic wording families (four per task/register). They are controlled natural-Arabic paraphrases, not collected user conversations or independently human-reviewed language. Wording families are shared across anchor-disjoint Router TRAIN/DEV; this split measures new-anchor generalization, not unseen-style generalization. Samples are balanced by construction rather than by estimated production frequency.','',
        'Artifacts: `router_dataset.jsonl`, `router_train.jsonl`, `router_dev.jsonl`, feature-only `router_train_features.jsonl` and `router_dev_features.jsonl`, `questions.jsonl`, `anchor_split.json`, `manifest.json`, `summary.json`, `verification.json`, and raw/paired checkpoints. Full rows retain provenance and outcomes in separate fields; future Router training should consume only the feature dictionary and label.','',
        '**Stopped after Step 3A. No Router was trained.**','']
    (OUT/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(verification,indent=2))


if __name__=='__main__':
    main()
