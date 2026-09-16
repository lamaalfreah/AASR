import tempfile
import unittest
from pathlib import Path
from evaluation.evaluate_structured_baseline import evaluate_split, infer_record, summarize
from test_spatial import QUESTIONS, ROWS, context


class EvaluationTests(unittest.TestCase):
    def test_test_split_refused_before_io(self):
        with self.assertRaisesRegex(ValueError, 'test gold is locked'):
            evaluate_split(Path('/does-not-exist'), 'test', {})

    def test_inference_never_reads_gold(self):
        class Guard(dict):
            def __getitem__(self, key):
                if key not in ('question','context'):
                    raise AssertionError('Gold field read by runtime')
                return super().__getitem__(key)
        row = Guard(question=QUESTIONS['nearest_category'], context=context(ROWS))
        self.assertEqual(infer_record(row).answer.text, 'مدرسة أولى')

    def test_gzip_evaluation_and_denominators(self):
        import gzip
        import json
        records = [dict(id='a', question=QUESTIONS['nearest_category'], context=context(ROWS),
                        task_type='nearest_category', answer='مدرسة أولى'),
                   dict(id='b', question='unsupported', context=context(ROWS),
                        task_type='nearest_category', answer='مدرسة أولى')]
        with tempfile.TemporaryDirectory() as directory:
            with gzip.open(Path(directory)/'validation.jsonl.gz', 'wt', encoding='utf-8') as f:
                for r in records:
                    f.write(json.dumps(r, ensure_ascii=False)+'\n')
            metrics, rows = evaluate_split(Path(directory), 'validation', {('validation','a'):'A',('validation','b'):'G'})
        self.assertEqual(metrics['total'], 2)
        self.assertEqual(metrics['accuracy_percent'], 50)
        self.assertEqual(metrics['reconstructable_accuracy_percent'], 100)
        self.assertEqual(metrics['abstention_percent'], 50)
        self.assertEqual(metrics['failure_stage_counts'], {'parser': 1})
        self.assertEqual(metrics['parse_success'], 1)
