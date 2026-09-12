import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
from dataclasses import asdict

from language.modal_checkpoint import MODEL, ModalCheckpointProvider
from language.long_parser import LongParser, parse_json
from spatial import parse, parse_context, execute
from test_spatial import context, ROWS, QUESTIONS
from evaluation.evaluate_modal_long import paired, run_locked


class ModalLongTests(unittest.TestCase):
    def response(self, messages, key, text=None):
        if text is None:
            text = json.dumps(dict(status='success', query=asdict(parse(
                QUESTIONS['nearest_category'], context(ROWS))), message=''))
        return dict(model=MODEL, revision='fixture-revision', request_digest=key,
                    text=text, input_tokens=10, output_tokens=20, inference_ms=25)

    def provider(self, directory, remote):
        p = ModalCheckpointProvider(remote, directory, {'revision': 'fixture-revision'})
        p.begin_case('local-case')
        return p

    def test_restart_skips_completed_generation(self):
        with tempfile.TemporaryDirectory() as d:
            remote = Mock(side_effect=self.response)
            p = self.provider(d, remote)
            first = p.complete([])
            self.assertEqual(remote.call_count, 1)
            second = self.provider(d, Mock(side_effect=AssertionError('repeated inference'))).complete([])
            self.assertEqual(first, second)
            self.assertEqual(len(list(Path(d).glob('*.json'))), 1)

    def test_invalid_output_saved_before_one_retry_and_geo_integration(self):
        with tempfile.TemporaryDirectory() as d:
            calls = []
            def remote(messages, key):
                calls.append(key)
                if len(calls) == 1:
                    return self.response(messages, key, 'not json')
                self.assertEqual(len(list(Path(d).glob('*.json'))), 1)
                return self.response(messages, key)
            ctx = parse_context(context(ROWS))
            result = LongParser(self.provider(d, remote)).parse(QUESTIONS['nearest_category'], ctx)
            self.assertEqual(result.invalid_responses, 1)
            self.assertEqual(result.llm_calls, 2)
            self.assertEqual(execute(result.query, ctx).answer.text, 'مدرسة أولى')
            self.assertEqual(len(list(Path(d).glob('*.json'))), 2)

    def test_interrupted_retry_resumes_without_repeating_first_call(self):
        with tempfile.TemporaryDirectory() as d:
            count = []
            def remote(messages, key):
                count.append(key)
                if len(count) == 1:
                    return self.response(messages, key, '{}')
                raise RuntimeError('simulated GPU interruption')
            ctx = parse_context(context(ROWS))
            with self.assertRaises(RuntimeError):
                LongParser(self.provider(d, remote)).parse(QUESTIONS['nearest_category'], ctx)
            resumed = Mock(side_effect=self.response)
            outcome = LongParser(self.provider(d, resumed)).parse(QUESTIONS['nearest_category'], ctx)
            self.assertEqual(resumed.call_count, 1)
            self.assertEqual(outcome.status, 'success')

    def test_payload_has_only_public_fields(self):
        with tempfile.TemporaryDirectory() as d:
            remote = Mock(side_effect=self.response)
            LongParser(self.provider(d, remote)).parse(QUESTIONS['nearest_category'], parse_context(context(ROWS)))
            messages = remote.call_args.args[0]
            payload = json.loads(messages[1]['content'])
            self.assertEqual(set(payload), {'question', 'context'})
            self.assertEqual(set(payload['context']), {'context_anchor_name', 'candidate_names', 'category_vocabulary'})
            for forbidden in ('local-case', 'gold_answer', 'expected_structured_query', 'source_record_id', 'complexity_level', 'language_register', 'latitude', 'longitude'):
                self.assertNotIn(forbidden, json.dumps(messages))

    def test_strict_schema_rejects_extra_field_and_fenced_json(self):
        response = self.response([], 'key')['text']
        data = json.loads(response)
        data['query']['final_answer'] = 'forbidden'
        with self.assertRaises(ValueError):
            parse_json(json.dumps(data))
        with self.assertRaises(ValueError):
            parse_json('```json\n'+response+'\n```')

    def test_pairwise_denominators_at_both_levels(self):
        rows = [dict(short_query_exact=s, long_query_exact=l,
                     short_answer_correct=l, long_answer_correct=s)
                for s, l in [(True, True), (True, False), (False, True), (False, False), (False, True)]]
        q = paired(rows, 'query_exact')
        self.assertEqual(q['long_only'], 2)
        self.assertAlmostEqual(q['rescue_percent'], 200/3)
        self.assertEqual(q['regression_percent'], 50)
        a = paired(rows, 'answer_correct')
        self.assertEqual(a['rescue_percent'], 50)
        self.assertAlmostEqual(a['regression_percent'], 200/3)

    def test_failed_smoke_stops_before_remote_allocation(self):
        with patch('evaluation.evaluate_modal_long.frozen_check'), \
             patch('evaluation.evaluate_modal_long.saved_rows', side_effect=[[], [{'long_query_exact': False}]]), \
             patch('evaluation.evaluate_modal_long.progress'), patch('evaluation.evaluate_modal_long.report'):
            prepare, model = Mock(), Mock()
            run_locked(prepare, model, 'fixture')
            prepare.assert_not_called(); model.assert_not_called()

    def test_different_model_is_persisted_and_refused(self):
        with tempfile.TemporaryDirectory() as d:
            def remote(messages, key):
                r = self.response(messages, key)
                r['model'] = 'different-model'
                return r
            with self.assertRaisesRegex(RuntimeError, 'identity mismatch'):
                self.provider(d, remote).complete([])
            self.assertEqual(len(list(Path(d).glob('*.json'))), 1)
