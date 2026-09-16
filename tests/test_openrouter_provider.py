import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from language.openrouter_provider import OpenRouterProvider,ProviderStopped,MODEL,atomic_json,load_local_key
from language.long_parser import LongParser
from spatial import parse,parse_context,execute
from dataclasses import asdict
from test_spatial import context,ROWS,QUESTIONS


class OpenRouterTests(unittest.TestCase):
    def provider(self,directory):
        env=Path(directory)/'.env';env.write_text('OPENROUTER_API_KEY="unit-test-placeholder"\n')
        return OpenRouterProvider(Path(directory)/'calls',env)

    def response(self,text=None):
        if text is None:text=json.dumps({'status':'success','query':asdict(parse(QUESTIONS['nearest_category'],context(ROWS))),'message':''})
        return {'model':MODEL,'id':'unit-test-response','choices':[{'message':{'content':text}}],
                'usage':{'prompt_tokens':10,'completion_tokens':20,'cost':0}}

    def test_local_key_only_and_auth_failure_does_not_reach_models(self):
        with tempfile.TemporaryDirectory() as d,patch.dict('os.environ',{},clear=True):
            p=self.provider(d)
            with patch.object(p,'request',side_effect=ProviderStopped('authentication','HTTP 401')) as req:
                with self.assertRaises(ProviderStopped):p.verify_model()
                self.assertEqual(req.call_args_list[0].args,('/key',));self.assertEqual(req.call_count,1)

    def test_exact_model_gate_and_free_pricing(self):
        with tempfile.TemporaryDirectory() as d,patch.dict('os.environ',{},clear=True):
            p=self.provider(d)
            with patch.object(p,'request',side_effect=[{}, {'data':[{'id':'qwen/qwen3-8b:free'}]}]):
                with self.assertRaisesRegex(ProviderStopped,'absent'):p.verify_model()
            self.assertFalse(p.verified)
            with patch.object(p,'request',side_effect=[{}, {'data':[{'id':MODEL,'pricing':{'prompt':'0','completion':'0'}}]}]):
                self.assertTrue(p.verify_model()['exact_model_listed'])

    def test_checkpoint_survives_restart_and_skips_completed_call(self):
        with tempfile.TemporaryDirectory() as d,patch.dict('os.environ',{},clear=True):
            p=self.provider(d);p.verified=True;p.begin_case('dev:fixture')
            with patch.object(p,'request',return_value=self.response()) as req:
                first=p.complete([{'role':'user','content':'question only'}]);self.assertEqual(req.call_count,1)
            files=list((Path(d)/'calls').glob('*.json'));self.assertEqual(len(files),1)
            self.assertNotIn('unit-test-placeholder',files[0].read_text())
            q=self.provider(d);q.verified=True;q.begin_case('dev:fixture')
            with patch.object(q,'request',side_effect=AssertionError('Repeated completed call')):
                second=q.complete([{'role':'user','content':'question only'}])
            self.assertEqual(first,second);self.assertEqual(q.replayed_calls,1)

    def test_invalid_output_checkpointed_before_bounded_retry(self):
        with tempfile.TemporaryDirectory() as d,patch.dict('os.environ',{},clear=True),patch('language.openrouter_provider.time.sleep'):
            p=self.provider(d);p.verified=True;p.begin_case('dev:fixture')
            with patch.object(p,'request',side_effect=[self.response('invalid JSON'),self.response()]):
                outcome=LongParser(p).parse(QUESTIONS['nearest_category'],parse_context(context(ROWS)))
            self.assertEqual(outcome.llm_calls,2);self.assertEqual(outcome.invalid_responses,1)
            self.assertEqual(len(list((Path(d)/'calls').glob('*.json'))),2)
            result=execute(outcome.query,parse_context(context(ROWS)))
            self.assertEqual(result.answer.text,'مدرسة أولى')

    def test_rate_limit_preserves_prior_checkpoint_and_no_auto_retry(self):
        with tempfile.TemporaryDirectory() as d,patch.dict('os.environ',{},clear=True),patch('language.openrouter_provider.time.sleep'):
            p=self.provider(d);p.verified=True;p.begin_case('dev:first')
            with patch.object(p,'request',return_value=self.response()):p.complete([])
            p.begin_case('dev:second')
            with patch.object(p,'request',side_effect=ProviderStopped('rate_limit','HTTP 429')) as req:
                with self.assertRaises(ProviderStopped):p.complete([])
                self.assertEqual(req.call_count,1)
            self.assertEqual(len(list((Path(d)/'calls').glob('*.json'))),1)

    def test_no_gold_or_credentials_in_generation_body(self):
        with tempfile.TemporaryDirectory() as d,patch.dict('os.environ',{},clear=True):
            p=self.provider(d);p.verified=True;p.begin_case('local-id-not-for-model')
            with patch.object(p,'request',return_value=self.response()) as req:
                LongParser(p).parse(QUESTIONS['nearest_category'],parse_context(context(ROWS)))
            payload=json.dumps(req.call_args.args[1])
            for forbidden in ('unit-test-placeholder','local-id-not-for-model','gold_answer','expected_structured_query','source_record_id','complexity_level','language_register'):
                self.assertNotIn(forbidden,payload)

    def test_different_model_response_refused_but_checkpoint_retained(self):
        with tempfile.TemporaryDirectory() as d,patch.dict('os.environ',{},clear=True):
            p=self.provider(d);p.verified=True;p.begin_case('dev:fixture')
            response=self.response();response['model']='qwen/qwen3-8b:free'
            with patch.object(p,'request',return_value=response):
                with self.assertRaisesRegex(ProviderStopped,'not the requested'):p.complete([])
            self.assertEqual(len(list((Path(d)/'calls').glob('*.json'))),1)
