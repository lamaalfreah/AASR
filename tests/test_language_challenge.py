from collections import Counter
from dataclasses import asdict,replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from evaluation.build_language_challenge import (OUT,DEV_INDICES,read_jsonl,register_level,
    frozen_hashes,semantic_check,train_records)
from evaluation.evaluate_language_challenge import paired,score_long
from language.long_parser import (LongParser,ProviderResponse,parse_json,safe_context,query_from_dict)
from language.qwen_provider import QwenProvider,ProviderUnavailable
from spatial import parse,parse_context,execute
from spatial.normalization import answer_matches
from spatial.schema import Location
from spatial.extensions import SiteQuery,execute_site_query
from test_spatial import ROWS,QUESTIONS,context


class FakeProvider:
    def __init__(self,outputs):self.outputs=iter(outputs);self.messages=[]
    def complete(self,messages):
        self.messages.append(json.loads(json.dumps(messages)))
        return ProviderResponse(next(self.outputs),10,20,'qwen3-4b')


class LanguageChallengeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.items=read_jsonl(OUT/'challenge/core_challenge.jsonl')
        cls.sources={r['source_record_id']:r for r in read_jsonl(OUT/'challenge/source_contexts.jsonl')}

    def test_counts_registers_and_complexity(self):
        self.assertEqual(len(self.items),320)
        self.assertEqual(set(Counter(r['task_type'] for r in self.items).values()),{40})
        self.assertEqual(Counter(r['language_register'] for r in self.items),
                         {'natural_msa':128,'conversational':96,'light_saudi':64,'indirect_compositional':32})
        self.assertEqual(Counter(r['complexity_level'] for r in self.items),{'L1':96,'L2':96,'L3':64,'L4':64})

    def test_splits_sources_anchors_wording_disjoint(self):
        dev=[r for r in self.items if r['challenge_split']=='dev'];ev=[r for r in self.items if r['challenge_split']=='eval']
        self.assertEqual((len(dev),len(ev)),(192,128))
        for field in ('source_record_id','wording_family','challenge_question'):
            self.assertFalse({r[field] for r in dev}&{r[field] for r in ev})
        anchors=lambda rows:{parse_context(self.sources[r['source_record_id']]['context']).anchor for r in rows}
        self.assertFalse(anchors(dev)&anchors(ev))
        self.assertEqual(set(Counter(r['task_type'] for r in dev).values()),{24})
        self.assertEqual(set(Counter(r['task_type'] for r in ev).values()),{16})

    def test_every_challenge_semantically_preserved(self):
        bank=json.loads((OUT/'challenge/paraphrase_bank.json').read_text())
        for item in self.items:
            with self.subTest(challenge=item['challenge_id']):
                source={'id':item['source_record_id'],'split':'train','context':self.sources[item['source_record_id']]['context'],
                        'question':item['original_question'],'answer':item['gold_answer']}
                i=int(item['wording_family'].rsplit(':',1)[1])-1
                checks=semantic_check(item,source,bank[item['task_type']][i],i)
                self.assertTrue(all(checks.values()))
                q=query_from_dict(item['expected_structured_query'])
                result=execute(q,parse_context(source['context']))
                self.assertTrue(answer_matches(result.answer,item['gold_answer']))

    def test_semantic_mutation_rejected(self):
        item=dict(self.items[0]);item['challenge_question']+=' ضمن خمسة كيلومترات'
        source={'id':item['source_record_id'],'split':'train','context':self.sources[item['source_record_id']]['context'],
                'question':item['original_question'],'answer':item['gold_answer']}
        bank=json.loads((OUT/'challenge/paraphrase_bank.json').read_text())
        i=int(item['wording_family'].rsplit(':',1)[1])-1
        with self.assertRaisesRegex(ValueError,'Unreviewed'):
            semantic_check(item,source,bank[item['task_type']][i],i)

    def test_no_test_raw_access(self):
        with self.assertRaisesRegex(ValueError,'TRAIN'):
            next(train_records(Path('/nonexistent/test.jsonl.gz')))
        frozen_hashes()

    def test_safe_inference_context_has_no_gold_or_coordinates(self):
        safe=safe_context(parse_context(context(ROWS)))
        self.assertEqual(set(safe),{'context_anchor_name','candidate_names','category_vocabulary'})
        encoded=json.dumps(safe)
        for word in ('latitude','longitude','gold_answer','expected_structured_query','task_type','record_id'):
            self.assertNotIn(word,encoded)

    def response(self):
        q=parse(QUESTIONS['nearest_category'],context(ROWS))
        return json.dumps({'status':'success','query':asdict(q),'message':''},ensure_ascii=False)

    def test_long_json_valid_and_typed(self):
        outcome=parse_json(self.response(),parse_context(context(ROWS)))
        self.assertEqual(outcome.query.operation,'nearest_category')

    def test_strict_json_rejects_invalid_types_extras_duplicates_fences(self):
        for text in ('```json\n'+self.response()+'\n```','{"status":"unsupported","query":null,"message":"","status":"success"}',
                     '{"status":"success","query":NaN,"message":""}',self.response()[:-1]+',"answer":"leak"}'):
            with self.subTest(text=text[:30]),self.assertRaises((ValueError,TypeError)):
                parse_json(text)
        data=json.loads(self.response())
        data['query']['categories']=[True]
        with self.assertRaises(ValueError):parse_json(json.dumps(data))
        data=json.loads(self.response());data['query']['origin']['name']='invented'
        with self.assertRaisesRegex(ValueError,'invented'):
            parse_json(json.dumps(data),parse_context(context(ROWS)))

    def test_invalid_long_response_bounded_retry(self):
        provider=FakeProvider(['bad',self.response()]);parser=LongParser(provider)
        outcome=parser.parse('سؤال',parse_context(context(ROWS)))
        self.assertEqual((outcome.status,outcome.llm_calls,outcome.invalid_responses),('success',2,1))
        self.assertEqual((outcome.input_tokens,outcome.output_tokens),(20,40))
        provider=FakeProvider(['bad','bad']);outcome=LongParser(provider).parse('سؤال',parse_context(context(ROWS)))
        self.assertEqual((outcome.status,outcome.llm_calls),('invalid_output',2))
        with self.assertRaises(ValueError):LongParser(provider,3)

    def test_clarification_and_unsupported(self):
        for status in ('needs_clarification','unsupported'):
            outcome=parse_json(json.dumps({'status':status,'query':None,'message':'حدد المطلوب'}))
            self.assertEqual(outcome.status,status)
            self.assertIsNone(outcome.query)

    def test_gold_not_sent_to_provider(self):
        provider=FakeProvider([self.response()])
        item={'challenge_question':'سؤال طبيعي','expected_structured_query':json.loads(self.response())['query'],
              'gold_answer':'SENTINEL_GOLD','task_type':'SENTINEL_TASK','source_record_id':'SENTINEL_SOURCE'}
        scores=score_long(item,parse_context(context(ROWS)),LongParser(provider))
        self.assertTrue(scores['long_query_exact'])
        payload=json.dumps(provider.messages)
        self.assertNotIn('SENTINEL',payload)
        self.assertNotIn('expected_structured_query',payload)

    def test_provider_configuration_and_exact_model_gate(self):
        with patch.dict('os.environ',{},clear=True),self.assertRaises(ProviderUnavailable):QwenProvider()
        env={'DASHSCOPE_API_KEY':'fake-test-only','DASHSCOPE_BASE_URL':'https://dashscope-intl.aliyuncs.com/compatible-mode/v1'}
        with patch.dict('os.environ',env,clear=True):
            provider=QwenProvider()
            with self.assertRaisesRegex(ProviderUnavailable,'verify_model'):provider.complete([])
            with patch.object(provider,'_request',return_value={'output':{'models':[{'model':'qwen3-8b'}]}}):
                with self.assertRaisesRegex(ProviderUnavailable,'not listed'):provider.verify_model()
            self.assertFalse(provider.verified)
            with patch.object(provider,'_request',return_value={'output':{'models':[{'model':'qwen3-4b'}]}}):
                self.assertTrue(provider.verify_model()['exact_model_listed'])

    def test_regression_undefined_without_short_correct(self):
        result=paired([{'short_correct':False,'long_correct':True,'outcome':'short_wrong/long_correct'}])
        self.assertEqual(result['long_rescue_percent'],100)
        self.assertIsNone(result['long_regression_percent'])

    def test_provider_rejects_model_substitution(self):
        env={'DASHSCOPE_API_KEY':'fake-test-only','DASHSCOPE_BASE_URL':'https://dashscope-intl.aliyuncs.com/compatible-mode/v1'}
        with patch.dict('os.environ',env,clear=True):
            provider=QwenProvider();provider.verified=True
            with patch.object(provider,'_request',return_value={'model':'qwen3-8b'}):
                with self.assertRaisesRegex(ProviderUnavailable,'different model'):provider.complete([])

    def test_count_collision_source_cannot_be_bypassed(self):
        ctx=parse_context(context(ROWS))
        q=asdict(parse(QUESTIONS['count_within_radius'],ctx));q['origin']['source']='context_anchor'
        text=json.dumps({'status':'success','query':q,'message':''})
        with self.assertRaisesRegex(ValueError,'unresolved'):parse_json(text,ctx)


class ExtensionTests(unittest.TestCase):
    def setUp(self):
        self.anchor=Location('a','مرجع',0,0)
        self.facilities=(Location('h','مستشفى',0,0,'مستشفى'),)
        self.sites=(Location('s1','١',0,.1),Location('s2','٢',0,.2),Location('s3','٣',.3,0))

    def test_best_requires_objective(self):
        result=execute_site_query(SiteQuery(facility_category='مستشفى'),self.sites,self.facilities,self.anchor)
        self.assertEqual(result.status,'needs_clarification')

    def test_unsupported_operation_and_external_data(self):
        for q in (SiteQuery(operation='diagnose'),SiteQuery(required_data=('population',)),SiteQuery(objective='best_healthcare')):
            self.assertEqual(execute_site_query(q,self.sites,self.facilities,self.anchor).status,'unsupported')

    def test_objective_max_min_ranking_and_exclusion(self):
        q=SiteQuery(objective='maximize_nearest_facility_distance',facility_category='مستشفى')
        self.assertEqual(execute_site_query(q,self.sites,self.facilities,self.anchor).selected_ids,('s3',))
        r=execute_site_query(replace(q,objective='minimize_nearest_facility_distance'),self.sites,self.facilities,self.anchor)
        self.assertEqual(r.selected_ids,('s1',))
        self.assertEqual(execute_site_query(replace(q,top_k=2),self.sites,self.facilities,self.anchor).selected_ids,('s3','s2'))
        self.assertEqual(execute_site_query(replace(q,min_distance_km=40),self.sites,self.facilities,self.anchor).status,'not_found')
        self.assertEqual(execute_site_query(replace(q,direction='شرق'),self.sites,self.facilities,self.anchor).selected_ids,('s2',))

    def test_invalid_objective_arguments_and_missing_data(self):
        q=SiteQuery(objective='maximize_nearest_facility_distance',facility_category='مستشفى')
        for q2 in (replace(q,min_distance_km=-1),replace(q,min_distance_km=float('nan')),replace(q,top_k=True)):
            self.assertEqual(execute_site_query(q2,self.sites,self.facilities,self.anchor).status,'invalid_query')
        self.assertEqual(execute_site_query(q,(),self.facilities,self.anchor).status,'needs_clarification')
        self.assertEqual(execute_site_query(q,self.sites,(),self.anchor).status,'unsupported')

    def test_rank_ties_preserved(self):
        sites=(self.sites[0],replace(self.sites[0],identity='different'))
        q=SiteQuery(objective='maximize_nearest_facility_distance',facility_category='مستشفى')
        self.assertEqual(len(execute_site_query(q,sites,self.facilities,self.anchor).selected_ids),2)

    def test_extension_counts_and_separation(self):
        items=read_jsonl(OUT/'challenge/extension_challenge.jsonl')
        self.assertEqual(len(items),80)
        self.assertEqual(Counter(i['intent_kind'] for i in items),{'underspecified_best':20,'missing_external_data':20,'explicit_geometric_objective':40})
        self.assertTrue(all(i['limitation'].endswith('not an LLM-parsed intent.') for i in items))
