#!/usr/bin/env python3
"""80 exploratory language intents with explicit, separately executable contracts."""
from dataclasses import asdict
import json
from pathlib import Path
import sys
BASE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(BASE))
from evaluation.build_language_challenge import OUT,read_jsonl,write_jsonl,frozen_hashes
from spatial import parse_context
from spatial.extensions import SiteQuery,execute_site_query

CLARIFY=[
'ما أفضل مكان لإنشاء منشأة صحية شمال {a}؟',
'أبي أفتح عيادة حول {a}، وين الأنسب؟',
'وين تنصح أختار موقع قريب من {a}؟',
'أي موقع من المواقع المعروضة أفضل لمستشفى قرب {a}؟',
'لو بنفتح مركزًا صحيًا عند {a}، أي مكان تختار؟',
'أبحث عن موقع مناسب لخدمة الناس حول {a}. ما اقتراحك؟',
'وش أحسن نقطة لعيادة شمال {a}؟',
'أحتاج موقعًا ممتازًا لمرفق صحي في منطقة {a}. أين أضعه؟',
'عندي أكثر من موقع حول {a}، أي واحد يناسب المشروع أكثر؟',
'كيف أختار أفضل نقطة لمركز طبي بالقرب من {a}؟',
'ودي بمكان كويس لعيادة في جهة شمال {a}. وين؟',
'ما الموقع الأفضل صحيًا من هذه المواقع قرب {a}؟',
'نبي نختار موقعًا لمستشفى عند {a}، رشّح لنا واحدًا.',
'أي مكان تنصح به لمشروعي الصحي حول {a}؟',
'أنا محتار بين المواقع القريبة من {a}، وش الأنسب؟',
'لو الأمر بيدك، أين تضع العيادة قرب {a}؟',
'أريد قرارًا بشأن أفضل موقع لخدمة المنطقة حول {a}.',
'أبي موقعًا له ميزة على الباقي شمال {a}. وش تقترح؟',
'من المواقع المعروضة حول {a}، أيها أفضل للخدمات الصحية؟',
'هل تختار لي موقعًا جيدًا لمركز صحي عند {a}؟'
]
UNSUPPORTED=[
('population','أي موقع حول {a} يخدم أكبر عدد من السكان؟'),
('traffic','وش الموقع الأسهل وصولًا بالسيارة وقت الزحمة حول {a}؟'),
('land_price','أي أرض قرب {a} أقل سعرًا لإنشاء عيادة؟'),
('healthcare_demand','ما الموقع الذي يغطي أكبر نقص في الخدمة الصحية حول {a}؟'),
('demographics','أين أضع مركز كبار السن حول {a} حسب توزيع الأعمار؟'),
('road_network','أي موقع حول {a} يصل إليه الإسعاف بأقل زمن قيادة؟'),
('land_ownership','أبي أرضًا متاحة للشراء شمال {a}، أي موقع منها مملوك للبلدية؟'),
('zoning','أي موقع حول {a} يسمح نظامه العمراني بمستشفى؟'),
('public_transport','ما الموقع الأقرب بحافلة عامة للسكان حول {a}؟'),
('facility_capacity','أي موقع عند {a} يقلل الضغط على المستشفيات حسب عدد أسرّتها؟'),
('population','أبي أكثر موقع حول {a} قريب من كثافة سكانية عالية.'),
('traffic','رتّب المواقع حول {a} حسب وقت الوصول في ساعة الذروة.'),
('land_price','وين أقل تكلفة لاستئجار أرض لعيادة حول {a}؟'),
('healthcare_demand','رشّح موقعًا قرب {a} حسب معدلات المرض في الأحياء.'),
('demographics','أي موقع حول {a} يخدم أكبر عدد من الأطفال؟'),
('road_network','أريد موقعًا لا يتجاوز الوصول إليه عشر دقائق بالسيارة من {a}.'),
('land_ownership','ما المواقع الخالية وغير المشغولة قرب {a}؟'),
('zoning','أي المواقع حول {a} حاصل على ترخيص منشأة صحية؟'),
('public_transport','أبي موقعًا حول {a} مناسبًا لذوي الإعاقة حسب وسائل النقل.'),
('facility_capacity','وين الأفضل حول {a} إذا حسبنا تخصصات المستشفيات وقدرتها الاستيعابية؟')
]
SOLVABLE=[
'من النقاط المعروضة حول {a}، أي نقطة أبعد عن أقرب مستشفى مذكور؟ اعتبر الأفضلية للبُعد فقط.',
'أبي النقطة اللي أقرب مستشفى مذكور لها هو الأبعد مقارنة بباقي النقاط حول {a}.',
'رتّب أفضل ثلاث نقاط حول {a} حسب بُعد كل نقطة عن أقرب مستشفى مذكور، من الأبعد للأقرب.',
'أي نقطة {d} {a} تزيد فيها المسافة إلى أقرب مستشفى مذكور على بقية النقاط في الجهة نفسها؟',
'من النقاط المقترحة {d} {a}، اختر الأبعد عن أقرب مستشفى مذكور، بشرط بُعد خمسة كيلومترات على الأقل.',
'وين النقطة الأقرب إلى أقرب مستشفى مذكور من بين النقاط المعروضة حول {a}؟ القرب هو المعيار الوحيد.',
'رتّب ثلاث نقاط حول {a} من الأقل إلى الأكثر بُعدًا عن أقرب مستشفى مذكور لكل منها.',
'أبي النقطة الأبعد عن أقرب مستشفى مذكور حول {a}، ولا يقل البُعد عن كيلومتر واحد.',
'بعد استبعاد النقاط التي يبعد أقرب مستشفى مذكور عنها أقل من ثلاثة كيلومترات، أي نقطة حول {a} هي الأبعد عن أقرب مستشفى؟',
'في جهة {d} من {a}، أي نقطة مقترحة هي الأقرب إلى أقرب مستشفى مذكور؟'
]


def main():
    frozen_hashes()
    contexts=read_jsonl(OUT/'challenge/source_contexts.jsonl')
    eligible=[]
    for row in contexts:
        ctx=parse_context(row['context'])
        hospitals=tuple(p for p in ctx.candidates if p.category=='مستشفى')
        sites=tuple(p for p in ctx.candidates if p.category!='مستشفى')[:8]
        if hospitals and len(sites)>=3:eligible.append((row,ctx,sites,hospitals))
    if len(eligible)<80:raise RuntimeError('Need 80 input contexts with supplied sites/facilities')
    items=[]
    for i,(row,ctx,sites,hospitals) in enumerate(eligible[:80]):
        a='«'+ctx.anchor.name+'»'; d=('شمال','شرق','جنوب','غرب')[i%4]
        if i<20:
            question=CLARIFY[i].format(a=a);q=SiteQuery(facility_category='مستشفى');kind='underspecified_best'
        elif i<40:
            requirement,template=UNSUPPORTED[i-20]
            question=template.format(a=a);q=SiteQuery(required_data=(requirement,));kind='missing_external_data'
        else:
            j=(i-40)%10
            q=SiteQuery(objective='minimize_nearest_facility_distance' if j in (5,6,9) else 'maximize_nearest_facility_distance',
                        facility_category='مستشفى',direction=d if j in (3,4,9) else None,
                        min_distance_km=5 if j==4 else 1 if j==7 else 3 if j==8 else None,
                        top_k=3 if j in (2,6) else 1)
            question=SOLVABLE[j].format(a=a,d=d);kind='explicit_geometric_objective'
        result=execute_site_query(q,sites,hospitals,ctx.anchor)
        items.append({'challenge_id':f'extension-{i+1:03d}','source_record_id':row['source_record_id'],
                      'challenge_question':question,'intent_kind':kind,'review_status':'agent_reviewed_contract_only',
                      'inference_context':{'anchor':asdict(ctx.anchor),'supplied_sites':[asdict(p) for p in sites],
                                           'listed_facilities':[asdict(p) for p in hospitals],
                                           'site_role':'Illustrative reference points copied from existing POIs; not available land or verified proposed sites'},
                      'expected_extension_query':asdict(q),'expected_status':result.status,
                      'expected_selected_ids':list(result.selected_ids),
                      'limitation':'Only supplied points/listed facilities. No healthcare suitability claim. Extension evaluation uses expected intent contract, not an LLM-parsed intent.'})
    write_jsonl(OUT/'challenge/extension_challenge.jsonl',items)
    print('Created 80 extension intents: 20 clarification, 20 missing-data, 40 geometric contracts.')


if __name__=='__main__':main()
