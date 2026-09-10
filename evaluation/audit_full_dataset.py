#!/usr/bin/env python3
"""CPU-only, streaming dataset audit. Not a production query parser/executor.

Raw files are read-only. Test gold values are excluded from answer/semantic
analysis. Schema presence/null/type inspection includes every split, but test
gold values are never enumerated, compared, or used to select conventions.
Geometry hypotheses are fixed in this audit and checked on train/validation.
"""
import argparse
import collections as co
import csv
import difflib
import functools
import gzip
import hashlib
import itertools
import io
import json
import math
from pathlib import Path
import re
import statistics
import tempfile
import unicodedata
import unittest

# Shared parsing/geometry, also usable when this audit is run as a script.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from spatial.context_parser import (parse_context_fields as parse_context,
                                    REG_MARK, ART_MARK, START, BODY, ANCHOR)
from spatial.query_parser import parse_question_fields as parse_question, PATTERNS
from spatial.geometry import valid_coord, distance, bearing, direction

SPLITS = ("train", "validation", "test")
TASKS = ("nearest_category", "cardinal_direction", "within_radius_yes_no",
         "closer_of_two", "count_within_radius", "nearest_of_two_categories",
         "two_hop_nearest", "spatial_multi_constraint")
GOLD = {"answer", "short_target", "long_target", "verified_rationale",
        "evidence_poi_ids", "answer_position", "intermediate_answer",
        "first_hop_category", "second_hop_category", "target_category",
        "direction_constraint", "radius_constraint_km", "num_qualifying_pois",
        "router_target_alpha"}
EXPECTED_TYPES = {
    **{k: {"str"} for k in ("id country language split context question answer "
       "task_type difficulty recommended_reasoning verified_rationale short_target "
       "long_target anchor_place wikidata_id wikipedia_title context_length_bin_chars").split()},
    **{k: {"int"} for k in ("context_chars article_chars num_context_pois").split()},
    **{k: {"int", "float"} for k in ("difficulty_score anchor_latitude anchor_longitude").split()},
    "sources": {"list"}, "evidence_poi_ids": {"list"},
    **{k: {"str", "null"} for k in ("answer_position intermediate_answer first_hop_category "
       "second_hop_category target_category direction_constraint").split()},
    **{k: {"int", "float", "null"} for k in ("radius_constraint_km "
       "num_qualifying_pois router_target_alpha").split()},
}
CAT_CODES = {"مستشفى": "hospital", "عيادة": "clinic", "صيدلية": "pharmacy",
             "مدرسة": "school", "جامعة": "university", "كلية": "college",
             "مطعم": "restaurant", "مقهى": "cafe", "بنك": "bank", "محطة وقود": "fuel"}
KNOWN_CATS = set(CAT_CODES) | {"مطعم وجبات سريعة", "صراف آلي", "روضة أطفال", "مكان عبادة"}
RECON = {"A": "deterministically reconstructable", "B": "ambiguous input",
         "C": "required candidate missing", "D": "malformed candidate registry",
         "E": "taxonomy/category uncertainty", "F": "suspected label inconsistency",
         "G": "insufficient information", "H": "other"}
FIELD_POLICY = {
    'question':('A','User question'), 'context':('A','Visible context supplied to system'),
    **{k:('B','Use only when extracted/recomputed from input or supplied by independent retrieval') for k in
       ('country language anchor_place anchor_latitude anchor_longitude wikidata_id wikipedia_title sources '
        'context_chars article_chars num_context_pois context_length_bin_chars').split()},
    **{k:('C','Training target/offline reference only; never feed raw annotation to inference') for k in
       ('task_type answer short_target long_target verified_rationale answer_position intermediate_answer '
        'first_hop_category second_hop_category target_category direction_constraint radius_constraint_km').split()},
    **{k:('D','Construction/gold metadata; not an inference feature') for k in
       ('id split difficulty difficulty_score recommended_reasoning router_target_alpha evidence_poi_ids num_qualifying_pois').split()},
}
FIX_RULES = {
    'multiline_candidate_names':('A','Parse complete numbered candidate blocks across newlines; preserve name text.'),
    'candidate_name_edge_whitespace':('A','Preserve raw names; any future normalized lookup key must be separate and collision-aware.'),
    'answer_edge_whitespace':('B','Retain raw target and define a separately documented comparison key; never silently rewrite raw labels.'),
    'duplicate_candidate_names':('C','Preserve separate candidate identities; duplicate name alone does not imply a query is ambiguous.'),
    'same_name_multiple_coordinates':('C','Keep distinct locations and flag identity uncertainty when referenced.'),
    'duplicate_candidate_coordinates':('E','Review co-located entities; do not merge on coordinates.'),
    'duplicate_candidate_rows':('E','Preserve duplicate rows until original OSM identities and multiplicity semantics are recovered.'),
    'named_identity_ambiguous':('C','Return ambiguity set; exclude from unique entity-link labels unless identity is independently supplied.'),
    'count_origin_name_collides_with_context_anchor':('C','Represent query origin explicitly; clarify whether named OSM candidate or context anchor when coordinates differ.'),
    'target_name_collides_with_context_anchor':('C','A target repeats the main anchor name at another coordinate; require explicit candidate identity before assigning a unique spatial answer.'),
    'query_origin_differs_from_context_anchor':('B','Resolve the origin named in the question from visible candidates; never substitute context main anchor.'),
    'multi_constraint_four_sector_count_differs':('B','Use separate +/-35-degree directional filter for multi-constraint tasks; four-sector classification is different.'),
    'multi_constraint_four_sector_answer_differs':('B','Use train-derived +/-35-degree filtering before nearest selection; do not change raw labels.'),
    'reconstructability_B':('D','Candidate for exclusion from single-answer/entity-identity training; retain for ambiguity-aware evaluation and clarification.'),
    'reconstructability_F':('G','Review source coordinate precision/rounding and label provenance; do not automatically relabel or drop.'),
    'answer_position_mismatch':('F','Re-derive comparison-position target from the two quoted alternatives on TRAIN only, after review.'),
    'article_chars_mismatch':('B','Treat trim-based article lengths as derived measurements; retain stored construction count.'),
    'exact_nearest_tie':('G','Preserve all exact minima; do not invent an ordinal tie break.'),
    'nearest_gap_within_0_2m':('G','Flag fixed 0.2m rounding-sensitivity envelope; inspect original precision before changing order.'),
    'radius_boundary_within_0_2m':('G','Flag fixed 0.2m radius sensitivity; threshold is diagnostic only.'),
    'repeated_name_gold_answer':('C','Answer text is not a unique entity identity; obtain ID from geometry or retain multiple matches.'),
    'count_would_change_if_names_deduplicated':('B','Count candidate records, not unique names; name deduplication changes valid counts.'),
    'intermediate_name_repeated_but_geometry_resolves':('B','Select first hop by geometry and retain its candidate identity; do not resolve the gold intermediate name by lookup.'),
}


def digest(x):
    return hashlib.sha256(x.encode("utf-8")).hexdigest()


def norm(x):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", x)).strip()


def alias(x):
    return re.sub(r"[\W_]+", "", norm(x).lower().translate(str.maketrans("أإآىة", "ااايه")))


def stats(v):
    if not v:
        return {"n": 0}
    a = sorted(v)
    def p(q):
        z = (len(a)-1)*q/100
        lo = int(z)
        return a[lo] + (a[min(lo+1, len(a)-1)]-a[lo])*(z-lo)
    return {"n": len(a), "min": a[0], "mean": round(statistics.mean(a), 6),
            **{f"p{q}": round(p(q), 6) for q in (50, 90, 95, 99)}, "max": a[-1]}





def identity(q, candidates, context_anchor=None):
    if q is None: return 'UNPARSED', []
    names = [q[k] for k in ('entity', 'entity1', 'entity2') if k in q]
    if context_anchor is not None and q['anchor'] != context_anchor:
        names.insert(0,q['anchor'])
    if not names: return 'NOT APPLICABLE', []
    matches = [[p for p in candidates if p['name'] == name] for name in names]
    if any(not m for m in matches): return 'NOT FOUND', matches
    if any(len(m) > 1 for m in matches): return 'AMBIGUOUS', matches
    return 'UNIQUE', matches


def semantic_probe(split, row, parsed, query):
    """Audit hypotheses only. No production API. Refuses test before gold access."""
    if split == 'test': return None
    t = row.get('task_type'); answer = row.get('answer')
    result = {"class": "H", "details": {}, "flags": []}
    def stop(cls, reason):
        result.update({"class": cls, "reason": reason}); return result
    if parsed['bad'] or any(x not in ('nonsequential_ordinals',) for x in parsed['errors']):
        return stop('D', 'invalid_context_or_candidate_structure')
    if query is None: return stop('G', 'unparsed_question')
    if query['operation'] != t: return stop('H', 'task_annotation_disagrees_with_input_pattern')
    if not parsed['anchor']: return stop('G', 'missing_anchor')
    an = parsed['anchor']; a = an[1:]
    ps = parsed['candidates']; d = result['details']
    if t=='count_within_radius' and query['anchor']==an[0]:
        origins=[p for p in ps if p['name']==query['anchor']]
        if any((p['lat'],p['lon'])!=a for p in origins):
            def count_from(ref):
                return str(sum(p['category']==query['category'] and distance(ref,(p['lat'],p['lon']))<=query['radius'] for p in ps))
            candidate_counts={count_from((p['lat'],p['lon'])) for p in origins}
            possible=candidate_counts|{count_from(a)}
            d.update({'origin_name_collision':True,'candidate_origin_agrees_with_gold':answer in candidate_counts,
                      'answer_in_possible_set':answer in possible,'possible_answers':sorted(possible),
                      'answer_invariant_despite_identity_ambiguity':len(possible)==1})
            return stop('B','context_anchor_and_candidate_share_name_but_coordinates_differ')
    references=[query[k] for k in ('entity','entity1','entity2') if k in query]
    if an[0] in references and any(p['name']==an[0] and (p['lat'],p['lon'])!=a for p in ps):
        d['target_context_name_collision']=True
        return stop('B','named_target_and_context_anchor_share_name_but_coordinates_differ')
    identity_status, matches = identity(query, ps, an[0])
    d['identity'] = identity_status
    if identity_status == 'NOT FOUND': return stop('C', 'named_target_missing')
    # Inspect all possible named matches, without using evidence IDs to choose one.
    if identity_status == 'AMBIGUOUS':
        if t == 'count_within_radius':
            opts=sorted({str(sum(p['category']==query['category'] and distance((origin['lat'],origin['lon']),(p['lat'],p['lon']))<=query['radius'] for p in ps)) for origin in matches[0]})
        elif t == 'cardinal_direction':
            opts = sorted({direction(bearing(a, (p['lat'], p['lon']))) for p in matches[0]})
        else:
            opts = set()
            for p, z in itertools.product(*matches):
                dp = distance(a, (p['lat'],p['lon'])); dz = distance(a, (z['lat'],z['lon']))
                if dp <= dz: opts.add(query['entity1'])
                if dz <= dp: opts.add(query['entity2'])
            opts = sorted(opts)
        d['possible_answers'] = opts; d['answer_in_possible_set'] = answer in opts
        d['answer_invariant_despite_identity_ambiguity'] = len(opts) == 1
        return stop('B', 'named_entity_identity_ambiguous')
    if query['anchor'] != an[0]:
        origins=[p for p in ps if p['name']==query['anchor']]
        if len(origins)!=1:return stop('G','query_origin_not_uniquely_resolved')
        a=(origins[0]['lat'],origins[0]['lon'])
        d['query_origin_is_candidate']=True
    cats = [query[k] for k in ('category','category1','category2') if k in query]
    if any(c not in KNOWN_CATS for c in cats): return stop('E', 'unrecognized_query_category')
    for c in cats:
        if not any(p['category'] == c for p in ps):
            # Absence is a legitimate negative/zero for radius questions.
            if t not in ('within_radius_yes_no','count_within_radius'):
                return stop('C', 'required_category_has_no_candidate')
    pos = lambda p: (p['lat'], p['lon'])
    def nearest(pool, ref):
        if not pool: return []
        values = [(distance(ref, pos(p)), p) for p in pool]
        minimum = min(z[0] for z in values)
        ordered = sorted(z[0] for z in values)
        if len(ordered)>1:
            d.setdefault('nearest_gaps_km',[]).append(ordered[1]-ordered[0])
        # Exact ties only. A separate 0.2m flag captures coordinate-rounding sensitivity.
        tied = [p for dist,p in values if dist == minimum]
        d.setdefault('selected_distances_km',[]).append(minimum)
        if len(tied)>1: result['flags'].append('exact_nearest_tie')
        if len(ordered)>1 and ordered[1]-ordered[0] <= .0002:
            result['flags'].append('nearest_gap_within_0_2m')
        return tied
    pool = lambda c: [p for p in ps if p['category'] == c]
    predicted = set()
    if t == 'cardinal_direction':
        b = bearing(a, pos(matches[0][0])); d['bearing'] = b
        d['four_match'] = direction(b) == answer; d['eight_match'] = direction(b,8) == answer
        d['distance_to_sector_boundary_deg'] = min(abs((b-v+180)%360-180) for v in (45,135,225,315))
        predicted.add(direction(b))
    elif t == 'closer_of_two':
        p,z=matches[0][0],matches[1][0]
        dp,dz=distance(a,pos(p)),distance(a,pos(z));d['comparison_gap_km']=abs(dp-dz)
        if dp<=dz: predicted.add(query['entity1'])
        if dz<=dp: predicted.add(query['entity2'])
        if abs(dp-dz)<=.0002:result['flags'].append('nearest_gap_within_0_2m')
        d['answer_position_consistent'] = row.get('answer_position') == ('first' if answer==query['entity1'] else 'second' if answer==query['entity2'] else None)
    elif t in ('within_radius_yes_no','count_within_radius'):
        ds=[distance(a,pos(p)) for p in pool(query['category'])]; radius=query['radius']
        n=sum(x<=radius for x in ds); strict=sum(x<radius for x in ds)
        d.update({'qualifying_count':n,'strict_count':strict,'exact_boundary_candidates':sum(x==radius for x in ds),
                  'nearest_radius_gap_km':min((abs(x-radius) for x in ds),default=None)})
        rounded=sum(round(x,3)<=radius for x in ds)
        d['rounded_3decimal_radius_matches_gold']=(str(rounded) if t=='count_within_radius' else ('نعم' if rounded else 'لا'))==answer
        d['unrounded_radius_matches_gold']=(str(n) if t=='count_within_radius' else ('نعم' if n else 'لا'))==answer
        if t=='count_within_radius':
            d['unique_name_count']=len({p['name'] for p in pool(query['category']) if distance(a,pos(p))<=radius})
            d['count_excluding_zero_distance']=sum(0<x<=radius for x in ds)
        # Fixed alternative radius constant: sensitivity audit, not convention tuning.
        d['count_radius_6371']=sum(x*6371/6371.0088<=radius for x in ds)
        predicted.add(str(n) if t=='count_within_radius' else ('نعم' if n else 'لا'))
    elif t == 'nearest_category':
        selected=nearest(pool(query['category']),a)
        predicted.update(p['name'] for p in selected)
        if len(selected)>1:
            d['first_visible_tied_candidate_matches_gold']=selected[0]['name']==answer
            d['tied_candidates']=[{'ordinal':p['ordinal'],'name':p['name']} for p in selected]
    elif t == 'nearest_of_two_categories':
        first=nearest(pool(query['category1']),a);second=nearest(pool(query['category2']),a)
        predicted.update(p['name'] for p in nearest(first+second,a))
    elif t == 'two_hop_nearest':
        first=nearest(pool(query['category1']),a)
        d['first_selected_names']=sorted({p['name'] for p in first})
        d['first_selected_count']=len(first)
        d['intermediate_reference_match']=row.get('intermediate_answer') in d['first_selected_names']
        d['intermediate_visible_matches']=sum(p['name']==row.get('intermediate_answer') for p in ps)
        finals=[]
        for p in first:finals.extend(nearest(pool(query['category2']),pos(p)))
        predicted.update(p['name'] for p in finals)
        d['category_metadata_match']=(CAT_CODES.get(query['category1'])==row.get('first_hop_category') and CAT_CODES.get(query['category2'])==row.get('second_hop_category'))
    elif t == 'spatial_multi_constraint':
        cat=pool(query['category'])
        center={'شمال':0,'شرق':90,'جنوب':180,'غرب':270}[query['direction']]
        # TRAIN hypothesis comparison: +/-35 degrees matches all 1,478 training
        # answers and counts, confirmed on 191 validation rows. Never test gold.
        dr=[p for p in cat if abs((bearing(a,pos(p))-center+180)%360-180)<=35]
        qual=[p for p in dr if distance(a,pos(p))<=query['radius']]
        d.update({'after_category':len(cat),'after_direction':len(dr),'after_radius':len(qual),
          'qualifying_count_metadata_match':len(qual)==row.get('num_qualifying_pois'),
          'query_metadata_match':CAT_CODES.get(query['category'])==row.get('target_category') and
           query['direction']==row.get('direction_constraint') and query['radius']==row.get('radius_constraint_km')})
        for width in (22.5,30,35,40,45,60,90):
            center={'شمال':0,'شرق':90,'جنوب':180,'غرب':270}.get(query['direction'])
            if center is None:continue
            subset=[p for p in cat if abs((bearing(a,pos(p))-center+180)%360-180)<=width and distance(a,pos(p))<=query['radius']]
            best=min(subset,key=lambda p:distance(a,pos(p))) if subset else None
            d[f'width_{width}_count_matches_gold']=len(subset)==row.get('num_qualifying_pois')
            d[f'width_{width}_answer_matches_gold']=best is not None and best['name']==answer
        d['unique_name_qualifying_count_matches_gold']=len({p['name'] for p in qual})==row.get('num_qualifying_pois')
        rounded=[p for p in dr if round(distance(a,pos(p)),3)<=query['radius']]
        d['rounded_radius_qualifying_count_matches_gold']=len(rounded)==row.get('num_qualifying_pois')
        d['evidence_list_length_matches_qualifying_metadata']=len(row.get('evidence_poi_ids',[]))==row.get('num_qualifying_pois')
        golds=[p for p in qual if p['name']==answer]
        d['gold_present_in_qualifying_set']=bool(golds)
        if golds:
            gd=min(distance(a,pos(p)) for p in golds)
            d['candidates_strictly_closer_than_gold']=sum(distance(a,pos(p))<gd for p in qual)
        predicted.update(p['name'] for p in nearest(qual,a))
        if not qual:return stop('F','zero_qualifying_candidates_for_entity_answer')
    d['predicted_answers']=sorted(predicted)
    if len(predicted)>1:return stop('B','exact_spatial_tie_changes_visible_answer')
    if not predicted:return stop('G','no_reconstructable_result')
    if answer not in predicted:
        return stop('F','visible_input_reconstruction_disagrees_with_gold')
    return stop('A','unique_visible_answer_agrees_with_gold')


def sha_file(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,sort_keys=True,allow_nan=False)+'\n',encoding='utf-8')


def write_csv(path, rows, fields):
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)


class Audit:
    def __init__(self, root, out):
        self.root,self.out=root,out
        self.counts=co.Counter();self.task=co.Counter();self.difficulty=co.Counter();self.cross=co.Counter()
        self.schema=co.defaultdict(lambda:co.defaultdict(lambda:{'present':0,'null':0,'empty_string':0,'empty_list':0,'types':co.Counter(),'element_types':co.Counter()}))
        self.schema_task=co.Counter();self.values=co.defaultdict(co.Counter)
        self.metrics=co.defaultdict(co.Counter);self.lengths=co.defaultdict(list)
        self.categories=co.Counter();self.unique_categories=co.Counter();self.templates=co.Counter();self.question_patterns=co.Counter()
        self.question_words=co.defaultdict(co.Counter);self.question_slots=co.Counter()
        self.indices=co.defaultdict(lambda:co.defaultdict(co.Counter));self.registry_sets={};self.registry_splits=co.defaultdict(co.Counter)
        self.ambiguity=co.Counter();self.recon=co.Counter();self.answers=co.defaultdict(co.Counter)
        self.radius=co.defaultdict(list);self.semantic=co.defaultdict(co.Counter);self.semantic_lengths=co.defaultdict(list)
        self.issue_counts=co.Counter();self.examples=co.defaultdict(list);self.probe_examples=[]

    def issue(self, code, split, task, rid, detail=''):
        self.issue_counts[(code,split,task)]+=1
        self.issue_writer.writerow({'split':split,'task_type':task,'id':rid,'issue':code,'detail':str(detail)[:500]})
        if len(self.examples[code])<5:self.examples[code].append({'split':split,'task_type':task,'id':rid,'detail':str(detail)[:500]})

    def idx(self, kind, value, split):
        if value is not None and value!='':self.indices[kind][value][split]+=1

    def consume(self, split, r, line_no):
        self.counts[split]+=1;n=self.counts[split];t=r.get('task_type','<missing>');rid=r.get('id')
        self.task[(split,t)]+=1;self.difficulty[(split,r.get('difficulty','<missing>'))]+=1
        self.cross[(split,t,r.get('difficulty','<missing>'))]+=1
        if not rid:self.issue('missing_id',split,t,f'line:{line_no}')
        if r.get('split')!=split:self.issue('split_field_mismatch',split,t,rid,r.get('split'))
        if set(r)!=set(EXPECTED_TYPES):self.issue('schema_key_set_mismatch',split,t,rid,sorted(set(r)^set(EXPECTED_TYPES)))
        if t not in TASKS:self.issue('unexpected_task',split,t,rid)
        for k,v in r.items():
            typ='null' if v is None else type(v).__name__;s=self.schema[k][split]
            s['present']+=1;s['null']+=v is None;s['types'][typ]+=1
            s['empty_string']+=isinstance(v,str) and v==''
            s['empty_list']+=isinstance(v,list) and not v
            self.schema_task[(k,split,t,'null' if v is None else 'nonnull')]+=1
            if isinstance(v,list):s['element_types'].update(type(z).__name__ for z in v)
            if k in EXPECTED_TYPES and typ not in EXPECTED_TYPES[k]:self.issue('unexpected_field_type',split,t,rid,k+':'+typ)
            # Never enumerate test gold values. Strings with huge vocabularies omitted.
            if not(split=='test' and k in GOLD) and k in {'country','language','difficulty','difficulty_score','recommended_reasoning','router_target_alpha','sources','answer_position','first_hop_category','second_hop_category','target_category','direction_constraint','radius_constraint_km','context_length_bin_chars'}:
                self.values[(split,k)][json.dumps(v,ensure_ascii=False,sort_keys=True)]+=1
        c=r.get('context');q=r.get('question')
        if not isinstance(c,str) or not isinstance(q,str):self.issue('invalid_text_input',split,t,rid);return
        p=parse_context(c);ps=p['candidates'];pq=parse_question(q)
        for k,x in [('context_chars',len(c)),('question_chars',len(q)),('article_chars',len(p['article'])),('registry_chars',len(p['registry'])),('candidate_count',len(ps))]:
            self.lengths[(split,k)].append(x)
        if len(c)!=r.get('context_chars'):self.issue('context_chars_mismatch',split,t,rid)
        if len(p['article'])!=r.get('article_chars'):self.issue('article_chars_mismatch',split,t,rid,{'observed':len(p['article']),'metadata':r.get('article_chars')})
        if len(ps)!=r.get('num_context_pois'):self.issue('candidate_count_mismatch',split,t,rid,{'parsed':len(ps),'metadata':r.get('num_context_pois')})
        if p['errors'] or p['bad']:self.issue('candidate_parse_or_coordinate_problem',split,t,rid,p['errors']+p['bad'])
        else:self.metrics[split]['candidate_parse_clean_records']+=1
        self.metrics[split]['candidate_blocks']+=p['blocks'];self.metrics[split]['parsed_candidates']+=len(ps)
        self.metrics[split]['malformed_candidate_blocks']+=len(p['bad'])
        self.metrics[split]['article_separated_records']+=p['article_separated']
        for err in p['errors']:self.metrics[split][err]+=1
        for z in ps:
            for err in z['errors']:self.metrics[split][err]+=1
        if any(z['name']!=z['name'].strip() for z in ps):self.issue('candidate_name_edge_whitespace',split,t,rid)
        multi=sum(z['multiline'] for z in ps);self.metrics[split]['multiline_candidate_occurrences']+=multi
        if multi:self.issue('multiline_candidate_names',split,t,rid,multi)
        names=co.Counter(z['name'] for z in ps);coordinates=co.Counter((z['lat'],z['lon']) for z in ps)
        rows=co.Counter((z['name'],z['category'],z['lat'],z['lon']) for z in ps)
        same=co.defaultdict(set)
        for z in ps:same[z['name']].add((z['lat'],z['lon']))
        properties={'duplicate_candidate_names':any(v>1 for v in names.values()),
                    'same_name_multiple_coordinates':any(len(v)>1 for v in same.values()),
                    'duplicate_candidate_coordinates':any(v>1 for v in coordinates.values()),
                    'duplicate_candidate_rows':any(v>1 for v in rows.values())}
        for key,flag in properties.items():
            if flag:self.issue(key,split,t,rid)
        self.metrics[split]['extra_duplicate_name_occurrences']+=sum(v-1 for v in names.values())
        self.metrics[split]['extra_duplicate_row_occurrences']+=sum(v-1 for v in rows.values())
        self.categories.update((split,z['category']) for z in ps)
        for z in ps:
            if z['category'] not in KNOWN_CATS:self.issue('unexpected_candidate_category',split,t,rid,z['category'])
        ch=digest(c);rh=digest(p['registry']);ah=digest(norm(p['article']))
        if rh not in self.registry_sets:
            fingerprints=frozenset(digest(json.dumps(z,ensure_ascii=False)) for z in sorted(rows))
            self.registry_sets[rh]=fingerprints
            self.unique_categories.update(z['category'] for z in ps)
        self.registry_splits[rh][split]+=1
        self.idx('id',rid,split);self.idx('wikidata_id',r.get('wikidata_id'),split)
        self.idx('anchor_place',r.get('anchor_place'),split)
        self.idx('anchor_coordinates',json.dumps([r.get('anchor_latitude'),r.get('anchor_longitude')]),split)
        if p['anchor']:
            self.idx('visible_anchor_coordinates',str(p['anchor'][1:]),split)
            if p['anchor'][0]!=r.get('anchor_place'):self.issue('anchor_name_metadata_mismatch',split,t,rid)
            try:
                dist=distance(p['anchor'][1:],(r['anchor_latitude'],r['anchor_longitude']))
                self.semantic_lengths[(split,'anchor_metadata_rounding_distance_m')].append(dist*1000)
                if not valid_coord(r['anchor_latitude'],r['anchor_longitude']):self.issue('invalid_anchor_metadata_coordinates',split,t,rid)
            except (TypeError,KeyError,ValueError):self.issue('invalid_anchor_metadata_coordinates',split,t,rid)
        self.idx('context_exact',ch,split);self.idx('context_whitespace_normalized',digest(norm(c)),split)
        self.idx('question_exact',digest(q),split);self.idx('question_context_exact',digest(q+'\0'+c),split)
        self.idx('registry_exact',rh,split)
        if p['article']:self.idx('article_normalized',ah,split)
        template=re.sub(r'«[^»]*»','<NAME>',q);template=re.sub(r'\d+(?:\.\d+)?','<NUM>',template)
        self.question_patterns[(split,t,template)]+=1
        self.question_words[split].update(re.findall(r'\w+',q))
        if pq:
            self.templates[(split,t,pq['operation'])]+=1
            if pq['operation']!=t:self.issue('question_task_mismatch',split,t,rid,pq['operation'])
            for key in ('category','category1','category2','direction','radius_text'):
                if key in pq:self.question_slots[(split,t,key,str(pq[key]))]+=1
            if 'radius' in pq:self.radius[(split,t)].append(pq['radius'])
        else:self.issue('unparsed_question_template',split,t,rid,q)
        status,matches=identity(pq,ps,p['anchor'][0] if p['anchor'] else None);self.ambiguity[(split,t,status)]+=1
        if pq and p['anchor'] and pq['operation']=='count_within_radius' and pq['anchor']==p['anchor'][0]:
            if any(z['name']==pq['anchor'] and (z['lat'],z['lon'])!=p['anchor'][1:] for z in ps):
                self.issue('count_origin_name_collides_with_context_anchor',split,t,rid)
        if pq and p['anchor']:
            references=[pq[k] for k in ('entity','entity1','entity2') if k in pq]
            if p['anchor'][0] in references and any(z['name']==p['anchor'][0] and (z['lat'],z['lon'])!=p['anchor'][1:] for z in ps):
                self.issue('target_name_collides_with_context_anchor',split,t,rid,{'question':q})
        self.metrics[split]['questions_with_latin_letters']+=bool(re.search('[A-Za-z]',q))
        if pq and p['anchor'] and pq['anchor']!=p['anchor'][0]:
            self.issue('query_origin_differs_from_context_anchor',split,t,rid,{'question':q,'context_anchor':p['anchor'][0]})
        if status in ('AMBIGUOUS','NOT FOUND','UNPARSED'):
            self.issue('named_identity_'+status.lower().replace(' ','_'),split,t,rid,{'question':q,'match_counts':[len(x) for x in matches]})
        # Explicit test firewall: all remaining analysis requires train/validation gold.
        if split=='test':return
        answer=r.get('answer');a=self.semantic[(split,t)]
        self.answers[(split,t)][answer]+=1
        if not isinstance(answer,str) or not answer.strip():self.issue('empty_or_invalid_answer',split,t,rid)
        if r.get('short_target')!=answer:self.issue('short_target_answer_mismatch',split,t,rid)
        expected_long=f"الإجابة: {answer}\nالتبرير المكاني الموثق: {r.get('verified_rationale')}"
        if r.get('long_target')!=expected_long:self.issue('long_target_format_mismatch',split,t,rid)
        if t=='cardinal_direction' and answer not in ('شمال','شرق','جنوب','غرب'):self.issue('unexpected_direction_answer',split,t,rid,answer)
        elif t=='within_radius_yes_no' and answer not in ('نعم','لا'):self.issue('unexpected_yesno_answer',split,t,rid,answer)
        elif t=='count_within_radius' and not re.fullmatch(r'[0-9]+',answer or ''):self.issue('unexpected_count_format',split,t,rid,answer)
        elif t not in ('cardinal_direction','within_radius_yes_no','count_within_radius'):
            a['entity_answer_examples']+=1;a['answer_name_exact_present']+=answer in names
            if names[answer]>1:self.issue('repeated_name_gold_answer',split,t,rid,answer)
            if answer not in names:
                aliases=[name for name in names if alias(name)==alias(answer)]
                self.issue('entity_answer_not_exactly_in_registry',split,t,rid,{'gold':answer,'normalized_matches':aliases[:3]})
        if isinstance(answer,str) and answer!=answer.strip():self.issue('answer_edge_whitespace',split,t,rid)
        if pq and t=='closer_of_two' and pq['entity1']!=pq['entity2'] and answer in (pq['entity1'],pq['entity2']):
            expected_position='first' if answer==pq['entity1'] else 'second'
            if r.get('answer_position')!=expected_position:self.issue('answer_position_mismatch',split,t,rid,{'expected':expected_position,'stored':r.get('answer_position')})
        probe=semantic_probe(split,r,p,pq);cls=probe['class'];self.recon[(split,t,cls)]+=1
        self.probe_writer.writerow({'split':split,'task_type':t,'id':rid,'class':cls,'reason':probe['reason']})
        if cls!='A':
            self.issue('reconstructability_'+cls,split,t,rid,{'reason':probe['reason'],'question':q,'gold':answer,'details':probe['details']})
            if len(self.probe_examples)<80:self.probe_examples.append({'split':split,'task_type':t,'id':rid,'question':q,'gold':answer,**probe})
        for flag in sorted(set(probe['flags'])):self.issue(flag,split,t,rid)
        d=probe['details']
        for key,val in d.items():
            if isinstance(val,bool):a[key+'_true' if val else key+'_false']+=1
            elif isinstance(val,(int,float)) and math.isfinite(val):self.semantic_lengths[(split,t+':'+key)].append(val)
            elif key=='nearest_gaps_km':self.semantic_lengths[(split,t+':nearest_gap_km')].extend(val)
        if d.get('intermediate_reference_match') is False:self.issue('intermediate_reconstruction_mismatch',split,t,rid,d)
        if d.get('qualifying_count_metadata_match') is False:self.issue('qualifying_count_metadata_mismatch',split,t,rid,d)
        if d.get('width_45_count_matches_gold') is False:self.issue('multi_constraint_four_sector_count_differs',split,t,rid)
        if d.get('width_45_answer_matches_gold') is False:self.issue('multi_constraint_four_sector_answer_differs',split,t,rid)
        if d.get('category_metadata_match') is False or d.get('query_metadata_match') is False:self.issue('query_slot_metadata_mismatch',split,t,rid,d)
        if 'intermediate_visible_matches' in d and d['intermediate_visible_matches']>1:self.issue('intermediate_name_repeated_but_geometry_resolves',split,t,rid,d['intermediate_visible_matches'])
        if d.get('unique_name_count') is not None and d['unique_name_count']!=d['qualifying_count']:self.issue('count_would_change_if_names_deduplicated',split,t,rid)
        if 'nearest_radius_gap_km' in d and d['nearest_radius_gap_km'] is not None and d['nearest_radius_gap_km']<=.0002:self.issue('radius_boundary_within_0_2m',split,t,rid)
        if 'count_radius_6371' in d and d['count_radius_6371']!=d['qualifying_count']:self.issue('earth_radius_constant_changes_count',split,t,rid)

    def integrity(self):
        result={}
        for kind,index in self.indices.items():
            pairs={}
            for x,y in itertools.combinations(SPLITS,2):
                shared=[(k,v) for k,v in index.items() if v[x] and v[y]]
                pairs[x+'__'+y]={'shared_distinct_values':len(shared),'affected_records':{x:sum(v[x] for k,v in shared),y:sum(v[y] for k,v in shared)},'example_keys':[str(k) for k,v in shared[:5]]}
            result[kind]={'distinct_values':len(index),'within_split_repeated_values':{s:sum(v[s]>1 for v in index.values()) for s in SPLITS},
              'within_split_extra_records':{s:sum(max(0,v[s]-1) for v in index.values()) for s in SPLITS},'cross_split':pairs}
        return result

    def near_registry(self):
        # Exact Jaccard comparisons for every cross-split registry pair able to
        # reach 0.9, using an inverted entity-fingerprint index and size pruning.
        # No embeddings; no full contexts retained. This is a context proxy.
        keys=list(self.registry_sets);sets=[self.registry_sets[k] for k in keys]
        sizes=[len(x) for x in sets];inverted=co.defaultdict(list)
        pair_counts=co.Counter();examples=[];affected=co.defaultdict(set)
        for i,items in enumerate(sets):
            overlaps=co.Counter()
            si=self.registry_splits[keys[i]]
            for item in items:
                for j in inverted[item]:
                    if min(sizes[i],sizes[j]) < .9*max(sizes[i],sizes[j]):continue
                    sj=self.registry_splits[keys[j]]
                    if not any(si[x] and sj[y] or si[y] and sj[x] for x,y in itertools.combinations(SPLITS,2)):continue
                    overlaps[j]+=1
            for j,intersection in sorted(overlaps.items()):
                jac=intersection/(sizes[i]+sizes[j]-intersection)
                if jac<.9:continue
                sj=self.registry_splits[keys[j]]
                for x,y in itertools.combinations(SPLITS,2):
                    if (si[x] and sj[y]) or (si[y] and sj[x]):
                        pair_counts[x+'__'+y]+=1
                        for k in (i,j):
                            for s in (x,y):
                                if self.registry_splits[keys[k]][s]:affected[(x+'__'+y,s)].add(keys[k])
                if len(examples)<10:examples.append({'registry1':keys[j],'registry2':keys[i],'jaccard':jac})
            for item in items:inverted[item].append(i)
        return {'method':'Complete cross-split pair search for candidate-row-set Jaccard >= 0.9 using inverted index; order/multiplicity ignored ONLY in this duplicate detector, not parsing/execution.',
                'limitation':'Near-context proxy over spatial registry; not a general natural-language semantic similarity search.',
                'unique_registries':len(keys),'pairs':dict(pair_counts),
                'affected_record_counts':{'|'.join(k):sum(self.registry_splits[h][k[1]] for h in vals) for k,vals in affected.items()},'examples':examples}

    def save(self, hashes):
        total=sum(self.counts.values());out=self.out
        def dist_rows(counter,labels):
            return [dict(zip(labels,key),count=n) for key,n in sorted(counter.items())]
        write_csv(out/'task_distribution.csv',dist_rows(self.task,['split','task_type']),['split','task_type','count'])
        write_csv(out/'difficulty_distribution.csv',dist_rows(self.difficulty,['split','difficulty']),['split','difficulty','count'])
        write_csv(out/'task_difficulty_distribution.csv',dist_rows(self.cross,['split','task_type','difficulty']),['split','task_type','difficulty','count'])
        for name,counter,status_key in [('ambiguity',self.ambiguity,'status'),('reconstructability',self.recon,'class')]:
            records=[]
            for (s,t,k),n in sorted(counter.items()):records.append({'split':s,'task_type':t,status_key:k,'count':n,'percent_of_task_split':round(100*n/self.task[(s,t)],6)})
            write_csv(out/(name+'_summary.csv'),records,['split','task_type',status_key,'count','percent_of_task_split'])
            aggregates={}
            for scope in ['overall']+list(SPLITS)+list(TASKS):
                filtered=[(s,t,k,n) for (s,t,k),n in counter.items() if scope=='overall' or s==scope or t==scope]
                den=sum(x[3] for x in filtered);cc=co.Counter()
                for s,t,k,n in filtered:cc[k]+=n
                aggregates[scope]={'n':den,'counts':dict(cc),'percentages':{k:round(100*n/den,6) for k,n in cc.items()} if den else {}}
            dump(out/(name+'_totals.json'),aggregates)
        schema={}
        for k,by in self.schema.items():
            presence=sum(z['present'] for z in by.values());null=sum(z['null'] for z in by.values());types=co.Counter()
            for z in by.values():types.update(z['types'])
            schema[k]={'presence_count':presence,'presence_percent':100*presence/total,'null_count':null,'null_percent_of_present':100*null/presence,
                       'empty_string_count':sum(z['empty_string'] for z in by.values()),'empty_list_count':sum(z['empty_list'] for z in by.values()),'types':types,'by_split':by}
        dump(out/'schema_summary.json',schema)
        write_csv(out/'schema_by_task.csv',dist_rows(self.schema_task,['field','split','task_type','state']),['field','split','task_type','state','count'])
        dump(out/'observed_metadata_values.json',{'|'.join(k):v for k,v in self.values.items()})
        dump(out/'length_summary.json',{'|'.join(k):stats(v) for k,v in self.lengths.items()} | {'all|'+key:stats([z for (s,k),v in self.lengths.items() if k==key for z in v]) for key in ('context_chars','question_chars','article_chars','registry_chars','candidate_count')})
        dump(out/'candidate_summary.json',{'by_split':self.metrics,'unique_registry_category_occurrences':self.unique_categories,
             'category_distribution_denominator':'Occurrences across records; repeated contexts count repeatedly. Unique registry counts also supplied.',
             'no_deduplication_applied':True})
        write_csv(out/'category_distribution.csv',dist_rows(self.categories,['split','category']),['split','category','count'])
        cats=sorted({c for s,c in self.categories});close=[]
        for a,b in itertools.combinations(cats,2):
            score=difflib.SequenceMatcher(None,alias(a),alias(b)).ratio()
            if score>=.65:close.append({'a':a,'b':b,'string_similarity':score})
        dump(out/'category_findings.json',{'categories':cats,'split_presence':{c:[s for s in SPLITS if self.categories[(s,c)]] for c in cats},
             'lexically_similar_pairs_not_merged':close,'normalization_proposals':'Preserve exact taxonomy; review restaurant/fast_food, bank/ATM, university/college and school/kindergarten semantically, never merge automatically.'})
        dump(out/'split_integrity_summary.json',self.integrity())
        dump(out/'near_duplicate_registry_summary.json',self.near_registry())
        write_csv(out/'question_templates.csv',dist_rows(self.question_patterns,['split','task_type','slot_masked_pattern']),['split','task_type','slot_masked_pattern','count'])
        write_csv(out/'question_slots.csv',dist_rows(self.question_slots,['split','task_type','slot','value']),['split','task_type','slot','value','count'])
        dump(out/'question_pattern_summary.json',{'regex_operation_matches':{'|'.join(k):v for k,v in self.templates.items()},
          'slot_masked_pattern_counts':{s:len({p for ss,t,p in self.question_patterns if ss==s}) for s in SPLITS},
          'word_types':{s:len(v) for s,v in self.question_words.items()},'word_tokens':{s:sum(v.values()) for s,v in self.question_words.items()},
          'note':'Rules identified from TRAIN only, frozen before validation/test input inspection; eight literal grammatical skeletons with slots. No classifier trained.'})
        dump(out/'answer_format_summary.json',{'scope':'TRAIN AND VALIDATION ONLY; TEST GOLD LOCKED',
          'by_split_task':{'|'.join(k):{'n':sum(v.values()),'unique_answers':len(v),'empty_answers':v.get('',0),
              'distribution':dict(v) if k[1] in ('cardinal_direction','within_radius_yes_no','count_within_radius') else None} for k,v in self.answers.items()}})
        dump(out/'radius_summary.json',{'|'.join(k):{'statistics':stats(v),'values':dict(co.Counter(map(str,v)))} for k,v in self.radius.items()})
        dump(out/'semantic_summary.json',{'scope':'TRAIN AND VALIDATION ONLY; metadata coordinate-rounding diagnostics may include TEST inputs',
            'fixed_hypotheses':{'distance':'Haversine sphere radius 6371.0088 km; visible context coordinates only',
             'direction':'classification: four 90-degree sectors, boundaries 45/135/225/315; multi-constraint filter: angular deviation <=35 degrees (TRAIN-derived, VALIDATION-confirmed)',
             'radius':'<=; compare < without tuning','taxonomy':'exact visible category strings',
             'ties':'retain all exact minima; report coordinate-rounding-sensitive gaps <=0.2m separately',
             'duplicates':'all rows preserved'},'counters':{'|'.join(k):v for k,v in self.semantic.items()},
             'numeric_statistics':{'|'.join(k):stats(v) for k,v in self.semantic_lengths.items()},'reconstructability_classes':RECON})
        dump(out/'representative_findings.json',{'issues':self.examples,'reconstruction_examples':self.probe_examples})
        write_csv(out/'issue_summary.csv',dist_rows(self.issue_counts,['issue','split','task_type']),['issue','split','task_type','count'])
        write_csv(out/'field_input_policy.csv',[{'field':k,'policy':v[0],'reason':v[1]} for k,v in sorted(FIELD_POLICY.items())],['field','policy','reason'])
        fixes=[]
        for (issue,s,t),n in sorted(self.issue_counts.items()):
            category,rule=FIX_RULES.get(issue,('G','Review audit finding before any derived-data transformation.'))
            fixes.append({'issue':issue,'split':s,'task_type':t,'affected_records':n,'remedy_class':category,'exact_rule':rule,'raw_untouched':True})
        write_csv(out/'recommended_remedies.csv',fixes,['issue','split','task_type','affected_records','remedy_class','exact_rule','raw_untouched'])
        summary={'valid_record_counts':dict(self.counts),'total':total,'raw_sha256':hashes,
           'test_gold_locked':True,'raw_unchanged':True,'issues':dict(co.Counter({code:sum(n for (c,s,t),n in self.issue_counts.items() if c==code) for code,s,t in self.issue_counts})),
           'task_imbalance_max_min':{s:max(self.task[(s,t)] for t in TASKS)/min(self.task[(s,t)] for t in TASKS) for s in SPLITS},
           'audit_version':1,'limitations':['No TEST answer/semantic evaluation.','No neural model or tokenizer loaded.',
            'Reconstructability is conditional on fixed visible-input semantic hypotheses, not proof of source-generator correctness.',
            'Raw coordinates may have more precision than visible context coordinates; no gold evidence IDs used for disambiguation.',
            'Near-duplicate audit uses exact normalized text and candidate-set similarity proxies.']}
        summary['integrity_checks']={code:sum(n for (c,s,t),n in self.issue_counts.items() if c==code) for code in
            ('malformed_json','blank_line','non_object_json','missing_id','split_field_mismatch','schema_key_set_mismatch',
             'unexpected_field_type','unexpected_task','invalid_text_input','candidate_count_mismatch','context_chars_mismatch')}
        summary['integrity_checks']['duplicate_id_extra_records']=sum(sum(v.values())-1 for v in self.indices['id'].values())
        dump(out/'dataset_audit_summary.json',summary)
        return summary

    def run(self):
        self.out.mkdir(parents=True,exist_ok=True)
        paths={s:self.root/(s+'.jsonl.gz') for s in SPLITS}
        hashes={s:sha_file(p) for s,p in paths.items()}
        with (self.out/'record_issues.csv').open('w',newline='',encoding='utf-8') as issues, (self.out/'reconstructability_records.csv').open('w',newline='',encoding='utf-8') as probes:
            self.issue_writer=csv.DictWriter(issues,fieldnames=['split','task_type','id','issue','detail']);self.issue_writer.writeheader()
            self.probe_writer=csv.DictWriter(probes,fieldnames=['split','task_type','id','class','reason']);self.probe_writer.writeheader()
            for s,path in paths.items():
                try:
                    with gzip.open(path,'rt',encoding='utf-8') as f:
                        for line_no,line in enumerate(f,1):
                            if not line.strip():self.issue('blank_line',s,'<unknown>',f'line:{line_no}');continue
                            try:r=json.loads(line)
                            except json.JSONDecodeError as e:self.issue('malformed_json',s,'<unknown>',f'line:{line_no}',e.msg);continue
                            if not isinstance(r,dict):self.issue('non_object_json',s,'<unknown>',f'line:{line_no}');continue
                            self.consume(s,r,line_no)
                    print(f'{s}: {self.counts[s]} records audited',flush=True)
                except (OSError,EOFError,UnicodeError) as e:
                    raise RuntimeError(f'Cannot stream {s}: {type(e).__name__}') from e
        if hashes!={s:sha_file(p) for s,p in paths.items()}:raise RuntimeError('Raw file checksum changed during audit')
        summary=self.save(hashes)
        print(json.dumps({'counts':summary['valid_record_counts'],'total':summary['total'],'raw_unchanged':True,'test_gold_locked':True},indent=2))


class AuditTests(unittest.TestCase):
    def context(self, rows):
        return 'المكان المرجعي هو «أ»\nخط العرض: 24.0\nخط الطول: 46.0\n'+ART_MARK+'\nنص\n'+REG_MARK+'\n'+rows
    def test_multiline_and_duplicates(self):
        row='الاسم: مدرسة\nمتعددة الأسطر؛ النوع: مدرسة؛ خط العرض: 24.1؛ خط الطول: 46.1.'
        p=parse_context(self.context('1. '+row+'\n2. '+row))
        self.assertEqual(len(p['candidates']),2);self.assertTrue(all(x['multiline'] for x in p['candidates']));self.assertFalse(p['bad'])
    def test_invalid_coordinates(self):
        p=parse_context(self.context('1. الاسم: س؛ النوع: مدرسة؛ خط العرض: 91؛ خط الطول: 181.'))
        self.assertIn('invalid_candidate_coordinates',p['candidates'][0]['errors'])
    def test_name_trailing_whitespace_is_preserved(self):
        p=parse_context(self.context('1. الاسم: latifa hospital ؛ النوع: مقهى؛ خط العرض: 24؛ خط الطول: 46.'))
        self.assertEqual(p['candidates'][0]['name'],'latifa hospital ')
    def test_query_origin_can_be_candidate(self):
        q={'operation':'count_within_radius','anchor':'س','category':'مدرسة','radius':1.5}
        ps=[{'name':'س','lat':24,'lon':46}]
        self.assertEqual(identity(q,ps,'أ')[0],'UNIQUE')
    def test_missing_coordinate(self):
        p=parse_context(self.context('1. الاسم: س؛ النوع: مدرسة؛ خط العرض: ؛ خط الطول: 46.'))
        self.assertIn('missing_latitude',p['candidates'][0]['errors'])
    def test_malformed_registry(self):
        p=parse_context(self.context('1. الاسم: broken'))
        self.assertEqual(len(p['bad']),1)
    def test_test_gold_firewall(self):
        class Forbidden(dict):
            def get(self,*args):raise AssertionError('test fields accessed')
        self.assertIsNone(semantic_probe('test',Forbidden(),None,None))
    def test_direction(self):
        self.assertEqual(direction(235.7),'غرب');self.assertEqual(direction(235.7,8),'جنوب غرب')
    def test_question(self):
        q=parse_question('كم عدد المعالم من نوع «مطعم» الواقعة ضمن مسافة 1.5 كم تقريبًا من «أ» وفق الإحداثيات المتاحة؟')
        self.assertEqual(q['radius'],1.5);self.assertEqual(q['operation'],'count_within_radius')
    def test_statistics(self):
        self.assertEqual(stats([1,2,3,4])['p50'],2.5)
    def test_gzip_streaming(self):
        with tempfile.TemporaryDirectory(prefix='aasr_audit_test_') as directory:
            p=Path(directory)/'tiny.jsonl.gz'
            with gzip.open(p,'wt',encoding='utf-8') as f:
                for i in range(3):f.write(json.dumps({'id':str(i),'question':'سؤال'},ensure_ascii=False)+'\n')
            with gzip.open(p,'rt',encoding='utf-8') as f:
                self.assertEqual([json.loads(line)['id'] for line in f],['0','1','2'])
    def test_test_gold_value_changes_cannot_change_structural_results(self):
        context=self.context('1. الاسم: س؛ النوع: مدرسة؛ خط العرض: 24.1؛ خط الطول: 46.1.')
        base={'id':'fixture','split':'test','task_type':'nearest_category','difficulty':'easy',
              'context':context,'question':'ما أقرب مدرسة إلى «أ» من بين المعالم الواردة في السياق؟',
              'anchor_place':'أ','anchor_latitude':24.0,'anchor_longitude':46.0,
              'num_context_pois':1,'context_chars':len(context),'article_chars':2}
        snapshots=[]
        for marker in ('hidden-a','hidden-b'):
            row=dict(base,**{k:marker for k in GOLD})
            audit=Audit(Path('.'),Path('.'));stream=io.StringIO()
            audit.issue_writer=csv.DictWriter(stream,fieldnames=['split','task_type','id','issue','detail'])
            audit.consume('test',row,1)
            snapshots.append((dict(audit.schema),dict(audit.values),dict(audit.metrics),dict(audit.ambiguity),
                              dict(audit.recon),dict(audit.answers),dict(audit.semantic),stream.getvalue()))
        self.assertEqual(snapshots[0],snapshots[1])
        self.assertEqual(snapshots[0][4],{})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    base=Path(__file__).resolve().parents[1]
    parser.add_argument('--raw-dir',type=Path,default=base/'data/raw')
    parser.add_argument('--output-dir',type=Path,default=base/'experiments/E5_lightweight_adaptive/data_audit')
    parser.add_argument('--self-test',action='store_true')
    a=parser.parse_args()
    if a.self_test:
        result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(AuditTests))
        raise SystemExit(0 if result.wasSuccessful() else 1)
    Audit(a.raw_dir,a.output_dir).run()


if __name__=='__main__':main()
