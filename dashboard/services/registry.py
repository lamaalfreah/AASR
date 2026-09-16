"""Bounded deterministic retrieval from a spatial-only local registry."""
import json
from functools import lru_cache
from collections import Counter
from django.conf import settings
from spatial.schema import CATEGORIES,Location,ParsedContext,StructuredQuery
from spatial.geometry import valid_coord,distance
from spatial.identity import resolve
from .current_location import bind_location

LIMIT=200

class RetrievalIssue(ValueError):
    def __init__(self,status,message):self.status=status;self.message=message

@lru_cache(maxsize=4)
def _load(path,modified):
    data=json.loads(path.read_text());rows=[];ids=set()
    for row in data['entities']:
        if set(row)!={'id','name_ar','category','latitude','longitude','source'}:raise ValueError('Invalid registry schema')
        if not row['id'] or row['id'] in ids:raise ValueError('Duplicate registry identity')
        if not isinstance(row['name_ar'],str) or not row['name_ar'].strip() or row['category'] not in CATEGORIES:raise ValueError('Invalid registry entity')
        if any(type(row[k]) not in (int,float) for k in ('latitude','longitude')) or not valid_coord(row['latitude'],row['longitude']):raise ValueError('Invalid registry coordinates')
        ids.add(row['id']);rows.append(Location(row['id'],row['name_ar'],row['latitude'],row['longitude'],row['category']))
    if not rows:raise ValueError('Empty registry')
    return tuple(rows)

def entities():
    path=settings.AASR_REGISTRY_PATH
    return _load(path,path.stat().st_mtime_ns)

def default_anchor():return next(p for p in entities() if p.category=='مستشفى')

def unique(points):return tuple({p.identity:p for p in points}.values())

def ranked(points,anchor):return sorted(points,key=lambda p:(distance(anchor.coordinates,p.coordinates),p.identity))

def preview_catalog():
    anchor=default_anchor()
    def row(p):return dict(name=p.name,category=p.category,lat=p.latitude,lng=p.longitude)
    return dict(name=f'سجل مكاني محلي مشتق من OpenStreetMap — {len(entities())} موقعًا؛ ليس مصدرًا حيًا',
                anchor=row(anchor),candidates=[row(p) for p in ranked(entities(),anchor) if p.identity!=anchor.identity][:12])

def initial_context(question,current_location):
    anchor=default_anchor()
    named=[p for p in entities() if p.name.strip() not in CATEGORIES and p.name.strip() in question]
    counts=Counter(p.name for p in named)
    if any(n>1 for n in counts.values()):
        raise RetrievalIssue('ambiguous','يوجد أكثر من موقع بهذا الاسم في السجل. حدّد موقعًا مميزًا.')
    # One explicit name can supply the reference. All named entities are retained
    # for the frozen parser to determine their actual roles.
    if len(named)==1:anchor=named[0]
    base,_=bind_location(question,ParsedContext(anchor,()),current_location)
    ordered=ranked(entities(),base.anchor)
    # A small quota per category supports comparisons/two-hop without guessing
    # the operation before parsing. Named entities are never lost to the quota.
    quota=Counter();subset=list(named)
    for p in ordered:
        if quota[p.category]<5:subset.append(p);quota[p.category]+=1
    subset=unique(subset)
    if len(subset)>LIMIT:raise RetrievalIssue('needs_clarification','حدّد عددًا أقل من المواقع في السؤال.')
    return ParsedContext(anchor,tuple(p for p in subset if p.identity!=anchor.identity))

def execution_context(query,context):
    all_points=unique((context.anchor,)+entities())
    # Avoid turning a known anchor into a second copy with a scoped identity.
    pool=ParsedContext(context.anchor,tuple(p for p in all_points if p.identity!=context.anchor.identity))
    origin=resolve(query.origin,pool)
    if origin.status!='UNIQUE':
        raise RetrievalIssue('ambiguous' if origin.status=='AMBIGUOUS' else 'not_found','لم يمكن تحديد الموقع المرجعي بشكل فريد.')
    anchor=origin.matches[0]
    required=[anchor]
    for ref in query.targets:
        found=resolve(ref,pool)
        if found.status!='UNIQUE':raise RetrievalIssue('ambiguous' if found.status=='AMBIGUOUS' else 'not_found','وضّح أسماء المواقع المطلوبة.')
        required.extend(found.matches)
    categories=set(query.categories)|{query.first_hop_category,query.second_hop_category}
    eligible=[p for p in entities() if p.category in categories and p.identity!=anchor.identity]
    if query.radius_km is not None:
        eligible=[p for p in eligible if distance(anchor.coordinates,p.coordinates)<=query.radius_km]
    elif query.operation in ('nearest_category','nearest_of_two_categories'):
        selected=[]
        for category in query.categories:
            ordered=ranked([p for p in eligible if p.category==category],anchor)
            if len(ordered)>40:
                cutoff=distance(anchor.coordinates,ordered[39].coordinates)
                ordered=[p for p in ordered if distance(anchor.coordinates,p.coordinates)<=cutoff]
            selected.extend(ordered)
        eligible=selected
    # Two-hop keeps both complete categories: no pruning around the wrong origin.
    candidates=unique(required+eligible)
    candidates=tuple(p for p in candidates if p.identity!=context.anchor.identity)
    if len(candidates)>LIMIT:
        raise RetrievalIssue('needs_clarification','نطاق البحث أكبر من حد العرض. حدّد نطاقًا أصغر؛ لم تُحسب نتيجة جزئية.')
    return ParsedContext(context.anchor,candidates)
