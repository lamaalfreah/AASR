"""Place grounding and bounded geographic context, not task classification."""
import re
from functools import lru_cache
from django.conf import settings
from language.neural_short import category_mentions
from spatial.schema import Location,ParsedContext,StructuredQuery,CATEGORIES
from spatial.geometry import distance,valid_coord
from spatial.identity import resolve
from spatial import execute
from .geo.poi_index import index,locations,normalize

class Clarification(ValueError):pass

@lru_cache(maxsize=1)
def aliases():
    result={}
    generic={normalize(c) for c in CATEGORIES}
    for row in index():
        for alias in row['aliases']:
            key=normalize(alias)
            if len(key)<4 or key in generic:continue
            if row['id'] not in [r['id'] for r in result.setdefault(key,[])]:result[key].append(row)
    reference=dict(id='configured:anchor',name=settings.AASR_DEFAULT_ANCHOR_NAME,
                   lat=settings.AASR_DEFAULT_LAT,lon=settings.AASR_DEFAULT_LON,category=None)
    result.setdefault(normalize(reference['name']),[]).append(reference)
    return result


@lru_cache(maxsize=1)
def patterns():
    return [(re.compile(r'(?<!\w)'+re.escape(alias)+r'(?!\w)'),rows) for alias,rows in aliases().items()]


def prepare(question,current_location=None):
    text=normalize(question)
    if current_location is not None and (not isinstance(current_location,dict) or set(current_location)!={'lat','lon'} or any(type(v) not in (int,float) for v in current_location.values()) or not valid_coord(current_location['lat'],current_location['lon'])):
        raise Clarification('إحداثيات الموقع غير صالحة.')
    matches=[]
    for pattern,rows in patterns():
        for match in pattern.finditer(text):
            matches.append((match.start(),match.end(),rows))
    kept=[]
    for item in sorted(matches,key=lambda m:(-(m[1]-m[0]),m[0])):
        if not any(item[0]<m[1] and m[0]<item[1] for m in kept):kept.append(item)
    kept.sort()
    if any(len(rows)>1 for _,_,rows in kept):raise Clarification('يوجد أكثر من مكان بهذا الاسم. اكتب اسمًا أكثر تحديدًا.')
    known=aliases()
    for quoted in re.findall(r'«([^»]+)»|"([^"]+)"',question):
        name=next(p for p in quoted if p)
        if normalize(name) not in known and normalize(name) not in {normalize(c) for c in CATEGORIES}:
            raise Clarification('لم أجد هذا الاسم في بيانات المنطقة المحلية. جرّب الاسم المسجل أو موقعًا آخر.')
    named=[]
    for _,_,rows in kept:
        row=rows[0]
        named.append(Location(row['id'],row['name'],row['lat'],row['lon'],row['category']))
    origin=None
    # Ground the reference using relational prepositions, independently of intent.
    for (start,_,_),point in zip(kept,named):
        if re.search(r'(?:بالنسبة\s+(?:الي|ل)|من|الي|حول|عند|عن|قرب)\s*[«"]?\s*$',text[:start]):origin=point
    if origin is None and len(named)==1:origin=named[0]
    if origin is None and len(named)>1:raise Clarification('حدّد الموقع المرجعي بعبارة مثل «بالنسبة إلى …».')
    if origin is None and current_location is not None:
        origin=Location('user:current','موقعي الحالي',current_location['lat'],current_location['lon'])
    if origin is None:
        if re.search(r'(?<!\w)(حولي|مني|موقعي|عندي|لي)(?!\w)',text):raise Clarification('حدّد موقعًا مرجعيًا بالاسم أو أرسل الموقع الحالي.')
        if re.search(r'(?:الي|من|حول|عند|بالنسبة|في)\s+[^\d]+$',text) and settings.AASR_DEFAULT_ANCHOR_NAME not in question and normalize(settings.AASR_DEFAULT_AREA_NAME) not in text:
            raise Clarification('لم أتمكن من تحديد المرجع في السجل المحلي. استخدم اسمًا معروفًا داخل منطقة التغطية.')
        origin=Location('configured:anchor',settings.AASR_DEFAULT_ANCHOR_NAME,settings.AASR_DEFAULT_LAT,settings.AASR_DEFAULT_LON)
    # Canonical names are quoted for the trained tokenizer; no task words are rewritten.
    for (start,end,_),point in reversed(list(zip(kept,named))):text=text[:start]+'«'+point.name+'»'+text[end:]
    text=text.replace('««','«').replace('»»','»')
    if origin.name not in text:text=text.rstrip('؟?!. ')+' من «'+origin.name+'»؟'
    requested={c for _,_,c in category_mentions(text)}
    pool=[p for p in locations() if (not requested or p.category in requested) and p.identity!=origin.identity]
    pool.sort(key=lambda p:(distance(origin.coordinates,p.coordinates),p.identity))
    visible={p.identity:p for p in named if p.identity!=origin.identity}
    # Balanced context for two categories, not arbitrary first 80 records.
    categories=sorted(requested) or sorted(CATEGORIES)
    for category in categories:
        for p in [p for p in pool if p.category==category][:max(1,settings.AASR_PARSER_CANDIDATES//len(categories))]:visible[p.identity]=p
    return text,ParsedContext(origin,tuple(visible.values()))


def execution_context(query,context):
    all_points={p.identity:p for p in locations()+context.candidates};all_points[context.anchor.identity]=context.anchor
    all_context=ParsedContext(context.anchor,tuple(p for p in all_points.values() if p.identity!=context.anchor.identity))
    resolutions=[resolve(ref,all_context) for ref in (query.origin,)+query.targets]
    if any(r.status!='UNIQUE' for r in resolutions):raise Clarification('لم يتم تحديد هوية المواقع بشكل فريد. وضّح الأسماء.')
    origin=resolutions[0].matches[0]
    radius=query.radius_km if query.radius_km is not None else settings.AASR_SEARCH_KM
    if radius>settings.ASAR_GEO_MAX_RADIUS_M/1000:raise Clarification('نطاق البحث أكبر من حد البيانات المحلية. اختر نطاقًا أصغر.')
    required={p.identity:p for r in resolutions for p in r.matches}
    categories=set(query.categories)|{query.first_hop_category,query.second_hop_category}
    candidates=[p for p in locations() if p.category in categories and p.identity!=context.anchor.identity and distance(origin.coordinates,p.coordinates)<=radius]
    if query.operation=='two_hop_nearest':
        first_query=StructuredQuery('nearest_category',query.origin,categories=(query.first_hop_category,))
        first_context=ParsedContext(context.anchor,tuple({p.identity:p for p in candidates+list(required.values()) if p.identity!=context.anchor.identity}.values()))
        first=execute(first_query,first_context)
        if first.status!='success':raise Clarification('لم يمكن تحديد الخطوة الأولى بشكل فريد ضمن النطاق المحلي.')
        hop=first.trace['selected_candidates'][0]
        candidates=[p for p in candidates if p.category==query.first_hop_category]
        candidates += [p for p in locations() if p.category==query.second_hop_category and distance((hop['latitude'],hop['longitude']),p.coordinates)<=radius]
    required.update({p.identity:p for p in candidates})
    required.pop(context.anchor.identity,None)
    return ParsedContext(context.anchor,tuple(required.values()))
