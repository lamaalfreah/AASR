"""Eight audited templates; no record metadata or learned rules."""
from __future__ import annotations
import re

PATTERNS = [
    ("nearest_category", r"ما أقرب (?P<category>.+?) إلى «(?P<anchor>.+?)» من بين المعالم الواردة في السياق؟"),
    ("cardinal_direction", r"في أي اتجاه يقع «(?P<entity>.+?)» بالنسبة إلى «(?P<anchor>.+?)»؟"),
    ("within_radius_yes_no", r"هل يوجد (?P<category>.+?) ضمن مسافة (?P<radius>[\d.]+) كم تقريبًا من «(?P<anchor>.+?)» من بين المعالم الواردة في السياق؟"),
    ("closer_of_two", r"أي الموقعين أقرب إلى «(?P<anchor>.+?)»: «(?P<entity1>.+?)» أم «(?P<entity2>.+?)»؟"),
    ("count_within_radius", r"كم عدد المعالم من نوع «(?P<category>.+?)» الواقعة ضمن مسافة (?P<radius>[\d.]+) كم تقريبًا من «(?P<anchor>.+?)» وفق الإحداثيات المتاحة؟"),
    ("nearest_of_two_categories", r"أيهما أقرب إلى «(?P<anchor>.+?)»: أقرب (?P<category1>.+?) أم أقرب (?P<category2>.+?)؟ اذكر اسم المعلم الأقرب\."),
    ("two_hop_nearest", r"انطلاقًا من «(?P<anchor>.+?)»، حدد أولًا أقرب (?P<category1>.+?)، ثم استخدم هذا الموقع كنقطة مرجعية جديدة لتحديد أقرب (?P<category2>.+?)\. ما اسم المعلم النهائي؟"),
    ("spatial_multi_constraint", r"ما أقرب (?P<category>.+?) يقع في اتجاه (?P<direction>.+?) من «(?P<anchor>.+?)» وضمن مسافة (?P<radius>[\d.]+) كم تقريبًا؟"),
]
PATTERNS = [(t, re.compile(p, re.S)) for t, p in PATTERNS]

def parse_question_fields(q):
    matches = [(t, m) for t, p in PATTERNS if (m := p.fullmatch(q))]
    if len(matches) != 1: return None
    t, m = matches[0]
    d = dict(m.groupdict(), operation=t)
    if 'radius' in d: d['radius_text'], d['radius'] = d['radius'], float(d['radius'])
    return d



from .context_parser import parse_context
from .schema import (CATEGORIES, DIRECTIONS, OPERATIONS, EntityReference,
                     ParsedContext, QueryError, StructuredQuery)
import math


def validate_query(query):
    """Validate replaceable parser outputs before any geometric execution."""
    if not isinstance(query, StructuredQuery):
        raise QueryError('invalid_query', 'expected_structured_query')
    if query.operation not in OPERATIONS:
        raise QueryError('unsupported', 'unsupported_operation')
    if not isinstance(query.origin, EntityReference) or not query.origin.name:
        raise QueryError('invalid_query', 'missing_origin')
    refs = (query.origin,) + query.targets
    if any(not isinstance(r, EntityReference) or not isinstance(r.name, str)
           or not r.name or r.source not in ('context_anchor', 'candidate', 'unspecified')
           for r in refs):
        raise QueryError('invalid_query', 'invalid_entity_reference')
    t = query.operation
    target_count = 1 if t == 'cardinal_direction' else 2 if t == 'closer_of_two' else 0
    category_count = 2 if t == 'nearest_of_two_categories' else (
        1 if t in ('nearest_category', 'within_radius_yes_no',
                   'count_within_radius', 'spatial_multi_constraint') else 0)
    if len(query.targets) != target_count or len(query.categories) != category_count:
        raise QueryError('invalid_query', 'wrong_argument_count')
    cats = query.categories
    if t == 'two_hop_nearest':
        cats += (query.first_hop_category, query.second_hop_category)
    elif query.first_hop_category is not None or query.second_hop_category is not None:
        raise QueryError('invalid_query', 'unexpected_hop_arguments')
    if any(c not in CATEGORIES for c in cats):
        raise QueryError('unsupported', 'unknown_category')
    needs_radius = t in ('within_radius_yes_no', 'count_within_radius', 'spatial_multi_constraint')
    if needs_radius:
        r = query.radius_km
        if isinstance(r, bool) or not isinstance(r, (float, int)) or not math.isfinite(r) or r < 0:
            raise QueryError('invalid_query', 'invalid_radius')
    elif query.radius_km is not None:
        raise QueryError('invalid_query', 'unexpected_radius')
    if t == 'spatial_multi_constraint':
        if query.direction not in DIRECTIONS:
            raise QueryError('unsupported', 'unknown_direction')
    elif query.direction is not None:
        raise QueryError('invalid_query', 'unexpected_direction')


def parse(question: str, context: str | ParsedContext) -> StructuredQuery:
    """Parse current dataset templates from public input; raise QueryError on failure."""
    if not isinstance(question, str):
        raise QueryError('invalid_query', 'question_must_be_text')
    ctx = parse_context(context) if isinstance(context, str) else context
    if not isinstance(ctx, ParsedContext) or ctx.errors or ctx.anchor is None:
        raise QueryError('invalid_query', 'invalid_context', 'context_parser')
    try:
        fields = parse_question_fields(question)
    except ValueError:
        raise QueryError('invalid_query', 'invalid_radius') from None
    if fields is None:
        raise QueryError('unsupported', 'unmatched_template')
    operation = fields['operation']
    # Ordinary templates explicitly use the context reference. Count wording
    # does not privilege it over a same-named candidate at another location.
    source = 'context_anchor' if fields['anchor'] == ctx.anchor.name else 'candidate'
    if operation == 'count_within_radius' and source == 'context_anchor':
        source = 'unspecified'
    query = StructuredQuery(
        operation, EntityReference(fields['anchor'], source),
        targets=tuple(EntityReference(fields[k]) for k in ('entity', 'entity1', 'entity2') if k in fields),
        categories=tuple(fields[k] for k in ('category', 'category1', 'category2')
                         if k in fields and operation != 'two_hop_nearest'),
        radius_km=fields.get('radius'), direction=fields.get('direction'),
        first_hop_category=fields.get('category1') if operation == 'two_hop_nearest' else None,
        second_hop_category=fields.get('category2') if operation == 'two_hop_nearest' else None,
    )
    validate_query(query)
    return query
