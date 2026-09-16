"""Deterministic execution of inference-only queries against visible candidates."""
from dataclasses import asdict
from .geometry import bearing, direction, distance, valid_coord
from .identity import resolve
from .query_parser import validate_query
from .schema import ExecutionResult, ParsedContext, QueryError, StructuredQuery, TypedAnswer


def execute(query: StructuredQuery, context: ParsedContext) -> ExecutionResult:
    """Context bundles the visible anchor and ordered candidates; no dataset row accepted."""
    trace = {'parsed_arguments': asdict(query) if isinstance(query, StructuredQuery) else None,
             'resolved_origin': None, 'identity_resolutions': [], 'steps': [],
             'selected_candidates': [], 'diagnostics': []}
    operation = query.operation if isinstance(query, StructuredQuery) else None
    trace['operation'] = operation

    def result(status, reason=None, answer=None, stage='executor'):
        trace.update(execution_status=status, final_answer=asdict(answer) if answer else None)
        return ExecutionResult(status, operation, answer, reason, stage, trace)

    try:
        validate_query(query)
    except QueryError as e:
        return result(e.status, e.reason, stage='query_validation')
    if not isinstance(context, ParsedContext) or context.errors or context.anchor is None:
        return result('invalid_query', 'invalid_context', stage='context_parser')
    all_locations = (context.anchor,) + context.candidates
    if any(not valid_coord(*p.coordinates) for p in all_locations):
        return result('invalid_query', 'invalid_coordinates', stage='context_parser')
    if len({p.identity for p in all_locations}) != len(all_locations):
        return result('invalid_query', 'duplicate_scoped_identity', stage='context_parser')
    resolutions = [resolve(ref, context) for ref in (query.origin,) + query.targets]
    trace['identity_resolutions'] = [dict(asdict(r), candidate_count=r.candidate_count) for r in resolutions]
    if any(r.status == 'NOT_FOUND' for r in resolutions):
        return result('not_found', 'referenced_entity_not_found', stage='identity')
    if any(r.status == 'AMBIGUOUS' for r in resolutions):
        return result('ambiguous', 'referenced_entity_ambiguous', stage='identity')
    origin = resolutions[0].matches[0]
    trace['resolved_origin'] = asdict(origin)
    targets = [r.matches[0] for r in resolutions[1:]]
    trace['radius_km'], trace['direction'] = query.radius_km, query.direction
    t = query.operation

    def pool(category):
        return [p for p in context.candidates if p.category == category]

    def measure(candidates, reference):
        return [(p, distance(reference.coordinates, p.coordinates)) for p in candidates]

    def nearest(candidates, reference, name):
        values = measure(candidates, reference)
        if not values:
            trace['steps'].append({'step': name, 'candidate_count': 0, 'selected': []})
            return []
        minimum = min(d for p, d in values)
        selected = [p for p, d in values if d == minimum]
        distances = sorted(d for p, d in values)
        trace['steps'].append({'step': name, 'reference': asdict(reference),
                               'candidate_count': len(values), 'distance_km': minimum,
                               'distances_km': {p.identity: d for p, d in values},
                               'selected': [asdict(p) for p in selected]})
        if len(selected) > 1:
            trace['diagnostics'].append('exact_nearest_tie')
        if len(distances) > 1 and distances[1] - distances[0] <= .0002:
            trace['diagnostics'].append('nearest_gap_within_0_2m')
        return selected

    def entity_result(selected):
        trace['selected_candidates'] = [asdict(p) for p in selected]
        if not selected:
            return result('not_found', 'no_qualifying_candidate')
        names = {p.name for p in selected}
        if len(names) != 1:
            return result('ambiguous', 'nearest_tie_distinct_answers')
        # Same visible answer may have multiple minimizing identities. Retain
        # all of them; successful text answer does not claim a unique entity.
        trace['selected_identity_unique'] = len(selected) == 1
        name = next(iter(names))
        return result('success', answer=TypedAnswer('entity', name, name))

    if t == 'cardinal_direction':
        p = targets[0]
        d = distance(origin.coordinates, p.coordinates)
        if d == 0:
            return result('invalid_query', 'bearing_undefined_at_same_location')
        b = bearing(origin.coordinates, p.coordinates)
        label = direction(b)
        trace['selected_candidates'] = [asdict(p)]
        trace['steps'].append({'step': 'direction', 'distance_km': d, 'bearing_degrees': b,
                               'sectors': 4, 'label': label})
        return result('success', answer=TypedAnswer('direction', label, label))
    if t == 'closer_of_two':
        return entity_result(nearest(targets, origin, 'compare_targets'))
    if t in ('within_radius_yes_no', 'count_within_radius'):
        measured = measure(pool(query.categories[0]), origin)
        qualifying = [(p, d) for p, d in measured if d <= query.radius_km]
        trace['selected_candidates'] = [asdict(p) for p, d in qualifying]
        trace['steps'].append({'step': 'radius_filter', 'category_count': len(measured),
            'qualifying_count': len(qualifying), 'radius_km': query.radius_km,
            'comparison': '<=', 'distances_km': {p.identity: d for p, d in measured}})
        if any(abs(d-query.radius_km) <= .0002 for p, d in measured):
            trace['diagnostics'].append('radius_boundary_within_0_2m')
        if t == 'within_radius_yes_no':
            return result('success', answer=TypedAnswer('boolean', bool(qualifying), 'نعم' if qualifying else 'لا'))
        return result('success', answer=TypedAnswer('integer', len(qualifying), str(len(qualifying))))
    if t == 'nearest_category':
        return entity_result(nearest(pool(query.categories[0]), origin, 'nearest_category'))
    if t == 'nearest_of_two_categories':
        groups = [nearest(pool(c), origin, f'category_{i+1}') for i, c in enumerate(query.categories)]
        if any(not g for g in groups):
            return result('not_found', 'required_category_missing')
        # Keep context order and each original row once, even if categories coincide.
        ids = {p.identity for g in groups for p in g}
        return entity_result(nearest([p for p in context.candidates if p.identity in ids], origin, 'compare_categories'))
    if t == 'two_hop_nearest':
        first = nearest(pool(query.first_hop_category), origin, 'first_hop')
        trace['first_hop_result'] = [asdict(p) for p in first]
        if not first:
            return result('not_found', 'first_hop_category_missing')
        if len(first) != 1:
            return result('ambiguous', 'first_hop_identity_tie')
        return entity_result(nearest(pool(query.second_hop_category), first[0], 'second_hop'))
    if t == 'spatial_multi_constraint':
        candidates = pool(query.categories[0])
        center = {'شمال': 0, 'شرق': 90, 'جنوب': 180, 'غرب': 270}[query.direction]
        measured = [(p, distance(origin.coordinates, p.coordinates), bearing(origin.coordinates, p.coordinates))
                    for p in candidates]
        # Bearing at identical coordinates is undefined. Do not assign it north.
        if any(d == 0 for p, d, b in measured):
            return result('invalid_query', 'direction_constraint_at_same_location')
        directional = [(p, d, b) for p, d, b in measured if abs((b-center+180) % 360-180) <= 35]
        qualifying = [p for p, d, b in directional if d <= query.radius_km]
        trace['steps'].append({'step': 'multi_constraint_filter', 'after_category': len(candidates),
            'after_direction': len(directional), 'after_radius': len(qualifying),
            'direction': query.direction, 'half_width_degrees': 35, 'radius_km': query.radius_km,
            'measurements': [{'identity': p.identity, 'distance_km': d, 'bearing_degrees': b}
                             for p, d, b in measured]})
        return entity_result(nearest(qualifying, origin, 'nearest_qualifying'))
    return result('unsupported', 'unsupported_operation')
