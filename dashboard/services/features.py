"""Frozen Router feature semantics, copied from Step 3A; no research-data imports."""
from spatial.schema import EntityReference
from spatial.identity import resolve

def safe_features(question, context, prediction, probabilities):
    import math
    query = prediction.query
    # The frozen extractor raises before returning partial slots. Do not invent
    # a missing-slot count; expose only its documented unresolved slot group.
    slot_errors = {'count_origin_not_explicit_unique':['origin'], 'origin_not_explicit':['origin'],
        'wrong_named_target_count':['targets'],'category_slot_count':['categories'],
        'two_category_slot_count':['categories'],'missing_or_conflicting_radius':['radius_km'],
        'direction_slot_count':['direction']}
    missing = [] if query else slot_errors.get(prediction.reason)
    refs = (query.origin,)+query.targets if query else tuple(
        EntityReference(name,'unspecified') for name in sorted(
            {p.name for p in context.candidates}|({context.anchor.name} if context.anchor else set())) if name in question)
    ambiguous = sum(resolve(ref,context).status=='AMBIGUOUS' for ref in refs)
    ranked = sorted(probabilities,reverse=True)
    constraint_count = (len(query.targets)+len(query.categories)+int(query.radius_km is not None)+
        int(query.direction is not None)+int(query.first_hop_category is not None)+int(query.second_hop_category is not None)) if query else None
    return dict(short_confidence=float(prediction.confidence),
        short_entropy=-sum(p*math.log(p) for p in probabilities if p>0),
        short_top1_top2_margin=ranked[0]-ranked[1],query_valid=query is not None,
        missing_slots=missing,entity_ambiguity_count=ambiguous,question_length_chars=len(question),
        question_length_words=len(question.split()),predicted_operation=prediction.operation,
        constraint_count=constraint_count)

