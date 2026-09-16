"""Validate the frozen query and public identities before spatial execution.

All geometry belongs exclusively to spatial.execute; this adapter does not
implement a second, inconsistent geometry verifier.
"""
from dataclasses import asdict
from spatial.identity import resolve
from spatial.query_parser import validate_query
from spatial.schema import QueryError


def validate_identity(query, context):
    try:
        validate_query(query)
    except QueryError as exc:
        return {'status':'invalid_query', 'reason':exc.reason, 'resolutions':[]}
    resolutions=[resolve(ref,context) for ref in (query.origin,)+query.targets]
    status='not_found' if any(r.status=='NOT_FOUND' for r in resolutions) else 'ambiguous' if any(r.status=='AMBIGUOUS' for r in resolutions) else 'valid'
    return {'status':status,'resolutions':[asdict(r) for r in resolutions]}
