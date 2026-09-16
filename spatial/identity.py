"""Exact, public-input identity resolution; never choose among duplicate names."""
from .schema import EntityReference, ParsedContext, Resolution


def resolve(reference: EntityReference, context: ParsedContext) -> Resolution:
    candidates = tuple(p for p in context.candidates if p.name == reference.name)
    anchor = context.anchor
    if reference.source == 'context_anchor':
        matches = (anchor,) if anchor and anchor.name == reference.name else ()
    elif reference.source == 'candidate':
        matches = candidates
    else:
        matches = candidates
        if anchor and anchor.name == reference.name:
            # Do not create a second coordinate interpretation solely because
            # the same reference also appears in the registry at that location.
            # Candidate duplicates themselves always remain separate matches.
            if not candidates or any(p.coordinates != anchor.coordinates for p in candidates):
                matches = (anchor,) + candidates
    status = 'NOT_FOUND' if not matches else 'UNIQUE' if len(matches) == 1 else 'AMBIGUOUS'
    message = {'UNIQUE': 'One visible location matches', 'NOT_FOUND': 'No visible location matches',
               'AMBIGUOUS': 'Multiple locations share this name; specify the location'}[status]
    return Resolution(reference, status, matches, message)
