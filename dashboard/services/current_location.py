"""Request-scoped location binding; no geocoding, geometry, or persistence."""
import math
import re
from dataclasses import replace
from spatial.schema import Location

CURRENT_NAME='موقعي الحالي'
SELF_REFERENCE=re.compile(r'(?<!\w)(?:حولي|حولنا|مني|عندي|موقعي(?: الحالي)?|بالقرب مني)(?!\w)')


def bind_location(question, context, current_location):
    personal=bool(SELF_REFERENCE.search(question) or re.search(r'(?:أقرب|اقرب|أبعد|ابعد|قريب).*?(?<!\w)لي(?!\w)',question))
    if current_location is None:
        return context, personal
    if (not isinstance(current_location,dict) or set(current_location)!={'lat','lon'}
        or any(type(current_location[k]) not in (int,float) or not math.isfinite(current_location[k]) for k in ('lat','lon'))
        or not -90<=current_location['lat']<=90 or not -180<=current_location['lon']<=180):
        raise ValueError('الموقع الحالي غير صالح.')
    # An explicitly named original reference keeps its original meaning.
    if context.anchor and context.anchor.name in question and not personal:
        return context, False
    anchor=Location('current_location',CURRENT_NAME,current_location['lat'],current_location['lon'],None)
    candidates=context.candidates
    if context.anchor:
        candidates=(context.anchor,)+candidates
    return replace(context,anchor=anchor,candidates=candidates),False
