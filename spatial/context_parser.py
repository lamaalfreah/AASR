"""Visible context grammar extracted unchanged from the full dataset audit."""
import functools
import re
from .geometry import valid_coord

REG_MARK = "سجل المعالم المكانية المستخرج من OpenStreetMap:"
ART_MARK = "خلفية نصية من ويكيبيديا العربية عن المكان:"
START = re.compile(r"(?m)^[ \t]*(\d+)\.[ \t]*الاسم:")
BODY = re.compile(r"\s*(?P<name>.*?)؛\s*النوع:\s*(?P<category>.*?)؛\s*خط العرض:\s*"
                  r"(?P<lat>.*?)؛\s*خط الطول:\s*(?P<lon>.*?)\s*\.\s*\Z", re.S)
ANCHOR = re.compile(r"المكان المرجعي هو «(?P<name>.*?)».*?خط العرض:\s*(?P<lat>[-+\d.]+)"
                    r"\s*خط الطول:\s*(?P<lon>[-+\d.]+)", re.S)

@functools.lru_cache(maxsize=32)
def parse_context_fields(context):
    """Parse visible records only, preserving identity, order, and duplicates."""
    errors = []
    if context.count(REG_MARK) != 1:
        errors.append("registry_marker_count")
    prefix, sep, registry = context.partition(REG_MARK)
    before, art_sep, article = prefix.partition(ART_MARK)
    am = ANCHOR.search(before)
    anchor = None
    if am:
        try:
            anchor = (am['name'], float(am['lat']), float(am['lon']))
            if not valid_coord(*anchor[1:]): errors.append("invalid_anchor_coordinates")
        except ValueError:
            errors.append("malformed_anchor_coordinates")
    else:
        errors.append("missing_anchor")
    starts = list(START.finditer(registry))
    candidates, bad = [], []
    for i, s in enumerate(starts):
        end = starts[i+1].start() if i+1 < len(starts) else len(registry)
        block = registry[s.end():end]
        m = BODY.fullmatch(block)
        if not m:
            bad.append({"ordinal": int(s[1]), "reason": "malformed_block", "excerpt": block[:120]})
            continue
        # Preserve trailing whitespace in the literal source name. It may also
        # appear in quoted questions/gold text; normalization is a separate audit.
        name, category = m['name'], m['category'].strip()
        reasons = []
        if not name: reasons.append("missing_name")
        if not category: reasons.append("missing_category")
        if not m['lat'].strip(): reasons.append("missing_latitude")
        if not m['lon'].strip(): reasons.append("missing_longitude")
        try:
            lat, lon = float(m['lat']), float(m['lon'])
            if not valid_coord(lat, lon): reasons.append("invalid_candidate_coordinates")
        except ValueError:
            lat = lon = float('nan')
            reasons.append("nonnumeric_coordinates")
        candidates.append({"ordinal": int(s[1]), "name": name, "category": category,
                           "lat": lat, "lon": lon, "multiline": '\n' in name,
                           "errors": reasons})
        if reasons: bad.append({"ordinal": int(s[1]), "reason": ','.join(reasons)})
    if sep and not starts: errors.append("no_candidate_blocks")
    if starts and registry[:starts[0].start()].strip(): errors.append("unparsed_registry_prefix")
    if [int(s[1]) for s in starts] != list(range(1, len(starts)+1)):
        errors.append("nonsequential_ordinals")
    return {"anchor": anchor, "article": article.strip(), "article_separated": bool(art_sep and sep),
            "registry": registry.strip(), "candidates": candidates, "bad": bad,
            "errors": errors, "blocks": len(starts)}



def parse_context(context):
    """Typed, immutable view; never drop invalid or duplicate rows silently."""
    from .schema import Location, ParsedContext
    if not isinstance(context, str):
        return ParsedContext(None, (), ('context_must_be_text',))
    raw = parse_context_fields(context)
    errors = list(raw['errors']) + [b['reason'] for b in raw['bad']]
    anchor = Location('anchor', *raw['anchor']) if raw['anchor'] else None
    candidates = tuple(Location(f"candidate:{p['ordinal']}", p['name'], p['lat'],
                                p['lon'], p['category']) for p in raw['candidates'])
    return ParsedContext(anchor, candidates, tuple(errors))
