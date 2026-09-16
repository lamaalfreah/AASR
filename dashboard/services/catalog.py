"""Public location catalogs only; no dataset retrieval or gold fields."""
import json
import math
from django.conf import settings
from spatial.schema import CATEGORIES, Location, ParsedContext


def load_catalog():
    if settings.AASR_USE_RUNTIME_REGISTRY:
        from .registry import preview_catalog
        return preview_catalog()
    return json.loads(settings.AASR_CATALOG_PATH.read_text(encoding='utf-8'))


def parse_catalog(data):
    if not isinstance(data, dict) or set(data) - {'name', 'anchor', 'candidates'}:
        raise ValueError('كتالوج المواقع غير صالح.')
    candidates = data.get('candidates')
    if not isinstance(candidates, list) or not 1 <= len(candidates) <= 200:
        raise ValueError('يجب توفير من 1 إلى 200 موقع مرشح.')
    def location(row, identity, candidate):
        if not isinstance(row, dict) or set(row) - {'name','lat','lng','category'}:
            raise ValueError('حقول الموقع غير صالحة.')
        name = row.get('name')
        if not isinstance(name, str) or not name.strip() or len(name) > 160 or any(ord(c)<32 for c in name):
            raise ValueError('اسم الموقع غير صالح.')
        lat, lng = row.get('lat'), row.get('lng')
        if any(type(v) not in (int,float) or not math.isfinite(v) for v in (lat,lng)) or not -90<=lat<=90 or not -180<=lng<=180:
            raise ValueError('إحداثيات الموقع غير صالحة.')
        category = row.get('category')
        if candidate and category not in CATEGORIES:
            raise ValueError('فئة الموقع غير مدعومة.')
        return Location(identity, name, lat, lng, category)
    return ParsedContext(location(data.get('anchor'), 'anchor', False),
                         tuple(location(r, f'candidate:{i}', True) for i,r in enumerate(candidates)))
