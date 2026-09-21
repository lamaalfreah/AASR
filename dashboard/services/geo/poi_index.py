"""Cached local OSM gazetteer. No research records, external keys, or model calls."""
import hashlib
import json
import math
import threading
from functools import lru_cache
from pathlib import Path
from django.conf import settings
from spatial.schema import Location
from language.neural_short import normalize_words

AMENITIES={'hospital':'مستشفى','clinic':'عيادة','pharmacy':'صيدلية','school':'مدرسة',
 'university':'جامعة','college':'كلية','restaurant':'مطعم','cafe':'مقهى','bank':'بنك',
 'atm':'صراف آلي','fuel':'محطة وقود','fast_food':'مطعم وجبات سريعة',
 'kindergarten':'روضة أطفال','place_of_worship':'مكان عبادة'}
_lock=threading.Lock()


def clean(value):
    return value.strip() if isinstance(value,str) and value.strip() else ''


def build_index():
    """Read the configured region once, retain only named POIs and public geometry."""
    from pyrosm import OSM
    pbf=Path(settings.ASAR_OSM_PBF_PATH)
    if not pbf.is_file():raise RuntimeError('Local OSM extract is unavailable')
    signature=hashlib.sha256(json.dumps([str(pbf),pbf.stat().st_size,pbf.stat().st_mtime_ns,
                                       settings.AASR_OSM_BBOX,AMENITIES,3],sort_keys=True).encode()).hexdigest()
    path=settings.BASE_DIR/'.runtime'/f'poi-index-{signature}.json'
    if path.exists():return json.loads(path.read_text())
    osm=OSM(str(pbf),bounding_box=settings.AASR_OSM_BBOX)
    frame=osm.get_pois(custom_filter={'amenity':list(AMENITIES),'healthcare':['hospital','clinic','pharmacy']},
                       extra_attributes=['name:ar','name:en','alt_name','official_name','short_name'])
    rows=[];seen=set()
    if frame is None:raise RuntimeError('No local POIs in configured extent')
    for _,row in frame.iterrows():
        tags=row.get('tags')
        if isinstance(tags,str):
            try:tags=json.loads(tags)
            except ValueError:tags={}
        if not isinstance(tags,dict):tags={}
        def tag(key):return clean(row.get(key)) or clean(tags.get(key))
        category=AMENITIES.get(tag('amenity')) or AMENITIES.get(tag('healthcare'))
        names=list(dict.fromkeys(filter(None,[tag('name:ar'),tag('name'),tag('name:en'),tag('alt_name'),tag('official_name'),tag('short_name')])))
        if not category or not names:continue
        geometry=row.get('geometry')
        if geometry is None or geometry.is_empty:continue
        point=geometry if geometry.geom_type=='Point' else geometry.representative_point()
        lat,lon=float(point.y),float(point.x)
        if not math.isfinite(lat) or not math.isfinite(lon):continue
        osm_id=str(int(row['id']));osm_type=tag('osm_type') or ('node' if geometry.geom_type=='Point' else 'way')
        key=f'osm:{osm_type}:{osm_id}'
        if key in seen:continue
        seen.add(key)
        rows.append(dict(id=key,name=names[0],aliases=names,category=category,lat=lat,lon=lon))
    if not rows:raise RuntimeError('No named local POIs')
    rows.sort(key=lambda row:row['id'])
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(rows,ensure_ascii=False));temporary.replace(path)
    return rows


@lru_cache(maxsize=1)
def index():
    with _lock:return build_index()


@lru_cache(maxsize=1)
def locations():return tuple(Location(r['id'],r['name'],r['lat'],r['lon'],r['category']) for r in index())


def normalize(text):return ' '.join(normalize_words(text).split())
