"""Explicit geometric objectives over supplied points, not real-world suitability."""
from __future__ import annotations
from dataclasses import asdict,dataclass,field
import math
from spatial.geometry import bearing,distance,valid_coord
from spatial.schema import Location


@dataclass(frozen=True)
class SiteQuery:
    operation: str = 'site_selection'
    objective: str | None = None  # maximize/minimize nearest-facility distance
    facility_category: str | None = None
    direction: str | None = None
    min_distance_km: float | None = None
    top_k: int = 1
    required_data: tuple[str,...] = ()


@dataclass
class SiteResult:
    status: str
    reason: str
    selected_ids: tuple[str,...] = ()
    trace: dict = field(default_factory=dict)


def execute_site_query(query: SiteQuery, sites: tuple[Location,...],
                       facilities: tuple[Location,...], anchor: Location | None) -> SiteResult:
    if query.operation!='site_selection':return SiteResult('unsupported','unsupported_extension_operation')
    if query.required_data:return SiteResult('unsupported','required_data_unavailable')
    if query.objective is None:return SiteResult('needs_clarification','define_best_with_measurable_objective')
    if query.objective not in ('maximize_nearest_facility_distance','minimize_nearest_facility_distance'):
        return SiteResult('unsupported','unsupported_objective')
    if not query.facility_category:return SiteResult('needs_clarification','specify_facility_category')
    if not sites:return SiteResult('needs_clarification','supply_candidate_locations')
    if type(query.top_k)!=int or query.top_k<1:return SiteResult('invalid_query','invalid_top_k')
    if query.direction not in (None,'شمال','شرق','جنوب','غرب'):
        return SiteResult('needs_clarification','specify_cardinal_direction')
    if query.direction and anchor is None:return SiteResult('needs_clarification','specify_direction_reference')
    radius=query.min_distance_km
    if radius is not None and (type(radius) not in (int,float) or not math.isfinite(radius) or radius<0):
        return SiteResult('invalid_query','invalid_minimum_distance')
    all_points=sites+facilities+((anchor,) if anchor else ())
    if any(not valid_coord(*p.coordinates) for p in all_points):return SiteResult('invalid_query','invalid_coordinates')
    if len({p.identity for p in sites})!=len(sites):return SiteResult('invalid_query','duplicate_site_identity')
    existing=tuple(p for p in facilities if p.category==query.facility_category)
    if not existing:return SiteResult('unsupported','no_existing_facility_coordinates')
    scored=[]
    center={'شمال':0,'شرق':90,'جنوب':180,'غرب':270}.get(query.direction)
    for site in sites:
        if center is not None:
            if site.coordinates==anchor.coordinates:continue
            b=bearing(anchor.coordinates,site.coordinates)
            if abs((b-center+180)%360-180)>35:continue
        ds=[(distance(site.coordinates,p.coordinates),p.identity) for p in existing]
        nearest=min(d for d,_ in ds)
        if radius is not None and nearest<radius:continue
        scored.append({'site_id':site.identity,'nearest_facility_distance_km':nearest,
                       'nearest_facility_ids':[pid for d,pid in ds if d==nearest]})
    trace={'query':asdict(query),'scope':'Supplied points and listed facilities only; not real healthcare suitability',
           'direction_half_width_degrees':35 if query.direction else None,'sites_supplied':len(sites),
           'facilities_supplied':len(existing),'qualifying_sites':len(scored)}
    if not scored:return SiteResult('not_found','no_candidate_satisfies_constraints',trace=trace)
    reverse=query.objective=='maximize_nearest_facility_distance'
    scored.sort(key=lambda r:r['nearest_facility_distance_km'],reverse=reverse)
    # Include every tie at the requested rank; no fabricated unique winner.
    boundary=scored[min(query.top_k,len(scored))-1]['nearest_facility_distance_km']
    selected=[r for r in scored if r['nearest_facility_distance_km']>=boundary] if reverse else [r for r in scored if r['nearest_facility_distance_km']<=boundary]
    trace.update(ranking=scored,tie_inclusive=True)
    return SiteResult('success','geometric_ranking_only',tuple(r['site_id'] for r in selected),trace)
